import json
from dataclasses import replace
from pathlib import Path

import pytest

from evaluate_smoke import evaluate_cases
from fintech_support.copilot import Prediction, SupportCopilot
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parents[1]


class Classifier:
    def __init__(self, intent="pending_transfer", confidence=0.95):
        self.prediction = Prediction(intent, confidence)

    def predict(self, question):
        return self.prediction


@pytest.fixture
def copilot(tmp_path):
    database = tmp_path / "demo.sqlite"
    initialize_demo_database(database, ROOT / "data/transactions/demo_records.json")
    return SupportCopilot(Classifier(), FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json")),
                          DemoBankingTools(database))


def transfer_case(**overrides):
    return {
        "question": "My transfer TX-TRF-001 is pending",
        "expected_intent": "pending_transfer", "expected_status": "answered",
        "expected_faq_id": "demo-transfer-01", "expected_tool": "lookup_demo_transaction",
        "expected_reference": "TX-TRF-001", "expected_found": True,
        "expected_record_status": "pending", "expected_reason_code": "PROCESSING", **overrides,
    }


def test_reports_metrics_with_explicit_denominators(copilot):
    result = evaluate_cases(copilot, [transfer_case()])
    assert result["passed"] == 1
    assert result["metrics"] == {name: 1.0 if name != "hallucination_rate" else 0.0 for name in (
        "accuracy", "macro_f1", "recall_at_1", "recall_at_3", "tool_selection_accuracy",
        "argument_accuracy", "tool_data_accuracy", "citation_accuracy", "faithfulness",
        "hallucination_rate", "delivered_faithfulness", "task_success_rate",
    )}
    assert result["metric_support"]["argument_accuracy"] == 1


def test_no_tool_cases_do_not_inflate_argument_accuracy(copilot):
    case = transfer_case(question="My transfer is pending", expected_tool=None)
    result = evaluate_cases(copilot, [case])
    assert result["metrics"]["argument_accuracy"] is None
    assert result["metric_support"]["argument_accuracy"] == 0


def test_bad_generated_claim_fails_despite_safe_fallback(copilot, monkeypatch):
    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", lambda *args:
                        "Your transfer completed for EUR 500. [demo-transfer-01]")
    result = evaluate_cases(copilot, [transfer_case()], llm_model="test", llm_api_key="test")
    assert result["passed"] == 0
    assert result["metrics"]["faithfulness"] == 0
    assert result["metrics"]["hallucination_rate"] == 1
    assert result["metrics"]["delivered_faithfulness"] == 1


def test_wrong_tool_arguments_are_separate_from_selection(copilot, monkeypatch):
    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", lambda *args: {
        "tool_calls": [{"function": {"name": "lookup_demo_transaction",
                                    "arguments": '{"reference":"TX-TRF-002"}'}}]
    })
    result = evaluate_cases(copilot, [transfer_case()], llm_model="test", llm_api_key="test",
                            llm_select_tool=True)
    assert result["failed"] == 1
    assert result["metrics"]["tool_selection_accuracy"] == 1
    assert result["metrics"]["argument_accuracy"] == 0
    assert "outside the authorized" in result["cases"][0]["error"]


def test_expected_invalid_input_is_a_successful_guardrail(copilot):
    case = transfer_case(question="My transfer is pending", record_reference="CARD-001",
                         expected_tool=None, expected_faq_id=None,
                         expected_error="TX-xxx")
    result = evaluate_cases(copilot, [case])
    assert result["passed"] == 1
    assert result["metrics"]["accuracy"] is None


def test_fabricated_tool_data_fails_independent_expected_values(copilot, monkeypatch):
    response = copilot.respond("My transfer TX-TRF-001 is pending")
    bad = replace(response.tool_result, status="completed", reason_code="SETTLED")
    monkeypatch.setattr(copilot, "respond", lambda *args, **kwargs: replace(response, tool_result=bad))
    result = evaluate_cases(copilot, [transfer_case()])
    assert result["metrics"]["tool_data_accuracy"] == 0


@pytest.mark.parametrize("llm_enabled", [False, True])
def test_all_hand_authored_cases_with_oracle_routing(copilot, monkeypatch, llm_enabled):
    def synthetic_completion(model, api_key, payload):
        if "tools" in payload:
            request = json.loads(payload["messages"][1]["content"])
            return {"tool_calls": [{"function": {
                "name": "lookup_demo_card" if request["intent"] == "lost_or_stolen_card"
                else "lookup_demo_transaction",
                "arguments": json.dumps({"reference": request["reference"]}),
            }}]}
        evidence = payload["messages"][1]["content"].split("Synthetic FAQ evidence:\n", 1)[1]
        first = evidence.splitlines()[0]
        citation = first.split(" ", 1)[0]
        content = next(faq.content for faq in copilot.retriever.faqs if faq.id == citation[1:-1])
        return {"content": content + " " + citation}

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", synthetic_completion)
    cases = json.loads((ROOT / "data/evaluation/copilot_smoke_cases.json").read_text())
    for case in cases:
        copilot.classifier = Classifier(case["expected_intent"])
        report = evaluate_cases(
            copilot, [case], llm_model="mock-model" if llm_enabled else None,
            llm_api_key="mock-key" if llm_enabled else None, llm_select_tool=llm_enabled,
        )
        row = report["cases"][0]
        if case.get("expected_error"):
            assert row["passed"] is True, row
            continue
        assert row["checks"]["tool_match"] is True, row
        assert row["checks"]["argument_match"] is not False, row
        assert row["checks"]["tool_data_match"] is True, row
        assert row["checks"]["faithfulness_match"] is True, row
        assert row["checks"]["delivered_grounding"] is True, row
        if not row["retrieved_faq_ids"]:
            assert "official support" in row["answer"]


def test_runtime_failure_is_recorded_and_later_cases_still_run(copilot, monkeypatch):
    respond = copilot.respond
    calls = []

    def flaky(question, **kwargs):
        calls.append(question)
        if len(calls) == 1:
            raise RuntimeError("NVIDIA rate limit reached")
        return respond(question, **kwargs)

    monkeypatch.setattr(copilot, "respond", flaky)
    result = evaluate_cases(copilot, [transfer_case(), transfer_case()])
    assert result["passed"] == 1
    assert result["failed"] == 1


def test_generation_coverage_does_not_imply_live_llm_evaluation(copilot):
    report = evaluate_cases(copilot, [transfer_case()])
    assert report["generation"]["evaluated_candidates"] == 0
    assert report["generation"]["faithfulness"] is None


def test_wrong_citation_attribution_is_not_counted_as_correct(copilot, monkeypatch):
    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", lambda *args:
                        "A pending status means completion has not been confirmed. [demo-transfer-03]")
    report = evaluate_cases(copilot, [transfer_case()], llm_model="test", llm_api_key="test")
    assert report["metrics"]["citation_accuracy"] == 0


def test_malformed_arguments_preserve_observed_tool_selection(copilot, monkeypatch):
    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", lambda *args: {
        "tool_calls": [{"function": {"name": "lookup_demo_transaction", "arguments": "{bad"}}]
    })
    report = evaluate_cases(copilot, [transfer_case()], llm_model="test", llm_api_key="test",
                            llm_select_tool=True)
    assert report["metrics"]["tool_selection_accuracy"] == 1
    assert report["metrics"]["argument_accuracy"] == 0
