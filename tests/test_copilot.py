import json
from io import BytesIO
from pathlib import Path

import pytest

from fintech_support.copilot import Prediction, SupportCopilot, IntentClassifier, grounded_answer
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parents[1]


class StubClassifier:
    def __init__(self, intent, confidence=0.95):
        self.prediction = Prediction(intent, confidence)

    def predict(self, question):
        return self.prediction


def make_copilot(intent, confidence=0.95):
    retriever = FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json"))
    return SupportCopilot(StubClassifier(intent, confidence), retriever)


def test_returns_synthetic_policy_with_source():
    response = make_copilot("lost_or_stolen_card").respond("I lost my card. Can I freeze it?")
    assert response.status == "answered"
    assert "[demo-lost-01]" in response.answer
    assert "cannot freeze cards" in response.answer
    assert all(hit.faq.source.startswith("Synthetic demo policy") for hit in response.retrieved)


def test_unsupported_intent_abstains():
    response = make_copilot("exchange_rate").respond("How do I get the exchange rate?")
    assert response.status == "unsupported_intent"
    assert not response.retrieved
    assert "official support" in response.answer


def test_low_confidence_abstains_without_llm():
    response = make_copilot("declined_card_payment", confidence=0.3).respond(
        "My card payment was rejected", min_confidence=0.5, llm_model="fake-model"
    )
    assert response.status == "low_confidence"
    assert not response.retrieved


def test_load_existing_baseline_model():
    model_path = ROOT / "outputs/baseline/model.joblib"
    if not model_path.is_file():
        pytest.skip("Baseline artifact is not present")
    classifier, name = IntentClassifier.from_artifacts(
        transformer_dir=ROOT / "outputs/distilbert/model",
        baseline_path=model_path,
        prefer_transformer=False,
    )
    prediction = classifier.predict("My card has been stolen")
    assert isinstance(prediction.intent, str)
    assert 0 <= prediction.confidence <= 1
    assert name == "TF-IDF + Logistic Regression"


def test_rejects_llm_citations_outside_retrieved_sources(monkeypatch):
    hits = tuple(make_copilot("lost_or_stolen_card").retriever.search(
        "My card was stolen", intent="lost_or_stolen_card"
    ))

    def fake_urlopen(request, timeout):
        return BytesIO(json.dumps({"choices": [{"message": {
            "content": "Please freeze it. [demo-card-99]"
        }}]}).encode())

    monkeypatch.setattr("fintech_support.copilot.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="outside the retrieved FAQ"):
        grounded_answer("My card was stolen", hits, llm_model="fake", llm_api_key="test-key")


def test_nvidia_chat_api_and_grounded_citation(monkeypatch):
    hits = tuple(make_copilot("lost_or_stolen_card").retriever.search(
        "My card was stolen", intent="lost_or_stolen_card"
    ))
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return BytesIO(json.dumps({"choices": [{"message": {
            "content": "This prototype cannot freeze cards or change account settings. [demo-lost-01]"
        }}]}).encode())

    monkeypatch.setattr("fintech_support.copilot.urlopen", fake_urlopen)
    answer = grounded_answer(
        "My card was stolen",
        hits,
        llm_model="nvidia/example-instruct-model",
        llm_api_key="test-key",
    )

    assert "[demo-lost-01]" in answer
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["timeout"] == 20.0
    assert captured["payload"]["model"] == "nvidia/example-instruct-model"
    assert captured["payload"]["temperature"] == 0


def make_tool_copilot(tmp_path, intent="declined_card_payment"):
    db = tmp_path / "synthetic.sqlite"
    initialize_demo_database(db, ROOT / "data/transactions/demo_records.json")
    return SupportCopilot(
        StubClassifier(intent),
        FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json")),
        DemoBankingTools(db),
    )


def test_nvidia_tool_call_uses_verified_synthetic_record(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    requests = []

    def fake_completion(model, api_key, payload):
        requests.append(payload)
        if "tools" in payload:
            return {"tool_calls": [{"function": {
                "name": "lookup_demo_transaction",
                "arguments": json.dumps({"reference": "TX-CARD-001"}),
            }}]}
        return {"content": "If it is frozen, review the in-app unfreeze option. [demo-card-01]"}

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", fake_completion)
    response = copilot.respond(
        "Why did my payment fail?",
        record_reference="TX-CARD-001",
        llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
    )
    assert response.tool_routing == "nvidia_function_call"
    assert response.generation_status == "accepted"
    assert response.tool_result.reason_code == "CARD_FROZEN"
    assert "reason_code=CARD_FROZEN" in response.answer
    assert "[demo-card-01]" in response.answer
    assert [row["function"]["name"] for row in requests[0]["tools"]] == [
        "lookup_demo_card", "lookup_demo_transaction"
    ]
    assert "reason_code=CARD_FROZEN" in requests[1]["messages"][1]["content"]
    assert len(requests) == 2


@pytest.mark.parametrize("tool,reference", [
    ("lookup_demo_card", "TX-CARD-001"),
    ("lookup_demo_transaction", "TX-CARD-002"),
    ("lookup_demo_transaction", "TX-CARD-001 extra"),
])
def test_nvidia_tool_call_rejects_wrong_selection(monkeypatch, tmp_path, tool, reference):
    copilot = make_tool_copilot(tmp_path)
    calls = []

    def fake_completion(model, api_key, payload):
        calls.append(payload)
        return {"tool_calls": [{"function": {
            "name": tool, "arguments": json.dumps({"reference": reference}),
        }}]}

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", fake_completion)
    with pytest.raises(RuntimeError, match="outside the authorized"):
        copilot.respond(
            "Why did my card payment fail?", record_reference="TX-CARD-001",
            llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("bad_message", [
    {"tool_calls": []},
    {"tool_calls": [{"function": {"name": "lookup_demo_transaction", "arguments": "{bad"}}]},
    {"tool_calls": [{"function": {"name": "lookup_demo_transaction", "arguments": "{}"}},
                    {"function": {"name": "lookup_demo_transaction", "arguments": "{}"}}]},
])
def test_nvidia_tool_call_rejects_malformed_responses(monkeypatch, tmp_path, bad_message):
    copilot = make_tool_copilot(tmp_path)
    monkeypatch.setattr(
        "fintech_support.copilot._request_nvidia_message",
        lambda model, api_key, payload: bad_message,
    )
    with pytest.raises(RuntimeError, match="tool call|tool arguments"):
        copilot.respond(
            "My card payment failed", record_reference="TX-CARD-001",
            llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
        )


def test_invalid_reference_fails_before_nvidia_call(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)

    def should_not_call(*args, **kwargs):
        pytest.fail("NVIDIA should not be called for an invalid demo reference")

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", should_not_call)
    with pytest.raises(ValueError, match="TX-xxx"):
        copilot.respond(
            "My card payment failed", record_reference="CARD-001",
            llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
        )


def test_rejected_generation_falls_back_and_preserves_audit(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path, "pending_transfer")
    invented = "Your transfer is completed for EUR 500. [demo-transfer-01]"
    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", lambda *args: invented)
    result = copilot.respond("My transfer TX-TRF-001 is pending", llm_model="test", llm_api_key="test")
    assert result.status == "grounding_fallback"
    assert result.generated_answer == invented
    assert result.grounding_issues
    assert "EUR 500" not in result.answer
    assert "status=pending" in result.answer


def test_not_found_does_not_ask_llm_to_infer_record(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path, "pending_transfer")

    def forbidden(*args):
        pytest.fail("A missing record must use deterministic guidance")

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", forbidden)
    result = copilot.respond("My transfer TX-TRF-999 is pending", llm_model="test", llm_api_key="test")
    assert result.tool_result.found is False
    assert "no demo record" in result.answer


def test_explicit_reference_cannot_override_different_question_reference(tmp_path):
    with pytest.raises(ValueError, match="reference"):
        make_tool_copilot(tmp_path, "pending_transfer").respond(
            "My transfer TX-TRF-001 is pending", record_reference="TX-TRF-002"
        )


def test_unsupported_intent_never_requires_tools():
    response = make_copilot("exchange_rate").respond("Exchange rate for TX-CARD-001?")
    assert response.status == "unsupported_intent"
    assert response.tool_result is None


def test_invalid_tool_result_fails_before_generation(monkeypatch, tmp_path):
    from fintech_support.demo_tools import ToolResult

    copilot = make_tool_copilot(tmp_path, "pending_transfer")
    monkeypatch.setattr(copilot.tools, "call", lambda *args:
                        ToolResult("lookup_demo_transaction", "TX-TRF-002", True, "pending", "PROCESSING"))
    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", lambda *args:
                        pytest.fail("Invalid tool results must not reach the LLM"))
    with pytest.raises(RuntimeError, match="Invalid tool result"):
        copilot.respond("My transfer TX-TRF-001 is pending", llm_model="test", llm_api_key="test")
