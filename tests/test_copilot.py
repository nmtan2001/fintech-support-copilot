import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

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
        return BytesIO(json.dumps({"candidates": [{"content": {"parts": [
            {"text": "Please freeze it. [demo-card-99]"}
        ]}}]}).encode())

    monkeypatch.setattr("fintech_support.copilot.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="outside the retrieved FAQ"):
        grounded_answer("My card was stolen", hits, llm_model="fake", llm_api_key="test-key")


def test_gemini_chat_api_and_grounded_citation(monkeypatch):
    hits = tuple(make_copilot("lost_or_stolen_card").retriever.search(
        "My card was stolen", intent="lost_or_stolen_card"
    ))
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("X-goog-api-key")
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return BytesIO(json.dumps({"candidates": [{"content": {"parts": [
            {"text": "This prototype cannot freeze cards or change account settings. [demo-lost-01]"}
        ]}}]}).encode())

    monkeypatch.setattr("fintech_support.copilot.urlopen", fake_urlopen)
    answer = grounded_answer(
        "My card was stolen",
        hits,
        llm_model="gemini-3.5-flash-lite",
        llm_api_key="test-key",
    )

    assert "[demo-lost-01]" in answer
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 30.0
    assert captured["payload"]["generationConfig"]["temperature"] == 0
    assert "systemInstruction" in captured["payload"]
    assert captured["payload"]["contents"][0]["role"] == "user"


def test_gemini_rejects_unsafe_model_id_before_http(monkeypatch):
    from fintech_support.copilot import _request_gemini_message

    monkeypatch.setattr("fintech_support.copilot.urlopen", lambda *args, **kwargs:
                        pytest.fail("Invalid model must not trigger an HTTP request"))
    with pytest.raises(ValueError, match="Invalid Gemini model ID"):
        _request_gemini_message("../other-api", "secret-key", {})


def test_gemini_http_errors_hide_credentials(monkeypatch):
    from fintech_support.copilot import _request_gemini_message

    def failed_request(request, timeout):
        raise HTTPError(request.full_url, 429, "quota exceeded", {}, None)

    monkeypatch.setattr("fintech_support.copilot.urlopen", failed_request)
    with pytest.raises(RuntimeError, match="rate limit or quota") as exc:
        _request_gemini_message("gemini-3.5-flash-lite", "secret-key", {})
    assert "secret-key" not in str(exc.value)


def test_gemini_empty_candidates_use_grounded_fallback(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    monkeypatch.setattr("fintech_support.copilot.urlopen", lambda *args, **kwargs:
                        BytesIO(b'{"candidates":[]}'))
    result = copilot.respond("My card payment was declined", llm_model="gemini-3.5-flash-lite",
                            llm_api_key="secret-key")
    assert result.status == "generation_fallback"
    assert "[demo-card-" in result.answer


def test_gemini_ignores_thought_parts(monkeypatch):
    from fintech_support.copilot import _request_gemini_completion

    payloads = []

    def completion(model, api_key, payload):
        payloads.append(payload)
        return {"parts": [{"text": "private reasoning", "thought": True},
                          {"text": "Verified demo response."}]}

    monkeypatch.setattr("fintech_support.copilot._request_gemini_message", completion)
    assert _request_gemini_completion("gemini-3.5-flash-lite", "test", [
        {"role": "user", "content": "support question"}
    ]) == "Verified demo response."
    assert len(payloads) == 1
    assert payloads[0]["generationConfig"]["maxOutputTokens"] == 2048


def test_empty_gemini_completion_uses_grounded_fallback(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    calls = []

    def empty_completion(model, api_key, payload):
        calls.append(payload)
        return {"parts": [{"text": "thinking", "thought": True}]}

    monkeypatch.setattr("fintech_support.copilot._request_gemini_message", empty_completion)
    result = copilot.respond(
        "My card payment was declined", llm_model="gemini-3.5-flash-lite", llm_api_key="test",
    )
    assert result.status == "generation_fallback"
    assert result.generation_status == "unavailable"
    assert "[demo-card-" in result.answer
    assert "no final answer" in result.grounding_issues[0]
    assert len(calls) == 1


def test_function_call_requires_demo_reference_even_if_checkbox_enabled(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)

    def valid_grounded_answer(*args):
        return "In the demo, ask the customer to verify their available balance and card limits in the app. [demo-card-02]"

    monkeypatch.setattr("fintech_support.copilot._request_gemini_completion", valid_grounded_answer)
    monkeypatch.setattr("fintech_support.copilot._select_gemini_tool", lambda *args:
                        pytest.fail("No synthetic ID was provided for a lookup"))
    result = copilot.respond(
        "My card payment was declined", llm_model="gemini-3.5-flash-lite",
        llm_api_key="test", llm_select_tool=True,
    )
    assert result.tool_result is None
    assert result.tool_routing is None
    assert result.generation_status == "accepted"


def make_tool_copilot(tmp_path, intent="declined_card_payment"):
    db = tmp_path / "synthetic.sqlite"
    initialize_demo_database(db, ROOT / "data/transactions/demo_records.json")
    return SupportCopilot(
        StubClassifier(intent),
        FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json")),
        DemoBankingTools(db),
    )


def test_gemini_tool_call_uses_verified_synthetic_record(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    requests = []

    def fake_completion(model, api_key, payload):
        requests.append(payload)
        if "tools" in payload:
            return {"parts": [{"functionCall": {
                "name": "lookup_demo_transaction",
                "args": {"reference": "TX-CARD-001"},
            }}]}
        return {"parts": [{"text": "If it is frozen, review the in-app unfreeze option. [demo-card-01]"}]}

    monkeypatch.setattr("fintech_support.copilot._request_gemini_message", fake_completion)
    response = copilot.respond(
        "Why did my payment fail?",
        record_reference="TX-CARD-001",
        llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
    )
    assert response.tool_routing == "gemini_function_call"
    assert response.generation_status == "accepted"
    assert response.tool_result.reason_code == "CARD_FROZEN"
    assert "reason_code=CARD_FROZEN" in response.answer
    assert "[demo-card-01]" in response.answer
    assert [row["name"] for row in requests[0]["tools"][0]["functionDeclarations"]] == [
        "lookup_demo_card", "lookup_demo_transaction"
    ]
    assert requests[0]["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert "reason_code=CARD_FROZEN" in requests[1]["contents"][0]["parts"][0]["text"]
    assert len(requests) == 2


def test_gemini_tool_selection_uses_output_budget(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    payloads = []

    def completion(model, api_key, payload):
        payloads.append(payload)
        if "tools" in payload:
            return {"parts": [{"functionCall": {
                "name": "lookup_demo_transaction", "args": {"reference": "TX-CARD-001"},
            }}]}
        return {"parts": [{"text": "If it is frozen, review the in-app unfreeze option. [demo-card-01]"}]}

    monkeypatch.setattr("fintech_support.copilot._request_gemini_message", completion)
    result = copilot.respond(
        "Why was TX-CARD-001 declined?", llm_model="gemini-3.5-flash-lite",
        llm_api_key="test", llm_select_tool=True,
    )
    assert result.tool_routing == "gemini_function_call"
    assert payloads[0]["generationConfig"]["maxOutputTokens"] == 1024


@pytest.mark.parametrize("tool,reference", [
    ("lookup_demo_card", "TX-CARD-001"),
    ("lookup_demo_transaction", "TX-CARD-002"),
    ("lookup_demo_transaction", "TX-CARD-001 extra"),
])
def test_gemini_tool_call_rejects_wrong_selection(monkeypatch, tmp_path, tool, reference):
    copilot = make_tool_copilot(tmp_path)
    calls = []

    def fake_completion(model, api_key, payload):
        calls.append(payload)
        return {"parts": [{"functionCall": {
            "name": tool, "args": {"reference": reference},
        }}]}

    monkeypatch.setattr("fintech_support.copilot._request_gemini_message", fake_completion)
    with pytest.raises(RuntimeError, match="outside the authorized"):
        copilot.respond(
            "Why did my card payment fail?", record_reference="TX-CARD-001",
            llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("bad_message", [
    {"parts": []},
    {"parts": [{"functionCall": "invalid"}]},
    {"parts": [{"functionCall": {"name": "lookup_demo_transaction", "args": "bad"}}]},
    {"parts": [{"functionCall": {"name": "lookup_demo_transaction", "args": {}}},
               {"functionCall": {"name": "lookup_demo_transaction", "args": {}}}]},
])
def test_gemini_tool_call_rejects_malformed_responses(monkeypatch, tmp_path, bad_message):
    copilot = make_tool_copilot(tmp_path)
    monkeypatch.setattr(
        "fintech_support.copilot._request_gemini_message",
        lambda model, api_key, payload: bad_message,
    )
    with pytest.raises(RuntimeError, match="tool call|tool arguments"):
        copilot.respond(
            "My card payment failed", record_reference="TX-CARD-001",
            llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
        )


def test_invalid_reference_fails_before_gemini_call(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)

    def should_not_call(*args, **kwargs):
        pytest.fail("Gemini should not be called for an invalid demo reference")

    monkeypatch.setattr("fintech_support.copilot._request_gemini_message", should_not_call)
    with pytest.raises(ValueError, match="TX-xxx"):
        copilot.respond(
            "My card payment failed", record_reference="CARD-001",
            llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
        )


def test_rejected_generation_falls_back_and_preserves_audit(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path, "pending_transfer")
    invented = "Your transfer is completed for EUR 500. [demo-transfer-01]"
    monkeypatch.setattr("fintech_support.copilot._request_gemini_completion", lambda *args: invented)
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

    monkeypatch.setattr("fintech_support.copilot._request_gemini_completion", forbidden)
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
    monkeypatch.setattr("fintech_support.copilot._request_gemini_completion", lambda *args:
                        pytest.fail("Invalid tool results must not reach the LLM"))
    with pytest.raises(RuntimeError, match="Invalid tool result"):
        copilot.respond("My transfer TX-TRF-001 is pending", llm_model="test", llm_api_key="test")
