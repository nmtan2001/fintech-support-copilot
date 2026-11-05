import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from fintech_support.copilot import (
    DEFAULT_NVIDIA_MODEL, NVIDIA_REQUEST_TIMEOUT_SECONDS, Prediction, SupportCopilot,
    IntentClassifier, grounded_answer,
)
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
    assert "[HELP-LOST-CARD-01]" in response.answer
    assert "freeze it immediately in Cards" in response.answer
    assert all(hit.faq.source.endswith("Help Centre") for hit in response.retrieved)


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


def test_nvidia_nim_chat_api_and_grounded_citation(monkeypatch):
    hits = tuple(make_copilot("lost_or_stolen_card").retriever.search(
        "My card was stolen", intent="lost_or_stolen_card"
    ))
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return BytesIO(json.dumps({"choices": [{"message": {
            "content": "If your card is lost or stolen, freeze it immediately in Cards. [HELP-LOST-CARD-01]"
        }}]}).encode())

    monkeypatch.setattr("fintech_support.copilot.urlopen", fake_urlopen)
    answer = grounded_answer(
        "My card was stolen",
        hits,
        llm_model=DEFAULT_NVIDIA_MODEL,
        llm_api_key="test-key",
    )

    assert "[HELP-LOST-CARD-01]" in answer
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["api_key"] == "Bearer test-key"
    assert captured["timeout"] == NVIDIA_REQUEST_TIMEOUT_SECONDS
    assert captured["payload"]["model"] == DEFAULT_NVIDIA_MODEL
    assert captured["payload"]["max_tokens"] == 2048
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["messages"][1]["role"] == "user"


def test_nvidia_nim_rejects_unsafe_model_id_before_http(monkeypatch):
    from fintech_support.copilot import _request_nvidia_message

    monkeypatch.setattr("fintech_support.copilot.urlopen", lambda *args, **kwargs:
                        pytest.fail("Invalid model must not trigger an HTTP request"))
    with pytest.raises(ValueError, match="Invalid NVIDIA NIM model ID"):
        _request_nvidia_message("bad\nmodel", "secret-key", {})


def test_nvidia_nim_http_errors_hide_credentials(monkeypatch):
    from fintech_support.copilot import _request_nvidia_message

    def failed_request(request, timeout):
        raise HTTPError(request.full_url, 429, "quota exceeded", {}, None)

    monkeypatch.setattr("fintech_support.copilot.urlopen", failed_request)
    with pytest.raises(RuntimeError, match="rate limit or quota") as exc:
        _request_nvidia_message(DEFAULT_NVIDIA_MODEL, "secret-key", {})
    assert "secret-key" not in str(exc.value)


def test_nvidia_nim_payment_required_has_actionable_error(monkeypatch):
    from fintech_support.copilot import _request_nvidia_message

    def payment_required(request, timeout):
        raise HTTPError(request.full_url, 402, "payment required", {}, None)

    monkeypatch.setattr("fintech_support.copilot.urlopen", payment_required)
    with pytest.raises(RuntimeError, match="insufficient API credits"):
        _request_nvidia_message(DEFAULT_NVIDIA_MODEL, "test-key", {})


def test_nvidia_nim_retries_transient_503_with_backoff(monkeypatch):
    from fintech_support.copilot import _request_nvidia_message

    attempts = []
    delays = []

    def flaky_request(request, timeout):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise HTTPError(request.full_url, 503, "temporarily unavailable", {}, None)
        return BytesIO(json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode())

    monkeypatch.setattr("fintech_support.copilot.urlopen", flaky_request)
    monkeypatch.setattr("fintech_support.copilot.time.sleep", delays.append)
    result = _request_nvidia_message(DEFAULT_NVIDIA_MODEL, "test-key", {})
    assert result["content"] == "OK"
    assert len(attempts) == 3
    assert len(delays) == 2
    assert 1 <= delays[0] < 1.25
    assert 2 <= delays[1] < 2.25


def test_nvidia_nim_503_after_retries_has_actionable_error(monkeypatch):
    from fintech_support.copilot import _request_nvidia_message

    monkeypatch.setattr("fintech_support.copilot.urlopen", lambda request, timeout:
                        (_ for _ in ()).throw(HTTPError(request.full_url, 503, "unavailable", {}, None)))
    monkeypatch.setattr("fintech_support.copilot.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="temporarily overloaded.*after retries"):
        _request_nvidia_message(DEFAULT_NVIDIA_MODEL, "test-key", {})


def test_nvidia_nim_empty_choices_use_grounded_fallback(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    monkeypatch.setattr("fintech_support.copilot.urlopen", lambda *args, **kwargs:
                        BytesIO(b'{"choices":[]}'))
    result = copilot.respond("My card payment was declined", llm_model=DEFAULT_NVIDIA_MODEL,
                            llm_api_key="secret-key")
    assert result.status == "generation_fallback"
    assert "[HELP-CARD-" in result.answer


def test_nvidia_nim_completion_preserves_chat_messages(monkeypatch):
    from fintech_support.copilot import _request_nvidia_completion

    payloads = []

    def completion(model, api_key, payload):
        payloads.append(payload)
        return {"content": "Verified demo response."}

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", completion)
    assert _request_nvidia_completion(DEFAULT_NVIDIA_MODEL, "test", [
        {"role": "user", "content": "support question"}
    ]) == "Verified demo response."
    assert len(payloads) == 1
    assert payloads[0]["messages"][0]["content"] == "support question"
    assert payloads[0]["max_tokens"] == 2048


def test_empty_nvidia_nim_completion_uses_grounded_fallback(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    calls = []

    def empty_completion(model, api_key, payload):
        calls.append(payload)
        return {"content": None}

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", empty_completion)
    result = copilot.respond(
        "My card payment was declined", llm_model=DEFAULT_NVIDIA_MODEL, llm_api_key="test",
    )
    assert result.status == "generation_fallback"
    assert result.generation_status == "unavailable"
    assert "[HELP-CARD-" in result.answer
    assert "no final answer" in result.grounding_issues[0]
    assert len(calls) == 1


def test_nvidia_generation_runtime_error_uses_grounded_fallback(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    monkeypatch.setattr(
        "fintech_support.copilot._request_nvidia_completion",
        lambda *args: (_ for _ in ()).throw(RuntimeError("inference timed out")),
    )
    result = copilot.respond(
        "My card payment was declined", llm_model=DEFAULT_NVIDIA_MODEL, llm_api_key="test",
    )
    assert result.status == "generation_fallback"
    assert result.generation_status == "unavailable"
    assert result.grounding_issues == ("inference timed out",)
    assert any(f"[{hit.faq.id}]" in result.answer for hit in result.retrieved)


def test_function_call_requires_demo_reference_even_if_checkbox_enabled(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)

    def valid_grounded_answer(*args):
        return "A card payment can be declined when the available balance is insufficient, a spending limit has been reached, or the merchant does not accept the card. [HELP-CARD-02]"

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", valid_grounded_answer)
    monkeypatch.setattr("fintech_support.copilot._select_nvidia_tool", lambda *args:
                        pytest.fail("No synthetic ID was provided for a lookup"))
    result = copilot.respond(
        "My card payment was declined", llm_model=DEFAULT_NVIDIA_MODEL,
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


def test_nvidia_nim_tool_call_uses_verified_synthetic_record(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    requests = []

    def fake_completion(model, api_key, payload):
        requests.append(payload)
        if "tools" in payload:
            return {"tool_calls": [{"type": "function", "function": {
                "name": "lookup_demo_transaction",
                "arguments": json.dumps({"reference": "TX-CARD-001"}),
            }}]}
        return {"content": "If it is frozen, follow the in-app steps to unfreeze it before trying again. [HELP-CARD-01]"}

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", fake_completion)
    response = copilot.respond(
        "Why did my payment fail?",
        record_reference="TX-CARD-001",
        llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
    )
    assert response.tool_routing == "nvidia_nim_function_call"
    assert response.generation_status == "accepted"
    assert response.tool_result.reason_code == "CARD_FROZEN"
    assert "Reason: the card was frozen." in response.answer
    assert "[HELP-CARD-01]" in response.answer
    assert requests[0]["tools"][0]["function"]["name"] == "lookup_demo_transaction"
    assert requests[0]["tool_choice"]["function"]["name"] == "lookup_demo_transaction"
    assert "Reason: the card was frozen." in requests[1]["messages"][1]["content"]
    assert len(requests) == 2


def test_nvidia_nim_tool_selection_uses_output_budget(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)
    payloads = []

    def completion(model, api_key, payload):
        payloads.append(payload)
        if "tools" in payload:
            return {"tool_calls": [{"type": "function", "function": {
                "name": "lookup_demo_transaction",
                "arguments": json.dumps({"reference": "TX-CARD-001"}),
            }}]}
        return {"content": "If it is frozen, follow the in-app steps to unfreeze it before trying again. [HELP-CARD-01]"}

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", completion)
    result = copilot.respond(
        "Why was TX-CARD-001 declined?", llm_model=DEFAULT_NVIDIA_MODEL,
        llm_api_key="test", llm_select_tool=True,
    )
    assert result.tool_routing == "nvidia_nim_function_call"
    assert payloads[0]["max_tokens"] == 1024


def test_nvidia_nim_tool_call_normalizes_reference_format(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)

    def completion(model, api_key, payload):
        if "tools" in payload:
            return {"tool_calls": [{"type": "function", "function": {
                "name": "lookup_demo_transaction",
                "arguments": json.dumps({"reference": " tx-card-001 "}),
            }}]}
        return {"content": "If it is frozen, follow the in-app steps to unfreeze it before trying again. [HELP-CARD-01]"}

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", completion)
    result = copilot.respond(
        "Why did my payment fail?", record_reference="TX-CARD-001",
        llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
    )
    assert result.tool_result.reference == "TX-CARD-001"
    assert result.tool_routing == "nvidia_nim_function_call"


@pytest.mark.parametrize("tool,reference", [
    ("lookup_demo_card", "TX-CARD-001"),
    ("lookup_demo_transaction", "TX-CARD-002"),
    ("lookup_demo_transaction", "TX-CARD-001 extra"),
])
def test_nvidia_nim_tool_call_rejects_wrong_selection(monkeypatch, tmp_path, tool, reference):
    copilot = make_tool_copilot(tmp_path)
    calls = []

    def fake_completion(model, api_key, payload):
        calls.append(payload)
        return {"tool_calls": [{"type": "function", "function": {
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
    {"tool_calls": ["invalid"]},
    {"tool_calls": [{"function": {"name": "lookup_demo_transaction", "arguments": "bad"}}]},
    {"tool_calls": [
        {"function": {"name": "lookup_demo_transaction", "arguments": "{}"}},
        {"function": {"name": "lookup_demo_transaction", "arguments": "{}"}},
    ]},
])
def test_nvidia_nim_tool_call_rejects_malformed_responses(monkeypatch, tmp_path, bad_message):
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


def test_invalid_reference_fails_before_nvidia_nim_call(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path)

    def should_not_call(*args, **kwargs):
        pytest.fail("NVIDIA NIM should not be called for an invalid demo reference")

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_message", should_not_call)
    with pytest.raises(ValueError, match="TX-xxx"):
        copilot.respond(
            "My card payment failed", record_reference="CARD-001",
            llm_model="demo-model", llm_api_key="fake-key", llm_select_tool=True,
        )


def test_rejected_generation_falls_back_and_preserves_audit(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path, "pending_transfer")
    invented = "Your transfer is completed for EUR 500. [HELP-TRANSFER-01]"
    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", lambda *args: invented)
    result = copilot.respond("My transfer TX-TRF-001 is pending", llm_model="test", llm_api_key="test")
    assert result.status == "grounding_fallback"
    assert result.generated_answer == invented
    assert result.grounding_issues
    assert "EUR 500" not in result.answer
    assert "Transfer TX-TRF-001 is pending." in result.answer


def test_not_found_does_not_ask_llm_to_infer_record(monkeypatch, tmp_path):
    copilot = make_tool_copilot(tmp_path, "pending_transfer")

    def forbidden(*args):
        pytest.fail("A missing record must use deterministic guidance")

    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", forbidden)
    result = copilot.respond("My transfer TX-TRF-999 is pending", llm_model="test", llm_api_key="test")
    assert result.tool_result.found is False
    assert "No matching transfer was found" in result.answer


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
