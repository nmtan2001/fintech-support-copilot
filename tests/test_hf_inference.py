import json
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from fintech_support.copilot import IntentClassifier, Prediction, SupportCopilot
from fintech_support.retrieval import FAQRetriever, load_faqs


def classifier(**kwargs):
    instance, name = IntentClassifier.from_huggingface(token="test-secret", **kwargs)
    assert name == "DistilBERT (HF Inference)"
    return instance


def test_remote_classifier_sends_softmax_request_without_loading_torch(monkeypatch):
    captured = []

    def response(request, timeout):
        captured.append((request, timeout))
        return BytesIO(b'[{"label":"pending_transfer","score":0.96}]')

    monkeypatch.setattr("fintech_support.copilot.urlopen", response)
    result = classifier().predict("My transfer is pending")
    assert result == Prediction("pending_transfer", 0.96)
    request, timeout = captured[0]
    assert request.full_url == (
        "https://router.huggingface.co/hf-inference/models/nmtan2001/fintech-support-distilbert"
    )
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert json.loads(request.data) == {
        "inputs": "My transfer is pending", "parameters": {"top_k": 1, "function_to_apply": "softmax"},
    }
    assert timeout == 30


def test_batch_predictions_preserve_order(monkeypatch):
    responses = iter([
        b'[{"label":"pending_transfer","score":0.96}]',
        b'[[{"label":"lost_or_stolen_card","score":0.91}]]',
    ])
    monkeypatch.setattr("fintech_support.copilot.urlopen", lambda *args, **kwargs: BytesIO(next(responses)))
    instance = classifier()
    assert instance.predict_many(["transfer pending", "card lost"]) == [
        Prediction("pending_transfer", 0.96), Prediction("lost_or_stolen_card", 0.91),
    ]
    assert instance.predict_many([]) == []
    with pytest.raises(ValueError, match="batch_size"):
        instance.predict_many(["transfer pending"], batch_size=0)


@pytest.mark.parametrize("body", [
    b'{}', b'[]', b'not json', b'[{"label":"LABEL_0","score":0.9}]',
    b'[{"label":"pending_transfer","score":1.2}]',
    b'[{"label":"pending_transfer","score":-0.1}]',
    b'[{"label":"pending_transfer","score":NaN}]',
    b'[{"label":"pending_transfer","score":true}]',
    b'[{"label":"","score":0.9}]', b'[{"label":"pending_transfer"}]',
])
def test_invalid_remote_predictions_fail_closed(monkeypatch, body):
    monkeypatch.setattr("fintech_support.copilot.urlopen", lambda *args, **kwargs: BytesIO(body))
    with pytest.raises(RuntimeError, match="invalid.*classification"):
        classifier().predict("transfer pending")


@pytest.mark.parametrize("status,reason", [
    (401, "token"), (403, "permission"), (404, "deployed"), (429, "rate limit"),
    (503, "unavailable"),
])
def test_http_failures_are_actionable_and_never_expose_token(monkeypatch, status, reason):
    def fail(request, timeout):
        raise HTTPError(request.full_url, status, "test-secret", {}, BytesIO(b'test-secret'))

    monkeypatch.setattr("fintech_support.copilot.urlopen", fail)
    with pytest.raises(RuntimeError, match=reason) as exc:
        classifier().predict("transfer pending")
    assert "test-secret" not in str(exc.value)


def test_dns_error_is_reported_without_local_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("DNS lookup failed")

    monkeypatch.setattr("fintech_support.copilot.urlopen", fail)
    with pytest.raises(RuntimeError, match="reach Hugging Face"):
        classifier().predict("transfer pending")


def test_token_is_required_and_environment_aliases_are_supported(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN"):
        IntentClassifier.from_huggingface()
    monkeypatch.setenv("HF_TOKEN", "test-secret")
    assert IntentClassifier.from_huggingface()[0] is not None
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "test-secret")
    assert IntentClassifier.from_huggingface()[0] is not None


@pytest.mark.parametrize("url", [
    "http://example.com", "https://example.com", "https://huggingface.cloud.attacker.example",
    "https://user:password@demo.endpoints.huggingface.cloud",
])
def test_endpoint_url_cannot_send_token_to_untrusted_host(url):
    with pytest.raises(ValueError, match="endpoint"):
        classifier(endpoint_url=url)


def test_dedicated_hf_endpoint_is_supported(monkeypatch):
    captured = []

    def response(request, timeout):
        captured.append(request.full_url)
        return BytesIO(b'[{"label":"pending_transfer","score":0.96}]')

    monkeypatch.setattr("fintech_support.copilot.urlopen", response)
    classifier(endpoint_url="https://demo.eu-west-1.aws.endpoints.huggingface.cloud").predict("transfer")
    assert captured == ["https://demo.eu-west-1.aws.endpoints.huggingface.cloud"]


def test_remote_intent_connects_to_actual_retrieval_and_sqlite(monkeypatch, tmp_path):
    from pathlib import Path
    from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("fintech_support.copilot.urlopen", lambda *args, **kwargs:
                        BytesIO(b'[{"label":"pending_transfer","score":0.96}]'))
    database = tmp_path / "demo.sqlite"
    initialize_demo_database(database, root / "data/transactions/demo_records.json")
    copilot = SupportCopilot(classifier(), FAQRetriever(load_faqs(root / "data/faq/demo_policies.json")),
                             DemoBankingTools(database))
    result = copilot.respond("My transfer TX-TRF-001 is pending")
    assert result.status == "answered"
    assert result.tool_result.status == "pending"
    assert result.tool_result.reason_code == "PROCESSING"
    assert "[demo-transfer-01]" in result.answer


def test_api_hf_backend_does_not_load_local_artifacts(monkeypatch):
    import api

    expected = classifier()
    monkeypatch.setenv("INTENT_BACKEND", "hf")
    monkeypatch.setenv("HF_TOKEN", "test-secret")
    monkeypatch.setattr(IntentClassifier, "from_huggingface", lambda **kwargs: (expected, "HF"))
    monkeypatch.setattr(IntentClassifier, "from_artifacts", lambda **kwargs:
                        pytest.fail("HF mode must not load local artifacts"))
    api.get_copilot.cache_clear()
    try:
        assert api.get_copilot().classifier is expected
    finally:
        api.get_copilot.cache_clear()


def test_streamlit_hf_mode_does_not_load_local_artifacts(monkeypatch):
    import app

    expected = classifier()
    captured = {}

    def load_remote(**kwargs):
        captured.update(kwargs)
        return expected, "DistilBERT (HF Inference)"

    monkeypatch.setattr(IntentClassifier, "from_huggingface", load_remote)
    monkeypatch.setattr(IntentClassifier, "from_artifacts", lambda **kwargs:
                        pytest.fail("HF mode must not load local artifacts"))
    instance, _ = app.load_classifier.__wrapped__(
        "DistilBERT (HF Inference)", "nmtan2001/fintech-support-distilbert", "test-secret", None,
    )
    assert instance is expected
    assert captured["token"] == "test-secret"


def test_evaluation_cli_supports_hf_without_local_artifacts(monkeypatch, tmp_path):
    import evaluate_smoke

    output = tmp_path / "hf.json"
    monkeypatch.setenv("HF_TOKEN", "test-secret")
    monkeypatch.setattr("sys.argv", ["evaluate_smoke.py", "--model", "hf", "--output", str(output)])
    monkeypatch.setattr(IntentClassifier, "from_artifacts", lambda **kwargs:
                        pytest.fail("HF mode must not load local artifacts"))
    monkeypatch.setattr("fintech_support.copilot.urlopen", lambda *args, **kwargs:
                        BytesIO(b'[{"label":"pending_transfer","score":0.96}]'))
    evaluate_smoke.main()
    report = json.loads(output.read_text())
    assert report["classifier_backend"] == "hf"
    assert report["hf_model_id"] == "nmtan2001/fintech-support-distilbert"
    assert report["count"] == 48
    assert "test-secret" not in output.read_text()
