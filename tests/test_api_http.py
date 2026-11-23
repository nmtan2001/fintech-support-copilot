"""Exercise routing, validation and lifespan without downloading a model."""
import pytest
from fastapi.testclient import TestClient

import api
from fintech_support.copilot import Prediction, SupportCopilot
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.retrieval import FAQRetriever, load_faqs


@pytest.fixture
def client(monkeypatch, tmp_path):
    class Classifier:
        def predict(self, question):
            if question == "Where is my transfer":
                return Prediction("pending_transfer", 0.83)
            return Prediction("declined_card_payment", 0.94)

        def predict_many(self, questions, *, batch_size=16):
            return [self.predict(question) for question in questions]

    database = tmp_path / "demo.sqlite"
    initialize_demo_database(database, api.ROOT / "data/transactions/demo_records.json")
    tools = DemoBankingTools(database)
    copilot = SupportCopilot(
        Classifier(), FAQRetriever(load_faqs(api.ROOT / "data/faq/demo_policies.json")), tools
    )
    monkeypatch.setattr(api, "get_copilot", lambda: copilot)
    monkeypatch.setattr(api, "get_demo_tools", lambda: tools)
    with TestClient(api.app) as session:
        yield session


def test_release_identity(client):
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json()["version"] == api.APP_VERSION
    assert response.json()["commit"] == api.APP_COMMIT


def test_lifespan_readiness(client):
    assert client.get("/ready").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_not_ready_after_shutdown(client):
    # Explicit state checks ensure shutdown clears readiness.
    with TestClient(api.app) as session:
        assert session.get("/ready").status_code == 200
    assert client.get("/ready").status_code == 503


def test_startup_failure_is_not_ready(monkeypatch):
    def fail():
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(api, "get_copilot", fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        with TestClient(api.app):
            pass
    assert api.app.state.ready is False


def test_support_http_contract(client):
    response = client.post("/support", json={
        "question": "My card payment was declined", "record_reference": "TX-CARD-001"
    })
    assert response.status_code == 200
    body = response.json()
    assert body["synthetic_only"] is True
    assert body["tool_result"]["reason_code"] == "CARD_FROZEN"
    assert body["retrieved_sources"]
    assert body["retrieved_sources"][0]["content"]
    assert body["retrieved_sources"][0]["title"]
    assert client.get("/faqs").json()["count"] > 0
    assert client.get("/demo-status/TX-CARD-001").json()["status"] == "declined"


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x", "min_confidence": 2}])
def test_invalid_http_input(client, payload):
    assert client.post("/support", json=payload).status_code == 422


def test_unavailable_response_hides_internal_details(client, monkeypatch):
    def fail():
        raise OSError("private/path/secret-value")
    monkeypatch.setattr(api, "get_copilot", fail)
    response = client.post("/support", json={"question": "My card was declined"})
    assert response.status_code == 503
    assert "secret-value" not in response.text


def test_capabilities(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert "declined_card_payment" in body["lookup_intents"]
    assert body["synthetic_only"] is True


def test_nvidia_is_disabled_even_with_credentials(client, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "unused-test-key")
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(api, "get_secret", lambda *args, **kwargs: None)
    assert client.get("/capabilities").json()["generation_available"] is False
    response = client.post("/support", json={"question": "My card declined", "use_llm": True})
    assert response.status_code == 400
    assert "disabled" in response.json()["detail"]


def test_http_client_preserves_evidence_and_lookup(client):
    from fintech_support.api_client import SupportAPIClient

    remote = SupportAPIClient("http://testserver", transport=client)
    result = remote.respond("My card payment was declined", record_reference="TX-CARD-001")
    assert result.tool_result.reason_code == "CARD_FROZEN"
    assert result.retrieved[0].faq.content
    assert result.generation_status == "disabled"
    assert remote.request_id


def test_http_client_reports_validation_error(client):
    from fintech_support.api_client import SupportAPIClient

    remote = SupportAPIClient("http://testserver", transport=client)
    with pytest.raises(ValueError, match="conflicts"):
        remote.respond("My TX-CARD-001 card payment declined", record_reference="TX-CARD-002")


def test_http_client_reports_backend_failure(client, monkeypatch):
    from fintech_support.api_client import SupportAPIClient

    def fail():
        raise OSError("private-path")

    monkeypatch.setattr(api, "get_copilot", fail)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        SupportAPIClient("http://testserver", transport=client).respond("My card declined")


def test_request_metrics_and_identifier(client):
    response = client.get("/faqs")
    assert len(response.headers["x-request-id"]) == 32
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'route="/faqs"' in metrics.text
    assert "copilot_http_requests_total" in metrics.text


def test_metrics_use_route_templates_without_record_identifiers(client):
    client.get("/demo-status/TX-CARD-001")
    metrics = client.get("/metrics").text
    assert 'route="/demo-status/{reference}"' in metrics
    assert "TX-CARD-001" not in metrics


def test_unavailable_demo_lookup_returns_sanitized_503(client, monkeypatch):
    def fail():
        raise RuntimeError("private database diagnostics")
    monkeypatch.setattr(api, "get_demo_tools", fail)
    response = client.get("/demo-status/TX-CARD-001")
    assert response.status_code == 503
    assert "private database diagnostics" not in response.text
    assert response.headers["x-request-id"]


def test_batch_classification_returns_model_identity(client):
    response = client.post("/classify/batch", json={"questions": ["Card declined", "Where is my transfer"]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["model_revision"] == api.MODEL_REVISION
    assert [row["intent"] for row in body["predictions"]] == ["declined_card_payment", "pending_transfer"]
    assert all("confidence_uncalibrated" in row for row in body["predictions"])


@pytest.mark.parametrize("questions", [[], [" "], ["x" * 2001], ["x"] * 17])
def test_batch_classification_rejects_invalid_input(client, questions):
    assert client.post("/classify/batch", json={"questions": questions}).status_code == 422

def test_remote_ui_uses_api_without_loading_local_models(client, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SUPPORT_API_URL", "http://testserver")
    monkeypatch.setenv("NVIDIA_API_KEY", "unused-test-key")
    monkeypatch.setattr("fintech_support.api_client.requests.Session", lambda: client)
    app = AppTest.from_file(api.ROOT / "app.py", default_timeout=120).run()
    assert not app.exception
    assert not any("Response path" == widget.label for widget in app.selectbox)
    assert not {"Intent classifier", "FAQ retrieval strategy"} & {widget.label for widget in app.selectbox}
    assert not app.tabs
    assert not app.slider
    assert len(app.chat_input) == 1
    assert not any(button.label.startswith("Run support analysis") for button in app.button)
