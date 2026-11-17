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
            return Prediction("declined_card_payment", 0.94)

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
