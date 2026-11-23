"""Exercise server-owned history, cited Z.ai answers and the chat workbench."""

import json

import pytest
from fastapi.testclient import TestClient

import api
from fintech_support.assistant import AssistantEngine
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.knowledge import HandbookRetriever
from fintech_support.sessions import InMemorySessionStore
from fintech_support.zai import ZaiClient


@pytest.fixture
def chat_client(monkeypatch, tmp_path):
    from fintech_support.conversations import ConversationService

    calls = []

    def transport(payload, timeout):
        calls.append(payload)
        instruction = payload["messages"][0]["content"]
        data = json.loads(payload["messages"][1]["content"])
        if "Resolve the latest" in instruction:
            if data["message"] == "My payment failed":
                value = {"decision": "clarify", "query": "payment failed", "lookups": [],
                         "clarification": "Was this a card payment or a transfer?"}
            else:
                refs = data["authorized_references"]
                value = {"decision": "answer", "query": "frozen card payment declined", "lookups": [
                    {"reference": ref} for ref in refs[:1]]}
        elif "Explain the answer" in instruction:
            key = next((key for key in data["evidence"] if key.startswith("TOOL:")), next(iter(data["evidence"])))
            quote = data["evidence"][key][:600]
            value = {"claims": [{"text": quote, "evidence_ids": [key], "quotes": {key: quote}}]}
        else:
            value = {"supported": True, "unsupported_claims": []}
        return {"choices": [{"message": {"content": json.dumps(value)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10}}

    database = tmp_path / "demo.sqlite"
    initialize_demo_database(database, api.ROOT / "data/transactions/demo_records.json")
    engine = AssistantEngine(HandbookRetriever.from_path(api.ROOT / "data/knowledge/handbook.json", backend="lexical"),
                             DemoBankingTools(database), ZaiClient(api_key="test-only", transport=transport))
    service = ConversationService(engine, InMemorySessionStore(max_turns=5))
    monkeypatch.setattr(api, "get_conversation_service", lambda: service)
    monkeypatch.setattr(api, "get_copilot", lambda: object())
    with TestClient(api.app) as client:
        yield client, calls, service


def create(client):
    response = client.post("/conversations")
    assert response.status_code == 201
    session = response.json()
    return session["id"], {"Authorization": "Bearer " + session["token"]}


def test_clarification_followup_and_citations_use_server_history(chat_client):
    client, calls, _ = chat_client
    sid, headers = create(client)
    path = f"/conversations/{sid}/messages"
    first = client.post(path, headers=headers, json={"message": "My payment failed"})
    assert first.json()["status"] == "clarification_required"
    assert first.json()["case_state"]["missing_fields"] == ["payment_type"]
    answer = client.post(path, headers=headers, json={"message": "It was my card TX-CARD-001"})
    assert answer.status_code == 200
    assert answer.json()["generation_status"] == "accepted"
    assert answer.json()["tool_results"][0]["reason_code"] == "CARD_FROZEN"
    assert answer.json()["citations"][0]["id"] == "TOOL:TX-CARD-001"
    followup = client.post(path, headers=headers, json={"message": "What about its status?"})
    assert followup.json()["tool_results"][0]["reference"] == "TX-CARD-001"
    assert followup.json()["generation_status"] == "deterministic"
    planner = json.loads(calls[-3]["messages"][1]["content"])
    assert planner["history"][0]["content"] == "My payment failed"
    history = client.get(f"/conversations/{sid}", headers=headers).json()
    assert len(history["messages"]) == 6
    assert "token_hash" not in history and "token" not in history


def test_sessions_require_token_and_forbid_client_history(chat_client):
    client, _, _ = chat_client
    sid, headers = create(client)
    path = f"/conversations/{sid}/messages"
    assert client.post(path, json={"message": "hello"}).status_code == 401
    assert client.post(path, headers={"Authorization": "Bearer wrong"}, json={"message": "hello"}).status_code == 404
    assert client.post(path, headers=headers, json={"message": "hello", "history": []}).status_code == 422
    assert client.post(path, headers=headers, json={"message": " "}).status_code == 422
    assert client.delete(f"/conversations/{sid}", headers=headers).status_code == 204
    assert client.get(f"/conversations/{sid}", headers=headers).status_code == 404


def test_provider_failure_returns_evidence_without_exposing_details(chat_client):
    client, _, service = chat_client

    def fail(*args, **kwargs):
        raise RuntimeError("private-api-key")

    service.engine.provider.chat_json = fail
    sid, headers = create(client)
    response = client.post(f"/conversations/{sid}/messages", headers=headers,
                           json={"message": "My frozen card payment declined"})
    assert response.status_code == 200
    assert response.json()["generation_status"] == "unavailable"
    assert response.json()["citations"]
    assert "private-api-key" not in response.text


def test_rejected_generation_returns_verified_facts_and_diagnostics(chat_client):
    client, _, service = chat_client
    original = service.engine.provider.chat_json

    def reject_verification(messages, **kwargs):
        if "Independently check" in messages[0]["content"]:
            return {"supported": False, "unsupported_claims": [0]}
        return original(messages, **kwargs)

    service.engine.provider.chat_json = reject_verification
    sid, headers = create(client)
    response = client.post(f"/conversations/{sid}/messages", headers=headers,
                           json={"message": "Why did my card payment TX-CARD-001 decline?"})
    assert response.status_code == 200
    body = response.json()
    assert body["fallback_reason"] == "verification_rejected"
    assert body["verification_supported"] is False
    assert body["rejected_claim_indexes"] == [0]
    assert "frozen" in body["answer"]
    assert [citation["id"] for citation in body["citations"]] == ["TOOL:TX-CARD-001"]


def test_chat_ui_calls_api_and_displays_cited_answer(chat_client, monkeypatch):
    from streamlit.testing.v1 import AppTest

    client, _, _ = chat_client
    monkeypatch.setenv("SUPPORT_API_URL", "http://testserver")
    monkeypatch.setattr("fintech_support.api_client.requests.Session", lambda: client)
    app = AppTest.from_file(api.ROOT / "app.py", default_timeout=120).run()
    assert not app.exception
    assert not app.chat_input[0].proto.is_auto_positioned_at_bottom
    records = next(widget for widget in app.selectbox if widget.label == "Choose a sample record")
    assert "TX-CARD-001 · Card payment" in records.options
    next(button for button in app.button if button.label == "Ask about this record").click().run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any("frozen" in item.value for item in app.markdown)
    app.chat_input[0].set_value("What is its status now?").run()
    assert not app.exception
    assert len(app.chat_message) == 4
    next(button for button in app.button if button.label == "New conversation").click().run()
    assert not app.chat_message
    profile = next(widget for widget in app.selectbox if widget.label == "Fictional customer profile")
    profile.set_value("DEMO-BETA").run()
    next(button for button in app.button if button.label == "Track a transfer").click().run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "TX-TRF-006" in app.session_state["chat_messages"][0]["content"]


def test_showcase_verifier_checks_real_conversation_route(chat_client, monkeypatch):
    from scripts.verify_conversation import verify_conversation

    client, _, _ = chat_client
    monkeypatch.setattr("fintech_support.api_client.requests.Session", lambda: client)
    assert verify_conversation("http://testserver", require_generation=True)["verified"] is True


def test_showcase_verifier_rejects_offline_fallback(chat_client, monkeypatch):
    from scripts.verify_conversation import verify_conversation

    client, _, service = chat_client
    service.engine.provider = None
    monkeypatch.setattr("fintech_support.api_client.requests.Session", lambda: client)
    with pytest.raises(RuntimeError, match="generation was not accepted"):
        verify_conversation("http://testserver", require_generation=True)


def test_customer_binding_and_case_queue_are_session_owned(chat_client):
    client, calls, _ = chat_client
    sid, headers = create(client)
    denied = client.post(f"/conversations/{sid}/messages", headers=headers, json={"message": "Check TX-CARD-006"})
    assert denied.status_code == 403
    assert not calls
    response = client.post(f"/conversations/{sid}/messages", headers=headers, json={"message": "I lost my card CARD-001"})
    assert response.json()["route"] == "sensitive"
    assert not calls
    queue = client.get(f"/conversations/{sid}/cases", headers=headers)
    assert len(queue.json()["cases"]) == 1
    assert client.get(f"/conversations/{sid}/cases").status_code == 401
    other_id, other_headers = create(client)
    assert client.get(f"/conversations/{other_id}/cases", headers=other_headers).json()["cases"] == []
    assert client.get(f"/conversations/{sid}/cases", headers=other_headers).status_code == 404
