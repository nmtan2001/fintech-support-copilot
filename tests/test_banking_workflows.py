"""Finance workflows bind records and queued cases to the demo customer."""

import json
from pathlib import Path

import pytest

from fintech_support.assistant import AssistantEngine
from fintech_support.conversations import ConversationService
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.knowledge import HandbookRetriever
from fintech_support.sessions import InMemorySessionStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def service(tmp_path):
    path = tmp_path / "bank.sqlite"
    initialize_demo_database(path, ROOT / "data/transactions/demo_records.json")
    engine = AssistantEngine(HandbookRetriever.from_path(ROOT / "data/knowledge/handbook.json", backend="lexical"),
                             DemoBankingTools(path))
    return ConversationService(engine, InMemorySessionStore())


def test_payment_clarification_is_structured_and_followup_remembers_reference(service):
    session = service.create()
    reply = service.respond(session["id"], session["token"], "My payment failed")
    assert reply["case_state"]["missing_fields"] == ["payment_type"]
    assert "card payment or a transfer" in reply["answer"]
    reply = service.respond(session["id"], session["token"], "It was my card TX-CARD-001")
    assert reply["route"] == "record"
    assert reply["case_state"]["active_reference"] == "TX-CARD-001"
    reply = service.respond(session["id"], session["token"], "What is its status now?")
    assert reply["tool_results"][0]["reason_code"] == "CARD_FROZEN"


def test_customer_cannot_read_another_profiles_record(service):
    session = service.create(customer_id="DEMO-ALPHA")
    with pytest.raises(PermissionError, match="not available"):
        service.respond(session["id"], session["token"], "Check TX-CARD-006")
    assert service.get(session["id"], session["token"])["turns"] == 0
    other = service.create(customer_id="DEMO-BETA")
    reply = service.respond(other["id"], other["token"], "Check TX-CARD-006")
    assert reply["tool_results"][0]["reference"] == "TX-CARD-006"


@pytest.mark.parametrize("message,issue", [
    ("I lost my card", "lost_card"),
    ("I don't recognize this card payment TX-CARD-001", "unrecognized_payment"),
    ("I want to dispute my purchase TX-CARD-001", "dispute"),
])
def test_sensitive_journey_is_deterministic_and_queued_once(service, message, issue):
    class ForbiddenProvider:
        def chat_json(self, *args, **kwargs):
            pytest.fail("Sensitive workflow must not delegate its steps to an LLM")

    service.engine.provider = ForbiddenProvider()
    session = service.create()
    reply = service.respond(session["id"], session["token"], message)
    assert reply["route"] == "sensitive"
    assert reply["case_state"]["issue"] == issue
    assert reply["citations"]
    assert reply["handoff"]["sent"] is False
    case_id = reply["handoff"]["case_id"]
    again = service.respond(session["id"], session["token"], message)
    assert again["handoff"]["case_id"] == case_id
    cases = service.cases(session["id"], session["token"])
    assert len(cases) == 1
    assert cases[0]["status"] == "awaiting_demo_review"
    assert cases[0]["customer_id"] == "DEMO-ALPHA"
    assert cases[0]["recent_messages"]


def test_general_fraud_policy_question_does_not_open_a_case(service):
    session = service.create()
    reply = service.respond(session["id"], session["token"], "What is your policy about fraud?")
    assert reply["route"] == "policy"
    assert service.cases(session["id"], session["token"]) == []


def test_demo_customer_fixture_covers_every_record_once():
    customers = json.loads((ROOT / "data/transactions/demo_customers.json").read_text())["customers"]
    records = json.loads((ROOT / "data/transactions/demo_records.json").read_text())
    refs = [ref for customer in customers for ref in customer["references"]]
    assert len(refs) == len(set(refs)) == len(records)
    assert set(refs) == {record["reference"] for record in records}
