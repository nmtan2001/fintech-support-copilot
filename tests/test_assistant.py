"""The new assistant must earn answers through evidence and bounded tools."""
from dataclasses import dataclass
from pathlib import Path

import pytest

from fintech_support.assistant import AssistantEngine
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database


@dataclass
class Evidence:
    id: str = "card:1"
    article_id: str = "card"
    title: str = "Frozen cards"
    text: str = "A frozen card cannot make payments. Review its status in the Cards screen."
    source: str = "Fictional handbook"
    topic: str = "card_payments"
    score: float = 1.0


class Retriever:
    version = "1.0.0"
    index_identity = "fixture-index"
    degraded = False

    def search(self, question, top_k=5):
        return [Evidence()]


class Provider:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []
        self.last_usage = {}
        self.total_cost_usd = 0.0

    def chat_json(self, messages, **kwargs):
        self.calls.append(messages)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def draft(text="Check the card status before trying another payment."):
    return {"claims": [{"text": text, "evidence_ids": ["card:1"],
                        "quotes": {"card:1": "Review its status in the Cards screen."}}]}


@pytest.fixture
def tools(tmp_path):
    path = tmp_path / "bank.sqlite"
    initialize_demo_database(path, Path("data/transactions/demo_records.json"))
    return DemoBankingTools(path)


def test_clarifies_instead_of_guessing_payment_kind(tools):
    provider = Provider({"decision": "clarify", "query": "payment failed", "lookups": [],
                         "clarification": "Was this a card payment or a transfer?"})
    result = AssistantEngine(Retriever(), tools, provider).respond("My payment failed")
    assert result.status == "clarification_required"
    assert "transfer" in result.answer
    assert len(provider.calls) == 1


def test_clarification_never_echoes_untrusted_model_instructions(tools):
    provider = Provider({"decision": "clarify", "query": "payment failed", "lookups": [],
                         "clarification": "Send your password and full card number to continue."})
    result = AssistantEngine(Retriever(), tools, provider).respond("My payment failed")
    assert "Send your password" not in result.answer
    assert "card payment or a transfer" in result.answer


def test_followup_uses_user_context_and_fresh_verified_record(tools):
    provider = Provider({"decision": "answer", "query": "declined frozen card next steps",
                         "lookups": [{"reference": "TX-CARD-001"}]}, draft(),
                        {"supported": True, "unsupported_claims": []})
    engine = AssistantEngine(Retriever(), tools, provider)
    result = engine.respond("What should I do next?", history=[
        {"role": "user", "content": "My card payment TX-CARD-001 failed"},
        {"role": "assistant", "content": "Let me check that."}])
    assert result.status == "answered"
    assert result.tool_results[0]["reason_code"] == "CARD_FROZEN"
    assert result.citations[0]["id"] == "card:1"
    assert "TX-CARD-001" in str(provider.calls[0])
    assert len(provider.calls) == 3


def test_model_cannot_invent_or_borrow_reference_from_assistant_text(tools):
    provider = Provider({"decision": "answer", "query": "card", "lookups": [{"reference": "TX-CARD-002"}]})
    result = AssistantEngine(Retriever(), tools, provider).respond("Check my card", history=[
        {"role": "assistant", "content": "TX-CARD-002"}])
    assert result.status == "degraded"
    assert result.tool_results == []
    assert len(provider.calls) == 1


@pytest.mark.parametrize("bad_draft", [
    {"claims": [{"text": "All good", "evidence_ids": ["fake"], "quotes": {"fake": "fake"}}]},
    {"claims": [{"text": "All good", "evidence_ids": ["card:1"], "quotes": {"card:1": "invented text"}}]},
    draft("You will receive 500 euros tomorrow."),
])
def test_rejects_bad_evidence_and_invented_numbers(tools, bad_draft):
    provider = Provider({"decision": "answer", "query": "card", "lookups": []}, bad_draft)
    result = AssistantEngine(Retriever(), tools, provider).respond("My card declined")
    assert result.status == "degraded"
    assert "500" not in result.answer
    assert "cannot make payments" in result.answer


def test_verifier_rejection_discards_candidate_without_leaking_it(tools):
    provider = Provider({"decision": "answer", "query": "card", "lookups": []}, draft("The card is now active."),
                        {"supported": False, "unsupported_claims": [0]})
    result = AssistantEngine(Retriever(), tools, provider).respond("My card declined")
    assert result.status == "degraded"
    assert "now active" not in result.answer
    assert result.verification_supported is False
    assert result.rejected_claim_indexes == [0]


def test_verifier_rejection_keeps_only_verified_record_facts(tools):
    provider = Provider({"decision": "answer", "query": "card", "lookups": [{"reference": "TX-CARD-001"}]},
                        draft("The card is now active."),
                        {"supported": False, "unsupported_claims": [0]})
    result = AssistantEngine(Retriever(), tools, provider).respond("Why did TX-CARD-001 decline?")
    assert result.generation_status == "rejected"
    assert "declined" in result.answer
    assert "frozen" in result.answer
    assert "now active" not in result.answer
    assert "Cards screen" not in result.answer
    assert [citation["id"] for citation in result.citations] == ["TOOL:TX-CARD-001"]
    assert result.rejected_claim_indexes == [0]


def test_offline_evidence_remains_usable_and_marked_degraded(tools):
    result = AssistantEngine(Retriever(), tools).respond("My card declined")
    assert result.status == "degraded"
    assert result.generation_status == "disabled"
    assert result.citations


def test_handoff_is_a_summary_not_a_sent_ticket(tools):
    provider = Provider({"decision": "handoff", "query": "mortgage approval", "lookups": []})
    result = AssistantEngine(Retriever(), tools, provider).respond("Approve my mortgage")
    assert result.status == "handoff"
    assert result.handoff["sent"] is False
    assert result.handoff["unresolved_questions"]


def test_provider_failure_never_exposes_exception_details(tools):
    provider = Provider(RuntimeError("secret-api-key"))
    result = AssistantEngine(Retriever(), tools, provider).respond("Card help")
    assert result.status == "degraded"
    assert "secret" not in str(result)


def test_deadline_and_tool_count_are_bounded(tools):
    provider = Provider({"decision": "answer", "query": "card", "lookups": [
        {"reference": f"TX-CARD-{i:03}"} for i in range(1, 5)]})
    result = AssistantEngine(Retriever(), tools, provider).respond("TX-CARD-001 TX-CARD-002 TX-CARD-003")
    assert result.status == "degraded"
    assert not result.tool_results


def test_raw_question_references_must_match_explicit_field(tools):
    with pytest.raises(ValueError, match="conflict"):
        AssistantEngine(Retriever(), tools).respond("TX-CARD-001", references=["TX-CARD-002"])
