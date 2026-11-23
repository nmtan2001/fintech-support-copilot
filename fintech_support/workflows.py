"""Explicit routes and fixed steps for the fictional banking support journeys."""

import re
from dataclasses import dataclass, field

from fintech_support.assistant import AssistantResponse
from fintech_support.demo_tools import DemoBankingTools

POLICY_ARTICLES = {
    "lost_card": "KB-CARD-STATUS-01",
    "unrecognized_payment": "KB-DISPUTES-01",
    "dispute": "KB-DISPUTES-02",
}


def reference_intent(reference):
    if reference.startswith("TX-CARD-"):
        return "declined_card_payment"
    if reference.startswith("TX-TRF-"):
        return "pending_transfer"
    return "lost_or_stolen_card"


@dataclass
class CaseState:
    route: str = "policy"
    issue: str = "general_policy"
    active_reference: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    workflow_stage: str = "ready"


def route_request(message, previous, references):
    text = message.casefold().replace("’", "'")
    reference = references[0] if references else previous.get("active_reference")
    followup = bool(re.search(r"\b(it|its|that|status|next|still|yes)\b", text))
    if re.search(r"\b(lost (my |the )?card|my card (is |was )?(lost|stolen)|stolen card)\b", text):
        return CaseState("sensitive", "lost_card", reference, [], "ready_for_review")
    if re.search(r"\b(unauthori[sz]ed|unrecogni[sz]ed|don't recogni[sz]e|do not recogni[sz]e|didn't make|did not make|i was scammed)\b", text):
        return CaseState("sensitive", "unrecognized_payment", reference, [], "ready_for_review")
    if re.search(r"\b(i want to dispute|dispute (my|this|the) (purchase|payment|transaction)|chargeback for)\b", text):
        return CaseState("sensitive", "dispute", reference, [], "ready_for_review")
    if re.search(r"\b(human (agent|support)|speak to (someone|an agent|a person)|escalate)\b", text):
        return CaseState("handoff", previous.get("issue", "general_support"), reference, [], "ready_for_review")
    if previous.get("route") == "sensitive" and followup:
        return CaseState("sensitive", previous["issue"], reference, [], "ready_for_review")
    if references or (reference and followup):
        issue = "card_status" if reference.startswith("CARD-") else (
            "card_payment" if reference.startswith("TX-CARD-") else "transfer")
        return CaseState("record", issue, reference)
    if "payment" in text and not re.search(r"\b(card|transfer)\b", text) and re.search(r"\b(my|failed)\b", text):
        return CaseState("clarify", "payment", None, ["payment_type"], "collecting_details")
    if previous.get("missing_fields") == ["payment_type"] and re.search(r"\b(card|transfer)\b", text):
        issue = "card_payment" if "card" in text else "transfer"
        return CaseState("clarify", issue, None, ["record_reference"], "collecting_details")
    if "my" in text and re.search(r"\b(status|record|transaction)\b", text):
        return CaseState("clarify", previous.get("issue", "record"), None, ["record_reference"], "collecting_details")
    return CaseState()


class ScopedBankingTools:
    """Customer and turn scope are enforced before any fixture read."""

    def __init__(self, tools, allowed_references):
        self.tools = tools
        self.allowed = set(allowed_references)
        self.results = {}

    def validate(self, intent, reference):
        name = DemoBankingTools.validate(intent, reference)
        if reference.strip().upper() not in self.allowed:
            raise PermissionError("Reference is not available for this demo profile and active request")
        return name

    def call(self, intent, reference):
        self.validate(intent, reference)
        key = (intent, reference.strip().upper())
        if key not in self.results:
            self.results[key] = self.tools.call(intent, reference)
        return self.results[key]


def sensitive_response(engine, state, tools, references):
    article_id = POLICY_ARTICLES[state.issue]
    hits = [hit for hit in engine.retriever.chunks if hit.article_id == article_id]
    if not hits:
        raise RuntimeError("Required banking workflow policy is missing")
    results = [tools.call(reference_intent(ref), ref) for ref in references]
    return AssistantResponse(
        status="handoff", answer="\n\n".join(f"{hit.text} [{hit.id}]" for hit in hits),
        citations=[engine._citation(hit) for hit in hits],
        generation_status="deterministic", knowledge_version=engine.retriever.version,
        retrieval_degraded=engine.retriever.degraded,
        tool_results=[vars(result) for result in results if result],
        handoff={"sent": False, "issue": state.issue, "references": references,
                 "attempted_steps": ["Provided the fixed fictional banking procedure"],
                 "unresolved_questions": ["Review the reported issue with a human support specialist"]},
    )
