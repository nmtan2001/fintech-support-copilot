"""Server-owned conversation history shared by HTTP and the standalone workbench."""

import os
import re
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from fintech_support.assistant import AssistantEngine, AssistantResponse
from fintech_support.demo_tools import REFERENCE_PATTERN, DemoBankingTools, initialize_demo_database
from fintech_support.knowledge import HandbookRetriever
from fintech_support.sessions import InMemorySessionStore, RedisSessionStore
from fintech_support.settings import get_secret
from fintech_support.zai import Budget, RedisBudget, ZaiClient
from fintech_support.clarifications import CLARIFICATIONS
from fintech_support.limits import RedisTurnLimiter, TurnLimiter
from fintech_support.workflows import ScopedBankingTools, reference_intent, route_request, sensitive_response

ROOT = Path(__file__).resolve().parents[1]


class ConversationService:
    def __init__(self, engine, store, *, customers=None, limiter=None):
        self.engine, self.store = engine, store
        self.limiter = limiter or TurnLimiter()
        document = customers or json.loads((ROOT / "data/transactions/demo_customers.json").read_text())
        if document.get("synthetic_only") is not True:
            raise ValueError("Demo customers must be explicitly synthetic")
        self.customers = {row["id"]: row for row in document["customers"]}

    def create(self, customer_id="DEMO-ALPHA"):
        if customer_id not in self.customers:
            raise ValueError("Unknown fictional customer profile")
        return self.store.create(customer_id=customer_id)

    def profiles(self):
        return [{"id": row["id"], "name": row["name"], "references": row["references"]}
                for row in self.customers.values()]

    def get(self, sid, token):
        return self.store.get(sid, token)

    def delete(self, sid, token):
        self.store.delete(sid, token)

    def cases(self, sid, token):
        return self.store.get(sid, token).get("cases", [])

    def respond(self, sid, token, message, *, references=None):
        references = list(references or [])
        customer_id = self.store.get(sid, token).get("customer_id", "DEMO-ALPHA")
        with self.limiter.turn(customer_id), self.store.turn(sid, token) as state:
            customer = state.get("customer_id", "DEMO-ALPHA")
            supplied = list(dict.fromkeys(match.upper() for match in REFERENCE_PATTERN.findall(message)))
            explicit = [ref.strip().upper() for ref in references]
            if explicit and supplied and not set(supplied).issubset(explicit):
                raise ValueError("Explicit references conflict with message references")
            supplied = list(dict.fromkeys(supplied + explicit))
            if len(supplied) > 3:
                raise ValueError("At most three references per turn")
            if any(ref not in self.customers[customer]["references"] for ref in supplied):
                raise PermissionError("Reference is not available for this demo profile")
            case = route_request(message, state.get("case_state", {}), supplied)
            selected = supplied or ([case.active_reference] if case.active_reference else [])
            scoped = ScopedBankingTools(self.engine.tools, selected)
            engine = copy.copy(self.engine)
            engine.tools = scoped
            if case.route == "clarify":
                response = AssistantResponse("clarification_required", CLARIFICATIONS[case.missing_fields[0]],
                                             generation_status="deterministic", knowledge_version=engine.retriever.version,
                                             retrieval_degraded=engine.retriever.degraded)
            elif case.route == "sensitive":
                response = sensitive_response(engine, case, scoped, selected)
            elif case.route == "handoff":
                response = AssistantResponse("handoff", "This request needs a human support review.",
                                             handoff={"sent": False, "issue": message, "references": selected},
                                             generation_status="deterministic", knowledge_version=engine.retriever.version)
            else:
                verified = [scoped.call(reference_intent(ref), ref) for ref in selected] if case.route == "record" else []
                status_only = verified and re.search(r"\b(status|check)\b", message, re.IGNORECASE) and not re.search(
                    r"\b(why|how|explain|next|should)\b", message, re.IGNORECASE)
                if status_only:
                    response = AssistantResponse("answered", "", generation_status="deterministic",
                                                 knowledge_version=engine.retriever.version,
                                                 retrieval_degraded=engine.retriever.degraded)
                else:
                    response = engine.respond(message, history=state["messages"], references=selected)
                for result in verified:
                    if result:
                        row = asdict(result)
                        if row not in response.tool_results:
                            response.tool_results.append(row)
                        key = "TOOL:" + result.reference
                        if not any(cite["id"] == key for cite in response.citations):
                            response.citations.append({"id": key, "article_id": None, "title": "Verified synthetic lookup",
                                                       "text": result.render(), "source": "Read-only synthetic SQLite record"})
                        if result.render() not in response.answer:
                            response.answer = (result.render() + f" [{key}]\n\n" + response.answer).strip()
            content = message
            if references:
                content += "\nSynthetic references supplied by user: " + ", ".join(references)
            state["messages"].extend([
                {"role": "user", "content": content},
                {"role": "assistant", "content": response.answer},
            ])
            state["references"].extend(match.upper() for match in REFERENCE_PATTERN.findall(content))
            if response.status == "clarification_required":
                case.route = "clarify"
                case.workflow_stage = "collecting_details"
                case.missing_fields = case.missing_fields or ["issue_details"]
            if response.handoff:
                if case.route != "sensitive":
                    case.route = "handoff"
                identity = f"{sid}:{case.issue}"
                case_id = "CASE-" + hashlib.sha256(identity.encode()).hexdigest()[:16].upper()
                queued = {"case_id": case_id, "customer_id": customer, "issue": case.issue,
                          "status": "awaiting_demo_review", "references": selected,
                          "summary": response.handoff, "recent_messages": copy.deepcopy(state["messages"][-8:]),
                          "updated_at": time.time()}
                cases = state.setdefault("cases", [])
                cases[:] = [row for row in cases if row["case_id"] != case_id] + [queued]
                response.handoff = {**response.handoff, "case_id": case_id, "queued": True,
                                    "status": "awaiting_demo_review", "sent": False}
                case.workflow_stage = "queued_for_review"
            state["case_state"] = asdict(case)
            result = {**asdict(response), "route": case.route, "case_state": asdict(case), "customer_id": customer}
        return result


def create_service(root: Path, *, tools=None, api_key=None):
    backend = os.environ.get("HANDBOOK_BACKEND", "lexical")
    retriever = HandbookRetriever.from_path(root / "data/knowledge/handbook.json", backend=backend,
                                            model_manifest_path=root / "retrieval-manifest.json")
    if tools is None:
        database = Path(os.environ.get("DEMO_DATABASE_PATH", root / "outputs/demo/demo.sqlite"))
        initialize_demo_database(database, root / "data/transactions/demo_records.json")
        tools = DemoBankingTools(database)
    key = api_key or get_secret("ZAI_API_KEY", root=root)
    redis_url = os.environ.get("REDIS_URL")
    store = RedisSessionStore(redis_url) if redis_url else InMemorySessionStore()
    budget = RedisBudget(store.client, key="copilot:budget:" + os.environ.get("SHOWCASE_RUN_ID", "runtime")) if redis_url else Budget()
    provider = ZaiClient(api_key=key, budget=budget) if key else None
    limiter = RedisTurnLimiter(store.client) if redis_url else TurnLimiter()
    return ConversationService(AssistantEngine(retriever, tools, provider), store, limiter=limiter)
