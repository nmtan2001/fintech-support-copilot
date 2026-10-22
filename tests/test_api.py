import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import api
from fintech_support.copilot import Prediction, SupportCopilot
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.retrieval import FAQRetriever, load_faqs


def test_health():
    assert api.health()["data"] == "synthetic_only"
    assert any(route.path == "/support" for route in api.app.routes)


def test_support_api_returns_tool_result(monkeypatch, tmp_path):
    class StubClassifier:
        def predict(self, question):
            return Prediction("declined_card_payment", 0.94)

    database = tmp_path / "demo.sqlite"
    initialize_demo_database(database, api.ROOT / "data/transactions/demo_records.json")
    copilot = SupportCopilot(
        StubClassifier(),
        FAQRetriever(load_faqs(api.ROOT / "data/faq/demo_policies.json")),
        DemoBankingTools(database),
    )
    monkeypatch.setattr(api, "get_copilot", lambda: copilot)
    body = api.support(api.SupportRequest(
        question="My card payment was declined", record_reference="TX-CARD-001"
    ))
    assert body["tool_result"]["status"] == "declined"
    assert body["tool_result"]["reason_code"] == "CARD_FROZEN"
    assert body["synthetic_only"] is True
    assert body["retrieved_sources"]
    with pytest.raises(HTTPException) as exc:
        api.support(api.SupportRequest(
            question="My payment declined", record_reference="CARD-001"
        ))
    assert exc.value.status_code == 400
    with pytest.raises(ValidationError):
        api.SupportRequest(question="")


def test_llm_tool_selection_requires_llm_generation():
    with pytest.raises(HTTPException) as exc:
        api.support(api.SupportRequest(question="My card was declined", llm_select_tool=True))
    assert exc.value.status_code == 400
    assert "use_llm" in exc.value.detail
