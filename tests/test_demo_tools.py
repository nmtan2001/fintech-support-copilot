import json
import sqlite3
from pathlib import Path

import pytest

from fintech_support.copilot import Prediction, SupportCopilot
from fintech_support.demo_tools import DemoBankingTools, extract_reference, initialize_demo_database
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parents[1]


class Classifier:
    def __init__(self, intent, confidence=0.99):
        self.prediction = Prediction(intent, confidence)

    def predict(self, question):
        return self.prediction


@pytest.fixture
def tools(tmp_path):
    path = tmp_path / "demo.sqlite"
    initialize_demo_database(path, ROOT / "data/transactions/demo_records.json")
    return DemoBankingTools(path)


def test_synthetic_data_and_read_only_lookup(tools):
    with sqlite3.connect(tools.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM demo_records").fetchone()[0] == 30
    original = tools.call("declined_card_payment", "TX-CARD-001")
    assert original.name == "lookup_demo_transaction"
    assert original.status == "declined"
    assert original.reason_code == "CARD_FROZEN"
    assert tools.call("pending_transfer", "TX-TRF-002").status == "completed"
    assert tools.call("lost_or_stolen_card", "CARD-001").status == "frozen"
    assert tools.call("declined_card_payment", "TX-CARD-999").found is False
    assert tools.call("exchange_rate", "TX-CARD-001") is None
    initialize_demo_database(tools.database_path, ROOT / "data/transactions/demo_records.json")
    assert tools.call("declined_card_payment", "TX-CARD-001") == original


@pytest.mark.parametrize("intent,ref", [
    ("declined_card_payment", "TX-TRF-001"),
    ("pending_transfer", "TX-CARD-001"),
    ("lost_or_stolen_card", "TX-CARD-001"),
    ("pending_transfer", "TX-TRF-001' OR 1=1 --"),
])
def test_wrong_or_malformed_reference_is_rejected(tools, intent, ref):
    with pytest.raises(ValueError):
        tools.call(intent, ref)


def test_extract_reference_requires_one_reference():
    assert extract_reference("Check tx-trf-001 please") == "TX-TRF-001"
    assert extract_reference("No reference supplied") is None
    with pytest.raises(ValueError, match="only one"):
        extract_reference("TX-TRF-001 and TX-TRF-002")


def test_corrupt_database_has_actionable_error(tmp_path):
    path = tmp_path / "corrupt.sqlite"
    path.write_text("not a SQLite database")
    with pytest.raises(RuntimeError, match="read synthetic demo database"):
        DemoBankingTools(path).call("pending_transfer", "TX-TRF-001")


def test_copilot_invokes_tools_only_after_supported_classification(tools):
    retriever = FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json"))
    copilot = SupportCopilot(Classifier("declined_card_payment"), retriever, tools)
    result = copilot.respond("My card payment failed. Check TX-CARD-001 please")
    assert result.tool_result is not None
    assert "status=declined" in result.answer
    assert "reason_code=CARD_FROZEN" in result.answer
    assert "[demo-card-" in result.answer
    assert copilot.respond("My card was declined").tool_result is None
    tool_only = copilot.respond("zqxwjkrn", record_reference="TX-CARD-001")
    assert tool_only.status == "tool_only"
    assert "status=declined" in tool_only.answer

    low = SupportCopilot(Classifier("declined_card_payment", 0.1), retriever, tools)
    assert low.respond("My card was declined, TX-CARD-001", min_confidence=0.5).tool_result is None

    unsupported = SupportCopilot(Classifier("exchange_rate"), retriever, tools)
    assert unsupported.respond("What is the exchange rate? TX-CARD-001").tool_result is None


def test_evaluation_fixture_covers_tool_calls_and_fallbacks():
    cases = json.loads((ROOT / "data/evaluation/copilot_smoke_cases.json").read_text())
    assert 30 <= len(cases) <= 50
    assert any(row.get("expected_tool") == "lookup_demo_transaction" for row in cases)
    assert any(row.get("expected_tool") == "lookup_demo_card" for row in cases)
    assert any(row.get("expected_found") is False for row in cases)
    assert any(row["expected_status"] == "unsupported_intent" for row in cases)
