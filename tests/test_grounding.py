from pathlib import Path

import pytest

from fintech_support.copilot import grounded_answer
from fintech_support.demo_tools import ToolResult
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def hits():
    return tuple(FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json")).search(
        "My transfer is pending", intent="pending_transfer"
    ))


@pytest.mark.parametrize("candidate", [
    "Your transfer is completed. [demo-transfer-01]",
    "The transfer amount is EUR 500. [demo-transfer-01]",
    "Your transfer was delayed by fraud screening. [demo-transfer-01]",
    "Your transfer arrives tomorrow. [demo-transfer-01]",
    "A pending status means completion has not been confirmed. [demo-transfer-03]",
    "A pending status means completion has not been confirmed.",
    "completion has not been confirmed [demo-transfer-01]",
])
def test_rejects_unsupported_claims_even_with_valid_citation(monkeypatch, hits, candidate):
    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", lambda *args: candidate)
    with pytest.raises(RuntimeError, match="ground|citation|evidence"):
        grounded_answer("My transfer is pending", hits, llm_model="test", llm_api_key="test")


def test_accepts_exact_faq_sentence(monkeypatch, hits):
    candidate = "A pending status means completion has not been confirmed. [demo-transfer-01]"
    monkeypatch.setattr("fintech_support.copilot._request_nvidia_completion", lambda *args: candidate)
    assert grounded_answer("My transfer is pending", hits, llm_model="test", llm_api_key="test") == candidate


@pytest.mark.parametrize("result", [
    ToolResult("lookup_demo_card", "TX-TRF-001", True, "pending", "PROCESSING"),
    ToolResult("lookup_demo_transaction", "TX-TRF-002", True, "pending", "PROCESSING"),
    ToolResult("lookup_demo_transaction", "TX-TRF-001", False, "pending", "PROCESSING"),
    ToolResult("lookup_demo_transaction", "TX-TRF-001", True, None, "PROCESSING"),
    ToolResult("lookup_demo_transaction", "TX-TRF-001", True, "pending", ""),
])
def test_tool_result_must_match_authorized_lookup(result):
    with pytest.raises(RuntimeError, match="lookup|result"):
        result.validate_for("pending_transfer", "TX-TRF-001")
