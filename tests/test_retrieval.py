import json
from pathlib import Path

import pytest

from evaluate_retrieval import evaluate_cases
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def retriever():
    return FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json"))


def test_unknown_intent_has_no_sources(retriever):
    assert retriever.search("Why is my transfer pending?", intent="exchange_rate") == []


def test_intent_gates_retrieval(retriever):
    results = retriever.search("My card was stolen", intent="lost_or_stolen_card")
    assert results
    assert all("lost_or_stolen_card" in result.faq.intents for result in results)


def test_unrelated_terms_do_not_retrieve(retriever):
    assert retriever.search("zqxwjkrn", intent="lost_or_stolen_card") == []


def test_evaluation_cases_are_faq_grounded(retriever):
    cases = json.loads((ROOT / "data/evaluation/faq_cases.json").read_text())
    result = evaluate_cases(retriever, cases)
    assert result["count"] >= 20
    assert result["recall_at_3"] == 1.0


def test_reject_duplicate_faq_ids(tmp_path):
    original = json.loads((ROOT / "data/faq/demo_policies.json").read_text())
    path = tmp_path / "faq.json"
    path.write_text(json.dumps([original[0], original[0]]))
    with pytest.raises(ValueError, match="Duplicate FAQ id"):
        load_faqs(path)
