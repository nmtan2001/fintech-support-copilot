import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fintech_support.copilot import Prediction, SupportCopilot, IntentClassifier, grounded_answer
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parents[1]


class StubClassifier:
    def __init__(self, intent, confidence=0.95):
        self.prediction = Prediction(intent, confidence)

    def predict(self, question):
        return self.prediction


def make_copilot(intent, confidence=0.95):
    retriever = FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json"))
    return SupportCopilot(StubClassifier(intent, confidence), retriever)


def test_returns_synthetic_policy_with_source():
    response = make_copilot("lost_or_stolen_card").respond("I lost my card. Can I freeze it?")
    assert response.status == "answered"
    assert "[demo-lost-01]" in response.answer
    assert "cannot freeze cards" in response.answer
    assert all(hit.faq.source.startswith("Synthetic demo policy") for hit in response.retrieved)


def test_unsupported_intent_abstains():
    response = make_copilot("exchange_rate").respond("How do I get the exchange rate?")
    assert response.status == "unsupported_intent"
    assert not response.retrieved
    assert "official support" in response.answer


def test_low_confidence_abstains_without_llm():
    response = make_copilot("declined_card_payment", confidence=0.3).respond(
        "My card payment was rejected", min_confidence=0.5, llm_model="fake-model"
    )
    assert response.status == "low_confidence"
    assert not response.retrieved


def test_load_existing_baseline_model():
    model_path = ROOT / "outputs/baseline/model.joblib"
    if not model_path.is_file():
        pytest.skip("Baseline artifact is not present")
    classifier, name = IntentClassifier.from_artifacts(
        transformer_dir=ROOT / "outputs/distilbert/model",
        baseline_path=model_path,
        prefer_transformer=False,
    )
    prediction = classifier.predict("My card has been stolen")
    assert isinstance(prediction.intent, str)
    assert 0 <= prediction.confidence <= 1
    assert name == "TF-IDF + Logistic Regression"


def test_rejects_llm_citations_outside_retrieved_sources(monkeypatch):
    hits = tuple(make_copilot("lost_or_stolen_card").retriever.search(
        "My card was stolen", intent="lost_or_stolen_card"
    ))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
                output_text="Please freeze it. [demo-card-99]"
            ))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    with pytest.raises(RuntimeError, match="outside the retrieved FAQ"):
        grounded_answer("My card was stolen", hits, llm_model="fake")
