import pytest

from analyze_errors import collect_errors
from fintech_support.copilot import IntentClassifier


class FakeModel:
    classes_ = ["card", "transfer"]

    def predict_proba(self, texts):
        import numpy as np

        return np.array([[0.8, 0.2] if "card" in text else [0.1, 0.9] for text in texts])


def test_batch_predictions():
    classifier = IntentClassifier(FakeModel())
    predictions = classifier.predict_many(["card lost", "transfer delayed"])
    assert [prediction.intent for prediction in predictions] == ["card", "transfer"]
    assert classifier.predict("card lost") == predictions[0]
    with pytest.raises(ValueError, match="batch_size"):
        classifier.predict_many(["card"], batch_size=0)


def test_error_summary_reports_top_confusion():
    result = collect_errors(
        ["first", "second", "third"],
        ["card", "transfer", "card"],
        ["transfer", "transfer", "transfer"],
    )
    assert result["error_count"] == 2
    assert result["top_confusions"] == [{"actual": "card", "predicted": "transfer", "count": 2}]
    with pytest.raises(ValueError, match="equal lengths"):
        collect_errors(["first"], ["card"], [])


def test_high_confidence_errors_sorted_first():
    result = collect_errors(
        ["one", "two", "three"],
        ["card", "transfer", "card"],
        ["transfer", "card", "card"],
        confidences=[0.2, 0.95, 0.7],
    )
    assert result["high_confidence_errors"] == 1
    assert result["examples"][0]["text"] == "two"
