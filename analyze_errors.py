"""Inspect incorrect classifications on validation or official test samples."""

import argparse
from collections import Counter
from pathlib import Path

from fintech_support.copilot import IntentClassifier
from fintech_support.data import read_split, training_split
from fintech_support.evaluation import evaluate, write_json


def collect_errors(texts: list[str], actual: list[str], predicted: list[str], *, sample_count: int = 20):
    if not (len(texts) == len(actual) == len(predicted)):
        raise ValueError("texts, actual, and predicted must have equal lengths")
    confusions = Counter((expected, received) for expected, received in zip(actual, predicted) if expected != received)
    mistakes = [
        {"text": text, "actual": expected, "predicted": received}
        for text, expected, received in zip(texts, actual, predicted)
        if expected != received
    ]
    return {
        "error_count": len(mistakes),
        "top_confusions": [
            {"actual": expected, "predicted": received, "count": count}
            for (expected, received), count in confusions.most_common(20)
        ],
        "examples": mistakes[:sample_count],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["distilbert", "baseline"], default="distilbert")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/error_analysis.json"))
    args = parser.parse_args()

    if args.model == "distilbert":
        transformer_dir = Path("outputs/distilbert/model")
        baseline_path = Path("outputs/nonexistent-baseline.joblib")
    else:
        transformer_dir = Path("outputs/nonexistent-transformer")
        baseline_path = Path("outputs/baseline/model.joblib")
    classifier, classifier_name = IntentClassifier.from_artifacts(
        transformer_dir=transformer_dir, baseline_path=baseline_path,
    )
    train = read_split(args.data_dir, "train")
    frame = (
        training_split(train, seed=args.seed)[1]
        if args.split == "validation" else read_split(args.data_dir, "test")
    )
    predictions = classifier.predict_many(frame["text"].tolist())
    actual = frame["category"].tolist()
    predicted = [prediction.intent for prediction in predictions]
    result = {
        "classifier": classifier_name,
        "split": args.split,
        "seed": args.seed,
        "metrics": evaluate(actual, predicted),
        **collect_errors(frame["text"].tolist(), actual, predicted),
    }
    write_json(args.output, result)
    print(f"{classifier_name}, {args.split}: {result['error_count']} errors / {len(frame)} examples")
    for row in result["top_confusions"][:5]:
        print(f"  {row['actual']} -> {row['predicted']}: {row['count']}")


if __name__ == "__main__":
    main()
