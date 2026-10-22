"""Predict a BANKING77 intent with the saved DistilBERT model."""

import argparse
from pathlib import Path

from fintech_support.copilot import IntentClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Customer question")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/distilbert/model"))
    args = parser.parse_args()

    classifier, _ = IntentClassifier.from_artifacts(
        transformer_dir=args.model_dir,
        baseline_path=Path("outputs/no-baseline-model.joblib"),
    )
    prediction = classifier.predict(args.text)
    print(f"intent={prediction.intent} confidence={prediction.confidence:.4f}")


if __name__ == "__main__":
    main()
