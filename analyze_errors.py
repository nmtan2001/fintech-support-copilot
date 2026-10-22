"""Inspect incorrect classifications on validation or official test samples."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from fintech_support.copilot import IntentClassifier
from fintech_support.data import read_split, training_split
from fintech_support.evaluation import evaluate, write_json


def collect_errors(
    texts: list[str], actual: list[str], predicted: list[str], *,
    confidences: list[float] | None = None, sample_count: int = 20,
):
    if not (len(texts) == len(actual) == len(predicted)):
        raise ValueError("texts, actual, and predicted must have equal lengths")
    if confidences is not None and len(confidences) != len(texts):
        raise ValueError("confidences must have the same length as texts")
    confusions = Counter((expected, received) for expected, received in zip(actual, predicted) if expected != received)
    mistakes = [
        {"text": text, "actual": expected, "predicted": received,
         **({"confidence_uncalibrated": float(confidences[i])} if confidences is not None else {})}
        for i, (text, expected, received) in enumerate(zip(texts, actual, predicted))
        if expected != received
    ]
    if confidences is not None:
        mistakes.sort(key=lambda row: -row["confidence_uncalibrated"])
    return {
        "error_count": len(mistakes),
        "high_confidence_errors": sum(
            row.get("confidence_uncalibrated", 0.0) >= 0.9 for row in mistakes
        ),
        "top_confusions": [
            {"actual": expected, "predicted": received, "count": count}
            for (expected, received), count in confusions.most_common(20)
        ],
        "examples": mistakes[:sample_count],
    }


def resolve_checkpoint_dir(model_dir: Path) -> Path:
    if (model_dir / "config.json").is_file() and (model_dir / "model.safetensors").is_file():
        return model_dir
    candidates = sorted(
        path.parent for path in model_dir.rglob("config.json")
        if (path.parent / "model.safetensors").is_file()
    ) if model_dir.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No DistilBERT checkpoint found under {model_dir}; expected config.json "
            "and model.safetensors"
        )
    raise ValueError(
        f"Found multiple DistilBERT checkpoints under {model_dir}; pass the exact model directory"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["distilbert", "baseline"], default="distilbert")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/distilbert/model"),
                        help="Checkpoint directory with config.json and model.safetensors")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-csv", type=Path, help="Export every incorrect example as CSV")
    args = parser.parse_args()
    if args.batch_size < 1 or args.sample_count < 1:
        parser.error("--batch-size and --sample-count must be positive")

    if args.model == "distilbert":
        transformer_dir = resolve_checkpoint_dir(args.model_dir)
        baseline_path = Path("outputs/nonexistent-baseline.joblib")
    else:
        transformer_dir = Path("outputs/nonexistent-transformer")
        baseline_path = Path("outputs/baseline/model.joblib")
    classifier, classifier_name = IntentClassifier.from_artifacts(
        transformer_dir=transformer_dir, baseline_path=baseline_path,
        device="auto",
    )
    print(f"Running {classifier_name} on {classifier.device}")
    train = read_split(args.data_dir, "train")
    frame = (
        training_split(train, seed=args.seed)[1]
        if args.split == "validation" else read_split(args.data_dir, "test")
    )
    predictions = classifier.predict_many(frame["text"].tolist(), batch_size=args.batch_size)
    actual = frame["category"].tolist()
    predicted = [prediction.intent for prediction in predictions]
    confidences = [prediction.confidence for prediction in predictions]
    errors = collect_errors(
        frame["text"].tolist(), actual, predicted,
        confidences=confidences, sample_count=args.sample_count,
    )
    result = {
        "classifier": classifier_name,
        "split": args.split,
        "seed": args.seed,
        "metrics": evaluate(actual, predicted),
        **errors,
    }
    output = args.output or Path(
        f"outputs/evaluation/{args.model}_{args.split}_error_analysis.json"
    )
    write_json(output, result)
    csv_path = args.output_csv or output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["text", "actual", "predicted", "confidence_uncalibrated"])
        writer.writeheader()
        for row in zip(frame["text"], actual, predicted, confidences):
            text, expected, received, confidence = row
            if expected != received:
                writer.writerow(dict(text=text, actual=expected, predicted=received,
                                     confidence_uncalibrated=confidence))
    print(f"{classifier_name}, {args.split}: {result['error_count']} errors / {len(frame)} examples")
    print(f"High-confidence errors (>=0.90 uncalibrated): {result['high_confidence_errors']}")
    print(f"Saved report: {output}; incorrect examples: {csv_path}")
    for row in result["top_confusions"][:5]:
        print(f"  {row['actual']} -> {row['predicted']}: {row['count']}")


if __name__ == "__main__":
    main()
