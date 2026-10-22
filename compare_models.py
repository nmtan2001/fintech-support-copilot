"""Compare saved baseline and Transformer reports without retraining."""

import argparse
import json
from pathlib import Path

from fintech_support.evaluation import write_json


def compare(baseline: dict, transformer: dict, *, split: str = "validation") -> dict:
    left, right = baseline[split], transformer[split]
    baseline_classes = left["per_class"]
    transformer_classes = right["per_class"]
    common = sorted(set(baseline_classes) & set(transformer_classes) - {"accuracy", "macro avg", "weighted avg", "micro avg"})
    differences = [
        {
            "intent": intent,
            "baseline_f1": baseline_classes[intent]["f1-score"],
            "transformer_f1": transformer_classes[intent]["f1-score"],
            "delta_f1": transformer_classes[intent]["f1-score"] - baseline_classes[intent]["f1-score"],
        }
        for intent in common
    ]
    return {
        "split": split,
        "baseline": {key: left[key] for key in ("accuracy", "macro_f1")},
        "transformer": {key: right[key] for key in ("accuracy", "macro_f1")},
        "per_intent": sorted(differences, key=lambda row: row["delta_f1"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("outputs/baseline/metrics.json"))
    parser.add_argument("--transformer", type=Path, default=Path("outputs/distilbert/metrics.json"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/model_comparison.json"))
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    transformer = json.loads(args.transformer.read_text(encoding="utf-8"))
    result = compare(baseline, transformer, split=args.split)
    write_json(args.output, result)
    print(f"{args.split}: baseline macro-F1={result['baseline']['macro_f1']:.4f}, transformer macro-F1={result['transformer']['macro_f1']:.4f}")
    print("Five largest decreases in per-intent F1:")
    for row in result["per_intent"][:5]:
        print(f"  {row['intent']}: {row['delta_f1']:+.4f}")


if __name__ == "__main__":
    main()
