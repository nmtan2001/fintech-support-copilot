"""Verify a downloaded DistilBERT checkpoint against the official BANKING77 labels."""

import argparse
import json
from pathlib import Path

from fintech_support.data import label_maps, read_split


def inspect_checkpoint(model_dir: Path, data_dir: Path) -> dict:
    root = model_dir.parent
    required = [
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "tokenizer.json",
        root / "metrics.json",
        root / "training_history.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete downloaded checkpoint: {missing}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((root / "training_history.json").read_text(encoding="utf-8"))
    train = read_split(data_dir, "train")
    test = read_split(data_dir, "test")
    expected_label2id, _ = label_maps(train)
    labels = set(config["id2label"].values())
    if config.get("model_type") != "distilbert" or labels != set(expected_label2id):
        raise ValueError("Checkpoint architecture or intent labels do not match BANKING77")
    if config.get("label2id") != expected_label2id:
        raise ValueError("Checkpoint label-to-ID mapping differs from BANKING77 training labels")
    if config.get("num_labels", len(labels)) != len(labels):
        raise ValueError("Checkpoint num_labels does not match label mapping")
    counts = metrics["counts"]
    if counts["train"] + counts["validation"] != len(train) or counts["test"] != len(test):
        raise ValueError("Metrics sample counts do not match the official dataset splits")
    if not 0 <= history["best_validation_macro_f1"] <= 1:
        raise ValueError("Invalid best validation metric")
    if abs(history["best_validation_macro_f1"] - metrics["validation"]["macro_f1"]) > 1e-6:
        raise ValueError("Saved checkpoint validation score differs from best recorded metric")
    return {
        "model_type": config["model_type"],
        "intent_count": len(labels),
        "train_test_exact_text_overlap": len(set(train["text"]) & set(test["text"])),
        "best_checkpoint": history["best_checkpoint"],
        "best_validation_macro_f1": history["best_validation_macro_f1"],
        "validation": {name: metrics["validation"][name] for name in ("accuracy", "macro_f1")},
        "test": {name: metrics["test"][name] for name in ("accuracy", "macro_f1")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/distilbert/model"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    summary = inspect_checkpoint(args.model_dir, args.data_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
