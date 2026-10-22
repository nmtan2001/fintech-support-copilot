"""Shared classification evaluation for baseline and Transformer models."""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score


def score(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        **score(y_true, y_pred),
        "per_class": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
