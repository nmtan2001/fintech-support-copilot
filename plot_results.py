"""Save classifier comparison and training-curve figures for the course report."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def plot_results(base_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fintech-support-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = json.loads((base_dir / "baseline/metrics.json").read_text(encoding="utf-8"))
    transformer = json.loads((base_dir / "distilbert/metrics.json").read_text(encoding="utf-8"))
    history = json.loads((base_dir / "distilbert/training_history.json").read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = [int(row["epoch"]) for row in history["history"] if "eval_macro_f1" in row]
    values = [row["eval_macro_f1"] for row in history["history"] if "eval_macro_f1" in row]
    if not epochs:
        raise ValueError("No epoch-level validation results in training history")
    ax.plot(epochs, values, marker="o", label="DistilBERT validation Macro-F1")
    ax.axhline(baseline["validation"]["macro_f1"], linestyle="--", label="Baseline validation Macro-F1")
    ax.set(xlabel="Training epoch", ylabel="Macro-F1", title="BANKING77 validation performance")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    curve_path = out_dir / "training_curve.png"
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    splits = ("validation", "test")
    categories = ("Accuracy", "Macro-F1")
    positions = list(range(len(splits) * len(categories)))
    metrics = ("accuracy", "macro_f1")
    base_values = [baseline[split][metric] for split in splits for metric in metrics]
    transformer_values = [transformer[split][metric] for split in splits for metric in metrics]
    ax.bar([p - 0.18 for p in positions], base_values, width=0.36, label="TF-IDF + Logistic Regression")
    ax.bar([p + 0.18 for p in positions], transformer_values, width=0.36, label="DistilBERT")
    ax.set_xticks(positions, [f"{split.title()}\n{category}" for split in splits for category in categories])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("BANKING77: baseline versus fine-tuned DistilBERT")
    ax.legend(loc="lower right")
    fig.tight_layout()
    comparison_path = out_dir / "model_comparison.png"
    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)
    return curve_path, comparison_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    for path in plot_results(args.base_dir, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
