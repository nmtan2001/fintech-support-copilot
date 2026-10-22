"""TF-IDF + Logistic Regression reference classifier."""

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from fintech_support.data import ensure_known_labels, label_maps, read_split, training_split
from fintech_support.evaluation import evaluate, write_json


def fit_baseline(train, seed: int = 42):
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    model.fit(train["text"], train["category"])
    return model


def run(data_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    source_train = read_split(data_dir, "train")
    test = read_split(data_dir, "test")
    label2id, _ = label_maps(source_train)
    ensure_known_labels(test, label2id)
    train, validation = training_split(source_train, seed=seed)

    model = fit_baseline(train, seed=seed)
    metrics = {
        "model": "tfidf_logistic_regression",
        "seed": seed,
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "validation": evaluate(validation["category"], model.predict(validation["text"])),
        "test": evaluate(test["category"], model.predict(test["text"])),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = run(args.data_dir, args.output_dir, args.seed)
    for split in ("validation", "test"):
        scores = metrics[split]
        print(f"{split}: accuracy={scores['accuracy']:.4f}, macro_f1={scores['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
