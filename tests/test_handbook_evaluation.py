"""Retrieval measurements must preserve the held-out split."""

import json

from scripts.evaluate_handbook import ROOT, evaluate


def test_retrieval_report_only_uses_answerable_development_cases():
    split = json.loads((ROOT / "data/evaluation/assistant_split.json").read_text())
    report = evaluate()
    ids = {row["id"] for row in report["cases"]}
    assert ids.issubset(set(split["dev"]))
    assert not ids & set(split["test"])
    assert report["count"] > 0
    assert 0 <= report["article_recall_at_5"] <= 1
