"""Measure handbook retrieval on the development split without provider calls."""

import argparse
import json
from pathlib import Path

from fintech_support.knowledge import HandbookRetriever

ROOT = Path(__file__).resolve().parents[1]


def evaluate(root=ROOT, *, backend="lexical"):
    split = json.loads((root / "data/evaluation/assistant_split.json").read_text())
    cases = json.loads((root / "data/evaluation/assistant_cases.json").read_text())["cases"]
    development_ids = set(split["dev"])
    retriever = HandbookRetriever.from_path(root / "data/knowledge/handbook.json", backend=backend,
                                            model_manifest_path=root / "retrieval-manifest.json")
    results = []
    for case in cases:
        if case["id"] not in development_ids or case["expected_action"] != "answered":
            continue
        gold = set(case["evidence_ids"])
        query = " ".join(message["content"] for message in case["messages"] if message["role"] == "user")
        hits = retriever.search(query, top_k=5)
        found = {hit.article_id for hit in hits}
        results.append({"id": case["id"], "gold": sorted(gold), "retrieved": sorted(found),
                        "recall_at_5": len(gold & found) / len(gold),
                        "hit_at_1": bool(hits and hits[0].article_id in gold)})
    if not results:
        raise ValueError("No answerable development cases are available")
    return {"partition": "dev", "backend": backend, "count": len(results),
            "query_source": "Concatenated user messages; no LLM rewriting or answer evaluation",
            "article_recall_at_5": sum(row["recall_at_5"] for row in results) / len(results),
            "hit_at_1": sum(row["hit_at_1"] for row in results) / len(results), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("lexical", "hybrid"), default="lexical")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/showcase/retrieval-evaluation.json")
    args = parser.parse_args()
    report = evaluate(backend=args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))


if __name__ == "__main__":
    main()
