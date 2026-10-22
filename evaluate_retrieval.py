"""Evaluate small FAQ retrieval against manually labeled demo queries."""

import argparse
import json
from pathlib import Path

from fintech_support.evaluation import write_json
from fintech_support.retrieval import FAQRetriever, load_faqs


def evaluate_cases(retriever: FAQRetriever, cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("Evaluation set must not be empty")
    results = []
    for case in cases:
        if not {"question", "intent", "expected_id"} <= case.keys():
            raise ValueError("Evaluation case requires question, intent, expected_id")
        hits = retriever.search(case["question"], intent=case["intent"], top_k=3)
        retrieved_ids = [hit.faq.id for hit in hits]
        results.append({
            "question": case["question"],
            "intent": case["intent"],
            "expected_id": case["expected_id"],
            "retrieved_ids": retrieved_ids,
            "hit_at_1": bool(retrieved_ids and retrieved_ids[0] == case["expected_id"]),
            "hit_at_3": case["expected_id"] in retrieved_ids,
        })
    return {
        "backend": retriever.backend,
        "count": len(results),
        "recall_at_1": sum(row["hit_at_1"] for row in results) / len(results),
        "recall_at_3": sum(row["hit_at_3"] for row in results) / len(results),
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faq", type=Path, default=Path("data/faq/demo_policies.json"))
    parser.add_argument("--cases", type=Path, default=Path("data/evaluation/faq_cases.json"))
    parser.add_argument("--backend", choices=["tfidf", "semantic"], default="tfidf")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/retrieval.json"))
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    result = evaluate_cases(FAQRetriever(load_faqs(args.faq), backend=args.backend), cases)
    write_json(args.output, result)
    print(f"{result['backend']}: {result['count']} cases; Recall@1={result['recall_at_1']:.4f}, Recall@3={result['recall_at_3']:.4f}")


if __name__ == "__main__":
    main()
