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
        expected_id = case["expected_id"]
        if expected_id is not None and (
            not isinstance(expected_id, str) or not expected_id.strip()
        ):
            raise ValueError("expected_id must be a FAQ id or null for an abstention case")
        hits = retriever.search(case["question"], intent=case["intent"], top_k=3)
        retrieved_ids = [hit.faq.id for hit in hits]
        is_negative = expected_id is None
        hit_at_1 = bool(expected_id and retrieved_ids and retrieved_ids[0] == expected_id)
        hit_at_3 = bool(expected_id and expected_id in retrieved_ids)
        results.append({
            "question": case["question"],
            "intent": case["intent"],
            "expected_id": expected_id,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": [
                {
                    "id": hit.faq.id,
                    "ranking_score": hit.score,
                    "tfidf_cosine": hit.tfidf_score,
                    "semantic_cosine": hit.semantic_score,
                }
                for hit in hits
            ],
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "abstained": not retrieved_ids,
            "case_correct": not retrieved_ids if is_negative else hit_at_1,
        })
    relevant = [row for row in results if row["expected_id"] is not None]
    negative = [row for row in results if row["expected_id"] is None]
    correct_top1 = sum(row["hit_at_1"] for row in relevant)
    return {
        "backend": retriever.backend,
        "count": len(results),
        "relevant_count": len(relevant),
        "negative_count": len(negative),
        "recall_at_1": correct_top1 / len(relevant) if relevant else None,
        "recall_at_3": sum(row["hit_at_3"] for row in relevant) / len(relevant) if relevant else None,
        "precision_at_1": correct_top1 / len(results),
        "negative_abstention_rate": (
            sum(row["abstained"] for row in negative) / len(negative) if negative else None
        ),
        "case_accuracy": sum(row["case_correct"] for row in results) / len(results),
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faq", type=Path, default=Path("data/faq/demo_policies.json"))
    parser.add_argument("--cases", type=Path, default=Path("data/evaluation/faq_cases.json"))
    parser.add_argument("--backend", choices=["tfidf", "semantic", "hybrid"], default="tfidf")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/retrieval.json"))
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    result = evaluate_cases(FAQRetriever(load_faqs(args.faq), backend=args.backend), cases)
    write_json(args.output, result)
    print(
        f"{result['backend']}: {result['relevant_count']} relevant, "
        f"{result['negative_count']} abstention cases; "
        f"Recall@1={result['recall_at_1']:.4f}, "
        f"Recall@3={result['recall_at_3']:.4f}, "
        f"Precision@1={result['precision_at_1']:.4f}, "
        f"negative abstention={result['negative_abstention_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
