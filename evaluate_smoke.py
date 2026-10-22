"""Run hand-authored end-to-end checks for the demo copilot."""

import argparse
import json
from pathlib import Path

from fintech_support.copilot import IntentClassifier, SupportCopilot
from fintech_support.evaluation import write_json
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parent


def evaluate_cases(copilot: SupportCopilot, cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("Smoke case set must not be empty")
    results = []
    for case in cases:
        required = {"question", "expected_intent", "expected_status", "expected_faq_id"}
        if not required <= case.keys():
            raise ValueError(f"Smoke case requires fields: {sorted(required)}")
        response = copilot.respond(case["question"])
        retrieved_ids = [hit.faq.id for hit in response.retrieved]
        intent_match = response.intent == case["expected_intent"]
        status_match = response.status == case["expected_status"]
        expected_id = case["expected_faq_id"]
        source_match = (
            expected_id in retrieved_ids if expected_id else not retrieved_ids
        )
        cited_match = (
            not expected_id or f"[{expected_id}]" in response.answer
        )
        results.append({
            "question": case["question"],
            "expected_intent": case["expected_intent"],
            "predicted_intent": response.intent,
            "confidence_uncalibrated": response.confidence,
            "expected_status": case["expected_status"],
            "actual_status": response.status,
            "expected_faq_id": expected_id,
            "retrieved_faq_ids": retrieved_ids,
            "checks": {
                "intent_match": intent_match,
                "status_match": status_match,
                "source_match": source_match,
                "citation_match": cited_match,
            },
            "passed": intent_match and status_match and source_match and cited_match,
            "answer": response.answer,
        })
    return {
        "count": len(results),
        "passed": sum(row["passed"] for row in results),
        "failed": sum(not row["passed"] for row in results),
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["distilbert", "baseline"], default="distilbert")
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data/evaluation/copilot_smoke_cases.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/evaluation/copilot_smoke.json"
    )
    args = parser.parse_args()

    classifier, name = IntentClassifier.from_artifacts(
        transformer_dir=ROOT / "outputs/distilbert/model",
        baseline_path=ROOT / "outputs/baseline/model.joblib",
        prefer_transformer=args.model == "distilbert",
    )
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    copilot = SupportCopilot(
        classifier,
        FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json")),
    )
    result = evaluate_cases(copilot, cases)
    result["classifier"] = name
    result["llm"] = "disabled; deterministic FAQ-grounded responses"
    write_json(args.output, result)
    print(f"{name}: {result['passed']}/{result['count']} smoke cases passed")
    for row in result["cases"]:
        print(
            f"{'PASS' if row['passed'] else 'FAIL'}: "
            f"{row['expected_intent']} -> {row['predicted_intent']}; "
            f"sources={row['retrieved_faq_ids']}"
        )
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
