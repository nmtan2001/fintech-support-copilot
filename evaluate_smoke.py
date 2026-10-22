"""Run hand-authored end-to-end checks for the demo copilot."""

import argparse
import json
import os
from pathlib import Path

from fintech_support.copilot import IntentClassifier, SupportCopilot, ToolSelectionError
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.evaluation import score, write_json
from fintech_support.grounding import assess_answer
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parent


CHECKS = (
    "intent_match", "status_match", "source_match", "citation_match", "tool_match",
    "argument_match", "tool_data_match", "faithfulness_match", "delivered_grounding",
    "input_rejected",
)


def _validate_case(case: dict) -> None:
    required = {"question", "expected_intent", "expected_status", "expected_faq_id"}
    if not isinstance(case, dict) or not required <= case.keys():
        raise ValueError(f"Smoke case requires fields: {sorted(required)}")
    if case.get("expected_tool"):
        tool_fields = {"expected_found", "expected_record_status", "expected_reason_code"}
        if not tool_fields <= case.keys() or not (
            case.get("expected_reference") or case.get("record_reference")
        ):
            raise ValueError("Tool cases require an explicit expected reference and expected record data")
    if "expected_error" in case and not (
        isinstance(case["expected_error"], str) and case["expected_error"].strip()
    ):
        raise ValueError("expected_error must be a nonempty validation error substring")


def evaluate_cases(
    copilot: SupportCopilot, cases: list[dict], *, llm_model: str | None = None,
    llm_api_key: str | None = None, llm_select_tool: bool = False,
    min_confidence: float = 0.0,
) -> dict:
    if not cases:
        raise ValueError("Smoke case set must not be empty")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    for case in cases:
        _validate_case(case)
    results = []
    for case in cases:
        checks = dict.fromkeys(CHECKS)
        row = {
            "question": case["question"], "expected_intent": case["expected_intent"],
            "expected_status": case["expected_status"],
            "expected_faq_id": case["expected_faq_id"], "checks": checks,
        }
        expected_tool = case.get("expected_tool")
        expected_reference = case.get("expected_reference", case.get("record_reference"))
        if expected_reference:
            expected_reference = expected_reference.strip().upper()
        expected_error = case.get("expected_error")
        try:
            response = copilot.respond(
                case["question"], record_reference=case.get("record_reference"),
                min_confidence=case.get("min_confidence", min_confidence),
                llm_model=llm_model, llm_api_key=llm_api_key,
                llm_select_tool=llm_select_tool,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            expected_rejection = (
                isinstance(exc, ValueError) and bool(expected_error) and expected_error in str(exc)
            )
            row.update(error=str(exc), error_type=type(exc).__name__, passed=expected_rejection)
            checks["input_rejected"] = expected_rejection
            if not expected_error:
                for check in ("status_match", "source_match", "citation_match", "tool_match"):
                    checks[check] = False
                if expected_tool:
                    checks["argument_match"] = False
                    checks["tool_data_match"] = False
                if isinstance(exc, ToolSelectionError):
                    row["proposed_tool"] = exc.name
                    row["proposed_arguments"] = exc.arguments
                    checks["tool_match"] = exc.name == expected_tool
                    if expected_tool:
                        checks["argument_match"] = exc.arguments == {"reference": expected_reference}
            results.append(row)
            continue
        ids = [hit.faq.id for hit in response.retrieved]
        tool = response.tool_result
        candidate = response.generated_answer if response.generated_answer is not None else response.answer
        assessment = assess_answer(
            candidate, response.retrieved, tool_result=tool, allow_fallback=True,
        )
        delivered = assess_answer(
            response.answer, response.retrieved, tool_result=tool, allow_fallback=True,
        )
        expected_id = case["expected_faq_id"]
        checks.update(
            intent_match=response.intent == case["expected_intent"],
            status_match=response.status == case["expected_status"],
            source_match=expected_id in ids if expected_id else not ids,
            citation_match=assessment.citations_valid and (
                not expected_id or any(f"[{source_id}]" in candidate for source_id in ids)
            ),
            tool_match=(tool.name if tool else None) == expected_tool,
            faithfulness_match=assessment.faithful,
            delivered_grounding=delivered.faithful,
            input_rejected=False if expected_error else None,
        )
        if expected_tool:
            checks["argument_match"] = tool is not None and tool.reference == expected_reference
            checks["tool_data_match"] = (
                tool is not None and tool.name == expected_tool and
                tool.reference == expected_reference and tool.found == case["expected_found"] and
                tool.status == case["expected_record_status"] and
                tool.reason_code == case["expected_reason_code"] and
                tool.render() in response.answer
            )
        else:
            checks["tool_data_match"] = tool is None
        row.update(
            predicted_intent=response.intent, confidence_uncalibrated=response.confidence,
            actual_status=response.status, retrieved_faq_ids=ids,
            hit_at_1=bool(ids and ids[0] == expected_id) if expected_id else None,
            hit_at_3=expected_id in ids if expected_id else None,
            tool=tool.name if tool else None, tool_reference=tool.reference if tool else None,
            tool_status=tool.status if tool else None, tool_reason_code=tool.reason_code if tool else None,
            tool_routing=response.tool_routing, generation_status=response.generation_status,
            grounding_issues=list(assessment.issues), generated_answer=response.generated_answer,
            answer=response.answer, passed=all(value for value in checks.values() if value is not None),
        )
        results.append(row)

    metrics = {}
    support = {}
    classified = [row for row in results if "predicted_intent" in row]
    classification = score(
        [row["expected_intent"] for row in classified],
        [row["predicted_intent"] for row in classified],
    ) if classified else {}
    for name in ("accuracy", "macro_f1"):
        metrics[name] = classification.get(name)
        support[name] = len(classified)
    for name, field in (("recall_at_1", "hit_at_1"), ("recall_at_3", "hit_at_3")):
        eligible = [row for case, row in zip(cases, results)
                    if case["expected_faq_id"] and not case.get("expected_error")]
        support[name] = len(eligible)
        metrics[name] = sum(bool(row.get(field)) for row in eligible) / len(eligible) if eligible else None
    mapping = {
        "tool_selection_accuracy": "tool_match", "argument_accuracy": "argument_match",
        "tool_data_accuracy": "tool_data_match", "citation_accuracy": "citation_match",
        "faithfulness": "faithfulness_match", "delivered_faithfulness": "delivered_grounding",
    }
    for name, check in mapping.items():
        eligible = [row for case, row in zip(cases, results)
                    if not case.get("expected_error") and row["checks"][check] is not None
                    and (name != "tool_data_accuracy" or case.get("expected_tool"))
                    and (name != "citation_accuracy" or case["expected_faq_id"] or
                         "[" in (row.get("generated_answer") or row.get("answer", "")))]
        support[name] = len(eligible)
        metrics[name] = (sum(row["checks"][check] for row in eligible) / len(eligible)
                         if eligible else None)
    metrics["hallucination_rate"] = (1 - metrics["faithfulness"]
                                     if metrics["faithfulness"] is not None else None)
    support["hallucination_rate"] = support["faithfulness"]
    passed = sum(row["passed"] for row in results)
    metrics["task_success_rate"] = passed / len(results)
    support["task_success_rate"] = len(results)
    generated = [row for row in results if row.get("generated_answer") is not None]
    return {
        "scope": "hand-authored synthetic pipeline cases; not BANKING77 test-set metrics",
        "grounding_method": "complete, attributed FAQ extracts and exact tool rendering; rejects paraphrases",
        "hallucination_rate_definition": "1 - strict extractive faithfulness; includes citation and format violations",
        "count": len(results), "passed": passed, "failed": len(results) - passed,
        "metrics": metrics, "metric_support": support,
        "classification_unobserved": len(results) - len(classified),
        "generation": {
            "evaluated_candidates": len(generated),
            "accepted": sum(row["generation_status"] == "accepted" for row in generated),
            "rejected": sum(row["generation_status"] == "rejected" for row in generated),
            "faithfulness": (sum(row["checks"]["faithfulness_match"] for row in generated) /
                             len(generated)) if generated else None,
        },
        "check_accuracy": {
            check: (sum(row["checks"][check] for row in results if row["checks"][check] is not None) /
                    sum(row["checks"][check] is not None for row in results))
            if any(row["checks"][check] is not None for row in results) else None
            for check in CHECKS
        },
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
    parser.add_argument("--llm-model", help="Optional NVIDIA model ID for an API-based evaluation run")
    parser.add_argument("--llm-tool-calling", action="store_true", help="Let NVIDIA select the read-only demo tool")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true", help="Exit with an error unless every case passes")
    args = parser.parse_args()
    if args.llm_tool_calling and not args.llm_model:
        parser.error("--llm-tool-calling also requires --llm-model")

    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")
    if args.model == "distilbert" and not (ROOT / "outputs/distilbert/model/config.json").is_file():
        parser.error("DistilBERT artifacts are missing; use --model baseline explicitly")

    classifier, name = IntentClassifier.from_artifacts(
        transformer_dir=ROOT / "outputs/distilbert/model",
        baseline_path=ROOT / "outputs/baseline/model.joblib",
        prefer_transformer=args.model == "distilbert",
    )
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    database_path = ROOT / "outputs/demo/demo.sqlite"
    initialize_demo_database(database_path, ROOT / "data/transactions/demo_records.json")
    copilot = SupportCopilot(
        classifier,
        FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json")),
        DemoBankingTools(database_path),
    )
    api_key = os.environ.get("NVIDIA_API_KEY") if args.llm_model else None
    if args.llm_model and not api_key:
        parser.error("NVIDIA_API_KEY must be set for --llm-model")
    result = evaluate_cases(
        copilot, cases, llm_model=args.llm_model, llm_api_key=api_key,
        llm_select_tool=args.llm_tool_calling, min_confidence=args.min_confidence,
    )
    result["classifier"] = name
    result["classifier_backend"] = args.model
    result["min_confidence"] = args.min_confidence
    result["evaluation_mode"] = "live_nvidia" if args.llm_model else "deterministic"
    result["llm"] = args.llm_model or "disabled; deterministic FAQ-grounded responses"
    result["llm_tool_calling"] = args.llm_tool_calling
    write_json(args.output, result)
    print(f"{name}: {result['passed']}/{result['count']} smoke cases passed")
    for metric, value in result["metrics"].items():
        formatted = f"{value:.4f}" if value is not None else "N/A"
        print(f"  {metric}: {formatted} (n={result['metric_support'][metric]})")
    for row in result["cases"]:
        print(
            f"{'PASS' if row['passed'] else 'FAIL'}: "
            f"{row['expected_intent']} -> {row.get('predicted_intent', '<error>')}; "
            f"sources={row.get('retrieved_faq_ids', [])}; "
            f"{row.get('error', '')}"
        )
    if args.strict and result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
