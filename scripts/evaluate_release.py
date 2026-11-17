"""Gate a pinned model on authored synthetic workflows, without an LLM."""
import csv
import hashlib
import json
import math
from pathlib import Path

import api
from evaluate_retrieval import evaluate_cases as evaluate_retrieval
from evaluate_smoke import evaluate_cases
from fintech_support.evaluation import score


def failures(metrics, thresholds):
    return [name for name, minimum in thresholds.items()
            if not isinstance(metrics.get(name), (int, float))
            or not math.isfinite(metrics[name]) or metrics[name] < minimum]


def main():
    root = api.ROOT
    manifest = json.loads((root / "model-manifest.json").read_text())
    copilot = api.get_copilot()
    result = evaluate_cases(copilot, json.loads((root / manifest["evaluation_set"]).read_text()))
    result["model_id"] = api.MODEL_ID
    result["model_revision"] = api.MODEL_REVISION
    result["app_version"] = api.APP_VERSION
    result["app_commit"] = api.APP_COMMIT
    retrieval = evaluate_retrieval(copilot.retriever,
                                  json.loads((root / "data/evaluation/faq_cases.json").read_text()))
    result["retrieval"] = retrieval
    benchmark = manifest["benchmark"]
    data_path = root / benchmark["path"]
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != benchmark["sha256"]:
        raise ValueError("BANKING77 benchmark checksum mismatch")
    with data_path.open(newline="") as source:
        examples = list(csv.DictReader(source))
    if len(examples) != benchmark["examples"]:
        raise ValueError("BANKING77 example count mismatch")
    predictions = copilot.classifier.predict_many([row["text"] for row in examples])
    result["benchmark"] = {**benchmark,
        "metrics": score([row["category"] for row in examples], [row.intent for row in predictions])}
    thresholds = manifest["release_thresholds"]
    result["thresholds"] = thresholds
    failed = ["benchmark." + key for key in failures(result["benchmark"]["metrics"], thresholds)]
    failed += ["workflow." + key for key in failures(result["metrics"], manifest["workflow_thresholds"])]
    result["gate_passed"] = not failed
    Path("/tmp/model-evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"benchmark": result["benchmark"]["metrics"], "workflow": result["metrics"], "gate_passed": not failed, "failed_metrics": failed}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
