"""Gate a pinned model on authored synthetic workflows, without an LLM."""
import json
import math
from pathlib import Path

import api
from evaluate_retrieval import evaluate_cases as evaluate_retrieval
from evaluate_smoke import evaluate_cases


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
    thresholds = manifest["release_thresholds"]
    result["thresholds"] = thresholds
    failed = failures(result["metrics"], thresholds)
    result["gate_passed"] = not failed
    Path("/tmp/model-evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"metrics": result["metrics"], "gate_passed": not failed, "failed_metrics": failed}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
