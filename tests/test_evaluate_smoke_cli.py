import sys

import evaluate_smoke


def test_cli_uses_model_id_when_local_checkpoint_is_missing(tmp_path, monkeypatch):
    cases = tmp_path / "cases.json"
    cases.write_text("[]", encoding="utf-8")
    model_id = "nmtan2001/fintech-support-distilbert"
    loaded = {}

    def fake_from_artifacts(cls, **kwargs):
        loaded.update(kwargs)
        return object(), "DistilBERT"

    monkeypatch.setattr(
        evaluate_smoke.IntentClassifier,
        "from_artifacts",
        classmethod(fake_from_artifacts),
    )
    def missing_checkpoint(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(evaluate_smoke, "resolve_checkpoint_dir", missing_checkpoint)
    monkeypatch.setattr(evaluate_smoke, "initialize_demo_database", lambda *args: None)
    monkeypatch.setattr(evaluate_smoke, "load_faqs", lambda path: [])
    monkeypatch.setattr(evaluate_smoke, "FAQRetriever", lambda faqs: object())
    monkeypatch.setattr(evaluate_smoke, "SupportCopilot", lambda *args: object())
    monkeypatch.setattr(
        evaluate_smoke,
        "evaluate_cases",
        lambda *args, **kwargs: {
            "count": 0,
            "passed": 0,
            "failed": 0,
            "metrics": {},
            "metric_support": {},
            "cases": [],
        },
    )
    monkeypatch.setattr(evaluate_smoke, "write_json", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_smoke.py",
            "--model",
            "distilbert",
            "--model-id",
            model_id,
            "--cases",
            str(cases),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    evaluate_smoke.main()

    assert loaded["model_id"] == model_id
    assert loaded["prefer_transformer"] is True
