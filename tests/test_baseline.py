import joblib
import pandas as pd

from baseline import run


def test_baseline_produces_reusable_model_and_metrics(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"text": f"card payment failed number {i}", "category": "card"}
        for i in range(20)
    ] + [
        {"text": f"transfer pending number {i}", "category": "transfer"}
        for i in range(20)
    ]
    pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(rows[:4] + rows[-4:]).to_csv(data_dir / "test.csv", index=False)

    output = tmp_path / "outputs"
    metrics = run(data_dir, output)
    model = joblib.load(output / "model.joblib")

    assert metrics["counts"] == {"train": 36, "validation": 4, "test": 8}
    assert metrics["test"]["macro_f1"] > 0.9
    assert model.predict(["card payment failed"])[0] == "card"
    assert (output / "metrics.json").is_file()
