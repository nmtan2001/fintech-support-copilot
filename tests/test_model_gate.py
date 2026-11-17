import pytest

from scripts.evaluate_release import failures


@pytest.mark.parametrize("metric", [None, float("nan"), float("inf"), 0.79])
def test_invalid_or_regressed_model_metrics_block_release(metric):
    assert failures({"accuracy": metric}, {"accuracy": 0.8}) == ["accuracy"]


def test_missing_metric_blocks_release():
    assert failures({}, {"accuracy": 0.8}) == ["accuracy"]


def test_threshold_boundary_passes():
    assert failures({"accuracy": 0.8}, {"accuracy": 0.8}) == []
