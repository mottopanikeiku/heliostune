import pytest

from heliostune.replay import BenchmarkTable, compare_methods
from heliostune.synthetic import synthetic_measurements


def test_protocol_requires_three_disjoint_banks() -> None:
    measurements = synthetic_measurements(replicates=2)
    table = BenchmarkTable(measurements)
    with pytest.raises(ValueError, match="replicate banks"):
        table.validate_protocol("sim-source", "sim-target")


def test_grouped_replay_returns_budget_curves_and_costs() -> None:
    summary = compare_methods(
        synthetic_measurements(),
        source_gpu="sim-source",
        target_gpu="sim-target",
        max_budget=2,
        seeds=2,
        transfer_strength=0.08,
    )

    assert summary["workloads"] == 96
    assert summary["configs"] == 36
    assert len(summary["folds"]) == 4
    assert all(fold["target_workloads"] == 24 for fold in summary["folds"])
    assert summary["experiment"]["bank_roles"]["2"] == "held-out final evaluation"
    assert summary["source_cost"]["visible_source_observations_per_fold"] == 72 * 36
    for method in (
        "static",
        "random",
        "nearest_shape",
        "cold_thompson",
        "transfer_thompson",
    ):
        points = summary["methods"][method]
        assert [point["budget"] for point in points] == [1, 2]
        assert all(point["mean_fraction_oracle"] > 0 for point in points)
    assert summary["methods"]["exhaustive"][0]["budget"] == 36
