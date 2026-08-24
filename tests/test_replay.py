from __future__ import annotations

from dataclasses import replace

import pytest

from heliostune.bandit import BayesianLinearBandit
from heliostune.configs import KernelConfig, Workload
from heliostune.replay import BenchmarkTable, compare_methods, eligible_source_workloads
from heliostune.schema import HardwareProfile, Measurement
from heliostune.synthetic import synthetic_measurements

_HARDWARE = (
    HardwareProfile("source", "Source GPU", (8, 9), 60, 24.0),
    HardwareProfile("target", "Target GPU", (8, 6), 72, 24.0),
)
_CONFIGS = (
    KernelConfig(16, 32, 32, 4, 3),
    KernelConfig(32, 32, 32, 4, 3),
)
_WORKLOADS = (
    Workload(1, 32, 32, "model-a", "shared-a", "decode"),
    Workload(2, 32, 32, "model-a", "unique-a", "decode"),
    Workload(1, 32, 32, "model-b", "shared-b", "decode"),
    Workload(4, 32, 32, "model-b", "unique-b", "decode"),
)


def _small_matrix(*, banks: tuple[int, ...] = (0, 1, 2)) -> tuple[Measurement, ...]:
    rows: list[Measurement] = []
    for hardware in _HARDWARE:
        for workload_index, workload in enumerate(_WORKLOADS):
            for config_index, config in enumerate(_CONFIGS):
                for bank in banks:
                    if hardware.gpu == "source":
                        latency = 1.0 + config_index + workload_index * 0.01
                    else:
                        latency = 2.0 - config_index + workload_index * 0.01
                    rows.append(
                        Measurement(
                            hardware=hardware,
                            workload=workload,
                            config=config,
                            bank=bank,
                            latency_ms=latency * (1.0 + bank * 0.001),
                            torch_latency_ms=3.0 + workload_index + bank * 0.01,
                            correct=True,
                        )
                    )
    return tuple(rows)


def test_protocol_requires_three_disjoint_banks() -> None:
    measurements = synthetic_measurements(banks=(0, 1))
    table = BenchmarkTable(measurements)
    with pytest.raises(ValueError, match="requires exactly banks"):
        table.validate_protocol("sim-source", "sim-target")


def test_protocol_rejects_extra_banks() -> None:
    table = BenchmarkTable(_small_matrix(banks=(0, 1, 2, 3)))
    with pytest.raises(ValueError, match=r"found \(0, 1, 2, 3\)"):
        table.validate_protocol("source", "target")


def test_matrix_gate_rejects_missing_cells_and_inconsistent_torch_timings() -> None:
    rows = _small_matrix()
    with pytest.raises(ValueError, match="missing matrix cell.*bank-2"):
        BenchmarkTable(rows[:-1])

    changed = list(rows)
    row = changed[1]
    changed[1] = replace(row, torch_latency_ms=row.torch_latency_ms + 1.0)
    with pytest.raises(ValueError, match="inconsistent duplicated torch timing"):
        BenchmarkTable(changed)


def test_matrix_gate_rejects_inconsistent_hardware_profiles() -> None:
    rows = list(_small_matrix())
    row = rows[0]
    rows[0] = replace(
        row,
        hardware=replace(row.hardware, device_name="Different source GPU"),
    )
    with pytest.raises(ValueError, match="inconsistent hardware profile"):
        BenchmarkTable(rows)


def test_fold_constructor_excludes_family_and_exact_target_shapes() -> None:
    table = BenchmarkTable(_small_matrix())
    heldout = tuple(workload for workload in _WORKLOADS if workload.model == "model-a")

    eligible, excluded = eligible_source_workloads(table, "source", "model-a", heldout)

    assert tuple(workload.key for workload in eligible) == (_WORKLOADS[3].key,)
    assert excluded == 1


def test_source_shape_poison_cannot_change_replay() -> None:
    corpus = _small_matrix()
    poisoned = tuple(
        replace(
            row,
            latency_ms=(0.001 if row.config == _CONFIGS[0] else 1_000.0),
        )
        if row.hardware.gpu == "source" and row.bank == 0 and row.workload.m == 1
        else row
        for row in corpus
    )

    original = compare_methods(
        corpus,
        source_gpu="source",
        target_gpu="target",
        max_budget=1,
        seeds=1,
    )
    replayed = compare_methods(
        poisoned,
        source_gpu="source",
        target_gpu="target",
        max_budget=1,
        seeds=1,
    )

    assert replayed["methods"] == original["methods"]
    assert all(fold["exact_shape_exclusions"] == 1 for fold in replayed["folds"])
    expected_rows = {"model-a": 2, "model-b": 2}
    assert replayed["source_cost"]["visible_source_observations_per_fold"] == expected_rows


def test_bandit_recommends_only_a_paid_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        BayesianLinearBandit,
        "choose",
        lambda _self, actions, _feature_fn: actions[0],
    )
    summary = compare_methods(
        _small_matrix(),
        source_gpu="source",
        target_gpu="target",
        max_budget=1,
        seeds=1,
    )

    assert summary["methods"]["cold_thompson"][0]["mean_fraction_oracle"] < 0.51
    assert summary["methods"]["transfer_thompson"][0]["mean_fraction_oracle"] < 0.51


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
    assert summary["source_cost"]["visible_source_observations_total"] == sum(
        fold["visible_bank0_source_observations"] for fold in summary["folds"]
    )
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
