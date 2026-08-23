from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from heliostune.configs import KernelConfig, Workload
from heliostune.multisource import compare_multisource
from heliostune.schema import HardwareProfile, Measurement

_CONFIGS = (
    KernelConfig(16, 32, 32, 4, 2),
    KernelConfig(32, 64, 32, 4, 3),
)
_WORKLOADS = (
    Workload(64, 128, 256, "alpha", "attention", "decode"),
    Workload(96, 192, 256, "alpha", "feedforward", "prefill"),
    # This shape deliberately coincides with alpha/attention. Neither row may
    # enter the other family's held-out source archive.
    Workload(64, 128, 256, "beta", "attention", "decode"),
    Workload(128, 256, 128, "beta", "feedforward", "prefill"),
)
_HARDWARE = (
    HardwareProfile("source-a", "Source A", (8, 0), 80, 24.0),
    HardwareProfile("source-b", "Source B", (8, 6), 60, 16.0),
    HardwareProfile("target", "Target", (9, 0), 100, 48.0),
)
_TARGET_CONFIG_FACTORS = (
    (0.5, 1.5),
    (1.4, 0.6),
    (0.8, 1.2),
    (1.8, 0.4),
)
_EXPECTED_METHODS = {
    "static_multisource",
    "torch",
    "random",
    "single_source_nearest",
    "multisource_retrieval",
    "cold_thompson",
    "pooled_source_thompson",
    "parhelion_thompson",
    "exhaustive",
    "heldout_reference",
}
_CURVE_METHODS = _EXPECTED_METHODS - {"exhaustive", "heldout_reference"}


def _corpus() -> tuple[Measurement, ...]:
    """Return a complete, deterministic GPU/workload/config/three-bank matrix."""

    rows: list[Measurement] = []
    source_factors = {
        "source-a": (0.65, 1.35),
        "source-b": (1.4, 0.6),
    }
    for gpu_index, hardware in enumerate(_HARDWARE):
        for workload_index, workload in enumerate(_WORKLOADS):
            base_latency = 1.0 + 0.2 * workload_index + 0.05 * gpu_index
            factors = (
                _TARGET_CONFIG_FACTORS[workload_index]
                if hardware.gpu == "target"
                else source_factors[hardware.gpu]
            )
            for config_index, config in enumerate(_CONFIGS):
                for bank in range(3):
                    bank_factor = 1.0 + (0.01 * bank if config_index == 0 else -0.01 * bank)
                    rows.append(
                        Measurement(
                            hardware=hardware,
                            workload=workload,
                            config=config,
                            latency_ms=base_latency * factors[config_index] * bank_factor,
                            torch_latency_ms=base_latency * 2.5,
                            correct=True,
                            replicate=bank,
                            max_abs_error=0.0,
                        )
                    )
    assert len(rows) == len(_HARDWARE) * len(_WORKLOADS) * len(_CONFIGS) * 3
    return tuple(rows)


def _development_replay(
    measurements: tuple[Measurement, ...] | None = None,
    *,
    source_gpus: tuple[str, str] = ("source-a", "source-b"),
    seeds: int = 3,
    **parameters: Any,
) -> dict[str, Any]:
    return compare_multisource(
        _corpus() if measurements is None else measurements,
        source_gpus=source_gpus,
        target_gpu="target",
        max_budget=2,
        seeds=seeds,
        protocol_role="development",
        **parameters,
    )


def _target_measurements(
    corpus: tuple[Measurement, ...],
) -> dict[tuple[str, str, int], Measurement]:
    return {
        (row.workload.key, row.config.key, row.replicate): row
        for row in corpus
        if row.hardware.gpu == "target"
    }


def test_development_replay_schema_and_parhelion_paid_anchor_contract() -> None:
    corpus = _corpus()
    result = _development_replay(corpus)

    assert result["protocol_role"] == "development"
    assert result["source_gpus"] == ["source-a", "source-b"]
    assert result["target_gpu"] == "target"
    assert result["measurement_banks"] == 3
    assert result["model_families"] == 2
    assert set(result["methods"]) == _EXPECTED_METHODS
    assert set(result["method_labels"]) == _EXPECTED_METHODS
    assert result["transfer_method"] == "parhelion_thompson"
    assert result["experiment"]["live_methods"] == [
        "random",
        "single_source_nearest",
        "multisource_retrieval",
        "cold_thompson",
        "pooled_source_thompson",
        "parhelion_thompson",
    ]

    for method in _CURVE_METHODS:
        curve = result["methods"][method]
        assert [point["budget"] for point in curve] == [1, 2]
        for point in curve:
            assert set(point) == {
                "budget",
                "mean_fraction_oracle",
                "ci95_low",
                "ci95_high",
            }
            assert point["mean_fraction_oracle"] > 0.0
            assert point["ci95_low"] <= point["mean_fraction_oracle"] <= point["ci95_high"]

    # At budget one, the only measured incumbent is the paid retrieval anchor.
    # Reconstruct its bank-2 score from the anchor keys reported for each fold.
    target_rows = _target_measurements(corpus)
    configs_by_key = {config.key: config for config in _CONFIGS}
    fold_anchor_scores: list[float] = []
    for fold in result["folds"]:
        heldout_workloads = [
            workload for workload in _WORKLOADS if workload.model == fold["heldout_model"]
        ]
        fractions: list[float] = []
        for workload in heldout_workloads:
            reference = min(
                _CONFIGS,
                key=lambda config: target_rows[(workload.key, config.key, 1)].latency_ms,
            )
            anchor = configs_by_key[fold["parhelion_anchor_configs"][workload.key]]
            reference_latency = target_rows[(workload.key, reference.key, 2)].latency_ms
            anchor_latency = target_rows[(workload.key, anchor.key, 2)].latency_ms
            assert reference_latency is not None
            assert anchor_latency is not None
            fractions.append(reference_latency / anchor_latency)
        fold_anchor_scores.append(math.prod(fractions) ** (1.0 / len(fractions)))

    budget_one = result["methods"]["parhelion_thompson"][0]
    assert budget_one["mean_fraction_oracle"] == pytest.approx(
        statistics.fmean(fold_anchor_scores)
    )
    assert budget_one["mean_fraction_oracle"] == pytest.approx(
        result["methods"]["multisource_retrieval"][0]["mean_fraction_oracle"]
    )
    assert "pays for its consensus retrieval anchor as query one" in result["methodology"]
    assert result["experiment"]["recommendation"] == (
        "best measured bank-0 incumbent for every live adaptive method"
    )
    assert result["experiment"]["target_budget_unit"] == (
        "distinct bank-0 configuration probes per held-out workload"
    )


def test_baseline_parameters_are_independent_and_nearest_is_bound_to_first_source() -> None:
    common_baselines = {
        "retrieval_k": 1,
        "retrieval_temperature": 0.9,
        "pooled_transfer_strength": 0.35,
    }
    low_transfer = _development_replay(
        k=1,
        temperature=0.2,
        transfer_strength=0.0,
        **common_baselines,
    )
    high_transfer = _development_replay(
        k=2,
        temperature=1.7,
        transfer_strength=1.0,
        **common_baselines,
    )

    assert low_transfer["hyperparameters"]["multisource_retrieval"] == {
        "k": 1,
        "temperature": 0.9,
    }
    assert low_transfer["hyperparameters"]["pooled_source_thompson"] == {
        "transfer_strength": 0.35
    }
    assert high_transfer["hyperparameters"]["parhelion"] == {
        "k": 2,
        "temperature": 1.7,
        "transfer_strength": 1.0,
    }
    assert (
        low_transfer["methods"]["multisource_retrieval"]
        == high_transfer["methods"]["multisource_retrieval"]
    )
    assert (
        low_transfer["methods"]["pooled_source_thompson"]
        == high_transfer["methods"]["pooled_source_thompson"]
    )

    reversed_sources = _development_replay(source_gpus=("source-b", "source-a"))
    nearest = reversed_sources["hyperparameters"]["single_source_nearest"]
    assert nearest == {
        "source_gpu": "source-b",
        "k": 1,
        "temperature": 1.0,
        "contract": "parameter-free one-nearest-workload retrieval",
    }
    assert all(
        fold["single_source_nearest_gpu"] == "source-b"
        for fold in reversed_sources["folds"]
    )
    assert reversed_sources["provenance"]["single_source_nearest"] == (
        "parameter-free k=1 retrieval bound to first declared source source-b"
    )
    assert (
        reversed_sources["methods"]["single_source_nearest"]
        != low_transfer["methods"]["single_source_nearest"]
    )


def test_source_archive_rejects_exact_target_shapes() -> None:
    corpus = _corpus()
    shape_counts = Counter((row.m, row.n, row.k) for row in _WORKLOADS)
    poisoned = tuple(
        replace(
            row,
            latency_ms=(0.001 if row.config == _CONFIGS[0] else 1_000.0),
        )
        if row.hardware.gpu in {"source-a", "source-b"}
        and row.replicate == 0
        and shape_counts[(row.workload.m, row.workload.n, row.workload.k)] > 1
        else row
        for row in corpus
    )

    original = _development_replay(corpus)
    replayed = _development_replay(poisoned)
    for fold in replayed["folds"]:
        assert fold["archive_excludes_heldout_family"] is True
        assert fold["archive_excludes_exact_target_shapes"] is True
        assert fold["excluded_exact_target_shapes_by_gpu"] == {
            "source-a": 1,
            "source-b": 1,
        }

    # The poisoned cells are exact-shape rows and are excluded in both family
    # folds. Any source-derived curve changing here is observable leakage.
    for method in (
        "static_multisource",
        "single_source_nearest",
        "multisource_retrieval",
        "pooled_source_thompson",
        "parhelion_thompson",
    ):
        assert replayed["methods"][method] == original["methods"][method]


def test_frozen_primary_comparator_reports_paired_student_t_delta_at_12_seeds() -> None:
    result = _development_replay(
        seeds=12,
        primary_comparator="torch",
        k=2,
        temperature=0.7,
        transfer_strength=0.2,
        retrieval_k=1,
        retrieval_temperature=1.1,
        pooled_transfer_strength=0.4,
    )

    assert result["primary_comparator"] == "torch"
    assert result["headline"]["primary_comparator"] == "torch"
    metrics = result["primary_metrics"]
    assert "External and frozen" in metrics["comparator_selection"]
    assert metrics["descriptive_target_strongest_legacy_method"] != "torch"
    paired_auc = metrics["paired_seed_fraction_reference_auc_1_to_8"]
    assert len(paired_auc["parhelion_thompson"]) == 12
    assert len(paired_auc["torch"]) == 12

    delta = metrics["paired_parhelion_vs_primary_auc_delta"]
    assert delta == result["headline"]["paired_auc_delta_vs_primary"]
    assert set(delta) == {
        "comparator",
        "mean_auc_delta",
        "ci95_low",
        "ci95_high",
        "paired_seeds",
        "degrees_of_freedom",
        "superiority_supported",
        "claim",
    }
    assert delta["comparator"] == "torch"
    assert delta["paired_seeds"] == 12
    assert delta["degrees_of_freedom"] == 11

    differences = [
        parhelion - comparator
        for parhelion, comparator in zip(
            paired_auc["parhelion_thompson"], paired_auc["torch"], strict=True
        )
    ]
    expected_mean = statistics.fmean(differences)
    expected_half_width = 2.200985160 * statistics.stdev(differences) / math.sqrt(12)
    assert delta["mean_auc_delta"] == pytest.approx(expected_mean)
    assert delta["ci95_low"] == pytest.approx(expected_mean - expected_half_width)
    assert delta["ci95_high"] == pytest.approx(expected_mean + expected_half_width)


class _DataAccessIsFailure:
    def __iter__(self):
        raise AssertionError("final-role argument guards accessed measurements")


@pytest.mark.parametrize(
    "override",
    [
        {"source_gpus": ("L4", "A10")},
        {"source_gpus": ("A10", "L4", "T4")},
        {"target_gpu": "H200"},
        {"seeds": 12},
        {"max_budget": 7},
        {"k": None},
        {"temperature": None},
        {"transfer_strength": None},
        {"retrieval_k": None},
        {"retrieval_temperature": None},
        {"pooled_transfer_strength": None},
        {"primary_comparator": None},
        {"primary_comparator": ""},
    ],
)
def test_final_role_rejects_nonfrozen_arguments_before_data_access(
    override: dict[str, Any],
) -> None:
    frozen: dict[str, Any] = {
        "source_gpus": ("L4", "A10", "T4"),
        "target_gpu": "H100",
        "max_budget": 8,
        "seeds": 30,
        "k": 3,
        "temperature": 0.7,
        "transfer_strength": 0.08,
        "retrieval_k": 3,
        "retrieval_temperature": 0.7,
        "pooled_transfer_strength": 0.08,
        "primary_comparator": "random",
        "protocol_role": "final",
    }
    with pytest.raises(ValueError):
        compare_multisource(_DataAccessIsFailure(), **(frozen | override))


def test_primary_comparator_rejects_nonprotocol_seed_count_before_data_access() -> None:
    with pytest.raises(ValueError, match="seed count 12 or 30"):
        compare_multisource(
            _DataAccessIsFailure(),
            source_gpus=("source-a", "source-b"),
            target_gpu="target",
            seeds=11,
            primary_comparator="random",
        )
