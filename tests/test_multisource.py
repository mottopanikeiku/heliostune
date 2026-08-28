from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from heliostune.artifacts import read_measurements
from heliostune.configs import KernelConfig, Workload
from heliostune.errors import SchemaError
from heliostune.multisource import compare_multisource
from heliostune.multisource_engine import (
    PreparedReplay,
    _best_observed,
    _incumbent_value,
    assemble_multisource_summary,
    evaluate_anchored_cold,
    evaluate_cold_thompson,
    evaluate_multisource_retrieval,
    evaluate_parhelion,
    evaluate_parhelion_no_forced_anchor,
    evaluate_pooled_source,
    parameter_independent_evaluations,
    prepare_multisource,
    serialize_workload_endpoints,
    validate_release_provenance,
)
from heliostune.replay import compare_methods
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
_RELEASE_PROVENANCE: dict[str, object] = {
    "algorithm_commit": "a" * 40,
    "freeze_commit": "b" * 40,
    "freeze_sha256": "c" * 64,
    "sole_h100_run": "https://modal.com/apps/example/main/ap-release",
    "raw_h100_sha256": "d" * 64,
    "final_archive_sha256": "e" * 64,
    "post_run_manifest_path": "benchmarks/post-run-manifest.json",
}


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
                            bank=bank,
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
        (row.workload.key, row.config.key, row.bank): row
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
    assert budget_one["mean_fraction_oracle"] == pytest.approx(statistics.fmean(fold_anchor_scores))
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


def test_fold_results_serialize_complete_curves_without_changing_aggregates() -> None:
    result = _development_replay()
    fold_results = result["fold_results"]

    assert [fold["heldout_model"] for fold in fold_results] == [
        fold["heldout_model"] for fold in result["folds"]
    ]
    assert len(fold_results) == result["model_families"]

    deterministic_methods = {
        "static_multisource",
        "torch",
        "single_source_nearest",
        "multisource_retrieval",
        "exhaustive",
        "heldout_reference",
    }
    for fold_result, fold in zip(fold_results, result["folds"], strict=True):
        assert set(fold_result) == {
            "heldout_model",
            "target_workloads",
            "visible_bank0_source_observations_by_gpu",
            "excluded_exact_target_shapes_by_gpu",
            "methods",
        }
        for metadata_key in (
            "heldout_model",
            "target_workloads",
            "visible_bank0_source_observations_by_gpu",
            "excluded_exact_target_shapes_by_gpu",
        ):
            assert fold_result[metadata_key] == fold[metadata_key]

        fold_methods = fold_result["methods"]
        assert set(fold_methods) == _EXPECTED_METHODS
        for points in fold_methods.values():
            assert all(
                set(point)
                == {
                    "budget",
                    "mean_fraction_oracle",
                    "ci95_low",
                    "ci95_high",
                }
                for point in points
            )
        for method in _CURVE_METHODS:
            assert [point["budget"] for point in fold_methods[method]] == [1, 2]
        for method in deterministic_methods:
            assert all(
                point["ci95_low"] == point["mean_fraction_oracle"] == point["ci95_high"]
                for point in fold_methods[method]
            )
        assert [point["budget"] for point in fold_methods["exhaustive"]] == [len(_CONFIGS)]
        assert fold_methods["heldout_reference"] == [
            {
                "budget": len(_CONFIGS),
                "mean_fraction_oracle": 1.0,
                "ci95_low": 1.0,
                "ci95_high": 1.0,
            }
        ]

    # Fold serialization is reporting-only: the existing top-level means still
    # equal the paired-seed/equal-fold aggregate used before this field existed.
    for method in _EXPECTED_METHODS:
        for point_index, aggregate_point in enumerate(result["methods"][method]):
            fold_means = [
                fold["methods"][method][point_index]["mean_fraction_oracle"]
                for fold in fold_results
            ]
            assert aggregate_point["mean_fraction_oracle"] == pytest.approx(
                statistics.fmean(fold_means)
            )

    # Deterministic top-level confidence bounds continue to reflect fold
    # variation, while their newly supplied per-fold bounds are zero-width.
    for method in deterministic_methods:
        for point_index, aggregate_point in enumerate(result["methods"][method]):
            fold_means = [
                fold["methods"][method][point_index]["mean_fraction_oracle"]
                for fold in fold_results
            ]
            half_width = 1.96 * statistics.stdev(fold_means) / math.sqrt(len(fold_means))
            mean = statistics.fmean(fold_means)
            assert aggregate_point["ci95_low"] == pytest.approx(max(0.0, mean - half_width))
            assert aggregate_point["ci95_high"] == pytest.approx(mean + half_width)


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
    assert low_transfer["hyperparameters"]["pooled_source_thompson"] == {"transfer_strength": 0.35}
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
        fold["single_source_nearest_gpu"] == "source-b" for fold in reversed_sources["folds"]
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
        and row.bank == 0
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


def test_seed_count_above_the_paired_seed_stride_is_rejected() -> None:
    with pytest.raises(ValueError, match="no greater than 100000"):
        prepare_multisource(
            _corpus(),
            source_gpus=("source-a", "source-b"),
            target_gpu="target",
            max_budget=2,
            seeds=100_001,
        )


def test_legacy_seed_count_above_stride_is_rejected_before_data_access() -> None:
    with pytest.raises(ValueError, match="no greater than 100000"):
        compare_methods(
            _DataAccessIsFailure(),
            source_gpu="source",
            target_gpu="target",
            seeds=100_001,
        )


def test_primary_comparator_rejects_nonprotocol_seed_count_before_data_access() -> None:
    with pytest.raises(ValueError, match="seed count 12 or 30"):
        compare_multisource(
            _DataAccessIsFailure(),
            source_gpus=("source-a", "source-b"),
            target_gpu="target",
            seeds=11,
            primary_comparator="random",
        )


def test_prepared_replay_is_immutable_and_matches_public_facade() -> None:
    prepared = prepare_multisource(
        _corpus(),
        source_gpus=("source-a", "source-b"),
        target_gpu="target",
        max_budget=2,
        seeds=3,
    )

    assert isinstance(prepared, PreparedReplay)
    assert len(prepared.bank0_rewards) == len(_HARDWARE) * len(_WORKLOADS) * len(_CONFIGS)
    assert all(
        fold.exact_shape_exclusions == {"source-a": 1, "source-b": 1} for fold in prepared.folds
    )
    assert all(
        not row.flags.writeable
        for fold in prepared.folds
        for row in fold.joint_feature_rows.values()
    )
    with pytest.raises(FrozenInstanceError):
        prepared.max_budget = 1
    with pytest.raises(TypeError):
        prepared.bank0_rewards[("source-a", "missing", "missing")] = 0.0

    retrieval = evaluate_multisource_retrieval(prepared, k=2, temperature=0.7)
    pooled = evaluate_pooled_source(prepared, transfer_strength=0.2)
    parhelion = evaluate_parhelion(
        prepared,
        k=2,
        temperature=0.7,
        transfer_strength=0.2,
        retrieval=retrieval,
    )
    assembled = assemble_multisource_summary(
        prepared,
        retrieval=retrieval,
        pooled=pooled,
        parhelion=parhelion,
        k=2,
        temperature=0.7,
        transfer_strength=0.2,
        retrieval_k=2,
        retrieval_temperature=0.7,
        pooled_transfer_strength=0.2,
        primary_comparator=None,
    )
    facade = _development_replay(
        k=2,
        temperature=0.7,
        transfer_strength=0.2,
        retrieval_k=2,
        retrieval_temperature=0.7,
        pooled_transfer_strength=0.2,
    )

    assert assembled == facade


def test_fold_precomputation_uses_fold_identity_when_reordered() -> None:
    prepared = prepare_multisource(
        _corpus(),
        source_gpus=("source-a", "source-b"),
        target_gpu="target",
        max_budget=2,
        seeds=3,
    )
    reordered = replace(prepared, folds=prepared.folds[::-1])
    retrieval = evaluate_multisource_retrieval(prepared, k=2, temperature=0.7)
    reordered_retrieval = evaluate_multisource_retrieval(reordered, k=2, temperature=0.7)

    original_evaluations = (
        evaluate_parhelion(
            prepared,
            k=2,
            temperature=0.7,
            transfer_strength=0.2,
            retrieval=retrieval,
        ),
        evaluate_anchored_cold(prepared, retrieval=retrieval),
        evaluate_parhelion_no_forced_anchor(
            prepared,
            k=2,
            temperature=0.7,
            transfer_strength=0.2,
        ),
    )
    reordered_evaluations = (
        evaluate_parhelion(
            reordered,
            k=2,
            temperature=0.7,
            transfer_strength=0.2,
            retrieval=reordered_retrieval,
        ),
        evaluate_anchored_cold(reordered, retrieval=reordered_retrieval),
        evaluate_parhelion_no_forced_anchor(
            reordered,
            k=2,
            temperature=0.7,
            transfer_strength=0.2,
        ),
    )

    for original, replayed in zip(original_evaluations, reordered_evaluations, strict=True):
        original_by_fold = {
            fold.index: tuple(
                seed_curves[position] for seed_curves in original.stochastic_seed_fold_curves
            )
            for position, fold in enumerate(prepared.folds)
        }
        replayed_by_fold = {
            fold.index: tuple(
                seed_curves[position] for seed_curves in replayed.stochastic_seed_fold_curves
            )
            for position, fold in enumerate(reordered.folds)
        }
        assert replayed_by_fold == original_by_fold


def test_release_provenance_accepts_mapping_and_serializes_plain_dict() -> None:
    provenance = MappingProxyType(_RELEASE_PROVENANCE)

    result = _development_replay(release_provenance=provenance)

    assert type(result["release_provenance"]) is dict
    assert result["release_provenance"] == _RELEASE_PROVENANCE
    validated = validate_release_provenance(provenance)
    assert dict(validated) == _RELEASE_PROVENANCE
    assert tuple(validated) == tuple(_RELEASE_PROVENANCE)


@pytest.mark.parametrize(
    ("provenance", "message"),
    [
        (
            _RELEASE_PROVENANCE | {"unexpected": "value"},
            "release_provenance has unknown fields",
        ),
        (
            {key: value for key, value in _RELEASE_PROVENANCE.items() if key != "algorithm_commit"},
            "release_provenance has missing fields",
        ),
        (
            _RELEASE_PROVENANCE | {"algorithm_commit": ""},
            r"release_provenance\['algorithm_commit'\] must be nonblank",
        ),
        (
            _RELEASE_PROVENANCE | {"algorithm_commit": "A" * 40},
            "algorithm_commit.*lowercase hexadecimal commit",
        ),
        (
            _RELEASE_PROVENANCE | {"freeze_commit": "b" * 39},
            "freeze_commit.*40-character",
        ),
        (
            _RELEASE_PROVENANCE | {"freeze_sha256": "c" * 63},
            "freeze_sha256.*64-character",
        ),
        (
            _RELEASE_PROVENANCE | {"raw_h100_sha256": "D" * 64},
            "raw_h100_sha256.*lowercase hexadecimal",
        ),
        (
            _RELEASE_PROVENANCE | {"final_archive_sha256": "g" * 64},
            "final_archive_sha256.*lowercase hexadecimal",
        ),
        (
            _RELEASE_PROVENANCE | {"sole_h100_run": "http://modal.com/apps/example"},
            "sole_h100_run.*HTTPS Modal URL",
        ),
        (
            _RELEASE_PROVENANCE | {"sole_h100_run": "https://modal.com.evil/apps/example"},
            "sole_h100_run.*HTTPS Modal URL",
        ),
        (
            _RELEASE_PROVENANCE | {"post_run_manifest_path": "/benchmarks/manifest.json"},
            "post_run_manifest_path.*normalized non-escaping",
        ),
        (
            _RELEASE_PROVENANCE | {"post_run_manifest_path": "benchmarks/../outside.json"},
            "post_run_manifest_path.*normalized non-escaping",
        ),
        (
            _RELEASE_PROVENANCE | {"post_run_manifest_path": "benchmarks//manifest.json"},
            "post_run_manifest_path.*normalized non-escaping",
        ),
    ],
)
def test_release_provenance_mapping_still_rejects_invalid_values(
    provenance: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SchemaError, match=message):
        _development_replay(release_provenance=MappingProxyType(provenance))


def test_causal_ablation_evaluators_share_schedule_and_paid_anchor() -> None:
    prepared = prepare_multisource(
        _corpus(),
        source_gpus=("source-a", "source-b"),
        target_gpu="target",
        max_budget=2,
        seeds=3,
    )
    retrieval = evaluate_multisource_retrieval(
        prepared,
        k=2,
        temperature=0.7,
        capture_endpoints=True,
    )
    parhelion = evaluate_parhelion(
        prepared,
        k=2,
        temperature=0.7,
        transfer_strength=0.0,
        retrieval=retrieval,
        capture_endpoints=True,
    )
    anchored = evaluate_anchored_cold(
        prepared,
        retrieval=retrieval,
        capture_endpoints=True,
    )
    cold = evaluate_cold_thompson(prepared, capture_endpoints=True)
    no_anchor = evaluate_parhelion_no_forced_anchor(
        prepared,
        k=2,
        temperature=0.7,
        transfer_strength=0.0,
        capture_endpoints=True,
    )
    pooled_zero = evaluate_pooled_source(
        prepared,
        transfer_strength=0.0,
        capture_endpoints=True,
    )

    assert parhelion.fold_metadata == anchored.fold_metadata == retrieval.fold_metadata
    for seed in range(prepared.seeds):
        for fold in range(len(prepared.folds)):
            expected_budget_one = retrieval.deterministic_fold_curves[fold][0]
            assert parhelion.stochastic_seed_fold_curves[seed][fold][0] == pytest.approx(
                expected_budget_one
            )
            assert anchored.stochastic_seed_fold_curves[seed][fold][0] == pytest.approx(
                expected_budget_one
            )
    assert pooled_zero.stochastic_seed_fold_curves == cold.stochastic_seed_fold_curves
    assert len(serialize_workload_endpoints(prepared, parhelion)) == (
        prepared.seeds * len(prepared.all_workloads)
    )
    assert len(serialize_workload_endpoints(prepared, anchored)) == (
        prepared.seeds * len(prepared.all_workloads)
    )
    assert len(serialize_workload_endpoints(prepared, no_anchor)) == (
        prepared.seeds * len(prepared.all_workloads)
    )


def test_published_folds_exclude_target_family_and_shape() -> None:
    repository = Path(__file__).resolve().parents[1]
    rows = read_measurements(repository / "benchmarks/data/parhelion-v2-measurements.jsonl.zst")

    prepared = prepare_multisource(
        rows,
        source_gpus=("L4", "A10", "T4"),
        target_gpu="H100",
        max_budget=1,
        seeds=1,
        protocol_role="development",
    )

    assert len(prepared.folds) == 4
    assert {fold.heldout_model for fold in prepared.folds} == set(prepared.model_families)
    assert len({fold.heldout_model for fold in prepared.folds}) == 4
    for fold in prepared.folds:
        assert {workload.model for workload in fold.target_workloads} == {fold.heldout_model}
        target_shapes = {(w.m, w.n, w.k) for w in fold.target_workloads}
        for gpu in prepared.source_gpus:
            source_workloads = fold.source_workloads[gpu]
            assert all(w.model != fold.heldout_model for w in source_workloads)
            assert not target_shapes & {(w.m, w.n, w.k) for w in source_workloads}
        assert all(o.source_gpu in prepared.source_gpus for o in fold.archive)
        assert all(o.workload.model != fold.heldout_model for o in fold.archive)

    # Two of the four published families share exact (M, N, K) shapes with other
    # families; the exclusion must fire for them and stay inert for the other two.
    assert {fold.heldout_model: dict(fold.exact_shape_exclusions) for fold in prepared.folds} == {
        "granite-3.1-8b": {"L4": 12, "A10": 12, "T4": 12},
        "mistral-7b": {"L4": 12, "A10": 12, "T4": 12},
        "phi-3-mini": {"L4": 0, "A10": 0, "T4": 0},
        "qwen2.5-7b": {"L4": 0, "A10": 0, "T4": 0},
    }


_LATENCY_FACTORS = st.lists(
    st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False),
    min_size=len(_HARDWARE) * len(_WORKLOADS) * len(_CONFIGS),
    max_size=len(_HARDWARE) * len(_WORKLOADS) * len(_CONFIGS),
)


def _factored_corpus(factors: list[float]) -> tuple[Measurement, ...]:
    rows: list[Measurement] = []
    values = iter(factors)
    for gpu_index, hardware in enumerate(_HARDWARE):
        for workload_index, workload in enumerate(_WORKLOADS):
            base_latency = 1.0 + 0.2 * workload_index + 0.05 * gpu_index
            for config_index, config in enumerate(_CONFIGS):
                factor = next(values)
                for bank in range(3):
                    bank_factor = 1.0 + (0.01 * bank if config_index == 0 else -0.01 * bank)
                    rows.append(
                        Measurement(
                            hardware=hardware,
                            workload=workload,
                            config=config,
                            latency_ms=base_latency * factor * bank_factor,
                            torch_latency_ms=base_latency * 2.5,
                            correct=True,
                            bank=bank,
                            max_abs_error=0.0,
                        )
                    )
    return tuple(rows)


@settings(max_examples=20)
@given(factors=_LATENCY_FACTORS)
def test_every_live_method_reaches_the_same_incumbent_once_the_budget_is_exhausted(
    factors: list[float],
) -> None:
    # Curves are scored on the evaluation bank while incumbents are selected on the
    # observation bank, so a curve may dip. What must hold is that a method which has
    # paid for every action recommends the observation-bank optimum, identically for
    # every method, seed, and fold.
    prepared = prepare_multisource(
        _factored_corpus(factors),
        source_gpus=("source-a", "source-b"),
        target_gpu="target",
        max_budget=len(_CONFIGS),
        seeds=2,
    )
    retrieval = evaluate_multisource_retrieval(prepared, k=2, temperature=0.7)
    evaluations = {
        "multisource_retrieval": retrieval,
        "pooled_source_thompson": evaluate_pooled_source(prepared, transfer_strength=0.2),
        "parhelion_thompson": evaluate_parhelion(
            prepared,
            k=2,
            temperature=0.7,
            transfer_strength=0.2,
            retrieval=retrieval,
        ),
        **{
            method: evaluation
            for method, evaluation in parameter_independent_evaluations(prepared).items()
            if method in _CURVE_METHODS
        },
    }
    exhausted = [
        _incumbent_value(
            prepared.table,
            prepared.target_gpu,
            fold.target_workloads,
            {workload.key: list(prepared.configs) for workload in fold.target_workloads},
        )
        for fold in prepared.folds
    ]

    for method, evaluation in evaluations.items():
        if method in {"static_multisource", "torch"}:
            continue
        for fold_curves in [
            evaluation.deterministic_fold_curves,
            *evaluation.stochastic_seed_fold_curves,
        ]:
            for fold_index, curve in enumerate(fold_curves):
                assert curve[-1] == pytest.approx(exhausted[fold_index]), method


@settings(max_examples=20)
@given(factors=_LATENCY_FACTORS, order=st.permutations(range(len(_CONFIGS))))
def test_incumbent_selection_never_worsens_in_the_observation_bank(
    factors: list[float],
    order: list[int],
) -> None:
    prepared = prepare_multisource(
        _factored_corpus(factors),
        source_gpus=("source-a", "source-b"),
        target_gpu="target",
        max_budget=len(_CONFIGS),
        seeds=1,
    )
    workload = prepared.folds[0].target_workloads[0]
    queried: list[KernelConfig] = []
    previous = math.inf

    for index in order:
        queried.append(prepared.configs[index])
        incumbent = _best_observed(prepared.table, prepared.target_gpu, workload, queried)
        latency = prepared.table.get(prepared.target_gpu, workload, incumbent, 0).latency_ms
        assert latency is not None
        assert latency <= previous
        previous = latency
