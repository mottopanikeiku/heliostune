"""Leakage-resistant multi-source replay for the Parhelion autotuner."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray

from heliostune.bandit import BayesianLinearBandit
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, KernelConfig, Workload
from heliostune.errors import SchemaError
from heliostune.features import V2_FEATURE_NAMES, v2_joint_features
from heliostune.replay import (
    PAIRED_SEED_STRIDE,
    BenchmarkTable,
    eligible_source_workloads,
    paired_seed,
)
from heliostune.retrieval import (
    RETRIEVAL_FEATURE_NAMES,
    ArchiveObservation,
    RetrievalIndex,
    log_tflops_reward,
)
from heliostune.schema import HardwareProfile, Measurement
from heliostune.uncertainty import student_t_critical_95
from heliostune.validation import exact_fields, nonblank_string

_OBSERVATION_BANK = 0
_REFERENCE_BANK = 1
_EVALUATION_BANK = 2
_NOISE_VARIANCE = 0.05
_PRIOR_PRECISION = 1.0

_METHOD_LABELS = {
    "static_multisource": "Static multi-source best",
    "torch": "torch.matmul",
    "random": "Random search",
    "single_source_nearest": "Single-source nearest-shape reuse",
    "multisource_retrieval": "Multi-source retrieval",
    "cold_thompson": "Cold Thompson sampling",
    "pooled_source_thompson": "Pooled-source Thompson sampling",
    "parhelion_thompson": "Parhelion retrieval-anchored Thompson sampling",
    "exhaustive": "Exhaustive autotuning",
    "heldout_reference": "Held-out exhaustive reference",
}

RELEASE_PROVENANCE_FIELDS = (
    "algorithm_commit",
    "freeze_commit",
    "freeze_sha256",
    "sole_h100_run",
    "raw_h100_sha256",
    "final_archive_sha256",
    "post_run_manifest_path",
)

_LIVE_METHODS = (
    "random",
    "single_source_nearest",
    "multisource_retrieval",
    "cold_thompson",
    "pooled_source_thompson",
    "parhelion_thompson",
)


def _latency(
    table: BenchmarkTable,
    gpu: str,
    workload: Workload,
    config: KernelConfig,
    bank: int,
) -> float:
    value = table.get(gpu, workload, config, bank).latency_ms
    if value is None or not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"validated cell {gpu}/{workload.key}/{config.key}/bank-{bank} "
            "has no finite positive latency"
        )
    return value


def _reward(
    table: BenchmarkTable,
    gpu: str,
    workload: Workload,
    config: KernelConfig,
) -> float:
    return log_tflops_reward(
        workload,
        _latency(table, gpu, workload, config, _OBSERVATION_BANK),
    )


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _evaluate(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    recommendations: dict[str, KernelConfig],
) -> float:
    return _geometric_mean(
        tuple(
            table.heldout_fraction(target_gpu, workload, recommendations[workload.key])
            for workload in workloads
        )
    )


def _best_observed(
    table: BenchmarkTable,
    target_gpu: str,
    workload: Workload,
    queried: Sequence[KernelConfig],
) -> KernelConfig:
    if not queried:
        raise ValueError("an incumbent requires at least one paid target observation")
    return min(
        queried,
        key=lambda config: (
            _latency(table, target_gpu, workload, config, _OBSERVATION_BANK),
            config.key,
        ),
    )


def _incumbent_value(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    queried: dict[str, list[KernelConfig]],
) -> float:
    recommendations = {
        workload.key: _best_observed(table, target_gpu, workload, queried[workload.key])
        for workload in workloads
    }
    return _evaluate(table, target_gpu, workloads, recommendations)


def _static_multisource_best(
    table: BenchmarkTable,
    source_gpus: Sequence[str],
    source_workloads: Mapping[str, tuple[Workload, ...]],
    configs: Sequence[KernelConfig],
) -> KernelConfig:
    contexts = tuple((gpu, workload) for gpu in source_gpus for workload in source_workloads[gpu])
    if not contexts:
        raise ValueError("no leakage-safe source workload remains in this fold")
    per_context_best = {
        (gpu, workload.key): min(
            _latency(table, gpu, workload, config, _OBSERVATION_BANK) for config in configs
        )
        for gpu, workload in contexts
    }
    return min(
        configs,
        key=lambda config: (
            _geometric_mean(
                tuple(
                    _latency(table, gpu, workload, config, _OBSERVATION_BANK)
                    / per_context_best[(gpu, workload.key)]
                    for gpu, workload in contexts
                )
            ),
            config.key,
        ),
    )


def _archive(
    table: BenchmarkTable,
    source_gpus: Sequence[str],
    source_workloads: Mapping[str, tuple[Workload, ...]],
    configs: Sequence[KernelConfig],
) -> tuple[ArchiveObservation, ...]:
    return tuple(
        ArchiveObservation(
            workload=workload,
            config=config,
            source_gpu=gpu,
            latency_ms=_latency(table, gpu, workload, config, _OBSERVATION_BANK),
        )
        for gpu in source_gpus
        for workload in source_workloads[gpu]
        for config in configs
    )


def _pooled_source_model(
    table: BenchmarkTable,
    source_gpus: Sequence[str],
    source_workloads: Mapping[str, tuple[Workload, ...]],
    configs: Sequence[KernelConfig],
) -> BayesianLinearBandit:
    model = BayesianLinearBandit(
        dimension=len(V2_FEATURE_NAMES),
        noise_variance=_NOISE_VARIANCE,
        prior_precision=_PRIOR_PRECISION,
        seed=0,
    )
    for gpu in source_gpus:
        hardware = table.hardware(gpu)
        for workload in source_workloads[gpu]:
            for config in configs:
                model.update(
                    v2_joint_features(workload, config, hardware),
                    _reward(table, gpu, workload, config),
                )
    return model


def _parhelion_features(
    retrieval: RetrievalIndex,
    workload: Workload,
    config: KernelConfig,
    hardware: HardwareProfile,
) -> NDArray[np.float64]:
    return np.concatenate(
        (
            v2_joint_features(workload, config, hardware),
            np.asarray(retrieval.score(workload, config).as_array(), dtype=np.float64),
        )
    )


def _feature_from_cache(
    cache: Mapping[tuple[str, str], NDArray[np.float64]],
    workload: Workload,
    config: KernelConfig,
) -> NDArray[np.float64]:
    return cache[(workload.key, config.key)]


def _parhelion_source_model(
    table: BenchmarkTable,
    source_gpus: Sequence[str],
    source_workloads: Mapping[str, tuple[Workload, ...]],
    configs: Sequence[KernelConfig],
    retrieval: RetrievalIndex,
) -> BayesianLinearBandit:
    model = BayesianLinearBandit(
        dimension=len(V2_FEATURE_NAMES) + len(RETRIEVAL_FEATURE_NAMES),
        noise_variance=_NOISE_VARIANCE,
        prior_precision=_PRIOR_PRECISION,
        seed=0,
    )
    source_families = {workload.model for gpu in source_gpus for workload in source_workloads[gpu]}
    # Retrieval features for an archive row are themselves cross-family. With a
    # single remaining source family there is no leakage-safe source posterior,
    # so Parhelion keeps its ridge prior and still uses target retrieval features.
    if len(source_families) < 2:
        return model
    for gpu in source_gpus:
        hardware = table.hardware(gpu)
        for workload in source_workloads[gpu]:
            for config in configs:
                model.update(
                    _parhelion_features(retrieval, workload, config, hardware),
                    _reward(table, gpu, workload, config),
                )
    return model


def _capture_incumbents(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    queried: Mapping[str, Sequence[KernelConfig]],
    destination: dict[str, KernelConfig] | None,
) -> None:
    if destination is None:
        return
    destination.update(
        {
            workload.key: _best_observed(
                table,
                target_gpu,
                workload,
                queried[workload.key],
            )
            for workload in workloads
        }
    )


def _random_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
    orders: Sequence[Sequence[Workload]],
    seed: int,
    endpoint_out: dict[str, KernelConfig] | None = None,
) -> list[float]:
    rng = np.random.default_rng(seed)
    queried: dict[str, list[KernelConfig]] = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for order in orders:
        for workload in order:
            available = tuple(config for config in configs if config not in queried[workload.key])
            selected = available[int(rng.integers(len(available)))]
            queried[workload.key].append(selected)
        curve.append(_incumbent_value(table, target_gpu, workloads, queried))
    _capture_incumbents(table, target_gpu, workloads, queried, endpoint_out)
    return curve


def _ranked_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    ranks: Mapping[str, Sequence[KernelConfig]],
    max_budget: int,
    endpoint_out: dict[str, KernelConfig] | None = None,
) -> list[float]:
    queried: dict[str, list[KernelConfig]] = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for budget_index in range(max_budget):
        for workload in workloads:
            queried[workload.key].append(ranks[workload.key][budget_index])
        curve.append(_incumbent_value(table, target_gpu, workloads, queried))
    _capture_incumbents(table, target_gpu, workloads, queried, endpoint_out)
    return curve


def _thompson_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
    orders: Sequence[Sequence[Workload]],
    model: BayesianLinearBandit,
    feature_fn: Callable[[Workload, KernelConfig], NDArray[np.float64]],
    endpoint_out: dict[str, KernelConfig] | None = None,
) -> list[float]:
    queried: dict[str, list[KernelConfig]] = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for order in orders:
        for workload in order:
            available = tuple(config for config in configs if config not in queried[workload.key])
            features_for_config = partial(feature_fn, workload)
            selected = model.choose(available, features_for_config)
            queried[workload.key].append(selected)
            model.update(
                feature_fn(workload, selected),
                _reward(table, target_gpu, workload, selected),
            )
        curve.append(_incumbent_value(table, target_gpu, workloads, queried))
    _capture_incumbents(table, target_gpu, workloads, queried, endpoint_out)
    return curve


def _parhelion_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
    orders: Sequence[Sequence[Workload]],
    model: BayesianLinearBandit,
    features: Mapping[tuple[str, str], NDArray[np.float64]],
    anchors: Mapping[str, KernelConfig],
    endpoint_out: dict[str, KernelConfig] | None = None,
) -> list[float]:
    queried: dict[str, list[KernelConfig]] = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for budget_index, order in enumerate(orders):
        for workload in order:
            if budget_index == 0:
                selected = anchors[workload.key]
            else:
                available = tuple(
                    config for config in configs if config not in queried[workload.key]
                )
                features_for_config = partial(
                    _feature_from_cache,
                    features,
                    workload,
                )
                selected = model.choose(available, features_for_config)
            queried[workload.key].append(selected)
            model.update(
                features[(workload.key, selected.key)],
                _reward(table, target_gpu, workload, selected),
            )
        curve.append(_incumbent_value(table, target_gpu, workloads, queried))
    _capture_incumbents(table, target_gpu, workloads, queried, endpoint_out)
    return curve


def _mean_curves(curves: Sequence[Sequence[float]]) -> list[float]:
    if not curves:
        raise ValueError("cannot average an empty set of fold curves")
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("fold curves must be a non-empty rectangular matrix")
    return [float(value) for value in np.mean(values, axis=0)]


def _aggregate(
    runs: Sequence[Sequence[float]], budgets: Sequence[int]
) -> list[dict[str, float | int]]:
    values = np.asarray(runs, dtype=np.float64)
    if values.shape != (len(runs), len(budgets)):
        raise ValueError("run matrix does not match budgets")
    points: list[dict[str, float | int]] = []
    for index, budget in enumerate(budgets):
        column = values[:, index]
        mean = float(np.mean(column))
        half_width = (
            0.0
            if len(column) == 1
            else float(1.96 * np.std(column, ddof=1) / math.sqrt(len(column)))
        )
        points.append(
            {
                "budget": budget,
                "mean_fraction_oracle": mean,
                "ci95_low": max(0.0, mean - half_width),
                "ci95_high": mean + half_width,
            }
        )
    return points


def _constant_curve(value: float, length: int) -> list[float]:
    return [value] * length


def _torch_value(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
) -> float:
    fractions: list[float] = []
    for workload in workloads:
        torch_latency = table.get(
            target_gpu, workload, configs[0], _EVALUATION_BANK
        ).torch_latency_ms
        if torch_latency is None or not math.isfinite(torch_latency) or torch_latency <= 0.0:
            raise ValueError(f"missing held-out torch latency for {target_gpu}/{workload.key}")
        reference = table.reference_config(target_gpu, workload)
        fractions.append(
            _latency(table, target_gpu, workload, reference, _EVALUATION_BANK) / torch_latency
        )
    return _geometric_mean(fractions)


def _validate_frozen_matrix(
    table: BenchmarkTable,
    gpus: Sequence[str],
    *,
    target_gpu: str,
    protocol_role: str,
) -> None:
    expected_workloads = {workload.key for workload in DEFAULT_WORKLOADS}
    expected_configs = {config.key for config in DEFAULT_CONFIGS}
    expected_records = len(expected_workloads) * len(expected_configs) * 3
    for gpu in gpus:
        if {workload.key for workload in table.workloads(gpu)} != expected_workloads:
            raise ValueError(
                f"{protocol_role} replay requires the frozen 96-workload corpus on {gpu}"
            )
        if {config.key for config in table.configs(gpu)} != expected_configs:
            raise ValueError(
                f"{protocol_role} replay requires the frozen 36-config corpus on {gpu}"
            )
        table.validate_matrix(gpu, (_OBSERVATION_BANK, _REFERENCE_BANK, _EVALUATION_BANK))
        gpu_records = sum(measurement.hardware.gpu == gpu for measurement in table.measurements)
        if gpu_records != expected_records:
            raise ValueError(
                f"{protocol_role} replay requires exactly {expected_records} records on {gpu}"
            )

    target = table.hardware(target_gpu)
    target_name = target.device_name.upper()
    if protocol_role == "validation" and (
        "T4" not in target_name or target.compute_capability != (7, 5)
    ):
        raise ValueError("validation target runtime identity is not an NVIDIA T4")
    if protocol_role == "final" and (
        "H100" not in target_name
        or "H200" in target_name
        or target.compute_capability != (9, 0)
        or not 75.0 <= target.total_memory_gb <= 85.0
    ):
        raise ValueError("final target runtime identity is not an 80 GB NVIDIA H100")


@dataclass(frozen=True, slots=True)
class PreparedFold:
    """Parameter-independent state for one leakage-safe model-family fold."""

    index: int
    heldout_model: str
    target_workloads: tuple[Workload, ...]
    source_workloads: Mapping[str, tuple[Workload, ...]]
    exact_shape_exclusions: Mapping[str, int]
    archive: tuple[ArchiveObservation, ...]
    pooled_source: BayesianLinearBandit
    target_hardware: HardwareProfile
    joint_feature_rows: Mapping[tuple[str, str], NDArray[np.float64]]
    paired_orders: tuple[tuple[tuple[Workload, ...], ...], ...]
    static_config: KernelConfig
    static_curve: tuple[float, ...]
    torch_curve: tuple[float, ...]
    nearest_curve: tuple[float, ...]
    random_seed_curves: tuple[tuple[float, ...], ...]
    cold_seed_curves: tuple[tuple[float, ...], ...]
    exhaustive_curve: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PreparedReplay:
    """Validated replay state shared by every parameter evaluator."""

    table: BenchmarkTable
    source_gpus: tuple[str, ...]
    target_gpu: str
    max_budget: int
    seeds: int
    protocol_role: str
    configs: tuple[KernelConfig, ...]
    all_workloads: tuple[Workload, ...]
    model_families: tuple[str, ...]
    budgets: tuple[int, ...]
    folds: tuple[PreparedFold, ...]
    bank0_rewards: Mapping[tuple[str, str, str], float]


@dataclass(frozen=True, slots=True)
class MethodEvaluation:
    """Fold curves produced by one isolated method evaluator."""

    method: str
    deterministic_fold_curves: tuple[tuple[float, ...], ...]
    stochastic_seed_fold_curves: tuple[tuple[tuple[float, ...], ...], ...]
    fold_metadata: tuple[tuple[tuple[str, str], ...], ...]
    endpoint_config_keys: tuple[
        tuple[tuple[tuple[str, str], ...], ...],
        ...,
    ] = ()


def _validate_prepare_inputs(
    source_gpus: Sequence[str],
    target_gpu: str,
    max_budget: int,
    seeds: int,
    protocol_role: str,
) -> tuple[str, ...]:
    if isinstance(source_gpus, (str, bytes)):
        raise TypeError("source_gpus must be a sequence of GPU names")
    sources = tuple(source_gpus)
    if len(sources) < 2:
        raise ValueError("multi-source replay requires at least two source GPUs")
    if any(type(gpu) is not str or not gpu or gpu != gpu.strip() for gpu in sources):
        raise ValueError("source_gpus must contain non-empty names without surrounding whitespace")
    if len(set(sources)) != len(sources):
        raise ValueError("source_gpus must be unique")
    if type(target_gpu) is not str or not target_gpu or target_gpu != target_gpu.strip():
        raise ValueError("target_gpu must be a non-empty name without surrounding whitespace")
    if target_gpu in sources:
        raise ValueError("target_gpu must not also be a source GPU")
    if isinstance(max_budget, bool) or not isinstance(max_budget, int) or max_budget <= 0:
        raise ValueError("max_budget must be a positive integer")
    if (
        isinstance(seeds, bool)
        or not isinstance(seeds, int)
        or seeds <= 0
        or seeds > PAIRED_SEED_STRIDE
    ):
        raise ValueError(f"seeds must be a positive integer no greater than {PAIRED_SEED_STRIDE}")
    if protocol_role not in {"development", "validation", "final"}:
        raise ValueError("protocol_role must be development, validation, or final")
    if protocol_role == "validation" and (
        sources != ("L4", "A10") or target_gpu != "T4" or seeds != 12 or max_budget != 8
    ):
        raise ValueError(
            "validation replay requires L4+A10 sources, T4 target, 12 seeds, and budget 8"
        )
    if protocol_role == "final" and (
        sources != ("L4", "A10", "T4") or target_gpu != "H100" or seeds != 30 or max_budget != 8
    ):
        raise ValueError(
            "final replay requires L4+A10+T4 sources, H100 target, 30 seeds, and budget 8"
        )
    return sources


def prepare_multisource(
    measurements: Iterable[Measurement],
    *,
    source_gpus: Sequence[str],
    target_gpu: str,
    max_budget: int = 8,
    seeds: int = 30,
    protocol_role: str = "development",
) -> PreparedReplay:
    """Validate once and prepare every parameter-independent replay object."""
    sources = _validate_prepare_inputs(
        source_gpus,
        target_gpu,
        max_budget,
        seeds,
        protocol_role,
    )
    table = BenchmarkTable(measurements)
    for source_gpu in sources:
        table.validate_protocol(source_gpu, target_gpu)
    if protocol_role in {"validation", "final"}:
        _validate_frozen_matrix(
            table,
            (*sources, target_gpu),
            target_gpu=target_gpu,
            protocol_role=protocol_role,
        )

    configs = table.configs(target_gpu)
    prepared_budget = min(max_budget, len(configs))
    budgets = tuple(range(1, prepared_budget + 1))
    all_workloads = table.workloads(target_gpu)
    model_families = tuple(sorted({workload.model for workload in all_workloads}))
    if len(model_families) < 2:
        raise ValueError("grouped replay requires at least two model families")

    bank0_rewards = MappingProxyType(
        {
            (
                measurement.hardware.gpu,
                measurement.workload.key,
                measurement.config.key,
            ): log_tflops_reward(measurement.workload, measurement.latency_ms)
            for measurement in table.measurements
            if measurement.bank == _OBSERVATION_BANK and measurement.latency_ms is not None
        }
    )
    folds: list[PreparedFold] = []
    for fold_index, heldout_model in enumerate(model_families):
        target_workloads = tuple(
            workload for workload in all_workloads if workload.model == heldout_model
        )
        source_workloads: dict[str, tuple[Workload, ...]] = {}
        exact_shape_exclusions: dict[str, int] = {}
        for source_gpu in sources:
            eligible, excluded_count = eligible_source_workloads(
                table,
                source_gpu,
                heldout_model,
                target_workloads,
            )
            if not eligible:
                raise ValueError(
                    f"no leakage-safe source workloads remain on {source_gpu!r} "
                    f"when holding out {heldout_model!r}"
                )
            source_workloads[source_gpu] = eligible
            exact_shape_exclusions[source_gpu] = excluded_count

        archive = _archive(table, sources, source_workloads, configs)
        single_source_retrieval = RetrievalIndex(
            _archive(table, (sources[0],), source_workloads, configs),
            k=1,
            temperature=1.0,
        )
        pooled_source = _pooled_source_model(table, sources, source_workloads, configs)
        static_config = _static_multisource_best(table, sources, source_workloads, configs)
        static_value = _evaluate(
            table,
            target_gpu,
            target_workloads,
            {workload.key: static_config for workload in target_workloads},
        )
        static_curve = tuple(_constant_curve(static_value, prepared_budget))
        torch_curve = tuple(
            _constant_curve(
                _torch_value(table, target_gpu, target_workloads, configs),
                prepared_budget,
            )
        )
        nearest_ranks = {
            workload.key: single_source_retrieval.rank(workload, configs)
            for workload in target_workloads
        }
        nearest_curve = tuple(
            _ranked_curve(
                table,
                target_gpu,
                target_workloads,
                nearest_ranks,
                prepared_budget,
            )
        )
        target_hardware = table.hardware(target_gpu)
        joint_feature_rows: dict[tuple[str, str], NDArray[np.float64]] = {}
        for workload in target_workloads:
            for config in configs:
                row = v2_joint_features(workload, config, target_hardware)
                row.setflags(write=False)
                joint_feature_rows[(workload.key, config.key)] = row

        paired_orders: list[tuple[tuple[Workload, ...], ...]] = []
        random_seed_curves: list[tuple[float, ...]] = []
        cold_seed_curves: list[tuple[float, ...]] = []
        for seed in range(seeds):
            policy_seed = paired_seed(fold_index, seed)
            order_rng = np.random.default_rng(policy_seed + 50_000)
            orders = tuple(
                tuple(
                    target_workloads[index]
                    for index in order_rng.permutation(len(target_workloads))
                )
                for _ in budgets
            )
            paired_orders.append(orders)
            random_seed_curves.append(
                tuple(
                    _random_curve(
                        table,
                        target_gpu,
                        target_workloads,
                        configs,
                        orders,
                        policy_seed,
                    )
                )
            )
            cold = BayesianLinearBandit(
                dimension=len(V2_FEATURE_NAMES),
                noise_variance=_NOISE_VARIANCE,
                prior_precision=_PRIOR_PRECISION,
                seed=policy_seed,
            )
            cold_seed_curves.append(
                tuple(
                    _thompson_curve(
                        table,
                        target_gpu,
                        target_workloads,
                        configs,
                        orders,
                        cold,
                        partial(_feature_from_cache, joint_feature_rows),
                    )
                )
            )

        exhaustive_recommendations = {
            workload.key: min(
                configs,
                key=lambda config: (
                    _latency(
                        table,
                        target_gpu,
                        workload,
                        config,
                        _OBSERVATION_BANK,
                    ),
                    config.key,
                ),
            )
            for workload in target_workloads
        }
        exhaustive_curve = (
            _evaluate(
                table,
                target_gpu,
                target_workloads,
                exhaustive_recommendations,
            ),
        )
        folds.append(
            PreparedFold(
                index=fold_index,
                heldout_model=heldout_model,
                target_workloads=target_workloads,
                source_workloads=MappingProxyType(dict(source_workloads)),
                exact_shape_exclusions=MappingProxyType(dict(exact_shape_exclusions)),
                archive=archive,
                pooled_source=pooled_source,
                target_hardware=target_hardware,
                joint_feature_rows=MappingProxyType(joint_feature_rows),
                paired_orders=tuple(paired_orders),
                static_config=static_config,
                static_curve=static_curve,
                torch_curve=torch_curve,
                nearest_curve=nearest_curve,
                random_seed_curves=tuple(random_seed_curves),
                cold_seed_curves=tuple(cold_seed_curves),
                exhaustive_curve=exhaustive_curve,
            )
        )

    return PreparedReplay(
        table=table,
        source_gpus=sources,
        target_gpu=target_gpu,
        max_budget=prepared_budget,
        seeds=seeds,
        protocol_role=protocol_role,
        configs=configs,
        all_workloads=all_workloads,
        model_families=model_families,
        budgets=budgets,
        folds=tuple(folds),
        bank0_rewards=bank0_rewards,
    )


def _validate_retrieval_parameters(k: int, temperature: float) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")


def _validate_transfer_strength(transfer_strength: float, *, name: str) -> None:
    if (
        isinstance(transfer_strength, bool)
        or not isinstance(transfer_strength, (int, float))
        or not math.isfinite(transfer_strength)
        or not 0 <= transfer_strength <= 1
    ):
        raise ValueError(f"{name} must be finite and between zero and one")


def parameter_independent_evaluations(
    prepared: PreparedReplay,
) -> Mapping[str, MethodEvaluation]:
    """Return static and parameter-free evaluations computed during preparation."""
    empty_metadata = tuple(() for _ in prepared.folds)
    return MappingProxyType(
        {
            "static_multisource": MethodEvaluation(
                "static_multisource",
                tuple(fold.static_curve for fold in prepared.folds),
                (),
                empty_metadata,
            ),
            "torch": MethodEvaluation(
                "torch",
                tuple(fold.torch_curve for fold in prepared.folds),
                (),
                empty_metadata,
            ),
            "single_source_nearest": MethodEvaluation(
                "single_source_nearest",
                tuple(fold.nearest_curve for fold in prepared.folds),
                (),
                empty_metadata,
            ),
            "random": MethodEvaluation(
                "random",
                (),
                tuple(
                    tuple(fold.random_seed_curves[seed] for fold in prepared.folds)
                    for seed in range(prepared.seeds)
                ),
                empty_metadata,
            ),
            "cold_thompson": MethodEvaluation(
                "cold_thompson",
                (),
                tuple(
                    tuple(fold.cold_seed_curves[seed] for fold in prepared.folds)
                    for seed in range(prepared.seeds)
                ),
                empty_metadata,
            ),
            "exhaustive": MethodEvaluation(
                "exhaustive",
                tuple(fold.exhaustive_curve for fold in prepared.folds),
                (),
                empty_metadata,
            ),
        }
    )


def _endpoint_pairs(
    fold: PreparedFold,
    endpoints: Mapping[str, KernelConfig],
) -> tuple[tuple[str, str], ...]:
    return tuple((workload.key, endpoints[workload.key].key) for workload in fold.target_workloads)


def _parhelion_feature_rows(
    retrieval: RetrievalIndex,
    fold: PreparedFold,
    configs: Sequence[KernelConfig],
) -> dict[tuple[str, str], NDArray[np.float64]]:
    rows: dict[tuple[str, str], NDArray[np.float64]] = {}
    for workload in fold.target_workloads:
        for config in configs:
            row = _parhelion_features(retrieval, workload, config, fold.target_hardware)
            row.setflags(write=False)
            rows[(workload.key, config.key)] = row
    return rows


def _anchors_from_metadata(
    config_by_key: Mapping[str, KernelConfig],
    fold: PreparedFold,
    anchor_pairs: tuple[tuple[str, str], ...],
) -> dict[str, KernelConfig]:
    anchor_keys = dict(anchor_pairs)
    try:
        return {
            workload.key: config_by_key[anchor_keys[workload.key]]
            for workload in fold.target_workloads
        }
    except KeyError as exc:
        raise ValueError("retrieval anchor metadata is incomplete") from exc


def _seeded_evaluation(
    prepared: PreparedReplay,
    *,
    method: str,
    fold_metadata: tuple[tuple[tuple[str, str], ...], ...],
    curve: Callable[[PreparedFold, int, int, dict[str, KernelConfig] | None], list[float]],
    capture_endpoints: bool,
) -> MethodEvaluation:
    """Run one stochastic method over the frozen fold-outer, seed-inner schedule."""
    per_seed_folds: list[list[tuple[float, ...]]] = [[] for _ in range(prepared.seeds)]
    per_seed_endpoints: list[list[tuple[tuple[str, str], ...]]] = [
        [] for _ in range(prepared.seeds)
    ]
    for fold in prepared.folds:
        for seed in range(prepared.seeds):
            endpoints: dict[str, KernelConfig] | None = {} if capture_endpoints else None
            per_seed_folds[seed].append(
                tuple(curve(fold, seed, paired_seed(fold.index, seed), endpoints))
            )
            if endpoints is not None:
                per_seed_endpoints[seed].append(_endpoint_pairs(fold, endpoints))
    return MethodEvaluation(
        method,
        (),
        tuple(tuple(folds) for folds in per_seed_folds),
        fold_metadata,
        (tuple(tuple(folds) for folds in per_seed_endpoints) if capture_endpoints else ()),
    )


def evaluate_multisource_retrieval(
    prepared: PreparedReplay,
    *,
    k: int,
    temperature: float,
    capture_endpoints: bool = False,
) -> MethodEvaluation:
    """Evaluate only deterministic multi-source retrieval."""
    _validate_retrieval_parameters(k, temperature)
    fold_curves: list[tuple[float, ...]] = []
    metadata: list[tuple[tuple[str, str], ...]] = []
    endpoint_folds: list[tuple[tuple[str, str], ...]] = []
    for fold in prepared.folds:
        retrieval = RetrievalIndex(fold.archive, k=k, temperature=float(temperature))
        ranks = {
            workload.key: retrieval.rank(workload, prepared.configs)
            for workload in fold.target_workloads
        }
        anchors = {workload.key: ranks[workload.key][0] for workload in fold.target_workloads}
        endpoints: dict[str, KernelConfig] | None = {} if capture_endpoints else None
        fold_curves.append(
            tuple(
                _ranked_curve(
                    prepared.table,
                    prepared.target_gpu,
                    fold.target_workloads,
                    ranks,
                    prepared.max_budget,
                    endpoint_out=endpoints,
                )
            )
        )
        if endpoints is not None:
            endpoint_folds.append(_endpoint_pairs(fold, endpoints))
        metadata.append(
            tuple((workload_key, config.key) for workload_key, config in anchors.items())
        )
    return MethodEvaluation(
        "multisource_retrieval",
        tuple(fold_curves),
        (),
        tuple(metadata),
        (tuple(endpoint_folds),) if capture_endpoints else (),
    )


def evaluate_pooled_source(
    prepared: PreparedReplay,
    *,
    transfer_strength: float,
    capture_endpoints: bool = False,
) -> MethodEvaluation:
    """Evaluate only pooled-source Thompson sampling."""
    _validate_transfer_strength(transfer_strength, name="pooled_transfer_strength")

    def curve(
        fold: PreparedFold,
        seed: int,
        policy_seed: int,
        endpoints: dict[str, KernelConfig] | None,
    ) -> list[float]:
        model = fold.pooled_source.transferred(
            transfer_strength=float(transfer_strength),
            seed=policy_seed,
        )
        return _thompson_curve(
            prepared.table,
            prepared.target_gpu,
            fold.target_workloads,
            prepared.configs,
            fold.paired_orders[seed],
            model,
            partial(_feature_from_cache, fold.joint_feature_rows),
            endpoint_out=endpoints,
        )

    return _seeded_evaluation(
        prepared,
        method="pooled_source_thompson",
        fold_metadata=tuple(() for _ in prepared.folds),
        curve=curve,
        capture_endpoints=capture_endpoints,
    )


def evaluate_parhelion(
    prepared: PreparedReplay,
    *,
    k: int,
    temperature: float,
    transfer_strength: float,
    retrieval: MethodEvaluation,
    capture_endpoints: bool = False,
) -> MethodEvaluation:
    """Evaluate only Parhelion using the independently selected paid anchor."""
    _validate_retrieval_parameters(k, temperature)
    _validate_transfer_strength(transfer_strength, name="transfer_strength")
    if retrieval.method != "multisource_retrieval":
        raise ValueError("Parhelion requires a multi-source retrieval anchor evaluation")
    if len(retrieval.fold_metadata) != len(prepared.folds):
        raise ValueError("retrieval anchor folds do not match prepared replay")

    config_by_key = {config.key: config for config in prepared.configs}
    source_models: dict[int, BayesianLinearBandit] = {}
    feature_rows_by_fold: dict[int, dict[tuple[str, str], NDArray[np.float64]]] = {}
    anchors_by_fold: dict[int, dict[str, KernelConfig]] = {}
    for fold, anchor_pairs in zip(prepared.folds, retrieval.fold_metadata, strict=True):
        parhelion_retrieval = RetrievalIndex(
            fold.archive,
            k=k,
            temperature=float(temperature),
        )
        source_models[fold.index] = _parhelion_source_model(
            prepared.table,
            prepared.source_gpus,
            fold.source_workloads,
            prepared.configs,
            parhelion_retrieval,
        )
        feature_rows_by_fold[fold.index] = _parhelion_feature_rows(
            parhelion_retrieval, fold, prepared.configs
        )
        anchors_by_fold[fold.index] = _anchors_from_metadata(config_by_key, fold, anchor_pairs)

    def curve(
        fold: PreparedFold,
        seed: int,
        policy_seed: int,
        endpoints: dict[str, KernelConfig] | None,
    ) -> list[float]:
        model = source_models[fold.index].transferred(
            transfer_strength=float(transfer_strength),
            seed=policy_seed,
        )
        return _parhelion_curve(
            prepared.table,
            prepared.target_gpu,
            fold.target_workloads,
            prepared.configs,
            fold.paired_orders[seed],
            model,
            feature_rows_by_fold[fold.index],
            anchors_by_fold[fold.index],
            endpoint_out=endpoints,
        )

    return _seeded_evaluation(
        prepared,
        method="parhelion_thompson",
        fold_metadata=tuple(retrieval.fold_metadata),
        curve=curve,
        capture_endpoints=capture_endpoints,
    )


def evaluate_random(
    prepared: PreparedReplay,
    *,
    capture_endpoints: bool = False,
) -> MethodEvaluation:
    """Evaluate uniform random search under the frozen paired order schedule."""
    if not capture_endpoints:
        return parameter_independent_evaluations(prepared)["random"]

    def curve(
        fold: PreparedFold,
        seed: int,
        policy_seed: int,
        endpoints: dict[str, KernelConfig] | None,
    ) -> list[float]:
        return _random_curve(
            prepared.table,
            prepared.target_gpu,
            fold.target_workloads,
            prepared.configs,
            fold.paired_orders[seed],
            policy_seed,
            endpoint_out=endpoints,
        )

    return _seeded_evaluation(
        prepared,
        method="random",
        fold_metadata=tuple(() for _ in prepared.folds),
        curve=curve,
        capture_endpoints=True,
    )


def evaluate_cold_thompson(
    prepared: PreparedReplay,
    *,
    capture_endpoints: bool = False,
) -> MethodEvaluation:
    """Evaluate the untouched base-feature ridge policy."""
    if not capture_endpoints:
        return parameter_independent_evaluations(prepared)["cold_thompson"]

    def curve(
        fold: PreparedFold,
        seed: int,
        policy_seed: int,
        endpoints: dict[str, KernelConfig] | None,
    ) -> list[float]:
        model = BayesianLinearBandit(
            dimension=len(V2_FEATURE_NAMES),
            noise_variance=_NOISE_VARIANCE,
            prior_precision=_PRIOR_PRECISION,
            seed=policy_seed,
        )
        return _thompson_curve(
            prepared.table,
            prepared.target_gpu,
            fold.target_workloads,
            prepared.configs,
            fold.paired_orders[seed],
            model,
            partial(_feature_from_cache, fold.joint_feature_rows),
            endpoint_out=endpoints,
        )

    return _seeded_evaluation(
        prepared,
        method="cold_thompson",
        fold_metadata=tuple(() for _ in prepared.folds),
        curve=curve,
        capture_endpoints=True,
    )


def evaluate_anchored_cold(
    prepared: PreparedReplay,
    *,
    retrieval: MethodEvaluation,
    capture_endpoints: bool = False,
) -> MethodEvaluation:
    """Force the paid retrieval anchor, then use untouched base v2 features."""
    if retrieval.method != "multisource_retrieval":
        raise ValueError("anchored cold requires a multi-source retrieval evaluation")
    config_by_key = {config.key: config for config in prepared.configs}
    anchors_by_fold = {
        fold.index: _anchors_from_metadata(config_by_key, fold, anchor_pairs)
        for fold, anchor_pairs in zip(prepared.folds, retrieval.fold_metadata, strict=True)
    }

    def curve(
        fold: PreparedFold,
        seed: int,
        policy_seed: int,
        endpoints: dict[str, KernelConfig] | None,
    ) -> list[float]:
        model = BayesianLinearBandit(
            dimension=len(V2_FEATURE_NAMES),
            noise_variance=_NOISE_VARIANCE,
            prior_precision=_PRIOR_PRECISION,
            seed=policy_seed,
        )
        return _parhelion_curve(
            prepared.table,
            prepared.target_gpu,
            fold.target_workloads,
            prepared.configs,
            fold.paired_orders[seed],
            model,
            fold.joint_feature_rows,
            anchors_by_fold[fold.index],
            endpoint_out=endpoints,
        )

    return _seeded_evaluation(
        prepared,
        method="anchored_cold_thompson",
        fold_metadata=tuple(retrieval.fold_metadata),
        curve=curve,
        capture_endpoints=capture_endpoints,
    )


def evaluate_parhelion_no_forced_anchor(
    prepared: PreparedReplay,
    *,
    k: int,
    temperature: float,
    transfer_strength: float,
    capture_endpoints: bool = False,
) -> MethodEvaluation:
    """Use v2 retrieval features with ordinary Thompson choice from budget one."""
    _validate_retrieval_parameters(k, temperature)
    _validate_transfer_strength(transfer_strength, name="transfer_strength")
    source_models: dict[int, BayesianLinearBandit] = {}
    feature_rows_by_fold: dict[int, dict[tuple[str, str], NDArray[np.float64]]] = {}
    for fold in prepared.folds:
        retrieval = RetrievalIndex(
            fold.archive,
            k=k,
            temperature=float(temperature),
        )
        source_models[fold.index] = _parhelion_source_model(
            prepared.table,
            prepared.source_gpus,
            fold.source_workloads,
            prepared.configs,
            retrieval,
        )
        feature_rows_by_fold[fold.index] = _parhelion_feature_rows(
            retrieval, fold, prepared.configs
        )

    def curve(
        fold: PreparedFold,
        seed: int,
        policy_seed: int,
        endpoints: dict[str, KernelConfig] | None,
    ) -> list[float]:
        model = source_models[fold.index].transferred(
            transfer_strength=float(transfer_strength),
            seed=policy_seed,
        )
        return _thompson_curve(
            prepared.table,
            prepared.target_gpu,
            fold.target_workloads,
            prepared.configs,
            fold.paired_orders[seed],
            model,
            partial(_feature_from_cache, feature_rows_by_fold[fold.index]),
            endpoint_out=endpoints,
        )

    return _seeded_evaluation(
        prepared,
        method="parhelion_no_forced_anchor",
        fold_metadata=tuple(() for _ in prepared.folds),
        curve=curve,
        capture_endpoints=capture_endpoints,
    )


def serialize_workload_endpoints(
    prepared: PreparedReplay,
    evaluation: MethodEvaluation,
) -> list[dict[str, object]]:
    """Serialize every captured budget-max workload recommendation and bank-2 score."""
    if not evaluation.endpoint_config_keys:
        raise ValueError(f"{evaluation.method} did not capture workload endpoints")
    config_by_key = {config.key: config for config in prepared.configs}
    records: list[dict[str, object]] = []
    stochastic = bool(evaluation.stochastic_seed_fold_curves)
    for seed_index, fold_endpoints in enumerate(evaluation.endpoint_config_keys):
        if len(fold_endpoints) != len(prepared.folds):
            raise ValueError(f"{evaluation.method} endpoint fold count does not match replay")
        for fold, endpoint_pairs in zip(
            prepared.folds,
            fold_endpoints,
            strict=True,
        ):
            endpoints = dict(endpoint_pairs)
            for workload in fold.target_workloads:
                try:
                    config = config_by_key[endpoints[workload.key]]
                except KeyError as exc:
                    raise ValueError(
                        f"{evaluation.method} endpoint is missing {workload.key}"
                    ) from exc
                recommendation_latency = _latency(
                    prepared.table,
                    prepared.target_gpu,
                    workload,
                    config,
                    _EVALUATION_BANK,
                )
                reference = prepared.table.reference_config(
                    prepared.target_gpu,
                    workload,
                )
                reference_latency = _latency(
                    prepared.table,
                    prepared.target_gpu,
                    workload,
                    reference,
                    _EVALUATION_BANK,
                )
                records.append(
                    {
                        "method": evaluation.method,
                        "seed": seed_index if stochastic else None,
                        "heldout_model": fold.heldout_model,
                        "workload_key": workload.key,
                        "config_key": config.key,
                        "bank0_paid_latency_ms": _latency(
                            prepared.table,
                            prepared.target_gpu,
                            workload,
                            config,
                            _OBSERVATION_BANK,
                        ),
                        "bank2_evaluation_latency_ms": recommendation_latency,
                        "bank2_reference_latency_ms": reference_latency,
                        "fraction_reference": reference_latency / recommendation_latency,
                        "tflops": workload.flops / recommendation_latency / 1e9,
                    }
                )
    return records


def _runs(evaluation: MethodEvaluation) -> list[list[float]]:
    if evaluation.deterministic_fold_curves:
        return [list(curve) for curve in evaluation.deterministic_fold_curves]
    return [
        _mean_curves(seed_fold_curves)
        for seed_fold_curves in evaluation.stochastic_seed_fold_curves
    ]


def evaluation_auc(prepared: PreparedReplay, evaluation: MethodEvaluation) -> float:
    """Return the protocol's equal-budget AUC for one isolated evaluation."""
    points = _aggregate(_runs(evaluation), prepared.budgets)
    return float(np.mean([point["mean_fraction_oracle"] for point in points]))


def _build_method_aggregates(
    prepared: PreparedReplay,
    evaluations: Mapping[str, MethodEvaluation],
    exhaustive: MethodEvaluation,
) -> tuple[
    set[str],
    dict[str, list[list[float]]],
    dict[str, list[dict[str, float | int]]],
]:
    deterministic = {
        method for method, evaluation in evaluations.items() if evaluation.deterministic_fold_curves
    }
    runs = {method: _runs(evaluation) for method, evaluation in evaluations.items()}
    methods = {
        method: _aggregate(runs[method], prepared.budgets)
        for method in (
            "static_multisource",
            "torch",
            "random",
            "single_source_nearest",
            "multisource_retrieval",
            "cold_thompson",
            "pooled_source_thompson",
            "parhelion_thompson",
        )
    }
    methods["exhaustive"] = _aggregate(
        [list(curve) for curve in exhaustive.deterministic_fold_curves],
        [len(prepared.configs)],
    )
    methods["heldout_reference"] = [
        {
            "budget": len(prepared.configs),
            "mean_fraction_oracle": 1.0,
            "ci95_low": 1.0,
            "ci95_high": 1.0,
        }
    ]
    return deterministic, runs, methods


def _build_paired_seed_auc(
    prepared: PreparedReplay,
    runs: Mapping[str, Sequence[Sequence[float]]],
    deterministic_methods: set[str],
    headline_budget: int,
) -> dict[str, list[float]]:
    paired: dict[str, list[float]] = {}
    for method, method_runs in runs.items():
        if method not in deterministic_methods:
            paired[method] = [
                float(np.mean(seed_curve[:headline_budget])) for seed_curve in method_runs
            ]
        else:
            fold_mean_auc = float(
                np.mean([np.mean(fold_curve[:headline_budget]) for fold_curve in method_runs])
            )
            paired[method] = [fold_mean_auc] * prepared.seeds
    return paired


def _build_paired_primary_effect(
    prepared: PreparedReplay,
    paired_seed_auc: Mapping[str, Sequence[float]],
    primary_comparator: str | None,
) -> dict[str, Any] | None:
    if primary_comparator is None:
        return None
    if primary_comparator not in paired_seed_auc:
        valid_comparators = ", ".join(sorted(paired_seed_auc))
        raise ValueError(
            f"primary_comparator must be one of {valid_comparators}, not {primary_comparator!r}"
        )
    delta_values = np.asarray(
        paired_seed_auc["parhelion_thompson"],
        dtype=np.float64,
    ) - np.asarray(paired_seed_auc[primary_comparator], dtype=np.float64)
    delta_mean = float(np.mean(delta_values))
    delta_half_width = float(
        student_t_critical_95(prepared.seeds - 1)
        * np.std(delta_values, ddof=1)
        / math.sqrt(len(delta_values))
    )
    delta_low = delta_mean - delta_half_width
    return {
        "comparator": primary_comparator,
        "mean_auc_delta": delta_mean,
        "ci95_low": delta_low,
        "ci95_high": delta_mean + delta_half_width,
        "paired_seeds": prepared.seeds,
        "degrees_of_freedom": max(0, prepared.seeds - 1),
        "superiority_supported": delta_low > 0.0,
        "claim": (
            "Parhelion has higher fraction-reference AUC than the frozen comparator."
            if delta_low > 0.0
            else None
        ),
    }


@dataclass(frozen=True, slots=True)
class ReleaseProvenance(Mapping[str, str]):
    """Syntax-validated caller-supplied release provenance.

    The serialized mapping carries exactly the seven published provenance
    fields.  Validating those fields does not independently authenticate the
    archive or manifest that a caller may associate with them.
    """

    algorithm_commit: str
    freeze_commit: str
    freeze_sha256: str
    sole_h100_run: str
    raw_h100_sha256: str
    final_archive_sha256: str
    post_run_manifest_path: str

    def __getitem__(self, key: str) -> str:
        if key not in RELEASE_PROVENANCE_FIELDS:
            raise KeyError(key)
        return cast(str, getattr(self, key))

    def __iter__(self) -> Iterator[str]:
        return iter(RELEASE_PROVENANCE_FIELDS)

    def __len__(self) -> int:
        return len(RELEASE_PROVENANCE_FIELDS)

    def to_dict(self) -> dict[str, str]:
        return {key: self[key] for key in RELEASE_PROVENANCE_FIELDS}


def _lowercase_hex(value: object, *, length: int, context: str, kind: str) -> str:
    result = nonblank_string(value, context=context)
    if len(result) != length or any(character not in "0123456789abcdef" for character in result):
        raise SchemaError(f"{context} must be a {length}-character lowercase hexadecimal {kind}")
    return result


def _modal_https_url(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    try:
        parsed = urlsplit(result)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise SchemaError(f"{context} must be an HTTPS Modal URL") from exc
    if (
        parsed.scheme != "https"
        or hostname is None
        or not (hostname == "modal.com" or hostname.endswith(".modal.com"))
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or not parsed.path.startswith("/")
        or parsed.path == "/"
        or parsed.query
        or parsed.fragment
    ):
        raise SchemaError(f"{context} must be an HTTPS Modal URL")
    return result


def _repository_manifest_path(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    path = PurePosixPath(result)
    if (
        "\\" in result
        or path.is_absolute()
        or path.as_posix() != result
        or len(path.parts) < 2
        or path.parts[0] != "benchmarks"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SchemaError(
            f"{context} must be a normalized non-escaping path under repository benchmarks"
        )
    return result


def validate_release_provenance(
    value: Mapping[str, object],
) -> ReleaseProvenance:
    """Validate the exact caller-supplied release provenance syntax."""
    fields = exact_fields(
        dict(value),
        required=RELEASE_PROVENANCE_FIELDS,
        context="release_provenance",
    )
    return ReleaseProvenance(
        algorithm_commit=_lowercase_hex(
            fields["algorithm_commit"],
            length=40,
            context="release_provenance['algorithm_commit']",
            kind="commit",
        ),
        freeze_commit=_lowercase_hex(
            fields["freeze_commit"],
            length=40,
            context="release_provenance['freeze_commit']",
            kind="commit",
        ),
        freeze_sha256=_lowercase_hex(
            fields["freeze_sha256"],
            length=64,
            context="release_provenance['freeze_sha256']",
            kind="SHA-256 digest",
        ),
        sole_h100_run=_modal_https_url(
            fields["sole_h100_run"],
            context="release_provenance['sole_h100_run']",
        ),
        raw_h100_sha256=_lowercase_hex(
            fields["raw_h100_sha256"],
            length=64,
            context="release_provenance['raw_h100_sha256']",
            kind="SHA-256 digest",
        ),
        final_archive_sha256=_lowercase_hex(
            fields["final_archive_sha256"],
            length=64,
            context="release_provenance['final_archive_sha256']",
            kind="SHA-256 digest",
        ),
        post_run_manifest_path=_repository_manifest_path(
            fields["post_run_manifest_path"],
            context="release_provenance['post_run_manifest_path']",
        ),
    )


def _validated_release_provenance(value: Mapping[str, object]) -> dict[str, str]:
    if isinstance(value, ReleaseProvenance):
        return value.to_dict()
    return validate_release_provenance(value).to_dict()


def assemble_multisource_summary(
    prepared: PreparedReplay,
    *,
    retrieval: MethodEvaluation,
    pooled: MethodEvaluation,
    parhelion: MethodEvaluation,
    k: int,
    temperature: float,
    transfer_strength: float,
    retrieval_k: int,
    retrieval_temperature: float,
    pooled_transfer_strength: float,
    primary_comparator: str | None,
    release_provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Assemble aggregate, fold, cost, and provenance output from evaluations."""
    release = (
        None if release_provenance is None else _validated_release_provenance(release_provenance)
    )
    independent = parameter_independent_evaluations(prepared)
    evaluations: dict[str, MethodEvaluation] = {
        "static_multisource": independent["static_multisource"],
        "torch": independent["torch"],
        "random": independent["random"],
        "single_source_nearest": independent["single_source_nearest"],
        "multisource_retrieval": retrieval,
        "cold_thompson": independent["cold_thompson"],
        "pooled_source_thompson": pooled,
        "parhelion_thompson": parhelion,
    }
    deterministic_methods, runs, methods = _build_method_aggregates(
        prepared,
        evaluations,
        independent["exhaustive"],
    )

    fold_details: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(prepared.folds):
        visible_rows = {
            gpu: len(fold.source_workloads[gpu]) * len(prepared.configs)
            for gpu in prepared.source_gpus
        }
        anchors = dict(retrieval.fold_metadata[fold_index])
        detail = {
            "heldout_model": fold.heldout_model,
            "target_workloads": len(fold.target_workloads),
            "source_workloads_by_gpu": {
                gpu: len(fold.source_workloads[gpu]) for gpu in prepared.source_gpus
            },
            "visible_bank0_source_observations_by_gpu": visible_rows,
            "excluded_exact_target_shapes_by_gpu": dict(fold.exact_shape_exclusions),
            "single_source_nearest_gpu": prepared.source_gpus[0],
            "static_config": fold.static_config.key,
            "parhelion_anchor_configs": anchors,
            "archive_excludes_heldout_family": True,
            "archive_excludes_exact_target_shapes": True,
            "target_probes_per_live_method_per_workload": prepared.max_budget,
        }
        fold_details.append(detail)
        fold_methods: dict[str, list[dict[str, float | int]]] = {}
        for method, evaluation in evaluations.items():
            if method in deterministic_methods:
                fold_methods[method] = _aggregate(
                    [evaluation.deterministic_fold_curves[fold_index]],
                    prepared.budgets,
                )
            else:
                fold_methods[method] = _aggregate(
                    [
                        seed_folds[fold_index]
                        for seed_folds in evaluation.stochastic_seed_fold_curves
                    ],
                    prepared.budgets,
                )
        fold_methods["exhaustive"] = _aggregate(
            [fold.exhaustive_curve],
            [len(prepared.configs)],
        )
        fold_methods["heldout_reference"] = [
            {
                "budget": len(prepared.configs),
                "mean_fraction_oracle": 1.0,
                "ci95_low": 1.0,
                "ci95_high": 1.0,
            }
        ]
        fold_results.append(
            {
                "heldout_model": fold.heldout_model,
                "target_workloads": len(fold.target_workloads),
                "visible_bank0_source_observations_by_gpu": visible_rows,
                "excluded_exact_target_shapes_by_gpu": dict(fold.exact_shape_exclusions),
                "methods": fold_methods,
            }
        )

    auc = {
        method: float(np.mean([point["mean_fraction_oracle"] for point in points]))
        for method, points in methods.items()
    }
    queries_to_95 = {
        method: next(
            (int(point["budget"]) for point in points if point["mean_fraction_oracle"] >= 0.95),
            None,
        )
        for method, points in methods.items()
    }
    headline_budget = min(8, prepared.max_budget)
    headline_index = headline_budget - 1
    paired_seed_auc = _build_paired_seed_auc(
        prepared,
        runs,
        deterministic_methods,
        headline_budget,
    )

    legacy_methods = (
        "static_multisource",
        "torch",
        "random",
        "single_source_nearest",
        "multisource_retrieval",
        "cold_thompson",
        "pooled_source_thompson",
    )
    descriptive_strongest = max(
        legacy_methods,
        key=lambda method: (auc[method], method),
    )
    paired_delta = _build_paired_primary_effect(
        prepared,
        paired_seed_auc,
        primary_comparator,
    )
    measurements_per_gpu = len(prepared.all_workloads) * len(prepared.configs) * 3
    visible_source_by_fold = [
        sum(fold["visible_bank0_source_observations_by_gpu"].values()) for fold in fold_details
    ]

    summary: dict[str, Any] = {
        "project": "HeliosTune v2",
        "data_kind": "measured",
        "source_gpu": " + ".join(prepared.source_gpus),
        "source_gpus": list(prepared.source_gpus),
        "target_gpu": prepared.target_gpu,
        "protocol_role": prepared.protocol_role,
        "primary_comparator": primary_comparator,
        "methodology": (
            "Grouped leave-one-model-family-out multi-source replay. Every source archive, "
            "normalization, and posterior excludes the held-out family and exact target shapes. "
            "Policies see only bank 0; bank 1 selects the reference and bank 2 evaluates "
            "recommendations. Parhelion pays for its consensus retrieval anchor as query one, "
            "then adapts a retrieval-augmented Thompson posterior and returns a measured incumbent."
        ),
        "workloads": len(prepared.all_workloads),
        "configs": len(prepared.configs),
        "model_families": len(prepared.model_families),
        "measurement_banks": 3,
        "max_budget": prepared.max_budget,
        "seeds": prepared.seeds,
        "transfer_strength": transfer_strength,
        "hyperparameters": {
            "parhelion": {
                "k": k,
                "temperature": temperature,
                "transfer_strength": transfer_strength,
            },
            "single_source_nearest": {
                "source_gpu": prepared.source_gpus[0],
                "k": 1,
                "temperature": 1.0,
                "contract": "parameter-free one-nearest-workload retrieval",
            },
            "multisource_retrieval": {
                "k": retrieval_k,
                "temperature": retrieval_temperature,
            },
            "pooled_source_thompson": {
                "transfer_strength": pooled_transfer_strength,
            },
            "selection_provenance": (
                "Caller-supplied frozen choices selected externally; this replay performs no "
                "hyperparameter or baseline selection."
            ),
        },
        "provenance": {
            "protocol_role": "caller-declared",
            "hyperparameters": "externally selected and frozen before this replay",
            "primary_comparator": (
                "caller-supplied and frozen before target evaluation"
                if primary_comparator is not None
                else "not supplied; no primary comparative endpoint is reported"
            ),
            "single_source_nearest": (
                "parameter-free k=1 retrieval bound to first declared source "
                f"{prepared.source_gpus[0]}"
            ),
            "reward": "log-TFLOP/s for every source and target posterior observation",
            "budget_one_anchor": (
                "Parhelion and multi-source retrieval use the same independently frozen "
                "retrieval rank-one action"
            ),
            "gpu_name_special_casing": "none",
        },
        "hardware": [
            *(prepared.table.hardware(gpu).to_dict() for gpu in prepared.source_gpus),
            prepared.table.hardware(prepared.target_gpu).to_dict(),
        ],
        "source_hardware": {
            "gpus": list(prepared.source_gpus),
            "profiles": [prepared.table.hardware(gpu).to_dict() for gpu in prepared.source_gpus],
        },
        "target_hardware": prepared.table.hardware(prepared.target_gpu).to_dict(),
        "methods": methods,
        "method_labels": _METHOD_LABELS,
        "transfer_method": "parhelion_thompson",
        "cold_method": "cold_thompson",
        "headline": {
            "budget": headline_budget,
            "parhelion_fraction_oracle": methods["parhelion_thompson"][headline_index][
                "mean_fraction_oracle"
            ],
            "retrieval_fraction_oracle": methods["multisource_retrieval"][headline_index][
                "mean_fraction_oracle"
            ],
            "cold_fraction_oracle": methods["cold_thompson"][headline_index][
                "mean_fraction_oracle"
            ],
            "parhelion_auc": auc["parhelion_thompson"],
            "descriptive_target_strongest_legacy_method": descriptive_strongest,
            "primary_comparator": primary_comparator,
            "paired_auc_delta_vs_primary": paired_delta,
            "trials_avoided_vs_exhaustive": len(prepared.configs) - headline_budget,
        },
        "auc": auc,
        "queries_to_95_percent_reference": queries_to_95,
        "primary_metrics": {
            "fraction_reference_auc": auc,
            "queries_to_95_percent_reference": queries_to_95,
            "primary_budget": headline_budget,
            "paired_seed_fraction_reference_auc_1_to_8": paired_seed_auc,
            "paired_parhelion_vs_primary_auc_delta": paired_delta,
            "descriptive_target_strongest_legacy_method": descriptive_strongest,
            "comparator_selection": (
                "External and frozen before final-domain evaluation; target-side strongest "
                "legacy is descriptive only and never becomes the primary comparator."
            ),
            "fold_aggregation": (
                "Stochastic curves average model-family folds within each paired seed before "
                "the primary AUC1..8 Student-t interval. Deterministic curve intervals reflect "
                "variation across equal-weight folds."
            ),
        },
        "source_cost": {
            "collected_measurements_per_source_gpu": measurements_per_gpu,
            "collected_measurements_all_source_gpus": (
                measurements_per_gpu * len(prepared.source_gpus)
            ),
            "visible_bank0_source_observations_by_fold": visible_source_by_fold,
            "disclosure": (
                "Policies use only bank-0 source observations from leakage-safe rows. All "
                "physical source acquisition and source reference/evaluation banks are outside "
                "the target online-query budget."
            ),
        },
        "target_collection_cost": {
            "physically_collected_measurements": measurements_per_gpu,
            "simulated_online_queries_per_live_method_per_workload": prepared.max_budget,
            "simulated_online_queries_per_live_method_by_fold": [
                int(fold["target_workloads"]) * prepared.max_budget for fold in fold_details
            ],
            "simulated_online_queries_per_live_method_all_folds": (
                len(prepared.all_workloads) * prepared.max_budget
            ),
            "simulated_online_queries_all_live_methods_all_folds": (
                len(_LIVE_METHODS) * len(prepared.all_workloads) * prepared.max_budget
            ),
            "budget_b_formula": (
                "Per live method: target workloads in one fold times b (24*b for each frozen "
                "four-family corpus fold), and all workloads times b (96*b across four folds)."
            ),
            "exhaustive_bank0_queries_per_workload": len(prepared.configs),
            "adaptation_scope": (
                "Each adaptive method uses one shared posterior and a paired, batched workload "
                "order per budget round; workloads are not tuned as independent posteriors."
            ),
            "disclosure": (
                "Replay simulates online selection from an already collected exhaustive target "
                "matrix. Physical collection includes all three banks and is not reduced by the "
                "reported online query budget."
            ),
        },
        "folds": fold_details,
        "fold_results": fold_results,
        "experiment": {
            "workload_keys": [workload.key for workload in prepared.all_workloads],
            "config_keys": [config.key for config in prepared.configs],
            "target_budget_unit": "distinct bank-0 configuration probes per held-out workload",
            "live_methods": list(_LIVE_METHODS),
            "reward": "log-TFLOP/s from the bank-0 selected-configuration latency",
            "retrieval_action_score": (
                "distance-weighted, per-workload-centered log-TFLOP/s advantage"
            ),
            "retrieval_distance": "frozen normalized Euclidean distance over log2(M, N, K)",
            "parhelion_features": [*V2_FEATURE_NAMES, *RETRIEVAL_FEATURE_NAMES],
            "bank_roles": {
                "0": "policy-visible observations and source archive",
                "1": "exhaustive-reference selection",
                "2": "held-out recommendation evaluation",
            },
            "aggregation": "geometric mean fraction of held-out exhaustive reference",
            "confidence_interval": (
                "Stochastic curves use normal 95% seed intervals after equal-fold averaging; "
                "the paired primary AUC1..8 endpoint uses a two-sided 95% Student-t interval; "
                "deterministic curves use fold variation."
            ),
            "recommendation": "best measured bank-0 incumbent for every live adaptive method",
            "adaptation_scope": (
                "one shared posterior with paired batched workload updates per method and seed"
            ),
            "budget_one_invariant": (
                "Parhelion and multi-source retrieval query and recommend the identical "
                "retrieval anchor at budget 1, charged once to each method."
            ),
        },
        "limitations": [
            "The source archive cost is paid before target tuning and is not included in target probes.",
            "Replay reports simulated online query efficiency from a physically exhaustive target matrix.",
            "The curated 36-action manifest is not the full space of legal Triton configurations.",
            "Steady-state kernel timings omit compilation and end-to-end serving overhead.",
            (
                "Protocol role is caller-declared; development and validation results are not "
                "untouched final-domain evidence."
                if prepared.protocol_role != "final"
                else "Final-domain status is caller-declared; replay cannot independently prove that the target was untouched before this call."
            ),
            "Hyperparameters and the primary comparator are selected externally; this function never selects them or branches on a GPU name.",
            "Three independent timing banks reduce leakage but do not model production interference.",
            "The shape-disjoint L4 nearest baseline is v1-inspired and not an exact reproduction of the published v1 family-only replay.",
        ],
    }
    if release is not None:
        summary["release_provenance"] = release
    return summary


def run_multisource(
    measurements: Iterable[Measurement],
    *,
    source_gpus: Sequence[str],
    target_gpu: str,
    max_budget: int = 8,
    seeds: int = 30,
    k: int | None = None,
    temperature: float | None = None,
    transfer_strength: float | None = None,
    retrieval_k: int | None = None,
    retrieval_temperature: float | None = None,
    pooled_transfer_strength: float | None = None,
    primary_comparator: str | None = None,
    protocol_role: str = "development",
    release_provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Prepare once, evaluate method-local parameters, and assemble one summary."""
    if release_provenance is not None and not isinstance(release_provenance, ReleaseProvenance):
        release_provenance = validate_release_provenance(release_provenance)
    parhelion_parameters_supplied = (
        k is not None and temperature is not None and transfer_strength is not None
    )
    retrieval_parameters_supplied = retrieval_k is not None and retrieval_temperature is not None
    pooled_parameter_supplied = pooled_transfer_strength is not None
    k = 3 if k is None else k
    temperature = 0.7 if temperature is None else temperature
    transfer_strength = 0.08 if transfer_strength is None else transfer_strength
    retrieval_k = k if retrieval_k is None else retrieval_k
    retrieval_temperature = temperature if retrieval_temperature is None else retrieval_temperature
    pooled_transfer_strength = (
        transfer_strength if pooled_transfer_strength is None else pooled_transfer_strength
    )
    _validate_retrieval_parameters(k, temperature)
    _validate_transfer_strength(transfer_strength, name="transfer_strength")
    _validate_retrieval_parameters(retrieval_k, retrieval_temperature)
    _validate_transfer_strength(
        pooled_transfer_strength,
        name="pooled_transfer_strength",
    )
    if protocol_role in {"validation", "final"} and not parhelion_parameters_supplied:
        raise ValueError(
            f"{protocol_role} replay requires explicit frozen Parhelion hyperparameters"
        )
    if protocol_role == "final" and (
        not retrieval_parameters_supplied or not pooled_parameter_supplied
    ):
        raise ValueError("final replay requires explicit frozen baseline hyperparameters")
    if primary_comparator is not None and (
        type(primary_comparator) is not str or not primary_comparator
    ):
        raise ValueError("primary_comparator must be a non-empty method key or None")
    if primary_comparator is not None and seeds not in {12, 30}:
        raise ValueError(
            "primary_comparator endpoints require the frozen protocol seed count 12 or 30"
        )
    if protocol_role == "final" and primary_comparator is None:
        raise ValueError("final replay requires a frozen primary_comparator")

    prepared = prepare_multisource(
        measurements,
        source_gpus=source_gpus,
        target_gpu=target_gpu,
        max_budget=max_budget,
        seeds=seeds,
        protocol_role=protocol_role,
    )
    retrieval_evaluation = evaluate_multisource_retrieval(
        prepared,
        k=retrieval_k,
        temperature=retrieval_temperature,
    )
    pooled_evaluation = evaluate_pooled_source(
        prepared,
        transfer_strength=pooled_transfer_strength,
    )
    parhelion_evaluation = evaluate_parhelion(
        prepared,
        k=k,
        temperature=temperature,
        transfer_strength=transfer_strength,
        retrieval=retrieval_evaluation,
    )
    return assemble_multisource_summary(
        prepared,
        retrieval=retrieval_evaluation,
        pooled=pooled_evaluation,
        parhelion=parhelion_evaluation,
        k=k,
        temperature=temperature,
        transfer_strength=transfer_strength,
        retrieval_k=retrieval_k,
        retrieval_temperature=retrieval_temperature,
        pooled_transfer_strength=pooled_transfer_strength,
        primary_comparator=primary_comparator,
        release_provenance=release_provenance,
    )


__all__ = [
    "MethodEvaluation",
    "PreparedFold",
    "PreparedReplay",
    "ReleaseProvenance",
    "assemble_multisource_summary",
    "evaluate_anchored_cold",
    "evaluate_cold_thompson",
    "evaluate_multisource_retrieval",
    "evaluate_parhelion",
    "evaluate_parhelion_no_forced_anchor",
    "evaluate_pooled_source",
    "evaluate_random",
    "evaluation_auc",
    "parameter_independent_evaluations",
    "prepare_multisource",
    "run_multisource",
    "serialize_workload_endpoints",
    "validate_release_provenance",
]
