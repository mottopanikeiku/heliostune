"""Leakage-resistant multi-source replay for the Parhelion autotuner."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from heliostune.bandit import BayesianLinearBandit
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, KernelConfig, Workload
from heliostune.features import FEATURE_NAMES, joint_features
from heliostune.replay import BenchmarkTable
from heliostune.retrieval import (
    RETRIEVAL_FEATURE_NAMES,
    ArchiveObservation,
    RetrievalIndex,
    log_tflops_reward,
)
from heliostune.schema import HardwareProfile, Measurement

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


def _shape(workload: Workload) -> tuple[int, int, int]:
    return workload.m, workload.n, workload.k


def _fold_source_workloads(
    table: BenchmarkTable,
    source_gpu: str,
    heldout_model: str,
    heldout_workloads: Sequence[Workload],
) -> tuple[tuple[Workload, ...], int]:
    """Exclude the held-out family and every coincident held-out target shape."""

    heldout_shapes = {_shape(workload) for workload in heldout_workloads}
    family_safe = tuple(
        workload
        for workload in table.workloads(source_gpu)
        if workload.model != heldout_model
    )
    eligible = tuple(
        workload for workload in family_safe if _shape(workload) not in heldout_shapes
    )
    return eligible, len(family_safe) - len(eligible)




def _static_multisource_best(
    table: BenchmarkTable,
    source_gpus: Sequence[str],
    source_workloads: dict[str, tuple[Workload, ...]],
    configs: Sequence[KernelConfig],
) -> KernelConfig:
    contexts = tuple(
        (gpu, workload)
        for gpu in source_gpus
        for workload in source_workloads[gpu]
    )
    if not contexts:
        raise ValueError("no leakage-safe source workload remains in this fold")
    per_context_best = {
        (gpu, workload.key): min(
            _latency(table, gpu, workload, config, _OBSERVATION_BANK)
            for config in configs
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
    source_workloads: dict[str, tuple[Workload, ...]],
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
    source_workloads: dict[str, tuple[Workload, ...]],
    configs: Sequence[KernelConfig],
) -> BayesianLinearBandit:
    model = BayesianLinearBandit(
        dimension=len(FEATURE_NAMES),
        noise_variance=_NOISE_VARIANCE,
        prior_precision=_PRIOR_PRECISION,
        seed=0,
    )
    for gpu in source_gpus:
        hardware = table.hardware(gpu)
        for workload in source_workloads[gpu]:
            for config in configs:
                model.update(
                    joint_features(workload, config, hardware),
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
            joint_features(workload, config, hardware),
            np.asarray(retrieval.score(workload, config).as_array(), dtype=np.float64),
        )
    )


def _parhelion_source_model(
    table: BenchmarkTable,
    source_gpus: Sequence[str],
    source_workloads: dict[str, tuple[Workload, ...]],
    configs: Sequence[KernelConfig],
    retrieval: RetrievalIndex,
) -> BayesianLinearBandit:
    model = BayesianLinearBandit(
        dimension=len(FEATURE_NAMES) + len(RETRIEVAL_FEATURE_NAMES),
        noise_variance=_NOISE_VARIANCE,
        prior_precision=_PRIOR_PRECISION,
        seed=0,
    )
    source_families = {
        workload.model
        for gpu in source_gpus
        for workload in source_workloads[gpu]
    }
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


def _random_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
    orders: Sequence[Sequence[Workload]],
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    queried: dict[str, list[KernelConfig]] = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for order in orders:
        for workload in order:
            available = tuple(
                config for config in configs if config not in queried[workload.key]
            )
            selected = available[int(rng.integers(len(available)))]
            queried[workload.key].append(selected)
        curve.append(_incumbent_value(table, target_gpu, workloads, queried))
    return curve


def _ranked_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    ranks: dict[str, Sequence[KernelConfig]],
    max_budget: int,
) -> list[float]:
    queried: dict[str, list[KernelConfig]] = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for budget_index in range(max_budget):
        for workload in workloads:
            queried[workload.key].append(ranks[workload.key][budget_index])
        curve.append(_incumbent_value(table, target_gpu, workloads, queried))
    return curve


def _thompson_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
    orders: Sequence[Sequence[Workload]],
    model: BayesianLinearBandit,
    feature_fn: Callable[[Workload, KernelConfig], NDArray[np.float64]],
) -> list[float]:
    queried: dict[str, list[KernelConfig]] = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for order in orders:
        for workload in order:
            available = tuple(
                config for config in configs if config not in queried[workload.key]
            )
            selected = model.choose(
                available,
                lambda config, workload=workload: feature_fn(workload, config),
            )
            queried[workload.key].append(selected)
            model.update(
                feature_fn(workload, selected),
                _reward(table, target_gpu, workload, selected),
            )
        curve.append(_incumbent_value(table, target_gpu, workloads, queried))
    return curve


def _parhelion_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
    orders: Sequence[Sequence[Workload]],
    model: BayesianLinearBandit,
    features: dict[tuple[str, str], NDArray[np.float64]],
    anchors: dict[str, KernelConfig],
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
                selected = model.choose(
                    available,
                    lambda config, workload=workload: features[(workload.key, config.key)],
                )
            queried[workload.key].append(selected)
            model.update(
                features[(workload.key, selected.key)],
                _reward(table, target_gpu, workload, selected),
            )
        curve.append(_incumbent_value(table, target_gpu, workloads, queried))
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
def _student_t_critical_95(degrees_of_freedom: int) -> float:
    frozen_protocol_values = {
        11: 2.200985160,
        29: 2.045229642,
    }
    return frozen_protocol_values.get(degrees_of_freedom, 1.96)




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
            _latency(table, target_gpu, workload, reference, _EVALUATION_BANK)
            / torch_latency
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
            raise ValueError(f"{protocol_role} replay requires the frozen 96-workload corpus on {gpu}")
        if {config.key for config in table.configs(gpu)} != expected_configs:
            raise ValueError(f"{protocol_role} replay requires the frozen 36-config corpus on {gpu}")
        if table.replicates(gpu) != (0, 1, 2):
            raise ValueError(f"{protocol_role} replay requires exactly banks 0, 1, and 2 on {gpu}")
        gpu_measurements = tuple(
            measurement for measurement in table.measurements if measurement.hardware.gpu == gpu
        )
        if len(gpu_measurements) != expected_records:
            raise ValueError(
                f"{protocol_role} replay requires exactly {expected_records} records on {gpu}"
            )
        for measurement in gpu_measurements:
            if (
                not measurement.usable
                or measurement.latency_ms is None
                or not math.isfinite(measurement.latency_ms)
                or measurement.latency_ms <= 0.0
                or not math.isfinite(measurement.torch_latency_ms)
                or measurement.torch_latency_ms <= 0.0
            ):
                raise ValueError(
                    f"invalid frozen cell {gpu}/{measurement.workload.key}/"
                    f"{measurement.config.key}/bank-{measurement.replicate}"
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


def compare_multisource(
    measurements: Iterable[Measurement],
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
) -> dict[str, Any]:
    """Compare Parhelion and independently tuned baselines in grouped replay.

    Every source-derived object is rebuilt for each held-out model-family fold.
    Only bank 0 is visible to policies, bank 1 selects the reference, and bank 2
    evaluates recommendations. Parhelion pays for its retrieval anchor as probe
    one and reports only the best bank-0 configuration it has actually queried.
    Baseline retrieval and pooled Thompson parameters are independent of Parhelion.
    """
    parhelion_parameters_supplied = (
        k is not None and temperature is not None and transfer_strength is not None
    )
    retrieval_parameters_supplied = retrieval_k is not None and retrieval_temperature is not None
    pooled_parameter_supplied = pooled_transfer_strength is not None
    k = 3 if k is None else k
    temperature = 0.7 if temperature is None else temperature
    transfer_strength = 0.08 if transfer_strength is None else transfer_strength

    if isinstance(source_gpus, (str, bytes)):
        raise TypeError("source_gpus must be a sequence of GPU names")
    sources = tuple(source_gpus)
    if len(sources) < 2:
        raise ValueError("multi-source replay requires at least two source GPUs")
    if any(not isinstance(gpu, str) or not gpu for gpu in sources):
        raise ValueError("source_gpus must contain non-empty strings")
    if len(set(sources)) != len(sources):
        raise ValueError("source_gpus must be unique")
    if target_gpu in sources:
        raise ValueError("target_gpu must not also be a source GPU")
    if isinstance(max_budget, bool) or not isinstance(max_budget, int) or max_budget <= 0:
        raise ValueError("max_budget must be a positive integer")
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds <= 0:
        raise ValueError("seeds must be a positive integer")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(transfer_strength) or not 0.0 <= transfer_strength <= 1.0:
        raise ValueError("transfer_strength must be finite and between zero and one")
    retrieval_k = k if retrieval_k is None else retrieval_k
    retrieval_temperature = (
        temperature if retrieval_temperature is None else retrieval_temperature
    )
    pooled_transfer_strength = (
        transfer_strength
        if pooled_transfer_strength is None
        else pooled_transfer_strength
    )
    if (
        isinstance(retrieval_k, bool)
        or not isinstance(retrieval_k, int)
        or retrieval_k <= 0
    ):
        raise ValueError("retrieval_k must be a positive integer")
    if not math.isfinite(retrieval_temperature) or retrieval_temperature <= 0.0:
        raise ValueError("retrieval_temperature must be finite and positive")
    if (
        not math.isfinite(pooled_transfer_strength)
        or not 0.0 <= pooled_transfer_strength <= 1.0
    ):
        raise ValueError("pooled_transfer_strength must be between zero and one")
    if protocol_role not in {"development", "validation", "final"}:
        raise ValueError("protocol_role must be development, validation, or final")
    if protocol_role in {"validation", "final"} and not parhelion_parameters_supplied:
        raise ValueError(f"{protocol_role} replay requires explicit frozen Parhelion hyperparameters")
    if protocol_role == "validation" and (
        sources != ("L4", "A10")
        or target_gpu != "T4"
        or seeds != 12
        or max_budget != 8
    ):
        raise ValueError(
            "validation replay requires L4+A10 sources, T4 target, 12 seeds, and budget 8"
        )
    if protocol_role == "final":
        if (
            sources != ("L4", "A10", "T4")
            or target_gpu != "H100"
            or seeds != 30
            or max_budget != 8
        ):
            raise ValueError(
                "final replay requires L4+A10+T4 sources, H100 target, 30 seeds, and budget 8"
            )
        if not retrieval_parameters_supplied or not pooled_parameter_supplied:
            raise ValueError("final replay requires explicit frozen baseline hyperparameters")
    if primary_comparator is not None and (
        not isinstance(primary_comparator, str) or not primary_comparator
    ):
        raise ValueError("primary_comparator must be a non-empty method key or None")
    if primary_comparator is not None and seeds not in {12, 30}:
        raise ValueError(
            "primary_comparator endpoints require the frozen protocol seed count 12 or 30"
        )
    if protocol_role == "final" and primary_comparator is None:
        raise ValueError("final replay requires a frozen primary_comparator")

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
    max_budget = min(max_budget, len(configs))
    budgets = list(range(1, max_budget + 1))
    all_workloads = table.workloads(target_gpu)
    model_families = tuple(sorted({workload.model for workload in all_workloads}))
    if len(model_families) < 2:
        raise ValueError("grouped replay requires at least two model families")

    deterministic_fold_curves: dict[str, list[list[float]]] = {
        "static_multisource": [],
        "torch": [],
        "single_source_nearest": [],
        "multisource_retrieval": [],
    }
    stochastic_fold_curves: dict[str, list[list[list[float]]]] = {
        method: [[] for _ in range(seeds)]
        for method in ("random", "cold_thompson", "pooled_source_thompson", "parhelion_thompson")
    }
    exhaustive_folds: list[list[float]] = []
    fold_details: list[dict[str, Any]] = []

    for fold_index, heldout_model in enumerate(model_families):
        target_workloads = tuple(
            workload for workload in all_workloads if workload.model == heldout_model
        )
        source_workloads: dict[str, tuple[Workload, ...]] = {}
        exact_shape_exclusions: dict[str, int] = {}
        for source_gpu in sources:
            eligible, excluded_count = _fold_source_workloads(
                table, source_gpu, heldout_model, target_workloads
            )
            if not eligible:
                raise ValueError(
                    f"no leakage-safe source workloads remain on {source_gpu!r} "
                    f"when holding out {heldout_model!r}"
                )
            source_workloads[source_gpu] = eligible
            exact_shape_exclusions[source_gpu] = excluded_count

        archive = _archive(table, sources, source_workloads, configs)
        single_source_archive = _archive(
            table,
            (sources[0],),
            source_workloads,
            configs,
        )
        single_source_retrieval = RetrievalIndex(
            single_source_archive,
            k=1,
            temperature=1.0,
        )
        baseline_retrieval = RetrievalIndex(
            archive,
            k=retrieval_k,
            temperature=retrieval_temperature,
        )
        parhelion_retrieval = RetrievalIndex(archive, k=k, temperature=temperature)
        pooled_source = _pooled_source_model(table, sources, source_workloads, configs)
        parhelion_source = _parhelion_source_model(
            table, sources, source_workloads, configs, parhelion_retrieval
        )

        static_config = _static_multisource_best(
            table, sources, source_workloads, configs
        )
        static_value = _evaluate(
            table,
            target_gpu,
            target_workloads,
            {workload.key: static_config for workload in target_workloads},
        )
        deterministic_fold_curves["static_multisource"].append(
            _constant_curve(static_value, max_budget)
        )
        deterministic_fold_curves["torch"].append(
            _constant_curve(
                _torch_value(table, target_gpu, target_workloads, configs), max_budget
            )
        )

        nearest_ranks = {
            workload.key: single_source_retrieval.rank(workload, configs)
            for workload in target_workloads
        }
        deterministic_fold_curves["single_source_nearest"].append(
            _ranked_curve(
                table, target_gpu, target_workloads, nearest_ranks, max_budget
            )
        )

        retrieval_ranks = {
            workload.key: baseline_retrieval.rank(workload, configs)
            for workload in target_workloads
        }
        anchors = {
            workload.key: retrieval_ranks[workload.key][0]
            for workload in target_workloads
        }
        deterministic_fold_curves["multisource_retrieval"].append(
            _ranked_curve(
                table, target_gpu, target_workloads, retrieval_ranks, max_budget
            )
        )

        target_hardware = table.hardware(target_gpu)
        joint_cache = {
            (workload.key, config.key): joint_features(workload, config, target_hardware)
            for workload in target_workloads
            for config in configs
        }
        parhelion_cache = {
            (workload.key, config.key): _parhelion_features(
                parhelion_retrieval, workload, config, target_hardware
            )
            for workload in target_workloads
            for config in configs
        }

        for seed in range(seeds):
            paired_seed = fold_index * 100_000 + seed
            order_rng = np.random.default_rng(paired_seed + 50_000)
            orders = tuple(
                tuple(
                    target_workloads[index]
                    for index in order_rng.permutation(len(target_workloads))
                )
                for _ in budgets
            )
            stochastic_fold_curves["random"][seed].append(
                _random_curve(
                    table,
                    target_gpu,
                    target_workloads,
                    configs,
                    orders,
                    paired_seed,
                )
            )

            cold = BayesianLinearBandit(
                dimension=len(FEATURE_NAMES),
                noise_variance=_NOISE_VARIANCE,
                prior_precision=_PRIOR_PRECISION,
                seed=paired_seed,
            )
            stochastic_fold_curves["cold_thompson"][seed].append(
                _thompson_curve(
                    table,
                    target_gpu,
                    target_workloads,
                    configs,
                    orders,
                    cold,
                    lambda workload, config, cache=joint_cache: cache[
                        (workload.key, config.key)
                    ],
                )
            )

            pooled = pooled_source.transferred(
                transfer_strength=pooled_transfer_strength,
                seed=paired_seed,
            )
            stochastic_fold_curves["pooled_source_thompson"][seed].append(
                _thompson_curve(
                    table,
                    target_gpu,
                    target_workloads,
                    configs,
                    orders,
                    pooled,
                    lambda workload, config, cache=joint_cache: cache[
                        (workload.key, config.key)
                    ],
                )
            )

            parhelion = parhelion_source.transferred(
                transfer_strength=transfer_strength,
                seed=paired_seed,
            )
            stochastic_fold_curves["parhelion_thompson"][seed].append(
                _parhelion_curve(
                    table,
                    target_gpu,
                    target_workloads,
                    configs,
                    orders,
                    parhelion,
                    parhelion_cache,
                    anchors,
                )
            )

        exhaustive_recommendations = {
            workload.key: min(
                configs,
                key=lambda config, workload=workload: (
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
        exhaustive_folds.append(
            [
                _evaluate(
                    table,
                    target_gpu,
                    target_workloads,
                    exhaustive_recommendations,
                )
            ]
        )

        visible_rows = {
            gpu: len(source_workloads[gpu]) * len(configs) for gpu in sources
        }
        fold_details.append(
            {
                "heldout_model": heldout_model,
                "target_workloads": len(target_workloads),
                "source_workloads_by_gpu": {
                    gpu: len(source_workloads[gpu]) for gpu in sources
                },
                "visible_bank0_source_observations_by_gpu": visible_rows,
                "excluded_exact_target_shapes_by_gpu": exact_shape_exclusions,
                "single_source_nearest_gpu": sources[0],
                "static_config": static_config.key,
                "parhelion_anchor_configs": {
                    workload.key: anchors[workload.key].key
                    for workload in target_workloads
                },
                "archive_excludes_heldout_family": True,
                "archive_excludes_exact_target_shapes": True,
                "target_probes_per_live_method_per_workload": max_budget,
            }
        )

    runs: dict[str, list[list[float]]] = {
        method: fold_curves
        for method, fold_curves in deterministic_fold_curves.items()
    }
    for method, per_seed_folds in stochastic_fold_curves.items():
        runs[method] = [_mean_curves(fold_curves) for fold_curves in per_seed_folds]

    methods = {
        method: _aggregate(runs[method], budgets)
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
    methods["exhaustive"] = _aggregate(exhaustive_folds, [len(configs)])
    methods["heldout_reference"] = [
        {
            "budget": len(configs),
            "mean_fraction_oracle": 1.0,
            "ci95_low": 1.0,
            "ci95_high": 1.0,
        }
    ]

    auc = {
        method: float(
            np.mean([point["mean_fraction_oracle"] for point in points])
        )
        for method, points in methods.items()
    }
    queries_to_95 = {
        method: next(
            (
                int(point["budget"])
                for point in points
                if point["mean_fraction_oracle"] >= 0.95
            ),
            None,
        )
        for method, points in methods.items()
    }
    headline_budget = min(8, max_budget)
    headline_index = headline_budget - 1
    stochastic_methods = frozenset(stochastic_fold_curves)
    paired_seed_auc: dict[str, list[float]] = {}
    for method, method_runs in runs.items():
        if method in stochastic_methods:
            paired_seed_auc[method] = [
                float(np.mean(seed_curve[:headline_budget]))
                for seed_curve in method_runs
            ]
        else:
            fold_mean_auc = float(
                np.mean(
                    [
                        np.mean(fold_curve[:headline_budget])
                        for fold_curve in method_runs
                    ]
                )
            )
            paired_seed_auc[method] = [fold_mean_auc] * seeds

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
    if primary_comparator is not None and primary_comparator not in paired_seed_auc:
        valid_comparators = ", ".join(sorted(paired_seed_auc))
        raise ValueError(
            f"primary_comparator must be one of {valid_comparators}, not "
            f"{primary_comparator!r}"
        )
    paired_delta: dict[str, Any] | None = None
    if primary_comparator is not None:
        delta_values = np.asarray(
            paired_seed_auc["parhelion_thompson"], dtype=np.float64
        ) - np.asarray(paired_seed_auc[primary_comparator], dtype=np.float64)
        delta_mean = float(np.mean(delta_values))
        delta_half_width = float(
            _student_t_critical_95(seeds - 1)
            * np.std(delta_values, ddof=1)
            / math.sqrt(len(delta_values))
        )
        delta_low = delta_mean - delta_half_width
        paired_delta = {
            "comparator": primary_comparator,
            "mean_auc_delta": delta_mean,
            "ci95_low": delta_low,
            "ci95_high": delta_mean + delta_half_width,
            "paired_seeds": seeds,
            "degrees_of_freedom": max(0, seeds - 1),
            "superiority_supported": delta_low > 0.0,
            "claim": (
                "Parhelion has higher fraction-reference AUC than the frozen comparator."
                if delta_low > 0.0
                else None
            ),
        }
    measurements_per_gpu = len(all_workloads) * len(configs) * 3
    visible_source_by_fold = [
        sum(fold["visible_bank0_source_observations_by_gpu"].values())
        for fold in fold_details
    ]

    return {
        "project": "HeliosTune v2",
        "data_kind": "measured",
        "source_gpu": " + ".join(sources),
        "source_gpus": list(sources),
        "target_gpu": target_gpu,
        "protocol_role": protocol_role,
        "primary_comparator": primary_comparator,
        "methodology": (
            "Grouped leave-one-model-family-out multi-source replay. Every source archive, "
            "normalization, and posterior excludes the held-out family and exact target shapes. "
            "Policies see only bank 0; bank 1 selects the reference and bank 2 evaluates "
            "recommendations. Parhelion pays for its consensus retrieval anchor as query one, "
            "then adapts a retrieval-augmented Thompson posterior and returns a measured incumbent."
        ),
        "workloads": len(all_workloads),
        "configs": len(configs),
        "model_families": len(model_families),
        "measurement_banks": 3,
        "max_budget": max_budget,
        "seeds": seeds,
        "transfer_strength": transfer_strength,
        "hyperparameters": {
            "parhelion": {
                "k": k,
                "temperature": temperature,
                "transfer_strength": transfer_strength,
            },
            "single_source_nearest": {
                "source_gpu": sources[0],
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
                f"parameter-free k=1 retrieval bound to first declared source {sources[0]}"
            ),
            "reward": "log-TFLOP/s for every source and target posterior observation",
            "budget_one_anchor": (
                "Parhelion and multi-source retrieval use the same independently frozen "
                "retrieval rank-one action"
            ),
            "gpu_name_special_casing": "none",
        },
        "hardware": [
            *(table.hardware(gpu).to_dict() for gpu in sources),
            table.hardware(target_gpu).to_dict(),
        ],
        "source_hardware": {
            "gpus": list(sources),
            "profiles": [table.hardware(gpu).to_dict() for gpu in sources],
        },
        "target_hardware": table.hardware(target_gpu).to_dict(),
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
            "trials_avoided_vs_exhaustive": len(configs) - headline_budget,
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
            "collected_measurements_all_source_gpus": measurements_per_gpu * len(sources),
            "visible_bank0_source_observations_by_fold": visible_source_by_fold,
            "disclosure": (
                "Policies use only bank-0 source observations from leakage-safe rows. All "
                "physical source acquisition and source reference/evaluation banks are outside "
                "the target online-query budget."
            ),
        },
        "target_collection_cost": {
            "physically_collected_measurements": measurements_per_gpu,
            "simulated_online_queries_per_live_method_per_workload": max_budget,
            "simulated_online_queries_per_live_method_by_fold": [
                int(fold["target_workloads"]) * max_budget for fold in fold_details
            ],
            "simulated_online_queries_per_live_method_all_folds": (
                len(all_workloads) * max_budget
            ),
            "simulated_online_queries_all_live_methods_all_folds": (
                len(_LIVE_METHODS) * len(all_workloads) * max_budget
            ),
            "budget_b_formula": (
                "Per live method: target workloads in one fold times b (24*b for each frozen "
                "four-family corpus fold), and all workloads times b (96*b across four folds)."
            ),
            "exhaustive_bank0_queries_per_workload": len(configs),
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
        "experiment": {
            "workload_keys": [workload.key for workload in all_workloads],
            "config_keys": [config.key for config in configs],
            "target_budget_unit": "distinct bank-0 configuration probes per held-out workload",
            "live_methods": list(_LIVE_METHODS),
            "reward": "log-TFLOP/s from the bank-0 selected-configuration latency",
            "retrieval_action_score": (
                "distance-weighted, per-workload-centered log-TFLOP/s advantage"
            ),
            "retrieval_distance": (
                "frozen normalized Euclidean distance over log2(M, N, K)"
            ),
            "parhelion_features": [*FEATURE_NAMES, *RETRIEVAL_FEATURE_NAMES],
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
                if protocol_role != "final"
                else "Final-domain status is caller-declared; replay cannot independently prove that the target was untouched before this call."
            ),
            "Hyperparameters and the primary comparator are selected externally; this function never selects them or branches on a GPU name.",
            "Three independent timing banks reduce leakage but do not model production interference.",
            "The shape-disjoint L4 nearest baseline is v1-inspired and not an exact reproduction of the published v1 family-only replay.",
        ],
    }


__all__ = ["compare_multisource"]
