"""Frozen Parhelion v3 preparation, policy simulation, and selection engine."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray

from heliostune.bandit import BayesianLinearBandit
from heliostune.configs import KernelConfig, Workload
from heliostune.features import V3_FEATURE_NAMES, v3_joint_features
from heliostune.protocol import (
    V3_BUDGETS,
    V3_K_GRID,
    V3_NOISE_VARIANCE,
    V3_PRIOR_PRECISION,
    V3_TEMPERATURE_GRID,
    V3_TRANSFER_STRENGTH_GRID,
    require_v3_runtime,
    v3_seed,
)
from heliostune.replay import BenchmarkTable, eligible_source_workloads
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


@dataclass(frozen=True, slots=True)
class V3Parameters:
    retrieval_k: int
    retrieval_temperature: float
    pooled_transfer_strength: float
    parhelion_k: int
    parhelion_temperature: float
    parhelion_transfer_strength: float


@dataclass(frozen=True, slots=True)
class V3Fold:
    index: int
    heldout_model: str
    target_workloads: tuple[Workload, ...]
    source_workloads: Mapping[str, tuple[Workload, ...]]
    archive: tuple[ArchiveObservation, ...]
    target_hardware: HardwareProfile
    base_features: Mapping[tuple[str, str], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class V3Prepared:
    table: BenchmarkTable
    source_gpus: tuple[str, ...]
    target_gpu: str
    configs: tuple[KernelConfig, ...]
    official_config_keys: frozenset[str]
    seeds: tuple[int, ...]
    folds: tuple[V3Fold, ...]
    budgets: tuple[int, ...] = V3_BUDGETS


_ConfigPairs = tuple[tuple[str, str], ...]
_BudgetPairs = tuple[_ConfigPairs, ...]
_FoldPairs = tuple[_BudgetPairs, ...]
_SeedPairs = tuple[_FoldPairs, ...]


@dataclass(frozen=True, slots=True)
class V3Evaluation:
    method: str
    recommendations: _SeedPairs
    probes: _SeedPairs
    deterministic: bool = False


def _latency(
    table: BenchmarkTable,
    gpu: str,
    workload: Workload,
    config: KernelConfig,
    bank: int,
) -> float:
    value = table.get(gpu, workload, config, bank).latency_ms
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid latency at {gpu}/{workload.key}/{config.key}/bank-{bank}")
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
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _archive(
    table: BenchmarkTable,
    source_gpus: Sequence[str],
    source_workloads: Mapping[str, Sequence[Workload]],
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
        for workload in sorted(source_workloads[gpu], key=lambda item: item.key)
        for config in sorted(configs, key=lambda item: item.key)
    )


def prepare_v3(
    protocol: Mapping[str, object],
    measurements: Iterable[Measurement],
    *,
    source_gpus: Sequence[str],
    target_gpu: str,
    retained_config_keys: Sequence[str],
    official_config_keys: Sequence[str],
    seeds: Sequence[int],
) -> V3Prepared:
    """Validate runtime before data iteration, then prepare complete five-bank folds."""
    require_v3_runtime(protocol)
    sources = tuple(source_gpus)
    if len(sources) < 2 or len(set(sources)) != len(sources) or target_gpu in sources:
        raise ValueError("v3 requires unique source GPUs distinct from target")
    retained = tuple(retained_config_keys)
    if not retained or retained != tuple(sorted(retained)) or len(set(retained)) != len(retained):
        raise ValueError("retained v3 config keys must be a nonempty sorted unique sequence")
    seed_values = tuple(seeds)
    if not seed_values or any(type(seed) is not int or seed < 0 for seed in seed_values):
        raise ValueError("v3 policy seeds must be non-negative integers")
    table = BenchmarkTable(measurements)
    all_gpus = (*sources, target_gpu)
    for gpu in all_gpus:
        table.validate_matrix(gpu, (0, 1, 2, 3, 4))
    target_workload_keys = {workload.key for workload in table.workloads(target_gpu)}
    target_config_keys = tuple(config.key for config in table.configs(target_gpu))
    if target_config_keys != retained:
        raise ValueError("target config order does not match retained v3 manifest")
    for gpu in sources:
        if {workload.key for workload in table.workloads(gpu)} != target_workload_keys:
            raise ValueError(f"v3 workload set differs on {gpu}")
        if tuple(config.key for config in table.configs(gpu)) != retained:
            raise ValueError(f"v3 config order differs on {gpu}")
    configs = table.configs(target_gpu)
    all_workloads = table.workloads(target_gpu)
    folds: list[V3Fold] = []
    for fold_index, heldout_model in enumerate(
        sorted({workload.model for workload in all_workloads})
    ):
        targets = tuple(workload for workload in all_workloads if workload.model == heldout_model)
        source_workloads: dict[str, tuple[Workload, ...]] = {}
        for gpu in sources:
            eligible, _excluded = eligible_source_workloads(
                table,
                gpu,
                heldout_model,
                targets,
            )
            if not eligible:
                raise ValueError(f"no leakage-safe v3 rows remain for {gpu}/{heldout_model}")
            source_workloads[gpu] = tuple(sorted(eligible, key=lambda item: item.key))
        target_hardware = table.hardware(target_gpu)
        base_features: dict[tuple[str, str], NDArray[np.float64]] = {}
        for workload in targets:
            for config in configs:
                row = v3_joint_features(workload, config, target_hardware)
                row.setflags(write=False)
                base_features[(workload.key, config.key)] = row
        folds.append(
            V3Fold(
                index=fold_index,
                heldout_model=heldout_model,
                target_workloads=targets,
                source_workloads=MappingProxyType(source_workloads),
                archive=_archive(table, sources, source_workloads, configs),
                target_hardware=target_hardware,
                base_features=MappingProxyType(base_features),
            )
        )
    return V3Prepared(
        table=table,
        source_gpus=sources,
        target_gpu=target_gpu,
        configs=configs,
        official_config_keys=frozenset(official_config_keys),
        seeds=seed_values,
        folds=tuple(folds),
    )


def _orders(
    prepared: V3Prepared, fold: V3Fold, policy_seed: int
) -> tuple[tuple[Workload, ...], ...]:
    orders: list[tuple[Workload, ...]] = []
    for round_index in range(len(prepared.budgets)):
        seed = v3_seed(
            purpose="replay-workload-order",
            gpu=prepared.target_gpu,
            heldout_model=fold.heldout_model,
            policy_seed=policy_seed,
            round_index=round_index,
        )
        rng = np.random.default_rng(seed)
        orders.append(
            tuple(
                fold.target_workloads[index]
                for index in rng.permutation(len(fold.target_workloads))
            )
        )
    return tuple(orders)


def _best(
    prepared: V3Prepared,
    fold: V3Fold,
    workload: Workload,
    queried: Sequence[KernelConfig],
) -> KernelConfig:
    if not queried:
        raise ValueError("a v3 incumbent requires a paid observation")
    return min(
        queried,
        key=lambda config: (
            _latency(
                prepared.table,
                prepared.target_gpu,
                workload,
                config,
                _OBSERVATION_BANK,
            ),
            config.key,
        ),
    )


def _pairs(fold: V3Fold, configs: Mapping[str, KernelConfig]) -> _ConfigPairs:
    return tuple((workload.key, configs[workload.key].key) for workload in fold.target_workloads)


def _ranked_evaluation(
    prepared: V3Prepared,
    method: str,
    ranks_by_fold: Sequence[Mapping[str, Sequence[KernelConfig]]],
) -> V3Evaluation:
    fold_recommendations: list[_BudgetPairs] = []
    fold_probes: list[_BudgetPairs] = []
    for fold, ranks in zip(prepared.folds, ranks_by_fold, strict=True):
        queried: dict[str, list[KernelConfig]] = {
            workload.key: [] for workload in fold.target_workloads
        }
        recommendations: list[_ConfigPairs] = []
        probes: list[_ConfigPairs] = []
        for budget_index in range(len(prepared.budgets)):
            round_probes: dict[str, KernelConfig] = {}
            for workload in fold.target_workloads:
                selected = ranks[workload.key][budget_index]
                queried[workload.key].append(selected)
                round_probes[workload.key] = selected
            incumbents = {
                workload.key: _best(prepared, fold, workload, queried[workload.key])
                for workload in fold.target_workloads
            }
            probes.append(_pairs(fold, round_probes))
            recommendations.append(_pairs(fold, incumbents))
        fold_recommendations.append(tuple(recommendations))
        fold_probes.append(tuple(probes))
    return V3Evaluation(
        method,
        (tuple(fold_recommendations),),
        (tuple(fold_probes),),
        deterministic=True,
    )


def evaluate_v3_retrieval(
    prepared: V3Prepared,
    *,
    k: int,
    temperature: float,
) -> V3Evaluation:
    ranks = []
    for fold in prepared.folds:
        retrieval = RetrievalIndex(fold.archive, k=k, temperature=temperature)
        ranks.append(
            {
                workload.key: retrieval.rank(workload, prepared.configs)
                for workload in fold.target_workloads
            }
        )
    return _ranked_evaluation(prepared, "multisource_retrieval", ranks)


def evaluate_v3_nearest(prepared: V3Prepared) -> V3Evaluation:
    ranks = []
    for fold in prepared.folds:
        l4_archive = tuple(
            observation for observation in fold.archive if observation.source_gpu == "L4"
        )
        retrieval = RetrievalIndex(l4_archive, k=1, temperature=1.0)
        ranks.append(
            {
                workload.key: retrieval.rank(workload, prepared.configs)
                for workload in fold.target_workloads
            }
        )
    return _ranked_evaluation(prepared, "single_source_nearest", ranks)


def _source_model(
    prepared: V3Prepared,
    fold: V3Fold,
    *,
    retrieval: RetrievalIndex | None = None,
) -> BayesianLinearBandit:
    dimension = len(V3_FEATURE_NAMES) + (
        len(RETRIEVAL_FEATURE_NAMES) if retrieval is not None else 0
    )
    model = BayesianLinearBandit(
        dimension=dimension,
        noise_variance=V3_NOISE_VARIANCE,
        prior_precision=V3_PRIOR_PRECISION,
        seed=0,
    )
    if retrieval is not None:
        source_families = {
            workload.model
            for gpu in prepared.source_gpus
            for workload in fold.source_workloads[gpu]
        }
        if len(source_families) < 2:
            return model
    for gpu in prepared.source_gpus:
        hardware = prepared.table.hardware(gpu)
        for workload in fold.source_workloads[gpu]:
            for config in prepared.configs:
                features = v3_joint_features(workload, config, hardware)
                if retrieval is not None:
                    features = np.concatenate(
                        (
                            features,
                            np.asarray(
                                retrieval.score(workload, config).as_array(),
                                dtype=np.float64,
                            ),
                        )
                    )
                model.update(
                    features,
                    _reward(prepared.table, gpu, workload, config),
                )
    return model


def _retrieval_features(
    retrieval: RetrievalIndex,
    fold: V3Fold,
    configs: Sequence[KernelConfig],
) -> Mapping[tuple[str, str], NDArray[np.float64]]:
    rows: dict[tuple[str, str], NDArray[np.float64]] = {}
    for workload in fold.target_workloads:
        for config in configs:
            row = np.concatenate(
                (
                    v3_joint_features(workload, config, fold.target_hardware),
                    np.asarray(retrieval.score(workload, config).as_array(), dtype=np.float64),
                )
            )
            row.setflags(write=False)
            rows[(workload.key, config.key)] = row
    return MappingProxyType(rows)


def _feature_from_cache(
    cache: Mapping[tuple[str, str], NDArray[np.float64]],
    workload: Workload,
    config: KernelConfig,
) -> NDArray[np.float64]:
    return cache[(workload.key, config.key)]


def _adaptive_evaluation(
    prepared: V3Prepared,
    *,
    method: str,
    purpose: str,
    feature_rows: Sequence[Mapping[tuple[str, str], NDArray[np.float64]]],
    source_models: Sequence[BayesianLinearBandit | None],
    transfer_strength: float,
    anchors: Sequence[Mapping[str, KernelConfig]] | None = None,
) -> V3Evaluation:
    seed_recommendations: list[_FoldPairs] = []
    seed_probes: list[_FoldPairs] = []
    for policy_seed in prepared.seeds:
        fold_recommendations: list[_BudgetPairs] = []
        fold_probes: list[_BudgetPairs] = []
        for fold, cache, source_model in zip(
            prepared.folds,
            feature_rows,
            source_models,
            strict=True,
        ):
            model_seed = v3_seed(
                purpose=purpose,
                gpu=prepared.target_gpu,
                heldout_model=fold.heldout_model,
                policy_seed=policy_seed,
            )
            dimension = len(next(iter(cache.values())))
            model = (
                BayesianLinearBandit(
                    dimension=dimension,
                    noise_variance=V3_NOISE_VARIANCE,
                    prior_precision=V3_PRIOR_PRECISION,
                    seed=model_seed,
                )
                if source_model is None
                else source_model.transferred(
                    transfer_strength=transfer_strength,
                    seed=model_seed,
                )
            )
            queried: dict[str, list[KernelConfig]] = {
                workload.key: [] for workload in fold.target_workloads
            }
            recommendations: list[_ConfigPairs] = []
            probes: list[_ConfigPairs] = []
            for budget_index, order in enumerate(_orders(prepared, fold, policy_seed)):
                round_probes: dict[str, KernelConfig] = {}
                for workload in order:
                    if budget_index == 0 and anchors is not None:
                        selected = anchors[fold.index][workload.key]
                    else:
                        available = tuple(
                            config
                            for config in prepared.configs
                            if config not in queried[workload.key]
                        )
                        selected = model.choose(
                            available,
                            partial(_feature_from_cache, cache, workload),
                        )
                    queried[workload.key].append(selected)
                    round_probes[workload.key] = selected
                    model.update(
                        cache[(workload.key, selected.key)],
                        _reward(
                            prepared.table,
                            prepared.target_gpu,
                            workload,
                            selected,
                        ),
                    )
                incumbents = {
                    workload.key: _best(
                        prepared,
                        fold,
                        workload,
                        queried[workload.key],
                    )
                    for workload in fold.target_workloads
                }
                probes.append(_pairs(fold, round_probes))
                recommendations.append(_pairs(fold, incumbents))
            fold_probes.append(tuple(probes))
            fold_recommendations.append(tuple(recommendations))
        seed_probes.append(tuple(fold_probes))
        seed_recommendations.append(tuple(fold_recommendations))
    return V3Evaluation(method, tuple(seed_recommendations), tuple(seed_probes))


def evaluate_v3_cold(prepared: V3Prepared) -> V3Evaluation:
    return _adaptive_evaluation(
        prepared,
        method="cold_thompson",
        purpose="cold-thompson",
        feature_rows=tuple(fold.base_features for fold in prepared.folds),
        source_models=(None,) * len(prepared.folds),
        transfer_strength=0.0,
    )


def evaluate_v3_anchored_cold(
    prepared: V3Prepared,
    retrieval_evaluation: V3Evaluation,
) -> V3Evaluation:
    anchors = _budget_recommendation_maps(prepared, retrieval_evaluation, 0, 0)
    return _adaptive_evaluation(
        prepared,
        method="anchored_cold_thompson",
        purpose="anchored-cold-thompson",
        feature_rows=tuple(fold.base_features for fold in prepared.folds),
        source_models=(None,) * len(prepared.folds),
        transfer_strength=0.0,
        anchors=anchors,
    )


def evaluate_v3_pooled(
    prepared: V3Prepared,
    *,
    transfer_strength: float,
) -> V3Evaluation:
    source_models = tuple(_source_model(prepared, fold) for fold in prepared.folds)
    return _adaptive_evaluation(
        prepared,
        method="pooled_source_thompson",
        purpose="cold-thompson",
        feature_rows=tuple(fold.base_features for fold in prepared.folds),
        source_models=source_models,
        transfer_strength=transfer_strength,
    )


def _parhelion_components(
    prepared: V3Prepared,
    *,
    k: int,
    temperature: float,
) -> tuple[
    tuple[Mapping[tuple[str, str], NDArray[np.float64]], ...],
    tuple[BayesianLinearBandit, ...],
]:
    features = []
    models = []
    for fold in prepared.folds:
        retrieval = RetrievalIndex(fold.archive, k=k, temperature=temperature)
        features.append(_retrieval_features(retrieval, fold, prepared.configs))
        models.append(_source_model(prepared, fold, retrieval=retrieval))
    return tuple(features), tuple(models)


def evaluate_v3_parhelion(
    prepared: V3Prepared,
    *,
    k: int,
    temperature: float,
    transfer_strength: float,
    retrieval_evaluation: V3Evaluation,
) -> V3Evaluation:
    features, models = _parhelion_components(prepared, k=k, temperature=temperature)
    anchors = _budget_recommendation_maps(prepared, retrieval_evaluation, 0, 0)
    return _adaptive_evaluation(
        prepared,
        method="parhelion_thompson",
        purpose="parhelion-thompson",
        feature_rows=features,
        source_models=models,
        transfer_strength=transfer_strength,
        anchors=anchors,
    )


def evaluate_v3_no_anchor(
    prepared: V3Prepared,
    *,
    k: int,
    temperature: float,
    transfer_strength: float,
) -> V3Evaluation:
    features, models = _parhelion_components(prepared, k=k, temperature=temperature)
    return _adaptive_evaluation(
        prepared,
        method="parhelion_no_forced_anchor",
        purpose="parhelion-no-anchor",
        feature_rows=features,
        source_models=models,
        transfer_strength=transfer_strength,
    )


def evaluate_v3_no_transfer(
    prepared: V3Prepared,
    *,
    k: int,
    temperature: float,
    retrieval_evaluation: V3Evaluation,
) -> V3Evaluation:
    features, _models = _parhelion_components(prepared, k=k, temperature=temperature)
    anchors = _budget_recommendation_maps(prepared, retrieval_evaluation, 0, 0)
    return _adaptive_evaluation(
        prepared,
        method="parhelion_no_transfer",
        purpose="parhelion-thompson",
        feature_rows=features,
        source_models=(None,) * len(prepared.folds),
        transfer_strength=0.0,
        anchors=anchors,
    )


def evaluate_v3_random(prepared: V3Prepared) -> V3Evaluation:
    seed_recommendations: list[_FoldPairs] = []
    seed_probes: list[_FoldPairs] = []
    for policy_seed in prepared.seeds:
        fold_recommendations: list[_BudgetPairs] = []
        fold_probes: list[_BudgetPairs] = []
        for fold in prepared.folds:
            rng = np.random.default_rng(
                v3_seed(
                    purpose="random-policy",
                    gpu=prepared.target_gpu,
                    heldout_model=fold.heldout_model,
                    policy_seed=policy_seed,
                )
            )
            queried: dict[str, list[KernelConfig]] = {
                workload.key: [] for workload in fold.target_workloads
            }
            recommendations: list[_ConfigPairs] = []
            probes: list[_ConfigPairs] = []
            for order in _orders(prepared, fold, policy_seed):
                round_probes: dict[str, KernelConfig] = {}
                for workload in order:
                    available = tuple(
                        config for config in prepared.configs if config not in queried[workload.key]
                    )
                    selected = available[int(rng.integers(len(available)))]
                    queried[workload.key].append(selected)
                    round_probes[workload.key] = selected
                incumbents = {
                    workload.key: _best(
                        prepared,
                        fold,
                        workload,
                        queried[workload.key],
                    )
                    for workload in fold.target_workloads
                }
                probes.append(_pairs(fold, round_probes))
                recommendations.append(_pairs(fold, incumbents))
            fold_probes.append(tuple(probes))
            fold_recommendations.append(tuple(recommendations))
        seed_probes.append(tuple(fold_probes))
        seed_recommendations.append(tuple(fold_recommendations))
    return V3Evaluation("random", tuple(seed_recommendations), tuple(seed_probes))


def _budget_recommendation_maps(
    prepared: V3Prepared,
    evaluation: V3Evaluation,
    seed_index: int,
    budget_index: int,
) -> tuple[dict[str, KernelConfig], ...]:
    config_by_key = {config.key: config for config in prepared.configs}
    return tuple(
        {
            workload_key: config_by_key[config_key]
            for workload_key, config_key in evaluation.recommendations[seed_index][fold.index][
                budget_index
            ]
        }
        for fold in prepared.folds
    )


def evaluation_seed_curves(
    prepared: V3Prepared,
    evaluation: V3Evaluation,
    *,
    bank: int = _EVALUATION_BANK,
) -> tuple[tuple[float, ...], ...]:
    """Return one equal-four-fold curve per policy seed for a fixed scoring bank."""
    if bank not in {2, 3, 4}:
        raise ValueError("v3 policy curves may be scored only on banks 2, 3, or 4")
    config_by_key = {config.key: config for config in prepared.configs}
    seed_curves: list[tuple[float, ...]] = []
    for seed_recommendations in evaluation.recommendations:
        fold_curves: list[list[float]] = []
        for fold, budget_pairs in zip(prepared.folds, seed_recommendations, strict=True):
            curve: list[float] = []
            for pairs in budget_pairs:
                recommendations = dict(pairs)
                fractions = []
                for workload in fold.target_workloads:
                    config = config_by_key[recommendations[workload.key]]
                    reference = prepared.table.reference_config(
                        prepared.target_gpu,
                        workload,
                    )
                    fractions.append(
                        _latency(
                            prepared.table,
                            prepared.target_gpu,
                            workload,
                            reference,
                            bank,
                        )
                        / _latency(
                            prepared.table,
                            prepared.target_gpu,
                            workload,
                            config,
                            bank,
                        )
                    )
                curve.append(_geometric_mean(fractions))
            fold_curves.append(curve)
        seed_curves.append(
            tuple(float(value) for value in np.mean(np.asarray(fold_curves), axis=0))
        )
    return tuple(seed_curves)


def evaluation_auc1_8(prepared: V3Prepared, evaluation: V3Evaluation) -> float:
    curves = evaluation_seed_curves(prepared, evaluation)
    return float(np.mean([np.mean(curve[:8]) for curve in curves]))


def select_v3_parameters(prepared: V3Prepared) -> dict[str, object]:
    """Run independent baseline grids then the fixed-winner Parhelion grid on A100."""
    retrieval_scores: list[dict[str, object]] = []
    retrieval_evaluations: dict[tuple[int, float], V3Evaluation] = {}
    for k in V3_K_GRID:
        for temperature in V3_TEMPERATURE_GRID:
            evaluation = evaluate_v3_retrieval(prepared, k=k, temperature=temperature)
            retrieval_evaluations[(k, temperature)] = evaluation
            retrieval_scores.append(
                {
                    "k": k,
                    "temperature": temperature,
                    "auc1_8": evaluation_auc1_8(prepared, evaluation),
                }
            )
    retrieval_winner = min(
        retrieval_scores,
        key=lambda row: (
            -cast(float, row["auc1_8"]),
            cast(int, row["k"]),
            cast(float, row["temperature"]),
        ),
    )
    pooled_scores: list[dict[str, object]] = []
    pooled_evaluations: dict[float, V3Evaluation] = {}
    for strength in V3_TRANSFER_STRENGTH_GRID:
        evaluation = evaluate_v3_pooled(prepared, transfer_strength=strength)
        pooled_evaluations[strength] = evaluation
        pooled_scores.append(
            {"transfer_strength": strength, "auc1_8": evaluation_auc1_8(prepared, evaluation)}
        )
    pooled_winner = min(
        pooled_scores,
        key=lambda row: (
            -cast(float, row["auc1_8"]),
            cast(float, row["transfer_strength"]),
        ),
    )
    retrieval_key = (
        cast(int, retrieval_winner["k"]),
        cast(float, retrieval_winner["temperature"]),
    )
    anchor = retrieval_evaluations[retrieval_key]
    parhelion_scores: list[dict[str, object]] = []
    parhelion_evaluations: dict[tuple[int, float, float], V3Evaluation] = {}
    for k in V3_K_GRID:
        for temperature in V3_TEMPERATURE_GRID:
            for strength in V3_TRANSFER_STRENGTH_GRID:
                evaluation = evaluate_v3_parhelion(
                    prepared,
                    k=k,
                    temperature=temperature,
                    transfer_strength=strength,
                    retrieval_evaluation=anchor,
                )
                key = (k, temperature, strength)
                parhelion_evaluations[key] = evaluation
                parhelion_scores.append(
                    {
                        "k": k,
                        "temperature": temperature,
                        "transfer_strength": strength,
                        "auc1_8": evaluation_auc1_8(prepared, evaluation),
                    }
                )
    parhelion_winner = min(
        parhelion_scores,
        key=lambda row: (
            -cast(float, row["auc1_8"]),
            cast(int, row["k"]),
            cast(float, row["temperature"]),
            cast(float, row["transfer_strength"]),
        ),
    )
    return {
        "schema_version": 1,
        "study_id": "parhelion-v3-a100-selection",
        "selection_gpu": prepared.target_gpu,
        "source_gpus": list(prepared.source_gpus),
        "seeds": list(prepared.seeds),
        "budgets": list(prepared.budgets),
        "primary_budgets": list(range(1, 9)),
        "candidate_scores": {
            "multisource_retrieval": retrieval_scores,
            "pooled_source_thompson": pooled_scores,
            "parhelion_thompson": parhelion_scores,
        },
        "selected": {
            "multisource_retrieval": retrieval_winner,
            "pooled_source_thompson": pooled_winner,
            "parhelion_thompson": parhelion_winner,
        },
    }


__all__ = [
    "V3Evaluation",
    "V3Fold",
    "V3Parameters",
    "V3Prepared",
    "evaluate_v3_anchored_cold",
    "evaluate_v3_cold",
    "evaluate_v3_nearest",
    "evaluate_v3_no_anchor",
    "evaluate_v3_no_transfer",
    "evaluate_v3_parhelion",
    "evaluate_v3_pooled",
    "evaluate_v3_random",
    "evaluate_v3_retrieval",
    "evaluation_auc1_8",
    "evaluation_seed_curves",
    "prepare_v3",
    "select_v3_parameters",
]
