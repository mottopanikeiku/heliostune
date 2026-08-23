"""Leakage-resistant replay over disjoint observation, reference, and evaluation banks."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import numpy as np

from heliostune.bandit import BayesianLinearBandit
from heliostune.configs import KernelConfig, Workload
from heliostune.features import FEATURE_NAMES, joint_features
from heliostune.schema import HardwareProfile, Measurement

ActionT = TypeVar("ActionT")
_OBSERVATION_BANK = 0
_REFERENCE_BANK = 1
_EVALUATION_BANK = 2
_METHOD_LABELS = {
    "static": "Static source-best",
    "torch": "torch.matmul",
    "random": "Random search",
    "nearest_shape": "Nearest-shape reuse",
    "cold_thompson": "Cold Thompson sampling",
    "transfer_thompson": "Helios transfer",
    "exhaustive": "Exhaustive autotuning",
    "heldout_reference": "Held-out exhaustive reference",
}


class BenchmarkTable:
    """Indexed benchmark observations with strict matrix and bank checks."""

    def __init__(self, measurements: Iterable[Measurement]) -> None:
        self.measurements = tuple(measurements)
        if not self.measurements:
            raise ValueError("benchmark table must not be empty")
        self._index: dict[tuple[str, str, str, int], Measurement] = {}
        for measurement in self.measurements:
            key = (
                measurement.hardware.gpu,
                measurement.workload.key,
                measurement.config.key,
                measurement.replicate,
            )
            if key in self._index:
                raise ValueError(f"duplicate measurement: {key}")
            self._index[key] = measurement

    @property
    def gpus(self) -> tuple[str, ...]:
        return tuple(sorted({item.hardware.gpu for item in self.measurements}))

    def hardware(self, gpu: str) -> HardwareProfile:
        profiles = {item.hardware for item in self.measurements if item.hardware.gpu == gpu}
        if len(profiles) != 1:
            raise ValueError(f"expected one hardware profile for {gpu!r}, found {len(profiles)}")
        return profiles.pop()

    def workloads(self, gpu: str) -> tuple[Workload, ...]:
        values = {
            item.workload.key: item.workload
            for item in self.measurements
            if item.hardware.gpu == gpu
        }
        return tuple(values[key] for key in sorted(values))

    def configs(self, gpu: str) -> tuple[KernelConfig, ...]:
        values = {
            item.config.key: item.config for item in self.measurements if item.hardware.gpu == gpu
        }
        return tuple(values[key] for key in sorted(values))

    def replicates(self, gpu: str) -> tuple[int, ...]:
        return tuple(
            sorted({item.replicate for item in self.measurements if item.hardware.gpu == gpu})
        )

    def get(
        self,
        gpu: str,
        workload: Workload,
        config: KernelConfig,
        replicate: int = _OBSERVATION_BANK,
    ) -> Measurement:
        try:
            return self._index[(gpu, workload.key, config.key, replicate)]
        except KeyError as exc:
            raise KeyError((gpu, workload.key, config.key, replicate)) from exc

    def validate_protocol(self, source_gpu: str, target_gpu: str) -> None:
        if source_gpu not in self.gpus or target_gpu not in self.gpus:
            raise ValueError(f"dataset GPUs are {self.gpus}, not {source_gpu!r}/{target_gpu!r}")
        source_workloads = {item.key for item in self.workloads(source_gpu)}
        target_workloads = {item.key for item in self.workloads(target_gpu)}
        source_configs = {item.key for item in self.configs(source_gpu)}
        target_configs = {item.key for item in self.configs(target_gpu)}
        if source_workloads != target_workloads:
            raise ValueError("source and target workload sets differ")
        if source_configs != target_configs:
            raise ValueError("source and target configuration sets differ")
        required_banks = {_OBSERVATION_BANK, _REFERENCE_BANK, _EVALUATION_BANK}
        for gpu in (source_gpu, target_gpu):
            if not required_banks.issubset(self.replicates(gpu)):
                raise ValueError(f"{gpu} requires independent replicate banks 0, 1, and 2")
            for workload in self.workloads(gpu):
                for config in self.configs(gpu):
                    for bank in required_banks:
                        measurement = self.get(gpu, workload, config, bank)
                        if not measurement.usable:
                            raise ValueError(
                                f"invalid frozen cell {gpu}/{workload.key}/{config.key}/bank-{bank}: "
                                f"{measurement.error}"
                            )

    def reference_config(self, gpu: str, workload: Workload) -> KernelConfig:
        """Select a configuration using only the independent reference bank."""

        return min(
            self.configs(gpu),
            key=lambda config: (
                self._latency(gpu, workload, config, _REFERENCE_BANK),
                config.key,
            ),
        )

    def heldout_fraction(
        self,
        gpu: str,
        workload: Workload,
        recommendation: KernelConfig,
    ) -> float:
        """Evaluate a recommendation and the reference winner only on held-out bank 2."""

        reference = self.reference_config(gpu, workload)
        return self._latency(gpu, workload, reference, _EVALUATION_BANK) / self._latency(
            gpu, workload, recommendation, _EVALUATION_BANK
        )

    def _latency(
        self,
        gpu: str,
        workload: Workload,
        config: KernelConfig,
        replicate: int,
    ) -> float:
        value = self.get(gpu, workload, config, replicate).latency_ms
        if value is None:
            raise ValueError("validated measurement unexpectedly has no latency")
        return value


def _reward(measurement: Measurement) -> float:
    if not measurement.usable or measurement.latency_ms is None:
        raise ValueError("invalid cells cannot be converted into arbitrary reward penalties")
    return -math.log(measurement.latency_ms)


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _evaluate_recommendations(
    table: BenchmarkTable,
    target_gpu: str,
    recommendations: dict[str, KernelConfig],
    workloads: Sequence[Workload],
) -> float:
    return _geometric_mean(
        [
            table.heldout_fraction(target_gpu, workload, recommendations[workload.key])
            for workload in workloads
        ]
    )


def _best_observed(
    table: BenchmarkTable,
    target_gpu: str,
    workload: Workload,
    queried: Sequence[KernelConfig],
) -> KernelConfig:
    return max(
        queried,
        key=lambda config: (
            _reward(table.get(target_gpu, workload, config, _OBSERVATION_BANK)),
            config.key,
        ),
    )


def _model_recommendation(
    model: BayesianLinearBandit,
    workload: Workload,
    configs: Sequence[KernelConfig],
    hardware: HardwareProfile,
) -> KernelConfig:
    coefficients = model.mean
    return max(
        configs,
        key=lambda config: (
            float(joint_features(workload, config, hardware) @ coefficients),
            config.key,
        ),
    )


def _source_model(
    table: BenchmarkTable,
    source_gpu: str,
    configs: Sequence[KernelConfig],
    workloads: Sequence[Workload],
    *,
    noise_variance: float,
    prior_precision: float,
) -> BayesianLinearBandit:
    model = BayesianLinearBandit(
        dimension=len(FEATURE_NAMES),
        noise_variance=noise_variance,
        prior_precision=prior_precision,
        seed=0,
    )
    hardware = table.hardware(source_gpu)
    for workload in workloads:
        for config in configs:
            observation = table.get(source_gpu, workload, config, _OBSERVATION_BANK)
            model.update(joint_features(workload, config, hardware), _reward(observation))
    return model


def _nearest_workload(workload: Workload, candidates: Sequence[Workload]) -> Workload:
    def distance(candidate: Workload) -> tuple[float, str]:
        squared = sum(
            (math.log2(left) - math.log2(right)) ** 2
            for left, right in zip(
                (workload.m, workload.n, workload.k),
                (candidate.m, candidate.n, candidate.k),
                strict=True,
            )
        )
        return squared, candidate.key

    return min(candidates, key=distance)


def _source_rank(
    table: BenchmarkTable,
    source_gpu: str,
    workload: Workload,
    configs: Sequence[KernelConfig],
) -> list[KernelConfig]:
    return sorted(
        configs,
        key=lambda config: (
            table._latency(source_gpu, workload, config, _OBSERVATION_BANK),
            config.key,
        ),
    )


def _static_source_best(
    table: BenchmarkTable,
    source_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
) -> KernelConfig:
    per_workload_best = {
        workload.key: min(
            table._latency(source_gpu, workload, config, _OBSERVATION_BANK) for config in configs
        )
        for workload in workloads
    }
    return min(
        configs,
        key=lambda config: (
            _geometric_mean(
                [
                    table._latency(source_gpu, workload, config, _OBSERVATION_BANK)
                    / per_workload_best[workload.key]
                    for workload in workloads
                ]
            ),
            config.key,
        ),
    )


def _random_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
    orders: Sequence[Sequence[Workload]],
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    queried = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for order in orders:
        for workload in order:
            available = [config for config in configs if config not in queried[workload.key]]
            queried[workload.key].append(available[int(rng.integers(len(available)))])
        recommendations = {
            workload.key: _best_observed(table, target_gpu, workload, queried[workload.key])
            for workload in workloads
        }
        curve.append(_evaluate_recommendations(table, target_gpu, recommendations, workloads))
    return curve


def _ranked_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    ranks: dict[str, Sequence[KernelConfig]],
    max_budget: int,
) -> list[float]:
    queried = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for budget_index in range(max_budget):
        for workload in workloads:
            queried[workload.key].append(ranks[workload.key][budget_index])
        recommendations = {
            workload.key: _best_observed(table, target_gpu, workload, queried[workload.key])
            for workload in workloads
        }
        curve.append(_evaluate_recommendations(table, target_gpu, recommendations, workloads))
    return curve


def _bandit_curve(
    table: BenchmarkTable,
    target_gpu: str,
    workloads: Sequence[Workload],
    configs: Sequence[KernelConfig],
    orders: Sequence[Sequence[Workload]],
    model: BayesianLinearBandit,
) -> list[float]:
    hardware = table.hardware(target_gpu)
    queried = {workload.key: [] for workload in workloads}
    curve: list[float] = []
    for order in orders:
        for workload in order:
            available = [config for config in configs if config not in queried[workload.key]]
            selected = model.choose(
                available,
                lambda config, workload=workload: joint_features(workload, config, hardware),
            )
            queried[workload.key].append(selected)
            model.update(
                joint_features(workload, selected, hardware),
                _reward(table.get(target_gpu, workload, selected, _OBSERVATION_BANK)),
            )
        recommendations = {
            workload.key: _model_recommendation(model, workload, configs, hardware)
            for workload in workloads
        }
        curve.append(_evaluate_recommendations(table, target_gpu, recommendations, workloads))
    return curve


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
    return [value for _ in range(length)]


def compare_methods(
    measurements: Iterable[Measurement],
    *,
    source_gpu: str,
    target_gpu: str,
    max_budget: int = 8,
    seeds: int = 30,
    transfer_strength: float = 0.08,
    noise_variance: float = 0.05,
    prior_precision: float = 1.0,
) -> dict[str, Any]:
    """Run grouped leave-one-model-family-out replay in one GPU direction.

    Bank 0 is the only target latency stream visible to policies. Bank 1 selects
    the exhaustive reference configuration. Bank 2 evaluates every policy and
    the bank-1 winner. A target model family is never present in its source
    posterior, static baseline, or nearest-shape archive.
    """

    if source_gpu == target_gpu:
        raise ValueError("source_gpu and target_gpu must differ")
    if max_budget <= 0 or seeds <= 0:
        raise ValueError("max_budget and seeds must be positive")
    if not 0 <= transfer_strength <= 1:
        raise ValueError("transfer_strength must be between zero and one")

    table = BenchmarkTable(measurements)
    table.validate_protocol(source_gpu, target_gpu)
    configs = table.configs(target_gpu)
    max_budget = min(max_budget, len(configs))
    budgets = list(range(1, max_budget + 1))
    all_workloads = table.workloads(target_gpu)
    models = sorted({workload.model for workload in all_workloads})
    if len(models) < 2:
        raise ValueError("grouped transfer evaluation requires at least two model families")

    runs: dict[str, list[list[float]]] = {
        "static": [],
        "torch": [],
        "random": [],
        "nearest_shape": [],
        "cold_thompson": [],
        "transfer_thompson": [],
    }
    fold_details: list[dict[str, Any]] = []
    for fold_index, heldout_model in enumerate(models):
        source_workloads = tuple(
            workload for workload in all_workloads if workload.model != heldout_model
        )
        target_workloads = tuple(
            workload for workload in all_workloads if workload.model == heldout_model
        )
        source_model = _source_model(
            table,
            source_gpu,
            configs,
            source_workloads,
            noise_variance=noise_variance,
            prior_precision=prior_precision,
        )

        static_config = _static_source_best(table, source_gpu, source_workloads, configs)
        static_value = _evaluate_recommendations(
            table,
            target_gpu,
            {workload.key: static_config for workload in target_workloads},
            target_workloads,
        )
        runs["static"].append(_constant_curve(static_value, max_budget))

        torch_value = _geometric_mean(
            [
                table._latency(
                    target_gpu,
                    workload,
                    table.reference_config(target_gpu, workload),
                    _EVALUATION_BANK,
                )
                / table.get(
                    target_gpu,
                    workload,
                    configs[0],
                    _EVALUATION_BANK,
                ).torch_latency_ms
                for workload in target_workloads
            ]
        )
        runs["torch"].append(_constant_curve(torch_value, max_budget))

        nearest_ranks = {
            workload.key: _source_rank(
                table,
                source_gpu,
                _nearest_workload(workload, source_workloads),
                configs,
            )
            for workload in target_workloads
        }
        runs["nearest_shape"].append(
            _ranked_curve(
                table,
                target_gpu,
                target_workloads,
                nearest_ranks,
                max_budget,
            )
        )

        for seed in range(seeds):
            paired_seed = fold_index * 100_000 + seed
            order_rng = np.random.default_rng(paired_seed + 50_000)
            orders = [
                tuple(
                    target_workloads[index]
                    for index in order_rng.permutation(len(target_workloads))
                )
                for _ in budgets
            ]
            runs["random"].append(
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
                noise_variance=noise_variance,
                prior_precision=prior_precision,
                seed=paired_seed,
            )
            runs["cold_thompson"].append(
                _bandit_curve(
                    table,
                    target_gpu,
                    target_workloads,
                    configs,
                    orders,
                    cold,
                )
            )
            transfer = source_model.transferred(
                transfer_strength=transfer_strength,
                seed=paired_seed,
            )
            runs["transfer_thompson"].append(
                _bandit_curve(
                    table,
                    target_gpu,
                    target_workloads,
                    configs,
                    orders,
                    transfer,
                )
            )

        fold_details.append(
            {
                "heldout_model": heldout_model,
                "source_workloads": len(source_workloads),
                "target_workloads": len(target_workloads),
                "static_config": static_config.key,
            }
        )

    methods = {method: _aggregate(method_runs, budgets) for method, method_runs in runs.items()}
    exhaustive_runs: list[list[float]] = []
    for heldout_model in models:
        target_workloads = tuple(
            workload for workload in all_workloads if workload.model == heldout_model
        )
        recommendations = {
            workload.key: min(
                configs,
                key=lambda config, workload=workload: (
                    table._latency(
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
        exhaustive_runs.append(
            [_evaluate_recommendations(table, target_gpu, recommendations, target_workloads)]
        )
    methods["exhaustive"] = _aggregate(exhaustive_runs, [len(configs)])
    methods["heldout_reference"] = [
        {
            "budget": len(configs),
            "mean_fraction_oracle": 1.0,
            "ci95_low": 1.0,
            "ci95_high": 1.0,
        }
    ]

    headline_budget = min(8, max_budget)
    point_index = headline_budget - 1
    curve_methods = (
        "static",
        "random",
        "nearest_shape",
        "cold_thompson",
        "transfer_thompson",
    )
    auc = {
        method: float(np.mean([point["mean_fraction_oracle"] for point in methods[method]]))
        for method in curve_methods
    }
    strongest_legacy = max(
        ("static", "random", "nearest_shape", "cold_thompson"),
        key=lambda method: (auc[method], method),
    )
    queries_to_95 = {
        method: next(
            (
                int(point["budget"])
                for point in methods[method]
                if point["mean_fraction_oracle"] >= 0.95
            ),
            None,
        )
        for method in curve_methods
    }
    measurements_per_gpu = len(all_workloads) * len(configs) * 3
    return {
        "project": "HeliosTune",
        "data_kind": "measured",
        "source_gpu": source_gpu,
        "target_gpu": target_gpu,
        "methodology": (
            "Grouped leave-one-model-family-out transfer. Policies see only timing bank 0; "
            "bank 1 selects the best-of-manifest reference and bank 2 evaluates every "
            "recommendation. Source acquisition is disclosed outside the target query budget."
        ),
        "workloads": len(all_workloads),
        "configs": len(configs),
        "model_families": len(models),
        "measurement_banks": 3,
        "max_budget": max_budget,
        "seeds": seeds,
        "transfer_strength": transfer_strength,
        "hardware": [table.hardware(source_gpu).to_dict(), table.hardware(target_gpu).to_dict()],
        "methods": methods,
        "method_labels": _METHOD_LABELS,
        "headline": {
            "budget": headline_budget,
            "transfer_fraction_oracle": methods["transfer_thompson"][point_index][
                "mean_fraction_oracle"
            ],
            "cold_fraction_oracle": methods["cold_thompson"][point_index]["mean_fraction_oracle"],
            "random_fraction_oracle": methods["random"][point_index]["mean_fraction_oracle"],
            "transfer_auc": auc["transfer_thompson"],
            "strongest_legacy_method": strongest_legacy,
            "auc_delta_vs_strongest_legacy": (auc["transfer_thompson"] - auc[strongest_legacy]),
            "trials_avoided_vs_exhaustive": len(configs) - headline_budget,
        },
        "primary_metrics": {
            "fraction_reference_auc": auc,
            "queries_to_95_percent_reference": queries_to_95,
            "primary_budget": headline_budget,
        },
        "source_cost": {
            "collected_measurements_per_gpu": measurements_per_gpu,
            "visible_source_observations_per_fold": (
                len(all_workloads) - len(all_workloads) // len(models)
            )
            * len(configs),
            "disclosure": (
                "The transfer prior sees bank-0 source measurements from non-held-out model "
                "families. Source acquisition and all reference/evaluation banks are excluded "
                "from the target query budget and disclosed separately."
            ),
        },
        "folds": fold_details,
        "experiment": {
            "workload_keys": [workload.key for workload in all_workloads],
            "config_keys": [config.key for config in configs],
            "target_budget_unit": "bank-0 configuration measurements per held-out workload",
            "reward": "-log(selected configuration latency_ms)",
            "bank_roles": {
                "0": "policy-visible observations",
                "1": "exhaustive-reference selection",
                "2": "held-out final evaluation",
            },
            "aggregation": "geometric mean fraction of held-out exhaustive reference",
            "confidence_interval": "normal 95% interval across grouped folds and policy seeds",
        },
        "limitations": [
            "The study demonstrates transfer between Modal L4 and A10, not arbitrary unseen GPUs.",
            "Replay measures steady-state kernel probes, not compilation or end-to-end serving latency.",
            "The 36-action manifest is curated rather than every legal Triton configuration.",
            "Three independent timing banks reduce leakage but do not model production interference.",
        ],
    }
