"""Retrieval statistics for the multi-source Parhelion tuning prior.

The index deliberately uses only source measurements.  When scoring a target
workload, every archive row from the same model family is excluded before the
nearest workload shapes are selected.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from heliostune.configs import KernelConfig, Workload

RETRIEVAL_FEATURE_NAMES: tuple[str, ...] = (
    "weighted_advantage_mean",
    "weighted_variance",
    "normalized_neighbor_distance",
    "source_gpu_agreement",
)
"""Names and order of the four features returned by :meth:`RetrievalStats.as_array`."""

# Workload dimensions in the existing joint feature map are normalized by the
# fixed exponent 14.  Keeping that constant here makes distances independent of
# archive composition and therefore safe to freeze before final evaluation.
_LOG_SHAPE_NORMALIZER = 14.0


@dataclass(frozen=True, slots=True)
class ArchiveObservation:
    """One usable source-GPU latency for an archive workload and action."""

    workload: Workload
    config: KernelConfig
    source_gpu: str
    latency_ms: float

    def __post_init__(self) -> None:
        if not self.source_gpu:
            raise ValueError("source_gpu must not be empty")
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0.0:
            raise ValueError("latency_ms must be finite and positive")


@dataclass(frozen=True, slots=True)
class RetrievalStats:
    """The four retrieval features appended to Parhelion's joint feature vector.

    ``source_gpu_agreement`` is in ``[0, 1]``.  It is one when the selected
    source GPUs agree on whether an action is above or below their local
    workload means, zero for an evenly split sign vote, and one when every
    source is exactly indifferent.
    """

    weighted_advantage_mean: float
    weighted_variance: float
    normalized_neighbor_distance: float
    source_gpu_agreement: float

    def __post_init__(self) -> None:
        values = self.as_array()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("retrieval statistics must be finite")
        if self.weighted_variance < 0.0:
            raise ValueError("weighted_variance must not be negative")
        if self.normalized_neighbor_distance < 0.0:
            raise ValueError("normalized_neighbor_distance must not be negative")
        if not 0.0 <= self.source_gpu_agreement <= 1.0:
            raise ValueError("source_gpu_agreement must be between zero and one")

    def as_array(self) -> tuple[float, float, float, float]:
        """Return the features in :data:`RETRIEVAL_FEATURE_NAMES` order."""

        return (
            self.weighted_advantage_mean,
            self.weighted_variance,
            self.normalized_neighbor_distance,
            self.source_gpu_agreement,
        )


@dataclass(frozen=True, slots=True)
class _SourceRow:
    workload: Workload
    source_gpu: str
    advantages: Mapping[KernelConfig, float]


@dataclass(frozen=True, slots=True)
class _Neighbor:
    distance: float
    weight: float
    rows: tuple[_SourceRow, ...]


def log_tflops_reward(workload: Workload, latency_ms: float) -> float:
    """Return the natural logarithm of achieved TFLOP/s.

    A matrix multiplication performs ``workload.flops`` operations and a
    millisecond is ``1e-3`` seconds, so TFLOP/s is
    ``workload.flops / (latency_ms * 1e9)``.  The logarithm is evaluated as a
    sum of logarithms to avoid overflow for otherwise valid finite latencies.
    """

    if not math.isfinite(latency_ms) or latency_ms <= 0.0:
        raise ValueError("latency_ms must be finite and positive")
    reward = math.log(workload.flops) - math.log(latency_ms) - math.log(1e9)
    if not math.isfinite(reward):
        raise ValueError("log-TFLOP/s reward must be finite")
    return reward


class RetrievalIndex:
    """Pre-indexed multi-source archive used by Parhelion retrieval.

    The archive must contain exactly one observation for every action in every
    ``(source_gpu, workload)`` row, and every workload must have the same
    nonempty source-GPU set.  ``k`` counts distinct neighboring workload
    shapes, not duplicated GPU rows.  Every source GPU contributes equally to
    a selected shape's distance weight.

    Distances are frozen normalized Euclidean distances between
    ``log2(M, N, K) / 14`` coordinates.  Neighbor weights are proportional to
    ``exp(-distance / temperature)`` and are evaluated in a shifted form for
    numerical stability.
    """

    def __init__(
        self,
        observations: Iterable[ArchiveObservation],
        k: int = 3,
        temperature: float = 0.7,
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

        archive = tuple(observations)
        if not archive:
            raise ValueError("retrieval archive must not be empty")

        grouped: dict[tuple[str, Workload], dict[KernelConfig, float]] = {}
        for observation in archive:
            row_key = (observation.source_gpu, observation.workload)
            action_rewards = grouped.setdefault(row_key, {})
            if observation.config in action_rewards:
                raise ValueError(
                    "duplicate archive action for "
                    f"{observation.source_gpu}/{observation.workload.key}/{observation.config.key}"
                )
            action_rewards[observation.config] = log_tflops_reward(
                observation.workload, observation.latency_ms
            )

        action_set = frozenset(observation.config for observation in archive)
        for (source_gpu, workload), action_rewards in grouped.items():
            row_actions = frozenset(action_rewards)
            if row_actions != action_set:
                missing = sorted(config.key for config in action_set - row_actions)
                extra = sorted(config.key for config in row_actions - action_set)
                details = []
                if missing:
                    details.append(f"missing {', '.join(missing)}")
                if extra:
                    details.append(f"unexpected {', '.join(extra)}")
                raise ValueError(
                    f"incomplete source action row {source_gpu}/{workload.key}: "
                    + "; ".join(details)
                )
        source_gpus = frozenset(source_gpu for source_gpu, _ in grouped)
        workload_sources: defaultdict[Workload, set[str]] = defaultdict(set)
        for source_gpu, workload in grouped:
            workload_sources[workload].add(source_gpu)
        for workload, available_sources in workload_sources.items():
            if available_sources != source_gpus:
                missing_sources = sorted(source_gpus - available_sources)
                raise ValueError(
                    f"incomplete source-GPU coverage for {workload.key}: "
                    f"missing {', '.join(missing_sources)}"
                )

        rows_by_workload: defaultdict[Workload, list[_SourceRow]] = defaultdict(list)
        for (source_gpu, workload), action_rewards in grouped.items():
            row_mean = math.fsum(
                action_rewards[config]
                for config in sorted(action_rewards, key=lambda action: action.key)
            ) / len(action_rewards)
            advantages = {config: reward - row_mean for config, reward in action_rewards.items()}
            rows_by_workload[workload].append(
                _SourceRow(
                    workload=workload,
                    source_gpu=source_gpu,
                    advantages=advantages,
                )
            )

        self.k = k
        self.temperature = float(temperature)
        self.observations = archive
        self.configs = tuple(sorted(action_set, key=lambda config: config.key))
        self._action_set = action_set
        self._rows_by_workload = {
            workload: tuple(sorted(rows, key=lambda row: row.source_gpu))
            for workload, rows in rows_by_workload.items()
        }
        self._neighbor_cache: dict[Workload, tuple[_Neighbor, ...]] = {}

    @staticmethod
    def normalized_distance(left: Workload, right: Workload) -> float:
        """Return the frozen normalized log-shape distance between workloads."""

        squared_distance = math.fsum(
            ((math.log2(a) - math.log2(b)) / _LOG_SHAPE_NORMALIZER) ** 2
            for a, b in zip(
                (left.m, left.n, left.k),
                (right.m, right.n, right.k),
                strict=True,
            )
        )
        return math.sqrt(squared_distance)

    def _neighbors(self, workload: Workload) -> tuple[_Neighbor, ...]:
        cached = self._neighbor_cache.get(workload)
        if cached is not None:
            return cached

        candidates = [
            (self.normalized_distance(workload, source_workload), source_workload)
            for source_workload in self._rows_by_workload
            if source_workload.model != workload.model
        ]
        if not candidates:
            raise ValueError(
                f"no eligible retrieval neighbor remains after excluding model family {workload.model!r}"
            )
        candidates.sort(key=lambda item: (item[0], item[1].key))
        selected = candidates[: self.k]
        nearest_distance = selected[0][0]
        neighbors = tuple(
            _Neighbor(
                distance=distance,
                weight=math.exp(-(distance - nearest_distance) / self.temperature),
                rows=self._rows_by_workload[source_workload],
            )
            for distance, source_workload in selected
        )
        self._neighbor_cache[workload] = neighbors
        return neighbors

    def score(self, workload: Workload, config: KernelConfig) -> RetrievalStats:
        """Return Parhelion retrieval features for one target workload/action pair.

        Source rows belonging to ``workload.model`` are excluded.  The method
        uses cached shape neighbors and pre-centered action rows, so scoring all
        actions for one workload does not rescan the observation table.
        """

        if config not in self._action_set:
            raise ValueError(f"configuration {config.key} is not present in the retrieval archive")

        neighbors = self._neighbors(workload)
        weighted_advantages: list[tuple[float, float]] = []
        gpu_weighted_sums: defaultdict[str, float] = defaultdict(float)
        gpu_weights: defaultdict[str, float] = defaultdict(float)

        for neighbor in neighbors:
            # A source shape has a fixed total influence even if measurements
            # happen to be available from more GPUs for that shape.
            row_weight = neighbor.weight / len(neighbor.rows)
            for row in neighbor.rows:
                advantage = row.advantages[config]
                weighted_advantages.append((row_weight, advantage))
                gpu_weighted_sums[row.source_gpu] += row_weight * advantage
                gpu_weights[row.source_gpu] += row_weight

        total_weight = math.fsum(weight for weight, _ in weighted_advantages)
        mean = (
            math.fsum(weight * advantage for weight, advantage in weighted_advantages)
            / total_weight
        )
        variance = (
            math.fsum(weight * (advantage - mean) ** 2 for weight, advantage in weighted_advantages)
            / total_weight
        )
        distance_weight = math.fsum(neighbor.weight for neighbor in neighbors)
        neighbor_distance = (
            math.fsum(neighbor.weight * neighbor.distance for neighbor in neighbors)
            / distance_weight
        )

        gpu_means = tuple(gpu_weighted_sums[gpu] / gpu_weights[gpu] for gpu in sorted(gpu_weights))
        signs = tuple((value > 0.0) - (value < 0.0) for value in gpu_means)
        agreement = 1.0 if not any(signs) else abs(math.fsum(signs)) / len(signs)

        return RetrievalStats(
            weighted_advantage_mean=mean,
            weighted_variance=max(0.0, variance),
            normalized_neighbor_distance=neighbor_distance,
            source_gpu_agreement=agreement,
        )

    def rank(
        self,
        workload: Workload,
        configs: Sequence[KernelConfig] | Iterable[KernelConfig],
    ) -> tuple[KernelConfig, ...]:
        """Rank unique actions by consensus retrieval score, then stable key.

        Higher weighted mean advantage is better.  Duplicate actions are
        rejected rather than silently changing the caller's action space.
        """

        actions = tuple(configs)
        if len(set(actions)) != len(actions):
            raise ValueError("configs must contain unique actions")
        return tuple(
            sorted(
                actions,
                key=lambda config: (
                    -self.score(workload, config).weighted_advantage_mean,
                    config.key,
                ),
            )
        )


__all__ = [
    "ArchiveObservation",
    "RETRIEVAL_FEATURE_NAMES",
    "RetrievalIndex",
    "RetrievalStats",
    "log_tflops_reward",
]
