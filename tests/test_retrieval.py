import math

import pytest

from heliostune.configs import KernelConfig, Workload
from heliostune.retrieval import (
    ArchiveObservation,
    RetrievalIndex,
    RetrievalStats,
    log_tflops_reward,
)


def _configs() -> tuple[KernelConfig, KernelConfig]:
    return (
        KernelConfig(16, 32, 16, 4, 2),
        KernelConfig(32, 32, 16, 4, 2),
    )


def _workload(model: str, *, m: int = 32, projection: str = "q") -> Workload:
    return Workload(m, 32, 32, model, projection, "test")


def _row(
    workload: Workload,
    source_gpu: str,
    configs: tuple[KernelConfig, KernelConfig],
    latencies: tuple[float, float],
) -> tuple[ArchiveObservation, ArchiveObservation]:
    return tuple(
        ArchiveObservation(workload, config, source_gpu, latency)
        for config, latency in zip(configs, latencies, strict=True)
    )


def test_log_tflops_reward_and_latency_validation() -> None:
    workload = _workload("target")
    config = _configs()[0]
    latency_ms = 0.5

    assert log_tflops_reward(workload, latency_ms) == pytest.approx(
        math.log(workload.flops / (latency_ms * 1e9))
    )
    for invalid in (0.0, -1.0, math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            log_tflops_reward(workload, invalid)
        with pytest.raises(ValueError, match="finite and positive"):
            ArchiveObservation(workload, config, "a10", invalid)


def test_incomplete_archive_rows_and_source_coverage_are_rejected() -> None:
    configs = _configs()
    first = _workload("source-a")
    second = _workload("source-b", m=64)

    incomplete_actions = (
        *_row(first, "a10", configs, (1.0, 2.0)),
        ArchiveObservation(second, configs[0], "a10", 1.0),
    )
    with pytest.raises(ValueError, match="incomplete source action row"):
        RetrievalIndex(incomplete_actions)

    incomplete_sources = (
        *_row(first, "a10", configs, (1.0, 2.0)),
        *_row(first, "l4", configs, (1.5, 2.5)),
        *_row(second, "a10", configs, (2.0, 3.0)),
    )
    with pytest.raises(ValueError, match="incomplete source-GPU coverage"):
        RetrievalIndex(incomplete_sources)


def test_same_model_family_is_excluded_from_neighbors() -> None:
    configs = _configs()
    target = _workload("family-a", projection="target")
    same_family = _workload("family-a", projection="archive")
    other_family = _workload("family-b", m=64)
    index = RetrievalIndex(
        (
            *_row(same_family, "a10", configs, (1.0, 16.0)),
            *_row(other_family, "a10", configs, (16.0, 1.0)),
        ),
        k=1,
    )

    stats = index.score(target, configs[1])
    assert stats.normalized_neighbor_distance == pytest.approx(
        RetrievalIndex.normalized_distance(target, other_family)
    )
    assert index.rank(target, configs) == (configs[1], configs[0])


def test_ranking_ties_are_key_ordered_independent_of_input_order() -> None:
    configs = _configs()
    source = _workload("source")
    target = _workload("target")
    index = RetrievalIndex(_row(source, "a10", configs, (2.0, 2.0)), k=1)
    expected = tuple(sorted(configs, key=lambda config: config.key))

    assert index.rank(target, reversed(configs)) == expected
    assert index.rank(target, configs) == expected


def test_source_gpus_are_balanced_and_emit_four_finite_features() -> None:
    configs = _configs()
    source = _workload("source")
    target = _workload("target")
    index = RetrievalIndex(
        (
            *_row(source, "a10", configs, (1.0, 4.0)),
            *_row(source, "l4", configs, (4.0, 1.0)),
        ),
        k=1,
    )

    stats = index.score(target, configs[0])
    features = stats.as_array()
    assert isinstance(stats, RetrievalStats)
    assert isinstance(features, tuple)
    assert len(features) == 4
    assert all(math.isfinite(value) for value in features)
    assert features == pytest.approx(
        (0.0, math.log(2.0) ** 2, 0.0, 0.0),
        abs=1e-12,
    )
