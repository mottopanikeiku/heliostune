import math
from typing import NamedTuple

import pytest
from hypothesis import given
from hypothesis import strategies as st

from heliostune.configs import KernelConfig, Workload
from heliostune.retrieval import (
    RETRIEVAL_FEATURE_NAMES,
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


class _Archive(NamedTuple):
    """A schema-valid archive together with a permutation of the same rows."""

    observations: tuple[ArchiveObservation, ...]
    permuted: tuple[ArchiveObservation, ...]
    configs: tuple[KernelConfig, ...]
    target: Workload
    k: int
    temperature: float


_SHAPE_DIMENSIONS = (16, 32, 64, 128, 256, 512)
_CONFIG_POOL = (
    KernelConfig(16, 32, 16, 4, 2),
    KernelConfig(32, 32, 16, 4, 2),
    KernelConfig(64, 64, 32, 8, 3),
    KernelConfig(32, 64, 32, 2, 4),
)
_SOURCE_GPU_POOL = ("a10", "l4", "t4")
_LATENCIES = st.floats(
    min_value=1e-2,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _archives(draw: st.DrawFn) -> _Archive:
    """Draw a fully populated multi-source archive and a target workload.

    Every ``(source_gpu, workload)`` row carries the identical action set and
    every workload is covered by every source GPU, which is what
    :class:`RetrievalIndex` requires. Source model families are all distinct
    from the target's, so no draw leaves the neighbor set empty.
    """
    shapes = draw(
        st.lists(
            st.tuples(
                st.sampled_from(_SHAPE_DIMENSIONS),
                st.sampled_from(_SHAPE_DIMENSIONS),
                st.sampled_from(_SHAPE_DIMENSIONS),
            ),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    configs = tuple(
        draw(
            st.lists(
                st.sampled_from(_CONFIG_POOL),
                min_size=1,
                max_size=len(_CONFIG_POOL),
                unique=True,
            )
        )
    )
    source_gpus = tuple(
        draw(
            st.lists(
                st.sampled_from(_SOURCE_GPU_POOL),
                min_size=1,
                max_size=len(_SOURCE_GPU_POOL),
                unique=True,
            )
        )
    )
    workloads = tuple(
        Workload(m, n, k, f"source-{index}", "q", "test") for index, (m, n, k) in enumerate(shapes)
    )
    observations = tuple(
        ArchiveObservation(workload, config, source_gpu, draw(_LATENCIES))
        for workload in workloads
        for source_gpu in source_gpus
        for config in configs
    )
    target = Workload(
        draw(st.sampled_from(_SHAPE_DIMENSIONS)),
        draw(st.sampled_from(_SHAPE_DIMENSIONS)),
        draw(st.sampled_from(_SHAPE_DIMENSIONS)),
        "target",
        "q",
        "test",
    )
    return _Archive(
        observations=observations,
        permuted=tuple(draw(st.permutations(observations))),
        configs=configs,
        target=target,
        k=draw(st.integers(min_value=1, max_value=len(_SHAPE_DIMENSIONS))),
        temperature=draw(
            st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False)
        ),
    )


def _index(archive: _Archive, observations: tuple[ArchiveObservation, ...]) -> RetrievalIndex:
    return RetrievalIndex(observations, k=archive.k, temperature=archive.temperature)


@given(
    weighted_advantage_mean=st.floats(
        min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
    weighted_variance=st.floats(
        min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
    normalized_neighbor_distance=st.floats(
        min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
    source_gpu_agreement=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_stats_array_follows_the_declared_feature_names(
    weighted_advantage_mean: float,
    weighted_variance: float,
    normalized_neighbor_distance: float,
    source_gpu_agreement: float,
) -> None:
    stats = RetrievalStats(
        weighted_advantage_mean=weighted_advantage_mean,
        weighted_variance=weighted_variance,
        normalized_neighbor_distance=normalized_neighbor_distance,
        source_gpu_agreement=source_gpu_agreement,
    )
    features = stats.as_array()

    assert len(features) == len(RETRIEVAL_FEATURE_NAMES)
    assert all(math.isfinite(value) for value in features)
    assert features == tuple(getattr(stats, name) for name in RETRIEVAL_FEATURE_NAMES)


@given(archive=_archives())
def test_scored_features_are_finite_and_length_matched(archive: _Archive) -> None:
    index = _index(archive, archive.observations)

    for config in archive.configs:
        features = index.score(archive.target, config).as_array()
        assert len(features) == len(RETRIEVAL_FEATURE_NAMES)
        assert all(math.isfinite(value) for value in features)


@given(archive=_archives())
def test_index_is_invariant_to_observation_order(archive: _Archive) -> None:
    original = _index(archive, archive.observations)
    permuted = _index(archive, archive.permuted)

    assert permuted.configs == original.configs
    for config in archive.configs:
        # ``math.fsum`` makes every aggregate exactly rounded, so a permutation
        # must reproduce the features bit-for-bit rather than approximately.
        assert (
            permuted.score(archive.target, config).as_array()
            == original.score(archive.target, config).as_array()
        )
    assert permuted.rank(archive.target, archive.configs) == original.rank(
        archive.target, archive.configs
    )


@given(archive=_archives())
def test_neighbor_weights_decay_with_distance_from_the_nearest_shape(
    archive: _Archive,
) -> None:
    index = _index(archive, archive.observations)
    neighbors = index._neighbors(archive.target)
    distances = [neighbor.distance for neighbor in neighbors]
    weights = [neighbor.weight for neighbor in neighbors]

    assert 0 < len(neighbors) <= archive.k
    assert distances == sorted(distances)
    # Weights are shifted exponentials, normalized only where they are consumed
    # in ``score``; the nearest shape therefore carries exactly 1.0 and the
    # remainder decay into ``(0, 1]`` without summing to one.
    assert weights[0] == 1.0
    assert all(0.0 < weight <= 1.0 for weight in weights)
    assert all(later <= earlier for earlier, later in zip(weights, weights[1:], strict=False))

    for config in archive.configs:
        # ``normalized_neighbor_distance`` is a weight-average of exactly these
        # distances, so it must sit inside their range up to one rounding step.
        distance = index.score(archive.target, config).normalized_neighbor_distance
        assert distances[0] - 1e-12 <= distance <= distances[-1] + 1e-12
