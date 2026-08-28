"""CPU-safe candidate and workload specifications for Hopper validation."""

from __future__ import annotations

from collections.abc import Sequence

from heliostune.configs import DEFAULT_WORKLOADS, Workload
from heliostune.errors import ProtocolError

#: Largest ``M`` the skinny kernel is a plausible candidate for. Above it the
#: broadcast-reduce inner product loses to a tensor-core tile by construction.
SKINNY_M_LIMIT = 8

#: Shapes the published corpus never produces. Every published ``N`` and ``K``
#: is a multiple of 128, so no frozen workload exercises a partial ``N`` tile, a
#: partial ``K`` tile, or a split-``K`` fan-out wider than the tile count.
#: ``N`` and ``K`` stay multiples of eight because a tensor descriptor needs
#: 16-byte aligned leading strides.
EDGE_WORKLOADS: tuple[Workload, ...] = (
    Workload(m=1, n=8, k=8, model="synthetic", projection="descriptor-minimum", regime="edge"),
    Workload(m=7, n=136, k=264, model="synthetic", projection="partial-tiles", regime="edge"),
    Workload(m=131, n=1032, k=520, model="synthetic", projection="partial-rows", regime="edge"),
)


def validation_workloads(
    workloads: Sequence[Workload] = DEFAULT_WORKLOADS,
    *,
    edges: Sequence[Workload] = EDGE_WORKLOADS,
) -> tuple[Workload, ...]:
    """Return a small representative sample of ``workloads`` plus the edge shapes.

    The sample keeps the cheapest and the most expensive published workload of
    every token regime, so all six ``M`` values and both ends of the ``N * K``
    range survive while the gate stays affordable enough to run before any
    timing happens.
    """
    if not workloads:
        raise ProtocolError("validation workload manifest must not be empty")
    by_regime: dict[str, list[Workload]] = {}
    for workload in workloads:
        if type(workload) is not Workload:
            raise ProtocolError("validation workload manifest must contain only Workload values")
        by_regime.setdefault(workload.regime, []).append(workload)

    sample: dict[str, Workload] = {}
    for regime in sorted(by_regime):
        ordered = sorted(by_regime[regime], key=lambda item: (item.n * item.k, item.key))
        for workload in (ordered[0], ordered[-1]):
            sample[workload.key] = workload
    for workload in edges:
        if type(workload) is not Workload:
            raise ProtocolError("validation edge manifest must contain only Workload values")
        sample[workload.key] = workload
    return tuple(sample.values())
