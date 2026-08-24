"""Public multi-source replay facade."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from heliostune.multisource_engine import run_multisource
from heliostune.schema import Measurement


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
    """Compare Parhelion and independently tuned baselines in grouped replay."""
    return run_multisource(
        measurements,
        source_gpus=source_gpus,
        target_gpu=target_gpu,
        max_budget=max_budget,
        seeds=seeds,
        k=k,
        temperature=temperature,
        transfer_strength=transfer_strength,
        retrieval_k=retrieval_k,
        retrieval_temperature=retrieval_temperature,
        pooled_transfer_strength=pooled_transfer_strength,
        primary_comparator=primary_comparator,
        protocol_role=protocol_role,
    )


__all__ = ["compare_multisource"]
