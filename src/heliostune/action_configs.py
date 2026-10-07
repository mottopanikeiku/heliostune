"""Frozen action expansion; enumeration never imports Torch or Triton.

The six ordinary/persistent candidates apply to every workload. Four additional
FP32-workspace tensor-core split-K candidates apply only when M <= 31. Candidate
IDs include the family, independently of the historical configuration IDs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from heliostune.configs import HopperGemmConfig, KernelConfig, Workload
from heliostune.errors import SchemaError
from heliostune.validation import exact_int

SPLIT_K_M_LIMIT = 31


@dataclass(frozen=True, slots=True)
class SplitKConfig:
    """Tensor-core partial tiles, followed by one FP32 Triton reduction.

    Each split receives a contiguous, block-K-aligned interval. The launcher
    allocates an unpadded [split_k, M, N] FP32 workspace and an FP16 output on
    every call. The fixed reduction uses 256 output elements and four warps.
    """

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    split_k: int

    def __post_init__(self) -> None:
        for name in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "split_k"):
            exact_int(getattr(self, name), context=f"split-K config {name}", minimum=1)
        for name in ("block_m", "block_n", "block_k", "split_k"):
            size = getattr(self, name)
            if size & (size - 1):
                raise SchemaError(f"{name} must be a power of two")
        for name in ("block_m", "block_n", "block_k"):
            if getattr(self, name) < 16:
                raise SchemaError(f"{name} must be at least 16")
        if self.num_warps not in {1, 2, 4, 8}:
            raise SchemaError("num_warps must be one of 1, 2, 4, or 8")
        if self.split_k < 2:
            raise SchemaError("split_k must be at least 2")

    @property
    def key(self) -> str:
        return (
            f"m{self.block_m}n{self.block_n}k{self.block_k}"
            f"-w{self.num_warps}s{self.num_stages}-split{self.split_k}"
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    """An immutable launch configuration with a family-qualified identity."""

    config: KernelConfig | HopperGemmConfig | SplitKConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, KernelConfig | HopperGemmConfig | SplitKConfig):
            raise TypeError("candidate config must be a tile, persistent, or split-K config")

    @property
    def family(self) -> Literal["tile", "persistent", "split_k"]:
        if isinstance(self.config, KernelConfig):
            return "tile"
        if isinstance(self.config, HopperGemmConfig):
            return "persistent"
        return "split_k"

    @property
    def key(self) -> str:
        return f"{self.family}:{self.config.key}"

    def to_dict(self) -> dict[str, int | bool]:
        return dict(self.config.to_dict())


NON_SPLIT_CANDIDATES: tuple[ActionCandidate, ...] = tuple(
    sorted(
        (
            ActionCandidate(KernelConfig(128, 128, 64, 4, 5)),
            ActionCandidate(KernelConfig(128, 256, 64, 8, 4)),
            ActionCandidate(KernelConfig(256, 128, 64, 8, 4)),
            ActionCandidate(KernelConfig(128, 128, 128, 8, 3)),
            ActionCandidate(HopperGemmConfig(128, 128, 64, 4, 4)),
            ActionCandidate(HopperGemmConfig(128, 256, 64, 8, 3)),
        ),
        key=lambda candidate: candidate.key,
    )
)
SPLIT_K_CANDIDATES: tuple[ActionCandidate, ...] = tuple(
    sorted(
        (
            ActionCandidate(SplitKConfig(16, 64, 64, 4, 3, 4)),
            ActionCandidate(SplitKConfig(16, 64, 64, 4, 3, 16)),
            ActionCandidate(SplitKConfig(16, 128, 64, 4, 3, 4)),
            ActionCandidate(SplitKConfig(16, 128, 64, 4, 3, 16)),
        ),
        key=lambda candidate: candidate.key,
    )
)
ALL_CANDIDATES: tuple[ActionCandidate, ...] = tuple(
    sorted((*NON_SPLIT_CANDIDATES, *SPLIT_K_CANDIDATES), key=lambda candidate: candidate.key)
)


def candidates_for(workload: Workload) -> tuple[ActionCandidate, ...]:
    """Return the predeclared candidates; no measurements influence enumeration."""
    return ALL_CANDIDATES if workload.m <= SPLIT_K_M_LIMIT else NON_SPLIT_CANDIDATES
