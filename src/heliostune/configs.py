"""Kernel launch configurations and representative LLM projection workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """A manually selected Triton matmul launch configuration."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    group_m: int = 8

    def __post_init__(self) -> None:
        if min(self.block_m, self.block_n, self.block_k, self.group_m) <= 0:
            raise ValueError("tile sizes and group_m must be positive")
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError("num_warps must be one of 1, 2, 4, or 8")
        if self.num_stages <= 0:
            raise ValueError("num_stages must be positive")

    @property
    def key(self) -> str:
        return (
            f"m{self.block_m}n{self.block_n}k{self.block_k}"
            f"-w{self.num_warps}s{self.num_stages}g{self.group_m}"
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KernelConfig:
        return cls(**{field: int(value[field]) for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class Workload:
    """A row-major ``A[M,K] @ B[K,N]`` workload."""

    m: int
    n: int
    k: int
    family: str

    def __post_init__(self) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError("matrix dimensions must be positive")
        if not self.family:
            raise ValueError("family must not be empty")

    @property
    def key(self) -> str:
        return f"{self.family}-m{self.m}-n{self.n}-k{self.k}"

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Workload:
        return cls(
            m=int(value["m"]),
            n=int(value["n"]),
            k=int(value["k"]),
            family=str(value["family"]),
        )


DEFAULT_CONFIGS: tuple[KernelConfig, ...] = (
    KernelConfig(16, 32, 32, 4, 3),
    KernelConfig(16, 64, 32, 4, 3),
    KernelConfig(16, 128, 32, 4, 4),
    KernelConfig(32, 32, 32, 4, 3),
    KernelConfig(32, 64, 32, 4, 3),
    KernelConfig(32, 128, 32, 4, 4),
    KernelConfig(64, 32, 32, 4, 3),
    KernelConfig(64, 64, 32, 4, 3),
    KernelConfig(64, 128, 32, 8, 3),
    KernelConfig(128, 64, 32, 4, 3),
    KernelConfig(128, 128, 32, 8, 3),
    KernelConfig(64, 64, 64, 8, 4),
)


_BATCH_SIZES = (1, 8, 32, 128)
DEFAULT_WORKLOADS: tuple[Workload, ...] = tuple(
    Workload(batch, n, k, family)
    for family, n, k in (
        ("qkv", 4096, 4096),
        ("ffn-up", 11008, 4096),
        ("ffn-down", 4096, 11008),
    )
    for batch in _BATCH_SIZES
)
