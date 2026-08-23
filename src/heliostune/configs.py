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
    """A row-major ``A[M,K] @ B[K,N]`` workload from a named model projection."""

    m: int
    n: int
    k: int
    model: str
    projection: str
    regime: str

    def __post_init__(self) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError("matrix dimensions must be positive")
        if not self.model or not self.projection or not self.regime:
            raise ValueError("model, projection, and regime must not be empty")

    @property
    def key(self) -> str:
        return f"{self.model}-{self.projection}-{self.regime}-m{self.m}-n{self.n}-k{self.k}"

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
            model=str(value["model"]),
            projection=str(value["projection"]),
            regime=str(value["regime"]),
        )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Public model dimensions used to derive the benchmark corpus."""

    name: str
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    key_value_heads: int
    config_url: str

    @property
    def key_value_size(self) -> int:
        return self.hidden_size // self.attention_heads * self.key_value_heads


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "mistral-7b",
        4096,
        14336,
        32,
        8,
        "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json",
    ),
    ModelSpec(
        "qwen2.5-7b",
        3584,
        18944,
        28,
        4,
        "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/config.json",
    ),
    ModelSpec(
        "phi-3-mini",
        3072,
        8192,
        32,
        32,
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/config.json",
    ),
    ModelSpec(
        "granite-3.1-8b",
        4096,
        12800,
        32,
        8,
        "https://huggingface.co/ibm-granite/granite-3.1-8b-instruct/resolve/main/config.json",
    ),
)


_TILES = tuple((block_m, block_n) for block_m in (16, 32, 64, 128) for block_n in (32, 64, 128))


def _tile_warps(block_m: int, block_n: int) -> int:
    return 8 if block_m * block_n >= 8192 else 4


DEFAULT_CONFIGS: tuple[KernelConfig, ...] = (
    tuple(
        KernelConfig(block_m, block_n, 32, _tile_warps(block_m, block_n), 3, 8)
        for block_m, block_n in _TILES
    )
    + tuple(
        KernelConfig(block_m, block_n, 64, _tile_warps(block_m, block_n), 2, 8)
        for block_m, block_n in _TILES
    )
    + tuple(
        KernelConfig(block_m, block_n, 32, _tile_warps(block_m, block_n), 4, 4)
        for block_m, block_n in _TILES
    )
)


_TOKEN_REGIMES: tuple[tuple[str, int], ...] = (
    ("decode-1", 1),
    ("decode-7", 7),
    ("mixed-31", 31),
    ("mixed-96", 96),
    ("prefill-257", 257),
    ("prefill-1024", 1024),
)


def _projection_shapes(model: ModelSpec) -> tuple[tuple[str, int, int], ...]:
    return (
        ("attention-qkv", model.hidden_size + 2 * model.key_value_size, model.hidden_size),
        ("attention-out", model.hidden_size, model.hidden_size),
        ("ffn-up", model.intermediate_size, model.hidden_size),
        ("ffn-down", model.hidden_size, model.intermediate_size),
    )


DEFAULT_WORKLOADS: tuple[Workload, ...] = tuple(
    Workload(
        m=tokens,
        n=n,
        k=k,
        model=model.name,
        projection=projection,
        regime=regime,
    )
    for model in MODEL_SPECS
    for projection, n, k in _projection_shapes(model)
    for regime, tokens in _TOKEN_REGIMES
)
