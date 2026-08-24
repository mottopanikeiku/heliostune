"""Kernel launch configurations and representative LLM projection workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from heliostune.errors import SchemaError
from heliostune.validation import exact_fields, exact_int, nonblank_string


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
        for name in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "group_m"):
            exact_int(getattr(self, name), context=f"kernel config {name}", minimum=1)
        for name in ("block_m", "block_n", "block_k"):
            size = getattr(self, name)
            if size & (size - 1):
                raise SchemaError(f"{name} must be a power of two")
        if self.num_warps not in {1, 2, 4, 8}:
            raise SchemaError("num_warps must be one of 1, 2, 4, or 8")

    @property
    def key(self) -> str:
        return (
            f"m{self.block_m}n{self.block_n}k{self.block_k}"
            f"-w{self.num_warps}s{self.num_stages}g{self.group_m}"
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> KernelConfig:
        fields = tuple(cls.__dataclass_fields__)
        data = exact_fields(value, required=fields, context="kernel config")
        return cls(
            **{
                field: exact_int(data[field], context=f"kernel config {field}", minimum=1)
                for field in fields
            }
        )


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
        for name in ("m", "n", "k"):
            exact_int(getattr(self, name), context=f"workload {name}", minimum=1)
        for name in ("model", "projection", "regime"):
            nonblank_string(getattr(self, name), context=f"workload {name}")

    @property
    def key(self) -> str:
        return f"{self.model}-{self.projection}-{self.regime}-m{self.m}-n{self.n}-k{self.k}"

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Workload:
        data = exact_fields(
            value,
            required=("m", "n", "k", "model", "projection", "regime"),
            context="workload",
        )
        return cls(
            m=exact_int(data["m"], context="workload m", minimum=1),
            n=exact_int(data["n"], context="workload n", minimum=1),
            k=exact_int(data["k"], context="workload k", minimum=1),
            model=nonblank_string(data["model"], context="workload model"),
            projection=nonblank_string(data["projection"], context="workload projection"),
            regime=nonblank_string(data["regime"], context="workload regime"),
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

    def __post_init__(self) -> None:
        nonblank_string(self.name, context="model spec name")
        nonblank_string(self.config_url, context="model spec config_url")
        for name in (
            "hidden_size",
            "intermediate_size",
            "attention_heads",
            "key_value_heads",
        ):
            exact_int(getattr(self, name), context=f"model spec {name}", minimum=1)
        if self.hidden_size % self.attention_heads:
            raise SchemaError("hidden_size must be divisible by attention_heads")
        if self.key_value_heads > self.attention_heads:
            raise SchemaError("key_value_heads must not exceed attention_heads")

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
