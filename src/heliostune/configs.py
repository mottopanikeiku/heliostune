"""Kernel launch configurations and representative LLM projection workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from heliostune.errors import SchemaError
from heliostune.validation import exact_bool, exact_fields, exact_int, nonblank_string


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
        "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/27d67f1b5f57dc0953326b2601d68371d40ea8da/config.json",
    ),
    ModelSpec(
        "qwen2.5-7b",
        3584,
        18944,
        28,
        4,
        "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/d149729398750b98c0af14eb82c78cfe92750796/config.json",
    ),
    ModelSpec(
        "phi-3-mini",
        3072,
        8192,
        32,
        32,
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/f39ac1d28e925b323eae81227eaba4464caced4e/config.json",
    ),
    ModelSpec(
        "granite-3.1-8b",
        4096,
        12800,
        32,
        8,
        "https://huggingface.co/ibm-granite/granite-3.1-8b-instruct/resolve/4009206d5fc95d2e65a7b7633e159d6e97e25d35/config.json",
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


TRITON_TUTORIAL_COMMIT = "a77e7c793abc0d0c923a9afb275058e2fe57a198"
TRITON_TUTORIAL_CONFIG_PATH = "python/tutorials/03-matrix-multiplication.py"
PARHELION_V3_OFFICIAL_CONFIGS: tuple[KernelConfig, ...] = (
    KernelConfig(128, 256, 64, 8, 3, 8),
    KernelConfig(64, 256, 32, 4, 4, 8),
    KernelConfig(128, 128, 32, 4, 4, 8),
    KernelConfig(128, 64, 32, 4, 4, 8),
    KernelConfig(64, 128, 32, 4, 4, 8),
    KernelConfig(128, 32, 32, 4, 4, 8),
    KernelConfig(64, 32, 32, 2, 5, 8),
    KernelConfig(32, 64, 32, 2, 5, 8),
    KernelConfig(128, 256, 128, 8, 3, 8),
    KernelConfig(256, 128, 128, 8, 3, 8),
    KernelConfig(256, 64, 128, 4, 4, 8),
    KernelConfig(64, 256, 128, 4, 4, 8),
    KernelConfig(128, 128, 128, 4, 4, 8),
    KernelConfig(128, 64, 64, 4, 4, 8),
    KernelConfig(64, 128, 64, 4, 4, 8),
    KernelConfig(128, 32, 64, 4, 4, 8),
)
PARHELION_V3_OFFICIAL_CONFIG_KEYS = frozenset(
    config.key for config in PARHELION_V3_OFFICIAL_CONFIGS
)
PARHELION_V3_CANDIDATE_CONFIGS: tuple[KernelConfig, ...] = tuple(
    sorted(
        {*DEFAULT_CONFIGS, *PARHELION_V3_OFFICIAL_CONFIGS},
        key=lambda config: config.key,
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


#: A tensor descriptor needs 16-byte aligned leading strides, so every FP16 block
#: dimension that sits innermost in a descriptor must be a multiple of eight
#: elements, as must the row length of the tensor the descriptor is built over.
TENSOR_DESCRIPTOR_ALIGNMENT = 8

#: ``tl.dot`` requires every matrix-instruction dimension to be at least sixteen.
TENSOR_CORE_MIN_DOT_DIMENSION = 16

#: Largest broadcast product the skinny kernel may materialise, counted in
#: float32 elements. ``a[:, :, None] * b[None, :, :]`` is one live
#: ``BLOCK_M x BLOCK_K x BLOCK_N`` register tensor, so 8192 elements is 64 floats
#: per thread across four warps and leaves room for the operands themselves.
SKINNY_GEMV_PRODUCT_LIMIT = 8192


@dataclass(frozen=True, slots=True)
class HopperGemmConfig:
    """A launch configuration for the persistent tensor-descriptor matmul.

    A subtiled epilogue needs a flattened persistent loop and warp specialisation
    forbids one on Hopper, so :data:`HOPPER_GEMM_CONFIGS` never sets both flags.
    The pair is not rejected here: a later architecture can flatten a
    warp-specialised loop, so the launcher decides against the live device.
    """

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    group_m: int = 8
    epilogue_subtile: bool = False
    warp_specialize: bool = False

    def __post_init__(self) -> None:
        for name in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "group_m"):
            exact_int(getattr(self, name), context=f"hopper gemm config {name}", minimum=1)
        for name in ("epilogue_subtile", "warp_specialize"):
            exact_bool(getattr(self, name), context=f"hopper gemm config {name}")
        for name in ("block_m", "block_n", "block_k"):
            size = getattr(self, name)
            if size & (size - 1):
                raise SchemaError(f"{name} must be a power of two")
        if self.num_warps not in {1, 2, 4, 8}:
            raise SchemaError("num_warps must be one of 1, 2, 4, or 8")
        for name in ("block_m", "block_n", "block_k"):
            if getattr(self, name) < TENSOR_CORE_MIN_DOT_DIMENSION:
                raise SchemaError(f"{name} must be at least {TENSOR_CORE_MIN_DOT_DIMENSION}")
        for name in ("block_n", "block_k"):
            if getattr(self, name) % TENSOR_DESCRIPTOR_ALIGNMENT:
                raise SchemaError(
                    f"{name} must be a multiple of {TENSOR_DESCRIPTOR_ALIGNMENT} elements"
                )
        if self.epilogue_subtile and (self.block_n // 2) % TENSOR_DESCRIPTOR_ALIGNMENT:
            raise SchemaError(
                "a subtiled epilogue stores block_n // 2 columns, which must be a "
                f"multiple of {TENSOR_DESCRIPTOR_ALIGNMENT} elements"
            )

    @property
    def key(self) -> str:
        return (
            f"hopper-m{self.block_m}n{self.block_n}k{self.block_k}"
            f"-w{self.num_warps}s{self.num_stages}g{self.group_m}"
            f"-sub{int(self.epilogue_subtile)}-ws{int(self.warp_specialize)}"
        )

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> HopperGemmConfig:
        fields = (
            "block_m",
            "block_n",
            "block_k",
            "num_warps",
            "num_stages",
            "group_m",
            "epilogue_subtile",
            "warp_specialize",
        )
        data = exact_fields(value, required=fields, context="hopper gemm config")
        return cls(
            block_m=exact_int(data["block_m"], context="hopper gemm config block_m", minimum=1),
            block_n=exact_int(data["block_n"], context="hopper gemm config block_n", minimum=1),
            block_k=exact_int(data["block_k"], context="hopper gemm config block_k", minimum=1),
            num_warps=exact_int(
                data["num_warps"], context="hopper gemm config num_warps", minimum=1
            ),
            num_stages=exact_int(
                data["num_stages"], context="hopper gemm config num_stages", minimum=1
            ),
            group_m=exact_int(data["group_m"], context="hopper gemm config group_m", minimum=1),
            epilogue_subtile=exact_bool(
                data["epilogue_subtile"],
                context="hopper gemm config epilogue_subtile",
            ),
            warp_specialize=exact_bool(
                data["warp_specialize"],
                context="hopper gemm config warp_specialize",
            ),
        )


@dataclass(frozen=True, slots=True)
class SkinnyGemvConfig:
    """A launch configuration for the split-``K`` skinny-``M`` GEMV candidate."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    split_k: int = 1

    def __post_init__(self) -> None:
        for name in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "split_k"):
            exact_int(getattr(self, name), context=f"skinny gemv config {name}", minimum=1)
        for name in ("block_m", "block_n", "block_k"):
            size = getattr(self, name)
            if size & (size - 1):
                raise SchemaError(f"{name} must be a power of two")
        if self.num_warps not in {1, 2, 4, 8}:
            raise SchemaError("num_warps must be one of 1, 2, 4, or 8")
        product = self.block_m * self.block_n * self.block_k
        if product > SKINNY_GEMV_PRODUCT_LIMIT:
            raise SchemaError(
                "block_m * block_n * block_k must not exceed "
                f"{SKINNY_GEMV_PRODUCT_LIMIT}, got {product}"
            )

    @property
    def key(self) -> str:
        return (
            f"gemv-m{self.block_m}n{self.block_n}k{self.block_k}"
            f"-w{self.num_warps}s{self.num_stages}-split{self.split_k}"
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> SkinnyGemvConfig:
        fields = tuple(cls.__dataclass_fields__)
        data = exact_fields(value, required=fields, context="skinny gemv config")
        return cls(
            **{
                field: exact_int(
                    data[field],
                    context=f"skinny gemv config {field}",
                    minimum=1,
                )
                for field in fields
            }
        )


#: ``(block_m, block_n, block_k, num_warps, num_stages)`` tiles for the persistent
#: kernel, modelled on the Hopper rows of the Triton tutorial autotune grid at
#: :data:`TRITON_TUTORIAL_CONFIG_PATH`. Every tile keeps its staged FP16 operands
#: inside Hopper's 228 KiB of shared memory per multiprocessor.
_HOPPER_TILES: tuple[tuple[int, int, int, int, int], ...] = (
    (64, 64, 64, 4, 4),
    (64, 128, 64, 4, 4),
    (128, 64, 64, 4, 4),
    (128, 64, 128, 4, 3),
    (128, 128, 64, 8, 3),
    (128, 128, 128, 8, 3),
    (128, 256, 64, 8, 3),
    (256, 64, 64, 8, 3),
    (256, 128, 64, 8, 3),
)

#: A subtiled epilogue drains the accumulator in two descriptor stores instead of
#: one, which only pays for a tile wide enough to make a single store the burst.
HOPPER_SUBTILE_MIN_BLOCK_N = 128


def _hopper_flag_variants(block_n: int) -> tuple[tuple[bool, bool], ...]:
    """Return the ``(epilogue_subtile, warp_specialize)`` variants for one tile."""
    plain_and_specialised = ((False, False), (False, True))
    if block_n < HOPPER_SUBTILE_MIN_BLOCK_N:
        return plain_and_specialised
    return (*plain_and_specialised, (True, False))


HOPPER_GEMM_CONFIGS: tuple[HopperGemmConfig, ...] = tuple(
    sorted(
        (
            HopperGemmConfig(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                epilogue_subtile=epilogue_subtile,
                warp_specialize=warp_specialize,
            )
            for block_m, block_n, block_k, num_warps, num_stages in _HOPPER_TILES
            for epilogue_subtile, warp_specialize in _hopper_flag_variants(block_n)
        ),
        key=lambda config: config.key,
    )
)


#: Eight warps once the broadcast product reaches half its limit, so no thread
#: ever holds more than thirty-two float32 elements of it.
SKINNY_GEMV_WIDE_PRODUCT = SKINNY_GEMV_PRODUCT_LIMIT // 2

#: The reduction is a plain load-and-accumulate chain, which the three-stage
#: pipeline of the frozen kernel already covers.
SKINNY_GEMV_STAGES = 3

#: A decode projection is bandwidth bound, so the split factor -- the number of
#: programs streaming disjoint ``K`` spans of ``B`` -- is the knob that matters.
SKINNY_GEMV_SPLITS = (1, 4, 16)


def _skinny_warps(block_m: int, block_n: int, block_k: int) -> int:
    return 8 if block_m * block_n * block_k >= SKINNY_GEMV_WIDE_PRODUCT else 4


#: Every power-of-two tile whose broadcast product fits the register budget.
_SKINNY_TILES: tuple[tuple[int, int, int], ...] = tuple(
    (block_m, block_n, block_k)
    for block_m in (1, 2, 4, 8)
    for block_n in (32, 64, 128, 256)
    for block_k in (32, 64)
    if block_m * block_n * block_k <= SKINNY_GEMV_PRODUCT_LIMIT
)

SKINNY_GEMV_CONFIGS: tuple[SkinnyGemvConfig, ...] = tuple(
    sorted(
        (
            SkinnyGemvConfig(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=_skinny_warps(block_m, block_n, block_k),
                num_stages=SKINNY_GEMV_STAGES,
                split_k=split_k,
            )
            for block_m, block_n, block_k in _SKINNY_TILES
            for split_k in SKINNY_GEMV_SPLITS
        ),
        key=lambda config: config.key,
    )
)
