"""Strict schemas and filesystem verification for narrow plugin/suite reference templates."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, cast

from heliostune.artifacts import read_json, strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.validation import exact_bool, exact_fields, exact_int, finite_float, nonblank_string

DType = Literal["fp32", "tf32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "int8", "int4", "uint4"]
Domain = Literal[
    "dense_gemm",
    "fused_mlp",
    "rmsnorm_residual",
    "attention",
    "kv_cache",
    "moe",
    "quantized_linear",
]
CapabilityState = Literal["unprobed", "available", "unavailable"]

DTYPE_VOCABULARY = (
    "fp32",
    "tf32",
    "fp16",
    "bf16",
    "fp8_e4m3fn",
    "fp8_e5m2",
    "int8",
    "int4",
    "uint4",
)
DOMAIN_VOCABULARY = (
    "dense_gemm",
    "fused_mlp",
    "rmsnorm_residual",
    "attention",
    "kv_cache",
    "moe",
    "quantized_linear",
)
EXECUTABLE_TEMPLATE_IDS = ("gated_mlp_epilogue.v1", "residual_rmsnorm.v1")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


def _enum(value: object, allowed: tuple[str, ...], context: str) -> str:
    result = nonblank_string(value, context=context)
    if result not in allowed:
        raise SchemaError(f"unknown {context} {result!r}")
    return result


def _array(value: object, context: str) -> list[object]:
    if type(value) is not list:
        raise SchemaError(f"{context} must be an array")
    return cast(list[object], value)


def _strings(value: object, context: str, *, unique: bool = False) -> tuple[str, ...]:
    result = tuple(
        nonblank_string(item, context=f"{context} item") for item in _array(value, context)
    )
    if unique and len(set(result)) != len(result):
        raise SchemaError(f"{context} must not contain duplicates")
    return result


def _digest(value: object, context: str) -> str:
    result = nonblank_string(value, context=context)
    if _DIGEST_RE.fullmatch(result) is None:
        raise SchemaError(f"{context} must be a lowercase 64-hex SHA-256 digest")
    return result


def _nullable_string(value: object, context: str) -> str | None:
    return None if value is None else nonblank_string(value, context=context)


def _nullable_int(value: object, context: str, minimum: int = 0) -> int | None:
    return None if value is None else exact_int(value, context=context, minimum=minimum)


def _number(value: object, context: str, *, positive: bool = False) -> float:
    return finite_float(
        value, context=context, strictly_positive=positive, minimum=None if positive else 0
    )


class _HasId(Protocol):
    id: str


def _unique_ids(items: tuple[_HasId, ...], context: str) -> None:
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise SchemaError(f"{context} IDs must be unique")


@dataclass(frozen=True, slots=True)
class PackingSpec:
    bits: int
    axis: str
    order: Literal["low_nibble_first", "high_nibble_first"]

    def __post_init__(self) -> None:
        if exact_int(self.bits, context="packing bits", minimum=1) != 4:
            raise SchemaError("packing bits must be 4")
        nonblank_string(self.axis, context="packing axis")
        _enum(self.order, ("low_nibble_first", "high_nibble_first"), "packing order")

    @classmethod
    def from_dict(cls, value: object) -> PackingSpec:
        data = exact_fields(value, required=("bits", "axis", "order"), context="packing")
        return cls(
            bits=exact_int(data["bits"], context="packing bits", minimum=1),
            axis=nonblank_string(data["axis"], context="packing axis"),
            order=cast(
                Literal["low_nibble_first", "high_nibble_first"],
                _enum(data["order"], ("low_nibble_first", "high_nibble_first"), "packing order"),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {"bits": self.bits, "axis": self.axis, "order": self.order}


@dataclass(frozen=True, slots=True)
class DTypeSpec:
    name: DType
    usage: Literal["input", "storage", "compute", "output"]
    packing: PackingSpec | None

    def __post_init__(self) -> None:
        _enum(self.name, DTYPE_VOCABULARY, "dtype name")
        usage = _enum(self.usage, ("input", "storage", "compute", "output"), "dtype usage")
        if self.name in {"int4", "uint4"}:
            if usage != "storage" or self.packing is None:
                raise SchemaError("int4/uint4 are storage-only and require packing metadata")
        elif self.packing is not None:
            raise SchemaError("packing metadata is only valid for int4/uint4 storage")
        if self.name == "tf32" and usage != "compute":
            raise SchemaError("tf32 is compute-only")

    @classmethod
    def from_dict(cls, value: object) -> DTypeSpec:
        data = exact_fields(value, required=("name", "usage", "packing"), context="dtype spec")
        return cls(
            name=cast(DType, _enum(data["name"], DTYPE_VOCABULARY, "dtype name")),
            usage=cast(
                Literal["input", "storage", "compute", "output"],
                _enum(data["usage"], ("input", "storage", "compute", "output"), "dtype usage"),
            ),
            packing=None if data["packing"] is None else PackingSpec.from_dict(data["packing"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "usage": self.usage,
            "packing": None if self.packing is None else self.packing.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    scheme: Literal["per_tensor", "per_channel", "per_group"]
    scale_dtype: Literal["fp32", "fp16", "bf16"]
    scale_layout: Literal["scalar", "channel", "group"]
    calibration: Literal["static", "dynamic"]
    group_size: int | None

    def __post_init__(self) -> None:
        scheme = _enum(
            self.scheme, ("per_tensor", "per_channel", "per_group"), "quantization scheme"
        )
        _enum(self.scale_dtype, ("fp32", "fp16", "bf16"), "quantization scale_dtype")
        layout = _enum(
            self.scale_layout, ("scalar", "channel", "group"), "quantization scale_layout"
        )
        _enum(self.calibration, ("static", "dynamic"), "quantization calibration")
        group = _nullable_int(self.group_size, "quantization group_size", 1)
        expected_layout = {"per_tensor": "scalar", "per_channel": "channel", "per_group": "group"}[
            scheme
        ]
        if layout != expected_layout:
            raise SchemaError(f"{scheme} quantization requires {expected_layout!r} scale_layout")
        if (scheme == "per_group") != (group is not None):
            raise SchemaError("group_size is required exactly for per_group quantization")

    @classmethod
    def from_dict(cls, value: object) -> QuantizationSpec:
        data = exact_fields(
            value,
            required=("scheme", "scale_dtype", "scale_layout", "calibration", "group_size"),
            context="quantization",
        )
        return cls(
            scheme=cast(
                Literal["per_tensor", "per_channel", "per_group"],
                _enum(
                    data["scheme"],
                    ("per_tensor", "per_channel", "per_group"),
                    "quantization scheme",
                ),
            ),
            scale_dtype=cast(
                Literal["fp32", "fp16", "bf16"],
                _enum(data["scale_dtype"], ("fp32", "fp16", "bf16"), "quantization scale_dtype"),
            ),
            scale_layout=cast(
                Literal["scalar", "channel", "group"],
                _enum(
                    data["scale_layout"],
                    ("scalar", "channel", "group"),
                    "quantization scale_layout",
                ),
            ),
            calibration=cast(
                Literal["static", "dynamic"],
                _enum(data["calibration"], ("static", "dynamic"), "quantization calibration"),
            ),
            group_size=_nullable_int(data["group_size"], "quantization group_size", 1),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "scale_dtype": self.scale_dtype,
            "scale_layout": self.scale_layout,
            "calibration": self.calibration,
            "group_size": self.group_size,
        }


@dataclass(frozen=True, slots=True)
class NumericContract:
    id: str
    input: DTypeSpec
    storage: DTypeSpec
    accumulation: DTypeSpec
    output: DTypeSpec
    tf32: bool
    quantization: QuantizationSpec | None

    def __post_init__(self) -> None:
        nonblank_string(self.id, context="numeric contract id")
        expected = (
            (self.input, "input"),
            (self.storage, "storage"),
            (self.accumulation, "compute"),
            (self.output, "output"),
        )
        if any(spec.usage != usage for spec, usage in expected):
            raise SchemaError("numeric contract dtype usage does not match its field")
        enabled = exact_bool(self.tf32, context="numeric contract tf32")
        if enabled != (self.accumulation.name == "tf32"):
            raise SchemaError("tf32 must be true exactly when accumulation dtype is tf32")
        names = {spec.name for spec, _ in expected}
        advanced = bool(names & {"fp8_e4m3fn", "fp8_e5m2", "int8", "int4", "uint4"})
        if advanced and self.quantization is None:
            raise SchemaError(
                "FP8 and integer numeric contracts require scale/calibration/layout quantization metadata"
            )
        if not advanced and self.quantization is not None:
            raise SchemaError("quantization metadata requires an FP8 or integer dtype")

    @classmethod
    def from_dict(cls, value: object) -> NumericContract:
        data = exact_fields(
            value,
            required=("id", "input", "storage", "accumulation", "output", "tf32", "quantization"),
            context="numeric contract",
        )
        return cls(
            id=nonblank_string(data["id"], context="numeric contract id"),
            input=DTypeSpec.from_dict(data["input"]),
            storage=DTypeSpec.from_dict(data["storage"]),
            accumulation=DTypeSpec.from_dict(data["accumulation"]),
            output=DTypeSpec.from_dict(data["output"]),
            tf32=exact_bool(data["tf32"], context="numeric contract tf32"),
            quantization=None
            if data["quantization"] is None
            else QuantizationSpec.from_dict(data["quantization"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "input": self.input.to_dict(),
            "storage": self.storage.to_dict(),
            "accumulation": self.accumulation.to_dict(),
            "output": self.output.to_dict(),
            "tf32": self.tf32,
            "quantization": None if self.quantization is None else self.quantization.to_dict(),
        }

    @property
    def is_initially_executable(self) -> bool:
        return (
            self.input.name in {"fp16", "bf16"}
            and self.storage.name in {"fp16", "bf16"}
            and self.accumulation.name == "fp32"
            and self.output.name in {"fp16", "bf16", "fp32"}
            and not self.tf32
            and self.quantization is None
        )


@dataclass(frozen=True, slots=True)
class TensorSpec:
    id: str
    role: Literal["input", "output", "parameter", "intermediate"]
    shape: tuple[str, ...]
    storage_dtype: DType
    logical_dtype: DType
    layout: Literal["row_major", "column_major", "strided", "packed"]
    contiguous: bool
    alignment: int
    quantization: QuantizationSpec | None
    packing: PackingSpec | None

    def __post_init__(self) -> None:
        nonblank_string(self.id, context="tensor id")
        _enum(self.role, ("input", "output", "parameter", "intermediate"), "tensor role")
        if not self.shape or len(set(self.shape)) != len(self.shape):
            raise SchemaError("tensor shape must contain unique dimension names")
        for dim in self.shape:
            nonblank_string(dim, context="tensor shape dimension")
        storage = _enum(self.storage_dtype, DTYPE_VOCABULARY, "tensor storage_dtype")
        logical = _enum(self.logical_dtype, DTYPE_VOCABULARY, "tensor logical_dtype")
        layout = _enum(
            self.layout, ("row_major", "column_major", "strided", "packed"), "tensor layout"
        )
        contiguous = exact_bool(self.contiguous, context="tensor contiguous")
        alignment = exact_int(self.alignment, context="tensor alignment", minimum=1)
        if alignment & (alignment - 1):
            raise SchemaError("tensor alignment must be a power of two")
        if layout == "strided" and contiguous:
            raise SchemaError("strided tensor must not claim contiguous storage")
        if "tf32" in {storage, logical}:
            raise SchemaError("tf32 is compute-only and cannot be a tensor dtype")
        if logical in {"int4", "uint4"}:
            raise SchemaError("int4/uint4 are storage-only tensor dtypes")
        if layout == "packed" and storage not in {"int4", "uint4"}:
            raise SchemaError("packed layout requires int4/uint4 storage")
        if storage in {"int4", "uint4"} and (
            layout != "packed" or self.quantization is None or self.packing is None
        ):
            raise SchemaError(
                "int4/uint4 tensor storage requires packed layout, packing metadata, and quantization metadata"
            )
        if storage not in {"int4", "uint4"} and self.packing is not None:
            raise SchemaError("tensor packing metadata is only valid for int4/uint4 storage")
        if self.packing is not None and self.packing.axis not in self.shape:
            raise SchemaError("tensor packing axis must name a tensor shape dimension")
        advanced = {"fp8_e4m3fn", "fp8_e5m2", "int8", "int4", "uint4"}
        if {storage, logical} & advanced and self.quantization is None:
            raise SchemaError(
                "FP8 and integer tensors require scale/calibration/layout quantization metadata"
            )
        if not ({storage, logical} & advanced) and self.quantization is not None:
            raise SchemaError("tensor quantization metadata requires an FP8 or integer dtype")

    @classmethod
    def from_dict(cls, value: object) -> TensorSpec:
        data = exact_fields(
            value,
            required=(
                "id",
                "role",
                "shape",
                "storage_dtype",
                "logical_dtype",
                "layout",
                "contiguous",
                "alignment",
                "quantization",
                "packing",
            ),
            context="tensor",
        )
        return cls(
            id=nonblank_string(data["id"], context="tensor id"),
            role=cast(
                Literal["input", "output", "parameter", "intermediate"],
                _enum(
                    data["role"], ("input", "output", "parameter", "intermediate"), "tensor role"
                ),
            ),
            shape=_strings(data["shape"], "tensor shape", unique=True),
            storage_dtype=cast(
                DType, _enum(data["storage_dtype"], DTYPE_VOCABULARY, "tensor storage_dtype")
            ),
            logical_dtype=cast(
                DType, _enum(data["logical_dtype"], DTYPE_VOCABULARY, "tensor logical_dtype")
            ),
            layout=cast(
                Literal["row_major", "column_major", "strided", "packed"],
                _enum(
                    data["layout"],
                    ("row_major", "column_major", "strided", "packed"),
                    "tensor layout",
                ),
            ),
            contiguous=exact_bool(data["contiguous"], context="tensor contiguous"),
            alignment=exact_int(data["alignment"], context="tensor alignment", minimum=1),
            quantization=None
            if data["quantization"] is None
            else QuantizationSpec.from_dict(data["quantization"]),
            packing=None if data["packing"] is None else PackingSpec.from_dict(data["packing"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "role": self.role,
            "shape": list(self.shape),
            "storage_dtype": self.storage_dtype,
            "logical_dtype": self.logical_dtype,
            "layout": self.layout,
            "contiguous": self.contiguous,
            "alignment": self.alignment,
            "quantization": None if self.quantization is None else self.quantization.to_dict(),
            "packing": None if self.packing is None else self.packing.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ShapeConstraint:
    dimension: str
    op: Literal["divisible_by", "min", "max", "equal"]
    value: int

    def __post_init__(self) -> None:
        nonblank_string(self.dimension, context="shape constraint dimension")
        _enum(self.op, ("divisible_by", "min", "max", "equal"), "shape constraint op")
        exact_int(self.value, context="shape constraint value", minimum=1)

    def applies(self, shape: dict[str, int]) -> bool:
        if self.dimension not in shape:
            return False
        actual = shape[self.dimension]
        return {
            "divisible_by": actual % self.value == 0,
            "min": actual >= self.value,
            "max": actual <= self.value,
            "equal": actual == self.value,
        }[self.op]

    @classmethod
    def from_dict(cls, value: object) -> ShapeConstraint:
        data = exact_fields(
            value, required=("dimension", "op", "value"), context="shape constraint"
        )
        return cls(
            nonblank_string(data["dimension"], context="shape constraint dimension"),
            cast(
                Literal["divisible_by", "min", "max", "equal"],
                _enum(data["op"], ("divisible_by", "min", "max", "equal"), "shape constraint op"),
            ),
            exact_int(data["value"], context="shape constraint value", minimum=1),
        )

    def to_dict(self) -> dict[str, object]:
        return {"dimension": self.dimension, "op": self.op, "value": self.value}


@dataclass(frozen=True, slots=True)
class Capability:
    state: CapabilityState
    evidence_sha256: str | None

    def __post_init__(self) -> None:
        state = _enum(self.state, ("unprobed", "available", "unavailable"), "capability state")
        evidence = (
            None
            if self.evidence_sha256 is None
            else _digest(self.evidence_sha256, "capability evidence_sha256")
        )
        if (state == "unprobed") != (evidence is None):
            raise SchemaError(
                "unprobed capability requires null evidence; available/unavailable require evidence"
            )

    @classmethod
    def from_dict(cls, value: object) -> Capability:
        data = exact_fields(value, required=("state", "evidence_sha256"), context="capability")
        return cls(
            cast(
                CapabilityState,
                _enum(data["state"], ("unprobed", "available", "unavailable"), "capability state"),
            ),
            None
            if data["evidence_sha256"] is None
            else _digest(data["evidence_sha256"], "capability evidence_sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"state": self.state, "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class Requirements:
    cuda: bool
    min_compute_capability: str | None
    features: tuple[Literal["tensor_cores", "fp8_tensor_cores", "int4_storage"], ...]

    def __post_init__(self) -> None:
        cuda = exact_bool(self.cuda, context="requirements cuda")
        capability = _nullable_string(
            self.min_compute_capability, "requirements min_compute_capability"
        )
        if not cuda and capability is not None:
            raise SchemaError("min_compute_capability requires cuda")
        allowed = ("tensor_cores", "fp8_tensor_cores", "int4_storage")
        for feature in self.features:
            _enum(feature, allowed, "requirements feature")
        if len(set(self.features)) != len(self.features):
            raise SchemaError("requirements features must be unique")

    @classmethod
    def from_dict(cls, value: object) -> Requirements:
        data = exact_fields(
            value, required=("cuda", "min_compute_capability", "features"), context="requirements"
        )
        features = _strings(data["features"], "requirements features", unique=True)
        return cls(
            exact_bool(data["cuda"], context="requirements cuda"),
            _nullable_string(data["min_compute_capability"], "requirements min_compute_capability"),
            cast(tuple[Literal["tensor_cores", "fp8_tensor_cores", "int4_storage"], ...], features),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cuda": self.cuda,
            "min_compute_capability": self.min_compute_capability,
            "features": list(self.features),
        }


@dataclass(frozen=True, slots=True)
class Arm:
    id: str
    role: Literal["candidate", "reference", "comparator"]
    entrypoint: str
    domains: tuple[Domain, ...]
    numeric_contract_ids: tuple[str, ...]
    constraints: tuple[ShapeConstraint, ...]
    local_capability: Capability
    remote_capability: Capability
    requirements: Requirements

    def __post_init__(self) -> None:
        nonblank_string(self.id, context="arm id")
        _enum(self.role, ("candidate", "reference", "comparator"), "arm role")
        nonblank_string(self.entrypoint, context="arm entrypoint")
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise SchemaError("arm domains must be nonempty and unique")
        for domain in self.domains:
            _enum(domain, DOMAIN_VOCABULARY, "arm domain")
        if not self.numeric_contract_ids or len(set(self.numeric_contract_ids)) != len(
            self.numeric_contract_ids
        ):
            raise SchemaError("arm numeric_contract_ids must be nonempty and unique")

    @classmethod
    def from_dict(cls, value: object) -> Arm:
        data = exact_fields(
            value,
            required=(
                "id",
                "role",
                "entrypoint",
                "domains",
                "numeric_contract_ids",
                "constraints",
                "local_capability",
                "remote_capability",
                "requirements",
            ),
            context="arm",
        )
        domains = _strings(data["domains"], "arm domains", unique=True)
        return cls(
            nonblank_string(data["id"], context="arm id"),
            cast(
                Literal["candidate", "reference", "comparator"],
                _enum(data["role"], ("candidate", "reference", "comparator"), "arm role"),
            ),
            nonblank_string(data["entrypoint"], context="arm entrypoint"),
            cast(tuple[Domain, ...], domains),
            _strings(data["numeric_contract_ids"], "arm numeric_contract_ids", unique=True),
            tuple(
                ShapeConstraint.from_dict(item)
                for item in _array(data["constraints"], "arm constraints")
            ),
            Capability.from_dict(data["local_capability"]),
            Capability.from_dict(data["remote_capability"]),
            Requirements.from_dict(data["requirements"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "role": self.role,
            "entrypoint": self.entrypoint,
            "domains": list(self.domains),
            "numeric_contract_ids": list(self.numeric_contract_ids),
            "constraints": [item.to_dict() for item in self.constraints],
            "local_capability": self.local_capability.to_dict(),
            "remote_capability": self.remote_capability.to_dict(),
            "requirements": self.requirements.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GatedMLPSemantics:
    kind: Literal["gated_mlp"]
    activation: Literal["silu", "gelu"]
    gate_up_layout: Literal["separate", "packed"]
    bias: bool
    residual: bool
    output_arity: int
    fusion_boundary: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind != "gated_mlp":
            raise SchemaError("gated MLP semantics kind must be 'gated_mlp'")
        _enum(self.activation, ("silu", "gelu"), "gated MLP activation")
        _enum(self.gate_up_layout, ("separate", "packed"), "gated MLP gate_up_layout")
        exact_bool(self.bias, context="gated MLP bias")
        exact_bool(self.residual, context="gated MLP residual")
        if exact_int(self.output_arity, context="gated MLP output_arity", minimum=1) != 1:
            raise SchemaError("gated MLP output_arity must be 1")
        expected = (
            ("gate_projection", "up_projection", self.activation, "gating_multiply")
            + (("bias_add",) if self.bias else ())
            + (("residual_add",) if self.residual else ())
        )
        if self.fusion_boundary != expected:
            raise SchemaError(
                "gated MLP fusion_boundary must exactly describe its ordered fused operations"
            )

    @classmethod
    def from_dict(cls, value: object) -> GatedMLPSemantics:
        data = exact_fields(
            value,
            required=(
                "kind",
                "activation",
                "gate_up_layout",
                "bias",
                "residual",
                "output_arity",
                "fusion_boundary",
            ),
            context="gated MLP semantics",
        )
        return cls(
            cast(Literal["gated_mlp"], _enum(data["kind"], ("gated_mlp",), "semantics kind")),
            cast(
                Literal["silu", "gelu"],
                _enum(data["activation"], ("silu", "gelu"), "gated MLP activation"),
            ),
            cast(
                Literal["separate", "packed"],
                _enum(data["gate_up_layout"], ("separate", "packed"), "gated MLP gate_up_layout"),
            ),
            exact_bool(data["bias"], context="gated MLP bias"),
            exact_bool(data["residual"], context="gated MLP residual"),
            exact_int(data["output_arity"], context="gated MLP output_arity", minimum=1),
            _strings(data["fusion_boundary"], "gated MLP fusion_boundary"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "activation": self.activation,
            "gate_up_layout": self.gate_up_layout,
            "bias": self.bias,
            "residual": self.residual,
            "output_arity": self.output_arity,
            "fusion_boundary": list(self.fusion_boundary),
        }


@dataclass(frozen=True, slots=True)
class RMSNormSemantics:
    kind: Literal["rmsnorm_residual"]
    epsilon: float
    gamma: bool
    residual_position: Literal["pre", "post"]
    output_arity: int
    fusion_boundary: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind != "rmsnorm_residual":
            raise SchemaError("RMSNorm semantics kind must be 'rmsnorm_residual'")
        _number(self.epsilon, "RMSNorm epsilon", positive=True)
        exact_bool(self.gamma, context="RMSNorm gamma")
        position = _enum(self.residual_position, ("pre", "post"), "RMSNorm residual_position")
        arity = exact_int(self.output_arity, context="RMSNorm output_arity", minimum=1)
        if arity not in {1, 2}:
            raise SchemaError("RMSNorm output_arity must be 1 or 2")
        expected = (
            (("residual_add", "rms_normalize") if position == "pre" else ("rms_normalize",))
            + (("gamma_multiply",) if self.gamma else ())
            + (("residual_add",) if position == "post" else ())
        )
        if self.fusion_boundary != expected:
            raise SchemaError(
                "RMSNorm fusion_boundary must exactly describe its ordered fused operations"
            )

    @classmethod
    def from_dict(cls, value: object) -> RMSNormSemantics:
        data = exact_fields(
            value,
            required=(
                "kind",
                "epsilon",
                "gamma",
                "residual_position",
                "output_arity",
                "fusion_boundary",
            ),
            context="RMSNorm semantics",
        )
        return cls(
            cast(
                Literal["rmsnorm_residual"],
                _enum(data["kind"], ("rmsnorm_residual",), "semantics kind"),
            ),
            _number(data["epsilon"], "RMSNorm epsilon", positive=True),
            exact_bool(data["gamma"], context="RMSNorm gamma"),
            cast(
                Literal["pre", "post"],
                _enum(data["residual_position"], ("pre", "post"), "RMSNorm residual_position"),
            ),
            exact_int(data["output_arity"], context="RMSNorm output_arity", minimum=1),
            _strings(data["fusion_boundary"], "RMSNorm fusion_boundary"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "residual_position": self.residual_position,
            "output_arity": self.output_arity,
            "fusion_boundary": list(self.fusion_boundary),
        }


Semantics = GatedMLPSemantics | RMSNormSemantics


def _semantics(value: object) -> Semantics:
    if type(value) is not dict:
        raise SchemaError("case semantics must be an object")
    kind = cast(dict[object, object], value).get("kind")
    if kind == "gated_mlp":
        return GatedMLPSemantics.from_dict(value)
    if kind == "rmsnorm_residual":
        return RMSNormSemantics.from_dict(value)
    raise SchemaError(f"unknown case semantics kind {kind!r}")


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    numeric_contract_id: str
    input_seed: int
    shape: tuple[tuple[str, int], ...]
    semantics: Semantics

    def __post_init__(self) -> None:
        nonblank_string(self.id, context="case id")
        nonblank_string(self.numeric_contract_id, context="case numeric_contract_id")
        exact_int(self.input_seed, context="case input_seed", minimum=0)
        names = [name for name, _ in self.shape]
        if not names or len(set(names)) != len(names):
            raise SchemaError("case shape dimensions must be nonempty and unique")
        for name, size in self.shape:
            nonblank_string(name, context="case shape dimension")
            exact_int(size, context="case shape size", minimum=1)

    @property
    def shape_dict(self) -> dict[str, int]:
        return dict(self.shape)

    @classmethod
    def from_dict(cls, value: object) -> Case:
        data = exact_fields(
            value,
            required=("id", "numeric_contract_id", "input_seed", "shape", "semantics"),
            context="case",
        )
        shape_data = data["shape"]
        if type(shape_data) is not dict or any(
            type(k) is not str for k in cast(dict[object, object], shape_data)
        ):
            raise SchemaError("case shape must be an object with string keys")
        shape = tuple(
            (key, exact_int(item, context=f"case shape {key}", minimum=1))
            for key, item in cast(dict[str, object], shape_data).items()
        )
        return cls(
            nonblank_string(data["id"], context="case id"),
            nonblank_string(data["numeric_contract_id"], context="case numeric_contract_id"),
            exact_int(data["input_seed"], context="case input_seed", minimum=0),
            shape,
            _semantics(data["semantics"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "numeric_contract_id": self.numeric_contract_id,
            "input_seed": self.input_seed,
            "shape": dict(self.shape),
            "semantics": self.semantics.to_dict(),
        }


def _validate_case_tensor_graph(
    template_id: str, tensors: tuple[TensorSpec, ...], case: Case
) -> None:
    actual = {tensor.id: (tensor.role, tensor.shape) for tensor in tensors}
    shape = case.shape_dict
    if isinstance(case.semantics, GatedMLPSemantics):
        semantics = case.semantics
        required_dimensions = {"batch", "hidden", "intermediate"}
        if not required_dimensions <= shape.keys():
            raise SchemaError("gated MLP case shape requires batch, hidden, and intermediate")
        expected: dict[str, tuple[str, tuple[str, ...]]] = {
            "input": ("input", ("batch", "hidden")),
            "output": ("output", ("batch", "intermediate")),
        }
        if semantics.gate_up_layout == "separate":
            expected.update(
                {
                    "gate_weight": ("parameter", ("intermediate", "hidden")),
                    "up_weight": ("parameter", ("intermediate", "hidden")),
                }
            )
        else:
            if "gate_up" not in shape or shape["gate_up"] != 2 * shape["intermediate"]:
                raise SchemaError(
                    "packed gated MLP case shape requires gate_up equal to twice intermediate"
                )
            expected["gate_up_weight"] = ("parameter", ("gate_up", "hidden"))
        if semantics.bias:
            expected["bias"] = ("parameter", ("intermediate",))
        if semantics.residual:
            expected["residual"] = ("input", ("batch", "intermediate"))
        if actual != expected:
            raise SchemaError(
                "gated MLP tensors must exactly match gate/up layout, bias, residual, and output semantics"
            )
        if semantics.output_arity != sum(tensor.role == "output" for tensor in tensors):
            raise SchemaError("gated MLP output_arity must equal the declared output tensor count")
        return

    if template_id != "residual_rmsnorm.v1" or not isinstance(case.semantics, RMSNormSemantics):
        raise SchemaError("case semantics do not match the suite template")
    rms_semantics = case.semantics
    if not {"tokens", "hidden"} <= shape.keys():
        raise SchemaError("RMSNorm case shape requires tokens and hidden")
    expected = {
        "input": ("input", ("tokens", "hidden")),
        "residual": ("input", ("tokens", "hidden")),
        "output": ("output", ("tokens", "hidden")),
    }
    if rms_semantics.gamma:
        expected["gamma"] = ("parameter", ("hidden",))
    if rms_semantics.output_arity == 2:
        expected["residual_output"] = ("output", ("tokens", "hidden"))
    if actual != expected:
        raise SchemaError(
            "RMSNorm tensors must exactly match residual, gamma, and output arity semantics"
        )
    if rms_semantics.output_arity != sum(tensor.role == "output" for tensor in tensors):
        raise SchemaError("RMSNorm output_arity must equal the declared output tensor count")


@dataclass(frozen=True, slots=True)
class CorrectnessPolicy:
    id: str
    reference_arm_id: str
    atol: float
    rtol: float

    def __post_init__(self) -> None:
        nonblank_string(self.id, context="correctness policy id")
        nonblank_string(self.reference_arm_id, context="correctness reference_arm_id")
        _number(self.atol, "correctness atol")
        _number(self.rtol, "correctness rtol")

    @classmethod
    def from_dict(cls, value: object) -> CorrectnessPolicy:
        data = exact_fields(
            value, required=("id", "reference_arm_id", "atol", "rtol"), context="correctness policy"
        )
        return cls(
            nonblank_string(data["id"], context="correctness policy id"),
            nonblank_string(data["reference_arm_id"], context="correctness reference_arm_id"),
            _number(data["atol"], "correctness atol"),
            _number(data["rtol"], "correctness rtol"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "reference_arm_id": self.reference_arm_id,
            "atol": self.atol,
            "rtol": self.rtol,
        }


@dataclass(frozen=True, slots=True)
class TimingPolicy:
    id: str
    warmups: int
    repetitions: int
    statistic: Literal["median", "minimum"]

    def __post_init__(self) -> None:
        nonblank_string(self.id, context="timing policy id")
        exact_int(self.warmups, context="timing warmups", minimum=0)
        exact_int(self.repetitions, context="timing repetitions", minimum=1)
        _enum(self.statistic, ("median", "minimum"), "timing statistic")

    @classmethod
    def from_dict(cls, value: object) -> TimingPolicy:
        data = exact_fields(
            value, required=("id", "warmups", "repetitions", "statistic"), context="timing policy"
        )
        return cls(
            nonblank_string(data["id"], context="timing policy id"),
            exact_int(data["warmups"], context="timing warmups", minimum=0),
            exact_int(data["repetitions"], context="timing repetitions", minimum=1),
            cast(
                Literal["median", "minimum"],
                _enum(data["statistic"], ("median", "minimum"), "timing statistic"),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "warmups": self.warmups,
            "repetitions": self.repetitions,
            "statistic": self.statistic,
        }


@dataclass(frozen=True, slots=True)
class ExpectedCell:
    id: str
    case_id: str
    arm_id: str
    input_seed: int
    stage: Literal["correctness", "timing"]
    correctness_policy_id: str
    timing_policy_id: str | None

    def __post_init__(self) -> None:
        nonblank_string(self.id, context="expected cell id")
        nonblank_string(self.case_id, context="expected cell case_id")
        nonblank_string(self.arm_id, context="expected cell arm_id")
        exact_int(self.input_seed, context="expected cell input_seed", minimum=0)
        stage = _enum(self.stage, ("correctness", "timing"), "expected cell stage")
        nonblank_string(self.correctness_policy_id, context="expected cell correctness_policy_id")
        timing = _nullable_string(self.timing_policy_id, "expected cell timing_policy_id")
        if (stage == "timing") != (timing is not None):
            raise SchemaError("timing_policy_id is required exactly for timing cells")

    @classmethod
    def from_dict(cls, value: object) -> ExpectedCell:
        data = exact_fields(
            value,
            required=(
                "id",
                "case_id",
                "arm_id",
                "input_seed",
                "stage",
                "correctness_policy_id",
                "timing_policy_id",
            ),
            context="expected cell",
        )
        return cls(
            nonblank_string(data["id"], context="expected cell id"),
            nonblank_string(data["case_id"], context="expected cell case_id"),
            nonblank_string(data["arm_id"], context="expected cell arm_id"),
            exact_int(data["input_seed"], context="expected cell input_seed", minimum=0),
            cast(
                Literal["correctness", "timing"],
                _enum(data["stage"], ("correctness", "timing"), "expected cell stage"),
            ),
            nonblank_string(
                data["correctness_policy_id"], context="expected cell correctness_policy_id"
            ),
            _nullable_string(data["timing_policy_id"], "expected cell timing_policy_id"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "input_seed": self.input_seed,
            "stage": self.stage,
            "correctness_policy_id": self.correctness_policy_id,
            "timing_policy_id": self.timing_policy_id,
        }


@dataclass(frozen=True, slots=True)
class Suite:
    schema: Literal["heliostune.suite/1"]
    suite_id: str
    revision: int
    plugin_id: str
    plugin_version: int
    template_id: str
    template_status: Literal["reference_template_not_execution_freeze"]
    domain: Domain
    numeric_contracts: tuple[NumericContract, ...]
    tensors: tuple[TensorSpec, ...]
    arms: tuple[Arm, ...]
    cases: tuple[Case, ...]
    correctness_policies: tuple[CorrectnessPolicy, ...]
    timing_policies: tuple[TimingPolicy, ...]
    expected_cells: tuple[ExpectedCell, ...]
    executor_rule: Literal["timing_requires_retained_passing_correctness_observation"]

    def __post_init__(self) -> None:
        if self.schema != "heliostune.suite/1":
            raise SchemaError("suite schema must be 'heliostune.suite/1'")
        nonblank_string(self.suite_id, context="suite id")
        exact_int(self.revision, context="suite revision", minimum=1)
        nonblank_string(self.plugin_id, context="suite plugin_id")
        exact_int(self.plugin_version, context="suite plugin_version", minimum=1)
        if self.template_id not in EXECUTABLE_TEMPLATE_IDS:
            raise SchemaError(f"unknown executable template_id {self.template_id!r}")
        if self.template_status != "reference_template_not_execution_freeze":
            raise SchemaError(
                "suite must be labeled as a reference template, not an execution freeze"
            )
        domain = _enum(self.domain, DOMAIN_VOCABULARY, "suite domain")
        expected_domain = {
            "gated_mlp_epilogue.v1": "fused_mlp",
            "residual_rmsnorm.v1": "rmsnorm_residual",
        }[self.template_id]
        if domain != expected_domain:
            raise SchemaError("suite template_id and domain disagree")
        for items, name in (
            (self.numeric_contracts, "numeric contract"),
            (self.tensors, "tensor"),
            (self.arms, "arm"),
            (self.cases, "case"),
            (self.correctness_policies, "correctness policy"),
            (self.timing_policies, "timing policy"),
            (self.expected_cells, "expected cell"),
        ):
            if not items:
                raise SchemaError(f"suite {name}s must be nonempty")
            _unique_ids(cast(tuple[_HasId, ...], items), f"suite {name}")
        contracts_by_id = {item.id: item for item in self.numeric_contracts}
        arms_by_id = {item.id: item for item in self.arms}
        cases_by_id = {item.id: item for item in self.cases}
        correctness_by_id = {item.id: item for item in self.correctness_policies}
        timing_ids = {item.id for item in self.timing_policies}
        if any(not item.is_initially_executable for item in self.numeric_contracts):
            raise SchemaError(
                "executable suite templates permit only fp16/bf16 input/storage, fp32 accumulation, fp16/bf16/fp32 output, TF32 false, and null quantization"
            )
        initial_tensor_dtypes = {"fp16", "bf16", "fp32"}
        if any(
            tensor.storage_dtype not in initial_tensor_dtypes
            or tensor.logical_dtype not in initial_tensor_dtypes
            or (tensor.role != "output" and "fp32" in {tensor.storage_dtype, tensor.logical_dtype})
            for tensor in self.tensors
        ):
            raise SchemaError(
                "executable suite template input, parameter, and intermediate tensors permit "
                "only fp16/bf16 storage and logical dtypes; output tensors additionally permit "
                "fp32 when every applicable case contract declares fp32 output"
            )
        for arm in self.arms:
            if domain not in arm.domains:
                raise SchemaError(f"arm {arm.id!r} does not declare the suite domain")
            if not set(arm.numeric_contract_ids) <= contracts_by_id.keys():
                raise SchemaError(f"arm {arm.id!r} references an unknown numeric contract")
        for case in self.cases:
            if case.numeric_contract_id not in contracts_by_id:
                raise SchemaError(f"case {case.id!r} references an unknown numeric contract")
            if self.template_id == "gated_mlp_epilogue.v1" and not isinstance(
                case.semantics, GatedMLPSemantics
            ):
                raise SchemaError("gated MLP suite requires gated MLP case semantics")
            if self.template_id == "residual_rmsnorm.v1" and not isinstance(
                case.semantics, RMSNormSemantics
            ):
                raise SchemaError("RMSNorm suite requires RMSNorm case semantics")
            _validate_case_tensor_graph(self.template_id, self.tensors, case)
        # Tensor declarations are suite-global and shared by every case. Arms advertise
        # implementation capabilities, so FP32 output eligibility intersects case contracts.
        applicable_contracts = {contracts_by_id[case.numeric_contract_id] for case in self.cases}
        if any(
            tensor.role == "output"
            and "fp32" in {tensor.storage_dtype, tensor.logical_dtype}
            and any(contract.output.name != "fp32" for contract in applicable_contracts)
            for tensor in self.tensors
        ):
            raise SchemaError(
                "fp32 output tensors require every applicable case numeric contract to declare "
                "fp32 output"
            )
        for policy in self.correctness_policies:
            if (
                policy.reference_arm_id not in arms_by_id
                or arms_by_id[policy.reference_arm_id].role != "reference"
            ):
                raise SchemaError("correctness policy reference_arm_id must name a reference arm")
        correctness_expected = {
            (cell.case_id, cell.arm_id, cell.input_seed, cell.correctness_policy_id)
            for cell in self.expected_cells
            if cell.stage == "correctness"
        }
        observed: set[tuple[str, str, int, str]] = set()
        for cell in self.expected_cells:
            if (
                cell.case_id not in cases_by_id
                or cell.arm_id not in arms_by_id
                or cell.correctness_policy_id not in correctness_by_id
                or (cell.timing_policy_id is not None and cell.timing_policy_id not in timing_ids)
            ):
                raise SchemaError(f"expected cell {cell.id!r} contains an unknown reference")
            case = cases_by_id[cell.case_id]
            arm = arms_by_id[cell.arm_id]
            if cell.input_seed != case.input_seed:
                raise SchemaError("expected cell input_seed must equal its case input_seed")
            if case.numeric_contract_id not in arm.numeric_contract_ids or any(
                not constraint.applies(case.shape_dict) for constraint in arm.constraints
            ):
                raise SchemaError(f"expected cell {cell.id!r} uses an inapplicable arm")
            key = (cell.case_id, cell.arm_id, cell.input_seed, cell.correctness_policy_id)
            if cell.stage == "timing" and key not in observed:
                raise SchemaError(
                    "timing expected cell requires an earlier correctness stage for the same case/arm/input seed"
                )
            if cell.stage == "correctness":
                policy = correctness_by_id[cell.correctness_policy_id]
                reference = arms_by_id[policy.reference_arm_id]
                reference_key = (
                    case.id,
                    reference.id,
                    case.input_seed,
                    policy.id,
                )
                if (
                    case.numeric_contract_id not in reference.numeric_contract_ids
                    or any(
                        not constraint.applies(case.shape_dict)
                        for constraint in reference.constraints
                    )
                    or reference_key not in correctness_expected
                ):
                    raise SchemaError(
                        f"correctness expected cell {cell.id!r} uses an inapplicable or unlisted reference arm"
                    )
                observed.add(key)
        if self.executor_rule != "timing_requires_retained_passing_correctness_observation":
            raise SchemaError(
                "suite must expose the runtime observation gate without claiming it statically"
            )

    @classmethod
    def from_dict(cls, value: object) -> Suite:
        fields = (
            "schema",
            "suite_id",
            "revision",
            "plugin_id",
            "plugin_version",
            "template_id",
            "template_status",
            "domain",
            "numeric_contracts",
            "tensors",
            "arms",
            "cases",
            "correctness_policies",
            "timing_policies",
            "expected_cells",
            "executor_rule",
        )
        data = exact_fields(value, required=fields, context="suite")
        return cls(
            cast(
                Literal["heliostune.suite/1"],
                nonblank_string(data["schema"], context="suite schema"),
            ),
            nonblank_string(data["suite_id"], context="suite id"),
            exact_int(data["revision"], context="suite revision", minimum=1),
            nonblank_string(data["plugin_id"], context="suite plugin_id"),
            exact_int(data["plugin_version"], context="suite plugin_version", minimum=1),
            nonblank_string(data["template_id"], context="suite template_id"),
            cast(
                Literal["reference_template_not_execution_freeze"],
                nonblank_string(data["template_status"], context="suite template_status"),
            ),
            cast(Domain, _enum(data["domain"], DOMAIN_VOCABULARY, "suite domain")),
            tuple(
                NumericContract.from_dict(x)
                for x in _array(data["numeric_contracts"], "suite numeric_contracts")
            ),
            tuple(TensorSpec.from_dict(x) for x in _array(data["tensors"], "suite tensors")),
            tuple(Arm.from_dict(x) for x in _array(data["arms"], "suite arms")),
            tuple(Case.from_dict(x) for x in _array(data["cases"], "suite cases")),
            tuple(
                CorrectnessPolicy.from_dict(x)
                for x in _array(data["correctness_policies"], "suite correctness_policies")
            ),
            tuple(
                TimingPolicy.from_dict(x)
                for x in _array(data["timing_policies"], "suite timing_policies")
            ),
            tuple(
                ExpectedCell.from_dict(x)
                for x in _array(data["expected_cells"], "suite expected_cells")
            ),
            cast(
                Literal["timing_requires_retained_passing_correctness_observation"],
                nonblank_string(data["executor_rule"], context="suite executor_rule"),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "suite_id": self.suite_id,
            "revision": self.revision,
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "template_id": self.template_id,
            "template_status": self.template_status,
            "domain": self.domain,
            "numeric_contracts": [x.to_dict() for x in self.numeric_contracts],
            "tensors": [x.to_dict() for x in self.tensors],
            "arms": [x.to_dict() for x in self.arms],
            "cases": [x.to_dict() for x in self.cases],
            "correctness_policies": [x.to_dict() for x in self.correctness_policies],
            "timing_policies": [x.to_dict() for x in self.timing_policies],
            "expected_cells": [x.to_dict() for x in self.expected_cells],
            "executor_rule": self.executor_rule,
        }


@dataclass(frozen=True, slots=True)
class SuiteRef:
    path: str
    sha256: str
    suite_id: str
    revision: int

    def __post_init__(self) -> None:
        path = nonblank_string(self.path, context="suite ref path")
        if (
            "\\" in path
            or "\x00" in path
            or PurePosixPath(path).is_absolute()
            or path != PurePosixPath(path).as_posix()
        ):
            raise SchemaError("suite ref path must be a normalized POSIX relative path")
        _digest(self.sha256, "suite ref sha256")
        nonblank_string(self.suite_id, context="suite ref suite_id")
        exact_int(self.revision, context="suite ref revision", minimum=1)

    @classmethod
    def from_dict(cls, value: object) -> SuiteRef:
        data = exact_fields(
            value, required=("path", "sha256", "suite_id", "revision"), context="suite ref"
        )
        return cls(
            nonblank_string(data["path"], context="suite ref path"),
            _digest(data["sha256"], "suite ref sha256"),
            nonblank_string(data["suite_id"], context="suite ref suite_id"),
            exact_int(data["revision"], context="suite ref revision", minimum=1),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "suite_id": self.suite_id,
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class Plugin:
    schema: Literal["heliostune.plugin/1"]
    plugin_id: str
    version: int
    template_status: Literal["reference_template_not_execution_freeze"]
    domains: tuple[Domain, ...]
    arm_ids: tuple[str, ...]
    suite_refs: tuple[SuiteRef, ...]

    def __post_init__(self) -> None:
        if self.schema != "heliostune.plugin/1":
            raise SchemaError("plugin schema must be 'heliostune.plugin/1'")
        nonblank_string(self.plugin_id, context="plugin id")
        exact_int(self.version, context="plugin version", minimum=1)
        if self.template_status != "reference_template_not_execution_freeze":
            raise SchemaError(
                "plugin must be labeled as a reference template, not an execution freeze"
            )
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise SchemaError("plugin domains must be nonempty and unique")
        for domain in self.domains:
            _enum(domain, DOMAIN_VOCABULARY, "plugin domain")
        if not self.arm_ids or len(set(self.arm_ids)) != len(self.arm_ids):
            raise SchemaError("plugin arm_ids must be nonempty and unique")
        if not self.suite_refs:
            raise SchemaError("plugin suite_refs must be nonempty")
        ids = [(ref.suite_id, ref.revision) for ref in self.suite_refs]
        if len(set(ids)) != len(ids):
            raise SchemaError("plugin suite refs must identify unique suite revisions")

    @classmethod
    def from_dict(cls, value: object) -> Plugin:
        data = exact_fields(
            value,
            required=(
                "schema",
                "plugin_id",
                "version",
                "template_status",
                "domains",
                "arm_ids",
                "suite_refs",
            ),
            context="plugin",
        )
        return cls(
            cast(
                Literal["heliostune.plugin/1"],
                nonblank_string(data["schema"], context="plugin schema"),
            ),
            nonblank_string(data["plugin_id"], context="plugin id"),
            exact_int(data["version"], context="plugin version", minimum=1),
            cast(
                Literal["reference_template_not_execution_freeze"],
                nonblank_string(data["template_status"], context="plugin template_status"),
            ),
            cast(tuple[Domain, ...], _strings(data["domains"], "plugin domains", unique=True)),
            _strings(data["arm_ids"], "plugin arm_ids", unique=True),
            tuple(SuiteRef.from_dict(x) for x in _array(data["suite_refs"], "plugin suite_refs")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "plugin_id": self.plugin_id,
            "version": self.version,
            "template_status": self.template_status,
            "domains": list(self.domains),
            "arm_ids": list(self.arm_ids),
            "suite_refs": [x.to_dict() for x in self.suite_refs],
        }


@dataclass(frozen=True, slots=True)
class VerifiedSuite:
    path: Path
    bytes: bytes
    sha256: str
    suite: Suite


@dataclass(frozen=True, slots=True)
class VerifiedPlugin:
    path: Path
    bytes: bytes
    sha256: str
    plugin: Plugin
    suites: tuple[VerifiedSuite, ...]


def load_suite(path: str | Path) -> Suite:
    return Suite.from_dict(read_json(path))


def load_plugin(path: str | Path) -> Plugin:
    return Plugin.from_dict(read_json(path))


def _read_verified(path: str | Path, context: str) -> tuple[Path, bytes]:
    resolved = Path(path).resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read {context} {resolved}: {exc}") from exc
    if not resolved.is_file():
        raise ArtifactError(f"{context} is not a regular file: {resolved}")
    return resolved, payload


def _parse_verified_json(payload: bytes, source: Path) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ArtifactError(f"cannot decode JSON artifact {source}: {exc}") from exc
    return strict_json_loads(text, source=source)


def verify_suite(path: str | Path) -> VerifiedSuite:
    """Structurally verify a suite; this does not claim runtime observations."""
    resolved, payload = _read_verified(path, "suite")
    suite = Suite.from_dict(_parse_verified_json(payload, resolved))
    return VerifiedSuite(resolved, payload, hashlib.sha256(payload).hexdigest(), suite)


def verify_plugin(path: str | Path) -> VerifiedPlugin:
    """Verify plugin bytes and the contained, digest-bound suites they reference."""
    resolved, payload = _read_verified(path, "plugin")
    plugin = Plugin.from_dict(_parse_verified_json(payload, resolved))
    containment_root = resolved.parent.parent.resolve()
    suites: list[VerifiedSuite] = []
    for ref in plugin.suite_refs:
        target = (resolved.parent / ref.path).resolve()
        if target == containment_root or containment_root not in target.parents:
            raise ArtifactError(f"suite ref path escapes plugin containment root: {ref.path!r}")
        verified = verify_suite(target)
        if verified.sha256 != ref.sha256:
            raise ArtifactError(
                f"suite digest mismatch for {ref.path!r}: expected {ref.sha256}, got {verified.sha256}"
            )
        if (verified.suite.suite_id, verified.suite.revision) != (
            ref.suite_id,
            ref.revision,
        ):
            raise ArtifactError(f"suite identity mismatch for {ref.path!r}")
        if (verified.suite.plugin_id, verified.suite.plugin_version) != (
            plugin.plugin_id,
            plugin.version,
        ):
            raise ArtifactError(f"suite plugin identity mismatch for {ref.path!r}")
        suites.append(verified)
    domains = tuple(dict.fromkeys(item.suite.domain for item in suites))
    arm_ids = tuple(dict.fromkeys(arm.id for item in suites for arm in item.suite.arms))
    if domains != plugin.domains:
        raise ArtifactError("plugin domains must exactly equal referenced suite domains in order")
    if arm_ids != plugin.arm_ids:
        raise ArtifactError("plugin arm_ids must exactly equal referenced suite arms in order")
    return VerifiedPlugin(
        resolved, payload, hashlib.sha256(payload).hexdigest(), plugin, tuple(suites)
    )
