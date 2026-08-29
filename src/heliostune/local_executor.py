from __future__ import annotations

import contextlib
import hashlib
import importlib
import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from .errors import SchemaError
from .scope import Case, ExpectedCell, GatedMLPSemantics, RMSNormSemantics, Suite, verify_suite

CapabilityReason = Literal[
    "torch_missing",
    "torch_version_mismatch",
    "cuda_unavailable",
    "rocm_unsupported",
    "compute_capability_too_low",
    "bf16_unsupported",
    "inductor_unavailable",
    "allocation_failed",
    "device_probe_failed",
]
CellStatus = Literal["passed", "failed", "blocked"]


@dataclass(frozen=True, slots=True)
class CapabilityProbe:
    available: bool
    reasons: tuple[CapabilityReason, ...]
    torch_version: str | None
    cuda_version: str | None
    rocm_version: str | None
    device_index: int | None
    device_name: str | None
    compute_capability: tuple[int, int] | None
    native_bf16: bool | None
    inductor_available: bool | None
    allocation_succeeded: bool
    detail: str | None = None

    @property
    def reason(self) -> CapabilityReason | None:
        return self.reasons[0] if self.reasons else None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["compute_capability"] = (
            None if self.compute_capability is None else list(self.compute_capability)
        )
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True, slots=True)
class TensorMaterialization:
    suite_sha256: str
    case_id: str
    arm_id: str
    input_seed: int
    tensor_order: tuple[str, ...]
    tensors: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_sha256": self.suite_sha256,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "input_seed": self.input_seed,
            "tensor_order": list(self.tensor_order),
            "tensors": [dict(item) for item in self.tensors],
        }


@dataclass(frozen=True, slots=True)
class CorrectnessObservation:
    status: CellStatus
    correctness_key: str
    failure_kind: str | None
    message: str | None
    output: Mapping[str, object] | None
    input_storage_unchanged: bool
    output_disjoint: bool
    finite: bool
    close: bool
    max_abs_error: float | None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["output"] = None if self.output is None else dict(self.output)
        return value


@dataclass(frozen=True, slots=True)
class TimingObservation:
    status: CellStatus
    correctness_key: str
    failure_kind: str | None
    message: str | None
    warmups: int
    repetitions: int
    samples_ms: tuple[float, ...]
    median_ms: float | None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["samples_ms"] = list(self.samples_ms)
        return value


@dataclass(frozen=True, slots=True)
class CellObservation:
    cell_id: str
    case_id: str
    arm_id: str
    stage: Literal["correctness", "timing"]
    status: CellStatus
    correctness: CorrectnessObservation | None
    timing: TimingObservation | None

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "stage": self.stage,
            "status": self.status,
            "correctness": None if self.correctness is None else self.correctness.to_dict(),
            "timing": None if self.timing is None else self.timing.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LocalExecutionResult:
    verified_suite_path: str
    verified_suite_sha256: str
    verified_suite_bytes: bytes
    suite_id: str
    capability: CapabilityProbe
    materialization: tuple[TensorMaterialization, ...]
    observations: tuple[CellObservation, ...]
    attempts: tuple[Mapping[str, object], ...]
    environment: Mapping[str, object]
    compile_outcomes: Mapping[str, Mapping[str, object]]
    summary: Mapping[str, object]
    outcome: Literal["completed", "failed", "aborted"]

    @property
    def suite_path(self) -> str:
        return self.verified_suite_path

    @property
    def suite_sha256(self) -> str:
        return self.verified_suite_sha256

    @property
    def suite_bytes(self) -> bytes:
        return self.verified_suite_bytes

    def to_dict(self, *, include_suite_bytes: bool = False) -> dict[str, object]:
        value: dict[str, object] = {
            "verified_suite_path": self.verified_suite_path,
            "verified_suite_sha256": self.verified_suite_sha256,
            "suite_id": self.suite_id,
            "capability": self.capability.to_dict(),
            "materialization": [item.to_dict() for item in self.materialization],
            "observations": [item.to_dict() for item in self.observations],
            "attempts": [dict(item) for item in self.attempts],
            "environment": dict(self.environment),
            "compile_outcomes": {key: dict(item) for key, item in self.compile_outcomes.items()},
            "summary": dict(self.summary),
            "outcome": self.outcome,
        }
        if include_suite_bytes:
            value["verified_suite_bytes"] = self.verified_suite_bytes
        return value


@dataclass(frozen=True, slots=True)
class _DrawInstruction:
    tensor_id: str
    role: str
    shape: tuple[int, ...]
    normal_scale: float
    normal_offset: float


class _ExecutionValidationError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


GATED_MLP_SUITE_SHA256 = "407487a6aa7dc157dcd4aa7bcab698168813bf0a79916d70d91163dc384fe8a8"
RMSNORM_SUITE_SHA256 = "a318a59bca434b97d073e0ae76f827814213c0a68b0c4263b19c81f98be8f9ee"

_FROZEN_SUITE_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "gated-mlp-epilogue-reference": GATED_MLP_SUITE_SHA256,
        "residual-rmsnorm-reference": RMSNORM_SUITE_SHA256,
    }
)


_FROZEN_TEMPLATE = {
    "gated_mlp_epilogue.v1": {
        "suite_id": "gated-mlp-epilogue-reference",
        "domain": "fused_mlp",
        "case_id": "mlp-case-001",
        "shape": {"batch": 8, "hidden": 4096, "intermediate": 11008},
        "semantics": {
            "kind": "gated_mlp",
            "activation": "silu",
            "gate_up_layout": "separate",
            "bias": False,
            "residual": False,
            "output_arity": 1,
            "fusion_boundary": [
                "gate_projection",
                "up_projection",
                "silu",
                "gating_multiply",
            ],
        },
        "tensor_ids": ("input", "gate_weight", "up_weight", "output"),
        "arm_entries": {
            "mlp-candidate": "reference_template.gated_mlp_candidate",
            "mlp-reference": "reference_template.gated_mlp_reference",
        },
        "arm_roles": (("mlp-candidate", "candidate"), ("mlp-reference", "reference")),
        "cell_ids": (
            "mlp-candidate-correctness",
            "mlp-candidate-timing",
            "mlp-reference-correctness",
            "mlp-reference-timing",
        ),
    },
    "residual_rmsnorm.v1": {
        "suite_id": "residual-rmsnorm-reference",
        "domain": "rmsnorm_residual",
        "case_id": "rmsnorm-case-001",
        "shape": {"tokens": 128, "hidden": 4096},
        "semantics": {
            "kind": "rmsnorm_residual",
            "epsilon": 1e-5,
            "gamma": True,
            "residual_position": "pre",
            "output_arity": 1,
            "fusion_boundary": ["residual_add", "rms_normalize", "gamma_multiply"],
        },
        "tensor_ids": ("input", "residual", "gamma", "output"),
        "arm_entries": {
            "rmsnorm-candidate": "reference_template.residual_rmsnorm_candidate",
            "rmsnorm-reference": "reference_template.residual_rmsnorm_reference",
        },
        "arm_roles": (("rmsnorm-candidate", "candidate"), ("rmsnorm-reference", "reference")),
        "cell_ids": (
            "rmsnorm-candidate-correctness",
            "rmsnorm-candidate-timing",
            "rmsnorm-reference-correctness",
            "rmsnorm-reference-timing",
        ),
    },
}

_ENTRYPOINTS = {
    "reference_template.gated_mlp_candidate": "gated_mlp_candidate",
    "reference_template.gated_mlp_reference": "gated_mlp_reference",
    "reference_template.residual_rmsnorm_candidate": "residual_rmsnorm_candidate",
    "reference_template.residual_rmsnorm_reference": "residual_rmsnorm_reference",
}

_PRECISION_POLICY: dict[str, object] = {
    "float32_matmul_precision": "highest",
    "allow_tf32": False,
    "allow_bf16_reduced_precision_reduction": False,
    "allow_fp16_reduced_precision_reduction": False,
    "allow_fp16_accumulation": False,
}

_AUTOCAST_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "device_type": "cuda",
        "enabled": False,
        "restore_ambient_state": True,
    }
)

_CELL_TRANSITIONS = {
    "pending": frozenset({"running"}),
    "running": frozenset({"passed", "failed"}),
    "passed": frozenset(),
    "failed": frozenset(),
}


def _advance_cell_state(current: str, target: str) -> str:
    if target not in _CELL_TRANSITIONS.get(current, frozenset()):
        raise ValueError(f"invalid cell transition {current!r} -> {target!r}")
    return target


def _validate_frozen_suite(suite: Suite, *, suite_sha256: str | None = None) -> None:
    frozen = _FROZEN_TEMPLATE.get(suite.template_id)
    if frozen is None:
        raise SchemaError(f"local executor does not recognize template_id {suite.template_id!r}")
    if suite.template_status != "reference_template_not_execution_freeze":
        raise SchemaError("local executor requires the frozen reference template status")
    if suite.suite_id != frozen["suite_id"] or suite.revision != 1:
        raise SchemaError("local executor recognizes only the two revision-1 frozen suites")
    if suite_sha256 is not None and suite_sha256 != _FROZEN_SUITE_SHA256[suite.suite_id]:
        raise SchemaError("local executor accepts only the exact committed frozen suite bytes")
    if suite.domain != frozen["domain"]:
        raise SchemaError("suite domain is outside the frozen local execution contract")
    if len(suite.numeric_contracts) != 1:
        raise SchemaError("frozen local execution requires exactly one numeric contract")
    contract = suite.numeric_contracts[0].to_dict()
    expected_contract = {
        "id": "bf16-fp32-bf16",
        "input": {"name": "bf16", "usage": "input", "packing": None},
        "storage": {"name": "bf16", "usage": "storage", "packing": None},
        "accumulation": {"name": "fp32", "usage": "compute", "packing": None},
        "output": {"name": "bf16", "usage": "output", "packing": None},
        "tf32": False,
        "quantization": None,
    }
    if contract != expected_contract:
        raise SchemaError("numeric contract is outside the frozen bf16-fp32-bf16 contract")
    if len(suite.cases) != 1:
        raise SchemaError("frozen local execution requires exactly one case")
    case = suite.cases[0]
    if (
        case.id != frozen["case_id"]
        or case.input_seed != 17
        or case.numeric_contract_id != "bf16-fp32-bf16"
        or case.shape_dict != frozen["shape"]
        or case.semantics.to_dict() != frozen["semantics"]
    ):
        raise SchemaError("case is outside the frozen shape, seed, contract, or semantics")
    if tuple(tensor.id for tensor in suite.tensors) != frozen["tensor_ids"]:
        raise SchemaError("tensor declarations or declared tensor order are outside the freeze")
    if any(
        tensor.storage_dtype != "bf16"
        or tensor.logical_dtype != "bf16"
        or tensor.layout != "row_major"
        or not tensor.contiguous
        for tensor in suite.tensors
    ):
        raise SchemaError("tensor dtype/layout declarations are outside the frozen contract")
    arm_entries = {arm.id: arm.entrypoint for arm in suite.arms}
    if arm_entries != frozen["arm_entries"] or any(
        arm.entrypoint not in _ENTRYPOINTS for arm in suite.arms
    ):
        raise SchemaError("suite arm entrypoints are outside the closed local registry")
    if tuple((arm.id, arm.role) for arm in suite.arms) != frozen["arm_roles"]:
        raise SchemaError("suite arm order or roles are outside the frozen contract")
    if (
        len(suite.correctness_policies) != 1
        or suite.correctness_policies[0].id != "default-correctness"
        or suite.correctness_policies[0].reference_arm_id != suite.arms[1].id
        or (suite.correctness_policies[0].atol, suite.correctness_policies[0].rtol) != (0.02, 0.02)
    ):
        raise SchemaError("correctness policy is outside the frozen tolerance")
    if (
        len(suite.timing_policies) != 1
        or suite.timing_policies[0].id != "default-timing"
        or (
            suite.timing_policies[0].warmups,
            suite.timing_policies[0].repetitions,
            suite.timing_policies[0].statistic,
        )
        != (10, 50, "median")
    ):
        raise SchemaError("timing policy is outside the frozen 10/50 median contract")
    if tuple(cell.id for cell in suite.expected_cells) != frozen["cell_ids"]:
        raise SchemaError("expected cell IDs or order are outside the frozen contract")
    for arm in suite.arms:
        cells = [cell for cell in suite.expected_cells if cell.arm_id == arm.id]
        if [cell.stage for cell in cells] != ["correctness", "timing"]:
            raise SchemaError("each frozen arm requires correctness then timing cells")
        if any(
            cell.case_id != case.id
            or cell.input_seed != 17
            or cell.correctness_policy_id != "default-correctness"
            for cell in cells
        ):
            raise SchemaError("expected cell key is outside the frozen contract")


def _safe_error(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _unavailable_probe(
    reason: CapabilityReason, *, version: str | None = None, detail: str | None = None
) -> CapabilityProbe:
    return CapabilityProbe(
        False, (reason,), version, None, None, None, None, None, None, None, False, detail
    )


def _probe_torch(torch: Any, suite: Suite) -> CapabilityProbe:
    del suite
    version = getattr(torch, "__version__", None)
    if not isinstance(version, str) or version.partition("+")[0] != "2.8.0":
        return _unavailable_probe("torch_version_mismatch", version=cast(str | None, version))
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    rocm_version = getattr(getattr(torch, "version", None), "hip", None)
    if rocm_version is not None:
        return CapabilityProbe(
            False,
            ("rocm_unsupported",),
            version,
            cast(str | None, cuda_version),
            str(rocm_version),
            None,
            None,
            None,
            None,
            None,
            False,
        )
    try:
        if not bool(torch.cuda.is_available()):
            return CapabilityProbe(
                False,
                ("cuda_unavailable",),
                version,
                cast(str | None, cuda_version),
                None,
                None,
                None,
                None,
                None,
                None,
                False,
            )
    except Exception as exc:
        return _unavailable_probe("device_probe_failed", version=version, detail=_safe_error(exc))
    try:
        device_index = int(torch.cuda.current_device())
        raw_cc = torch.cuda.get_device_capability(device_index)
        cc = (int(raw_cc[0]), int(raw_cc[1]))
        name = str(torch.cuda.get_device_name(device_index))
    except Exception as exc:
        return CapabilityProbe(
            False,
            ("device_probe_failed",),
            version,
            cast(str | None, cuda_version),
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            _safe_error(exc),
        )
    if cc < (8, 0):
        return CapabilityProbe(
            False,
            ("compute_capability_too_low",),
            version,
            cast(str | None, cuda_version),
            None,
            device_index,
            name,
            cc,
            None,
            None,
            False,
        )
    try:
        native_bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        native_bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:
        return CapabilityProbe(
            False,
            ("device_probe_failed",),
            version,
            cast(str | None, cuda_version),
            None,
            device_index,
            name,
            cc,
            None,
            None,
            False,
            _safe_error(exc),
        )
    if not native_bf16:
        return CapabilityProbe(
            False,
            ("bf16_unsupported",),
            version,
            cast(str | None, cuda_version),
            None,
            device_index,
            name,
            cc,
            False,
            None,
            False,
        )
    try:
        backends = tuple(torch.compiler.list_backends())
        inductor = callable(getattr(torch, "compile", None)) and "inductor" in backends
    except Exception:
        inductor = False
    if not inductor:
        return CapabilityProbe(
            False,
            ("inductor_unavailable",),
            version,
            cast(str | None, cuda_version),
            None,
            device_index,
            name,
            cc,
            True,
            False,
            False,
        )
    try:
        allocation = torch.empty((1,), device=f"cuda:{device_index}", dtype=torch.bfloat16)
        torch.cuda.synchronize(device_index)
        del allocation
    except Exception as exc:
        return CapabilityProbe(
            False,
            ("allocation_failed",),
            version,
            cast(str | None, cuda_version),
            None,
            device_index,
            name,
            cc,
            True,
            True,
            False,
            _safe_error(exc),
        )
    return CapabilityProbe(
        True,
        (),
        version,
        cast(str | None, cuda_version),
        None,
        device_index,
        name,
        cc,
        True,
        True,
        True,
    )


def probe_local_capability(suite: Suite) -> CapabilityProbe:
    _validate_frozen_suite(suite)
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return _unavailable_probe("torch_missing", detail=_safe_error(exc))
    return _probe_torch(torch, suite)


def _resolve_draw_schedule(suite: Suite, case: Case) -> tuple[_DrawInstruction, ...]:
    dimensions = case.shape_dict
    schedule: list[_DrawInstruction] = []
    for tensor in suite.tensors:
        if tensor.role == "output":
            continue
        shape = tuple(dimensions[name] for name in tensor.shape)
        scale = 1.0
        offset = 0.0
        if suite.template_id == "gated_mlp_epilogue.v1" and tensor.role == "parameter":
            scale = 1.0 / math.sqrt(dimensions["hidden"])
        if suite.template_id == "residual_rmsnorm.v1" and tensor.id == "gamma":
            scale = 0.02
            offset = 1.0
        schedule.append(_DrawInstruction(tensor.id, tensor.role, shape, scale, offset))
    return tuple(schedule)


def _tensor_storage_bytes(torch: Any, tensor: Any) -> bytes:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(order="C")
    return cast(bytes, raw)


def _tensor_hash(torch: Any, tensor: Any) -> str:
    return hashlib.sha256(_tensor_storage_bytes(torch, tensor)).hexdigest()


def _storage_pointer(tensor: Any) -> int:
    if hasattr(tensor, "untyped_storage"):
        return int(tensor.untyped_storage().data_ptr())
    return int(tensor.data_ptr())


def _materialize_arm(
    torch: Any, suite: Suite, case: Case, arm_id: str, suite_sha256: str, device_index: int
) -> tuple[dict[str, Any], TensorMaterialization]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.input_seed)
    tensors: dict[str, Any] = {}
    descriptors: list[Mapping[str, object]] = []
    pointers: set[int] = set()
    schedule = _resolve_draw_schedule(suite, case)
    specs = {tensor.id: tensor for tensor in suite.tensors}
    for draw in schedule:
        cpu_fp32 = torch.randn(draw.shape, generator=generator, dtype=torch.float32, device="cpu")
        if draw.normal_scale != 1.0:
            cpu_fp32 = cpu_fp32 * draw.normal_scale
        if draw.normal_offset:
            cpu_fp32 = cpu_fp32 + draw.normal_offset
        cpu_bf16 = cpu_fp32.to(dtype=torch.bfloat16).contiguous()
        storage_sha256 = _tensor_hash(torch, cpu_bf16)
        cuda_tensor = cpu_bf16.to(
            device=f"cuda:{device_index}", dtype=torch.bfloat16, non_blocking=False
        ).contiguous()
        pointer = _storage_pointer(cuda_tensor)
        spec = specs[draw.tensor_id]
        if (
            tuple(cuda_tensor.shape) != draw.shape
            or cuda_tensor.dtype != torch.bfloat16
            or str(cuda_tensor.device) != f"cuda:{device_index}"
            or cuda_tensor.layout != torch.strided
            or not bool(cuda_tensor.is_contiguous())
            or pointer % spec.alignment
        ):
            raise _ExecutionValidationError(
                "materialization",
                f"materialized tensor {draw.tensor_id!r} violates its frozen storage contract",
            )
        if pointer in pointers:
            raise _ExecutionValidationError(
                "input_alias", "materialized tensors do not have disjoint storage"
            )
        pointers.add(pointer)
        tensors[draw.tensor_id] = cuda_tensor
        descriptors.append(
            {
                "tensor_id": draw.tensor_id,
                "role": draw.role,
                "shape": list(draw.shape),
                "draw": "normal_0_1_fp32_cpu",
                "normal_scale": draw.normal_scale,
                "normal_offset": draw.normal_offset,
                "cpu_dtype": "float32",
                "storage_dtype": "bfloat16",
                "device": f"cuda:{device_index}",
                "contiguous": True,
                "alignment_bytes": spec.alignment,
                "alignment_satisfied": True,
                "storage_sha256": storage_sha256,
            }
        )
    torch.cuda.synchronize(device_index)
    record = TensorMaterialization(
        suite_sha256,
        case.id,
        arm_id,
        case.input_seed,
        tuple(draw.tensor_id for draw in schedule),
        tuple(descriptors),
    )
    return tensors, record


def _gated_mlp_reference(torch: Any, x: Any, gate_weight: Any, up_weight: Any) -> Any:
    gate = torch.mm(x.float(), gate_weight.float().T)
    up = torch.mm(x.float(), up_weight.float().T)
    return (torch.nn.functional.silu(gate, inplace=False) * up).to(dtype=torch.bfloat16)


def _gated_mlp_candidate(torch: Any, x: Any, gate_weight: Any, up_weight: Any) -> Any:
    gate = torch.mm(x, gate_weight.T, out_dtype=torch.float32)
    up = torch.mm(x, up_weight.T, out_dtype=torch.float32)
    return (torch.nn.functional.silu(gate, inplace=False) * up).to(dtype=torch.bfloat16)


def _residual_rmsnorm(torch: Any, x: Any, residual: Any, gamma: Any, epsilon: float) -> Any:
    z = x.float() + residual.float()
    mean_square = torch.mean(z * z, dim=-1, keepdim=True, dtype=torch.float32)
    return (z * torch.rsqrt(mean_square + epsilon) * gamma.float()).to(dtype=torch.bfloat16)


def _make_kernel(torch: Any, entrypoint: str, semantics: object) -> Callable[..., Any]:
    kind = _ENTRYPOINTS.get(entrypoint)
    if kind is None:
        raise SchemaError(f"entrypoint {entrypoint!r} is not in the closed local registry")
    if kind == "gated_mlp_reference":
        return lambda x, gate_weight, up_weight: _gated_mlp_reference(
            torch, x, gate_weight, up_weight
        )
    if kind == "gated_mlp_candidate":
        return lambda x, gate_weight, up_weight: _gated_mlp_candidate(
            torch, x, gate_weight, up_weight
        )
    if not isinstance(semantics, RMSNormSemantics):
        raise SchemaError("RMSNorm entrypoint requires frozen RMSNorm semantics")
    epsilon = semantics.epsilon
    return lambda x, residual, gamma: _residual_rmsnorm(torch, x, residual, gamma, epsilon)


def _kernel_arguments(suite: Suite, tensors: Mapping[str, Any]) -> tuple[Any, ...]:
    if suite.template_id == "gated_mlp_epilogue.v1":
        return tensors["input"], tensors["gate_weight"], tensors["up_weight"]
    return tensors["input"], tensors["residual"], tensors["gamma"]


def _correctness_gate_key(suite_sha256: str, case: Case, cell: ExpectedCell) -> str:
    fields = (
        "heliostune.correctness-key/1",
        suite_sha256,
        case.id,
        cell.arm_id,
        str(cell.input_seed),
        case.numeric_contract_id,
        "highest|tf32=0|bf16rr=0|fp16rr=0|fp16acc=0",
    )
    return hashlib.sha256("\0".join(fields).encode()).hexdigest()


def _timing_gate_allows(correctness_key: str, retained_passes: set[str]) -> bool:
    return correctness_key in retained_passes


def _output_descriptor(actual: Any) -> dict[str, object]:
    return {
        "shape": list(actual.shape),
        "device": str(actual.device),
        "dtype": str(actual.dtype),
        "layout": str(actual.layout),
        "contiguous": bool(actual.is_contiguous()),
    }


def _validate_correctness(
    torch: Any,
    *,
    actual: Any,
    expected: Any,
    inputs: Mapping[str, Any],
    before_hashes: Mapping[str, str],
    expected_shape: tuple[int, ...],
    atol: float,
    rtol: float,
    correctness_key: str,
) -> CorrectnessObservation:
    descriptor: Mapping[str, object] | None = None
    unchanged = disjoint = finite = close = False
    max_abs_error: float | None = None
    try:
        descriptor = _output_descriptor(actual)
        if tuple(actual.shape) != expected_shape:
            raise _ExecutionValidationError("shape", "output shape does not match the suite")
        if str(actual.device) != str(expected.device):
            raise _ExecutionValidationError("device", "output device does not match reference")
        if actual.dtype != torch.bfloat16:
            raise _ExecutionValidationError("dtype", "output dtype is not bfloat16")
        if actual.layout != torch.strided or not bool(actual.is_contiguous()):
            raise _ExecutionValidationError("layout", "output is not contiguous strided layout")
        output_pointer = _storage_pointer(actual)
        disjoint = all(output_pointer != _storage_pointer(item) for item in inputs.values())
        if not disjoint:
            raise _ExecutionValidationError("alias", "output aliases an input storage")
        unchanged = all(
            _tensor_hash(torch, item) == before_hashes[name] for name, item in inputs.items()
        )
        if not unchanged:
            raise _ExecutionValidationError("mutation", "an input tensor was mutated")
        finite = bool(torch.isfinite(actual).all().item())
        if not finite:
            raise _ExecutionValidationError("nonfinite", "output contains a non-finite value")
        try:
            max_abs_error = float(torch.max(torch.abs(actual.float() - expected.float())).item())
        except Exception:
            max_abs_error = None
        try:
            torch.testing.assert_close(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
                equal_nan=False,
                check_device=True,
                check_dtype=True,
                check_layout=True,
                check_stride=True,
            )
        except AssertionError as exc:
            raise _ExecutionValidationError("tolerance", _safe_error(exc)) from exc
        close = True
        return CorrectnessObservation(
            "passed",
            correctness_key,
            None,
            None,
            descriptor,
            unchanged,
            disjoint,
            finite,
            close,
            max_abs_error,
        )
    except _ExecutionValidationError as exc:
        return CorrectnessObservation(
            "failed",
            correctness_key,
            exc.kind,
            str(exc),
            descriptor,
            unchanged,
            disjoint,
            finite,
            close,
            max_abs_error,
        )


@contextlib.contextmanager
def _precision_flags(torch: Any) -> Iterator[None]:
    matmul = torch.backends.cuda.matmul
    old_precision = torch.get_float32_matmul_precision()
    attributes = (
        "allow_tf32",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    )
    old_values = {name: getattr(matmul, name) for name in attributes}
    try:
        torch.set_float32_matmul_precision("highest")
        matmul.allow_tf32 = False
        matmul.allow_bf16_reduced_precision_reduction = False
        matmul.allow_fp16_reduced_precision_reduction = False
        matmul.allow_fp16_accumulation = False
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        for name, value in old_values.items():
            setattr(matmul, name, value)


@contextlib.contextmanager
def _cuda_autocast_disabled(torch: Any) -> Iterator[None]:
    with torch.autocast(device_type="cuda", enabled=False):
        yield


def _environment_schema(capability: CapabilityProbe) -> dict[str, object]:
    return {
        "schema": "heliostune.local-environment/1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": capability.torch_version,
        "cuda_version": capability.cuda_version,
        "rocm_version": capability.rocm_version,
        "device_index": capability.device_index,
        "device_name": capability.device_name,
        "compute_capability": None
        if capability.compute_capability is None
        else list(capability.compute_capability),
        "precision_policy": dict(_PRECISION_POLICY),
        "autocast_policy": dict(_AUTOCAST_POLICY),
        "backend_invoked": None,
        "fusion_claim": False,
    }


def _failed_cell(cell: ExpectedCell, key: str, kind: str, message: str) -> CellObservation:
    if cell.stage == "correctness":
        correctness = CorrectnessObservation(
            "failed", key, kind, message, None, False, False, False, False, None
        )
        return CellObservation(
            cell.id, cell.case_id, cell.arm_id, cell.stage, "failed", correctness, None
        )
    timing = TimingObservation("failed", key, kind, message, 0, 0, (), None)
    return CellObservation(cell.id, cell.case_id, cell.arm_id, cell.stage, "failed", None, timing)


def _record_transition(
    attempts: list[Mapping[str, object]],
    states: dict[str, str],
    cell: ExpectedCell,
    target: str,
    reason: str | None = None,
) -> None:
    source = states[cell.id]
    states[cell.id] = _advance_cell_state(source, target)
    status = {"running": "running", "passed": "success", "failed": "failure"}[target]
    attempts.append(
        {
            "attempt_id": len(attempts) + 1,
            "cell_id": cell.id,
            "stage": cell.stage,
            "status": status,
            "from_state": source,
            "to_state": target,
            "reason": reason,
        }
    )


def _with_timing_failure(
    timing: TimingObservation, *, failure_kind: str, message: str
) -> TimingObservation:
    return TimingObservation(
        "failed",
        timing.correctness_key,
        failure_kind,
        message,
        timing.warmups,
        0,
        (),
        None,
    )


def _timing_observation(
    torch: Any,
    *,
    kernel: Callable[..., Any],
    arguments: Sequence[Any],
    device_index: int,
    correctness_key: str,
    warmups: int,
    repetitions: int,
) -> TimingObservation:
    samples: list[float] = []
    try:
        for _ in range(warmups):
            kernel(*arguments)
        torch.cuda.synchronize(device_index)
    except Exception as exc:
        return TimingObservation(
            "failed", correctness_key, "warmup", _safe_error(exc), warmups, 0, (), None
        )
    try:
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            kernel(*arguments)
            end.record()
            torch.cuda.synchronize(device_index)
            samples.append(float(start.elapsed_time(end)))
    except Exception as exc:
        return TimingObservation(
            "failed",
            correctness_key,
            "timing",
            _safe_error(exc),
            warmups,
            0,
            (),
            None,
        )
    return TimingObservation(
        "passed",
        correctness_key,
        None,
        None,
        warmups,
        repetitions,
        tuple(samples),
        statistics.median(samples),
    )


def _force_eager_reason(torch: Any) -> str | None:
    disabled = os.environ.get("TORCHDYNAMO_DISABLE")
    if disabled is not None and disabled.strip().lower() not in {"", "0", "false", "no", "off"}:
        return "TORCHDYNAMO_DISABLE requests eager execution"
    config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if bool(getattr(config, "disable", False)):
        return "torch._dynamo.config.disable requests eager execution"
    if bool(getattr(config, "suppress_errors", False)):
        return "torch._dynamo.config.suppress_errors permits eager fallback"
    return None


def _lookup_inductor_backend(torch: Any) -> Callable[..., Any]:
    registry = getattr(getattr(getattr(torch, "_dynamo", None), "backends", None), "registry", None)
    if registry is None:
        registry = importlib.import_module("torch._dynamo.backends.registry")
    backend = registry.lookup_backend("inductor")
    if not callable(backend):
        raise RuntimeError("the pinned Inductor backend is not callable")
    return cast(Callable[..., Any], backend)


def _compile_candidate(
    torch: Any,
    kernel: Callable[..., Any],
    backend_state: dict[str, bool] | None = None,
) -> Callable[..., Any]:
    reason = _force_eager_reason(torch)
    if reason is not None:
        raise RuntimeError(reason)
    state = {"invoked": False} if backend_state is None else backend_state
    state["invoked"] = False

    def recording_inductor_backend(graph_module: Any, example_inputs: Sequence[Any]) -> Any:
        state["invoked"] = True
        return _lookup_inductor_backend(torch)(graph_module, example_inputs)

    compiled = torch.compile(
        kernel,
        backend=recording_inductor_backend,
        fullgraph=True,
        dynamic=False,
        mode="default",
    )
    if not callable(compiled):
        raise RuntimeError("torch.compile did not return a callable")
    if compiled is kernel:
        raise RuntimeError("torch.compile returned the original eager callable")
    return cast(Callable[..., Any], compiled)


def _first_candidate_call(
    torch: Any,
    kernel: Callable[..., Any],
    arguments: Sequence[Any],
    device_index: int,
    backend_state: Mapping[str, bool],
) -> Any:
    try:
        actual = kernel(*arguments)
        torch.cuda.synchronize(device_index)
    except Exception as exc:
        raise _ExecutionValidationError("compile_failed", _safe_error(exc)) from exc
    if not backend_state.get("invoked", False):
        raise _ExecutionValidationError(
            "compile_failed",
            "compiled callable completed without invoking the Inductor backend",
        )
    return actual


def _summary(
    suite: Suite, observations: Sequence[CellObservation], outcome: str
) -> dict[str, object]:
    terminal_ids = [item.cell_id for item in observations]
    return {
        "expected_cell_ids": [cell.id for cell in suite.expected_cells],
        "terminal_cell_ids": terminal_ids,
        "passed": sum(item.status == "passed" for item in observations),
        "failed": sum(item.status == "failed" for item in observations),
        "blocked": sum(item.status == "blocked" for item in observations),
        "all_cells_terminal": terminal_ids == [cell.id for cell in suite.expected_cells],
        "outcome": outcome,
        "fusion_claim": False,
    }


def run_local_suite(suite_path: str | Path) -> LocalExecutionResult:
    verified = verify_suite(suite_path)
    suite = verified.suite
    _validate_frozen_suite(suite, suite_sha256=verified.sha256)
    capability = probe_local_capability(suite)
    environment = _environment_schema(capability)
    states = {cell.id: "pending" for cell in suite.expected_cells}
    attempts: list[Mapping[str, object]] = []
    case = suite.cases[0]
    keys = {
        cell.id: _correctness_gate_key(verified.sha256, case, cell) for cell in suite.expected_cells
    }
    if not capability.available:
        summary = _summary(suite, (), "aborted")
        summary["capability_reasons"] = list(capability.reasons)
        return LocalExecutionResult(
            str(verified.path),
            verified.sha256,
            verified.bytes,
            suite.suite_id,
            capability,
            (),
            (),
            (),
            environment,
            {},
            summary,
            "aborted",
        )

    torch = importlib.import_module("torch")
    device_index = cast(int, capability.device_index)
    arms = {arm.id: arm for arm in suite.arms}
    correctness_policy = suite.correctness_policies[0]
    timing_policy = suite.timing_policies[0]
    output_spec = next(item for item in suite.tensors if item.role == "output")
    output_shape = tuple(case.shape_dict[name] for name in output_spec.shape)
    retained_inputs: dict[str, dict[str, Any]] = {}
    retained_kernels: dict[str, Callable[..., Any]] = {}
    retained_passes: set[str] = set()
    materialization: list[TensorMaterialization] = []
    observations: list[CellObservation] = []
    compile_outcomes: dict[str, Mapping[str, object]] = {}
    all_arm_pointers: set[int] = set()

    try:
        with _precision_flags(torch), _cuda_autocast_disabled(torch):
            for cell in suite.expected_cells:
                arm = arms[cell.arm_id]
                key = keys[cell.id]
                if cell.stage == "timing":
                    if not _timing_gate_allows(key, retained_passes):
                        _record_transition(attempts, states, cell, "running")
                        _record_transition(attempts, states, cell, "failed", "correctness_gate")
                        observations.append(
                            _failed_cell(
                                cell,
                                key,
                                "correctness_gate",
                                "no retained passing correctness observation for the exact key",
                            )
                        )
                        continue
                    _record_transition(attempts, states, cell, "running")
                    try:
                        inputs = retained_inputs[arm.id]
                        before = {name: _tensor_hash(torch, item) for name, item in inputs.items()}
                        timing = _timing_observation(
                            torch,
                            kernel=retained_kernels[arm.id],
                            arguments=_kernel_arguments(suite, inputs),
                            device_index=device_index,
                            correctness_key=key,
                            warmups=timing_policy.warmups,
                            repetitions=timing_policy.repetitions,
                        )
                        if any(
                            _tensor_hash(torch, item) != before[name]
                            for name, item in inputs.items()
                        ):
                            timing = _with_timing_failure(
                                timing,
                                failure_kind="mutation",
                                message="an input tensor was mutated during timing",
                            )
                    except Exception as exc:
                        timing = TimingObservation(
                            "failed",
                            key,
                            "execution",
                            _safe_error(exc),
                            0,
                            0,
                            (),
                            None,
                        )
                    _record_transition(
                        attempts,
                        states,
                        cell,
                        "passed" if timing.status == "passed" else "failed",
                        timing.failure_kind,
                    )
                    observations.append(
                        CellObservation(
                            cell.id,
                            cell.case_id,
                            cell.arm_id,
                            cell.stage,
                            timing.status,
                            None,
                            timing,
                        )
                    )
                    continue

                _record_transition(attempts, states, cell, "running")
                try:
                    inputs, materialized = _materialize_arm(
                        torch, suite, case, arm.id, verified.sha256, device_index
                    )
                    pointers = {_storage_pointer(item) for item in inputs.values()}
                    if pointers & all_arm_pointers:
                        raise _ExecutionValidationError(
                            "input_alias", "arm materializations do not have disjoint storage"
                        )
                    all_arm_pointers.update(pointers)
                    materialization.append(materialized)
                    retained_inputs[arm.id] = inputs
                    eager_kernel = _make_kernel(torch, arm.entrypoint, case.semantics)
                    backend_state: dict[str, bool] | None = None
                    if arm.role == "candidate":
                        backend_state = {"invoked": False}
                        compile_started = time.perf_counter_ns()
                        base_outcome: dict[str, object] = {
                            "case_id": case.id,
                            "arm_id": arm.id,
                            "entrypoint": arm.entrypoint,
                            "status": "compile_failed",
                            "error": None,
                            "wrapper_create_ns": None,
                            "first_call_ns": None,
                            "eager_fallback": False,
                            "backend_invoked": False,
                            "callable_distinct": False,
                            "autocast_policy": dict(_AUTOCAST_POLICY),
                        }
                        try:
                            kernel = _compile_candidate(torch, eager_kernel, backend_state)
                        except Exception as exc:
                            base_outcome.update(
                                error=_safe_error(exc),
                                wrapper_create_ns=time.perf_counter_ns() - compile_started,
                                backend_invoked=backend_state["invoked"],
                            )
                            compile_outcomes[arm.id] = base_outcome
                            raise _ExecutionValidationError(
                                "compile_failed", _safe_error(exc)
                            ) from exc
                        base_outcome.update(
                            status="wrapper_created",
                            wrapper_create_ns=time.perf_counter_ns() - compile_started,
                            callable_distinct=True,
                        )
                        compile_outcomes[arm.id] = base_outcome
                    else:
                        kernel = eager_kernel
                    retained_kernels[arm.id] = kernel
                    arguments = _kernel_arguments(suite, inputs)
                    before_hashes = {
                        name: _tensor_hash(torch, item) for name, item in inputs.items()
                    }
                    oracle = _make_kernel(
                        torch,
                        "reference_template.gated_mlp_reference"
                        if isinstance(case.semantics, GatedMLPSemantics)
                        else "reference_template.residual_rmsnorm_reference",
                        case.semantics,
                    )
                    expected = oracle(*arguments)
                    first_started = time.perf_counter_ns()
                    try:
                        if arm.role == "candidate":
                            assert backend_state is not None
                            actual = _first_candidate_call(
                                torch, kernel, arguments, device_index, backend_state
                            )
                        else:
                            actual = kernel(*arguments)
                            torch.cuda.synchronize(device_index)
                    except Exception as exc:
                        if arm.role == "candidate":
                            prior = dict(compile_outcomes[arm.id])
                            prior.update(
                                status="compile_failed",
                                error=str(exc)
                                if isinstance(exc, _ExecutionValidationError)
                                else _safe_error(exc),
                                first_call_ns=time.perf_counter_ns() - first_started,
                                backend_invoked=bool(backend_state and backend_state["invoked"]),
                            )
                            compile_outcomes[arm.id] = prior
                        if isinstance(exc, _ExecutionValidationError):
                            raise
                        raise _ExecutionValidationError("runtime", _safe_error(exc)) from exc
                    if arm.role == "candidate":
                        prior = dict(compile_outcomes[arm.id])
                        prior.update(
                            status="compiled_and_first_call_completed",
                            first_call_ns=time.perf_counter_ns() - first_started,
                            backend_invoked=True,
                        )
                        compile_outcomes[arm.id] = prior
                    correctness = _validate_correctness(
                        torch,
                        actual=actual,
                        expected=expected,
                        inputs=inputs,
                        before_hashes=before_hashes,
                        expected_shape=output_shape,
                        atol=correctness_policy.atol,
                        rtol=correctness_policy.rtol,
                        correctness_key=key,
                    )
                except _ExecutionValidationError as exc:
                    correctness = CorrectnessObservation(
                        "failed", key, exc.kind, str(exc), None, False, False, False, False, None
                    )
                except Exception as exc:
                    correctness = CorrectnessObservation(
                        "failed",
                        key,
                        "runtime",
                        _safe_error(exc),
                        None,
                        False,
                        False,
                        False,
                        False,
                        None,
                    )
                if correctness.status == "passed":
                    retained_passes.add(key)
                _record_transition(
                    attempts,
                    states,
                    cell,
                    "passed" if correctness.status == "passed" else "failed",
                    correctness.failure_kind,
                )
                observations.append(
                    CellObservation(
                        cell.id,
                        cell.case_id,
                        cell.arm_id,
                        cell.stage,
                        correctness.status,
                        correctness,
                        None,
                    )
                )
    except Exception as exc:
        message = _safe_error(exc)
        for cell in suite.expected_cells:
            if states[cell.id] == "pending":
                _record_transition(attempts, states, cell, "running")
                _record_transition(attempts, states, cell, "failed", "executor")
                observations.append(_failed_cell(cell, keys[cell.id], "executor", message))

    environment["backend_invoked"] = any(
        bool(item.get("backend_invoked")) for item in compile_outcomes.values()
    )

    typed_outcome: Literal["completed", "failed", "aborted"] = (
        "completed" if all(item.status == "passed" for item in observations) else "failed"
    )
    return LocalExecutionResult(
        str(verified.path),
        verified.sha256,
        verified.bytes,
        suite.suite_id,
        capability,
        tuple(materialization),
        tuple(observations),
        tuple(attempts),
        environment,
        compile_outcomes,
        _summary(suite, observations, typed_outcome),
        typed_outcome,
    )


__all__ = [
    "GATED_MLP_SUITE_SHA256",
    "RMSNORM_SUITE_SHA256",
    "CapabilityProbe",
    "TensorMaterialization",
    "CorrectnessObservation",
    "TimingObservation",
    "CellObservation",
    "LocalExecutionResult",
    "probe_local_capability",
    "run_local_suite",
]
