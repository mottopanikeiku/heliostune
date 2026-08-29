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

from .artifacts import strict_json_loads
from .errors import SchemaError
from .scope import Case, ExpectedCell, GatedMLPSemantics, RMSNormSemantics, Suite, verify_suite
from .validation import (
    exact_bool,
    exact_fields,
    exact_int,
    exact_object,
    finite_float,
    integer_pair,
    nonblank_string,
    optional_finite_float,
    optional_nonblank_string,
)

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

    @classmethod
    def from_dict(cls, value: object) -> CapabilityProbe:
        data = exact_fields(
            value,
            required=(
                "available",
                "reasons",
                "torch_version",
                "cuda_version",
                "rocm_version",
                "device_index",
                "device_name",
                "compute_capability",
                "native_bf16",
                "inductor_available",
                "allocation_succeeded",
                "detail",
            ),
            context="local capability probe",
        )
        reasons = tuple(
            cast(
                CapabilityReason,
                _enum(item, _CAPABILITY_REASONS, "local capability reason"),
            )
            for item in _array(data["reasons"], "local capability reasons")
        )
        if len(set(reasons)) != len(reasons):
            raise SchemaError("local capability reasons must not contain duplicates")
        compute_capability = (
            None
            if data["compute_capability"] is None
            else integer_pair(
                data["compute_capability"], context="local capability compute_capability"
            )
        )
        result = cls(
            exact_bool(data["available"], context="local capability available"),
            reasons,
            optional_nonblank_string(
                data["torch_version"], context="local capability torch_version"
            ),
            optional_nonblank_string(data["cuda_version"], context="local capability cuda_version"),
            optional_nonblank_string(data["rocm_version"], context="local capability rocm_version"),
            _optional_int(data["device_index"], context="local capability device_index", minimum=0),
            optional_nonblank_string(data["device_name"], context="local capability device_name"),
            compute_capability,
            _optional_bool(data["native_bf16"], context="local capability native_bf16"),
            _optional_bool(
                data["inductor_available"],
                context="local capability inductor_available",
            ),
            exact_bool(
                data["allocation_succeeded"],
                context="local capability allocation_succeeded",
            ),
            optional_nonblank_string(data["detail"], context="local capability detail"),
        )
        _validate_capability_probe(result)
        return result


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

    @classmethod
    def from_dict(cls, value: object) -> TensorMaterialization:
        data = exact_fields(
            value,
            required=(
                "suite_sha256",
                "case_id",
                "arm_id",
                "input_seed",
                "tensor_order",
                "tensors",
            ),
            context="tensor materialization",
        )
        return cls(
            _digest(data["suite_sha256"], "tensor materialization suite_sha256"),
            nonblank_string(data["case_id"], context="tensor materialization case_id"),
            nonblank_string(data["arm_id"], context="tensor materialization arm_id"),
            exact_int(data["input_seed"], context="tensor materialization input_seed", minimum=0),
            tuple(
                nonblank_string(item, context="tensor materialization tensor_order item")
                for item in _array(data["tensor_order"], "tensor materialization tensor_order")
            ),
            tuple(
                _parse_materialization_descriptor(item)
                for item in _array(data["tensors"], "tensor materialization tensors")
            ),
        )


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

    @classmethod
    def from_dict(cls, value: object) -> CorrectnessObservation:
        data = exact_fields(
            value,
            required=(
                "status",
                "correctness_key",
                "failure_kind",
                "message",
                "output",
                "input_storage_unchanged",
                "output_disjoint",
                "finite",
                "close",
                "max_abs_error",
            ),
            context="correctness observation",
        )
        result = cls(
            cast(CellStatus, _enum(data["status"], _CELL_STATUSES, "correctness status")),
            _digest(data["correctness_key"], "correctness observation correctness_key"),
            optional_nonblank_string(
                data["failure_kind"], context="correctness observation failure_kind"
            ),
            optional_nonblank_string(data["message"], context="correctness observation message"),
            None if data["output"] is None else _parse_output_descriptor(data["output"]),
            exact_bool(
                data["input_storage_unchanged"],
                context="correctness observation input_storage_unchanged",
            ),
            exact_bool(
                data["output_disjoint"],
                context="correctness observation output_disjoint",
            ),
            exact_bool(data["finite"], context="correctness observation finite"),
            exact_bool(data["close"], context="correctness observation close"),
            optional_finite_float(
                data["max_abs_error"],
                context="correctness observation max_abs_error",
                minimum=0,
            ),
        )
        _validate_failure_evidence(result, "standalone")
        evidence = (
            result.input_storage_unchanged,
            result.output_disjoint,
            result.finite,
            result.close,
        )
        if result.status == "passed" and (
            result.output is None
            or result.max_abs_error is None
            or evidence != (True, True, True, True)
        ):
            raise SchemaError("passing correctness observation lacks exact passing evidence")
        if result.status != "passed" and all(evidence):
            raise SchemaError("failed correctness observation masquerades as passing")
        return result


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

    @classmethod
    def from_dict(cls, value: object) -> TimingObservation:
        data = exact_fields(
            value,
            required=(
                "status",
                "correctness_key",
                "failure_kind",
                "message",
                "warmups",
                "repetitions",
                "samples_ms",
                "median_ms",
            ),
            context="timing observation",
        )
        result = cls(
            cast(CellStatus, _enum(data["status"], _CELL_STATUSES, "timing status")),
            _digest(data["correctness_key"], "timing observation correctness_key"),
            optional_nonblank_string(
                data["failure_kind"], context="timing observation failure_kind"
            ),
            optional_nonblank_string(data["message"], context="timing observation message"),
            exact_int(data["warmups"], context="timing observation warmups", minimum=0),
            exact_int(data["repetitions"], context="timing observation repetitions", minimum=0),
            tuple(
                finite_float(
                    item,
                    context="timing observation samples_ms item",
                    strictly_positive=True,
                )
                for item in _array(data["samples_ms"], "timing observation samples_ms")
            ),
            optional_finite_float(
                data["median_ms"],
                context="timing observation median_ms",
                strictly_positive=True,
            ),
        )
        _validate_failure_evidence(result, "standalone")
        if result.status == "passed":
            if (
                result.repetitions == 0
                or len(result.samples_ms) != result.repetitions
                or result.median_ms is None
                or not math.isclose(
                    result.median_ms,
                    statistics.median(result.samples_ms),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise SchemaError("passing timing observation has inconsistent timing evidence")
        elif result.repetitions != 0 or result.samples_ms or result.median_ms is not None:
            raise SchemaError("failed timing observation contains positive timing evidence")
        return result


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

    @classmethod
    def from_dict(cls, value: object) -> CellObservation:
        data = exact_fields(
            value,
            required=(
                "cell_id",
                "case_id",
                "arm_id",
                "stage",
                "status",
                "correctness",
                "timing",
            ),
            context="cell observation",
        )
        result = cls(
            nonblank_string(data["cell_id"], context="cell observation cell_id"),
            nonblank_string(data["case_id"], context="cell observation case_id"),
            nonblank_string(data["arm_id"], context="cell observation arm_id"),
            cast(
                Literal["correctness", "timing"],
                _enum(data["stage"], _CELL_STAGES, "cell observation stage"),
            ),
            cast(CellStatus, _enum(data["status"], _CELL_STATUSES, "cell observation status")),
            None
            if data["correctness"] is None
            else CorrectnessObservation.from_dict(data["correctness"]),
            None if data["timing"] is None else TimingObservation.from_dict(data["timing"]),
        )
        nested = result.correctness if result.stage == "correctness" else result.timing
        other = result.timing if result.stage == "correctness" else result.correctness
        if nested is None or other is not None:
            raise SchemaError("cell observation has the wrong nested record for its stage")
        if nested.status != result.status:
            raise SchemaError("cell observation status does not match its nested evidence")
        return result


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

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        verified_suite_path: str,
        verified_suite_sha256: str,
        verified_suite_bytes: bytes,
    ) -> LocalExecutionResult:
        logical_path = nonblank_string(verified_suite_path, context="verified local suite path")
        digest = _digest(verified_suite_sha256, "verified local suite SHA-256")
        if type(verified_suite_bytes) is not bytes:
            raise SchemaError("verified local suite bytes must be bytes")
        if hashlib.sha256(verified_suite_bytes).hexdigest() != digest:
            raise SchemaError("verified local suite SHA-256 does not match its exact bytes")
        try:
            suite_text = verified_suite_bytes.decode("utf-8")
        except UnicodeError as exc:
            raise SchemaError("verified local suite bytes must be UTF-8") from exc
        suite = Suite.from_dict(strict_json_loads(suite_text, source=Path(logical_path)))
        _validate_frozen_suite(suite, suite_sha256=digest)

        data = exact_fields(
            value,
            required=(
                "verified_suite_path",
                "verified_suite_sha256",
                "suite_id",
                "capability",
                "materialization",
                "observations",
                "attempts",
                "environment",
                "compile_outcomes",
                "summary",
                "outcome",
            ),
            context="local execution result",
        )
        nonblank_string(data["verified_suite_path"], context="serialized verified suite path")
        if _digest(data["verified_suite_sha256"], "serialized verified suite SHA-256") != digest:
            raise SchemaError("serialized local suite SHA-256 does not match verified suite")
        if nonblank_string(data["suite_id"], context="serialized suite_id") != suite.suite_id:
            raise SchemaError("serialized local suite_id does not match verified suite")

        capability = CapabilityProbe.from_dict(data["capability"])
        materialization = tuple(
            TensorMaterialization.from_dict(item)
            for item in _array(data["materialization"], "local materialization")
        )
        observations = tuple(
            CellObservation.from_dict(item)
            for item in _array(data["observations"], "local observations")
        )
        attempts = tuple(
            _parse_attempt(item) for item in _array(data["attempts"], "local attempts")
        )
        environment = _parse_environment(data["environment"], capability)
        compile_outcomes = _parse_compile_outcomes(data["compile_outcomes"], suite)
        outcome = cast(
            Literal["completed", "failed", "aborted"],
            _enum(data["outcome"], _OUTCOMES, "local execution outcome"),
        )
        summary = _parse_summary(data["summary"], suite, observations, capability, outcome)
        _validate_deserialized_result(
            suite=suite,
            suite_sha256=digest,
            capability=capability,
            materialization=materialization,
            observations=observations,
            attempts=attempts,
            environment=environment,
            compile_outcomes=compile_outcomes,
            summary=summary,
            outcome=outcome,
        )
        return cls(
            logical_path,
            digest,
            verified_suite_bytes,
            suite.suite_id,
            capability,
            materialization,
            observations,
            attempts,
            environment,
            compile_outcomes,
            summary,
            outcome,
        )


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


_CAPABILITY_REASONS = (
    "torch_missing",
    "torch_version_mismatch",
    "cuda_unavailable",
    "rocm_unsupported",
    "compute_capability_too_low",
    "bf16_unsupported",
    "inductor_unavailable",
    "allocation_failed",
    "device_probe_failed",
)
_CELL_STATUSES = ("passed", "failed", "blocked")
_CELL_STAGES = ("correctness", "timing")
_OUTCOMES = ("completed", "failed", "aborted")
_MATERIALIZATION_DESCRIPTOR_FIELDS = (
    "tensor_id",
    "role",
    "shape",
    "draw",
    "normal_scale",
    "normal_offset",
    "cpu_dtype",
    "storage_dtype",
    "device",
    "contiguous",
    "alignment_bytes",
    "alignment_satisfied",
    "storage_sha256",
)
_COMPILE_OUTCOME_FIELDS = (
    "case_id",
    "arm_id",
    "entrypoint",
    "status",
    "error",
    "wrapper_create_ns",
    "first_call_ns",
    "eager_fallback",
    "backend_invoked",
    "callable_distinct",
    "autocast_policy",
)


def _array(value: object, context: str) -> list[object]:
    if type(value) is not list:
        raise SchemaError(f"{context} must be an array")
    return cast(list[object], value)


def _enum(value: object, allowed: tuple[str, ...], context: str) -> str:
    result = nonblank_string(value, context=context)
    if result not in allowed:
        raise SchemaError(f"unknown {context} {result!r}")
    return result


def _digest(value: object, context: str) -> str:
    result = nonblank_string(value, context=context)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise SchemaError(f"{context} must be a lowercase 64-hex SHA-256 digest")
    return result


def _optional_bool(value: object, *, context: str) -> bool | None:
    return None if value is None else exact_bool(value, context=context)


def _optional_int(value: object, *, context: str, minimum: int = 0) -> int | None:
    return None if value is None else exact_int(value, context=context, minimum=minimum)


def _parse_materialization_descriptor(value: object) -> Mapping[str, object]:
    data = exact_fields(
        value,
        required=_MATERIALIZATION_DESCRIPTOR_FIELDS,
        context="tensor materialization descriptor",
    )
    return {
        "tensor_id": nonblank_string(
            data["tensor_id"], context="tensor materialization descriptor tensor_id"
        ),
        "role": nonblank_string(data["role"], context="tensor materialization descriptor role"),
        "shape": [
            exact_int(
                item,
                context="tensor materialization descriptor shape item",
                minimum=1,
            )
            for item in _array(data["shape"], "tensor materialization descriptor shape")
        ],
        "draw": nonblank_string(data["draw"], context="tensor materialization descriptor draw"),
        "normal_scale": finite_float(
            data["normal_scale"],
            context="tensor materialization descriptor normal_scale",
            strictly_positive=True,
        ),
        "normal_offset": finite_float(
            data["normal_offset"],
            context="tensor materialization descriptor normal_offset",
        ),
        "cpu_dtype": nonblank_string(
            data["cpu_dtype"], context="tensor materialization descriptor cpu_dtype"
        ),
        "storage_dtype": nonblank_string(
            data["storage_dtype"],
            context="tensor materialization descriptor storage_dtype",
        ),
        "device": nonblank_string(
            data["device"], context="tensor materialization descriptor device"
        ),
        "contiguous": exact_bool(
            data["contiguous"], context="tensor materialization descriptor contiguous"
        ),
        "alignment_bytes": exact_int(
            data["alignment_bytes"],
            context="tensor materialization descriptor alignment_bytes",
            minimum=1,
        ),
        "alignment_satisfied": exact_bool(
            data["alignment_satisfied"],
            context="tensor materialization descriptor alignment_satisfied",
        ),
        "storage_sha256": _digest(
            data["storage_sha256"],
            "tensor materialization descriptor storage_sha256",
        ),
    }


def _parse_output_descriptor(value: object) -> Mapping[str, object]:
    data = exact_fields(
        value,
        required=("shape", "device", "dtype", "layout", "contiguous"),
        context="correctness output descriptor",
    )
    return {
        "shape": [
            exact_int(item, context="correctness output shape item", minimum=1)
            for item in _array(data["shape"], "correctness output shape")
        ],
        "device": nonblank_string(data["device"], context="correctness output device"),
        "dtype": nonblank_string(data["dtype"], context="correctness output dtype"),
        "layout": nonblank_string(data["layout"], context="correctness output layout"),
        "contiguous": exact_bool(data["contiguous"], context="correctness output contiguous"),
    }


def _parse_attempt(value: object) -> Mapping[str, object]:
    data = exact_fields(
        value,
        required=(
            "attempt_id",
            "cell_id",
            "stage",
            "status",
            "from_state",
            "to_state",
            "reason",
        ),
        context="local execution attempt",
    )
    return {
        "attempt_id": exact_int(data["attempt_id"], context="local attempt attempt_id", minimum=1),
        "cell_id": nonblank_string(data["cell_id"], context="local attempt cell_id"),
        "stage": _enum(data["stage"], _CELL_STAGES, "local attempt stage"),
        "status": _enum(data["status"], ("running", "success", "failure"), "local attempt status"),
        "from_state": _enum(
            data["from_state"],
            ("pending", "running"),
            "local attempt from_state",
        ),
        "to_state": _enum(
            data["to_state"],
            ("running", "passed", "failed"),
            "local attempt to_state",
        ),
        "reason": optional_nonblank_string(data["reason"], context="local attempt reason"),
    }


def _parse_environment(value: object, capability: CapabilityProbe) -> Mapping[str, object]:
    data = exact_fields(
        value,
        required=(
            "schema",
            "python",
            "implementation",
            "platform",
            "torch_version",
            "cuda_version",
            "rocm_version",
            "device_index",
            "device_name",
            "compute_capability",
            "precision_policy",
            "autocast_policy",
            "backend_invoked",
            "fusion_claim",
        ),
        context="local execution environment",
    )
    precision = exact_fields(
        data["precision_policy"],
        required=(
            "float32_matmul_precision",
            "allow_tf32",
            "allow_bf16_reduced_precision_reduction",
            "allow_fp16_reduced_precision_reduction",
            "allow_fp16_accumulation",
        ),
        context="local precision policy",
    )
    parsed_precision: dict[str, object] = {
        "float32_matmul_precision": nonblank_string(
            precision["float32_matmul_precision"],
            context="local precision float32_matmul_precision",
        ),
        "allow_tf32": exact_bool(precision["allow_tf32"], context="local precision allow_tf32"),
        "allow_bf16_reduced_precision_reduction": exact_bool(
            precision["allow_bf16_reduced_precision_reduction"],
            context="local precision allow_bf16_reduced_precision_reduction",
        ),
        "allow_fp16_reduced_precision_reduction": exact_bool(
            precision["allow_fp16_reduced_precision_reduction"],
            context="local precision allow_fp16_reduced_precision_reduction",
        ),
        "allow_fp16_accumulation": exact_bool(
            precision["allow_fp16_accumulation"],
            context="local precision allow_fp16_accumulation",
        ),
    }
    autocast = _parse_autocast_policy(data["autocast_policy"])
    compute_capability = (
        None
        if data["compute_capability"] is None
        else integer_pair(
            data["compute_capability"],
            context="local environment compute_capability",
        )
    )
    result: dict[str, object] = {
        "schema": nonblank_string(data["schema"], context="local environment schema"),
        "python": nonblank_string(data["python"], context="local environment python"),
        "implementation": nonblank_string(
            data["implementation"], context="local environment implementation"
        ),
        "platform": nonblank_string(data["platform"], context="local environment platform"),
        "torch_version": optional_nonblank_string(
            data["torch_version"], context="local environment torch_version"
        ),
        "cuda_version": optional_nonblank_string(
            data["cuda_version"], context="local environment cuda_version"
        ),
        "rocm_version": optional_nonblank_string(
            data["rocm_version"], context="local environment rocm_version"
        ),
        "device_index": _optional_int(
            data["device_index"], context="local environment device_index", minimum=0
        ),
        "device_name": optional_nonblank_string(
            data["device_name"], context="local environment device_name"
        ),
        "compute_capability": (None if compute_capability is None else list(compute_capability)),
        "precision_policy": parsed_precision,
        "autocast_policy": autocast,
        "backend_invoked": _optional_bool(
            data["backend_invoked"], context="local environment backend_invoked"
        ),
        "fusion_claim": exact_bool(data["fusion_claim"], context="local environment fusion_claim"),
    }
    if result["schema"] != "heliostune.local-environment/1":
        raise SchemaError("local environment schema is not heliostune.local-environment/1")
    if parsed_precision != _PRECISION_POLICY or autocast != dict(_AUTOCAST_POLICY):
        raise SchemaError("local execution environment has an incorrect execution policy")
    capability_fields = (
        "torch_version",
        "cuda_version",
        "rocm_version",
        "device_index",
        "device_name",
        "compute_capability",
    )
    capability_values: tuple[object, ...] = (
        capability.torch_version,
        capability.cuda_version,
        capability.rocm_version,
        capability.device_index,
        capability.device_name,
        None if capability.compute_capability is None else list(capability.compute_capability),
    )
    if tuple(result[name] for name in capability_fields) != capability_values:
        raise SchemaError("local environment does not match its capability probe")
    if result["fusion_claim"] is not False:
        raise SchemaError("local execution environment must not claim fusion")
    return result


def _parse_autocast_policy(value: object) -> dict[str, object]:
    data = exact_fields(
        value,
        required=("device_type", "enabled", "restore_ambient_state"),
        context="local autocast policy",
    )
    return {
        "device_type": nonblank_string(data["device_type"], context="local autocast device_type"),
        "enabled": exact_bool(data["enabled"], context="local autocast enabled"),
        "restore_ambient_state": exact_bool(
            data["restore_ambient_state"],
            context="local autocast restore_ambient_state",
        ),
    }


def _parse_compile_outcomes(value: object, suite: Suite) -> Mapping[str, Mapping[str, object]]:
    raw = exact_object(value, context="local compile outcomes")
    arms = {arm.id: arm for arm in suite.arms}
    result: dict[str, Mapping[str, object]] = {}
    for key, item in raw.items():
        if key not in arms or arms[key].role != "candidate":
            raise SchemaError("local compile outcomes may describe candidate arms only")
        data = exact_fields(
            item,
            required=_COMPILE_OUTCOME_FIELDS,
            context=f"local compile outcome {key!r}",
        )
        autocast = _parse_autocast_policy(data["autocast_policy"])
        record: dict[str, object] = {
            "case_id": nonblank_string(
                data["case_id"], context=f"local compile outcome {key!r} case_id"
            ),
            "arm_id": nonblank_string(
                data["arm_id"], context=f"local compile outcome {key!r} arm_id"
            ),
            "entrypoint": nonblank_string(
                data["entrypoint"],
                context=f"local compile outcome {key!r} entrypoint",
            ),
            "status": _enum(
                data["status"],
                (
                    "compile_failed",
                    "wrapper_created",
                    "compiled_and_first_call_completed",
                ),
                f"local compile outcome {key!r} status",
            ),
            "error": optional_nonblank_string(
                data["error"], context=f"local compile outcome {key!r} error"
            ),
            "wrapper_create_ns": _optional_int(
                data["wrapper_create_ns"],
                context=f"local compile outcome {key!r} wrapper_create_ns",
                minimum=0,
            ),
            "first_call_ns": _optional_int(
                data["first_call_ns"],
                context=f"local compile outcome {key!r} first_call_ns",
                minimum=0,
            ),
            "eager_fallback": exact_bool(
                data["eager_fallback"],
                context=f"local compile outcome {key!r} eager_fallback",
            ),
            "backend_invoked": exact_bool(
                data["backend_invoked"],
                context=f"local compile outcome {key!r} backend_invoked",
            ),
            "callable_distinct": exact_bool(
                data["callable_distinct"],
                context=f"local compile outcome {key!r} callable_distinct",
            ),
            "autocast_policy": autocast,
        }
        if record["arm_id"] != key or record["entrypoint"] != arms[key].entrypoint:
            raise SchemaError(f"local compile outcome {key!r} has incorrect arm linkage")
        if autocast != dict(_AUTOCAST_POLICY) or record["eager_fallback"] is not False:
            raise SchemaError(f"local compile outcome {key!r} has invalid compile policy")
        status = record["status"]
        if status == "compile_failed":
            wrapper_failure = (
                record["first_call_ns"] is None and record["callable_distinct"] is False
            )
            first_call_failure = (
                record["first_call_ns"] is not None
                and record["wrapper_create_ns"] is not None
                and record["callable_distinct"] is True
            )
            if record["error"] is None or not (wrapper_failure or first_call_failure):
                raise SchemaError(
                    f"local compile outcome {key!r} has inconsistent failure evidence"
                )
        elif status == "wrapper_created":
            if (
                record["error"] is not None
                or record["wrapper_create_ns"] is None
                or record["first_call_ns"] is not None
                or record["callable_distinct"] is not True
            ):
                raise SchemaError(
                    f"local compile outcome {key!r} has inconsistent wrapper evidence"
                )
        elif (
            record["error"] is not None
            or record["wrapper_create_ns"] is None
            or record["first_call_ns"] is None
            or record["backend_invoked"] is not True
            or record["callable_distinct"] is not True
        ):
            raise SchemaError(f"local compile outcome {key!r} has inconsistent completed evidence")
        result[key] = record
    return result


def _parse_summary(
    value: object,
    suite: Suite,
    observations: Sequence[CellObservation],
    capability: CapabilityProbe,
    outcome: str,
) -> Mapping[str, object]:
    required = [
        "expected_cell_ids",
        "terminal_cell_ids",
        "passed",
        "failed",
        "blocked",
        "all_cells_terminal",
        "outcome",
        "fusion_claim",
    ]
    if not capability.available:
        required.append("capability_reasons")
    data = exact_fields(value, required=required, context="local execution summary")
    result: dict[str, object] = {
        "expected_cell_ids": [
            nonblank_string(item, context="local summary expected_cell_ids item")
            for item in _array(data["expected_cell_ids"], "local summary expected_cell_ids")
        ],
        "terminal_cell_ids": [
            nonblank_string(item, context="local summary terminal_cell_ids item")
            for item in _array(data["terminal_cell_ids"], "local summary terminal_cell_ids")
        ],
        "passed": exact_int(data["passed"], context="local summary passed", minimum=0),
        "failed": exact_int(data["failed"], context="local summary failed", minimum=0),
        "blocked": exact_int(data["blocked"], context="local summary blocked", minimum=0),
        "all_cells_terminal": exact_bool(
            data["all_cells_terminal"],
            context="local summary all_cells_terminal",
        ),
        "outcome": _enum(data["outcome"], _OUTCOMES, "local summary outcome"),
        "fusion_claim": exact_bool(data["fusion_claim"], context="local summary fusion_claim"),
    }
    if not capability.available:
        result["capability_reasons"] = [
            _enum(item, _CAPABILITY_REASONS, "local summary capability reason")
            for item in _array(data["capability_reasons"], "local summary capability_reasons")
        ]
    expected_ids = [cell.id for cell in suite.expected_cells]
    statuses = [item.status for item in observations]
    expected_values: dict[str, object] = {
        "expected_cell_ids": expected_ids,
        "terminal_cell_ids": [item.cell_id for item in observations],
        "passed": statuses.count("passed"),
        "failed": statuses.count("failed"),
        "blocked": statuses.count("blocked"),
        "all_cells_terminal": len(observations) == len(expected_ids),
        "outcome": outcome,
        "fusion_claim": False,
    }
    if not capability.available:
        expected_values["capability_reasons"] = list(capability.reasons)
    if result != expected_values:
        raise SchemaError("local execution summary does not match exact execution evidence")
    return result


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


def _validate_deserialized_result(
    *,
    suite: Suite,
    suite_sha256: str,
    capability: CapabilityProbe,
    materialization: tuple[TensorMaterialization, ...],
    observations: tuple[CellObservation, ...],
    attempts: tuple[Mapping[str, object], ...],
    environment: Mapping[str, object],
    compile_outcomes: Mapping[str, Mapping[str, object]],
    summary: Mapping[str, object],
    outcome: str,
) -> None:
    del summary
    _validate_capability_probe(capability)
    if not capability.available:
        if (
            outcome != "aborted"
            or materialization
            or observations
            or attempts
            or compile_outcomes
            or environment["backend_invoked"] is not None
        ):
            raise SchemaError(
                "capability-unavailable execution must not contain execution evidence"
            )
        return
    if outcome == "aborted":
        raise SchemaError("capability-available execution cannot be aborted")
    if environment["backend_invoked"] is not any(
        item["backend_invoked"] is True for item in compile_outcomes.values()
    ):
        raise SchemaError("local environment backend evidence does not match compile outcomes")

    expected_cells = suite.expected_cells
    if len(observations) != len(expected_cells):
        raise SchemaError("capability-available execution must observe every expected cell")
    states: dict[str, str] = {}
    terminal_status: dict[str, str] = {}
    if len(attempts) != 2 * len(expected_cells):
        raise SchemaError("local attempts must contain exactly two transitions per expected cell")
    for index, attempt in enumerate(attempts, start=1):
        if attempt["attempt_id"] != index:
            raise SchemaError("local attempt IDs must be contiguous and ordered")
        cell = expected_cells[(index - 1) // 2]
        if attempt["cell_id"] != cell.id or attempt["stage"] != cell.stage:
            raise SchemaError("local attempts must retain exact expected-cell order")
        previous = states.get(cell.id, "pending")
        if attempt["from_state"] != previous:
            raise SchemaError("local attempt from_state does not match prior transition")
        target = cast(str, attempt["to_state"])
        try:
            states[cell.id] = _advance_cell_state(previous, target)
        except ValueError as exc:
            raise SchemaError(str(exc)) from exc
        expected_attempt_status = {
            "running": "running",
            "passed": "success",
            "failed": "failure",
        }[target]
        if attempt["status"] != expected_attempt_status:
            raise SchemaError("local attempt status does not match its transition")
        if target in {"running", "passed"} and attempt["reason"] is not None:
            raise SchemaError("non-failing local transition must not contain a reason")
        if target == "failed" and attempt["reason"] is None:
            raise SchemaError("failing local transition must contain a reason")
        if target in {"passed", "failed"}:
            terminal_status[cell.id] = target

    cases = {case.id: case for case in suite.cases}
    timing_policies = {policy.id: policy for policy in suite.timing_policies}
    output_spec = next(item for item in suite.tensors if item.role == "output")
    retained_passes: set[str] = set()
    for cell, observation in zip(expected_cells, observations, strict=True):
        if (
            observation.cell_id != cell.id
            or observation.case_id != cell.case_id
            or observation.arm_id != cell.arm_id
            or observation.stage != cell.stage
        ):
            raise SchemaError(
                "local observations must retain exact expected-cell order and linkage"
            )
        if observation.status == "blocked":
            raise SchemaError("local executor does not emit blocked cell observations")
        if terminal_status.get(cell.id) != observation.status:
            raise SchemaError("local observation status does not match terminal attempt")
        case = cases[cell.case_id]
        key = _correctness_gate_key(suite_sha256, case, cell)
        nested: CorrectnessObservation | TimingObservation
        if cell.stage == "correctness":
            if observation.correctness is None or observation.timing is not None:
                raise SchemaError("correctness cell must contain only correctness evidence")
            nested = observation.correctness
        else:
            if observation.timing is None or observation.correctness is not None:
                raise SchemaError("timing cell must contain only timing evidence")
            nested = observation.timing
        if nested.status != observation.status or nested.correctness_key != key:
            raise SchemaError(
                "nested observation does not match its cell status and correctness key"
            )
        final_attempt = attempts[2 * expected_cells.index(cell) + 1]
        if final_attempt["reason"] != nested.failure_kind:
            raise SchemaError("terminal attempt reason does not match observation failure kind")
        _validate_failure_evidence(nested, cell.id)

        if isinstance(nested, CorrectnessObservation):
            evidence = (
                nested.input_storage_unchanged,
                nested.output_disjoint,
                nested.finite,
                nested.close,
            )
            if nested.status == "passed":
                if (
                    evidence != (True, True, True, True)
                    or nested.output is None
                    or nested.max_abs_error is None
                ):
                    raise SchemaError(
                        f"passing correctness observation {cell.id!r} lacks exact passing evidence"
                    )
                expected_shape = [case.shape_dict[name] for name in output_spec.shape]
                if nested.output != {
                    "shape": expected_shape,
                    "device": f"cuda:{capability.device_index}",
                    "dtype": "torch.bfloat16",
                    "layout": "torch.strided",
                    "contiguous": True,
                }:
                    raise SchemaError(
                        f"passing correctness observation {cell.id!r} has invalid output evidence"
                    )
                retained_passes.add(key)
            elif all(evidence):
                raise SchemaError(
                    f"failed correctness observation {cell.id!r} masquerades as passing"
                )
        else:
            policy = timing_policies[cast(str, cell.timing_policy_id)]
            gate_passed = key in retained_passes
            if gate_passed == (nested.failure_kind == "correctness_gate"):
                raise SchemaError(
                    f"timing observation {cell.id!r} does not match its correctness gate"
                )
            if nested.status == "passed":
                if (
                    not gate_passed
                    or nested.warmups != policy.warmups
                    or nested.repetitions != policy.repetitions
                    or len(nested.samples_ms) != policy.repetitions
                    or nested.median_ms is None
                    or not math.isclose(
                        nested.median_ms,
                        statistics.median(nested.samples_ms),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise SchemaError(
                        f"passing timing observation {cell.id!r} violates its timing policy"
                    )
            elif (
                nested.warmups not in {0, policy.warmups}
                or nested.repetitions != 0
                or nested.samples_ms
                or nested.median_ms is not None
            ):
                raise SchemaError(
                    f"failed timing observation {cell.id!r} contains positive timing evidence"
                )

    all_passed = all(item.status == "passed" for item in observations)
    if outcome != ("completed" if all_passed else "failed"):
        raise SchemaError("local execution outcome does not match cell observations")
    _validate_materialization_linkage(
        suite, suite_sha256, capability, materialization, observations
    )
    _validate_compile_linkage(suite, compile_outcomes, observations)


def _validate_capability_probe(capability: CapabilityProbe) -> None:
    torch_280 = (
        capability.torch_version is not None
        and capability.torch_version.partition("+")[0] == "2.8.0"
    )
    device_evidence = (
        capability.device_index is not None
        and capability.device_name is not None
        and capability.compute_capability is not None
    )
    no_device_evidence = (
        capability.device_index is None
        and capability.device_name is None
        and capability.compute_capability is None
    )

    if capability.available:
        if (
            capability.reasons
            or not torch_280
            or capability.cuda_version is None
            or capability.rocm_version is not None
            or not device_evidence
            or cast(tuple[int, int], capability.compute_capability) < (8, 0)
            or capability.native_bf16 is not True
            or capability.inductor_available is not True
            or capability.allocation_succeeded is not True
            or capability.detail is not None
        ):
            raise SchemaError("available local capability lacks exact passing evidence")
        return

    if len(capability.reasons) != 1 or capability.allocation_succeeded is not False:
        raise SchemaError("unavailable local capability must contain one rejection reason")

    reason = capability.reasons[0]
    no_torch_or_device_evidence = (
        capability.torch_version is None
        and capability.cuda_version is None
        and capability.rocm_version is None
        and no_device_evidence
        and capability.native_bf16 is None
        and capability.inductor_available is None
    )
    version_rejected = (
        (capability.torch_version is None or not torch_280)
        and capability.cuda_version is None
        and capability.rocm_version is None
        and no_device_evidence
        and capability.native_bf16 is None
        and capability.inductor_available is None
        and capability.detail is None
    )
    torch_only = (
        torch_280
        and capability.rocm_version is None
        and no_device_evidence
        and capability.native_bf16 is None
        and capability.inductor_available is None
    )
    cuda_device = (
        torch_280
        and capability.cuda_version is not None
        and capability.rocm_version is None
        and device_evidence
    )

    valid = False
    if reason == "torch_missing":
        valid = no_torch_or_device_evidence and capability.detail is not None
    elif reason == "torch_version_mismatch":
        valid = version_rejected
    elif reason == "cuda_unavailable":
        valid = torch_only and capability.detail is None
    elif reason == "rocm_unsupported":
        valid = (
            torch_280
            and capability.rocm_version is not None
            and no_device_evidence
            and capability.native_bf16 is None
            and capability.inductor_available is None
            and capability.detail is None
        )
    elif reason == "compute_capability_too_low":
        valid = (
            cuda_device
            and cast(tuple[int, int], capability.compute_capability) < (8, 0)
            and capability.native_bf16 is None
            and capability.inductor_available is None
            and capability.detail is None
        )
    elif reason == "bf16_unsupported":
        valid = (
            cuda_device
            and cast(tuple[int, int], capability.compute_capability) >= (8, 0)
            and capability.native_bf16 is False
            and capability.inductor_available is None
            and capability.detail is None
        )
    elif reason == "inductor_unavailable":
        valid = (
            cuda_device
            and cast(tuple[int, int], capability.compute_capability) >= (8, 0)
            and capability.native_bf16 is True
            and capability.inductor_available is False
            and capability.detail is None
        )
    elif reason == "allocation_failed":
        valid = (
            cuda_device
            and cast(tuple[int, int], capability.compute_capability) >= (8, 0)
            and capability.native_bf16 is True
            and capability.inductor_available is True
            and capability.detail is not None
        )
    elif reason == "device_probe_failed":
        valid = (
            torch_280
            and capability.rocm_version is None
            and capability.native_bf16 is None
            and capability.inductor_available is None
            and capability.detail is not None
            and (
                no_device_evidence
                or (
                    capability.cuda_version is not None
                    and device_evidence
                    and cast(tuple[int, int], capability.compute_capability) >= (8, 0)
                )
            )
        )
    if not valid:
        raise SchemaError(
            f"unavailable local capability reason {reason!r} has inconsistent probe evidence"
        )


def _validate_failure_evidence(
    observation: CorrectnessObservation | TimingObservation, cell_id: str
) -> None:
    if observation.status == "passed":
        if observation.failure_kind is not None or observation.message is not None:
            raise SchemaError(f"passing observation {cell_id!r} contains failure evidence")
    elif observation.failure_kind is None or observation.message is None:
        raise SchemaError(f"failed observation {cell_id!r} lacks failure evidence")


def _validate_materialization_linkage(
    suite: Suite,
    suite_sha256: str,
    capability: CapabilityProbe,
    materialization: tuple[TensorMaterialization, ...],
    observations: tuple[CellObservation, ...],
) -> None:
    case_by_id = {case.id: case for case in suite.cases}
    arm_by_id = {arm.id: arm for arm in suite.arms}
    expected_pairs = [
        (cell.case_id, cell.arm_id) for cell in suite.expected_cells if cell.stage == "correctness"
    ]
    actual_pairs = [(item.case_id, item.arm_id) for item in materialization]
    if len(set(actual_pairs)) != len(actual_pairs) or actual_pairs != [
        pair for pair in expected_pairs if pair in set(actual_pairs)
    ]:
        raise SchemaError("tensor materialization has unexpected, duplicate, or unordered linkage")
    passing_pairs = {
        (item.case_id, item.arm_id)
        for item in observations
        if item.stage == "correctness" and item.status == "passed"
    }
    if not passing_pairs <= set(actual_pairs):
        raise SchemaError("passing correctness requires exact tensor materialization")
    specs_by_id = {item.id: item for item in suite.tensors if item.role != "output"}
    for record in materialization:
        if (
            record.suite_sha256 != suite_sha256
            or record.case_id not in case_by_id
            or record.arm_id not in arm_by_id
        ):
            raise SchemaError("tensor materialization does not bind the verified suite")
        case = case_by_id[record.case_id]
        schedule = _resolve_draw_schedule(suite, case)
        if record.input_seed != case.input_seed:
            raise SchemaError("tensor materialization input seed does not match its case")
        if record.tensor_order != tuple(item.tensor_id for item in schedule) or len(
            record.tensors
        ) != len(schedule):
            raise SchemaError("tensor materialization does not match suite tensor order")
        for draw, descriptor in zip(schedule, record.tensors, strict=True):
            spec = specs_by_id[draw.tensor_id]
            expected = {
                "tensor_id": draw.tensor_id,
                "role": draw.role,
                "shape": list(draw.shape),
                "draw": "normal_0_1_fp32_cpu",
                "normal_scale": draw.normal_scale,
                "normal_offset": draw.normal_offset,
                "cpu_dtype": "float32",
                "storage_dtype": "bfloat16",
                "device": f"cuda:{capability.device_index}",
                "contiguous": True,
                "alignment_bytes": spec.alignment,
                "alignment_satisfied": True,
                "storage_sha256": descriptor["storage_sha256"],
            }
            if descriptor != expected:
                raise SchemaError(
                    f"tensor materialization descriptor for {draw.tensor_id!r} violates suite contract"
                )


def _validate_compile_linkage(
    suite: Suite,
    compile_outcomes: Mapping[str, Mapping[str, object]],
    observations: tuple[CellObservation, ...],
) -> None:
    arms = {arm.id: arm for arm in suite.arms}
    candidate_pairs = {
        (cell.case_id, cell.arm_id)
        for cell in suite.expected_cells
        if cell.stage == "correctness" and arms[cell.arm_id].role == "candidate"
    }
    for key, record in compile_outcomes.items():
        if (record["case_id"], key) not in candidate_pairs:
            raise SchemaError(f"local compile outcome {key!r} has incorrect case linkage")
    passed_candidates = {
        (item.case_id, item.arm_id)
        for item in observations
        if item.stage == "correctness"
        and item.status == "passed"
        and arms[item.arm_id].role == "candidate"
    }
    records = {(cast(str, item["case_id"]), key): item for key, item in compile_outcomes.items()}
    for pair in passed_candidates:
        completed_record = records.get(pair)
        if (
            completed_record is None
            or completed_record["status"] != "compiled_and_first_call_completed"
        ):
            raise SchemaError("passing candidate correctness requires completed compile evidence")


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
    raw_version = getattr(torch, "__version__", None)
    if not isinstance(raw_version, str):
        return _unavailable_probe("torch_version_mismatch")
    version = raw_version
    if version.partition("+")[0] != "2.8.0":
        return _unavailable_probe("torch_version_mismatch", version=version)
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
