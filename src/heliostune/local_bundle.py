"""Write structurally closed exploratory bundles for local fusion execution."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import math
import os
import secrets
import stat
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from heliostune.artifacts import strict_json_dumps
from heliostune.errors import ArtifactError, SchemaError
from heliostune.methodology import (
    EvidenceBundleV1,
    ProtocolV1,
    VerifiedBundle,
    attempt_chain_descriptor_bytes,
    encode_attempt_journal,
    plugin_suite_path,
    plugin_suite_role,
    selected_suite_descriptor_bytes,
    verify_bundle_v1_from_directory_fd,
)
from heliostune.scope import (
    ExpectedCell,
    Suite,
    VerifiedPlugin,
    VerifiedSuite,
    verify_plugin,
    verify_suite,
)

if TYPE_CHECKING:
    from heliostune.local_executor import CellObservation, LocalExecutionResult
    from heliostune.scope import Case, TensorSpec


_PROTOCOL_ROLES = (
    "plugin",
    "workloads",
    "candidates",
    "comparators",
    "splits",
    "numerics",
    "timing",
    "analyzer",
    "expected_cells",
    "environment_predicate",
    "failure_policy",
)
_EXTRA_ROLES = (
    "selected_suite",
    "attempt_chain",
    "terminal_cells",
    "observations",
    "capability_probe",
    "tensor_materialization",
    "execution_summary",
)
_ROLE_PATHS = {role: f"{role}.json" for role in (*_PROTOCOL_ROLES, *_EXTRA_ROLES)}
_ROLE_PATHS["observations"] = "observations.jsonl"
_RENAME_NOREPLACE = 1
_COMPILE_OUTCOME_FIELDS = {
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
}
_ATTEMPTS_PATH = "attempts.jsonl"
_PROTOCOL_PATH = "protocol.json"
_ROOT_PATH = "bundle.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_value(value: object) -> object:
    """Convert frozen result records to their lossless strict-JSON representation."""

    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SchemaError("local execution mappings must have string keys")
            converted[key] = _json_value(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise SchemaError(
        f"local execution value is not representable as strict JSON: {type(value).__name__}"
    )


def _canonical_json(value: object) -> bytes:
    return strict_json_dumps(_json_value(value)).encode("utf-8")


def _canonical_jsonl(rows: Sequence[object]) -> bytes:
    return "".join(strict_json_dumps(_json_value(row), compact=True) + "\n" for row in rows).encode(
        "utf-8"
    )


def _result_member(result: object, name: str) -> object:
    try:
        return getattr(result, name)
    except AttributeError as exc:
        raise SchemaError(f"local execution result is missing {name!r}") from exc


def _verified_result_suite(result: object) -> VerifiedSuite:
    raw_path = _result_member(result, "verified_suite_path")
    if type(raw_path) is not str:
        raise SchemaError("local execution verified_suite_path must be a string")
    verified = verify_suite(raw_path)
    declared_sha256 = _result_member(result, "verified_suite_sha256")
    if declared_sha256 != verified.sha256:
        raise ArtifactError(
            "local execution suite SHA-256 does not match the current exact suite bytes"
        )
    declared_bytes = _result_member(result, "verified_suite_bytes")
    if type(declared_bytes) is not bytes or declared_bytes != verified.bytes:
        raise ArtifactError(
            "local execution verified suite bytes do not match the current exact suite bytes"
        )
    if _result_member(result, "suite_id") != verified.suite.suite_id:
        raise ArtifactError(
            "local execution suite identity does not match its verified suite bytes"
        )
    return verified


def _bind_plugin_suite(
    plugin_path: str | Path, selected: VerifiedSuite
) -> tuple[VerifiedPlugin, int]:
    plugin = verify_plugin(plugin_path)
    matches = tuple(
        index
        for index, suite in enumerate(plugin.suites)
        if suite.sha256 == selected.sha256
        and suite.suite.suite_id == selected.suite.suite_id
        and suite.suite.revision == selected.suite.revision
    )
    if len(matches) != 1:
        raise ArtifactError("selected suite is not exactly one digest-bound plugin suite reference")
    if (selected.suite.plugin_id, selected.suite.plugin_version) != (
        plugin.plugin.plugin_id,
        plugin.plugin.version,
    ):
        raise ArtifactError("selected suite does not belong to the supplied plugin")
    return plugin, matches[0]


def _role_values(suite: Suite) -> dict[str, object]:
    candidates = [arm.to_dict() for arm in suite.arms if arm.role == "candidate"]
    comparators = [arm.to_dict() for arm in suite.arms if arm.role == "reference"]
    return {
        "workloads": {
            "suite_id": suite.suite_id,
            "revision": suite.revision,
            "template_id": suite.template_id,
            "domain": suite.domain,
            "tensors": [tensor.to_dict() for tensor in suite.tensors],
            "cases": [case.to_dict() for case in suite.cases],
        },
        "candidates": candidates,
        "comparators": comparators,
        "splits": {
            "policy": "single_frozen_suite_no_data_split",
            "case_ids": [case.id for case in suite.cases],
        },
        "numerics": {
            "numeric_contracts": [contract.to_dict() for contract in suite.numeric_contracts],
            "correctness_policies": [policy.to_dict() for policy in suite.correctness_policies],
        },
        "timing": {
            "timing_policies": [policy.to_dict() for policy in suite.timing_policies],
            "executor_rule": suite.executor_rule,
        },
        "analyzer": {"claims": [], "policy": "no_analysis_for_exploratory_local_execution"},
        "expected_cells": [cell.id for cell in suite.expected_cells],
        "environment_predicate": {
            "torch_version": "2.8.0",
            "accelerator": "cuda",
            "rocm": False,
            "minimum_compute_capability": "8.0",
            "native_bfloat16": True,
            "inductor_backend": True,
        },
        "failure_policy": {
            "retry_policy": "none",
            "max_physical_attempts": 1,
            "compile_or_runtime_failure": "retain_failure_without_eager_fallback",
            "capability_unavailable": "abort_before_cell_execution",
            "timing_gate": "exact_retained_passing_correctness_for_same_case_arm_seed_contract",
        },
    }


def _correctness_key(suite_sha256: str, suite: Suite, cell: ExpectedCell) -> str:
    case = next(item for item in suite.cases if item.id == cell.case_id)
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


def _finite_number(value: object, *, positive: bool = False) -> bool:
    if type(value) not in {int, float}:
        return False
    numeric = cast(int | float, value)
    return math.isfinite(numeric) and (numeric > 0 if positive else numeric >= 0)


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _validate_output_descriptor(
    output: object,
    *,
    case: Case,
    output_spec: TensorSpec,
    device_index: int,
    cell_id: str,
) -> None:
    expected_fields = {"shape", "device", "dtype", "layout", "contiguous"}
    if not isinstance(output, Mapping) or set(output) != expected_fields:
        raise SchemaError(
            f"passing correctness observation {cell_id!r} has an invalid output descriptor"
        )
    shape_dict = case.shape_dict
    expected_shape = [shape_dict[name] for name in output_spec.shape]
    shape = output["shape"]
    if (
        type(shape) is not list
        or any(type(dimension) is not int for dimension in shape)
        or shape != expected_shape
    ):
        raise SchemaError(f"passing correctness observation {cell_id!r} has the wrong output shape")
    if (
        output_spec.logical_dtype != "bf16"
        or output["device"] != f"cuda:{device_index}"
        or output["dtype"] != "torch.bfloat16"
        or output["layout"] != "torch.strided"
        or output["contiguous"] is not True
    ):
        raise SchemaError(
            f"passing correctness observation {cell_id!r} violates the output tensor contract"
        )


def _validate_observations(
    suite: Suite,
    suite_sha256: str,
    observations: Sequence[CellObservation],
    terminal_ids: tuple[str, ...],
    attempts: tuple[dict[str, str], ...],
    device_index: int,
) -> tuple[CellObservation, ...]:
    expected = {cell.id: cell for cell in suite.expected_cells}
    cases = {case.id: case for case in suite.cases}
    timing_policies = {policy.id: policy for policy in suite.timing_policies}
    final_attempts = {row["cell_id"]: row["status"] for row in attempts}
    output_spec = next(item for item in suite.tensors if item.role == "output")
    validated: list[CellObservation] = []
    if len(observations) != len(terminal_ids):
        raise SchemaError("local observations must exactly cover every terminal cell")

    for index, observation in enumerate(observations):
        cell_id = getattr(observation, "cell_id", None)
        if index >= len(terminal_ids) or cell_id != terminal_ids[index]:
            raise SchemaError(
                "local observations must exactly match terminal cells in expected order"
            )
        cell = expected[cell_id]
        status = getattr(observation, "status", None)
        if status not in {"passed", "failed", "blocked"}:
            raise SchemaError(f"local observation {cell_id!r} has invalid status")
        for name in ("case_id", "arm_id", "stage"):
            if getattr(observation, name, None) != getattr(cell, name):
                raise SchemaError(f"local observation {cell_id!r} does not match expected {name}")
        case = cases[cell.case_id]
        if cell.input_seed != case.input_seed:
            raise SchemaError(f"local observation {cell_id!r} has inconsistent input seed linkage")
        expected_key = _correctness_key(suite_sha256, suite, cell)
        correctness = getattr(observation, "correctness", None)
        timing = getattr(observation, "timing", None)
        nested = correctness if cell.stage == "correctness" else timing
        other = timing if cell.stage == "correctness" else correctness
        if nested is None or other is not None:
            raise SchemaError(
                f"local observation {cell_id!r} has the wrong nested record for {cell.stage}"
            )
        if getattr(nested, "status", None) != status:
            raise SchemaError(f"local observation {cell_id!r} nested status does not match")
        if getattr(nested, "correctness_key", None) != expected_key:
            raise SchemaError(
                f"local observation {cell_id!r} correctness key does not bind the exact suite, "
                "case, arm, seed, and numeric contract"
            )
        expected_attempt = "success" if status == "passed" else "failure"
        if final_attempts[cell_id] != expected_attempt:
            raise SchemaError(f"observation status does not match attempt status for {cell_id!r}")

        failure_kind = getattr(nested, "failure_kind", None)
        message = getattr(nested, "message", None)
        if status == "passed":
            if failure_kind is not None or message is not None:
                raise SchemaError(f"passing observation {cell_id!r} contains failure evidence")
        elif (
            type(failure_kind) is not str
            or not failure_kind
            or type(message) is not str
            or not message
        ):
            raise SchemaError(f"failed observation {cell_id!r} lacks exact failure evidence")

        if cell.stage == "correctness":
            boolean_fields = (
                "input_storage_unchanged",
                "output_disjoint",
                "finite",
                "close",
            )
            values = tuple(getattr(correctness, name, None) for name in boolean_fields)
            if any(type(value) is not bool for value in values):
                raise SchemaError(
                    f"correctness observation {cell_id!r} has invalid boolean evidence"
                )
            max_abs_error = getattr(correctness, "max_abs_error", None)
            if max_abs_error is not None and not _finite_number(max_abs_error):
                raise SchemaError(
                    f"correctness observation {cell_id!r} has invalid maximum absolute error"
                )
            if status == "passed":
                if not all(values) or max_abs_error is None:
                    raise SchemaError(
                        f"passing correctness observation {cell_id!r} lacks passing evidence"
                    )
                _validate_output_descriptor(
                    getattr(correctness, "output", None),
                    case=case,
                    output_spec=output_spec,
                    device_index=device_index,
                    cell_id=cast(str, cell_id),
                )
            elif all(values):
                raise SchemaError(
                    f"failed correctness observation {cell_id!r} masquerades as passing"
                )
        else:
            policy = timing_policies[cast(str, cell.timing_policy_id)]
            warmups = getattr(timing, "warmups", None)
            repetitions = getattr(timing, "repetitions", None)
            samples = getattr(timing, "samples_ms", None)
            median_ms = getattr(timing, "median_ms", None)
            if (
                type(warmups) is not int
                or type(repetitions) is not int
                or not isinstance(samples, Sequence)
                or isinstance(samples, (str, bytes, bytearray))
            ):
                raise SchemaError(f"timing observation {cell_id!r} has an invalid timing shape")
            if status == "passed":
                if (
                    warmups != policy.warmups
                    or repetitions != policy.repetitions
                    or len(samples) != policy.repetitions
                    or not all(_finite_number(sample, positive=True) for sample in samples)
                    or not _finite_number(median_ms, positive=True)
                ):
                    raise SchemaError(
                        f"passing timing observation {cell_id!r} does not satisfy its timing policy"
                    )
                recomputed = statistics.median(samples)
                if not math.isclose(
                    cast(float, median_ms),
                    recomputed,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise SchemaError(
                        f"passing timing observation {cell_id!r} has an incorrect median"
                    )
            elif (
                warmups not in {0, policy.warmups}
                or repetitions != 0
                or len(samples) != 0
                or median_ms is not None
            ):
                raise SchemaError(
                    f"failed timing observation {cell_id!r} contains positive timing results"
                )
        validated.append(observation)
    return tuple(validated)


_MATERIALIZATION_DESCRIPTOR_FIELDS = {
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
}


def _draw_parameters(suite: Suite, case: Case, tensor: TensorSpec) -> tuple[float, float]:
    scale = 1.0
    offset = 0.0
    dimensions = case.shape_dict
    if suite.template_id == "gated_mlp_epilogue.v1" and tensor.role == "parameter":
        scale = 1.0 / math.sqrt(dimensions["hidden"])
    if suite.template_id == "residual_rmsnorm.v1" and tensor.id == "gamma":
        scale = 0.02
        offset = 1.0
    return scale, offset


def _validate_materialization(
    result: object,
    suite: Suite,
    suite_sha256: str,
    observations: Sequence[CellObservation],
    device_index: int,
) -> tuple[object, ...]:
    raw = _result_member(result, "materialization")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise SchemaError("local tensor materialization must be a sequence")
    cases = {case.id: case for case in suite.cases}
    arms = {arm.id: arm for arm in suite.arms}
    input_specs = tuple(item for item in suite.tensors if item.role != "output")
    tensor_ids = tuple(item.id for item in input_specs)
    expected_pairs = {
        (cell.case_id, cell.arm_id) for cell in suite.expected_cells if cell.stage == "correctness"
    }
    passed_pairs = {
        (item.case_id, item.arm_id)
        for item in observations
        if item.stage == "correctness" and item.status == "passed"
    }
    seen: set[tuple[str, str]] = set()
    validated: list[object] = []
    for record in raw:
        case_id = getattr(record, "case_id", None)
        arm_id = getattr(record, "arm_id", None)
        pair = (case_id, arm_id)
        if pair not in expected_pairs or pair in seen:
            raise SchemaError("tensor materialization has an unexpected or duplicate case/arm")
        case = cases[cast(str, case_id)]
        if arm_id not in arms:
            raise SchemaError("tensor materialization references an unknown arm")
        if getattr(record, "suite_sha256", None) != suite_sha256:
            raise ArtifactError("tensor materialization suite SHA-256 does not match the suite")
        input_seed = getattr(record, "input_seed", None)
        if type(input_seed) is not int or input_seed != case.input_seed:
            raise SchemaError("tensor materialization input seed does not match its case")
        order = getattr(record, "tensor_order", None)
        tensors = getattr(record, "tensors", None)
        if (
            not isinstance(order, Sequence)
            or isinstance(order, (str, bytes, bytearray))
            or tuple(order) != tensor_ids
            or not isinstance(tensors, Sequence)
            or isinstance(tensors, (str, bytes, bytearray))
            or len(tensors) != len(tensor_ids)
        ):
            raise SchemaError("tensor materialization does not match the suite tensor order")
        for spec, descriptor in zip(input_specs, tensors, strict=True):
            if not isinstance(descriptor, Mapping) or set(descriptor) != (
                _MATERIALIZATION_DESCRIPTOR_FIELDS
            ):
                raise SchemaError("tensor materialization descriptor has an invalid evidence shape")
            scale, offset = _draw_parameters(suite, case, spec)
            shape = descriptor["shape"]
            expected_shape = [case.shape_dict[name] for name in spec.shape]
            if (
                type(descriptor["tensor_id"]) is not str
                or descriptor["tensor_id"] != spec.id
                or type(descriptor["role"]) is not str
                or descriptor["role"] != spec.role
                or type(shape) is not list
                or any(type(dimension) is not int for dimension in shape)
                or shape != expected_shape
                or descriptor["draw"] != "normal_0_1_fp32_cpu"
                or type(descriptor["normal_scale"]) is not float
                or descriptor["normal_scale"] != scale
                or type(descriptor["normal_offset"]) is not float
                or descriptor["normal_offset"] != offset
                or descriptor["cpu_dtype"] != "float32"
                or descriptor["storage_dtype"] != "bfloat16"
                or spec.storage_dtype != "bf16"
                or spec.logical_dtype != "bf16"
                or descriptor["device"] != f"cuda:{device_index}"
                or descriptor["contiguous"] is not spec.contiguous
                or type(descriptor["alignment_bytes"]) is not int
                or descriptor["alignment_bytes"] != spec.alignment
                or descriptor["alignment_satisfied"] is not True
            ):
                raise SchemaError(
                    f"tensor materialization descriptor for {spec.id!r} violates its suite contract"
                )
            digest = descriptor["storage_sha256"]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SchemaError("tensor materialization storage SHA-256 is invalid")
        seen.add(pair)
        validated.append(record)
    if not passed_pairs <= seen:
        raise SchemaError("passing correctness observations require exact tensor materialization")
    return tuple(validated)


def _validate_compile_outcomes(
    result: object,
    suite: Suite,
    observations: Sequence[CellObservation],
) -> dict[str, object]:
    raw = _result_member(result, "compile_outcomes")
    if not isinstance(raw, Mapping):
        raise SchemaError("local compile outcomes must be a mapping")
    arms = {arm.id: arm for arm in suite.arms}
    candidate_pairs = {
        (cell.case_id, cell.arm_id)
        for cell in suite.expected_cells
        if cell.stage == "correctness" and arms[cell.arm_id].role == "candidate"
    }
    passed_candidate_pairs = {
        (item.case_id, item.arm_id)
        for item in observations
        if item.stage == "correctness"
        and item.status == "passed"
        and arms[item.arm_id].role == "candidate"
    }
    exact_autocast_policy = {
        "device_type": "cuda",
        "enabled": False,
        "restore_ambient_state": True,
    }
    validated: dict[str, object] = {}
    validated_pairs: dict[tuple[str, str], Mapping[str, object]] = {}
    for key, record in raw.items():
        if type(key) is not str or key not in arms or arms[key].role != "candidate":
            raise SchemaError("compile outcomes may describe candidate arms only")
        if not isinstance(record, Mapping):
            raise SchemaError(f"compile outcome {key!r} must be a mapping")
        if set(record) != _COMPILE_OUTCOME_FIELDS:
            raise SchemaError(f"compile outcome {key!r} has an invalid evidence shape")
        case_id = record["case_id"]
        arm_id = record["arm_id"]
        if (
            type(case_id) is not str
            or type(arm_id) is not str
            or arm_id != key
            or (case_id, arm_id) not in candidate_pairs
        ):
            raise SchemaError(f"compile outcome {key!r} has incorrect case/arm linkage")
        if type(record["entrypoint"]) is not str or record["entrypoint"] != arms[key].entrypoint:
            raise SchemaError(f"compile outcome {key!r} does not bind the exact entrypoint")
        if record["eager_fallback"] is not False:
            raise SchemaError(f"compile outcome {key!r} must prohibit eager fallback")
        autocast_policy = record["autocast_policy"]
        if (
            type(record["backend_invoked"]) is not bool
            or type(record["callable_distinct"]) is not bool
            or type(autocast_policy) is not dict
            or set(autocast_policy) != set(exact_autocast_policy)
            or autocast_policy["device_type"] != "cuda"
            or autocast_policy["enabled"] is not False
            or autocast_policy["restore_ambient_state"] is not True
        ):
            raise SchemaError(f"compile outcome {key!r} has invalid compile evidence")
        status = record["status"]
        error = record["error"]
        wrapper_create_ns = record["wrapper_create_ns"]
        first_call_ns = record["first_call_ns"]
        if status == "compile_failed":
            wrapper_failure = first_call_ns is None and record["callable_distinct"] is False
            lazy_first_call_failure = (
                _nonnegative_int(first_call_ns)
                and record["callable_distinct"] is True
                and _nonnegative_int(wrapper_create_ns)
            )
            if (
                type(error) is not str
                or not error
                or (wrapper_create_ns is not None and not _nonnegative_int(wrapper_create_ns))
                or not (wrapper_failure or lazy_first_call_failure)
            ):
                raise SchemaError(f"compile outcome {key!r} has inconsistent failure evidence")
        elif status == "wrapper_created":
            if (
                error is not None
                or not _nonnegative_int(wrapper_create_ns)
                or first_call_ns is not None
                or record["callable_distinct"] is not True
            ):
                raise SchemaError(f"compile outcome {key!r} has inconsistent wrapper evidence")
        elif status == "compiled_and_first_call_completed":
            if (
                error is not None
                or not _nonnegative_int(wrapper_create_ns)
                or not _nonnegative_int(first_call_ns)
                or record["backend_invoked"] is not True
                or record["callable_distinct"] is not True
            ):
                raise SchemaError(f"compile outcome {key!r} has inconsistent completed evidence")
        else:
            raise SchemaError(f"compile outcome {key!r} has an invalid status")
        pair = (case_id, arm_id)
        validated_pairs[pair] = record
        validated[key] = dict(record)
    for pair in passed_candidate_pairs:
        record = validated_pairs.get(pair)
        if (
            record is None
            or record["status"] != "compiled_and_first_call_completed"
            or not _positive_int(record["wrapper_create_ns"])
            or not _positive_int(record["first_call_ns"])
        ):
            raise SchemaError(
                "passing candidate correctness requires completed exact compile evidence"
            )
    return validated


def _reject_summary_overrides(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise SchemaError("local execution summary keys must be strings")
            if (
                (key == "claims" and item not in ([], ()))
                or (key == "fusion_claim" and item is not False)
                or key.startswith("publication")
            ):
                raise SchemaError(f"local execution summary may not override {key!r}")
            _reject_summary_overrides(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_summary_overrides(item)


def _contains_execution_evidence(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {
                "compile_outcomes",
                "materialization",
                "observations",
                "samples_ms",
                "first_call_ns",
                "wrapper_create_ns",
            }:
                return True
            if (
                key
                in {
                    "backend_invoked",
                    "callable_distinct",
                    "eager_fallback",
                    "execution_completed",
                }
                and item is not None
                and item is not False
            ):
                return True
            if _contains_execution_evidence(item):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_execution_evidence(item) for item in value)
    return False


def _execution_summary(
    result: object,
    *,
    outcome: str,
    suite: Suite,
    selected: VerifiedSuite,
    terminal_ids: tuple[str, ...],
    observations: Sequence[CellObservation],
    compile_outcomes: Mapping[str, object],
) -> dict[str, object]:
    raw_summary = _result_member(result, "summary")
    if not isinstance(raw_summary, Mapping):
        raise SchemaError("local execution summary must be a mapping")
    _reject_summary_overrides(raw_summary)
    environment = _result_member(result, "environment")
    if not isinstance(environment, Mapping):
        raise SchemaError("local execution environment must be a mapping")
    _reject_summary_overrides(environment)
    safe_environment = dict(environment)
    safe_environment["fusion_claim"] = False
    statuses = [item.status for item in observations]
    return {
        "outcome": outcome,
        "environment": safe_environment,
        "compile_outcomes": dict(compile_outcomes),
        "summary": {
            "expected_cell_ids": [cell.id for cell in suite.expected_cells],
            "terminal_cell_ids": list(terminal_ids),
            "passed": statuses.count("passed"),
            "failed": statuses.count("failed"),
            "blocked": statuses.count("blocked"),
            "all_cells_terminal": len(terminal_ids) == len(suite.expected_cells),
            "outcome": outcome,
            "claims": [],
            "fusion_claim": False,
        },
        "claims": [],
        "fusion_claim": False,
        "suite_path": str(selected.path),
        "suite_sha256": selected.sha256,
        "suite_bytes": len(selected.bytes),
    }


def _attempt_rows(result: object) -> tuple[dict[str, str], ...]:
    raw = _result_member(result, "attempts")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise SchemaError("local execution attempts must be a sequence")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for transition in raw:
        if isinstance(transition, Mapping):
            cell_id = transition.get("cell_id")
            status = transition.get("status", transition.get("to_state"))
            from_state = transition.get("from_state")
        else:
            cell_id = getattr(transition, "cell_id", None)
            status = getattr(transition, "status", getattr(transition, "to_state", None))
            from_state = getattr(transition, "from_state", None)
        if type(cell_id) is not str or not cell_id:
            raise SchemaError("local execution attempt cell_id must be a nonblank string")
        if status not in {"pending", "running", "success", "failure"}:
            raise SchemaError(f"invalid local execution attempt status {status!r}")
        if cell_id not in seen and status != "pending" and from_state == "pending":
            rows.append({"cell_id": cell_id, "status": "pending"})
        rows.append({"cell_id": cell_id, "status": cast(str, status)})
        seen.add(cell_id)
    return tuple(rows)


def _terminal_state(
    expected_ids: tuple[str, ...],
    attempts: tuple[dict[str, str], ...],
    outcome: object,
) -> tuple[tuple[str, ...], int, int, int]:
    allowed: dict[str | None, set[str]] = {
        None: {"pending"},
        "pending": {"running", "success", "failure"},
        "running": {"success", "failure"},
        "success": set(),
        "failure": set(),
    }
    states: dict[str, str] = {}
    journal_ids: list[str] = []
    for row in attempts:
        cell_id = row["cell_id"]
        status = row["status"]
        if cell_id not in expected_ids:
            raise SchemaError(f"attempt references unknown expected cell {cell_id!r}")
        previous = states.get(cell_id)
        if status not in allowed[previous]:
            raise SchemaError(
                f"invalid local attempt transition for {cell_id!r}: {previous!r} -> {status!r}"
            )
        if previous is None:
            journal_ids.append(cell_id)
        states[cell_id] = status
    if tuple(journal_ids) != expected_ids[: len(journal_ids)]:
        raise SchemaError("local attempt cells must retain exact expected-cell prefix order")
    terminal_ids = tuple(
        cell_id for cell_id in expected_ids if states.get(cell_id) in {"success", "failure"}
    )
    if terminal_ids != expected_ids[: len(terminal_ids)]:
        raise SchemaError("local terminal cells must retain an exact expected-cell prefix")
    if outcome == "completed" and terminal_ids != expected_ids:
        raise SchemaError("completed local execution must terminate every expected cell")
    successes = sum(states[cell_id] == "success" for cell_id in terminal_ids)
    failures = len(terminal_ids) - successes
    return terminal_ids, len(states), successes, failures


def _protocol_payload(
    *,
    suite: Suite,
    plugin_id: str,
    plugin_version: int,
    role_payloads: Mapping[str, bytes],
    created_at: str,
) -> bytes:
    protocol = ProtocolV1.from_dict(
        {
            "schema": "heliostune.protocol/1",
            "study_id": f"local-{suite.suite_id}",
            "revision": suite.revision,
            "created_at": created_at,
            "evidence_class": "exploratory",
            "parent_protocol_sha256": None,
            "plugin": {
                "id": plugin_id,
                "version": str(plugin_version),
                "artifact_sha256": _sha256(role_payloads["plugin"]),
            },
            "semantic": {
                f"{role}_sha256": _sha256(role_payloads[role])
                for role in (
                    "workloads",
                    "candidates",
                    "comparators",
                    "splits",
                    "numerics",
                    "timing",
                )
            },
            "analysis": {
                "analyzer_sha256": _sha256(role_payloads["analyzer"]),
                "claims": [],
            },
            "execution": {
                "executor_api": "heliostune.local_executor/1",
                "expected_cells_sha256": _sha256(role_payloads["expected_cells"]),
                "expected_cell_count": len(suite.expected_cells),
                "environment_predicate_sha256": _sha256(role_payloads["environment_predicate"]),
                "failure_policy_sha256": _sha256(role_payloads["failure_policy"]),
                "retry_policy": "none",
                "max_physical_attempts": 1,
                "wall_limit_s": 86400,
                "paid_plan_sha256": None,
            },
        }
    )
    return _canonical_json(protocol.to_dict())


def _artifact_entry(role: str, path: str, payload: bytes) -> dict[str, object]:
    media_type = "application/x-ndjson" if role == "observations" else "application/json"
    return {
        "role": role,
        "path": path,
        "media_type": media_type,
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _require_descriptor_publication() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise ArtifactError("descriptor-pinned bundle publication is unsupported on this platform")
    required_dir_fd = (os.open, os.mkdir, os.rename, os.stat, os.unlink, os.rmdir)
    if any(function not in os.supports_dir_fd for function in required_dir_fd):
        raise ArtifactError(
            "descriptor-relative bundle publication is unsupported on this platform"
        )


def _same_identity(value: os.stat_result, expected: tuple[int, int]) -> bool:
    return stat.S_ISDIR(value.st_mode) and (value.st_dev, value.st_ino) == expected


def _write_file_at(directory_fd: int, name: str, payload: bytes) -> None:
    temporary = f".{name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    file_fd = -1
    try:
        file_fd = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        view = memoryview(payload)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError("short write while staging bundle artifact")
            view = view[written:]
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = -1
        os.rename(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except BaseException:
        if file_fd >= 0:
            os.close(file_fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory_fd)
        raise


def _remove_staged_directory(parent_fd: int, directory_fd: int, name: str) -> None:
    for entry in os.listdir(directory_fd):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(entry, dir_fd=directory_fd)
    with contextlib.suppress(FileNotFoundError):
        os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _published_verified_bundle(verified: VerifiedBundle, destination: Path) -> VerifiedBundle:
    protocol = replace(verified.protocol, path=destination / _PROTOCOL_PATH)
    return replace(
        verified,
        protocol=protocol,
        root_path=destination / _ROOT_PATH,
        referenced_paths=tuple(destination / path.name for path in verified.referenced_paths),
    )


def _rename_directory_noreplace(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise ArtifactError(
            "atomic no-replace bundle publication is unsupported on this platform"
        ) from exc
    result = renameat2(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _require_publication_controls(verified: VerifiedBundle) -> None:
    if verified.publication_eligible:
        raise ArtifactError("exploratory local bundle unexpectedly became publication eligible")
    required_controls = {
        "plugin suite custody": verified.limitations.plugin_suite_custody,
        "attempt journal hash chain": verified.limitations.attempt_journal_hash_chain,
        "attempt reconciliation": verified.limitations.attempt_reconciliation,
    }
    unchecked = tuple(name for name, status in required_controls.items() if status != "checked")
    if unchecked:
        raise ArtifactError(
            "staged bundle verification did not check required controls: " + ", ".join(unchecked)
        )


def _require_same_verified_content(before: VerifiedBundle, after: VerifiedBundle) -> None:
    if (
        before.bundle != after.bundle
        or before.root_sha256 != after.root_sha256
        or before.root_bytes != after.root_bytes
        or before.protocol.protocol != after.protocol.protocol
        or before.protocol.sha256 != after.protocol.sha256
        or before.protocol.bytes != after.protocol.bytes
        or tuple(path.name for path in before.referenced_paths)
        != tuple(path.name for path in after.referenced_paths)
        or before.limitations != after.limitations
    ):
        raise ArtifactError("published bundle content identity changed after staging verification")


def _publish_staged_bundle(
    destination: Path,
    payloads: Sequence[tuple[str, bytes]],
    root_payload: bytes,
) -> VerifiedBundle:
    _require_descriptor_publication()
    destination = destination.absolute()
    final_name = destination.name
    if not final_name or final_name in {".", ".."}:
        raise ArtifactError(f"invalid bundle output directory: {destination}")
    parent_path = destination.parent
    parent_fd = -1
    staging_fd = -1
    staging_name = ""
    cleanup_name = ""
    published = False
    try:
        parent_fd = os.open(
            parent_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        parent_stat = os.fstat(parent_fd)
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        if not _same_identity(os.stat(parent_path, follow_symlinks=False), parent_identity):
            raise ArtifactError("bundle output parent identity changed while it was opened")
        try:
            os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ArtifactError(f"bundle output directory already exists: {destination}")

        for _ in range(32):
            staging_name = f".heliostune-bundle-{secrets.token_hex(16)}"
            try:
                os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            break
        else:
            raise ArtifactError("cannot allocate a unique bundle staging directory")
        cleanup_name = staging_name

        staging_fd = os.open(
            staging_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        os.fchmod(staging_fd, 0o700)
        staging_stat = os.fstat(staging_fd)
        staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
        if not _same_identity(
            os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False),
            staging_identity,
        ):
            raise ArtifactError("bundle staging directory identity changed while it was opened")
        if stat.S_IMODE(staging_stat.st_mode) != 0o700:
            raise ArtifactError("bundle staging directory does not have mode 0700")

        for name, payload in payloads:
            _write_file_at(staging_fd, name, payload)
        os.fsync(staging_fd)
        _write_file_at(staging_fd, _ROOT_PATH, root_payload)

        verified = verify_bundle_v1_from_directory_fd(
            staging_fd,
            diagnostic_directory=destination,
        )
        _require_publication_controls(verified)

        if not _same_identity(os.fstat(parent_fd), parent_identity) or not _same_identity(
            os.stat(parent_path, follow_symlinks=False), parent_identity
        ):
            raise ArtifactError("bundle output parent identity changed during staging")
        if not _same_identity(os.fstat(staging_fd), staging_identity) or not _same_identity(
            os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False),
            staging_identity,
        ):
            raise ArtifactError("bundle staging directory identity changed before publication")
        try:
            os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ArtifactError(f"bundle output directory already exists: {destination}")
        _rename_directory_noreplace(parent_fd, staging_name, final_name)
        cleanup_name = final_name
        os.fsync(parent_fd)
        published_verified = verify_bundle_v1_from_directory_fd(
            staging_fd,
            diagnostic_directory=destination,
        )
        _require_publication_controls(published_verified)
        _require_same_verified_content(verified, published_verified)
        if not _same_identity(os.stat(parent_path, follow_symlinks=False), parent_identity):
            raise ArtifactError("bundle output parent identity changed during publication")
        if not _same_identity(
            os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False),
            staging_identity,
        ):
            raise ArtifactError("published bundle identity changed during publication")
        published_result = _published_verified_bundle(published_verified, destination)
        published = True
        return published_result
    except OSError as exc:
        raise ArtifactError(f"cannot publish bundle output directory {destination}: {exc}") from exc
    finally:
        if staging_fd >= 0:
            if not published:
                with contextlib.suppress(OSError):
                    _remove_staged_directory(parent_fd, staging_fd, cleanup_name)
            os.close(staging_fd)
        elif parent_fd >= 0 and cleanup_name and not published:
            with contextlib.suppress(OSError):
                os.rmdir(cleanup_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def write_local_bundle(
    result: LocalExecutionResult,
    *,
    plugin_path: str | Path,
    output_dir: str | Path,
) -> VerifiedBundle:
    """Validate, stage, verify, and atomically publish a closed exploratory bundle.

    ``output_dir`` and its final root remain absent until every descriptor-pinned
    staged artifact has been fsynced and the staged root has verified. If a hostile
    actor renames the parent pathname or the published destination after the atomic
    publication step, an unreported orphan may remain respectively in the
    renamed-away parent or at the destination's new pathname. Neither case returns
    success.
    """

    selected = _verified_result_suite(result)
    plugin, selected_suite_index = _bind_plugin_suite(plugin_path, selected)
    suite = selected.suite

    expected_ids = tuple(cell.id for cell in suite.expected_cells)
    outcome = _result_member(result, "outcome")
    if outcome not in {"completed", "failed", "aborted"}:
        raise SchemaError(f"invalid local execution outcome {outcome!r}")
    attempts = _attempt_rows(result)
    terminal_ids, logical_attempts, successes, failures = _terminal_state(
        expected_ids, attempts, outcome
    )
    capability = _result_member(result, "capability")
    available = getattr(capability, "available", None)
    device_index = getattr(capability, "device_index", None)
    if type(available) is not bool:
        raise SchemaError("local capability availability must be a boolean")
    if available and (type(device_index) is not int or device_index < 0):
        raise SchemaError("available local capability must identify an exact CUDA device")

    observations_raw = _result_member(result, "observations")
    if not isinstance(observations_raw, Sequence) or isinstance(
        observations_raw, (str, bytes, bytearray)
    ):
        raise SchemaError("local execution observations must be a sequence")
    observations = _validate_observations(
        suite,
        selected.sha256,
        observations_raw,
        terminal_ids,
        attempts,
        0 if device_index is None else device_index,
    )
    if outcome == "completed" and any(item.status != "passed" for item in observations):
        raise SchemaError("completed local execution may contain passing observations only")
    if outcome == "failed" and all(item.status == "passed" for item in observations):
        raise SchemaError("failed local execution must contain a failed or blocked observation")
    materialization = _validate_materialization(
        result,
        suite,
        selected.sha256,
        observations,
        0 if device_index is None else device_index,
    )
    compile_outcomes = _validate_compile_outcomes(result, suite, observations)

    environment = _result_member(result, "environment")
    if not isinstance(environment, Mapping):
        raise SchemaError("local execution environment must be a mapping")
    if not available and (
        outcome != "aborted"
        or attempts
        or observations
        or materialization
        or compile_outcomes
        or _contains_execution_evidence(environment)
    ):
        raise SchemaError("capability-unavailable execution must not contain execution evidence")

    execution_summary = _execution_summary(
        result,
        outcome=outcome,
        suite=suite,
        selected=selected,
        terminal_ids=terminal_ids,
        observations=observations,
        compile_outcomes=compile_outcomes,
    )
    role_payloads: dict[str, bytes] = {"plugin": plugin.bytes}
    role_payloads.update(
        {role: _canonical_json(value) for role, value in _role_values(suite).items()}
    )
    role_payloads.update(
        {
            "selected_suite": selected_suite_descriptor_bytes(selected_suite_index),
            "attempt_chain": attempt_chain_descriptor_bytes(),
            "terminal_cells": _canonical_json(terminal_ids),
            "observations": _canonical_jsonl(observations),
            "capability_probe": _canonical_json(capability),
            "tensor_materialization": _canonical_json(materialization),
            "execution_summary": _canonical_json(execution_summary),
        }
    )

    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    protocol_payload = _protocol_payload(
        suite=suite,
        plugin_id=plugin.plugin.plugin_id,
        plugin_version=plugin.plugin.version,
        role_payloads=role_payloads,
        created_at=created_at,
    )
    attempts_payload, attempt_chain_head = encode_attempt_journal(attempts)

    artifact_payloads = [
        (role, _ROLE_PATHS[role], role_payloads[role]) for role in (*_PROTOCOL_ROLES, *_EXTRA_ROLES)
    ]
    artifact_payloads[1:1] = [
        (plugin_suite_role(index), plugin_suite_path(index), verified_suite.bytes)
        for index, verified_suite in enumerate(plugin.suites)
    ]
    artifacts = [_artifact_entry(role, path, payload) for role, path, payload in artifact_payloads]
    root = EvidenceBundleV1.from_dict(
        {
            "schema": "heliostune.bundle/1",
            "bundle_id": f"local-{suite.suite_id}-{selected.sha256[:12]}",
            "created_at": created_at,
            "protocol": {
                "path": _PROTOCOL_PATH,
                "sha256": _sha256(protocol_payload),
                "bytes": len(protocol_payload),
            },
            "lifecycle": {"state": "SEALED", "outcome": outcome},
            "attempts": {
                "path": _ATTEMPTS_PATH,
                "sha256": _sha256(attempts_payload),
                "hash_chain_head": attempt_chain_head,
                "logical": logical_attempts,
                "physical": logical_attempts,
                "terminal": len(terminal_ids),
                "orphaned": 0,
            },
            "coverage": {
                "expected_cells": len(expected_ids),
                "terminal_cells": len(terminal_ids),
                "successes": successes,
                "failures": failures,
            },
            "artifacts": artifacts,
            "provenance": {"attestation": "none", "offline_reproduction": "not_checked"},
            "signatures": [],
        }
    )
    root_payload = _canonical_json(root.to_dict())

    payloads = [(path, payload) for _, path, payload in artifact_payloads]
    payloads.extend(
        (
            (_PROTOCOL_PATH, protocol_payload),
            (_ATTEMPTS_PATH, attempts_payload),
        )
    )
    return _publish_staged_bundle(Path(output_dir), payloads, root_payload)


__all__ = ["write_local_bundle"]
