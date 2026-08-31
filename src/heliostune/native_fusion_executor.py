"""Strict CPU-import-safe executor for the frozen native Triton RMSNorm suite."""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .artifacts import strict_json_loads
from .errors import ArtifactError, SchemaError
from .scope import Suite, verify_suite
from .validation import (
    exact_bool,
    exact_fields,
    exact_int,
    exact_object,
    nonblank_string,
    optional_finite_float,
    optional_nonblank_string,
)

NATIVE_RMSNORM_SUITE_SHA256 = "23f7397f2adee93cd9f7919aaf075c0f8b5e92cd6d4257ce4c54197d3c98035f"
_SCHEMA: Literal["heliostune.local_executor/2"] = "heliostune.local_executor/2"
_ENV_SCHEMA = "heliostune.local-environment/2"
_CASE = "rmsnorm-case-001"
_NATIVE = ("rmsnorm-triton-w4", "rmsnorm-triton-w8", "rmsnorm-triton-w16", "rmsnorm-triton-w32")
_COMPARATOR = "rmsnorm-inductor-comparator"
_EAGER = "rmsnorm-eager-reference"
# Serialized evidence follows the suite exactly; execution places the comparator
# before the eager arm without changing serialized custody order.
_RUNTIME_ARMS = (*_NATIVE, _EAGER, _COMPARATOR)
_EXECUTION_ARMS = (*_NATIVE, _COMPARATOR, _EAGER)
_ENTRYPOINT: Mapping[str, str] = {
    "rmsnorm-triton-w4": "heliostune_fusion_v2::residual_rmsnorm_w4",
    "rmsnorm-triton-w8": "heliostune_fusion_v2::residual_rmsnorm_w8",
    "rmsnorm-triton-w16": "heliostune_fusion_v2::residual_rmsnorm_w16",
    "rmsnorm-triton-w32": "heliostune_fusion_v2::residual_rmsnorm_w32",
    _COMPARATOR: "reference_template.residual_rmsnorm_candidate",
    _EAGER: "reference_template.residual_rmsnorm_reference",
}
_BACKEND = {**{arm: "native_triton" for arm in _NATIVE}, _COMPARATOR: "inductor", _EAGER: "eager"}
_CONFIG = {
    arm: {"block_size": 4096, "num_warps": warps, "num_stages": 1}
    for arm, warps in zip(_NATIVE, (4, 8, 16, 32), strict=True)
}
_CORRECTNESS_IDS = tuple(f"{arm}-correctness" for arm in _RUNTIME_ARMS)
_COMPILE_IDS = tuple(f"{arm}-correctness" for arm in (*_NATIVE, _COMPARATOR))
_NATIVE_IDS = tuple(f"{arm}-correctness" for arm in _NATIVE)
_CELL_IDS = tuple(cell for arm in _RUNTIME_ARMS for cell in (f"{arm}-correctness", f"{arm}-timing"))
_VALIDATION_PROBES = ("zeros", "cancellation", "overflow")
_EXECUTOR_SOURCE_NAMES = (
    "fusion_kernels.py",
    "_fusion_gpu.py",
    "native_fusion_executor.py",
    "local_executor.py",
)


def _capture_executor_sources() -> dict[str, object]:
    package_dir = Path(__file__).resolve().parent
    sources: list[dict[str, object]] = []
    for name in _EXECUTOR_SOURCE_NAMES:
        path = package_dir / name
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"cannot read installed native executor source {path}: {exc}") from exc
        sources.append(
            {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    return {"schema": "heliostune.executor-sources/1", "sources": sources}


_IMPORTED_EXECUTOR_SOURCES = _capture_executor_sources()


def _bound_executor_sources() -> dict[str, object]:
    current = _capture_executor_sources()
    if current != _IMPORTED_EXECUTOR_SOURCES:
        raise ArtifactError("native executor sources changed after module import")
    return {
        "schema": _IMPORTED_EXECUTOR_SOURCES["schema"],
        "sources": [
            dict(cast(Mapping[str, object], item))
            for item in cast(Sequence[object], _IMPORTED_EXECUTOR_SOURCES["sources"])
        ],
    }


def _parse_executor_sources(value: object) -> dict[str, object]:
    data = exact_fields(
        value, required=("schema", "sources"), context="native executor_sources"
    )
    if data["schema"] != "heliostune.executor-sources/1":
        raise SchemaError("native executor_sources schema differs")
    raw_sources = _array(data["sources"], "native executor_sources.sources")
    if len(raw_sources) != len(_EXECUTOR_SOURCE_NAMES):
        raise SchemaError("native executor_sources does not contain the exact source inventory")
    sources: list[dict[str, object]] = []
    for expected_name, raw in zip(_EXECUTOR_SOURCE_NAMES, raw_sources, strict=True):
        item = exact_fields(
            raw,
            required=("path", "bytes", "sha256"),
            context=f"native executor source {expected_name}",
        )
        if item["path"] != expected_name:
            raise SchemaError("native executor_sources order/path differs")
        sources.append(
            {
                "path": expected_name,
                "bytes": exact_int(
                    item["bytes"], context=f"native executor source {expected_name} bytes", minimum=1
                ),
                "sha256": _digest(
                    item["sha256"], f"native executor source {expected_name} SHA-256"
                ),
            }
        )
    return {"schema": "heliostune.executor-sources/1", "sources": sources}


_POLICIES = {
    "float32_matmul_precision": "highest",
    "allow_tf32": False,
    "allow_bf16_reduced_precision_reduction": False,
    "allow_fp16_reduced_precision_reduction": False,
    "allow_fp16_accumulation": False,
}
_AUTOCAST = {"device_type": "cuda", "enabled": False, "dtype": None, "cache_enabled": None}


def _array(value: object, context: str) -> list[object]:
    if type(value) is not list:
        raise SchemaError(f"{context} must be an array")
    return cast(list[object], value)


def _enum(value: object, allowed: Sequence[str], context: str) -> str:
    result = nonblank_string(value, context=context)
    if result not in allowed:
        raise SchemaError(f"{context} must be one of {tuple(allowed)!r}")
    return result


def _digest(value: object, context: str) -> str:
    result = nonblank_string(value, context=context)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise SchemaError(f"{context} must be a lowercase SHA-256 digest")
    return result


def _optional_int(value: object, context: str, minimum: int = 0) -> int | None:
    return None if value is None else exact_int(value, context=context, minimum=minimum)


def _optional_bool(value: object, context: str) -> bool | None:
    return None if value is None else exact_bool(value, context=context)


def _optional_digest(value: object, context: str) -> str | None:
    return None if value is None else _digest(value, context)


def _safe_error(exc: BaseException) -> str:
    encoded = f"{type(exc).__name__}: {exc}".encode(errors="replace")
    if len(encoded) > 4096:
        encoded = encoded[:4080] + b"...[truncated]"
    return encoded.decode(errors="replace")


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


def _compile_comparator(
    torch: Any, kernel: Callable[..., Any], state: dict[str, object]
) -> Callable[..., Any]:
    reason = _force_eager_reason(torch)
    if reason is not None:
        raise RuntimeError(reason)
    backend = _lookup_inductor_backend(torch)
    state.update(invoked=False, completed=False, error=None, callable_distinct=False)

    def recording_inductor_backend(graph_module: Any, example_inputs: Sequence[Any]) -> Any:
        state["invoked"] = True
        try:
            compiled_graph = backend(graph_module, example_inputs)
        except Exception as exc:
            state["error"] = _safe_error(exc)
            raise
        state["completed"] = True
        return compiled_graph

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
    state["callable_distinct"] = True
    return cast(Callable[..., Any], compiled)


def _names_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(names).encode()).hexdigest()


def _correctness_key(cell_id: str) -> str:
    arm_id = cell_id.removesuffix("-correctness").removesuffix("-timing")
    fields = (
        "heliostune.correctness-key/1",
        NATIVE_RMSNORM_SUITE_SHA256,
        _CASE,
        arm_id,
        "17",
        "bf16-fp32-bf16",
        "highest|tf32=0|bf16rr=0|fp16rr=0|fp16acc=0",
    )
    return hashlib.sha256("\0".join(fields).encode()).hexdigest()


def _validate_frozen_suite(suite: Suite, digest: str) -> None:
    """Authenticate every execution selector before torch/triton can be imported."""
    if digest != NATIVE_RMSNORM_SUITE_SHA256:
        raise SchemaError("native fusion suite digest is outside the frozen contract")
    if (
        suite.suite_id,
        suite.revision,
        suite.plugin_id,
        suite.plugin_version,
        suite.template_id,
        suite.domain,
    ) != (
        "residual-rmsnorm-triton",
        1,
        "fusion-triton-rmsnorm-plugin",
        1,
        "residual_rmsnorm_triton.v1",
        "rmsnorm_residual",
    ):
        raise SchemaError("native fusion suite identity/template differs")
    if len(suite.cases) != 1:
        raise SchemaError("native fusion suite must contain one case")
    case = suite.cases[0]
    if (
        case.id != _CASE
        or case.input_seed != 17
        or case.numeric_contract_id != "bf16-fp32-bf16"
        or case.shape_dict != {"tokens": 128, "hidden": 4096}
    ):
        raise SchemaError("native fusion case differs")
    suite_order = (*_NATIVE, _EAGER, _COMPARATOR)
    if tuple(arm.id for arm in suite.arms) != suite_order:
        raise SchemaError("native fusion arm order differs")
    if any(arm.entrypoint != _ENTRYPOINT[arm.id] for arm in suite.arms):
        raise SchemaError("native fusion entrypoint differs")
    expected_roles = (*("candidate" for _ in _NATIVE), "reference", "comparator")
    if tuple(arm.role for arm in suite.arms) != expected_roles:
        raise SchemaError("native fusion arm roles differ")
    if [policy.to_dict() for policy in suite.correctness_policies] != [
        {"id": "default-correctness", "reference_arm_id": _EAGER, "atol": 1e-5, "rtol": 0.0078125}
    ]:
        raise SchemaError("native fusion correctness policy differs")
    if [policy.to_dict() for policy in suite.timing_policies] != [
        {"id": "default-timing", "warmups": 10, "repetitions": 50, "statistic": "median"}
    ]:
        raise SchemaError("native fusion timing policy differs")
    expected_suite_cells = tuple(
        cell for arm in suite_order for cell in (f"{arm}-correctness", f"{arm}-timing")
    )
    if tuple(cell.id for cell in suite.expected_cells) != expected_suite_cells:
        raise SchemaError("native fusion expected cells differ")
    for cell in suite.expected_cells:
        if (
            cell.case_id != _CASE
            or cell.input_seed != 17
            or cell.correctness_policy_id != "default-correctness"
            or (cell.stage == "timing") != (cell.timing_policy_id == "default-timing")
        ):
            raise SchemaError("native fusion cell policy/linkage differs")
    if suite.executor_rule != "timing_requires_retained_passing_correctness_observation":
        raise SchemaError("native fusion timing gate differs")


def _parse_config(value: object, context: str) -> dict[str, int]:
    data = exact_fields(value, required=("block_size", "num_warps", "num_stages"), context=context)
    return {
        key: exact_int(data[key], context=f"{context}.{key}", minimum=1)
        for key in ("block_size", "num_warps", "num_stages")
    }


def _link(data: Mapping[str, object], cell_id: str, allowed: Sequence[str], context: str) -> str:
    arm = nonblank_string(data["arm_id"], context=f"{context}.arm_id")
    if arm not in allowed or cell_id != f"{arm}-correctness":
        raise SchemaError(f"{context} key/arm linkage differs")
    if (
        nonblank_string(data["case_id"], context=f"{context}.case_id") != _CASE
        or nonblank_string(data["entrypoint"], context=f"{context}.entrypoint") != _ENTRYPOINT[arm]
    ):
        raise SchemaError(f"{context} frozen linkage differs")
    return arm


def _parse_stage(value: object, cell_id: str) -> dict[str, object]:
    keys = (
        "case_id",
        "arm_id",
        "entrypoint",
        "backend_kind",
        "status",
        "failure_kind",
        "error",
        "correctness_passed",
        "resource_passed",
        "profile_passed",
        "validation_passed",
        "timing_allowed",
    )
    data = exact_fields(value, required=keys, context=f"stage_outcomes.{cell_id}")
    arm = _link(data, cell_id, _RUNTIME_ARMS, "stage outcome")
    result: dict[str, object] = {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "backend_kind": _enum(
            data["backend_kind"], ("native_triton", "inductor", "eager"), "stage backend_kind"
        ),
        "status": _enum(data["status"], ("completed", "failed", "blocked"), "stage status"),
        "failure_kind": optional_nonblank_string(
            data["failure_kind"], context="stage failure_kind"
        ),
        "error": optional_nonblank_string(data["error"], context="stage error"),
        "correctness_passed": exact_bool(
            data["correctness_passed"], context="stage correctness_passed"
        ),
        "resource_passed": _optional_bool(data["resource_passed"], "stage resource_passed"),
        "profile_passed": _optional_bool(data["profile_passed"], "stage profile_passed"),
        "validation_passed": _optional_bool(data["validation_passed"], "stage validation_passed"),
        "timing_allowed": exact_bool(data["timing_allowed"], context="stage timing_allowed"),
    }
    if result["backend_kind"] != _BACKEND[arm]:
        raise SchemaError("stage backend differs from closed arm map")
    native = arm in _NATIVE
    native_gates = (
        result["resource_passed"],
        result["profile_passed"],
        result["validation_passed"],
    )
    if native != all(item is not None for item in native_gates):
        raise SchemaError("native-only stage gate nullability differs")
    gate = bool(result["correctness_passed"])
    if native:
        gate = gate and all(item is True for item in native_gates)
    if result["timing_allowed"] is not gate:
        raise SchemaError("stage timing gate differs")
    if result["status"] == "completed":
        if not gate or result["failure_kind"] is not None or result["error"] is not None:
            raise SchemaError("completed stage lacks exact passing evidence")
    elif gate or result["failure_kind"] is None or result["error"] is None:
        raise SchemaError("non-completed stage lacks exact failure evidence")
    return result


def _parse_compile(value: object, cell_id: str) -> dict[str, object]:
    keys = (
        "case_id",
        "arm_id",
        "entrypoint",
        "backend_kind",
        "status",
        "error",
        "compile_ns",
        "backend_invoked",
        "fullgraph",
        "dynamic",
        "mode",
        "callable_distinct",
        "eager_fallback",
        "kernel_name",
        "kernel_hash",
        "config",
    )
    data = exact_fields(value, required=keys, context=f"compile_evidence.{cell_id}")
    arm = _link(data, cell_id, (*_NATIVE, _COMPARATOR), "compile evidence")
    config = None if data["config"] is None else _parse_config(data["config"], "compile config")
    result: dict[str, object] = {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "backend_kind": _enum(
            data["backend_kind"], ("native_triton", "inductor"), "compile backend_kind"
        ),
        "status": _enum(data["status"], ("compiled", "failed", "blocked"), "compile status"),
        "error": optional_nonblank_string(data["error"], context="compile error"),
        "compile_ns": _optional_int(data["compile_ns"], "compile_ns"),
        "backend_invoked": exact_bool(data["backend_invoked"], context="compile backend_invoked"),
        "fullgraph": exact_bool(data["fullgraph"], context="compile fullgraph"),
        "dynamic": exact_bool(data["dynamic"], context="compile dynamic"),
        "mode": nonblank_string(data["mode"], context="compile mode"),
        "callable_distinct": exact_bool(
            data["callable_distinct"], context="compile callable_distinct"
        ),
        "eager_fallback": exact_bool(data["eager_fallback"], context="compile eager_fallback"),
        "kernel_name": optional_nonblank_string(data["kernel_name"], context="compile kernel_name"),
        "kernel_hash": _optional_digest(data["kernel_hash"], "compile kernel_hash"),
        "config": config,
    }
    native = arm in _NATIVE
    if (
        result["backend_kind"] != _BACKEND[arm]
        or result["dynamic"] is not False
        or result["eager_fallback"] is not False
        or (
            native
            and (
                result["fullgraph"] is not False
                or result["mode"] != "native_triton"
                or config != _CONFIG[arm]
            )
        )
        or (
            not native
            and (
                result["fullgraph"] is not True or result["mode"] != "default" or config is not None
            )
        )
    ):
        raise SchemaError("compile policy/config differs")
    if result["status"] == "compiled":
        if (
            result["error"] is not None
            or result["compile_ns"] is None
            or result["backend_invoked"] is not True
            or result["callable_distinct"] is not True
            or (native and (result["kernel_name"] is None or result["kernel_hash"] is None))
            or (
                not native
                and (result["kernel_name"] is not None or result["kernel_hash"] is not None)
            )
        ):
            raise SchemaError("compiled evidence lacks exact passing evidence")
    elif result["status"] == "failed":
        if (
            result["error"] is None
            or result["compile_ns"] is None
            or result["kernel_name"] is not None
            or result["kernel_hash"] is not None
            or (native and (result["backend_invoked"] or result["callable_distinct"]))
        ):
            raise SchemaError("failed compile evidence is inconsistent")
    elif (
        result["error"] is None
        or result["compile_ns"] is not None
        or result["backend_invoked"]
        or result["callable_distinct"]
        or result["kernel_name"] is not None
        or result["kernel_hash"] is not None
    ):
        raise SchemaError("blocked compile evidence contains compile claims")
    return result


def _parse_target(value: object) -> dict[str, object]:
    data = exact_fields(value, required=("backend", "arch", "warp_size"), context="resource target")
    result: dict[str, object] = {
        "backend": nonblank_string(data["backend"], context="target backend"),
        "arch": nonblank_string(data["arch"], context="target arch"),
        "warp_size": exact_int(data["warp_size"], context="target warp_size", minimum=1),
    }
    if result != {"backend": "cuda", "arch": "90", "warp_size": 32}:
        raise SchemaError("resource target is not pinned Triton H100 cuda/90/32")
    return result


def _parse_metadata(value: object) -> dict[str, int]:
    data = exact_fields(
        value,
        required=("shared", "num_warps", "num_ctas", "num_stages"),
        context="resource metadata",
    )
    return {
        "shared": exact_int(data["shared"], context="metadata shared", minimum=0),
        "num_warps": exact_int(data["num_warps"], context="metadata num_warps", minimum=1),
        "num_ctas": exact_int(data["num_ctas"], context="metadata num_ctas", minimum=1),
        "num_stages": exact_int(data["num_stages"], context="metadata num_stages", minimum=1),
    }


def _parse_asm(value: object) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for raw in _array(value, "asm_stages"):
        data = exact_fields(raw, required=("stage", "bytes", "sha256"), context="asm stage")
        result.append(
            {
                "stage": nonblank_string(data["stage"], context="asm stage name"),
                "bytes": exact_int(data["bytes"], context="asm stage bytes", minimum=0),
                "sha256": _digest(data["sha256"], "asm stage sha256"),
            }
        )
    names = [cast(str, item["stage"]) for item in result]
    if names != sorted(names) or len(names) != len(set(names)):
        raise SchemaError("asm stages must be unique and sorted")
    return result


def _parse_resource(value: object, cell_id: str) -> dict[str, object]:
    keys = (
        "case_id",
        "arm_id",
        "entrypoint",
        "status",
        "error",
        "kernel_name",
        "kernel_hash",
        "target",
        "metadata",
        "n_regs",
        "n_spills",
        "n_max_threads",
        "asm_stages",
        "resource_gate_passed",
    )
    data = exact_fields(value, required=keys, context=f"resource_evidence.{cell_id}")
    arm = _link(data, cell_id, _NATIVE, "resource evidence")
    result: dict[str, object] = {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": _enum(data["status"], ("compiled", "failed", "blocked"), "resource status"),
        "error": optional_nonblank_string(data["error"], context="resource error"),
        "kernel_name": optional_nonblank_string(
            data["kernel_name"], context="resource kernel_name"
        ),
        "kernel_hash": _optional_digest(data["kernel_hash"], "resource kernel_hash"),
        "target": None if data["target"] is None else _parse_target(data["target"]),
        "metadata": None if data["metadata"] is None else _parse_metadata(data["metadata"]),
        "n_regs": _optional_int(data["n_regs"], "resource n_regs"),
        "n_spills": _optional_int(data["n_spills"], "resource n_spills"),
        "n_max_threads": _optional_int(data["n_max_threads"], "resource n_max_threads", 1),
        "asm_stages": _parse_asm(data["asm_stages"]),
        "resource_gate_passed": exact_bool(data["resource_gate_passed"], context="resource gate"),
    }
    complete = (
        result["kernel_name"] is not None
        and result["kernel_hash"] is not None
        and result["target"] is not None
        and result["metadata"] is not None
        and result["n_regs"] is not None
        and result["n_spills"] == 0
        and result["n_max_threads"] is not None
        and bool(result["asm_stages"])
    )
    if result["status"] == "compiled":
        metadata = cast(Mapping[str, object], result["metadata"])
        if (
            result["error"] is not None
            or result["resource_gate_passed"] is not True
            or not complete
            or metadata["num_warps"] != _CONFIG[arm]["num_warps"]
            or metadata["num_ctas"] != 1
            or metadata["num_stages"] != 1
        ):
            raise SchemaError("compiled resource lacks zero-spill exact-config evidence")
    elif result["error"] is None or result["resource_gate_passed"] is not False or complete:
        raise SchemaError("non-completed resource evidence is inconsistent")
    if result["status"] == "blocked" and (
        any(
            result[key] is not None
            for key in (
                "kernel_name",
                "kernel_hash",
                "target",
                "metadata",
                "n_regs",
                "n_spills",
                "n_max_threads",
            )
        )
        or result["asm_stages"] != []
        or result["resource_gate_passed"] is not False
    ):
        raise SchemaError("blocked resource evidence contains resource claims")
    return result


def _parse_profile(value: object, cell_id: str) -> dict[str, object]:
    keys = (
        "case_id",
        "arm_id",
        "entrypoint",
        "status",
        "error",
        "method",
        "warmed",
        "expected_kernel_name",
        "expected_kernel_hash",
        "config",
        "invocation_count",
        "cuda_event_count",
        "cuda_event_names_sample",
        "cuda_event_names_sha256",
        "exact_name_match_count",
        "output_revalidated",
        "inputs_revalidated",
        "one_kernel_gate_passed",
    )
    data = exact_fields(value, required=keys, context=f"profile_evidence.{cell_id}")
    arm = _link(data, cell_id, _NATIVE, "profile evidence")
    sample = [
        nonblank_string(item, context="profile event name")
        for item in _array(data["cuda_event_names_sample"], "profile event sample")
    ]
    if len(sample) > 32:
        raise SchemaError("profile event sample exceeds 32")
    invocation_count = exact_int(
        data["invocation_count"], context="profile invocation_count", minimum=0
    )
    event_count = exact_int(data["cuda_event_count"], context="profile cuda_event_count", minimum=0)
    match_count = exact_int(
        data["exact_name_match_count"], context="profile match count", minimum=0
    )
    result: dict[str, object] = {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": _enum(data["status"], ("profiled", "failed", "blocked"), "profile status"),
        "error": optional_nonblank_string(data["error"], context="profile error"),
        "method": nonblank_string(data["method"], context="profile method"),
        "warmed": exact_bool(data["warmed"], context="profile warmed"),
        "expected_kernel_name": optional_nonblank_string(
            data["expected_kernel_name"], context="expected kernel name"
        ),
        "expected_kernel_hash": _optional_digest(
            data["expected_kernel_hash"], "expected kernel hash"
        ),
        "config": _parse_config(data["config"], "profile config"),
        "invocation_count": invocation_count,
        "cuda_event_count": event_count,
        "cuda_event_names_sample": sample,
        "cuda_event_names_sha256": _optional_digest(
            data["cuda_event_names_sha256"], "profile names sha256"
        ),
        "exact_name_match_count": match_count,
        "output_revalidated": exact_bool(
            data["output_revalidated"], context="profile output_revalidated"
        ),
        "inputs_revalidated": exact_bool(
            data["inputs_revalidated"], context="profile inputs_revalidated"
        ),
        "one_kernel_gate_passed": exact_bool(
            data["one_kernel_gate_passed"], context="profile gate"
        ),
    }
    if result["method"] != "torch.profiler.cuda_events" or result["config"] != _CONFIG[arm]:
        raise SchemaError("profile method/config differs")
    if event_count < len(sample):
        raise SchemaError("profile sample exceeds total CUDA event count")
    if event_count == 0:
        if sample or result["cuda_event_names_sha256"] is not None:
            raise SchemaError("empty profile must not claim CUDA event names")
    elif event_count <= 32 and (
        len(sample) != event_count
        or result["cuda_event_names_sha256"] != _names_digest(sample)
        or (
            result["expected_kernel_name"] is not None
            and result["exact_name_match_count"]
            != sample.count(cast(str, result["expected_kernel_name"]))
        )
    ):
        raise SchemaError("profile full event sample/digest/match count differs")
    complete = (
        result["warmed"] is True
        and result["expected_kernel_name"] is not None
        and result["expected_kernel_hash"] is not None
        and invocation_count == 1
        and event_count == 1
        and sample == [result["expected_kernel_name"]]
        and result["cuda_event_names_sha256"] == _names_digest(sample)
        and match_count == 1
        and result["output_revalidated"] is True
        and result["inputs_revalidated"] is True
    )
    if result["status"] == "profiled":
        if (
            result["error"] is not None
            or result["one_kernel_gate_passed"] is not True
            or not complete
        ):
            raise SchemaError("profile lacks exact one-kernel passing evidence")
    elif result["error"] is None or result["one_kernel_gate_passed"] is not False or complete:
        raise SchemaError("non-completed profile evidence is inconsistent")
    if result["status"] == "blocked" and (
        result["warmed"] is not False
        or invocation_count != 0
        or event_count != 0
        or sample
        or result["cuda_event_names_sha256"] is not None
        or match_count != 0
        or result["output_revalidated"] is not False
        or result["inputs_revalidated"] is not False
    ):
        raise SchemaError("blocked profile evidence contains profile claims")
    return result


def _parse_validation_probe(value: object, context: str) -> dict[str, object]:
    keys = (
        "id",
        "passed",
        "deterministic",
        "inputs_unchanged",
        "output_disjoint",
        "value_class_match",
        "sign_match",
        "finite_close",
        "max_abs_error",
    )
    data = exact_fields(value, required=keys, context=context)
    result: dict[str, object] = {
        "id": _enum(data["id"], _VALIDATION_PROBES, f"{context}.id"),
        "passed": exact_bool(data["passed"], context=f"{context}.passed"),
        "deterministic": exact_bool(data["deterministic"], context=f"{context}.deterministic"),
        "inputs_unchanged": exact_bool(
            data["inputs_unchanged"], context=f"{context}.inputs_unchanged"
        ),
        "output_disjoint": exact_bool(
            data["output_disjoint"], context=f"{context}.output_disjoint"
        ),
        "value_class_match": exact_bool(
            data["value_class_match"], context=f"{context}.value_class_match"
        ),
        "sign_match": exact_bool(data["sign_match"], context=f"{context}.sign_match"),
        "finite_close": exact_bool(data["finite_close"], context=f"{context}.finite_close"),
        "max_abs_error": optional_finite_float(
            data["max_abs_error"], context=f"{context}.max_abs_error", minimum=0
        ),
    }
    checks = (
        result["deterministic"],
        result["inputs_unchanged"],
        result["output_disjoint"],
        result["value_class_match"],
        result["sign_match"],
        result["finite_close"],
    )
    if result["passed"] is not all(item is True for item in checks):
        raise SchemaError(f"{context} passed flag differs from probe evidence")
    return result


def _parse_validation(value: object, cell_id: str) -> dict[str, object]:
    keys = (
        "case_id",
        "arm_id",
        "entrypoint",
        "status",
        "error",
        "probes",
        "validation_gate_passed",
    )
    data = exact_fields(value, required=keys, context=f"validation_evidence.{cell_id}")
    arm = _link(data, cell_id, _NATIVE, "validation evidence")
    probes = [
        _parse_validation_probe(item, f"validation_evidence.{cell_id}.probes")
        for item in _array(data["probes"], f"validation_evidence.{cell_id}.probes")
    ]
    if tuple(item["id"] for item in probes) != _VALIDATION_PROBES:
        raise SchemaError("validation probes must contain zeros/cancellation/overflow in order")
    result: dict[str, object] = {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": _enum(data["status"], ("validated", "failed", "blocked"), "validation status"),
        "error": optional_nonblank_string(data["error"], context="validation error"),
        "probes": probes,
        "validation_gate_passed": exact_bool(
            data["validation_gate_passed"], context="validation gate"
        ),
    }
    gate = all(item["passed"] is True for item in probes)
    if result["status"] == "validated":
        if result["error"] is not None or result["validation_gate_passed"] is not True or not gate:
            raise SchemaError("validated evidence lacks all passing probes")
    elif result["error"] is None or result["validation_gate_passed"] is not False or gate:
        raise SchemaError("non-completed validation evidence is inconsistent")
    if result["status"] == "blocked" and any(
        probe["passed"] is not False
        or probe["deterministic"] is not False
        or probe["inputs_unchanged"] is not False
        or probe["output_disjoint"] is not False
        or probe["value_class_match"] is not False
        or probe["sign_match"] is not False
        or probe["finite_close"] is not False
        or probe["max_abs_error"] is not None
        for probe in probes
    ):
        raise SchemaError("blocked validation contains probe claims")
    return result


def _parse_environment(value: object, capability: Any) -> dict[str, object]:
    keys = (
        "schema",
        "python",
        "implementation",
        "platform",
        "torch_version",
        "triton_version",
        "cuda_version",
        "rocm_version",
        "device_index",
        "device_name",
        "compute_capability",
        "precision_policy",
        "autocast_policy",
        "backend_invoked",
        "fusion_claim",
    )
    data = exact_fields(value, required=keys, context="native environment")
    compute = (
        None
        if data["compute_capability"] is None
        else [
            exact_int(item, context="compute capability", minimum=0)
            for item in _array(data["compute_capability"], "compute capability")
        ]
    )
    if compute is not None and len(compute) != 2:
        raise SchemaError("compute capability must contain two integers")
    result: dict[str, object] = {
        "schema": nonblank_string(data["schema"], context="environment schema"),
        "python": nonblank_string(data["python"], context="environment python"),
        "implementation": nonblank_string(
            data["implementation"], context="environment implementation"
        ),
        "platform": nonblank_string(data["platform"], context="environment platform"),
        "torch_version": optional_nonblank_string(
            data["torch_version"], context="environment torch_version"
        ),
        "triton_version": optional_nonblank_string(
            data["triton_version"], context="environment triton_version"
        ),
        "cuda_version": optional_nonblank_string(
            data["cuda_version"], context="environment cuda_version"
        ),
        "rocm_version": optional_nonblank_string(
            data["rocm_version"], context="environment rocm_version"
        ),
        "device_index": _optional_int(data["device_index"], "environment device_index"),
        "device_name": optional_nonblank_string(
            data["device_name"], context="environment device_name"
        ),
        "compute_capability": compute,
        "precision_policy": dict(
            exact_object(data["precision_policy"], context="precision policy")
        ),
        "autocast_policy": dict(exact_object(data["autocast_policy"], context="autocast policy")),
        "backend_invoked": _optional_bool(data["backend_invoked"], "environment backend_invoked"),
        "fusion_claim": exact_bool(data["fusion_claim"], context="environment fusion_claim"),
    }
    capability_link = (
        capability.torch_version,
        capability.cuda_version,
        capability.rocm_version,
        capability.device_index,
        capability.device_name,
        None if capability.compute_capability is None else list(capability.compute_capability),
    )
    if (
        result["schema"] != _ENV_SCHEMA
        or result["precision_policy"] != _POLICIES
        or result["autocast_policy"] != _AUTOCAST
        or result["fusion_claim"] is not False
        or tuple(
            result[key]
            for key in (
                "torch_version",
                "cuda_version",
                "rocm_version",
                "device_index",
                "device_name",
                "compute_capability",
            )
        )
        != capability_link
    ):
        raise SchemaError("environment policy/capability linkage differs")
    if capability.available and (
        result["triton_version"] != "3.4.0"
        or result["compute_capability"] != [9, 0]
        or "H100" not in cast(str, result["device_name"])
    ):
        raise SchemaError("available environment is not pinned H100/Triton 3.4")
    return result


def _parse_attempts(value: object) -> tuple[Mapping[str, object], ...]:
    from .local_executor import _parse_attempt

    return tuple(_parse_attempt(item) for item in _array(value, "native attempts"))


def _counts(
    stage: Mapping[str, Mapping[str, object]],
    compile_evidence: Mapping[str, Mapping[str, object]],
    resource: Mapping[str, Mapping[str, object]],
    profile: Mapping[str, Mapping[str, object]],
    validation: Mapping[str, Mapping[str, object]],
) -> dict[str, int]:
    return {
        "stage_completed": sum(item["status"] == "completed" for item in stage.values()),
        "stage_failed": sum(item["status"] == "failed" for item in stage.values()),
        "stage_blocked": sum(item["status"] == "blocked" for item in stage.values()),
        "compile_compiled": sum(item["status"] == "compiled" for item in compile_evidence.values()),
        "compile_failed": sum(item["status"] == "failed" for item in compile_evidence.values()),
        "compile_blocked": sum(item["status"] == "blocked" for item in compile_evidence.values()),
        "resource_passed": sum(item["status"] == "compiled" for item in resource.values()),
        "resource_failed": sum(item["status"] == "failed" for item in resource.values()),
        "resource_blocked": sum(item["status"] == "blocked" for item in resource.values()),
        "profile_passed": sum(item["status"] == "profiled" for item in profile.values()),
        "profile_failed": sum(item["status"] == "failed" for item in profile.values()),
        "profile_blocked": sum(item["status"] == "blocked" for item in profile.values()),
        "validation_passed": sum(item["status"] == "validated" for item in validation.values()),
        "validation_failed": sum(item["status"] == "failed" for item in validation.values()),
        "validation_blocked": sum(item["status"] == "blocked" for item in validation.values()),
    }


def _parse_summary(
    value: object,
    observations: Sequence[Any],
    outcome: str,
    expected_counts: Mapping[str, int],
) -> dict[str, object]:
    keys = (
        "expected_cell_ids",
        "terminal_cell_ids",
        "passed",
        "failed",
        "blocked",
        "all_cells_terminal",
        "counts",
        "outcome",
        "fusion_claim",
    )
    data = exact_fields(value, required=keys, context="native summary")
    count_data = exact_fields(
        data["counts"], required=tuple(expected_counts), context="native summary counts"
    )
    counts = {
        key: exact_int(count_data[key], context=f"summary counts.{key}", minimum=0)
        for key in expected_counts
    }
    statuses = [item.status for item in observations]
    result = {
        "expected_cell_ids": [
            nonblank_string(item, context="expected cell id")
            for item in _array(data["expected_cell_ids"], "expected cell ids")
        ],
        "terminal_cell_ids": [
            nonblank_string(item, context="terminal cell id")
            for item in _array(data["terminal_cell_ids"], "terminal cell ids")
        ],
        "passed": exact_int(data["passed"], context="summary passed", minimum=0),
        "failed": exact_int(data["failed"], context="summary failed", minimum=0),
        "blocked": exact_int(data["blocked"], context="summary blocked", minimum=0),
        "all_cells_terminal": exact_bool(
            data["all_cells_terminal"], context="summary all_cells_terminal"
        ),
        "counts": counts,
        "outcome": _enum(data["outcome"], ("completed", "failed", "aborted"), "summary outcome"),
        "fusion_claim": exact_bool(data["fusion_claim"], context="summary fusion_claim"),
    }
    if (
        result["expected_cell_ids"] != list(_CELL_IDS)
        or result["terminal_cell_ids"] != [item.cell_id for item in observations]
        or result["passed"] != statuses.count("passed")
        or result["failed"] != statuses.count("failed")
        or result["blocked"] != statuses.count("blocked")
        or result["all_cells_terminal"] is not (len(observations) == len(_CELL_IDS))
        or counts != expected_counts
        or result["outcome"] != outcome
        or result["fusion_claim"] is not False
    ):
        raise SchemaError("summary does not match native observations/evidence")
    return result


@dataclass(frozen=True, slots=True)
class NativeFusionExecutionResult:
    schema: Literal["heliostune.local_executor/2"]
    verified_suite_path: str
    verified_suite_sha256: str
    verified_suite_bytes: bytes
    suite_id: str
    capability: Any
    materialization: tuple[Any, ...]
    observations: tuple[Any, ...]
    attempts: tuple[Mapping[str, object], ...]
    environment: Mapping[str, object]
    stage_outcomes: Mapping[str, Mapping[str, object]]
    compile_evidence: Mapping[str, Mapping[str, object]]
    resource_evidence: Mapping[str, Mapping[str, object]]
    profile_evidence: Mapping[str, Mapping[str, object]]
    validation_evidence: Mapping[str, Mapping[str, object]]
    executor_sources: Mapping[str, object]
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
        result: dict[str, object] = {
            "schema": self.schema,
            "verified_suite_path": self.verified_suite_path,
            "verified_suite_sha256": self.verified_suite_sha256,
            "suite_id": self.suite_id,
            "capability": self.capability.to_dict(),
            "materialization": [item.to_dict() for item in self.materialization],
            "observations": [item.to_dict() for item in self.observations],
            "attempts": [dict(item) for item in self.attempts],
            "environment": dict(self.environment),
            "stage_outcomes": {key: dict(item) for key, item in self.stage_outcomes.items()},
            "compile_evidence": {key: dict(item) for key, item in self.compile_evidence.items()},
            "resource_evidence": {key: dict(item) for key, item in self.resource_evidence.items()},
            "profile_evidence": {key: dict(item) for key, item in self.profile_evidence.items()},
            "validation_evidence": {
                key: dict(item) for key, item in self.validation_evidence.items()
            },
            "executor_sources": {
                "schema": self.executor_sources["schema"],
                "sources": [
                    dict(cast(Mapping[str, object], item))
                    for item in cast(Sequence[object], self.executor_sources["sources"])
                ],
            },
            "summary": dict(self.summary),
            "outcome": self.outcome,
        }
        if include_suite_bytes:
            result["verified_suite_bytes"] = self.verified_suite_bytes
        return result

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        verified_suite_path: str,
        verified_suite_sha256: str,
        verified_suite_bytes: bytes,
    ) -> NativeFusionExecutionResult:
        path = nonblank_string(verified_suite_path, context="verified native suite path")
        digest = _digest(verified_suite_sha256, "verified native suite SHA-256")
        if (
            type(verified_suite_bytes) is not bytes
            or hashlib.sha256(verified_suite_bytes).hexdigest() != digest
        ):
            raise SchemaError("verified native suite bytes do not match their SHA-256")
        try:
            suite = Suite.from_dict(
                strict_json_loads(verified_suite_bytes.decode(), source=Path(path))
            )
        except UnicodeError as exc:
            raise SchemaError("verified native suite bytes must be UTF-8") from exc
        _validate_frozen_suite(suite, digest)
        keys = (
            "schema",
            "verified_suite_path",
            "verified_suite_sha256",
            "suite_id",
            "capability",
            "materialization",
            "observations",
            "attempts",
            "environment",
            "stage_outcomes",
            "compile_evidence",
            "resource_evidence",
            "profile_evidence",
            "validation_evidence",
            "executor_sources",
            "summary",
            "outcome",
        )
        data = exact_fields(value, required=keys, context="native fusion result")
        if data["schema"] != _SCHEMA:
            raise SchemaError("native fusion result schema differs")
        nonblank_string(data["verified_suite_path"], context="serialized suite path")
        if (
            _digest(data["verified_suite_sha256"], "serialized suite digest") != digest
            or data["suite_id"] != suite.suite_id
        ):
            raise SchemaError("serialized native suite binding differs")
        from .local_executor import CapabilityProbe, CellObservation, TensorMaterialization

        capability = CapabilityProbe.from_dict(data["capability"])
        materialization = tuple(
            TensorMaterialization.from_dict(item)
            for item in _array(data["materialization"], "materialization")
        )
        observations = tuple(
            CellObservation.from_dict(item) for item in _array(data["observations"], "observations")
        )
        attempts = _parse_attempts(data["attempts"])
        environment = _parse_environment(data["environment"], capability)
        stage_raw = exact_object(data["stage_outcomes"], context="stage_outcomes")
        compile_raw = exact_object(data["compile_evidence"], context="compile_evidence")
        resource_raw = exact_object(data["resource_evidence"], context="resource_evidence")
        profile_raw = exact_object(data["profile_evidence"], context="profile_evidence")
        validation_raw = exact_object(data["validation_evidence"], context="validation_evidence")
        executor_sources = _parse_executor_sources(data["executor_sources"])
        if (
            set(stage_raw) != set(_CORRECTNESS_IDS)
            or set(compile_raw) != set(_COMPILE_IDS)
            or set(resource_raw) != set(_NATIVE_IDS)
            or set(profile_raw) != set(_NATIVE_IDS)
            or set(validation_raw) != set(_NATIVE_IDS)
        ):
            raise SchemaError("native evidence mappings do not have exact frozen cell IDs")
        stage = {cell: _parse_stage(stage_raw[cell], cell) for cell in _CORRECTNESS_IDS}
        compile_evidence = {cell: _parse_compile(compile_raw[cell], cell) for cell in _COMPILE_IDS}
        resource = {cell: _parse_resource(resource_raw[cell], cell) for cell in _NATIVE_IDS}
        profile = {cell: _parse_profile(profile_raw[cell], cell) for cell in _NATIVE_IDS}
        validation = {cell: _parse_validation(validation_raw[cell], cell) for cell in _NATIVE_IDS}
        outcome = cast(
            Literal["completed", "failed", "aborted"],
            _enum(data["outcome"], ("completed", "failed", "aborted"), "native outcome"),
        )
        counts = _counts(stage, compile_evidence, resource, profile, validation)
        summary = _parse_summary(data["summary"], observations, outcome, counts)
        _validate_cross_links(
            capability,
            materialization,
            observations,
            attempts,
            environment,
            stage,
            compile_evidence,
            resource,
            profile,
            validation,
            outcome,
        )
        return cls(
            _SCHEMA,
            path,
            digest,
            verified_suite_bytes,
            suite.suite_id,
            capability,
            materialization,
            observations,
            attempts,
            environment,
            stage,
            compile_evidence,
            resource,
            profile,
            validation,
            executor_sources,
            summary,
            outcome,
        )



def _derived_stage_failure(
    cell: str,
    arm: str,
    correctness: Any,
    compile_evidence: Mapping[str, Mapping[str, object]],
    resource: Mapping[str, Mapping[str, object]],
    validation: Mapping[str, Mapping[str, object]],
    profile: Mapping[str, Mapping[str, object]],
) -> tuple[str | None, str | None]:
    if arm != _EAGER and compile_evidence[cell]["status"] != "compiled":
        return "compile_failed", cast(str, compile_evidence[cell]["error"])
    if arm in _NATIVE and resource[cell]["resource_gate_passed"] is not True:
        return "resource_gate", cast(str, resource[cell]["error"])
    if correctness.status != "passed":
        nested = cast(Any, correctness.correctness)
        return cast(str, nested.failure_kind), cast(str, nested.message)
    if arm in _NATIVE and validation[cell]["validation_gate_passed"] is not True:
        return "validation_gate", cast(str, validation[cell]["error"])
    if arm in _NATIVE and profile[cell]["one_kernel_gate_passed"] is not True:
        return "profile_gate", cast(str, profile[cell]["error"])
    return None, None

def _validate_cross_links(
    capability: Any,
    materialization: Sequence[Any],
    observations: Sequence[Any],
    attempts: Sequence[Mapping[str, object]],
    environment: Mapping[str, object],
    stage: Mapping[str, Mapping[str, object]],
    compile_evidence: Mapping[str, Mapping[str, object]],
    resource: Mapping[str, Mapping[str, object]],
    profile: Mapping[str, Mapping[str, object]],
    validation: Mapping[str, Mapping[str, object]],
    outcome: str,
) -> None:
    if tuple(item.cell_id for item in observations) != _CELL_IDS:
        raise SchemaError("observations do not follow frozen runtime order")
    materialized_arms = tuple(item.arm_id for item in materialization)
    if capability.available:
        if materialized_arms != _RUNTIME_ARMS[: len(materialized_arms)]:
            raise SchemaError("materializations must be a suite-order prefix")
    elif materialization:
        raise SchemaError("capability-rejected execution must not claim materialization")
    for record in materialization:
        if (
            record.suite_sha256 != NATIVE_RMSNORM_SUITE_SHA256
            or record.case_id != _CASE
            or record.input_seed != 17
            or record.tensor_order != ("input", "residual", "gamma")
            or len(record.tensors) != 3
        ):
            raise SchemaError("materialization linkage/order differs")
        for descriptor, tensor_id, role, shape, scale, offset in zip(
            record.tensors,
            ("input", "residual", "gamma"),
            ("input", "input", "parameter"),
            ((128, 4096), (128, 4096), (4096,)),
            (1.0, 1.0, 0.02),
            (0.0, 0.0, 1.0),
            strict=True,
        ):
            expected_descriptor = {
                "tensor_id": tensor_id,
                "role": role,
                "shape": list(shape),
                "draw": "normal_0_1_fp32_cpu",
                "normal_scale": scale,
                "normal_offset": offset,
                "cpu_dtype": "float32",
                "storage_dtype": "bfloat16",
                "device": f"cuda:{capability.device_index}",
                "contiguous": True,
                "alignment_bytes": 16,
                "alignment_satisfied": True,
                "storage_sha256": descriptor["storage_sha256"],
            }
            if descriptor != expected_descriptor:
                raise SchemaError("materialization tensor descriptor differs")
    if any(
        observation.stage == "correctness"
        and observation.status == "passed"
        and observation.arm_id not in materialized_arms
        for observation in observations
    ):
        raise SchemaError("passing correctness lacks materialization")
    if materialization:
        reference_record = (
            next(item for item in materialization if item.arm_id == _EAGER)
            if _EAGER in materialized_arms
            else materialization[0]
        )
        reference_hashes = {
            descriptor["tensor_id"]: descriptor["storage_sha256"]
            for descriptor in reference_record.tensors
        }
        for record in materialization:
            if {
                descriptor["tensor_id"]: descriptor["storage_sha256"]
                for descriptor in record.tensors
            } != reference_hashes:
                raise SchemaError("cross-arm materialization hashes differ from reference")
    if len(attempts) != 2 * len(observations) or [item["attempt_id"] for item in attempts] != list(
        range(1, len(attempts) + 1)
    ):
        raise SchemaError("attempt IDs/count are not exact and contiguous")
    by_id = {item.cell_id: item for item in observations}
    for cell, item in stage.items():
        arm = cast(str, item["arm_id"])
        if item["correctness_passed"] is not (by_id[cell].status == "passed"):
            raise SchemaError("stage/correctness linkage differs")
        if arm in _NATIVE:
            if (
                item["resource_passed"] is not (resource[cell]["resource_gate_passed"] is True)
                or item["profile_passed"] is not (profile[cell]["one_kernel_gate_passed"] is True)
                or item["validation_passed"]
                is not (validation[cell]["validation_gate_passed"] is True)
            ):
                raise SchemaError("stage native gate linkage differs")
            names = (
                compile_evidence[cell]["kernel_name"],
                resource[cell]["kernel_name"],
                profile[cell]["expected_kernel_name"],
            )
            hashes = (
                compile_evidence[cell]["kernel_hash"],
                resource[cell]["kernel_hash"],
                profile[cell]["expected_kernel_hash"],
            )
            if len(set(names)) != 1 or len(set(hashes)) != 1:
                raise SchemaError("compile/resource/profile kernel identity linkage differs")
        if capability.available:
            expected_failure = _derived_stage_failure(
                cell,
                arm,
                by_id[cell],
                compile_evidence,
                resource,
                validation,
                profile,
            )
            if (
                item["failure_kind"],
                item["error"],
            ) != expected_failure:
                raise SchemaError("stage failure kind/error differs from underlying evidence")
        if (
            arm != _EAGER
            and item["timing_allowed"] is True
            and compile_evidence[cell]["status"] != "compiled"
        ):
            raise SchemaError("timing allowed without completed compile evidence")
        if not item["timing_allowed"] and by_id[f"{arm}-timing"].status == "passed":
            raise SchemaError("timing passed through a closed gate")
    for index, observation in enumerate(observations):
        arm = observation.arm_id
        expected_key = _correctness_key(f"{arm}-correctness")
        if observation.case_id != _CASE or observation.cell_id != f"{arm}-{observation.stage}":
            raise SchemaError("observation cell linkage differs")
        nested = (
            observation.correctness if observation.stage == "correctness" else observation.timing
        )
        if nested.correctness_key != expected_key:
            raise SchemaError("observation correctness key differs from /1 formula")
        if observation.stage == "correctness" and observation.status == "passed":
            expected_output = {
                "shape": [128, 4096],
                "device": f"cuda:{capability.device_index}",
                "dtype": "torch.bfloat16",
                "layout": "torch.strided",
                "contiguous": True,
            }
            if observation.correctness.output != expected_output:
                raise SchemaError("passing correctness output descriptor differs")
        if observation.stage == "timing":
            timing = observation.timing
            if timing.status == "passed" and (
                timing.warmups != 10 or timing.repetitions != 50 or len(timing.samples_ms) != 50
            ):
                raise SchemaError("passing timing does not use exact 10/50 policy")
            if timing.status == "blocked" and (
                timing.warmups != 0
                or timing.repetitions != 0
                or timing.samples_ms
                or timing.median_ms is not None
            ):
                raise SchemaError("blocked timing contains timing claims")
        running, terminal = attempts[2 * index : 2 * index + 2]
        expected_target = "passed" if observation.status == "passed" else "failed"
        expected_reason = None if observation.status == "passed" else nested.failure_kind
        if (
            running["cell_id"] != observation.cell_id
            or terminal["cell_id"] != observation.cell_id
            or running["stage"] != observation.stage
            or terminal["stage"] != observation.stage
            or running["from_state"] != "pending"
            or running["to_state"] != "running"
            or running["status"] != "running"
            or running["reason"] is not None
            or terminal["from_state"] != "running"
            or terminal["to_state"] != expected_target
            or terminal["status"] != ("success" if expected_target == "passed" else "failure")
            or terminal["reason"] != expected_reason
        ):
            raise SchemaError("attempt transition/reason does not match observation")
    available = bool(capability.available)
    if not available:
        capability_message = str(
            capability.detail or capability.reason or "native capability unavailable"
        )
        for observation in observations:
            if observation.status != "blocked":
                raise SchemaError("capability-rejected observation must be blocked")
            if observation.stage == "correctness":
                correctness = observation.correctness
                if (
                    correctness.failure_kind,
                    correctness.message,
                    correctness.output,
                    correctness.input_storage_unchanged,
                    correctness.output_disjoint,
                    correctness.finite,
                    correctness.close,
                    correctness.max_abs_error,
                ) != ("capability", capability_message, None, False, False, False, False, None):
                    raise SchemaError(
                        "capability-rejected blocked correctness evidence/linkage differs"
                    )
            else:
                timing = observation.timing
                if (
                    timing.failure_kind,
                    timing.message,
                    timing.warmups,
                    timing.repetitions,
                    timing.samples_ms,
                    timing.median_ms,
                ) != ("capability", capability_message, 0, 0, (), None):
                    raise SchemaError(
                        "capability-rejected blocked timing evidence/linkage differs"
                    )
        if any(
            item["failure_kind"] != "capability" or item["error"] != capability_message
            for item in stage.values()
        ):
            raise SchemaError("capability-rejected stage failure linkage differs")
        if not all(
            item["status"] == "blocked"
            for item in (
                *stage.values(),
                *compile_evidence.values(),
                *resource.values(),
                *profile.values(),
                *validation.values(),
            )
        ):
            raise SchemaError("capability-rejected evidence must be blocked")
    if environment["backend_invoked"] is not any(
        item["backend_invoked"] is True for item in compile_evidence.values()
    ):
        raise SchemaError("environment backend flag does not match compile evidence")
    all_passed = all(item.status == "passed" for item in observations)
    if outcome == "completed" and (
        not capability.available
        or not all_passed
        or any(item["status"] != "completed" for item in stage.values())
    ):
        raise SchemaError("completed outcome lacks all passing terminal evidence")
    if outcome == "aborted" and capability.available:
        raise SchemaError("aborted outcome requires capability rejection")
    if outcome == "failed" and (not capability.available or all_passed):
        raise SchemaError("failed outcome does not match capable non-passing execution")


def _attempt(
    attempts: list[Mapping[str, object]],
    cell: str,
    stage: str,
    target: str,
    reason: str | None = None,
) -> None:
    seen = any(item["cell_id"] == cell for item in attempts)
    attempts.append(
        {
            "attempt_id": len(attempts) + 1,
            "cell_id": cell,
            "stage": stage,
            "status": "running"
            if target == "running"
            else ("success" if target == "passed" else "failure"),
            "from_state": "running" if seen else "pending",
            "to_state": target,
            "reason": reason,
        }
    )


def _blocked_compile(arm: str, error: str) -> dict[str, object]:
    native = arm in _NATIVE
    return {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "backend_kind": _BACKEND[arm],
        "status": "blocked",
        "error": error,
        "compile_ns": None,
        "backend_invoked": False,
        "fullgraph": not native,
        "dynamic": False,
        "mode": "native_triton" if native else "default",
        "callable_distinct": False,
        "eager_fallback": False,
        "kernel_name": None,
        "kernel_hash": None,
        "config": dict(_CONFIG[arm]) if native else None,
    }


def _blocked_resource(arm: str, error: str) -> dict[str, object]:
    return {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": "blocked",
        "error": error,
        "kernel_name": None,
        "kernel_hash": None,
        "target": None,
        "metadata": None,
        "n_regs": None,
        "n_spills": None,
        "n_max_threads": None,
        "asm_stages": [],
        "resource_gate_passed": False,
    }


def _blocked_profile(
    arm: str, error: str, name: str | None = None, kernel_hash: str | None = None
) -> dict[str, object]:
    return {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": "blocked",
        "error": error,
        "method": "torch.profiler.cuda_events",
        "warmed": False,
        "expected_kernel_name": name,
        "expected_kernel_hash": kernel_hash,
        "config": dict(_CONFIG[arm]),
        "invocation_count": 0,
        "cuda_event_count": 0,
        "cuda_event_names_sample": [],
        "cuda_event_names_sha256": None,
        "exact_name_match_count": 0,
        "output_revalidated": False,
        "inputs_revalidated": False,
        "one_kernel_gate_passed": False,
    }


def _blocked_validation(arm: str, error: str, *, status: str = "blocked") -> dict[str, object]:
    return {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": status,
        "error": error,
        "probes": [
            {
                "id": probe_id,
                "passed": False,
                "deterministic": False,
                "inputs_unchanged": False,
                "output_disjoint": False,
                "value_class_match": False,
                "sign_match": False,
                "finite_close": False,
                "max_abs_error": None,
            }
            for probe_id in _VALIDATION_PROBES
        ],
        "validation_gate_passed": False,
    }


def _environment(
    capability: Any, triton_version: str | None, backend_invoked: bool | None
) -> dict[str, object]:
    return {
        "schema": _ENV_SCHEMA,
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "torch_version": capability.torch_version,
        "triton_version": triton_version,
        "cuda_version": capability.cuda_version,
        "rocm_version": capability.rocm_version,
        "device_index": capability.device_index,
        "device_name": capability.device_name,
        "compute_capability": None
        if capability.compute_capability is None
        else list(capability.compute_capability),
        "precision_policy": dict(_POLICIES),
        "autocast_policy": dict(_AUTOCAST),
        "backend_invoked": backend_invoked,
        "fusion_claim": False,
    }


def _probe_capability() -> tuple[Any, Any | None, str | None]:
    from .local_executor import CapabilityProbe, _probe_torch

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return (
            CapabilityProbe(
                False,
                ("torch_missing",),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                _safe_error(exc),
            ),
            None,
            None,
        )
    probed = _probe_torch(torch, cast(Any, None))
    # torch.__version__ is TorchVersion (a str subclass), while the strict wire
    # schema intentionally accepts only an exact built-in string.
    capability = CapabilityProbe(
        probed.available,
        probed.reasons,
        None if probed.torch_version is None else str(probed.torch_version),
        None if probed.cuda_version is None else str(probed.cuda_version),
        None if probed.rocm_version is None else str(probed.rocm_version),
        probed.device_index,
        probed.device_name,
        probed.compute_capability,
        probed.native_bf16,
        probed.inductor_available,
        probed.allocation_succeeded,
        probed.detail,
    )
    if not capability.available:
        return capability, torch, None
    try:
        triton_version = str(importlib.import_module("triton").__version__)
    except Exception as exc:
        rejected = CapabilityProbe(
            False,
            ("device_probe_failed",),
            capability.torch_version,
            capability.cuda_version,
            None,
            capability.device_index,
            capability.device_name,
            capability.compute_capability,
            None,
            None,
            False,
            _safe_error(exc),
        )
        return rejected, torch, None
    if (
        capability.compute_capability != (9, 0)
        or "H100" not in cast(str, capability.device_name)
        or triton_version != "3.4.0"
    ):
        detail = f"requires exact H100 sm90 and Triton 3.4.0; got {capability.device_name!r}, {capability.compute_capability!r}, {triton_version!r}"
        rejected = CapabilityProbe(
            False,
            ("device_probe_failed",),
            capability.torch_version,
            capability.cuda_version,
            None,
            capability.device_index,
            capability.device_name,
            capability.compute_capability,
            None,
            None,
            False,
            detail,
        )
        return rejected, torch, triton_version
    return capability, torch, triton_version


def _blocked_observation(arm: str, stage: str, message: str, kind: str = "capability") -> Any:
    from .local_executor import CellObservation, CorrectnessObservation, TimingObservation

    key = _correctness_key(f"{arm}-correctness")
    if stage == "correctness":
        correctness = CorrectnessObservation(
            "blocked", key, kind, message, None, False, False, False, False, None
        )
        return CellObservation(
            f"{arm}-correctness", _CASE, arm, "correctness", "blocked", correctness, None
        )
    timing = TimingObservation("blocked", key, kind, message, 0, 0, (), None)
    return CellObservation(f"{arm}-timing", _CASE, arm, "timing", "blocked", None, timing)


def _failed_correctness(arm: str, kind: str, message: str) -> Any:
    from .local_executor import CellObservation, CorrectnessObservation

    nested = CorrectnessObservation(
        "failed",
        _correctness_key(f"{arm}-correctness"),
        kind,
        message,
        None,
        False,
        False,
        False,
        False,
        None,
    )
    return CellObservation(f"{arm}-correctness", _CASE, arm, "correctness", "failed", nested, None)


def _blocked_timing(arm: str, message: str) -> Any:
    from .local_executor import CellObservation, TimingObservation

    nested = TimingObservation(
        "blocked",
        _correctness_key(f"{arm}-correctness"),
        "correctness_gate",
        message,
        0,
        0,
        (),
        None,
    )
    return CellObservation(f"{arm}-timing", _CASE, arm, "timing", "blocked", None, nested)


def _failed_timing(arm: str, kind: str, message: str) -> Any:
    from .local_executor import CellObservation, TimingObservation

    nested = TimingObservation(
        "failed",
        _correctness_key(f"{arm}-correctness"),
        kind,
        message,
        0,
        0,
        (),
        None,
    )
    return CellObservation(f"{arm}-timing", _CASE, arm, "timing", "failed", None, nested)


def _summary(
    observations: Sequence[Any],
    stage: Mapping[str, Mapping[str, object]],
    compile_evidence: Mapping[str, Mapping[str, object]],
    resource: Mapping[str, Mapping[str, object]],
    profile: Mapping[str, Mapping[str, object]],
    validation: Mapping[str, Mapping[str, object]],
    outcome: str,
) -> dict[str, object]:
    statuses = [item.status for item in observations]
    return {
        "expected_cell_ids": list(_CELL_IDS),
        "terminal_cell_ids": [item.cell_id for item in observations],
        "passed": statuses.count("passed"),
        "failed": statuses.count("failed"),
        "blocked": statuses.count("blocked"),
        "all_cells_terminal": len(observations) == len(_CELL_IDS),
        "counts": _counts(stage, compile_evidence, resource, profile, validation),
        "outcome": outcome,
        "fusion_claim": False,
    }


def _aborted_result(
    verified: Any,
    capability: Any,
    triton_version: str | None,
    executor_sources: Mapping[str, object],
) -> NativeFusionExecutionResult:
    message = str(capability.detail or capability.reason or "native capability unavailable")
    observations = tuple(
        _blocked_observation(arm, stage, message)
        for arm in _RUNTIME_ARMS
        for stage in ("correctness", "timing")
    )
    attempts: list[Mapping[str, object]] = []
    for observation in observations:
        _attempt(attempts, observation.cell_id, observation.stage, "running")
        nested = (
            observation.correctness if observation.stage == "correctness" else observation.timing
        )
        _attempt(attempts, observation.cell_id, observation.stage, "failed", nested.failure_kind)
    stage_outcomes = {
        f"{arm}-correctness": {
            "case_id": _CASE,
            "arm_id": arm,
            "entrypoint": _ENTRYPOINT[arm],
            "backend_kind": _BACKEND[arm],
            "status": "blocked",
            "failure_kind": "capability",
            "error": message,
            "correctness_passed": False,
            "resource_passed": False if arm in _NATIVE else None,
            "profile_passed": False if arm in _NATIVE else None,
            "validation_passed": False if arm in _NATIVE else None,
            "timing_allowed": False,
        }
        for arm in _RUNTIME_ARMS
    }
    compile_evidence = {
        f"{arm}-correctness": _blocked_compile(arm, message) for arm in (*_NATIVE, _COMPARATOR)
    }
    resource = {f"{arm}-correctness": _blocked_resource(arm, message) for arm in _NATIVE}
    profile = {f"{arm}-correctness": _blocked_profile(arm, message) for arm in _NATIVE}
    validation = {f"{arm}-correctness": _blocked_validation(arm, message) for arm in _NATIVE}
    summary = _summary(
        observations, stage_outcomes, compile_evidence, resource, profile, validation, "aborted"
    )
    return NativeFusionExecutionResult(
        _SCHEMA,
        str(verified.path),
        verified.sha256,
        verified.bytes,
        verified.suite.suite_id,
        capability,
        (),
        observations,
        tuple(attempts),
        _environment(capability, triton_version, False),
        stage_outcomes,
        compile_evidence,
        resource,
        profile,
        validation,
        executor_sources,
        summary,
        "aborted",
    )


def _failed_result(
    verified: Any,
    capability: Any,
    triton_version: str | None,
    message: str,
    materialization: Sequence[Any],
    executor_sources: Mapping[str, object],
    compile_evidence: Mapping[str, Mapping[str, object]] | None = None,
    resource_evidence: Mapping[str, Mapping[str, object]] | None = None,
) -> NativeFusionExecutionResult:
    compile_evidence = {} if compile_evidence is None else compile_evidence
    resource_evidence = {} if resource_evidence is None else resource_evidence
    observations = tuple(
        _blocked_observation(arm, cell_stage, message, "executor")
        for arm in _RUNTIME_ARMS
        for cell_stage in ("correctness", "timing")
    )
    attempts: list[Mapping[str, object]] = []
    for observation in observations:
        _attempt(attempts, observation.cell_id, observation.stage, "running")
        nested = (
            observation.correctness if observation.stage == "correctness" else observation.timing
        )
        _attempt(attempts, observation.cell_id, observation.stage, "failed", nested.failure_kind)
    stage = {
        f"{arm}-correctness": {
            "case_id": _CASE,
            "arm_id": arm,
            "entrypoint": _ENTRYPOINT[arm],
            "backend_kind": _BACKEND[arm],
            "status": "blocked",
            "failure_kind": "executor",
            "error": message,
            "correctness_passed": False,
            "resource_passed": False if arm in _NATIVE else None,
            "profile_passed": False if arm in _NATIVE else None,
            "validation_passed": False if arm in _NATIVE else None,
            "timing_allowed": False,
        }
        for arm in _RUNTIME_ARMS
    }
    compile_records = {
        cell: dict(
            compile_evidence.get(
                cell,
                _blocked_compile(cell.removesuffix("-correctness"), message),
            )
        )
        for cell in _COMPILE_IDS
    }
    resource_records = {
        cell: dict(
            resource_evidence.get(
                cell,
                _blocked_resource(cell.removesuffix("-correctness"), message),
            )
        )
        for cell in _NATIVE_IDS
    }
    profiles = {}
    for arm in _NATIVE:
        cell = f"{arm}-correctness"
        resource_record = resource_records[cell]
        profiles[cell] = _blocked_profile(
            arm,
            message,
            cast(str | None, resource_record["kernel_name"]),
            cast(str | None, resource_record["kernel_hash"]),
        )
        stage[cell] = {
            **stage[cell],
            "resource_passed": resource_record["resource_gate_passed"] is True,
        }
    validations = {f"{arm}-correctness": _blocked_validation(arm, message) for arm in _NATIVE}
    for arm in _RUNTIME_ARMS:
        cell = f"{arm}-correctness"
        if arm != _EAGER and compile_records[cell]["status"] != "compiled":
            failure_kind = "compile_failed"
            failure_error = cast(str, compile_records[cell]["error"])
        elif arm in _NATIVE and resource_records[cell]["resource_gate_passed"] is not True:
            failure_kind = "resource_gate"
            failure_error = cast(str, resource_records[cell]["error"])
        else:
            failure_kind = "executor"
            failure_error = message
        stage[cell] = {
            **stage[cell],
            "failure_kind": failure_kind,
            "error": failure_error,
        }
    summary = _summary(
        observations,
        stage,
        compile_records,
        resource_records,
        profiles,
        validations,
        "failed",
    )
    backend_invoked = any(item["backend_invoked"] is True for item in compile_records.values())
    return NativeFusionExecutionResult(
        _SCHEMA,
        str(verified.path),
        verified.sha256,
        verified.bytes,
        verified.suite.suite_id,
        capability,
        tuple(materialization),
        observations,
        tuple(attempts),
        _environment(capability, triton_version, backend_invoked),
        stage,
        compile_records,
        resource_records,
        profiles,
        validations,
        executor_sources,
        summary,
        "failed",
    )


def _run_correctness(
    torch: Any,
    arm: str,
    kernel: Callable[..., Any],
    arguments: Sequence[Any],
    expected: Any,
    inputs: Mapping[str, Any],
    device_index: int,
) -> tuple[Any, Any | None]:
    from .local_executor import (
        CellObservation,
        CorrectnessObservation,
        _storage_pointer,
        _tensor_hash,
        _validate_correctness,
    )

    try:
        before = {name: _tensor_hash(torch, tensor) for name, tensor in inputs.items()}
        actual = kernel(*arguments)
        torch.cuda.synchronize(device_index)
        repeated = kernel(*arguments)
        torch.cuda.synchronize(device_index)
        evidence = _validate_correctness(
            torch,
            actual=actual,
            expected=expected,
            inputs=inputs,
            before_hashes=before,
            expected_shape=(128, 4096),
            atol=1e-5,
            rtol=0.0078125,
            correctness_key=_correctness_key(f"{arm}-correctness"),
        )
        if evidence.status == "passed":
            disjoint = all(
                _storage_pointer(repeated) != _storage_pointer(tensor) for tensor in inputs.values()
            )
            unchanged = all(
                _tensor_hash(torch, tensor) == before[name] for name, tensor in inputs.items()
            )
            classes = all(
                bool(torch.equal(fn(actual), fn(expected)))
                for fn in (torch.isnan, torch.isposinf, torch.isneginf)
            )
            signs = bool(torch.equal(torch.signbit(actual), torch.signbit(expected)))
            deterministic = bool(torch.equal(actual, repeated))
            failure: tuple[str, str] | None = None
            if not disjoint or not unchanged:
                failure = ("mutation", "determinism invocation aliased or mutated an input")
            elif not classes:
                failure = ("value_class", "finite/NaN/infinity classes differ from reference")
            elif not signs:
                failure = ("sign", "output sign bits differ from reference")
            elif not deterministic:
                failure = ("determinism", "repeated output is not bitwise deterministic")
            if failure is not None:
                evidence = CorrectnessObservation(
                    "failed",
                    evidence.correctness_key,
                    failure[0],
                    failure[1],
                    evidence.output,
                    unchanged,
                    disjoint,
                    evidence.finite,
                    evidence.close,
                    evidence.max_abs_error,
                )
        return CellObservation(
            f"{arm}-correctness", _CASE, arm, "correctness", evidence.status, evidence, None
        ), actual
    except Exception as exc:
        return _failed_correctness(arm, "runtime", _safe_error(exc)), None


def _run_validation_battery(
    torch: Any,
    arm: str,
    kernel: Callable[..., Any],
    reference: Callable[..., Any],
    device_index: int,
) -> Mapping[str, object]:
    from .local_executor import _storage_pointer, _tensor_hash

    device = f"cuda:{device_index}"
    probes: list[dict[str, object]] = []
    error: str | None = None
    for probe_id in _VALIDATION_PROBES:
        try:
            if probe_id == "zeros":
                x = torch.zeros((128, 4096), device=device, dtype=torch.bfloat16)
                residual = torch.zeros_like(x)
            elif probe_id == "cancellation":
                row = (
                    torch.arange(4096, device=device, dtype=torch.float32)
                    .remainder(17)
                    .sub(8)
                    .to(dtype=torch.bfloat16)
                )
                x = row.expand(128, 4096).contiguous()
                residual = (-x).contiguous()
            else:
                x = torch.zeros((128, 4096), device=device, dtype=torch.bfloat16)
                residual = torch.zeros_like(x)
                maximum = torch.finfo(torch.bfloat16).max
                x[:, 0] = maximum
                residual[:, 0] = maximum
                x[:, 1] = -maximum
                residual[:, 1] = -maximum
            gamma = torch.ones((4096,), device=device, dtype=torch.bfloat16)
            inputs = {"input": x, "residual": residual, "gamma": gamma}
            before = {name: _tensor_hash(torch, tensor) for name, tensor in inputs.items()}
            expected = reference(x, residual, gamma)
            actual = kernel(x, residual, gamma)
            torch.cuda.synchronize(device_index)
            repeated = kernel(x, residual, gamma)
            torch.cuda.synchronize(device_index)
            deterministic = _tensor_hash(torch, actual) == _tensor_hash(torch, repeated)
            inputs_unchanged = all(
                _tensor_hash(torch, tensor) == before[name] for name, tensor in inputs.items()
            )
            output_disjoint = all(
                _storage_pointer(output) != _storage_pointer(tensor)
                for output in (actual, repeated)
                for tensor in inputs.values()
            )
            descriptor_ok = (
                tuple(actual.shape) == (128, 4096)
                and str(actual.device) == device
                and actual.dtype == torch.bfloat16
                and actual.layout == torch.strided
                and bool(actual.is_contiguous())
            )
            value_class_match = all(
                bool(torch.equal(classifier(actual), classifier(expected)))
                for classifier in (torch.isnan, torch.isposinf, torch.isneginf)
            )
            finite_mask = torch.isfinite(expected)
            finite_class_match = bool(torch.equal(torch.isfinite(actual), finite_mask))
            value_class_match = value_class_match and finite_class_match
            finite_actual = actual[finite_mask]
            finite_expected = expected[finite_mask]
            sign_match = bool(
                torch.equal(torch.signbit(finite_actual), torch.signbit(finite_expected))
            )
            if int(finite_expected.numel()) == 0:
                finite_close = True
                max_abs_error: float | None = None
            else:
                max_abs_error = float(
                    torch.max(torch.abs(finite_actual.float() - finite_expected.float())).item()
                )
                try:
                    torch.testing.assert_close(
                        finite_actual,
                        finite_expected,
                        atol=1e-5,
                        rtol=0.0078125,
                        equal_nan=False,
                        check_device=True,
                        check_dtype=True,
                        check_layout=True,
                        check_stride=True,
                    )
                    finite_close = True
                except AssertionError:
                    finite_close = False
            passed = all(
                (
                    descriptor_ok,
                    deterministic,
                    inputs_unchanged,
                    output_disjoint,
                    value_class_match,
                    sign_match,
                    finite_close,
                )
            )
            probes.append(
                {
                    "id": probe_id,
                    "passed": passed,
                    "deterministic": deterministic,
                    "inputs_unchanged": inputs_unchanged,
                    "output_disjoint": output_disjoint,
                    "value_class_match": value_class_match,
                    "sign_match": sign_match,
                    "finite_close": finite_close and descriptor_ok,
                    "max_abs_error": max_abs_error,
                }
            )
        except Exception as exc:
            error = _safe_error(exc)
            probes.append(
                {
                    "id": probe_id,
                    "passed": False,
                    "deterministic": False,
                    "inputs_unchanged": False,
                    "output_disjoint": False,
                    "value_class_match": False,
                    "sign_match": False,
                    "finite_close": False,
                    "max_abs_error": None,
                }
            )
    gate = all(probe["passed"] is True for probe in probes)
    if not gate and error is None:
        error = "one or more structured validation probes failed"
    return {
        "case_id": _CASE,
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": "validated" if gate else "failed",
        "error": None if gate else error,
        "probes": probes,
        "validation_gate_passed": gate,
    }


def _profile_once(
    torch: Any,
    arm: str,
    kernel: Callable[..., Any],
    arguments: Sequence[Any],
    expected: Any,
    inputs: Mapping[str, Any],
    device_index: int,
    name: str,
    kernel_hash: str,
) -> Mapping[str, object]:
    from .local_executor import _tensor_hash, _validate_correctness

    warmed = False
    try:
        before = {key: _tensor_hash(torch, tensor) for key, tensor in inputs.items()}
        kernel(*arguments)
        torch.cuda.synchronize(device_index)
        warmed = True
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
            actual = kernel(*arguments)
            torch.cuda.synchronize(device_index)
        names = [
            str(event.name)
            for event in profiler.events()
            if getattr(event, "device_type", None) == torch.autograd.DeviceType.CUDA
        ]
        correctness = _validate_correctness(
            torch,
            actual=actual,
            expected=expected,
            inputs=inputs,
            before_hashes=before,
            expected_shape=(128, 4096),
            atol=1e-5,
            rtol=0.0078125,
            correctness_key=_correctness_key(f"{arm}-correctness"),
        )
        inputs_ok = all(
            _tensor_hash(torch, tensor) == before[key] for key, tensor in inputs.items()
        )
        output_ok = correctness.status == "passed"
        matches = names.count(name)
        gate = output_ok and inputs_ok and names == [name]
        return {
            "case_id": _CASE,
            "arm_id": arm,
            "entrypoint": _ENTRYPOINT[arm],
            "status": "profiled" if gate else "failed",
            "error": None
            if gate
            else "profile lacks exactly one expected CUDA kernel event or revalidation failed",
            "method": "torch.profiler.cuda_events",
            "warmed": warmed,
            "expected_kernel_name": name,
            "expected_kernel_hash": kernel_hash,
            "config": dict(_CONFIG[arm]),
            "invocation_count": 1,
            "cuda_event_count": len(names),
            "cuda_event_names_sample": names[:32],
            "cuda_event_names_sha256": _names_digest(names),
            "exact_name_match_count": matches,
            "output_revalidated": output_ok,
            "inputs_revalidated": inputs_ok,
            "one_kernel_gate_passed": gate,
        }
    except Exception as exc:
        result = _blocked_profile(arm, _safe_error(exc), name, kernel_hash)
        result.update(status="failed", warmed=warmed)
        return result


def _timing(
    torch: Any,
    arm: str,
    kernel: Callable[..., Any],
    arguments: Sequence[Any],
    inputs: Mapping[str, Any],
    device_index: int,
) -> Any:
    from .local_executor import (
        CellObservation,
        _tensor_hash,
        _timing_observation,
        _with_timing_failure,
    )

    try:
        before = {key: _tensor_hash(torch, tensor) for key, tensor in inputs.items()}
        evidence = _timing_observation(
            torch,
            kernel=kernel,
            arguments=arguments,
            device_index=device_index,
            correctness_key=_correctness_key(f"{arm}-correctness"),
            warmups=10,
            repetitions=50,
        )
        if any(_tensor_hash(torch, tensor) != before[key] for key, tensor in inputs.items()):
            evidence = _with_timing_failure(
                evidence, failure_kind="mutation", message="timing mutated an input"
            )
    except Exception as exc:
        from .local_executor import TimingObservation

        evidence = TimingObservation(
            "failed",
            _correctness_key(f"{arm}-correctness"),
            "execution",
            _safe_error(exc),
            0,
            0,
            (),
            None,
        )
    return CellObservation(f"{arm}-timing", _CASE, arm, "timing", evidence.status, None, evidence)


def run_native_fusion_suite(suite_path: str | Path) -> NativeFusionExecutionResult:
    """Run one authenticated native suite; every resource/profile/timing gate is fail-closed."""
    verified = verify_suite(suite_path)
    _validate_frozen_suite(verified.suite, verified.sha256)
    executor_sources = _bound_executor_sources()
    capability, torch, triton_version = _probe_capability()
    if not capability.available:
        return _aborted_result(verified, capability, triton_version, executor_sources)
    assert torch is not None

    # Optional and v1-private imports occur only after frozen-suite authentication.
    from .fusion_kernels import (
        RESIDUAL_RMSNORM_CONFIGS,
        compile_residual_rmsnorm,
        compiled_kernel_evidence,
        load_residual_rmsnorm,
    )
    from .local_executor import (
        _cuda_autocast_disabled,
        _materialize_arm,
        _precision_flags,
        _residual_rmsnorm,
    )

    suite = verified.suite
    case = suite.cases[0]
    device_index = cast(int, capability.device_index)
    config_by_arm = {config.config_id: config for config in RESIDUAL_RMSNORM_CONFIGS}
    attempts: list[Mapping[str, object]] = []
    materialization: list[Any] = []
    inputs: dict[str, Mapping[str, Any]] = {}
    kernels: dict[str, Callable[..., Any]] = {}
    compile_evidence: dict[str, Mapping[str, object]] = {}
    resource: dict[str, Mapping[str, object]] = {}
    profile: dict[str, Mapping[str, object]] = {}
    validation: dict[str, Mapping[str, object]] = {}
    observations: dict[str, Any] = {}
    stage: dict[str, Mapping[str, object]] = {}

    with _precision_flags(torch), _cuda_autocast_disabled(torch):
        for arm in _RUNTIME_ARMS:
            try:
                materialized_tensors, record = _materialize_arm(
                    torch, suite, case, arm, verified.sha256, device_index
                )
            except Exception as exc:
                return _failed_result(
                    verified,
                    capability,
                    triton_version,
                    _safe_error(exc),
                    materialization,
                    executor_sources,
                )
            inputs[arm] = materialized_tensors
            materialization.append(record)

        # All four native kernels are compiled and resource-gated before correctness begins.
        for arm in _NATIVE:
            cell = f"{arm}-correctness"
            native_tensors = inputs[arm]
            started = time.perf_counter_ns()
            try:
                compiled = compile_residual_rmsnorm(
                    _ENTRYPOINT[arm],
                    native_tensors["input"],
                    native_tensors["residual"],
                    native_tensors["gamma"],
                )
                raw = compiled_kernel_evidence(compiled, config_by_arm[arm])
                target = cast(Mapping[str, object], raw["target"])
                metadata = cast(Mapping[str, object], raw["metadata"])
                normalized: dict[str, object] = {
                    "case_id": _CASE,
                    "arm_id": arm,
                    "entrypoint": _ENTRYPOINT[arm],
                    "status": raw["status"],
                    "error": raw["error"],
                    "kernel_name": raw["kernel_name"],
                    "kernel_hash": raw["kernel_hash"],
                    "target": {
                        "backend": str(target["backend"]),
                        "arch": str(target["arch"]),
                        "warp_size": target["warp_size"],
                    },
                    "metadata": {
                        key: metadata[key]
                        for key in ("shared", "num_warps", "num_ctas", "num_stages")
                    },
                    "n_regs": raw["n_regs"],
                    "n_spills": raw["n_spills"],
                    "n_max_threads": raw["n_max_threads"],
                    "asm_stages": [
                        {key: item[key] for key in ("stage", "bytes", "sha256")}
                        for item in cast(Sequence[Mapping[str, object]], raw["asm_stages"])
                    ],
                    "resource_gate_passed": raw["resource_gate_passed"],
                }
                resource_passed = normalized["resource_gate_passed"] is True
                normalized["status"] = "compiled" if resource_passed else "failed"
                normalized["error"] = (
                    None if resource_passed else "native resource gate requires n_spills=0"
                )
                resource[cell] = normalized
                compile_evidence[cell] = {
                    "case_id": _CASE,
                    "arm_id": arm,
                    "entrypoint": _ENTRYPOINT[arm],
                    "backend_kind": "native_triton",
                    "status": "compiled",
                    "error": None,
                    "compile_ns": time.perf_counter_ns() - started,
                    "backend_invoked": True,
                    "fullgraph": False,
                    "dynamic": False,
                    "mode": "native_triton",
                    "callable_distinct": True,
                    "eager_fallback": False,
                    "kernel_name": raw["kernel_name"],
                    "kernel_hash": raw["kernel_hash"],
                    "config": dict(_CONFIG[arm]),
                }
                kernels[arm] = cast(Callable[..., Any], load_residual_rmsnorm(_ENTRYPOINT[arm]))
            except Exception as exc:
                compile_error = _safe_error(exc)
                resource[cell] = {**_blocked_resource(arm, compile_error), "status": "failed"}
                compile_evidence[cell] = {
                    **_blocked_compile(arm, compile_error),
                    "status": "failed",
                    "compile_ns": time.perf_counter_ns() - started,
                }

        comparator_state: dict[str, object] = {
            "invoked": False,
            "completed": False,
            "error": None,
            "callable_distinct": False,
        }
        comparator_started = 0
        comparator_error: str | None = None

        def formula(x: Any, residual: Any, gamma: Any) -> Any:
            return _residual_rmsnorm(torch, x, residual, gamma, 1e-5)

        kernels[_EAGER] = formula

        reference_inputs = inputs[_EAGER]
        try:
            expected = formula(
                reference_inputs["input"], reference_inputs["residual"], reference_inputs["gamma"]
            )
            torch.cuda.synchronize(device_index)
        except Exception as exc:
            return _failed_result(
                verified,
                capability,
                triton_version,
                _safe_error(exc),
                materialization,
                executor_sources,
                compile_evidence,
                resource,
            )

        # Candidate: correctness/determinism -> warmed one-invocation profile -> timing.
        # Comparator: compile/no-fallback -> correctness -> timing. Eager: correctness -> timing.
        for arm in _EXECUTION_ARMS:
            cell = f"{arm}-correctness"
            timing_cell = f"{arm}-timing"
            _attempt(attempts, cell, "correctness", "running")
            arm_tensors = inputs[arm]
            arguments = (arm_tensors["input"], arm_tensors["residual"], arm_tensors["gamma"])
            resource_pass: bool | None = None
            profile_pass: bool | None = None
            validation_pass: bool | None = None
            if arm == _COMPARATOR:
                comparator_started = time.perf_counter_ns()
                try:
                    kernels[_COMPARATOR] = _compile_comparator(torch, formula, comparator_state)
                except Exception as exc:
                    comparator_error = cast(
                        str,
                        comparator_state["error"] or _safe_error(exc),
                    )

            if arm in _NATIVE and compile_evidence[cell]["status"] != "compiled":
                compile_error = cast(str, compile_evidence[cell]["error"])
                correctness = _failed_correctness(arm, "compile_failed", compile_error)
                resource_pass = False
                validation[cell] = _blocked_validation(
                    arm, "validation blocked by compile failure"
                )
                validation_pass = False
                profile[cell] = _blocked_profile(
                    arm,
                    "profile blocked by compile failure",
                    cast(str | None, resource[cell]["kernel_name"]),
                    cast(str | None, resource[cell]["kernel_hash"]),
                )
                profile_pass = False
            elif arm in _NATIVE and resource[cell]["resource_gate_passed"] is not True:
                resource_error = cast(str, resource[cell]["error"])
                correctness = _failed_correctness(arm, "resource_gate", resource_error)
                resource_pass = False
                validation[cell] = _blocked_validation(arm, "validation blocked by resource gate")
                validation_pass = False
                profile[cell] = _blocked_profile(
                    arm,
                    "profile blocked by resource gate",
                    cast(str | None, resource[cell]["kernel_name"]),
                    cast(str | None, resource[cell]["kernel_hash"]),
                )
                profile_pass = False
            elif arm == _COMPARATOR and comparator_error is not None:
                correctness = _failed_correctness(arm, "compile_failed", comparator_error)
                compile_evidence[cell] = {
                    **_blocked_compile(arm, comparator_error),
                    "status": "failed",
                    "compile_ns": time.perf_counter_ns() - comparator_started,
                    "backend_invoked": bool(comparator_state["invoked"]),
                    "callable_distinct": bool(comparator_state["callable_distinct"]),
                }
            else:
                try:
                    correctness, _ = _run_correctness(
                        torch, arm, kernels[arm], arguments, expected, arm_tensors, device_index
                    )
                except Exception as exc:
                    correctness = _failed_correctness(arm, "execution", _safe_error(exc))
                if arm == _COMPARATOR:
                    invoked = bool(comparator_state["invoked"])
                    completed = bool(comparator_state["completed"])
                    comparator_compile_error = cast(str | None, comparator_state["error"])
                    if not completed:
                        comparator_compile_error = (
                            comparator_compile_error
                            or "recording Inductor backend did not complete successfully"
                        )
                        correctness = _failed_correctness(
                            arm, "compile_failed", comparator_compile_error
                        )
                    compile_evidence[cell] = {
                        "case_id": _CASE,
                        "arm_id": arm,
                        "entrypoint": _ENTRYPOINT[arm],
                        "backend_kind": "inductor",
                        "status": "compiled" if completed else "failed",
                        "error": None if completed else comparator_compile_error,
                        "compile_ns": time.perf_counter_ns() - comparator_started,
                        "backend_invoked": invoked,
                        "fullgraph": True,
                        "dynamic": False,
                        "mode": "default",
                        "callable_distinct": bool(comparator_state["callable_distinct"]),
                        "eager_fallback": False,
                        "kernel_name": None,
                        "kernel_hash": None,
                        "config": None,
                    }
                if arm in _NATIVE:
                    resource_pass = True
                    if correctness.status == "passed":
                        try:
                            validation[cell] = _run_validation_battery(
                                torch, arm, kernels[arm], formula, device_index
                            )
                        except Exception as exc:
                            validation[cell] = _blocked_validation(
                                arm, _safe_error(exc), status="failed"
                            )
                    else:
                        validation[cell] = _blocked_validation(
                            arm, "validation blocked by correctness"
                        )
                    validation_pass = validation[cell]["validation_gate_passed"] is True
                    if correctness.status == "passed" and validation_pass:
                        try:
                            profile[cell] = _profile_once(
                                torch,
                                arm,
                                kernels[arm],
                                arguments,
                                expected,
                                arm_tensors,
                                device_index,
                                cast(str, resource[cell]["kernel_name"]),
                                cast(str, resource[cell]["kernel_hash"]),
                            )
                        except Exception as exc:
                            profile_failure = _blocked_profile(
                                arm,
                                _safe_error(exc),
                                cast(str, resource[cell]["kernel_name"]),
                                cast(str, resource[cell]["kernel_hash"]),
                            )
                            profile_failure["status"] = "failed"
                            profile[cell] = profile_failure
                    else:
                        profile[cell] = _blocked_profile(
                            arm,
                            "profile blocked by correctness/validation",
                            cast(Any, resource[cell]["kernel_name"]),
                            cast(Any, resource[cell]["kernel_hash"]),
                        )
                    profile_pass = profile[cell]["one_kernel_gate_passed"] is True

            failure, error = _derived_stage_failure(
                cell,
                arm,
                correctness,
                compile_evidence,
                resource,
                validation,
                profile,
            )

            correctness_passed = correctness.status == "passed"
            timing_allowed = correctness_passed and (
                arm not in _NATIVE
                or (resource_pass is True and validation_pass is True and profile_pass is True)
            )
            stage[cell] = {
                "case_id": _CASE,
                "arm_id": arm,
                "entrypoint": _ENTRYPOINT[arm],
                "backend_kind": _BACKEND[arm],
                "status": "completed" if timing_allowed else "failed",
                "failure_kind": failure,
                "error": error,
                "correctness_passed": correctness_passed,
                "resource_passed": resource_pass,
                "profile_passed": profile_pass,
                "validation_passed": validation_pass,
                "timing_allowed": timing_allowed,
            }
            observations[cell] = correctness
            correctness_reason = (
                None if correctness_passed else cast(Any, correctness.correctness).failure_kind
            )
            _attempt(
                attempts,
                cell,
                "correctness",
                "passed" if correctness_passed else "failed",
                correctness_reason,
            )

            _attempt(attempts, timing_cell, "timing", "running")
            if timing_allowed:
                try:
                    timing = _timing(torch, arm, kernels[arm], arguments, arm_tensors, device_index)
                except Exception as exc:
                    timing = _failed_timing(arm, "execution", _safe_error(exc))
            else:
                timing = _blocked_timing(
                    arm,
                    "timing blocked by correctness/resource/validation/profile gates",
                )
            observations[timing_cell] = timing
            timing_failure = (
                None if timing.status == "passed" else cast(Any, timing.timing).failure_kind
            )
            _attempt(
                attempts,
                timing_cell,
                "timing",
                "passed" if timing.status == "passed" else "failed",
                timing_failure,
            )

    ordered = tuple(observations[cell] for cell in _CELL_IDS)
    serialized_attempts: list[Mapping[str, object]] = []
    for cell_id in _CELL_IDS:
        for item in attempts:
            if item["cell_id"] == cell_id:
                serialized_attempts.append({**item, "attempt_id": len(serialized_attempts) + 1})
    outcome: Literal["completed", "failed", "aborted"] = (
        "completed" if all(item.status == "passed" for item in ordered) else "failed"
    )
    environment = _environment(
        capability,
        triton_version,
        any(item["backend_invoked"] for item in compile_evidence.values()),
    )
    summary = _summary(ordered, stage, compile_evidence, resource, profile, validation, outcome)
    return NativeFusionExecutionResult(
        _SCHEMA,
        str(verified.path),
        verified.sha256,
        verified.bytes,
        suite.suite_id,
        capability,
        tuple(materialization),
        ordered,
        tuple(serialized_attempts),
        environment,
        stage,
        compile_evidence,
        resource,
        profile,
        validation,
        executor_sources,
        summary,
        outcome,
    )


__all__ = ["NATIVE_RMSNORM_SUITE_SHA256", "NativeFusionExecutionResult", "run_native_fusion_suite"]
