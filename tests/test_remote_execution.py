from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard

import heliostune.local_executor as local
import heliostune.remote_execution as remote
from heliostune.errors import ArtifactError, SchemaError
from heliostune.fusion_execution_registry import (
    FUSION_EXECUTION_REGISTRY,
    fusion_execution_spec,
)
from heliostune.local_executor import CapabilityProbe, LocalExecutionResult, TensorMaterialization
from heliostune.native_fusion_executor import NativeFusionExecutionResult
from heliostune.remote_execution import (
    RECEIPT_LIMITATIONS,
    RECEIPT_SCHEMA,
    REMOTE_RESULT_ENVELOPE_MAX_BYTES,
    REMOTE_RESULT_TRANSPORT_MAX_BYTES,
    SERVER_TIMEOUT_SECONDS,
    TRANSPORT_SCHEMA,
    RemoteIntent,
    RemoteJournal,
    RemoteResultEnvelope,
    canonical_json_bytes,
    canonical_json_line_bytes,
    create_remote_records,
    decode_remote_request,
    encode_remote_request,
    open_remote_records,
    protect_remote_output,
    remote_artifact_paths,
    sha256_bytes,
    validate_remote_result,
    verify_remote_receipt,
    verify_remote_receipt_payloads,
    write_remote_receipt,
)
from heliostune.schema import HardwareProfile
from heliostune.scope import verify_suite
from heliostune.wheel_verifier import source_digest, source_entries

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/suites/gated-mlp-epilogue-v1.json"
PLUGIN = ROOT / "benchmarks/plugins/fusion-reference-plugin-v1.json"
NATIVE_SUITE = ROOT / "benchmarks/suites/residual-rmsnorm-triton-v1.json"
NATIVE_PLUGIN = ROOT / "benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json"


def _intent(tmp_path: Path) -> tuple[RemoteIntent, bytes, bytes, bytes]:
    suite = verify_suite(SUITE)
    plugin = PLUGIN.read_bytes()
    manifest = b'{"supplemental":true}\n'
    return (
        RemoteIntent(
            suite_path=str(SUITE),
            output_path=str((tmp_path / "receipt").absolute()),
            suite_sha256=suite.sha256,
            plugin_path=str(PLUGIN),
            plugin_sha256=sha256_bytes(plugin),
            wheel_filename="heliostune-1.0.0-py3-none-any.whl",
            wheel_sha256="2" * 64,
            manifest_sha256=sha256_bytes(manifest),
            head_commit="4" * 40,
            source_sha256="5" * 64,
        ),
        suite.bytes,
        plugin,
        manifest,
    )


def _native_intent(tmp_path: Path) -> tuple[RemoteIntent, bytes, bytes, bytes]:
    suite = verify_suite(NATIVE_SUITE)
    plugin = NATIVE_PLUGIN.read_bytes()
    manifest = b'{"supplemental":true}\n'
    return (
        RemoteIntent(
            suite_path=str(NATIVE_SUITE),
            output_path=str((tmp_path / "native-receipt").absolute()),
            suite_sha256=suite.sha256,
            plugin_path=str(NATIVE_PLUGIN),
            plugin_sha256=sha256_bytes(plugin),
            wheel_filename="heliostune-1.0.0-py3-none-any.whl",
            wheel_sha256="2" * 64,
            manifest_sha256=sha256_bytes(manifest),
            head_commit="4" * 40,
            source_sha256=source_digest(source_entries(ROOT / "src/heliostune")),
        ),
        suite.bytes,
        plugin,
        manifest,
    )


def _hardware(
    *,
    cuda_version: str | None = "12.8",
    torch_version: str = "2.8.0",
) -> HardwareProfile:
    return HardwareProfile(
        gpu="H100",
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability=(9, 0),
        multiprocessor_count=120,
        total_memory_gb=79.1,
        cuda_version=cuda_version,
        torch_version=torch_version,
        triton_version="3.4.0",
    )


def _aborted_result(
    intent: RemoteIntent,
    suite_bytes: bytes,
    *,
    capability: CapabilityProbe | None = None,
) -> LocalExecutionResult:
    hardware = _hardware()
    if capability is None:
        capability = CapabilityProbe(
            available=False,
            reasons=("inductor_unavailable",),
            torch_version=hardware.torch_version,
            cuda_version=hardware.cuda_version,
            rocm_version=None,
            device_index=0,
            device_name=hardware.device_name,
            compute_capability=hardware.compute_capability,
            native_bf16=True,
            inductor_available=False,
            allocation_succeeded=False,
            detail=None,
        )
    environment = {
        "schema": "heliostune.local-environment/1",
        "python": "3.11.14",
        "implementation": "CPython",
        "platform": "Linux",
        "torch_version": capability.torch_version,
        "cuda_version": capability.cuda_version,
        "rocm_version": capability.rocm_version,
        "device_index": capability.device_index,
        "device_name": capability.device_name,
        "compute_capability": (
            None if capability.compute_capability is None else list(capability.compute_capability)
        ),
        "precision_policy": {
            "float32_matmul_precision": "highest",
            "allow_tf32": False,
            "allow_bf16_reduced_precision_reduction": False,
            "allow_fp16_reduced_precision_reduction": False,
            "allow_fp16_accumulation": False,
        },
        "autocast_policy": {
            "device_type": "cuda",
            "enabled": False,
            "restore_ambient_state": True,
        },
        "backend_invoked": None,
        "fusion_claim": False,
    }
    expected_ids = [cell.id for cell in verify_suite(SUITE).suite.expected_cells]
    return LocalExecutionResult(
        verified_suite_path=intent.suite_path,
        verified_suite_sha256=intent.suite_sha256,
        verified_suite_bytes=suite_bytes,
        suite_id="gated-mlp-epilogue-reference",
        capability=capability,
        materialization=(),
        observations=(),
        attempts=(),
        environment=environment,
        compile_outcomes={},
        summary={
            "expected_cell_ids": expected_ids,
            "terminal_cell_ids": [],
            "passed": 0,
            "failed": 0,
            "blocked": 0,
            "all_cells_terminal": False,
            "outcome": "aborted",
            "fusion_claim": False,
            "candidate_reference_arithmetic": "candidate_reference_identical",
            "candidate_distinction": "fullgraph_inductor_compilation_only",
            "capability_reasons": list(capability.reasons),
        },
        outcome="aborted",
    )


def _envelope(
    intent: RemoteIntent,
    result: LocalExecutionResult,
    request_digest: str,
    *,
    hardware: HardwareProfile | None = None,
) -> RemoteResultEnvelope:
    return RemoteResultEnvelope(
        request_digest=request_digest,
        suite_path=intent.suite_path,
        suite_sha256=intent.suite_sha256,
        plugin_path=intent.plugin_path,
        plugin_sha256=intent.plugin_sha256,
        wheel_filename=intent.wheel_filename,
        wheel_sha256=intent.wheel_sha256,
        manifest_sha256=intent.manifest_sha256,
        head_commit=intent.head_commit,
        source_sha256=intent.source_sha256,
        gpu=intent.gpu,
        gpu_selector=intent.gpu_selector,
        hardware=_hardware() if hardware is None else hardware,
        environment=result.environment,
        result=result.to_dict(),
    )


def _rewrap(transport_json: str, **updates: object) -> str:
    wrapper = json.loads(transport_json)
    assert type(wrapper) is dict
    wrapper.update(updates)
    return canonical_json_line_bytes(wrapper).decode("utf-8")


def _transport_for_bytes(payload: bytes, *, encoding: str = "zstd-base64") -> str:
    compressed = zstandard.ZstdCompressor(
        level=19,
        threads=0,
        write_checksum=True,
        write_content_size=True,
        write_dict_id=False,
    ).compress(payload)
    return canonical_json_line_bytes(
        {
            "schema": TRANSPORT_SCHEMA,
            "encoding": encoding,
            "payload": base64.b64encode(compressed).decode("ascii"),
            "uncompressed_bytes": len(payload),
            "uncompressed_sha256": sha256_bytes(payload),
        }
    ).decode("utf-8")


def _compact_root(payload: bytes) -> dict[str, object]:
    value = remote._compact_decode(payload)
    assert type(value) is dict
    return value


def _compact_bytes(root: dict[str, object]) -> bytes:
    output = bytearray(remote._COMPACT_MAGIC)
    remote._compact_write_value(output, root)
    return bytes(output)


def _compact_tag_count(value: object, tag: str) -> int:
    if type(value) is dict:
        mapping = dict(value)
        return int(tag in mapping) + sum(_compact_tag_count(item, tag) for item in mapping.values())
    if type(value) is list:
        return sum(_compact_tag_count(item, tag) for item in value)
    return 0


def _fixture_digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _fixture_float32(value: float) -> float:
    return float(struct.unpack(">f", struct.pack(">f", value))[0])


def _native_completed_envelope(tmp_path: Path) -> RemoteResultEnvelope:
    intent, suite_bytes, _, _ = _native_intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    native_arms = (
        "rmsnorm-triton-w4",
        "rmsnorm-triton-w8",
        "rmsnorm-triton-w16",
        "rmsnorm-triton-w32",
    )
    eager_arm = "rmsnorm-eager-reference"
    comparator_arm = "rmsnorm-inductor-comparator"
    runtime_arms = (*native_arms, eager_arm, comparator_arm)
    compile_arms = (*native_arms, comparator_arm)
    entrypoints = {
        **{
            arm: f"heliostune_fusion_v2::residual_rmsnorm_w{arm.rsplit('w', 1)[1]}"
            for arm in native_arms
        },
        eager_arm: "reference_template.residual_rmsnorm_reference",
        comparator_arm: "reference_template.residual_rmsnorm_candidate",
    }
    configs = {
        arm: {"block_size": 4096, "num_warps": warps, "num_stages": 1}
        for arm, warps in zip(native_arms, (4, 8, 16, 32), strict=True)
    }
    tensor_hashes = {
        tensor_id: _fixture_digest(f"materialization:{tensor_id}")
        for tensor_id in ("input", "residual", "gamma")
    }
    materialization = [
        {
            "suite_sha256": intent.suite_sha256,
            "case_id": "rmsnorm-case-001",
            "arm_id": arm,
            "input_seed": 17,
            "tensor_order": ["input", "residual", "gamma"],
            "tensors": [
                {
                    "tensor_id": tensor_id,
                    "role": role,
                    "shape": shape,
                    "draw": "normal_0_1_fp32_cpu",
                    "normal_scale": scale,
                    "normal_offset": offset,
                    "cpu_dtype": "float32",
                    "storage_dtype": "bfloat16",
                    "device": "cuda:0",
                    "contiguous": True,
                    "alignment_bytes": 16,
                    "alignment_satisfied": True,
                    "storage_sha256": tensor_hashes[tensor_id],
                }
                for tensor_id, role, shape, scale, offset in (
                    ("input", "input", [128, 4096], 1.0, 0.0),
                    ("residual", "input", [128, 4096], 1.0, 0.0),
                    ("gamma", "parameter", [4096], 0.02, 1.0),
                )
            ],
        }
        for arm in runtime_arms
    ]
    observations: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    for arm_index, arm in enumerate(runtime_arms):
        correctness_key = hashlib.sha256(
            "\0".join(
                (
                    "heliostune.correctness-key/1",
                    intent.suite_sha256,
                    "rmsnorm-case-001",
                    arm,
                    "17",
                    "bf16-fp32-bf16",
                    "highest|tf32=0|bf16rr=0|fp16rr=0|fp16acc=0",
                )
            ).encode()
        ).hexdigest()
        samples = [
            _fixture_float32(0.020 + (0.004 * arm_index) + (sample * 0.00001))
            for sample in range(50)
        ]
        for stage in ("correctness", "timing"):
            cell_id = f"{arm}-{stage}"
            observations.append(
                {
                    "cell_id": cell_id,
                    "case_id": "rmsnorm-case-001",
                    "arm_id": arm,
                    "stage": stage,
                    "status": "passed",
                    "correctness": (
                        {
                            "status": "passed",
                            "correctness_key": correctness_key,
                            "failure_kind": None,
                            "message": None,
                            "output": {
                                "shape": [128, 4096],
                                "device": "cuda:0",
                                "dtype": "torch.bfloat16",
                                "layout": "torch.strided",
                                "contiguous": True,
                            },
                            "input_storage_unchanged": True,
                            "output_disjoint": True,
                            "finite": True,
                            "close": True,
                            "max_abs_error": _fixture_float32(0.00390625),
                        }
                        if stage == "correctness"
                        else None
                    ),
                    "timing": (
                        {
                            "status": "passed",
                            "correctness_key": correctness_key,
                            "failure_kind": None,
                            "message": None,
                            "warmups": 10,
                            "repetitions": 50,
                            "samples_ms": samples,
                            "median_ms": (samples[24] + samples[25]) / 2,
                        }
                        if stage == "timing"
                        else None
                    ),
                }
            )
            attempt_id = len(attempts) + 1
            attempts.extend(
                (
                    {
                        "attempt_id": attempt_id,
                        "cell_id": cell_id,
                        "stage": stage,
                        "status": "running",
                        "from_state": "pending",
                        "to_state": "running",
                        "reason": None,
                    },
                    {
                        "attempt_id": attempt_id + 1,
                        "cell_id": cell_id,
                        "stage": stage,
                        "status": "success",
                        "from_state": "running",
                        "to_state": "passed",
                        "reason": None,
                    },
                )
            )
    compile_evidence: dict[str, dict[str, object]] = {}
    for arm in compile_arms:
        native_arm = arm in native_arms
        compile_evidence[f"{arm}-correctness"] = {
            "case_id": "rmsnorm-case-001",
            "arm_id": arm,
            "entrypoint": entrypoints[arm],
            "backend_kind": "native_triton" if native_arm else "inductor",
            "status": "compiled",
            "error": None,
            "compile_ns": 813_000_000 + len(compile_evidence),
            "backend_invoked": True,
            "fullgraph": not native_arm,
            "dynamic": False,
            "mode": "native_triton" if native_arm else "default",
            "callable_distinct": True,
            "eager_fallback": False,
            "kernel_name": (f"residual_rmsnorm_{arm.rsplit('w', 1)[1]}" if native_arm else None),
            "kernel_hash": _fixture_digest(f"kernel:{arm}") if native_arm else None,
            "config": configs[arm] if native_arm else None,
        }
    resource_evidence: dict[str, dict[str, object]] = {}
    profile_evidence: dict[str, dict[str, object]] = {}
    validation_evidence: dict[str, dict[str, object]] = {}
    for arm in native_arms:
        cell_id = f"{arm}-correctness"
        kernel_name = f"residual_rmsnorm_{arm.rsplit('w', 1)[1]}"
        kernel_hash = _fixture_digest(f"kernel:{arm}")
        resource_evidence[cell_id] = {
            "case_id": "rmsnorm-case-001",
            "arm_id": arm,
            "entrypoint": entrypoints[arm],
            "status": "compiled",
            "error": None,
            "kernel_name": kernel_name,
            "kernel_hash": kernel_hash,
            "target": {"backend": "cuda", "arch": "90", "warp_size": 32},
            "metadata": {
                "shared": 2048,
                "num_warps": configs[arm]["num_warps"],
                "num_ctas": 1,
                "num_stages": 1,
            },
            "n_regs": 56,
            "n_spills": 0,
            "n_max_threads": 1024,
            "asm_stages": [
                {
                    "stage": "cubin",
                    "bytes": 18_000,
                    "sha256": _fixture_digest(f"asm:{arm}:cubin"),
                }
            ],
            "resource_gate_passed": True,
        }
        event_names = [kernel_name]
        profile_evidence[cell_id] = {
            "case_id": "rmsnorm-case-001",
            "arm_id": arm,
            "entrypoint": entrypoints[arm],
            "status": "profiled",
            "error": None,
            "method": "torch.profiler.cuda_events",
            "warmed": True,
            "expected_kernel_name": kernel_name,
            "expected_kernel_hash": kernel_hash,
            "config": configs[arm],
            "invocation_count": 1,
            "cuda_event_count": 1,
            "cuda_event_names_sample": event_names,
            "cuda_event_names_sha256": _fixture_digest("\0".join(event_names)),
            "exact_name_match_count": 1,
            "output_revalidated": True,
            "inputs_revalidated": True,
            "one_kernel_gate_passed": True,
        }
        validation_evidence[cell_id] = {
            "case_id": "rmsnorm-case-001",
            "arm_id": arm,
            "entrypoint": entrypoints[arm],
            "status": "validated",
            "error": None,
            "probes": [
                {
                    "id": probe,
                    "passed": True,
                    "deterministic": True,
                    "inputs_unchanged": True,
                    "output_disjoint": True,
                    "value_class_match": True,
                    "sign_match": True,
                    "finite_close": True,
                    "max_abs_error": _fixture_float32(error),
                }
                for probe, error in (
                    ("zeros", 0.0),
                    ("cancellation", 0.001953125),
                    ("overflow", 0.00390625),
                )
            ],
            "validation_gate_passed": True,
        }
    stage_outcomes = {
        f"{arm}-correctness": {
            "case_id": "rmsnorm-case-001",
            "arm_id": arm,
            "entrypoint": entrypoints[arm],
            "backend_kind": (
                "native_triton"
                if arm in native_arms
                else "eager"
                if arm == eager_arm
                else "inductor"
            ),
            "status": "completed",
            "failure_kind": None,
            "error": None,
            "correctness_passed": True,
            "resource_passed": True if arm in native_arms else None,
            "profile_passed": True if arm in native_arms else None,
            "validation_passed": True if arm in native_arms else None,
            "timing_allowed": True,
        }
        for arm in runtime_arms
    }
    environment = {
        "schema": "heliostune.local-environment/2",
        "python": "3.11.14",
        "implementation": "CPython",
        "platform": "Linux-6.8.0-x86_64",
        "torch_version": "2.8.0+cu128",
        "triton_version": "3.4.0",
        "cuda_version": "12.8",
        "rocm_version": None,
        "device_index": 0,
        "device_name": "NVIDIA H100 80GB HBM3",
        "compute_capability": [9, 0],
        "precision_policy": {
            "float32_matmul_precision": "highest",
            "allow_tf32": False,
            "allow_bf16_reduced_precision_reduction": False,
            "allow_fp16_reduced_precision_reduction": False,
            "allow_fp16_accumulation": False,
        },
        "autocast_policy": {
            "device_type": "cuda",
            "enabled": False,
            "dtype": None,
            "cache_enabled": None,
        },
        "backend_invoked": True,
        "fusion_claim": False,
    }
    package_dir = ROOT / "src/heliostune"
    package_sources = source_entries(package_dir)
    source_names = (
        "fusion_kernels.py",
        "_fusion_gpu.py",
        "native_fusion_executor.py",
        "local_executor.py",
    )
    executor_sources = {
        "schema": "heliostune.executor-sources/1",
        "package_source_sha256": source_digest(package_sources),
        "package_source_count": len(package_sources),
        "sources": [
            {
                "path": name,
                "bytes": len((package_dir / name).read_bytes()),
                "sha256": sha256_bytes((package_dir / name).read_bytes()),
            }
            for name in source_names
        ],
    }
    cell_ids = [f"{arm}-{stage}" for arm in runtime_arms for stage in ("correctness", "timing")]
    result = {
        "schema": "heliostune.local_executor/2",
        "verified_suite_path": intent.suite_path,
        "verified_suite_sha256": intent.suite_sha256,
        "suite_id": "residual-rmsnorm-triton",
        "capability": {
            "available": True,
            "reasons": [],
            "torch_version": "2.8.0+cu128",
            "cuda_version": "12.8",
            "rocm_version": None,
            "device_index": 0,
            "device_name": "NVIDIA H100 80GB HBM3",
            "compute_capability": [9, 0],
            "native_bf16": True,
            "inductor_available": True,
            "allocation_succeeded": True,
            "detail": None,
        },
        "materialization": materialization,
        "observations": observations,
        "attempts": attempts,
        "environment": environment,
        "stage_outcomes": stage_outcomes,
        "compile_evidence": compile_evidence,
        "resource_evidence": resource_evidence,
        "profile_evidence": profile_evidence,
        "validation_evidence": validation_evidence,
        "executor_sources": executor_sources,
        "summary": {
            "expected_cell_ids": cell_ids,
            "terminal_cell_ids": cell_ids,
            "passed": 12,
            "failed": 0,
            "blocked": 0,
            "all_cells_terminal": True,
            "counts": {
                "stage_completed": 6,
                "stage_failed": 0,
                "stage_blocked": 0,
                "compile_compiled": 5,
                "compile_failed": 0,
                "compile_blocked": 0,
                "resource_passed": 4,
                "resource_failed": 0,
                "resource_blocked": 0,
                "profile_passed": 4,
                "profile_failed": 0,
                "profile_blocked": 0,
                "validation_passed": 4,
                "validation_failed": 0,
                "validation_blocked": 0,
            },
            "outcome": "completed",
            "fusion_claim": False,
        },
        "outcome": "completed",
    }
    parsed_result = NativeFusionExecutionResult.from_dict(
        result,
        verified_suite_path=intent.suite_path,
        verified_suite_sha256=intent.suite_sha256,
        verified_suite_bytes=suite_bytes,
    )
    assert parsed_result.to_dict() == result
    return RemoteResultEnvelope(
        request_digest=request_digest,
        suite_path=intent.suite_path,
        suite_sha256=intent.suite_sha256,
        plugin_path=intent.plugin_path,
        plugin_sha256=intent.plugin_sha256,
        wheel_filename=intent.wheel_filename,
        wheel_sha256=intent.wheel_sha256,
        manifest_sha256=intent.manifest_sha256,
        head_commit=intent.head_commit,
        source_sha256=intent.source_sha256,
        gpu=intent.gpu,
        gpu_selector=intent.gpu_selector,
        hardware=_hardware(torch_version="2.8.0+cu128"),
        environment=parsed_result.environment,
        result=parsed_result.to_dict(),
    )


def _failed_compile_result(
    intent: RemoteIntent, suite_bytes: bytes, compile_error: str
) -> LocalExecutionResult:
    suite = verify_suite(SUITE).suite
    capability = CapabilityProbe(
        True,
        (),
        "2.8.0",
        "12.8",
        None,
        0,
        _hardware().device_name,
        (9, 0),
        True,
        True,
        True,
        None,
    )
    environment = dict(_aborted_result(intent, suite_bytes).environment)
    environment.update(
        torch_version=capability.torch_version,
        cuda_version=capability.cuda_version,
        device_index=capability.device_index,
        device_name=capability.device_name,
        compute_capability=list(capability.compute_capability or ()),
        backend_invoked=False,
    )
    attempts: list[dict[str, object]] = []
    observations: list[local.CellObservation] = []
    case = suite.cases[0]
    for cell in suite.expected_cells:
        failure_kind = (
            "compile_failed"
            if cell.stage == "correctness" and cell.arm_id == "mlp-candidate"
            else "correctness_gate"
            if cell.stage == "timing"
            else "runtime"
        )
        message = compile_error if failure_kind == "compile_failed" else "not executed"
        attempts.extend(
            (
                {
                    "attempt_id": len(attempts) + 1,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": "running",
                    "from_state": "pending",
                    "to_state": "running",
                    "reason": None,
                },
                {
                    "attempt_id": len(attempts) + 2,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": "failure",
                    "from_state": "running",
                    "to_state": "failed",
                    "reason": failure_kind,
                },
            )
        )
        key = local._correctness_gate_key(intent.suite_sha256, case, cell)
        correctness = (
            local.CorrectnessObservation(
                "failed",
                key,
                failure_kind,
                message,
                None,
                False,
                False,
                False,
                False,
                None,
            )
            if cell.stage == "correctness"
            else None
        )
        timing = (
            local.TimingObservation("failed", key, failure_kind, message, 0, 0, (), None)
            if cell.stage == "timing"
            else None
        )
        observations.append(
            local.CellObservation(
                cell.id,
                cell.case_id,
                cell.arm_id,
                cell.stage,
                "failed",
                correctness,
                timing,
            )
        )
    expected_ids = [cell.id for cell in suite.expected_cells]
    return LocalExecutionResult(
        intent.suite_path,
        intent.suite_sha256,
        suite_bytes,
        suite.suite_id,
        capability,
        (),
        tuple(observations),
        tuple(attempts),
        environment,
        {
            "mlp-candidate": {
                "case_id": case.id,
                "arm_id": "mlp-candidate",
                "entrypoint": "reference_template.gated_mlp_candidate",
                "status": "compile_failed",
                "error": compile_error,
                "wrapper_create_ns": 1,
                "first_call_ns": None,
                "eager_fallback": False,
                "backend_invoked": False,
                "callable_distinct": False,
                "autocast_policy": dict(local._AUTOCAST_POLICY),
            }
        },
        {
            "expected_cell_ids": expected_ids,
            "terminal_cell_ids": expected_ids,
            "passed": 0,
            "failed": len(expected_ids),
            "blocked": 0,
            "all_cells_terminal": True,
            "outcome": "failed",
            "fusion_claim": False,
            "candidate_reference_arithmetic": "candidate_reference_identical",
            "candidate_distinction": "fullgraph_inductor_compilation_only",
        },
        "failed",
    )


def test_result_envelope_freezes_inline_utf8_boundary(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    base = _envelope(intent, _aborted_result(intent, suite_bytes), request_digest)
    empty = canonical_json_bytes({**base.to_dict(), "result": {"padding": ""}})
    padding = "x" * (REMOTE_RESULT_ENVELOPE_MAX_BYTES - len(empty))
    at_limit = replace(base, result={"padding": padding}).to_json()

    assert len(at_limit.encode("utf-8")) == REMOTE_RESULT_ENVELOPE_MAX_BYTES
    assert RemoteResultEnvelope.from_json(at_limit).result == {"padding": padding}
    with pytest.raises(SchemaError, match="inline limit") as caught:
        replace(base, result={"padding": padding + "x"}).to_json()
    assert len(str(caught.value).encode("utf-8")) < 128
    with pytest.raises(SchemaError, match="inline limit"):
        RemoteResultEnvelope.from_json(at_limit + "x")


def test_realistic_completed_four_cell_transport_is_deterministic_and_inline(
    tmp_path: Path,
) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    result = _failed_compile_result(intent, suite_bytes, "representative compile failure")
    base = _envelope(intent, result, request_digest)
    cells = verify_suite(SUITE).suite.expected_cells
    observations: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        correctness_key = sha256_bytes(
            f"{intent.suite_sha256}:{cell.case_id}:{cell.arm_id}".encode()
        )
        samples = [float(sample) / 1000 for sample in range(1, 51)]
        observations.append(
            {
                "cell_id": cell.id,
                "case_id": cell.case_id,
                "arm_id": cell.arm_id,
                "stage": cell.stage,
                "status": "passed",
                "correctness": (
                    {
                        "status": "passed",
                        "correctness_key": correctness_key,
                        "failure_kind": None,
                        "message": None,
                        "output": {
                            "shape": [8, 11008],
                            "device": "cuda:0",
                            "dtype": "torch.bfloat16",
                            "layout": "torch.strided",
                            "contiguous": True,
                        },
                        "input_storage_unchanged": True,
                        "output_disjoint": True,
                        "finite": True,
                        "close": True,
                        "max_abs_error": 0.0,
                    }
                    if cell.stage == "correctness"
                    else None
                ),
                "timing": (
                    {
                        "status": "passed",
                        "correctness_key": correctness_key,
                        "failure_kind": None,
                        "message": None,
                        "warmups": 10,
                        "repetitions": 50,
                        "samples_ms": samples,
                        "median_ms": 0.0255,
                    }
                    if cell.stage == "timing"
                    else None
                ),
            }
        )
        attempts.extend(
            (
                {
                    "attempt_id": 2 * index + 1,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": "running",
                    "from_state": "pending",
                    "to_state": "running",
                    "reason": None,
                },
                {
                    "attempt_id": 2 * index + 2,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": "success",
                    "from_state": "running",
                    "to_state": "passed",
                    "reason": None,
                },
            )
        )
    completed = result.to_dict()
    completed["observations"] = observations
    completed["attempts"] = attempts
    completed["compile_outcomes"] = {
        "mlp-candidate": {
            **dict(result.compile_outcomes["mlp-candidate"]),
            "status": "compiled_and_first_call_completed",
            "error": None,
            "backend_invoked": True,
            "callable_distinct": True,
        }
    }
    completed["summary"] = {
        **dict(result.summary),
        "expected_cell_ids": [cell.id for cell in cells],
        "terminal_cell_ids": [cell.id for cell in cells],
        "passed": len(cells),
        "failed": 0,
        "blocked": 0,
        "all_cells_terminal": True,
        "outcome": "completed",
        "fusion_claim": False,
    }
    completed["outcome"] = "completed"
    envelope = replace(base, result=completed)

    transport = envelope.to_transport_json()
    wrapper = json.loads(transport)

    assert len(cells) == 4
    assert len(transport.encode("utf-8")) < REMOTE_RESULT_TRANSPORT_MAX_BYTES
    assert len(pickle.dumps(transport, protocol=4)) < 8 * 1024
    assert set(wrapper) == {
        "schema",
        "encoding",
        "payload",
        "uncompressed_bytes",
        "uncompressed_sha256",
    }
    assert wrapper["schema"] == TRANSPORT_SCHEMA
    assert wrapper["encoding"] == "zstd-base64"
    assert transport == envelope.to_transport_json()
    assert RemoteResultEnvelope.from_transport_json(transport) == envelope


def test_native_completed_transport_uses_compact_fallback_with_margin(tmp_path: Path) -> None:
    envelope = _native_completed_envelope(tmp_path)
    envelope_bytes = envelope.to_json().encode("utf-8")
    legacy = _transport_for_bytes(envelope_bytes)
    legacy_wrapper = json.loads(legacy)
    assert type(legacy_wrapper) is dict
    legacy_frame = base64.b64decode(str(legacy_wrapper["payload"]), validate=True)
    assert zstandard.ZstdDecompressor().decompress(legacy_frame) == envelope_bytes

    transport = envelope.to_transport_json()
    wrapper = json.loads(transport)
    assert type(wrapper) is dict
    compact_size = len(transport.encode("utf-8"))
    legacy_size = len(legacy.encode("utf-8"))
    packed_size = int(wrapper["uncompressed_bytes"])
    serialized_size = len(pickle.dumps(transport, protocol=4))
    margin_limit = (REMOTE_RESULT_TRANSPORT_MAX_BYTES * 9) // 10
    assert legacy_size > REMOTE_RESULT_TRANSPORT_MAX_BYTES
    assert packed_size <= REMOTE_RESULT_ENVELOPE_MAX_BYTES
    assert compact_size <= margin_limit
    assert serialized_size <= margin_limit
    assert wrapper["encoding"] == "zstd-base64-compact-v1"
    assert transport == envelope.to_transport_json()
    assert RemoteResultEnvelope.from_transport_json(transport) == envelope

    intent, suite_bytes, _, _ = _native_intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    assert request_digest == envelope.request_digest
    parsed_envelope, parsed_result = validate_remote_result(
        transport,
        intent=intent,
        request_digest=request_digest,
        verified_suite_bytes=suite_bytes,
    )
    assert parsed_envelope == envelope
    assert isinstance(parsed_result, NativeFusionExecutionResult)
    assert parsed_result.to_dict() == envelope.result
    assert parsed_envelope.result == parsed_result.to_dict()

    result = envelope.result
    assert type(result) is dict
    assert len(result["materialization"]) == 6
    assert len(result["observations"]) == 12
    assert len(result["attempts"]) == 24
    assert len(result["stage_outcomes"]) == 6
    assert len(result["compile_evidence"]) == 5
    assert len(result["resource_evidence"]) == 4
    assert len(result["validation_evidence"]) == 4
    assert len(result["profile_evidence"]) == 4
    compact_frame = base64.b64decode(str(wrapper["payload"]), validate=True)
    packed = zstandard.ZstdDecompressor().decompress(compact_frame)
    assert len(packed) == packed_size
    compact_root = _compact_root(packed)
    assert _compact_tag_count(compact_root, "$f4") == 6
    assert _compact_tag_count(compact_root, "$f8") == 0


def test_compact_uint64_codec_is_bounded_canonical_and_rejects_continuation_bomb() -> None:
    encoded_max = bytearray()
    remote._compact_write_uint(encoded_max, (1 << 64) - 1)
    assert bytes(encoded_max) == (b"\xff" * 9) + b"\x01"
    assert remote._compact_read_uint(bytes(encoded_max), 0) == ((1 << 64) - 1, 10)

    for unsupported in (-1, 1 << 64):
        output = bytearray()
        with pytest.raises(SchemaError, match="outside the uint64 range"):
            remote._compact_write_uint(output, unsupported)
        assert output == b""

    for supported in (-(1 << 63), (1 << 63) - 1):
        output = bytearray()
        remote._compact_write_value(output, supported)
        decoded, offset = remote._compact_read_value(bytes(output), 0)
        assert (decoded, offset) == (supported, len(output))

    for unsupported in (-(1 << 63) - 1, 1 << 63):
        output = bytearray()
        with pytest.raises(SchemaError, match="outside the int64 range"):
            remote._compact_write_value(output, unsupported)
        assert output == b""

    for malformed, match in (
        (b"\x80\x00", "noncanonical integer"),
        ((b"\xff" * 9) + b"\x00", "noncanonical integer"),
        ((b"\xff" * 9) + b"\x02", "outside the uint64 range"),
        ((b"\xff" * 10), "outside the uint64 range"),
    ):
        with pytest.raises(SchemaError, match=match):
            remote._compact_read_uint(malformed, 0)

    continuation_count = REMOTE_RESULT_ENVELOPE_MAX_BYTES - len(remote._COMPACT_MAGIC) - 1
    packed = remote._COMPACT_MAGIC + b"\x07" + (b"\xff" * continuation_count)
    transport = _transport_for_bytes(packed, encoding="zstd-base64-compact-v1")
    wrapper = json.loads(transport)
    assert len(packed) == REMOTE_RESULT_ENVELOPE_MAX_BYTES
    assert int(wrapper["uncompressed_bytes"]) == REMOTE_RESULT_ENVELOPE_MAX_BYTES
    assert len(transport.encode("utf-8")) <= (REMOTE_RESULT_TRANSPORT_MAX_BYTES * 9) // 10
    with pytest.raises(SchemaError, match="outside the uint64 range"):
        RemoteResultEnvelope.from_transport_json(transport)


def test_compact_float64_fallback_and_exact_digest_whitelist(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    base = _envelope(intent, _aborted_result(intent, suite_bytes), request_digest)
    envelope = replace(
        base,
        result={
            "samples_ms": [0.1],
            "digest": "a" * 64,
            "arbitrary_sha256": "b" * 64,
            "sha256": "c" * 64,
        },
    )
    compact = remote._compact_envelope_bytes(envelope.to_dict())
    compact_root = _compact_root(compact)
    packed_envelope = compact_root["envelope"]
    assert type(packed_envelope) is dict
    packed_result = packed_envelope["result"]
    assert type(packed_result) is dict

    assert set(packed_result["samples_ms"]) == {"$f8"}
    assert packed_result["digest"] == "a" * 64
    assert packed_result["arbitrary_sha256"] == "b" * 64
    sha_tag = packed_result["sha256"]
    assert type(sha_tag) is dict
    assert set(sha_tag) == {"$h"}
    assert sha_tag["$h"] == bytes.fromhex("c" * 64)
    assert remote._compact_unpack_value(packed_envelope) == envelope.to_dict()
    transport = _transport_for_bytes(compact, encoding="zstd-base64-compact-v1")
    with pytest.raises(SchemaError, match="canonical encoding"):
        RemoteResultEnvelope.from_transport_json(transport)


def test_compact_rejects_malformed_ambiguous_and_colliding_tags(tmp_path: Path) -> None:
    envelope = _native_completed_envelope(tmp_path)
    compact = remote._compact_envelope_bytes(envelope.to_dict())
    root = _compact_root(compact)
    packed_envelope = root["envelope"]
    assert type(packed_envelope) is dict
    digest_mapping = packed_envelope["request_digest"]
    assert type(digest_mapping) is dict
    digest_tag = digest_mapping["$h"]
    assert type(digest_tag) is bytes

    malformed_roots: list[tuple[dict[str, object], str]] = []
    wrong_length = _compact_root(compact)
    wrong_length_envelope = wrong_length["envelope"]
    assert type(wrong_length_envelope) is dict
    wrong_length_envelope["request_digest"] = {"$h": b"\x00"}
    malformed_roots.append((wrong_length, "wrong byte length"))
    wrong_type = _compact_root(compact)
    wrong_type_envelope = wrong_type["envelope"]
    assert type(wrong_type_envelope) is dict
    wrong_type_envelope["request_digest"] = {"$h": "not-bytes"}
    malformed_roots.append((wrong_type, "wrong byte length"))
    ambiguous = _compact_root(compact)
    ambiguous_envelope = ambiguous["envelope"]
    assert type(ambiguous_envelope) is dict
    ambiguous_envelope["request_digest"] = {"$h": digest_tag, "$f4": b""}
    malformed_roots.append((ambiguous, "ambiguous tag"))
    wrong_context = _compact_root(compact)
    wrong_context_envelope = wrong_context["envelope"]
    assert type(wrong_context_envelope) is dict
    wrong_context_result = wrong_context_envelope["result"]
    assert type(wrong_context_result) is dict
    wrong_context_result["ordinary"] = {"$h": digest_tag}
    malformed_roots.append((wrong_context, "not valid for this field"))
    wrong_float_length = _compact_root(compact)
    wrong_float_envelope = wrong_float_length["envelope"]
    assert type(wrong_float_envelope) is dict
    wrong_float_result = wrong_float_envelope["result"]
    assert type(wrong_float_result) is dict
    wrong_float_observations = wrong_float_result["observations"]
    assert type(wrong_float_observations) is list
    timing = next(
        item["timing"]
        for item in wrong_float_observations
        if type(item) is dict and item["timing"] is not None
    )
    assert type(timing) is dict
    timing["samples_ms"] = {"$f4": b"\x00"}
    malformed_roots.append((wrong_float_length, "wrong decoded byte length"))

    for malformed, match in malformed_roots:
        with pytest.raises(SchemaError, match=match):
            RemoteResultEnvelope.from_transport_json(
                _transport_for_bytes(
                    _compact_bytes(malformed),
                    encoding="zstd-base64-compact-v1",
                )
            )

    collision = envelope.to_dict()
    assert isinstance(collision["result"], dict)
    collision["result"]["collision"] = {"$h": digest_tag}
    with pytest.raises(SchemaError, match="reserved tag"):
        remote._compact_envelope_bytes(collision)
    with pytest.raises(SchemaError, match="finite floats"):
        remote._compact_envelope_bytes(
            {**envelope.to_dict(), "result": {"samples_ms": [float("inf")]}}
        )


def test_compact_rejects_noncanonical_float_width_binary_zstd_and_wrapper(
    tmp_path: Path,
) -> None:
    envelope = _native_completed_envelope(tmp_path)
    compact = remote._compact_envelope_bytes(envelope.to_dict())
    transport = _transport_for_bytes(compact, encoding="zstd-base64-compact-v1")
    wrapper = json.loads(transport)
    assert type(wrapper) is dict

    root = _compact_root(compact)
    packed_envelope = root["envelope"]
    assert type(packed_envelope) is dict
    packed_result = packed_envelope["result"]
    assert type(packed_result) is dict
    observations = packed_result["observations"]
    assert type(observations) is list
    timing = next(
        item["timing"] for item in observations if type(item) is dict and item["timing"] is not None
    )
    assert type(timing) is dict
    packed_samples = timing["samples_ms"]
    assert type(packed_samples) is dict
    raw_float32 = packed_samples["$f4"]
    assert type(raw_float32) is bytes
    float32_values = struct.unpack(f">{len(raw_float32) // 4}f", raw_float32)
    float64_raw = struct.pack(f">{len(float32_values)}d", *float32_values)
    timing["samples_ms"] = {"$f8": float64_raw}
    nonminimal = _compact_bytes(root)
    with pytest.raises(SchemaError, match="canonical compact form"):
        RemoteResultEnvelope.from_transport_json(
            _transport_for_bytes(nonminimal, encoding="zstd-base64-compact-v1")
        )

    with pytest.raises(SchemaError, match="binary magic"):
        RemoteResultEnvelope.from_transport_json(
            _transport_for_bytes(
                compact.removeprefix(remote._COMPACT_MAGIC),
                encoding="zstd-base64-compact-v1",
            )
        )

    with pytest.raises(SchemaError, match="trailing bytes"):
        RemoteResultEnvelope.from_transport_json(
            _transport_for_bytes(
                compact + b"\x00",
                encoding="zstd-base64-compact-v1",
            )
        )

    noncanonical_frame = zstandard.ZstdCompressor(
        level=1,
        write_checksum=False,
        write_content_size=True,
        write_dict_id=False,
    ).compress(compact)
    with pytest.raises(SchemaError, match="canonical zstd frame"):
        RemoteResultEnvelope.from_transport_json(
            _rewrap(
                transport,
                payload=base64.b64encode(noncanonical_frame).decode("ascii"),
            )
        )

    with pytest.raises(SchemaError, match="digest does not match"):
        RemoteResultEnvelope.from_transport_json(_rewrap(transport, uncompressed_sha256="0" * 64))
    with pytest.raises(SchemaError, match="byte count"):
        RemoteResultEnvelope.from_transport_json(
            _rewrap(
                transport,
                uncompressed_bytes=int(wrapper["uncompressed_bytes"]) + 1,
            )
        )
    frame = base64.b64decode(str(wrapper["payload"]), validate=True)
    with pytest.raises(SchemaError, match="one bounded zstd frame"):
        RemoteResultEnvelope.from_transport_json(
            _rewrap(
                transport,
                payload=base64.b64encode(frame + b"trailing").decode("ascii"),
            )
        )


def test_legacy_transport_wrapper_bytes_are_unchanged(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    envelope = _envelope(intent, _aborted_result(intent, suite_bytes), request_digest)
    expected = _transport_for_bytes(envelope.to_json().encode("utf-8"))

    assert envelope.to_transport_json() == expected
    assert json.loads(expected)["encoding"] == "zstd-base64"
    assert RemoteResultEnvelope.from_transport_json(expected) == envelope


def test_transport_freezes_exact_utf8_boundary_and_bounded_error(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    intent = replace(intent, output_path="/tmp/heliostune-transport-boundary")
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    base = _envelope(intent, _aborted_result(intent, suite_bytes), request_digest)
    entropy = "".join(hashlib.sha256(str(index).encode()).hexdigest() for index in range(1000))

    within_limit = replace(base, result={"padding": entropy[:6500]}).to_transport_json()

    assert len(pickle.dumps(within_limit, protocol=4)) < 8 * 1024
    assert len(within_limit.encode("utf-8")) <= REMOTE_RESULT_TRANSPORT_MAX_BYTES
    assert RemoteResultEnvelope.from_transport_json(within_limit).result == {
        "padding": entropy[:6500]
    }
    with pytest.raises(SchemaError, match="inline limit") as caught:
        replace(base, result={"padding": entropy[:12000]}).to_transport_json()
    assert len(str(caught.value).encode("utf-8")) < 128
    with pytest.raises(SchemaError, match="inline limit"):
        RemoteResultEnvelope.from_transport_json(within_limit + ("x" * 6144))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", 1, "schema must be a string"),
        ("encoding", 1, "encoding must be a string"),
        ("payload", 1, "payload must be a string"),
        ("uncompressed_bytes", True, "uncompressed_bytes must be an integer"),
        ("uncompressed_sha256", 1, "uncompressed_sha256 must be a string"),
    ],
)
def test_transport_requires_exact_field_types(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    transport = _envelope(
        intent, _aborted_result(intent, suite_bytes), request_digest
    ).to_transport_json()

    with pytest.raises(SchemaError, match=match):
        RemoteResultEnvelope.from_transport_json(_rewrap(transport, **{field: value}))


def test_transport_rejects_base64_digest_count_encoding_and_unknown_fields(
    tmp_path: Path,
) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    transport = _envelope(
        intent, _aborted_result(intent, suite_bytes), request_digest
    ).to_transport_json()
    wrapper = json.loads(transport)
    assert type(wrapper) is dict

    with pytest.raises(SchemaError, match="canonical base64"):
        RemoteResultEnvelope.from_transport_json(_rewrap(transport, payload="not-base64!"))
    with pytest.raises(SchemaError, match="digest does not match"):
        RemoteResultEnvelope.from_transport_json(_rewrap(transport, uncompressed_sha256="0" * 64))
    with pytest.raises(SchemaError, match="byte count"):
        RemoteResultEnvelope.from_transport_json(
            _rewrap(
                transport,
                uncompressed_bytes=int(wrapper["uncompressed_bytes"]) + 1,
            )
        )
    with pytest.raises(SchemaError, match="encoding must be 'zstd-base64'"):
        RemoteResultEnvelope.from_transport_json(_rewrap(transport, encoding="gzip-base64"))
    wrapper["extra"] = None
    with pytest.raises(SchemaError, match="unknown fields"):
        RemoteResultEnvelope.from_transport_json(canonical_json_line_bytes(wrapper).decode("utf-8"))


def test_transport_rejects_bomb_trailing_data_and_second_frame(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    transport = _envelope(
        intent, _aborted_result(intent, suite_bytes), request_digest
    ).to_transport_json()
    wrapper = json.loads(transport)
    assert type(wrapper) is dict
    frame = base64.b64decode(str(wrapper["payload"]), validate=True)
    empty_frame = zstandard.ZstdCompressor(write_content_size=True).compress(b"")

    bomb = b"x" * (REMOTE_RESULT_ENVELOPE_MAX_BYTES + 1)
    bomb_transport = _transport_for_bytes(bomb)
    assert len(bomb_transport.encode("utf-8")) < REMOTE_RESULT_TRANSPORT_MAX_BYTES
    with pytest.raises(SchemaError, match="byte count exceeds limit"):
        RemoteResultEnvelope.from_transport_json(bomb_transport)

    for suffix in (b"trailing", empty_frame):
        malformed = _rewrap(
            transport,
            payload=base64.b64encode(frame + suffix).decode("ascii"),
        )
        with pytest.raises(SchemaError, match="one bounded zstd frame"):
            RemoteResultEnvelope.from_transport_json(malformed)


def test_transport_rejects_noncanonical_wrapper_and_envelope(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    envelope = _envelope(intent, _aborted_result(intent, suite_bytes), request_digest)
    transport = envelope.to_transport_json()

    with pytest.raises(SchemaError, match="canonical JSON"):
        RemoteResultEnvelope.from_transport_json(
            canonical_json_bytes(json.loads(transport)).decode("utf-8")
        )
    envelope_bytes = envelope.to_json().encode("utf-8")
    noncanonical_frame = zstandard.ZstdCompressor(
        level=1,
        write_checksum=False,
        write_content_size=True,
        write_dict_id=False,
    ).compress(envelope_bytes)
    with pytest.raises(SchemaError, match="canonical zstd frame"):
        RemoteResultEnvelope.from_transport_json(
            _rewrap(
                transport,
                payload=base64.b64encode(noncanonical_frame).decode("ascii"),
            )
        )
    with pytest.raises(SchemaError, match="canonical JSON"):
        RemoteResultEnvelope.from_transport_json(
            _transport_for_bytes(b" " + envelope.to_json().encode("utf-8"))
        )


def test_failed_compile_huge_error_envelope_is_bounded_and_validates(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    compile_error = local._safe_error(
        TypeError("meta_mm() takes 2 positional arguments but 3 were given\n" + ("x" * 2_000_000))
    )
    result = _failed_compile_result(intent, suite_bytes, compile_error)
    envelope = _envelope(intent, result, request_digest)
    payload = envelope.to_transport_json()

    assert len(compile_error.encode("utf-8")) <= 4096
    assert len(payload.encode("utf-8")) < REMOTE_RESULT_TRANSPORT_MAX_BYTES
    assert len(pickle.dumps(payload, protocol=4)) < 8 * 1024
    parsed_envelope, parsed_result = validate_remote_result(
        payload,
        intent=intent,
        request_digest=request_digest,
        verified_suite_bytes=suite_bytes,
    )
    assert parsed_envelope.to_json() == envelope.to_json()
    assert parsed_envelope.result["outcome"] == "failed"
    assert isinstance(parsed_result, LocalExecutionResult)
    assert parsed_result.compile_outcomes["mlp-candidate"]["error"] == compile_error


def test_request_is_canonical_and_binds_exact_suite_bytes(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    payload = encode_remote_request(intent, suite_bytes)
    decoded_intent, decoded_bytes, request_digest = decode_remote_request(payload)
    assert decoded_intent == intent
    assert decoded_bytes == suite_bytes
    assert len(request_digest) == 64
    forged = json.loads(payload)
    forged["suite_utf8"] += " "
    with pytest.raises(SchemaError, match="suite digest"):
        decode_remote_request(canonical_json_bytes(forged).decode())
    with pytest.raises(SchemaError, match="canonical"):
        decode_remote_request(" " + payload)


def test_pinned_records_are_exclusive_and_spawn_acknowledgement_is_terminal(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        assert records.intent_bytes() == intent.to_bytes()
        assert records.intent_snapshot == intent.to_bytes()
        assert records.journal_snapshot is None
        records.journal.append("spawn_acknowledgement_lost", detail="RuntimeError: transport")
        records.journal.append("unresolved", detail="RuntimeError: transport")
        rows = [json.loads(line) for line in records.journal.bytes().splitlines()]
        assert [row["state"] for row in rows] == [
            "intent",
            "spawn_acknowledgement_lost",
            "unresolved",
        ]
        with pytest.raises(ArtifactError, match="remote intent"):
            protect_remote_output(intent.output_path)
    finally:
        records.close()


def test_replaced_tombstone_path_cannot_hide_or_forge_retained_intent(
    tmp_path: Path,
) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        intent_path, _ = remote_artifact_paths(intent.output_path)
        intent_path.unlink()
        intent_path.write_bytes(canonical_json_bytes({"forged": True}))
        assert records.intent_bytes() == intent.to_bytes()
        with pytest.raises(ArtifactError, match="tombstone identity changed"):
            records.assert_parent_identity()
    finally:
        records.close()


def test_journal_rejects_illegal_transition_and_duplicate_path(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    journal = RemoteJournal.create(path, "a" * 64)
    with pytest.raises(SchemaError, match="illegal"):
        journal.append("completed", call_id="fc-1")
    journal.close()
    with pytest.raises(FileExistsError):
        RemoteJournal.create(path, "a" * 64)


def test_result_envelope_uses_strict_local_parser_and_rejects_contradiction(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    envelope = _envelope(intent, _aborted_result(intent, suite_bytes), request_digest)
    parsed, result = validate_remote_result(
        envelope.to_transport_json(),
        intent=intent,
        request_digest=request_digest,
        verified_suite_bytes=suite_bytes,
    )
    assert parsed == envelope
    assert result.outcome == "aborted"
    forged = envelope.to_dict()
    assert isinstance(forged["result"], dict)
    assert isinstance(forged["result"]["capability"], dict)
    forged["result"]["capability"]["inductor_available"] = True
    forged_envelope = RemoteResultEnvelope.from_json(canonical_json_bytes(forged).decode())
    with pytest.raises(SchemaError, match="inconsistent probe evidence"):
        validate_remote_result(
            forged_envelope.to_transport_json(),
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_bytes,
        )


@pytest.mark.parametrize(
    ("capability", "hardware"),
    [
        (
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
                "ModuleNotFoundError: No module named 'torch'",
            ),
            _hardware(),
        ),
        (
            CapabilityProbe(
                False,
                ("torch_version_mismatch",),
                "2.7.1",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                None,
            ),
            _hardware(torch_version="2.7.1"),
        ),
        (
            CapabilityProbe(
                False,
                ("cuda_unavailable",),
                "2.8.0",
                "12.8",
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                None,
            ),
            _hardware(),
        ),
    ],
    ids=("torch-missing", "torch-version-mismatch", "cuda-unavailable"),
)
def test_valid_early_capability_aborts_preserve_null_stage_evidence(
    tmp_path: Path,
    capability: CapabilityProbe,
    hardware: HardwareProfile,
) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    envelope = _envelope(
        intent,
        _aborted_result(intent, suite_bytes, capability=capability),
        request_digest,
        hardware=hardware,
    )
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawned", call_id="fc-early-abort")
        records.journal.append("retrieval_started", call_id="fc-early-abort")
        records.journal.append("aborted", call_id="fc-early-abort")
        write_remote_receipt(
            records,
            status="aborted",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=envelope.to_json(),
            client_spawn_count=1,
        )
    finally:
        records.close()
    verified = verify_remote_receipt(intent.output_path)
    assert verified.receipt.status == "aborted"
    assert verified.result is not None and verified.result.capability == capability
    _, result = validate_remote_result(
        envelope.to_transport_json(),
        intent=intent,
        request_digest=request_digest,
        verified_suite_bytes=suite_bytes,
    )
    assert result.outcome == "aborted"
    assert result.capability == capability


def test_early_capability_abort_rejects_hardware_and_environment_contradictions(
    tmp_path: Path,
) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    capability = CapabilityProbe(
        False,
        ("torch_version_mismatch",),
        "2.7.1",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        None,
    )
    result = _aborted_result(intent, suite_bytes, capability=capability)
    mismatched = _envelope(intent, result, request_digest)
    with pytest.raises(SchemaError, match="capability field torch_version"):
        validate_remote_result(
            mismatched.to_transport_json(),
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_bytes,
        )

    fabricated = _envelope(
        intent,
        _aborted_result(
            intent,
            suite_bytes,
            capability=CapabilityProbe(
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
                "ModuleNotFoundError: No module named 'torch'",
            ),
        ),
        request_digest,
    ).to_dict()
    assert isinstance(fabricated["environment"], dict)
    assert isinstance(fabricated["result"], dict)
    assert isinstance(fabricated["result"]["environment"], dict)
    fabricated["environment"]["device_name"] = _hardware().device_name
    fabricated["result"]["environment"]["device_name"] = _hardware().device_name
    fabricated_envelope = RemoteResultEnvelope.from_json(canonical_json_bytes(fabricated).decode())
    with pytest.raises(SchemaError, match="environment does not match its capability"):
        validate_remote_result(
            fabricated_envelope.to_transport_json(),
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_bytes,
        )


def test_aborted_receipt_is_root_last_strict_and_tamper_evident(tmp_path: Path) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    result_payload = _envelope(
        intent, _aborted_result(intent, suite_bytes), request_digest
    ).to_json()
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawned", call_id="fc-one")
        records.journal.append("retrieval_started", call_id="fc-one")
        records.journal.append("aborted", call_id="fc-one")
        published = write_remote_receipt(
            records,
            status="aborted",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=result_payload,
            client_spawn_count=1,
        )
    finally:
        records.close()
    assert published.root_path == Path(intent.output_path) / "receipt.json"
    verified = verify_remote_receipt(intent.output_path)
    assert verified.receipt.schema == RECEIPT_SCHEMA
    assert verified.receipt.status == "aborted"
    assert verified.result is not None and verified.result.outcome == "aborted"
    receipt_files = {path.name: path.read_bytes() for path in Path(intent.output_path).iterdir()}
    verified_payloads = verify_remote_receipt_payloads(
        receipt_files, logical_output_path=intent.output_path
    )
    assert verified_payloads.receipt == verified.receipt
    assert verified_payloads.intent == verified.intent
    assert verified_payloads.journal == verified.journal
    assert verified_payloads.envelope == verified.envelope
    assert verified_payloads.result == verified.result
    assert verified_payloads.root_path == Path(intent.output_path) / "receipt.json"
    with pytest.raises(SchemaError, match="location differs"):
        verify_remote_receipt_payloads(receipt_files, logical_output_path=tmp_path / "other-output")
    root = json.loads((Path(intent.output_path) / "receipt.json").read_text())
    assert root["provider_attempts_observable"] is False
    assert root["provider_physical_attempts"] is None
    assert root["client_spawn_count"] == 1
    assert root["per_execution_timeout_s"] == SERVER_TIMEOUT_SECONDS
    assert root["total_gpu_seconds_upper_bound"] is None
    assert root["actual_cost_usd"] is None
    assert root["publication_eligible"] is False
    assert root["attestation"] == "none"
    assert tuple(root["limitations"]) == RECEIPT_LIMITATIONS
    assert "heliostune.bundle/1" not in (Path(intent.output_path) / "receipt.json").read_text()

    result_path = Path(intent.output_path) / root["result"]["path"]
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(ArtifactError, match="digest/size"):
        verify_remote_receipt(intent.output_path)


def test_live_writer_retains_authorized_journal_if_same_inode_mutates_after_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    result_payload = _envelope(
        intent, _aborted_result(intent, suite_bytes), request_digest
    ).to_json()
    records = create_remote_records(intent.output_path, intent, request_digest)
    records.journal.append("spawned", call_id="fc-one")
    records.journal.append("retrieval_started", call_id="fc-one")
    records.journal.append("aborted", call_id="fc-one")
    authorized = records.journal.bytes()
    forged = authorized.replace(b"fc-one", b"fc-two")
    assert len(forged) == len(authorized) and forged != authorized
    _, journal_path = remote_artifact_paths(intent.output_path)
    journal_identity = journal_path.stat().st_ino
    original_mkdir = os.mkdir
    mutated = False

    def mkdir_after_boundary(path: str, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        nonlocal mutated
        if not mutated and path.startswith(".heliostune-remote-receipt-"):
            descriptor = os.open(journal_path, os.O_WRONLY)
            try:
                assert os.pwrite(descriptor, forged, 0) == len(forged)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            mutated = True
        original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr("heliostune.remote_execution.os.mkdir", mkdir_after_boundary)
    try:
        write_remote_receipt(
            records,
            status="aborted",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=result_payload,
            client_spawn_count=1,
        )
        assert records.journal_snapshot == authorized
    finally:
        records.close()

    assert mutated
    assert journal_path.stat().st_ino == journal_identity
    assert journal_path.read_bytes() == forged
    assert (Path(intent.output_path) / "journal.jsonl").read_bytes() == authorized
    assert b"fc-two" not in (Path(intent.output_path) / "journal.jsonl").read_bytes()


def test_receipt_rejects_semantic_status_and_path_tampering(tmp_path: Path) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    payload = _envelope(intent, _aborted_result(intent, suite_bytes), request_digest).to_json()
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawned", call_id="fc-one")
        records.journal.append("retrieval_started", call_id="fc-one")
        records.journal.append("aborted", call_id="fc-one")
        write_remote_receipt(
            records,
            status="aborted",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=payload,
            client_spawn_count=1,
        )
    finally:
        records.close()
    root_path = Path(intent.output_path) / "receipt.json"
    root = json.loads(root_path.read_text())
    root["status"] = "failed"
    root_path.write_bytes(canonical_json_bytes(root))
    with pytest.raises(SchemaError, match="status differs"):
        verify_remote_receipt(intent.output_path)
    root["status"] = "aborted"
    root["bindings"]["suite"]["path"] = "../suite.json"
    root_path.write_bytes(canonical_json_bytes(root))
    with pytest.raises(SchemaError, match="plain filename"):
        verify_remote_receipt(intent.output_path)


def test_unresolved_receipt_has_no_result_or_false_cancellation_claim(tmp_path: Path) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawn_acknowledgement_lost", detail="transport lost")
        records.journal.append("unresolved", detail="transport lost")
        terminal_journal = records.journal.bytes()
        assert records.journal_snapshot is None
        write_remote_receipt(
            records,
            status="unresolved",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=None,
            client_spawn_count=1,
        )
        assert records.journal_snapshot == terminal_journal
    finally:
        records.close()
    verified = verify_remote_receipt(intent.output_path)
    assert verified.receipt.result is None
    assert [record.state for record in verified.journal] == [
        "intent",
        "spawn_acknowledgement_lost",
        "unresolved",
    ]


def test_parent_substitution_cannot_publish_into_forged_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(parent)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    records = create_remote_records(intent.output_path, intent, request_digest)
    moved = tmp_path / "moved-parent"
    os.rename(parent, moved)
    parent.mkdir()
    try:
        with pytest.raises(ArtifactError, match="parent identity changed"):
            records.assert_parent_identity()
        assert not (parent / "receipt").exists()
        assert (moved / "receipt.remote-intent.json").read_bytes() == intent.to_bytes()
        records.journal.append("spawn_acknowledgement_lost", detail="parent changed")
        records.journal.append("unresolved", detail="parent changed")
        with pytest.raises(ArtifactError, match="parent identity changed"):
            write_remote_receipt(
                records,
                status="unresolved",
                request_digest=request_digest,
                suite_bytes=suite_bytes,
                plugin_bytes=plugin_bytes,
                manifest_bytes=manifest_bytes,
                result_payload=None,
                client_spawn_count=1,
            )
    finally:
        records.close()


def test_receipt_staging_uses_fresh_nonempty_inventory_repeatedly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    result_payload = _envelope(
        intent, _aborted_result(intent, suite_bytes), request_digest
    ).to_json()
    original_verify = remote._verify_receipt_fd
    inventories: list[set[str]] = []

    def verify_twice(directory_fd: int, root_path: Path) -> remote.VerifiedRemoteReceipt:
        inventories.append(set(os.listdir(f"/proc/self/fd/{directory_fd}")))
        first = original_verify(directory_fd, root_path)
        inventories.append(set(os.listdir(f"/proc/self/fd/{directory_fd}")))
        second = original_verify(directory_fd, root_path)
        assert first == second
        return first

    monkeypatch.setattr(remote, "_verify_receipt_fd", verify_twice)
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawned", call_id="fc-inventory")
        records.journal.append("retrieval_started", call_id="fc-inventory")
        records.journal.append("aborted", call_id="fc-inventory")
        write_remote_receipt(
            records,
            status="aborted",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=result_payload,
            client_spawn_count=1,
        )
    finally:
        records.close()

    assert len(inventories) == 2
    assert inventories[0] == inventories[1]
    assert inventories[0] == {
        "intent.json",
        "journal.jsonl",
        "plugin.json",
        "receipt.json",
        "result.json",
        "suite.json",
        "wheel.manifest.json",
    }


def test_open_existing_records_strictly_restores_completed_journal_read_only(
    tmp_path: Path,
) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    created = create_remote_records(intent.output_path, intent, request_digest)
    created.journal.append("spawned", call_id="fc-completed")
    created.journal.append("retrieval_started", call_id="fc-completed")
    created.journal.append("completed", call_id="fc-completed")
    journal_before = created.journal.bytes()
    created.close()

    opened = open_remote_records(intent.output_path)
    try:
        assert opened.intent == intent
        assert opened.intent_bytes() == intent.to_bytes()
        assert opened.journal.bytes() == journal_before
        assert opened.intent_snapshot == intent.to_bytes()
        assert opened.journal_snapshot == journal_before
        assert opened.journal.request_digest == request_digest
        assert opened.journal.state == "completed"
        assert opened.journal.call_id == "fc-completed"
        with pytest.raises(RuntimeError, match="read-only"):
            opened.journal.append("failed", call_id="fc-completed")
        opened.assert_parent_identity()
    finally:
        opened.close()
    _, journal_path = remote_artifact_paths(intent.output_path)
    with RemoteJournal.open_existing(journal_path, request_digest) as journal:
        assert journal.state == "completed"
        assert journal.call_id == "fc-completed"
        assert journal.bytes() == journal_before
    with pytest.raises(SchemaError, match="request binding"):
        RemoteJournal.open_existing(journal_path, "f" * 64)
    assert not Path(intent.output_path).exists()


def test_open_existing_records_rejects_noncanonical_or_illegal_journal(
    tmp_path: Path,
) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)

    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    created = create_remote_records(intent.output_path, intent, request_digest)
    created.journal.append("spawned", call_id="fc-invalid")
    created.close()
    _, journal_path = remote_artifact_paths(intent.output_path)
    rows = [json.loads(line) for line in journal_path.read_bytes().splitlines()]
    rows[1]["sequence"] = 7
    journal_path.write_bytes(b"".join(canonical_json_line_bytes(row) for row in rows))

    with pytest.raises(SchemaError, match="sequence or request binding"):
        open_remote_records(intent.output_path)


def test_frozen_execution_registry_selects_legacy_and_native_modal_apis() -> None:
    assert tuple(FUSION_EXECUTION_REGISTRY) == (
        "407487a6aa7dc157dcd4aa7bcab698168813bf0a79916d70d91163dc384fe8a8",
        "a318a59bca434b97d073e0ae76f827814213c0a68b0c4263b19c81f98be8f9ee",
        "23f7397f2adee93cd9f7919aaf075c0f8b5e92cd6d4257ce4c54197d3c98035f",
    )
    assert fusion_execution_spec(verify_suite(SUITE).sha256).modal_executor_api.endswith("/1")
    native = fusion_execution_spec(verify_suite(NATIVE_SUITE).sha256)
    assert native.modal_executor_api == "heliostune.modal_fusion_executor/2"
    assert (native.suite_id, native.plugin_id) == (
        "residual-rmsnorm-triton",
        "fusion-triton-rmsnorm-plugin",
    )
    with pytest.raises(SchemaError, match="unsupported suite"):
        fusion_execution_spec("0" * 64)
    with pytest.raises(SchemaError, match="64-character lowercase"):
        fusion_execution_spec("A" * 64)
    with pytest.raises(TypeError):
        FUSION_EXECUTION_REGISTRY[native.suite_sha256] = native  # type: ignore[index]


def test_four_distinct_native_compiler_errors_fit_and_round_trip_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import heliostune.fusion_kernels as fusion
    import heliostune.local_executor as legacy
    import heliostune.native_fusion_executor as native

    intent, suite_bytes, _, _ = _native_intent(tmp_path)
    capability = CapabilityProbe(
        True,
        (),
        "2.8.0",
        "12.8",
        None,
        0,
        _hardware().device_name,
        (9, 0),
        True,
        True,
        True,
        None,
    )
    registry = SimpleNamespace(lookup_backend=lambda _name: lambda graph, _inputs: graph)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda _device: None),
        _dynamo=SimpleNamespace(
            config=SimpleNamespace(disable=False, suppress_errors=False),
            backends=SimpleNamespace(registry=registry),
        ),
    )
    monkeypatch.setattr(native, "_probe_capability", lambda: (capability, fake_torch, "3.4.0"))
    monkeypatch.setattr(legacy, "_precision_flags", lambda _torch: nullcontext())
    monkeypatch.setattr(legacy, "_cuda_autocast_disabled", lambda _torch: nullcontext())

    def materialize(
        _torch: object,
        _suite: object,
        _case: object,
        arm: str,
        digest: str,
        _device: int,
    ) -> tuple[dict[str, object], TensorMaterialization]:
        descriptors = tuple(
            {
                "tensor_id": tensor_id,
                "role": role,
                "shape": shape,
                "draw": "normal_0_1_fp32_cpu",
                "normal_scale": scale,
                "normal_offset": offset,
                "cpu_dtype": "float32",
                "storage_dtype": "bfloat16",
                "device": "cuda:0",
                "contiguous": True,
                "alignment_bytes": 16,
                "alignment_satisfied": True,
                "storage_sha256": storage_digest,
            }
            for tensor_id, role, shape, scale, offset, storage_digest in (
                ("input", "input", [128, 4096], 1.0, 0.0, "1" * 64),
                ("residual", "input", [128, 4096], 1.0, 0.0, "2" * 64),
                ("gamma", "parameter", [4096], 0.02, 1.0, "3" * 64),
            )
        )
        return (
            {"input": object(), "residual": object(), "gamma": object()},
            TensorMaterialization(
                digest,
                "rmsnorm-case-001",
                arm,
                17,
                ("input", "residual", "gamma"),
                descriptors,
            ),
        )

    monkeypatch.setattr(legacy, "_materialize_arm", materialize)
    monkeypatch.setattr(legacy, "_residual_rmsnorm", lambda *_args: object())
    errors = {
        native._ENTRYPOINT[arm]: "".join(
            hashlib.sha256(f"{arm}:{index}".encode()).hexdigest() for index in range(24)
        )
        for arm in native._NATIVE
    }

    def fail_compile(entrypoint: str, *_args: object) -> object:
        raise RuntimeError(errors[entrypoint])

    monkeypatch.setattr(fusion, "compile_residual_rmsnorm", fail_compile)
    result = native.run_native_fusion_suite(NATIVE_SUITE)
    compiler_errors = [
        str(result.compile_evidence[f"{arm}-correctness"]["error"]) for arm in native._NATIVE
    ]
    assert len(set(compiler_errors)) == 4
    assert all(
        error.startswith("RuntimeError: ")
        and len(error.encode("utf-8")) <= 384
        and "...[truncated sha256=" in error
        for error in compiler_errors
    )
    assert result.executor_sources["package_source_sha256"] == intent.source_sha256
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    envelope = RemoteResultEnvelope(
        request_digest=request_digest,
        suite_path=intent.suite_path,
        suite_sha256=intent.suite_sha256,
        plugin_path=intent.plugin_path,
        plugin_sha256=intent.plugin_sha256,
        wheel_filename=intent.wheel_filename,
        wheel_sha256=intent.wheel_sha256,
        manifest_sha256=intent.manifest_sha256,
        head_commit=intent.head_commit,
        source_sha256=intent.source_sha256,
        gpu=intent.gpu,
        gpu_selector=intent.gpu_selector,
        hardware=_hardware(),
        environment=result.environment,
        result=result.to_dict(),
    )

    transport = envelope.to_transport_json()
    assert len(transport.encode("utf-8")) < REMOTE_RESULT_TRANSPORT_MAX_BYTES
    assert RemoteResultEnvelope.from_transport_json(transport) == envelope
    _, parsed = validate_remote_result(
        transport,
        intent=intent,
        request_digest=request_digest,
        verified_suite_bytes=suite_bytes,
    )
    assert parsed.to_dict() == result.to_dict()


def test_native_aborted_result_and_receipt_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import heliostune.native_fusion_executor as native

    intent, suite_bytes, plugin_bytes, manifest_bytes = _native_intent(tmp_path)
    capability = CapabilityProbe(
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
        "strict CPU fake",
    )
    monkeypatch.setattr(native, "_probe_capability", lambda: (capability, None, None))
    result = native.run_native_fusion_suite(NATIVE_SUITE)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    envelope = RemoteResultEnvelope(
        request_digest=request_digest,
        suite_path=intent.suite_path,
        suite_sha256=intent.suite_sha256,
        plugin_path=intent.plugin_path,
        plugin_sha256=intent.plugin_sha256,
        wheel_filename=intent.wheel_filename,
        wheel_sha256=intent.wheel_sha256,
        manifest_sha256=intent.manifest_sha256,
        head_commit=intent.head_commit,
        source_sha256=intent.source_sha256,
        gpu=intent.gpu,
        gpu_selector=intent.gpu_selector,
        hardware=_hardware(),
        environment=result.environment,
        result=result.to_dict(),
    )
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawned", call_id="fc-native-abort")
        records.journal.append("retrieval_started", call_id="fc-native-abort")
        records.journal.append("aborted", call_id="fc-native-abort")
        write_remote_receipt(
            records,
            status="aborted",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=envelope.to_json(),
            client_spawn_count=1,
        )
    finally:
        records.close()

    verified = verify_remote_receipt(intent.output_path)
    assert isinstance(verified.result, native.NativeFusionExecutionResult)
    assert verified.result.to_dict() == result.to_dict()
    assert verified.receipt.bindings["executor_api"] == "heliostune.modal_fusion_executor/2"
    _, parsed = validate_remote_result(
        envelope.to_transport_json(),
        intent=intent,
        request_digest=request_digest,
        verified_suite_bytes=suite_bytes,
    )
    assert isinstance(parsed, native.NativeFusionExecutionResult)

    wrong_schema = envelope.to_dict()
    assert isinstance(wrong_schema["result"], dict)
    wrong_schema["result"]["schema"] = "heliostune.local_executor/1"
    with pytest.raises(SchemaError, match="native fusion result schema"):
        validate_remote_result(
            RemoteResultEnvelope.from_json(
                canonical_json_bytes(wrong_schema).decode()
            ).to_transport_json(),
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_bytes,
        )
    wrong_source = envelope.to_dict()
    assert isinstance(wrong_source["result"], dict)
    assert isinstance(wrong_source["result"]["executor_sources"], dict)
    wrong_source["result"]["executor_sources"]["package_source_sha256"] = "0" * 64
    with pytest.raises(SchemaError, match="package source digest"):
        validate_remote_result(
            RemoteResultEnvelope.from_json(
                canonical_json_bytes(wrong_source).decode()
            ).to_transport_json(),
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_bytes,
        )


def test_unresolved_native_receipt_binds_modal_v2_and_rejects_wrong_api(tmp_path: Path) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _native_intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawn_acknowledgement_lost", detail="transport lost")
        records.journal.append("unresolved", detail="transport lost")
        write_remote_receipt(
            records,
            status="unresolved",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=None,
            client_spawn_count=1,
        )
    finally:
        records.close()

    root_path = Path(intent.output_path) / "receipt.json"
    root = json.loads(root_path.read_text())
    assert root["bindings"]["executor_api"] == "heliostune.modal_fusion_executor/2"
    assert verify_remote_receipt(intent.output_path).receipt.status == "unresolved"
    root["bindings"]["executor_api"] = "heliostune.modal_fusion_executor/1"
    root_path.write_bytes(canonical_json_bytes(root))
    with pytest.raises(SchemaError, match="exact intent policy"):
        verify_remote_receipt(intent.output_path)


def test_remote_execution_import_does_not_load_gpu_modules() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import heliostune.remote_execution; "
                "assert 'heliostune.native_fusion_executor' not in sys.modules; "
                "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr
