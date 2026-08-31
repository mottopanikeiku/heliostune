"""Publish closed exploratory bundles for native Triton fusion execution."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import strict_json_dumps
from .errors import ArtifactError, SchemaError
from .local_bundle import _publish_staged_bundle
from .methodology import EvidenceBundleV1, ProtocolV1, VerifiedBundle
from .native_fusion_executor import (
    NativeFusionExecutionResult,
    _bound_executor_sources,
    _validate_frozen_suite,
)
from .scope import Suite, VerifiedSuite, verify_plugin, verify_suite

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
_NATIVE_EXTRA_ROLES = (
    "suite",
    "terminal_cells",
    "observations",
    "capability_probe",
    "tensor_materialization",
    "execution_summary",
    "stage_outcomes",
    "compile_evidence",
    "resource_evidence",
    "validation_evidence",
    "profile_evidence",
    "executor_sources",
)
_ROLE_PATHS = {role: f"{role}.json" for role in (*_PROTOCOL_ROLES, *_NATIVE_EXTRA_ROLES)}
_ROLE_PATHS["observations"] = "observations.jsonl"
_PROTOCOL_PATH = "protocol.json"
_ATTEMPTS_PATH = "attempts.jsonl"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return strict_json_dumps(value).encode("utf-8")


def _canonical_jsonl(rows: Sequence[object]) -> bytes:
    return "".join(strict_json_dumps(row, compact=True) + "\n" for row in rows).encode("utf-8")


def _strict_result(
    result: NativeFusionExecutionResult,
) -> tuple[NativeFusionExecutionResult, VerifiedSuite]:
    if not isinstance(result, NativeFusionExecutionResult):
        raise SchemaError("native bundle requires a NativeFusionExecutionResult")
    parsed = NativeFusionExecutionResult.from_dict(
        result.to_dict(),
        verified_suite_path=result.verified_suite_path,
        verified_suite_sha256=result.verified_suite_sha256,
        verified_suite_bytes=result.verified_suite_bytes,
    )
    selected = verify_suite(parsed.verified_suite_path)
    if (
        selected.sha256 != parsed.verified_suite_sha256
        or selected.bytes != parsed.verified_suite_bytes
        or selected.suite.suite_id != parsed.suite_id
    ):
        raise ArtifactError(
            "native execution suite custody does not match the current exact suite source"
        )
    return parsed, selected


def _bind_plugin(plugin_path: str | Path, selected: VerifiedSuite) -> tuple[bytes, str, int]:
    plugin = verify_plugin(plugin_path)
    matches = tuple(
        item
        for item in plugin.suites
        if item.sha256 == selected.sha256
        and item.suite.suite_id == selected.suite.suite_id
        and item.suite.revision == selected.suite.revision
    )
    if len(matches) != 1:
        raise ArtifactError("selected native suite is not exactly one digest-bound plugin suite")
    if (selected.suite.plugin_id, selected.suite.plugin_version) != (
        plugin.plugin.plugin_id,
        plugin.plugin.version,
    ):
        raise ArtifactError("selected native suite does not belong to the supplied plugin")
    return plugin.bytes, plugin.plugin.plugin_id, plugin.plugin.version


def _protocol_roles(suite: Suite) -> dict[str, object]:
    return {
        "workloads": {
            "suite_id": suite.suite_id,
            "revision": suite.revision,
            "template_id": suite.template_id,
            "domain": suite.domain,
            "tensors": [tensor.to_dict() for tensor in suite.tensors],
            "cases": [case.to_dict() for case in suite.cases],
        },
        "candidates": [arm.to_dict() for arm in suite.arms if arm.role == "candidate"],
        "comparators": [arm.to_dict() for arm in suite.arms if arm.role != "candidate"],
        "splits": {
            "policy": "single_frozen_suite_no_data_split",
            "case_ids": [case.id for case in suite.cases],
        },
        "numerics": {
            "numeric_contracts": [item.to_dict() for item in suite.numeric_contracts],
            "correctness_policies": [item.to_dict() for item in suite.correctness_policies],
        },
        "timing": {
            "timing_policies": [item.to_dict() for item in suite.timing_policies],
            "executor_rule": suite.executor_rule,
        },
        "analyzer": {
            "claims": [],
            "policy": "no_analysis_for_exploratory_native_fusion_execution",
        },
        "expected_cells": [cell.id for cell in suite.expected_cells],
        "environment_predicate": {
            "accelerator": "cuda",
            "gpu": "H100",
            "architecture": "sm90",
            "compute_capability": [9, 0],
            "torch_version": "2.8.0",
            "triton_version": "3.4.0",
            "rocm": False,
            "native_bfloat16": True,
            "inductor_backend": True,
        },
        "failure_policy": {
            "retry_policy": "none",
            "max_physical_attempts": 1,
            "capability_unavailable": "terminalize_all_cells_blocked_without_backend_invocation",
            "compile_failure": "retain_failure_without_eager_fallback",
            "resource_gate": "native_n_spills_must_equal_zero",
            "correctness_gate": "passing_same_case_arm_seed_contract_required",
            "validation_gate": "zeros_cancellation_overflow_all_pass_before_profile_and_timing",
            "profile_gate": "one_expected_cuda_kernel_and_no_unexpected_cuda_kernels",
            "timing_gate": "all_applicable_compile_resource_correctness_validation_profile_gates_pass",
        },
    }


def _preflight_destination(output_dir: str | Path) -> None:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ArtifactError("descriptor-pinned native bundle preflight is unsupported")
    destination = Path(output_dir)
    if destination.name in {"", ".", ".."}:
        raise ArtifactError(f"invalid native bundle output destination: {destination}")
    parent_fd = -1
    try:
        parent_fd = os.open(
            destination.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ArtifactError(f"native bundle output destination must be absent: {destination}")
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactError(
            f"cannot open real native bundle output parent {destination.parent}: {exc}"
        ) from exc
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)


def preflight_native_fusion_bundle(
    suite_path: str | Path,
    *,
    plugin_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Reject invalid native custody and publication paths before paid execution."""

    selected = verify_suite(suite_path)
    _validate_frozen_suite(selected.suite, selected.sha256)
    _bind_plugin(plugin_path, selected)
    _bound_executor_sources()
    _preflight_destination(output_dir)


def _attempt_journal(
    result: NativeFusionExecutionResult,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...], int, int]:
    rows: list[dict[str, str]] = []
    terminal_ids: list[str] = []
    successes = 0
    failures = 0
    attempts = result.attempts
    for index, observation in enumerate(result.observations):
        running, terminal = attempts[2 * index : 2 * index + 2]
        cell_id = observation.cell_id
        if running["cell_id"] != cell_id or terminal["cell_id"] != cell_id:
            raise SchemaError("native attempt journal differs from strict observation order")
        terminal_status = "success" if observation.status == "passed" else "failure"
        if terminal["status"] != terminal_status:
            raise SchemaError("native attempt terminal status differs from its observation")
        rows.extend(
            (
                {"cell_id": cell_id, "status": "pending"},
                {"cell_id": cell_id, "status": "running"},
                {"cell_id": cell_id, "status": terminal_status},
            )
        )
        terminal_ids.append(cell_id)
        if terminal_status == "success":
            successes += 1
        else:
            failures += 1
    return tuple(rows), tuple(terminal_ids), successes, failures


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
            "study_id": f"native-{suite.suite_id}",
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
                "executor_api": "heliostune.native_fusion_executor/2",
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


def _artifact_entry(role: str, payload: bytes) -> dict[str, object]:
    return {
        "role": role,
        "path": _ROLE_PATHS[role],
        "media_type": "application/x-ndjson" if role == "observations" else "application/json",
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def write_native_fusion_bundle(
    result: NativeFusionExecutionResult,
    *,
    plugin_path: str | Path,
    output_dir: str | Path,
) -> VerifiedBundle:
    """Strictly validate and atomically publish one native execution result."""

    parsed, selected = _strict_result(result)
    if _bound_executor_sources() != parsed.executor_sources:
        raise ArtifactError("native executor source inventory changed since execution")
    plugin_bytes, plugin_id, plugin_version = _bind_plugin(plugin_path, selected)
    suite = selected.suite
    expected_ids = tuple(cell.id for cell in suite.expected_cells)
    attempts, terminal_ids, successes, failures = _attempt_journal(parsed)
    if terminal_ids != tuple(item.cell_id for item in parsed.observations):
        raise SchemaError("native terminal cells differ from strict observations")

    execution_summary: dict[str, object] = {
        "outcome": parsed.outcome,
        "environment": dict(parsed.environment),
        "summary": dict(parsed.summary),
        "claims": [],
        "fusion_claim": False,
        "publication_eligible": False,
        "suite_path": str(selected.path),
        "suite_sha256": selected.sha256,
        "suite_bytes": len(selected.bytes),
    }
    role_payloads: dict[str, bytes] = {"plugin": plugin_bytes}
    role_payloads.update(
        {role: _canonical_json(value) for role, value in _protocol_roles(suite).items()}
    )
    role_payloads.update(
        {
            "suite": selected.bytes,
            "terminal_cells": _canonical_json(terminal_ids),
            "observations": _canonical_jsonl([item.to_dict() for item in parsed.observations]),
            "capability_probe": _canonical_json(parsed.capability.to_dict()),
            "tensor_materialization": _canonical_json(
                [item.to_dict() for item in parsed.materialization]
            ),
            "execution_summary": _canonical_json(execution_summary),
            "stage_outcomes": _canonical_json(dict(parsed.stage_outcomes)),
            "compile_evidence": _canonical_json(dict(parsed.compile_evidence)),
            "resource_evidence": _canonical_json(dict(parsed.resource_evidence)),
            "validation_evidence": _canonical_json(dict(parsed.validation_evidence)),
            "profile_evidence": _canonical_json(dict(parsed.profile_evidence)),
            "executor_sources": _canonical_json(parsed.executor_sources),
        }
    )

    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    protocol_payload = _protocol_payload(
        suite=suite,
        plugin_id=plugin_id,
        plugin_version=plugin_version,
        role_payloads=role_payloads,
        created_at=created_at,
    )
    attempts_payload = _canonical_jsonl(attempts)
    artifacts = [
        _artifact_entry(role, role_payloads[role])
        for role in (*_PROTOCOL_ROLES, *_NATIVE_EXTRA_ROLES)
    ]
    logical_attempts = len(terminal_ids)
    root = EvidenceBundleV1.from_dict(
        {
            "schema": "heliostune.bundle/1",
            "bundle_id": f"native-{suite.suite_id}-{selected.sha256[:12]}",
            "created_at": created_at,
            "protocol": {
                "path": _PROTOCOL_PATH,
                "sha256": _sha256(protocol_payload),
                "bytes": len(protocol_payload),
            },
            "lifecycle": {"state": "SEALED", "outcome": parsed.outcome},
            "attempts": {
                "path": _ATTEMPTS_PATH,
                "sha256": _sha256(attempts_payload),
                "hash_chain_head": _sha256(attempts_payload),
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
    payloads = [
        (_ROLE_PATHS[role], role_payloads[role])
        for role in (*_PROTOCOL_ROLES, *_NATIVE_EXTRA_ROLES)
    ]
    payloads.extend(((_PROTOCOL_PATH, protocol_payload), (_ATTEMPTS_PATH, attempts_payload)))
    return _publish_staged_bundle(Path(output_dir), payloads, root_payload)


__all__ = ["preflight_native_fusion_bundle", "write_native_fusion_bundle"]
