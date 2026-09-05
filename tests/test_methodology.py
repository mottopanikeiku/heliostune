from __future__ import annotations

import copy
import hashlib
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from heliostune.artifacts import strict_json_dumps, strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.methodology import (
    CapturedBundleArtifactV1,
    ClaimSpec,
    EvidenceBundleV1,
    ProtocolV1,
    VerificationLimitations,
    VerifiedBundle,
    attempt_chain_descriptor_bytes,
    capture_bundle_artifacts_v1_from_directory_fd,
    encode_attempt_journal,
    load_bundle_v1,
    load_protocol_v1,
    plugin_suite_path,
    plugin_suite_role,
    selected_suite_descriptor_bytes,
    verify_bundle_v1,
    verify_bundle_v1_from_directory_fd,
    verify_protocol_v1,
)

D = "0" * 64
D1 = "1" * 64
D2 = "2" * 64
D3 = "3" * 64


def descriptive_claim(*, claim_id: str = "latency-description") -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "kind": "descriptive",
        "candidate_id": "candidate",
        "comparator_id": "comparator",
        "reference_id": None,
        "estimand_ast_sha256": D,
        "units": "milliseconds",
        "direction": "lower",
        "scope_sha256": D1,
        "population_sha256": D2,
        "delta": None,
        "alpha": None,
        "multiplicity_family": None,
        "stopping": "none",
    }


def inferential_claim(kind: str = "superiority") -> dict[str, object]:
    claim = descriptive_claim(claim_id=f"{kind}-claim")
    claim.update(
        {
            "kind": kind,
            "delta": 0.01,
            "alpha": 0.05,
            "multiplicity_family": "primary",
            "stopping": "fixed_n",
        }
    )
    if kind == "scoped_exhaustive_dominance":
        claim["reference_id"] = "evaluation-oracle"
    return claim


def protocol_dict(
    *, evidence_class: str = "exploratory", claims: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "schema": "heliostune.protocol/1",
        "study_id": "methodology-test",
        "revision": 1,
        "created_at": "2026-08-26T12:34:56.123456+00:00",
        "evidence_class": evidence_class,
        "parent_protocol_sha256": None,
        "plugin": {"id": "test-plugin", "version": "1", "artifact_sha256": D},
        "semantic": {
            "workloads_sha256": D,
            "candidates_sha256": D1,
            "comparators_sha256": D2,
            "splits_sha256": D3,
            "numerics_sha256": "4" * 64,
            "timing_sha256": "5" * 64,
        },
        "analysis": {"analyzer_sha256": "6" * 64, "claims": claims or []},
        "execution": {
            "executor_api": "heliostune.executor/1",
            "expected_cells_sha256": "7" * 64,
            "expected_cell_count": 0 if evidence_class == "exploratory" else 1,
            "environment_predicate_sha256": "8" * 64,
            "failure_policy_sha256": "9" * 64,
            "retry_policy": "none",
            "max_physical_attempts": 1,
            "wall_limit_s": 300,
            "paid_plan_sha256": None,
        },
    }


def bundle_dict(*, state: str = "SEALED") -> dict[str, object]:
    return {
        "schema": "heliostune.bundle/1",
        "bundle_id": "methodology-bundle-test",
        "created_at": "2026-08-26T12:35:00Z",
        "protocol": {"path": "protocol.json", "sha256": D, "bytes": 100},
        "lifecycle": {"state": state, "outcome": "completed"},
        "attempts": {
            "path": "attempts/journal.jsonl",
            "sha256": D1,
            "hash_chain_head": D2,
            "logical": 2,
            "physical": 2,
            "terminal": 2,
            "orphaned": 0,
        },
        "coverage": {
            "expected_cells": 2,
            "terminal_cells": 2,
            "successes": 1,
            "failures": 1,
        },
        "artifacts": [
            {
                "role": "raw-measurements",
                "path": "raw/measurements.jsonl",
                "media_type": "application/x-ndjson",
                "bytes": 123,
                "sha256": D3,
            }
        ],
        "provenance": {
            "attestation": "self_attested_backend",
            "offline_reproduction": "partial",
        },
        "signatures": [
            {
                "scheme": "ed25519",
                "signer": "test-key",
                "subject_sha256": "4" * 64,
                "signature": "deterministic-test-signature",
            }
        ],
    }


def nested_set(document: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    target: dict[str, Any] = document
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def nested_del(document: dict[str, object], path: str) -> None:
    parts = path.split(".")
    target: dict[str, Any] = document
    for part in parts[:-1]:
        target = target[part]
    del target[parts[-1]]


def test_protocol_exact_canonical_roundtrip_and_loader(tmp_path: Path) -> None:
    raw = protocol_dict(claims=[descriptive_claim()])
    protocol = ProtocolV1.from_dict(raw)
    canonical = strict_json_dumps(protocol.to_dict())
    assert (
        strict_json_dumps(ProtocolV1.from_dict(strict_json_loads(canonical)).to_dict()) == canonical
    )

    path = tmp_path / "protocol.json"
    path.write_text(canonical, encoding="utf-8")
    assert load_protocol_v1(path) == protocol


def test_bundle_exact_canonical_roundtrip_and_loader(tmp_path: Path) -> None:
    raw = bundle_dict()
    bundle = EvidenceBundleV1.from_dict(raw)
    canonical = strict_json_dumps(bundle.to_dict())
    assert (
        strict_json_dumps(EvidenceBundleV1.from_dict(strict_json_loads(canonical)).to_dict())
        == canonical
    )

    path = tmp_path / "bundle.json"
    path.write_text(canonical, encoding="utf-8")
    assert load_bundle_v1(path) == bundle


def test_reference_template_is_valid_and_explicitly_non_study() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "methodology-protocol-v1-template.json"
    protocol = load_protocol_v1(path)
    assert path.read_text(encoding="utf-8") == strict_json_dumps(protocol.to_dict())
    assert protocol.evidence_class == "exploratory"
    assert protocol.analysis.claims == ()
    assert protocol.execution.wall_limit_s == 300
    assert protocol.execution.retry_policy == "none"
    assert protocol.execution.max_physical_attempts == 1
    assert protocol.execution.paid_plan_sha256 is None
    assert "template-reference-not-a-study-freeze" in protocol.study_id


@pytest.mark.parametrize(
    ("parser", "factory", "container"),
    [
        (ProtocolV1.from_dict, protocol_dict, "root"),
        (ProtocolV1.from_dict, protocol_dict, "plugin"),
        (ProtocolV1.from_dict, protocol_dict, "semantic"),
        (ProtocolV1.from_dict, protocol_dict, "analysis"),
        (ProtocolV1.from_dict, protocol_dict, "execution"),
        (EvidenceBundleV1.from_dict, bundle_dict, "root"),
        (EvidenceBundleV1.from_dict, bundle_dict, "protocol"),
        (EvidenceBundleV1.from_dict, bundle_dict, "lifecycle"),
        (EvidenceBundleV1.from_dict, bundle_dict, "attempts"),
        (EvidenceBundleV1.from_dict, bundle_dict, "coverage"),
        (EvidenceBundleV1.from_dict, bundle_dict, "provenance"),
    ],
)
def test_unknown_fields_are_rejected(parser: Any, factory: Any, container: str) -> None:
    raw = factory()
    target: dict[str, Any] = raw if container == "root" else raw[container]
    target["unknown"] = "forbidden"
    with pytest.raises(SchemaError, match="unknown fields"):
        parser(raw)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("study_id", None),
        ("revision", "1"),
        ("plugin", []),
        ("plugin.id", 1),
        ("semantic.workloads_sha256", None),
        ("analysis.claims", {}),
        ("execution.expected_cell_count", True),
        ("execution.wall_limit_s", 0),
        ("execution.paid_plan_sha256", False),
    ],
)
def test_protocol_exact_types_and_ranges(path: str, replacement: object) -> None:
    raw = protocol_dict()
    nested_set(raw, path, replacement)
    with pytest.raises(SchemaError):
        ProtocolV1.from_dict(raw)


@pytest.mark.parametrize(
    "path",
    [
        "schema",
        "study_id",
        "revision",
        "created_at",
        "evidence_class",
        "parent_protocol_sha256",
        "plugin.artifact_sha256",
        "semantic.splits_sha256",
        "analysis.claims",
        "execution.retry_policy",
    ],
)
def test_missing_protocol_fields_are_rejected(path: str) -> None:
    raw = protocol_dict()
    nested_del(raw, path)
    with pytest.raises(SchemaError, match="missing fields"):
        ProtocolV1.from_dict(raw)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("bundle_id", 1),
        ("protocol.bytes", True),
        ("attempts.logical", True),
        ("coverage.expected_cells", -1),
        ("artifacts", {}),
        ("provenance", []),
        ("signatures", {}),
    ],
)
def test_bundle_exact_types_and_ranges(path: str, replacement: object) -> None:
    raw = bundle_dict()
    nested_set(raw, path, replacement)
    with pytest.raises(SchemaError):
        EvidenceBundleV1.from_dict(raw)


@pytest.mark.parametrize(
    "path",
    [
        "schema",
        "bundle_id",
        "created_at",
        "protocol.path",
        "lifecycle.outcome",
        "attempts.hash_chain_head",
        "coverage.failures",
        "artifacts",
        "provenance.attestation",
        "signatures",
    ],
)
def test_missing_bundle_fields_are_rejected(path: str) -> None:
    raw = bundle_dict()
    nested_del(raw, path)
    with pytest.raises(SchemaError, match="missing fields"):
        EvidenceBundleV1.from_dict(raw)


def test_claim_unknown_missing_and_bool_number_are_rejected() -> None:
    claim = descriptive_claim()
    claim["extra"] = 1
    with pytest.raises(SchemaError, match="unknown fields"):
        ClaimSpec.from_dict(claim)

    claim = inferential_claim()
    del claim["scope_sha256"]
    with pytest.raises(SchemaError, match="missing fields"):
        ClaimSpec.from_dict(claim)

    claim = inferential_claim()
    claim["alpha"] = True
    with pytest.raises(SchemaError, match="must be a number"):
        ClaimSpec.from_dict(claim)


@pytest.mark.parametrize(
    "path",
    [
        "parent_protocol_sha256",
        "plugin.artifact_sha256",
        "semantic.workloads_sha256",
        "semantic.candidates_sha256",
        "semantic.comparators_sha256",
        "semantic.splits_sha256",
        "semantic.numerics_sha256",
        "semantic.timing_sha256",
        "analysis.analyzer_sha256",
        "execution.expected_cells_sha256",
        "execution.environment_predicate_sha256",
        "execution.failure_policy_sha256",
        "execution.paid_plan_sha256",
    ],
)
def test_every_protocol_digest_field_is_strict_lowercase_hex(path: str) -> None:
    raw = protocol_dict()
    nested_set(raw, path, "A" * 64)
    with pytest.raises(SchemaError, match="lowercase 64-hex"):
        ProtocolV1.from_dict(raw)


@pytest.mark.parametrize(
    "path",
    [
        "protocol.sha256",
        "attempts.sha256",
        "attempts.hash_chain_head",
        "artifacts.0.sha256",
        "signatures.0.subject_sha256",
    ],
)
def test_every_bundle_digest_field_is_strict_lowercase_hex(path: str) -> None:
    raw = bundle_dict()
    if ".0." in path:
        first, _, last = path.partition(".0.")
        cast_list = raw[first]
        assert isinstance(cast_list, list)
        cast_list[0][last] = "g" * 64
    else:
        nested_set(raw, path, "g" * 64)
    with pytest.raises(SchemaError, match="lowercase 64-hex"):
        EvidenceBundleV1.from_dict(raw)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-26 12:00:00Z",
        "2026-02-30T12:00:00Z",
        "2026-08-26T12:00:00",
        "2026-08-26T25:00:00Z",
        " 2026-08-26T12:00:00Z",
    ],
)
def test_rfc3339_timestamps_are_enforced(timestamp: str) -> None:
    protocol = protocol_dict()
    protocol["created_at"] = timestamp
    with pytest.raises(SchemaError, match="RFC3339|nonblank"):
        ProtocolV1.from_dict(protocol)

    bundle = bundle_dict()
    bundle["created_at"] = timestamp
    with pytest.raises(SchemaError, match="RFC3339|nonblank"):
        EvidenceBundleV1.from_dict(bundle)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("schema", "heliostune.protocol/0"),
        ("evidence_class", "final"),
        ("execution.retry_policy", "retry"),
    ],
)
def test_protocol_enums_are_closed(path: str, value: str) -> None:
    raw = protocol_dict()
    nested_set(raw, path, value)
    with pytest.raises(SchemaError):
        ProtocolV1.from_dict(raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("schema", "heliostune.evidence-bundle/1.0.0"),
        ("lifecycle.state", "ARCHIVED"),
        ("lifecycle.outcome", "in_progress"),
        ("provenance.attestation", "operator_signed"),
        ("provenance.offline_reproduction", "failed"),
    ],
)
def test_bundle_enums_are_closed(path: str, value: str) -> None:
    raw = bundle_dict()
    nested_set(raw, path, value)
    with pytest.raises(SchemaError):
        EvidenceBundleV1.from_dict(raw)


def test_preterminal_lifecycle_requires_pending_and_accepts_it() -> None:
    pending = bundle_dict(state="FROZEN")
    lifecycle = pending["lifecycle"]
    assert isinstance(lifecycle, dict)
    lifecycle["outcome"] = "pending"
    assert EvidenceBundleV1.from_dict(pending).lifecycle.outcome == "pending"

    premature = bundle_dict(state="FROZEN")
    with pytest.raises(SchemaError, match="requires outcome 'pending'"):
        EvidenceBundleV1.from_dict(premature)

    sealed_pending = bundle_dict(state="SEALED")
    sealed_lifecycle = sealed_pending["lifecycle"]
    assert isinstance(sealed_lifecycle, dict)
    sealed_lifecycle["outcome"] = "pending"
    with pytest.raises(SchemaError, match="requires a terminal outcome"):
        EvidenceBundleV1.from_dict(sealed_pending)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("kind", "difference"),
        ("direction", "ascending"),
        ("stopping", "optional"),
    ],
)
def test_claim_enums_are_closed(path: str, value: str) -> None:
    claim = descriptive_claim()
    claim[path] = value
    with pytest.raises(SchemaError):
        ClaimSpec.from_dict(claim)


@pytest.mark.parametrize(
    ("retry_policy", "attempts"),
    [("none", 2), ("pre_measurement_infrastructure", 1)],
)
def test_retry_policy_and_max_attempts_are_consistent(retry_policy: str, attempts: int) -> None:
    raw = protocol_dict()
    execution = raw["execution"]
    assert isinstance(execution, dict)
    execution["retry_policy"] = retry_policy
    execution["max_physical_attempts"] = attempts
    with pytest.raises(SchemaError, match="max_physical_attempts|retry"):
        ProtocolV1.from_dict(raw)


def test_bounded_premeasurement_retry_is_valid() -> None:
    raw = protocol_dict()
    execution = raw["execution"]
    assert isinstance(execution, dict)
    execution["retry_policy"] = "pre_measurement_infrastructure"
    execution["max_physical_attempts"] = 2
    assert ProtocolV1.from_dict(raw).execution.max_physical_attempts == 2


@pytest.mark.parametrize("evidence_class", ["exploratory", "engineering_gate"])
def test_nonconfirmatory_tiers_reject_inferential_claims(evidence_class: str) -> None:
    raw = protocol_dict(evidence_class=evidence_class, claims=[inferential_claim()])
    with pytest.raises(SchemaError, match="cannot contain inferential"):
        ProtocolV1.from_dict(raw)


def test_engineering_and_confirmatory_protocols_require_cells() -> None:
    for evidence_class in ("engineering_gate", "confirmatory"):
        raw = protocol_dict(evidence_class=evidence_class)
        execution = raw["execution"]
        assert isinstance(execution, dict)
        execution["expected_cell_count"] = 0
        with pytest.raises(SchemaError, match="at least one expected cell"):
            ProtocolV1.from_dict(raw)


@pytest.mark.parametrize("field", ["delta", "alpha", "multiplicity_family"])
def test_confirmatory_inferential_claim_requires_frozen_analysis_fields(field: str) -> None:
    claim = inferential_claim()
    claim[field] = None
    raw = protocol_dict(evidence_class="confirmatory", claims=[claim])
    with pytest.raises(SchemaError, match=field):
        ProtocolV1.from_dict(raw)


@pytest.mark.parametrize("kind", ["noninferiority", "equivalence"])
def test_margin_claims_require_positive_delta(kind: str) -> None:
    claim = inferential_claim(kind)
    claim["delta"] = 0.0
    with pytest.raises(SchemaError, match="positive delta"):
        ProtocolV1.from_dict(protocol_dict(evidence_class="confirmatory", claims=[claim]))


def test_confirmatory_inference_requires_nonoptional_stopping() -> None:
    claim = inferential_claim()
    claim["stopping"] = "none"
    with pytest.raises(SchemaError, match="stopping"):
        ProtocolV1.from_dict(protocol_dict(evidence_class="confirmatory", claims=[claim]))


def test_scoped_dominance_requires_reference_and_claim_ids_are_unique() -> None:
    claim = inferential_claim("scoped_exhaustive_dominance")
    claim["reference_id"] = None
    with pytest.raises(SchemaError, match="reference_id"):
        ProtocolV1.from_dict(protocol_dict(evidence_class="confirmatory", claims=[claim]))

    first = descriptive_claim()
    second = descriptive_claim()
    with pytest.raises(SchemaError, match="claim_id values must be unique"):
        ProtocolV1.from_dict(protocol_dict(claims=[first, second]))


@pytest.mark.parametrize(
    "path",
    [
        "../protocol.json",
        "/protocol.json",
        "./protocol.json",
        "protocol//root.json",
        "protocol/../root.json",
        "protocol\\root.json",
        "C:/protocol.json",
        "protocol.json/",
    ],
)
def test_bundle_paths_are_normalized_relative_and_nonescaping(path: str) -> None:
    raw = bundle_dict()
    protocol = raw["protocol"]
    assert isinstance(protocol, dict)
    protocol["path"] = path
    with pytest.raises(SchemaError, match="path|normalized"):
        EvidenceBundleV1.from_dict(raw)


def test_artifact_paths_are_unique_in_the_closed_root() -> None:
    raw = bundle_dict()
    artifacts = raw["artifacts"]
    assert isinstance(artifacts, list)
    duplicate = copy.deepcopy(artifacts[0])
    duplicate["role"] = "other"
    artifacts.append(duplicate)
    with pytest.raises(SchemaError, match="paths must be unique"):
        EvidenceBundleV1.from_dict(raw)

    raw = bundle_dict()
    artifacts = raw["artifacts"]
    assert isinstance(artifacts, list)
    artifacts[0]["path"] = "protocol.json"
    with pytest.raises(SchemaError, match="paths must be unique"):
        EvidenceBundleV1.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physical", 1),
        ("terminal", 3),
        ("orphaned", 1),
    ],
)
def test_attempt_summary_inconsistencies_are_rejected(field: str, value: int) -> None:
    raw = bundle_dict()
    attempts = raw["attempts"]
    assert isinstance(attempts, dict)
    attempts[field] = value
    with pytest.raises(SchemaError, match="attempt"):
        EvidenceBundleV1.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terminal_cells", 3),
        ("successes", 2),
        ("failures", 2),
    ],
)
def test_coverage_inconsistencies_are_rejected(field: str, value: int) -> None:
    raw = bundle_dict()
    coverage = raw["coverage"]
    assert isinstance(coverage, dict)
    coverage[field] = value
    with pytest.raises(SchemaError, match="coverage|terminal cells"):
        EvidenceBundleV1.from_dict(raw)


def test_incomplete_sealed_bundle_remains_representable_as_failure_evidence() -> None:
    raw = bundle_dict()
    attempts = raw["attempts"]
    coverage = raw["coverage"]
    lifecycle = raw["lifecycle"]
    assert isinstance(attempts, dict)
    assert isinstance(coverage, dict)
    assert isinstance(lifecycle, dict)
    attempts["terminal"] = 1
    coverage.update({"terminal_cells": 1, "successes": 0, "failures": 1})
    lifecycle["outcome"] = "failed"
    bundle = EvidenceBundleV1.from_dict(raw)
    assert bundle.lifecycle.state == "SEALED"
    assert bundle.coverage.terminal_cells < bundle.coverage.expected_cells


def test_legacy_protocol_and_bundle_are_not_accepted_as_v1() -> None:
    with pytest.raises(SchemaError):
        ProtocolV1.from_dict({"schema_version": 2, "protocol_role": "final"})
    with pytest.raises(SchemaError):
        EvidenceBundleV1.from_dict({"schema": "heliostune.evidence-bundle/legacy_unverified"})


def _file_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_closed_bundle(
    directory: Path,
    *,
    evidence_class: str = "exploratory",
    state: str = "SEALED",
    outcome: str | None = None,
    expected_cells: int = 2,
    terminal_cells: int | None = None,
    raw_rows: int = 1,
    retry_policy: str = "none",
    max_physical_attempts: int = 1,
    physical_attempts: int | None = None,
    orphaned_attempts: int = 0,
) -> Path:
    terminal = expected_cells if terminal_cells is None else terminal_cells
    lifecycle_outcome = (
        "pending"
        if outcome is None and state in {"DRAFT", "RESOLVED", "FROZEN", "DISPATCHED"}
        else (outcome or "completed")
    )
    expected_ids = [f"cell-{index}" for index in range(expected_cells)]
    terminal_ids = expected_ids[:terminal]
    role_payloads = {
        "plugin": b"plugin artifact\n",
        "workloads": b"workloads\n",
        "candidates": b"candidates\n",
        "comparators": b"comparators\n",
        "splits": b"splits\n",
        "numerics": b"numerics\n",
        "timing": b"timing\n",
        "analyzer": b"analyzer\n",
        "expected_cells": strict_json_dumps(expected_ids).encode("utf-8"),
        "terminal_cells": strict_json_dumps(terminal_ids).encode("utf-8"),
        "environment_predicate": b"environment predicate\n",
        "failure_policy": b"failure policy\n",
    }

    protocol_raw = protocol_dict(evidence_class=evidence_class)
    plugin = protocol_raw["plugin"]
    semantic = protocol_raw["semantic"]
    analysis = protocol_raw["analysis"]
    execution = protocol_raw["execution"]
    assert isinstance(plugin, dict)
    assert isinstance(semantic, dict)
    assert isinstance(analysis, dict)
    assert isinstance(execution, dict)
    plugin["artifact_sha256"] = _file_digest(role_payloads["plugin"])
    for role in ("workloads", "candidates", "comparators", "splits", "numerics", "timing"):
        semantic[f"{role}_sha256"] = _file_digest(role_payloads[role])
    analysis["analyzer_sha256"] = _file_digest(role_payloads["analyzer"])
    execution.update(
        {
            "expected_cells_sha256": _file_digest(role_payloads["expected_cells"]),
            "expected_cell_count": expected_cells,
            "environment_predicate_sha256": _file_digest(role_payloads["environment_predicate"]),
            "failure_policy_sha256": _file_digest(role_payloads["failure_policy"]),
            "retry_policy": retry_policy,
            "max_physical_attempts": max_physical_attempts,
        }
    )
    protocol_payload = strict_json_dumps(protocol_raw).encode("utf-8")

    transitions: list[dict[str, str]] = []
    for cell_id in terminal_ids:
        transitions.extend(
            (
                {"cell_id": cell_id, "status": "pending"},
                {"cell_id": cell_id, "status": "success"},
            )
        )
    attempts_payload = "".join(
        strict_json_dumps(record, compact=True) + "\n" for record in transitions
    ).encode("utf-8")
    measurements_payload = b"".join(b'{"measurement":"retained"}\n' for _ in range(raw_rows))
    bundle_raw = bundle_dict(state=state)
    bundle_raw["lifecycle"] = {"state": state, "outcome": lifecycle_outcome}
    bundle_raw["protocol"] = {
        "path": "protocol.json",
        "sha256": _file_digest(protocol_payload),
        "bytes": len(protocol_payload),
    }
    bundle_raw["attempts"] = {
        "path": "attempts/journal.jsonl",
        "sha256": _file_digest(attempts_payload),
        "hash_chain_head": D2,
        "logical": terminal,
        "physical": terminal if physical_attempts is None else physical_attempts,
        "terminal": terminal,
        "orphaned": orphaned_attempts,
    }
    bundle_raw["coverage"] = {
        "expected_cells": expected_cells,
        "terminal_cells": terminal,
        "successes": terminal,
        "failures": 0,
    }
    artifacts: list[dict[str, object]] = []
    for role, payload in role_payloads.items():
        artifacts.append(
            {
                "role": role,
                "path": f"artifacts/{role}.json",
                "media_type": (
                    "application/json"
                    if role in {"expected_cells", "terminal_cells"}
                    else "application/octet-stream"
                ),
                "bytes": len(payload),
                "sha256": _file_digest(payload),
            }
        )
    artifacts.append(
        {
            "role": "raw-measurements",
            "path": "raw/measurements.jsonl",
            "media_type": "application/x-ndjson",
            "bytes": len(measurements_payload),
            "sha256": _file_digest(measurements_payload),
        }
    )
    bundle_raw["artifacts"] = artifacts

    (directory / "attempts").mkdir(parents=True)
    (directory / "artifacts").mkdir()
    (directory / "raw").mkdir()
    (directory / "protocol.json").write_bytes(protocol_payload)
    (directory / "attempts/journal.jsonl").write_bytes(attempts_payload)
    for role, payload in role_payloads.items():
        (directory / f"artifacts/{role}.json").write_bytes(payload)
    (directory / "raw/measurements.jsonl").write_bytes(measurements_payload)
    root = directory / "bundle.json"
    root.write_text(strict_json_dumps(bundle_raw), encoding="utf-8")
    return root


def _artifact_by_role(document: dict[str, Any], role: str) -> dict[str, Any]:
    artifacts = document["artifacts"]
    assert isinstance(artifacts, list)
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("role") == role
    ]
    assert len(matches) == 1
    return matches[0]


def _append_json_artifact(
    root_raw: dict[str, Any],
    directory: Path,
    *,
    role: str,
    path: str,
    payload: bytes,
) -> None:
    target = directory / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    artifacts = root_raw["artifacts"]
    assert isinstance(artifacts, list)
    artifacts.append(
        {
            "role": role,
            "path": path,
            "media_type": "application/json",
            "bytes": len(payload),
            "sha256": _file_digest(payload),
        }
    )


def _enable_attempt_chain(root: Path) -> None:
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    attempts = root_raw["attempts"]
    assert isinstance(attempts, dict)
    journal_path = root.parent / str(attempts["path"])
    transitions: list[dict[str, object]] = []
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        transition = strict_json_loads(line)
        assert isinstance(transition, dict)
        transitions.append(transition)
    payload, head = encode_attempt_journal(transitions)
    journal_path.write_bytes(payload)
    attempts["sha256"] = _file_digest(payload)
    attempts["hash_chain_head"] = head
    _append_json_artifact(
        root_raw,
        root.parent,
        role="attempt_chain",
        path="attempt_chain.json",
        payload=attempt_chain_descriptor_bytes(),
    )
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")


def _enable_plugin_suite_custody(root: Path, *, selected_index: int = 0) -> None:
    repository = Path(__file__).parents[1]
    plugin_payload = (
        repository / "benchmarks/plugins/fusion-reference-plugin-v1.json"
    ).read_bytes()
    suite_payloads = [
        (repository / "benchmarks/suites/gated-mlp-epilogue-v1.json").read_bytes(),
        (repository / "benchmarks/suites/residual-rmsnorm-v1.json").read_bytes(),
    ]
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    plugin_artifact = _artifact_by_role(root_raw, "plugin")
    plugin_path = root.parent / str(plugin_artifact["path"])
    plugin_path.write_bytes(plugin_payload)
    plugin_artifact["bytes"] = len(plugin_payload)
    plugin_artifact["sha256"] = _file_digest(plugin_payload)

    protocol_path = root.parent / "protocol.json"
    protocol_raw = strict_json_loads(protocol_path.read_text(encoding="utf-8"))
    assert isinstance(protocol_raw, dict)
    plugin_binding = protocol_raw["plugin"]
    assert isinstance(plugin_binding, dict)
    plugin_binding.update(
        {
            "id": "fusion-reference-plugin",
            "version": "1",
            "artifact_sha256": _file_digest(plugin_payload),
        }
    )
    protocol_payload = strict_json_dumps(protocol_raw).encode("utf-8")
    protocol_path.write_bytes(protocol_payload)
    root_binding = root_raw["protocol"]
    assert isinstance(root_binding, dict)
    root_binding["bytes"] = len(protocol_payload)
    root_binding["sha256"] = _file_digest(protocol_payload)

    for index, suite_payload in enumerate(suite_payloads):
        _append_json_artifact(
            root_raw,
            root.parent,
            role=plugin_suite_role(index),
            path=plugin_suite_path(index),
            payload=suite_payload,
        )
    _append_json_artifact(
        root_raw,
        root.parent,
        role="selected_suite",
        path="selected_suite.json",
        payload=selected_suite_descriptor_bytes(selected_index),
    )
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")


def _bind_parent_protocol(root: Path, *, evidence_class: str = "exploratory") -> Path:
    parent_payload = strict_json_dumps(protocol_dict(evidence_class=evidence_class)).encode("utf-8")
    parent_path = root.parent / "artifacts/parent_protocol.json"
    parent_path.write_bytes(parent_payload)

    protocol_path = root.parent / "protocol.json"
    protocol_raw = strict_json_loads(protocol_path.read_text(encoding="utf-8"))
    assert isinstance(protocol_raw, dict)
    protocol_raw["parent_protocol_sha256"] = _file_digest(parent_payload)
    protocol_payload = strict_json_dumps(protocol_raw).encode("utf-8")
    protocol_path.write_bytes(protocol_payload)

    bundle_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(bundle_raw, dict)
    binding = bundle_raw["protocol"]
    assert isinstance(binding, dict)
    binding["sha256"] = _file_digest(protocol_payload)
    binding["bytes"] = len(protocol_payload)
    artifacts = bundle_raw["artifacts"]
    assert isinstance(artifacts, list)
    artifacts.append(
        {
            "role": "parent_protocol",
            "path": "artifacts/parent_protocol.json",
            "media_type": "application/json",
            "bytes": len(parent_payload),
            "sha256": _file_digest(parent_payload),
        }
    )
    root.write_text(strict_json_dumps(bundle_raw), encoding="utf-8")
    return parent_path


@pytest.mark.parametrize("evidence_class", ["exploratory", "engineering_gate", "confirmatory"])
def test_verify_closed_bundle_for_each_evidence_class(tmp_path: Path, evidence_class: str) -> None:
    root = _write_closed_bundle(tmp_path, evidence_class=evidence_class)

    verified = verify_bundle_v1(root)

    assert isinstance(verified, VerifiedBundle)
    assert verified.bundle.lifecycle.state == "SEALED"
    assert verified.protocol.protocol.evidence_class == evidence_class
    assert verified.protocol.path == (tmp_path / "protocol.json").resolve()
    assert verified.protocol.sha256 == verified.bundle.protocol.sha256
    assert verified.protocol.bytes == verified.bundle.protocol.bytes
    assert verified.root_path == root.resolve()
    assert verified.root_bytes == len(root.read_bytes())
    assert verified.root_sha256 == _file_digest(root.read_bytes())
    assert verified.attempts_bytes == len((tmp_path / "attempts/journal.jsonl").read_bytes())
    root_directory = root.parent.stat()
    assert verified.root_directory_identity == (
        root_directory.st_dev,
        root_directory.st_ino,
    )
    root_parent_directory = root.parent.parent.stat()
    assert verified.root_parent_directory_identity == (
        root_parent_directory.st_dev,
        root_parent_directory.st_ino,
    )
    assert verified.referenced_paths[:2] == (
        (tmp_path / "protocol.json").resolve(),
        (tmp_path / "attempts/journal.jsonl").resolve(),
    )
    assert (tmp_path / "raw/measurements.jsonl").resolve() in verified.referenced_paths
    assert len(verified.referenced_paths) == len(verified.bundle.artifacts) + 2


def test_verify_protocol_binds_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    payload = strict_json_dumps(protocol_dict()).encode("utf-8")
    path.write_bytes(payload)

    verified = verify_protocol_v1(path)

    assert verified.protocol == ProtocolV1.from_dict(protocol_dict())
    assert verified.path == path.resolve()
    assert verified.bytes == len(payload)
    assert verified.sha256 == _file_digest(payload)


@pytest.mark.parametrize(
    ("relative_path", "failure"),
    [
        ("protocol.json", "protocol"),
        ("attempts/journal.jsonl", "attempts"),
        ("raw/measurements.jsonl", "artifact"),
    ],
)
def test_verify_bundle_rejects_missing_references(
    tmp_path: Path, relative_path: str, failure: str
) -> None:
    root = _write_closed_bundle(tmp_path)
    (tmp_path / relative_path).unlink()

    with pytest.raises(ArtifactError, match=failure):
        verify_bundle_v1(root)


def test_verify_bundle_requires_every_protocol_digest_role(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    artifacts = root_raw["artifacts"]
    assert isinstance(artifacts, list)
    root_raw["artifacts"] = [
        artifact
        for artifact in artifacts
        if not isinstance(artifact, dict) or artifact.get("role") != "workloads"
    ]
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="missing protocol digest role 'workloads'"):
        verify_bundle_v1(root)


def test_verify_bundle_rejects_protocol_digest_role_mismatch(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    replacement = b"different plugin bytes\n"
    (tmp_path / "artifacts/plugin.json").write_bytes(replacement)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    artifact = _artifact_by_role(root_raw, "plugin")
    artifact["bytes"] = len(replacement)
    artifact["sha256"] = _file_digest(replacement)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="does not match its protocol SHA-256"):
        verify_bundle_v1(root)


def test_verify_bundle_rejects_dangling_parent_protocol_reference(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    parent_path = _bind_parent_protocol(root)
    parent_path.unlink()

    with pytest.raises(ArtifactError, match="parent_protocol"):
        verify_bundle_v1(root)


def test_exploratory_parent_nonpromotion_is_not_claimed_checked(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path, evidence_class="confirmatory")
    _bind_parent_protocol(root, evidence_class="exploratory")

    verified = verify_bundle_v1(root)

    assert verified.limitations.protocol_ancestry == "not_checked"
    assert verified.limitations.evidence_nonpromotion == "not_checked"
    assert verified.publication_eligible is False


@pytest.mark.parametrize(
    "terminal_payload",
    [
        b'{"cell_id":"cell-0"}\n',
        b'["cell-0","cell-0"]\n',
        b'[""]\n',
    ],
)
def test_terminal_cell_identity_artifact_is_strict(tmp_path: Path, terminal_payload: bytes) -> None:
    root = _write_closed_bundle(tmp_path)
    (tmp_path / "artifacts/terminal_cells.json").write_bytes(terminal_payload)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    artifact = _artifact_by_role(root_raw, "terminal_cells")
    artifact["bytes"] = len(terminal_payload)
    artifact["sha256"] = _file_digest(terminal_payload)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="terminal_cells"):
        verify_bundle_v1(root)


def test_attempt_journal_requires_strict_cell_status_transitions(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    invalid = b'{"cell_id":"cell-0","status":"success"}\n'
    (tmp_path / "attempts/journal.jsonl").write_bytes(invalid)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    attempts = root_raw["attempts"]
    assert isinstance(attempts, dict)
    attempts["sha256"] = _file_digest(invalid)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="invalid attempt transition"):
        verify_bundle_v1(root)


def test_retry_none_rejects_extra_physical_attempt(tmp_path: Path) -> None:
    root = _write_closed_bundle(
        tmp_path,
        expected_cells=1,
        physical_attempts=2,
    )

    with pytest.raises(ArtifactError, match="physical attempts to equal logical attempts"):
        verify_bundle_v1(root)


def test_pre_measurement_retry_accepts_bounded_physical_attempts(tmp_path: Path) -> None:
    root = _write_closed_bundle(
        tmp_path,
        expected_cells=1,
        retry_policy="pre_measurement_infrastructure",
        max_physical_attempts=2,
        physical_attempts=2,
        orphaned_attempts=1,
    )

    verified = verify_bundle_v1(root)

    assert verified.limitations.attempt_reconciliation == "not_checked"


def test_pre_measurement_retry_rejects_overbound_physical_attempts(tmp_path: Path) -> None:
    root = _write_closed_bundle(
        tmp_path,
        expected_cells=1,
        retry_policy="pre_measurement_infrastructure",
        max_physical_attempts=2,
        physical_attempts=3,
    )

    with pytest.raises(ArtifactError, match="exceed the retry-policy bound"):
        verify_bundle_v1(root)


@pytest.mark.parametrize("mutation", ["tampered", "size", "digest"])
def test_verify_bundle_rejects_artifact_identity_faults(tmp_path: Path, mutation: str) -> None:
    root = _write_closed_bundle(tmp_path)
    raw_path = tmp_path / "raw/measurements.jsonl"
    if mutation == "tampered":
        raw_path.write_bytes(raw_path.read_bytes() + b"x")
    else:
        root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
        assert isinstance(root_raw, dict)
        artifact = _artifact_by_role(root_raw, "raw-measurements")
        if mutation == "size":
            artifact["bytes"] = artifact["bytes"] + 1
        else:
            artifact["sha256"] = D
        root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="byte count|SHA-256"):
        verify_bundle_v1(root)


@pytest.mark.parametrize("mutation", ["tampered", "size", "digest"])
def test_verify_bundle_rejects_protocol_identity_faults(tmp_path: Path, mutation: str) -> None:
    root = _write_closed_bundle(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    if mutation == "tampered":
        protocol_path.write_bytes(protocol_path.read_bytes() + b" ")
    else:
        root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
        assert isinstance(root_raw, dict)
        binding = root_raw["protocol"]
        assert isinstance(binding, dict)
        if mutation == "size":
            binding["bytes"] = binding["bytes"] + 1
        else:
            binding["sha256"] = D
        root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="byte count|SHA-256"):
        verify_bundle_v1(root)


def test_verify_bundle_rejects_attempt_journal_digest_fault(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    (tmp_path / "attempts/journal.jsonl").write_bytes(b"tampered\n")

    with pytest.raises(ArtifactError, match="attempts SHA-256"):
        verify_bundle_v1(root)


def test_verify_bundle_rejects_duplicate_artifact_roles(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    second_payload = (tmp_path / "artifacts/plugin.json").read_bytes()
    (tmp_path / "raw/second.jsonl").write_bytes(second_payload)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    artifacts = root_raw["artifacts"]
    assert isinstance(artifacts, list)
    artifacts.append(
        {
            "role": "plugin",
            "path": "raw/second.jsonl",
            "media_type": "application/octet-stream",
            "bytes": len(second_payload),
            "sha256": _file_digest(second_payload),
        }
    )
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(SchemaError, match="artifact roles must be unique"):
        verify_bundle_v1(root)


def test_verify_bundle_rejects_resolved_path_aliases(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    (tmp_path / "raw/alias.jsonl").symlink_to("measurements.jsonl")
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    first = _artifact_by_role(root_raw, "raw-measurements")
    duplicate = copy.deepcopy(first)
    duplicate["role"] = "alias"
    duplicate["path"] = "raw/alias.jsonl"
    artifacts = root_raw["artifacts"]
    assert isinstance(artifacts, list)
    artifacts.append(duplicate)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="symlinks"):
        verify_bundle_v1(root)


def test_verify_bundle_never_follows_symlink_escape(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    root = _write_closed_bundle(bundle_dir)
    outside = tmp_path / "outside.jsonl"
    payload = b"outside\n"
    outside.write_bytes(payload)
    escaped = bundle_dir / "raw/escaped.jsonl"
    escaped.symlink_to(outside)

    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    artifact = _artifact_by_role(root_raw, "raw-measurements")
    artifact.update(
        {
            "path": "raw/escaped.jsonl",
            "bytes": len(payload),
            "sha256": _file_digest(payload),
        }
    )
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="escapes"):
        verify_bundle_v1(root)


@pytest.mark.parametrize("state", ["DRAFT", "RESOLVED", "FROZEN", "DISPATCHED"])
def test_verify_bundle_requires_closed_lifecycle(tmp_path: Path, state: str) -> None:
    root = _write_closed_bundle(tmp_path, state=state)

    with pytest.raises(ArtifactError, match="not closed"):
        verify_bundle_v1(root)


@pytest.mark.parametrize("evidence_class", ["engineering_gate", "confirmatory"])
def test_strict_evidence_tiers_require_complete_coverage(
    tmp_path: Path, evidence_class: str
) -> None:
    root = _write_closed_bundle(
        tmp_path,
        evidence_class=evidence_class,
        outcome="failed",
        expected_cells=2,
        terminal_cells=1,
    )

    with pytest.raises(ArtifactError, match="close every expected cell"):
        verify_bundle_v1(root)


def test_confirmatory_raw_prefix_cannot_close_two_declared_cells(tmp_path: Path) -> None:
    root = _write_closed_bundle(
        tmp_path,
        evidence_class="confirmatory",
        outcome="failed",
        expected_cells=2,
        terminal_cells=1,
        raw_rows=1,
    )

    with pytest.raises(ArtifactError, match="close every expected cell"):
        verify_bundle_v1(root)


def test_completed_exploratory_bundle_requires_complete_coverage(tmp_path: Path) -> None:
    root = _write_closed_bundle(
        tmp_path,
        evidence_class="exploratory",
        outcome="completed",
        expected_cells=2,
        terminal_cells=1,
    )

    with pytest.raises(ArtifactError, match="completed bundle"):
        verify_bundle_v1(root)


def test_failed_exploratory_prefix_is_closed_and_truthful(tmp_path: Path) -> None:
    root = _write_closed_bundle(
        tmp_path,
        evidence_class="exploratory",
        outcome="failed",
        expected_cells=2,
        terminal_cells=1,
    )

    verified = verify_bundle_v1(root)

    assert verified.bundle.coverage.terminal_cells == 1
    assert verified.bundle.lifecycle.outcome == "failed"


def test_incomplete_exploratory_journal_accepts_the_next_running_prefix_cell(
    tmp_path: Path,
) -> None:
    root = _write_closed_bundle(
        tmp_path,
        outcome="failed",
        expected_cells=3,
        terminal_cells=1,
        physical_attempts=2,
    )
    running = b'{"cell_id":"cell-1","status":"pending"}\n{"cell_id":"cell-1","status":"running"}\n'
    journal_path = tmp_path / "attempts/journal.jsonl"
    journal_payload = journal_path.read_bytes() + running
    journal_path.write_bytes(journal_payload)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    attempts = root_raw["attempts"]
    assert isinstance(attempts, dict)
    attempts["sha256"] = _file_digest(journal_payload)
    attempts["logical"] = 2
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    verified = verify_bundle_v1(root)

    assert verified.limitations.attempt_reconciliation == "not_checked"


def test_incomplete_exploratory_journal_cannot_skip_a_prefix_cell(tmp_path: Path) -> None:
    root = _write_closed_bundle(
        tmp_path,
        outcome="failed",
        expected_cells=3,
        terminal_cells=1,
        physical_attempts=2,
    )
    skipped = b'{"cell_id":"cell-2","status":"pending"}\n{"cell_id":"cell-2","status":"running"}\n'
    journal_path = tmp_path / "attempts/journal.jsonl"
    journal_payload = journal_path.read_bytes() + skipped
    journal_path.write_bytes(journal_payload)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    attempts = root_raw["attempts"]
    assert isinstance(attempts, dict)
    attempts["sha256"] = _file_digest(journal_payload)
    attempts["logical"] = 2
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="journal.*expected-cell prefix"):
        verify_bundle_v1(root)


def test_bundle_coverage_must_match_protocol_expected_count(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    coverage = root_raw["coverage"]
    assert isinstance(coverage, dict)
    coverage["expected_cells"] = 3
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="parsed expected_cells"):
        verify_bundle_v1(root)


def test_signatures_are_structural_only_and_limitations_are_explicit(
    tmp_path: Path,
) -> None:
    root = _write_closed_bundle(tmp_path)

    verified = verify_bundle_v1(root)

    assert len(verified.bundle.signatures) == 1
    assert verified.limitations.protocol_ancestry == "not_checked"
    assert verified.limitations.evidence_nonpromotion == "not_checked"
    assert verified.limitations.semantic_content_beyond_digests == "not_checked"
    assert verified.limitations.plugin_suite_custody == "not_checked"
    assert verified.limitations.attempt_journal_hash_chain == "not_checked"
    assert VerificationLimitations().attempt_reconciliation == "not_checked"
    assert verified.limitations.attempt_reconciliation == "checked"
    assert verified.limitations.claim_eligibility == "not_checked"
    assert verified.limitations.analyzer_replay == "not_checked"
    assert verified.limitations.provenance_tier_derivation == "not_checked"
    assert verified.limitations.signature_cryptography == "not_checked"
    assert verified.limitations.catalog_membership == "not_checked"
    assert verified.limitations.offline_reproduction == "not_checked"
    assert verified.publication_eligible is False


def test_filesystem_verifiers_reject_legacy_documents(tmp_path: Path) -> None:
    protocol = tmp_path / "legacy-protocol.json"
    protocol.write_text('{"schema_version":2,"protocol_role":"final"}', encoding="utf-8")
    with pytest.raises(SchemaError):
        verify_protocol_v1(protocol)

    root = tmp_path / "legacy-bundle.json"
    root.write_text(
        '{"schema":"heliostune.evidence-bundle/legacy_unverified"}',
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        verify_bundle_v1(root)


@pytest.mark.parametrize("target", ["missing", "directory", "invalid-utf8", "invalid-json"])
def test_verify_protocol_faults_are_closed(tmp_path: Path, target: str) -> None:
    path = tmp_path / "protocol.json"
    if target == "directory":
        path.mkdir()
    elif target == "invalid-utf8":
        path.write_bytes(b"\xff")
    elif target == "invalid-json":
        path.write_text("{", encoding="utf-8")

    expected = SchemaError if target == "invalid-json" else ArtifactError
    with pytest.raises(expected):
        verify_protocol_v1(path)


def test_verify_bundle_rejects_reference_to_directory(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    artifact = tmp_path / "raw/measurements.jsonl"
    artifact.unlink()
    artifact.mkdir()

    with pytest.raises(ArtifactError, match="regular file"):
        verify_bundle_v1(root)


def test_verify_bundle_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    artifact = tmp_path / "raw/measurements.jsonl"
    artifact.unlink()
    os.mkfifo(artifact)

    with pytest.raises(ArtifactError, match="regular file"):
        verify_bundle_v1(root)


@pytest.mark.parametrize("verifier", [verify_protocol_v1, verify_bundle_v1])
def test_verifier_entry_path_cannot_be_symlink_escape(tmp_path: Path, verifier: Any) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    if verifier is verify_protocol_v1:
        target = outside / "protocol.json"
        target.write_text(strict_json_dumps(protocol_dict()), encoding="utf-8")
    else:
        target = _write_closed_bundle(outside)

    inside = tmp_path / "inside"
    inside.mkdir()
    link = inside / target.name
    link.symlink_to(target)

    with pytest.raises(ArtifactError, match="escapes"):
        verifier(link)


def test_attempt_chain_helpers_have_deterministic_empty_one_and_multi_vectors() -> None:
    empty_payload, empty_head = encode_attempt_journal(())
    assert empty_payload == b""
    assert empty_head == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    one_payload, one_head = encode_attempt_journal(({"cell_id": "cell-a", "status": "pending"},))
    assert one_payload == (
        b'{"cell_id":"cell-a","predecessor_sha256":'
        b'"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",'
        b'"status":"pending"}\n'
    )
    assert one_head == "81b40efebe452a227faaa00b49a524cf31372bface6966899e9a4d5a674108e9"

    multi_payload, multi_head = encode_attempt_journal(
        (
            {"cell_id": "cell-a", "status": "pending"},
            {"cell_id": "cell-a", "status": "running"},
            {"cell_id": "cell-a", "status": "success"},
        )
    )
    assert multi_payload.startswith(one_payload)
    assert multi_head == "00ceccaa4e7172518364a816638d6d88f5e41ce6777aa633b05d8b07440c5c6b"
    assert plugin_suite_role(2) == "plugin_suite_2"
    assert plugin_suite_path(2) == "plugin_suite_2.json"
    assert strict_json_loads(attempt_chain_descriptor_bytes().decode("utf-8")) == {
        "schema": "heliostune.attempt-chain/1"
    }
    assert strict_json_loads(selected_suite_descriptor_bytes(1).decode("utf-8")) == {
        "plugin_suite_index": 1,
        "schema": "heliostune.selected-suite/1",
    }


def test_complete_bundle_reports_new_controls_checked_without_publication_eligibility(
    tmp_path: Path,
) -> None:
    root = _write_closed_bundle(tmp_path)
    _enable_plugin_suite_custody(root, selected_index=1)
    _enable_attempt_chain(root)

    verified = verify_bundle_v1(root)

    assert verified.limitations.plugin_suite_custody == "checked"
    assert verified.limitations.attempt_journal_hash_chain == "checked"
    assert verified.limitations.attempt_reconciliation == "checked"
    assert verified.publication_eligible is False


def test_legacy_bundle_remains_accepted_without_promoting_new_controls(tmp_path: Path) -> None:
    verified = verify_bundle_v1(_write_closed_bundle(tmp_path))

    assert verified.limitations.plugin_suite_custody == "not_checked"
    assert verified.limitations.attempt_journal_hash_chain == "not_checked"
    assert verified.limitations.attempt_reconciliation == "checked"


@pytest.mark.parametrize("line_ending", ["missing_final_lf", "crlf"])
def test_legacy_attempt_rows_preserve_historical_line_splitting(
    tmp_path: Path,
    line_ending: str,
) -> None:
    root = _write_closed_bundle(tmp_path)
    journal = tmp_path / "attempts/journal.jsonl"
    payload = journal.read_bytes()
    if line_ending == "missing_final_lf":
        payload = payload.removesuffix(b"\n")
    else:
        payload = payload.replace(b"\n", b"\r\n")
    journal.write_bytes(payload)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    attempts = root_raw["attempts"]
    assert isinstance(attempts, dict)
    attempts["sha256"] = _file_digest(payload)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    verified = verify_bundle_v1(root)

    assert verified.limitations.attempt_journal_hash_chain == "not_checked"
    assert verified.limitations.attempt_reconciliation == "checked"


def test_orphaned_attempt_aggregate_remains_not_checked(tmp_path: Path) -> None:
    root = _write_closed_bundle(
        tmp_path,
        outcome="failed",
        expected_cells=1,
        terminal_cells=0,
        physical_attempts=1,
        orphaned_attempts=1,
    )
    journal = tmp_path / "attempts/journal.jsonl"
    payload = b'{"cell_id":"cell-0","status":"pending"}\n'
    journal.write_bytes(payload)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    attempts = root_raw["attempts"]
    assert isinstance(attempts, dict)
    attempts["sha256"] = _file_digest(payload)
    attempts["logical"] = 1
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    verified = verify_bundle_v1(root)

    assert verified.limitations.attempt_reconciliation == "not_checked"


@pytest.mark.parametrize("mutation", ["malformed_descriptor", "reserved_prefix"])
def test_attempt_chain_opt_in_cannot_downgrade_on_partial_metadata(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _write_closed_bundle(tmp_path)
    _enable_attempt_chain(root)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    artifact = _artifact_by_role(root_raw, "attempt_chain")
    if mutation == "malformed_descriptor":
        payload = b'{"schema":"wrong"}\n'
        (tmp_path / "attempt_chain.json").write_bytes(payload)
        artifact["bytes"] = len(payload)
        artifact["sha256"] = _file_digest(payload)
    else:
        artifact["role"] = "attempt_chain_extra"
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="attempt chain"):
        verify_bundle_v1(root)


@pytest.mark.parametrize("outcome", ["completed", "aborted"])
def test_empty_chained_journal_verifies_against_h0(tmp_path: Path, outcome: str) -> None:
    root = _write_closed_bundle(tmp_path, expected_cells=0, outcome=outcome)
    _enable_attempt_chain(root)

    verified = verify_bundle_v1(root)

    assert verified.bundle.attempts.hash_chain_head == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert verified.limitations.attempt_journal_hash_chain == "checked"
    assert verified.limitations.attempt_reconciliation == "checked"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("predecessor", "predecessor mismatch"),
        ("noncanonical", "not canonical"),
        ("missing_lf", "LF-terminated"),
        ("crlf", "carriage returns"),
        ("mixed", "missing fields"),
        ("reordered", "predecessor mismatch"),
        ("truncated", "final head mismatch"),
        ("wrong_head", "final head mismatch"),
    ],
)
def test_chained_attempt_journal_faults_fail_closed(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = _write_closed_bundle(tmp_path)
    _enable_attempt_chain(root)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    attempts = root_raw["attempts"]
    assert isinstance(attempts, dict)
    journal = tmp_path / "attempts/journal.jsonl"
    rows = journal.read_bytes().splitlines(keepends=True)
    if mutation == "predecessor":
        row = strict_json_loads(rows[0].decode("utf-8"))
        assert isinstance(row, dict)
        row["predecessor_sha256"] = "0" * 64
        rows[0] = (strict_json_dumps(row, compact=True) + "\n").encode("utf-8")
    elif mutation == "noncanonical":
        row = strict_json_loads(rows[0].decode("utf-8"))
        assert isinstance(row, dict)
        reordered = {
            "status": row["status"],
            "predecessor_sha256": row["predecessor_sha256"],
            "cell_id": row["cell_id"],
        }
        rows[0] = (strict_json_dumps(reordered, compact=True) + "\n").encode("utf-8")
    elif mutation == "missing_lf":
        rows[-1] = rows[-1].removesuffix(b"\n")
    elif mutation == "crlf":
        rows[0] = rows[0].removesuffix(b"\n") + b"\r\n"
    elif mutation == "mixed":
        row = strict_json_loads(rows[1].decode("utf-8"))
        assert isinstance(row, dict)
        del row["predecessor_sha256"]
        rows[1] = (strict_json_dumps(row, compact=True) + "\n").encode("utf-8")
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "truncated":
        rows.pop()
    elif mutation == "wrong_head":
        attempts["hash_chain_head"] = "0" * 64
    payload = b"".join(rows)
    journal.write_bytes(payload)
    attempts["sha256"] = _file_digest(payload)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match=match):
        verify_bundle_v1(root)


def test_chained_nonempty_closed_journal_cannot_have_live_tail(tmp_path: Path) -> None:
    root = _write_closed_bundle(
        tmp_path,
        outcome="failed",
        expected_cells=1,
        terminal_cells=0,
        physical_attempts=1,
    )
    journal = tmp_path / "attempts/journal.jsonl"
    journal.write_bytes(b'{"cell_id":"cell-0","status":"pending"}\n')
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    attempts = root_raw["attempts"]
    assert isinstance(attempts, dict)
    attempts["sha256"] = _file_digest(journal.read_bytes())
    attempts["logical"] = 1
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")
    _enable_attempt_chain(root)

    with pytest.raises(ArtifactError, match="live state"):
        verify_bundle_v1(root)


@pytest.mark.parametrize(
    "mutation,match",
    [("duplicate_terminal", "invalid attempt transition"), ("unknown_cell", "not present")],
)
def test_chained_attempt_journal_rejects_duplicate_terminal_and_unknown_rows(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = _write_closed_bundle(tmp_path)
    _enable_attempt_chain(root)
    journal = tmp_path / "attempts/journal.jsonl"
    transitions: list[dict[str, object]] = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        row = strict_json_loads(line)
        assert isinstance(row, dict)
        transitions.append({"cell_id": row["cell_id"], "status": row["status"]})
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    attempts = root_raw["attempts"]
    assert isinstance(attempts, dict)
    if mutation == "duplicate_terminal":
        transitions.append({"cell_id": "cell-0", "status": "failure"})
    else:
        transitions.append({"cell_id": "unknown-cell", "status": "pending"})
        attempts["logical"] = 3
        attempts["physical"] = 3
    payload, head = encode_attempt_journal(transitions)
    journal.write_bytes(payload)
    attempts["sha256"] = _file_digest(payload)
    attempts["hash_chain_head"] = head
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match=match):
        verify_bundle_v1(root)


@pytest.mark.parametrize("mutation", ["missing_selected", "hole", "extra", "malformed"])
def test_plugin_suite_reserved_roles_require_exact_complete_inventory(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _write_closed_bundle(tmp_path)
    _enable_plugin_suite_custody(root)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    artifacts = root_raw["artifacts"]
    assert isinstance(artifacts, list)
    if mutation == "missing_selected":
        artifacts[:] = [
            artifact
            for artifact in artifacts
            if not isinstance(artifact, dict) or artifact.get("role") != "selected_suite"
        ]
    elif mutation == "hole":
        _artifact_by_role(root_raw, "plugin_suite_1")["role"] = "plugin_suite_2"
    elif mutation == "extra":
        payload = (tmp_path / "plugin_suite_1.json").read_bytes()
        _append_json_artifact(
            root_raw,
            tmp_path,
            role="plugin_suite_2",
            path="plugin_suite_2.json",
            payload=payload,
        )
    else:
        _artifact_by_role(root_raw, "plugin_suite_1")["role"] = "plugin_suite_01"
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="plugin suite|reserved"):
        verify_bundle_v1(root)


def test_plugin_suite_custody_rejects_duplicate_ref_paths_before_checked(
    tmp_path: Path,
) -> None:
    root = _write_closed_bundle(tmp_path)
    _enable_plugin_suite_custody(root)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    plugin_artifact = _artifact_by_role(root_raw, "plugin")
    plugin_path = tmp_path / str(plugin_artifact["path"])
    plugin_raw = strict_json_loads(plugin_path.read_text(encoding="utf-8"))
    assert isinstance(plugin_raw, dict)
    refs = plugin_raw["suite_refs"]
    assert isinstance(refs, list)
    first, second = refs
    assert isinstance(first, dict) and isinstance(second, dict)
    second["path"] = first["path"]
    plugin_payload = strict_json_dumps(plugin_raw).encode("utf-8")
    plugin_path.write_bytes(plugin_payload)
    plugin_artifact["bytes"] = len(plugin_payload)
    plugin_artifact["sha256"] = _file_digest(plugin_payload)

    protocol_path = tmp_path / "protocol.json"
    protocol_raw = strict_json_loads(protocol_path.read_text(encoding="utf-8"))
    assert isinstance(protocol_raw, dict)
    protocol_plugin = protocol_raw["plugin"]
    assert isinstance(protocol_plugin, dict)
    protocol_plugin["artifact_sha256"] = _file_digest(plugin_payload)
    protocol_payload = strict_json_dumps(protocol_raw).encode("utf-8")
    protocol_path.write_bytes(protocol_payload)
    protocol_binding = root_raw["protocol"]
    assert isinstance(protocol_binding, dict)
    protocol_binding["bytes"] = len(protocol_payload)
    protocol_binding["sha256"] = _file_digest(protocol_payload)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(SchemaError, match="ref paths must be unique"):
        verify_bundle_v1(root)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema":"heliostune.selected-suite/1","plugin_suite_index":0}\n',
        selected_suite_descriptor_bytes(2),
        b'{"plugin_suite_index":0,"schema":"wrong"}\n',
    ],
)
def test_selected_suite_descriptor_is_exact_and_in_range(
    tmp_path: Path,
    payload: bytes,
) -> None:
    root = _write_closed_bundle(tmp_path)
    _enable_plugin_suite_custody(root)
    selected = tmp_path / "selected_suite.json"
    selected.write_bytes(payload)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    artifact = _artifact_by_role(root_raw, "selected_suite")
    artifact["bytes"] = len(payload)
    artifact["sha256"] = _file_digest(payload)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="selected suite"):
        verify_bundle_v1(root)


@pytest.mark.parametrize(
    "field,value,match",
    [("id", "other-plugin", "plugin ID"), ("version", "01", "canonical decimal")],
)
def test_protocol_plugin_identity_is_bound_to_inventory(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    root = _write_closed_bundle(tmp_path)
    _enable_plugin_suite_custody(root)
    protocol_path = tmp_path / "protocol.json"
    protocol_raw = strict_json_loads(protocol_path.read_text(encoding="utf-8"))
    assert isinstance(protocol_raw, dict)
    plugin = protocol_raw["plugin"]
    assert isinstance(plugin, dict)
    plugin[field] = value
    protocol_payload = strict_json_dumps(protocol_raw).encode("utf-8")
    protocol_path.write_bytes(protocol_payload)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    binding = root_raw["protocol"]
    assert isinstance(binding, dict)
    binding["bytes"] = len(protocol_payload)
    binding["sha256"] = _file_digest(protocol_payload)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match=match):
        verify_bundle_v1(root)


def test_bundle_rejects_symlinked_intermediate_component(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    root = _write_closed_bundle(bundle_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = (bundle_dir / "raw/measurements.jsonl").read_bytes()
    (outside / "measurements.jsonl").write_bytes(payload)
    (bundle_dir / "linked").symlink_to(outside, target_is_directory=True)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    artifact = _artifact_by_role(root_raw, "raw-measurements")
    artifact["path"] = "linked/measurements.jsonl"
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="symlinks"):
        verify_bundle_v1(root)


def test_bundle_rejects_hard_link_file_identity_alias(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    source = tmp_path / "raw/measurements.jsonl"
    alias = tmp_path / "raw/hard-link.jsonl"
    alias.hardlink_to(source)
    root_raw = strict_json_loads(root.read_text(encoding="utf-8"))
    assert isinstance(root_raw, dict)
    original = _artifact_by_role(root_raw, "raw-measurements")
    duplicate = copy.deepcopy(original)
    duplicate["role"] = "hard-link-alias"
    duplicate["path"] = "raw/hard-link.jsonl"
    artifacts = root_raw["artifacts"]
    assert isinstance(artifacts, list)
    artifacts.append(duplicate)
    root.write_text(strict_json_dumps(root_raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="file identity already used"):
        verify_bundle_v1(root)


def test_bundle_verifies_through_pinned_staging_directory_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_closed_bundle(tmp_path)
    directory_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))

    def forbid_resolution(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("pinned directory verifier resolved a diagnostic path")

    monkeypatch.setattr(Path, "resolve", forbid_resolution)
    try:
        verified = verify_bundle_v1_from_directory_fd(
            directory_fd,
            diagnostic_directory=tmp_path,
        )
        assert stat.S_ISDIR(os.fstat(directory_fd).st_mode)
    finally:
        os.close(directory_fd)

    assert verified.root_path == root


def test_ordinary_path_verifier_preserves_proc_fd_compatibility(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    directory_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        verified = verify_bundle_v1(f"/proc/self/fd/{directory_fd}/bundle.json")
    finally:
        os.close(directory_fd)

    assert verified.root_path == root


def test_pinned_directory_fd_ignores_substituted_diagnostic_path(tmp_path: Path) -> None:
    original = tmp_path / "staging"
    original.mkdir()
    _write_closed_bundle(original)
    directory_fd = os.open(original, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    pinned = tmp_path / "pinned"
    original.rename(pinned)
    original.mkdir()
    (original / "bundle.json").write_text("{}", encoding="utf-8")
    try:
        verified = verify_bundle_v1_from_directory_fd(
            directory_fd,
            diagnostic_directory=original,
        )
        with pytest.raises(SchemaError):
            verify_bundle_v1(original / "bundle.json")
    finally:
        os.close(directory_fd)

    assert verified.bundle.bundle_id == "methodology-bundle-test"
    assert verified.root_path == original / "bundle.json"


def test_parent_identity_capture_needs_search_but_not_read_permission(tmp_path: Path) -> None:
    searchable_parent = tmp_path / "searchable-only"
    root = _write_closed_bundle(searchable_parent / "bundle")
    searchable_parent.chmod(0o111)
    try:
        verified = verify_bundle_v1(root)
        identity = searchable_parent.stat()
    finally:
        searchable_parent.chmod(0o755)

    assert verified.root_parent_directory_identity == (
        identity.st_dev,
        identity.st_ino,
    )


@pytest.mark.parametrize("capability", ["dir_fd", "follow_symlinks"])
def test_parent_identity_capture_requires_descriptor_relative_stat_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
) -> None:
    root = _write_closed_bundle(tmp_path)
    support_name = f"supports_{capability}"
    supported = set(getattr(os, support_name))
    supported.discard(os.stat)
    monkeypatch.setattr(os, support_name, supported)

    with pytest.raises(ArtifactError, match="descriptor-pinned bundle verification is unsupported"):
        verify_bundle_v1(root)


def test_directory_fd_entrypoint_rejects_file_fd_without_closing_caller(tmp_path: Path) -> None:
    root = _write_closed_bundle(tmp_path)
    file_fd = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(ArtifactError, match="does not refer to a directory"):
            verify_bundle_v1_from_directory_fd(file_fd)
        assert os.fstat(file_fd).st_ino == root.stat().st_ino
    finally:
        os.close(file_fd)


def test_descriptor_capture_returns_exact_artifacts_in_requested_order(
    tmp_path: Path,
) -> None:
    root = _write_closed_bundle(tmp_path)
    verified = verify_bundle_v1(root)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        captured = capture_bundle_artifacts_v1_from_directory_fd(
            directory_fd,
            verified,
            ("timing", "plugin"),
            diagnostic_directory=tmp_path,
        )
        assert all(type(item) is CapturedBundleArtifactV1 for item in captured)
        assert tuple(item.artifact.role for item in captured) == ("timing", "plugin")
        assert captured[0].payload == (tmp_path / "artifacts/timing.json").read_bytes()
        assert captured[1].payload == (tmp_path / "artifacts/plugin.json").read_bytes()
        assert os.fstat(directory_fd).st_ino == tmp_path.stat().st_ino
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize("roles", [(), ("plugin", "plugin"), ("missing",)])
def test_descriptor_capture_rejects_invalid_or_missing_roles(
    tmp_path: Path,
    roles: tuple[str, ...],
) -> None:
    root = _write_closed_bundle(tmp_path)
    verified = verify_bundle_v1(root)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises((SchemaError, ArtifactError)):
            capture_bundle_artifacts_v1_from_directory_fd(
                directory_fd,
                verified,
                roles,
                diagnostic_directory=tmp_path,
            )
    finally:
        os.close(directory_fd)


def test_bundle_rejects_oversized_sparse_artifact_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_closed_bundle(tmp_path)
    document = strict_json_loads(root.read_text(encoding="utf-8"), source=root)
    assert isinstance(document, dict)
    artifact = _artifact_by_role(document, "plugin")
    oversized_bytes = 32 * 1024 * 1024 + 1
    artifact_path = tmp_path / str(artifact["path"])
    with artifact_path.open("r+b") as sparse_file:
        sparse_file.truncate(oversized_bytes)
    artifact["bytes"] = oversized_bytes
    root.write_text(strict_json_dumps(document), encoding="utf-8")

    oversized_identity = artifact_path.stat().st_ino
    original_read = os.read

    def guarded_read(descriptor: int, size: int) -> bytes:
        if os.fstat(descriptor).st_ino == oversized_identity:
            raise AssertionError("oversized artifact payload was read")
        return original_read(descriptor, size)

    monkeypatch.setattr(os, "read", guarded_read)
    with pytest.raises(ArtifactError, match="exceeds.*byte limit"):
        verify_bundle_v1(root)


def test_verification_limitations_reject_caller_forged_replay_status() -> None:
    with pytest.raises(SchemaError, match="analyzer_replay"):
        VerificationLimitations(analyzer_replay="checked")  # type: ignore[arg-type]
    with pytest.raises(SchemaError, match="offline_reproduction"):
        VerificationLimitations(offline_reproduction="checked")  # type: ignore[arg-type]
