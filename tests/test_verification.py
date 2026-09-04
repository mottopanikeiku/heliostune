from __future__ import annotations

import copy
import json
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest
from test_methodology import _write_closed_bundle

import heliostune.artifacts as artifact_io
import heliostune.verification as verification
from heliostune.artifacts import strict_json_dumps
from heliostune.errors import ArtifactError, SchemaError
from heliostune.methodology import (
    Lifecycle,
    VerificationLimitations,
    verify_bundle_v1,
    verify_bundle_v1_from_directory_fd,
)
from heliostune.verification import (
    VERIFICATION_CONTROL_NAMES_V1,
    VERIFIER_SOURCE_PATHS_V1,
    FileIdentityV1,
    VerificationControlsV1,
    VerificationRecordV1,
    build_verification_record_v1,
    encode_verification_record_v1,
    load_verification_record_v1,
    write_verification_record_v1,
)


def _record(tmp_path: Path) -> tuple[VerificationRecordV1, Any]:
    verified = verify_bundle_v1(_write_closed_bundle(tmp_path / "bundle"))
    return build_verification_record_v1(verified), verified


def _set(document: dict[str, object], path: str, value: object) -> None:
    target: Any = document
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value


def _delete(document: dict[str, object], path: str) -> None:
    target: Any = document
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        del target[int(parts[-1])]
    else:
        del target[parts[-1]]


def test_build_is_canonical_complete_and_deferred(tmp_path: Path) -> None:
    record, verified = _record(tmp_path)
    document = record.to_dict()

    assert record.schema == "heliostune.verification-record/1"
    assert record.bundle.root.bytes == verified.root_bytes
    assert record.bundle.root.sha256 == verified.root_sha256
    assert record.bundle.protocol.path == verified.bundle.protocol.path
    assert record.bundle.protocol.bytes == verified.protocol.bytes
    assert record.bundle.protocol.sha256 == verified.protocol.sha256
    assert record.bundle.protocol.study_id == verified.protocol.protocol.study_id
    assert record.bundle.protocol.revision == verified.protocol.protocol.revision
    assert record.bundle.attempts.path == verified.bundle.attempts.path
    assert record.bundle.attempts.bytes == verified.attempts_bytes
    assert record.bundle.attempts.sha256 == verified.bundle.attempts.sha256
    assert record.bundle.attempts.hash_chain_head == verified.bundle.attempts.hash_chain_head
    assert tuple(artifact.role for artifact in record.bundle.artifacts) == tuple(
        sorted(artifact.role for artifact in verified.bundle.artifacts)
    )
    assert tuple(document["controls"]) == VERIFICATION_CONTROL_NAMES_V1
    assert record.claim_eligible is False
    assert record.publication_eligible is False
    assert record.has_failed_controls is False
    assert encode_verification_record_v1(record) == strict_json_dumps(document).encode("utf-8")


def test_control_name_tuple_covers_exact_dataclass_fields() -> None:
    assert tuple(field.name for field in fields(VerificationControlsV1)) == (
        "protocol_ancestry",
        "evidence_nonpromotion",
        "semantic_content_beyond_digests",
        "plugin_suite_custody",
        "attempt_journal_hash_chain",
        "attempt_reconciliation",
        "claim_eligibility",
        "analyzer_replay",
        "provenance_tier_derivation",
        "signature_cryptography",
        "catalog_membership",
        "offline_reproduction",
    )

    assert tuple(field.name for field in fields(VerificationLimitations)) == (
        VERIFICATION_CONTROL_NAMES_V1
    )


@pytest.mark.parametrize("status", ["not_checked", "not_applicable", "failed"])
@pytest.mark.parametrize("name", VERIFICATION_CONTROL_NAMES_V1)
def test_every_nonchecked_status_forces_both_eligibility_flags(
    tmp_path: Path, name: str, status: str
) -> None:
    record, _ = _record(tmp_path)
    values = {control: "checked" for control in VERIFICATION_CONTROL_NAMES_V1}
    values[name] = status
    controls = VerificationControlsV1.from_dict(values)

    changed = replace(
        record,
        controls=controls,
        claim_eligible=False,
        publication_eligible=False,
    )

    assert changed.controls.all_checked is False
    assert changed.has_failed_controls is (status == "failed")
    assert VerificationRecordV1.from_dict(changed.to_dict()) == changed


@pytest.mark.parametrize("state", ["VERIFIED", "ANALYZED", "PUBLISHED"])
def test_lifecycle_labels_cannot_promote_deferred_controls(tmp_path: Path, state: str) -> None:
    record, _ = _record(tmp_path)
    changed = replace(record, lifecycle=Lifecycle(state=state, outcome="completed"))

    assert changed.claim_eligible is False
    assert changed.publication_eligible is False


def test_all_checked_is_the_only_eligibility_formula(tmp_path: Path) -> None:
    record, _ = _record(tmp_path)
    controls = VerificationControlsV1.from_dict(
        {name: "checked" for name in VERIFICATION_CONTROL_NAMES_V1}
    )
    eligible = replace(
        record,
        lifecycle=Lifecycle(state="SEALED", outcome="failed"),
        evidence_class="exploratory",
        controls=controls,
        claim_eligible=True,
        publication_eligible=True,
    )

    assert eligible.controls.all_checked is True
    assert VerificationRecordV1.from_dict(eligible.to_dict()) == eligible


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("schema", "heliostune.verification-record/2"),
        ("bundle.schema", "heliostune.bundle/2"),
        ("bundle.root.bytes", True),
        ("bundle.root.sha256", "A" * 64),
        ("bundle.protocol.path", "/absolute/protocol.json"),
        ("bundle.protocol.bytes", -1),
        ("bundle.protocol.sha256", "f" * 63),
        ("bundle.protocol.study_id", ""),
        ("bundle.protocol.revision", 0),
        ("bundle.attempts.path", "../attempts.jsonl"),
        ("bundle.attempts.bytes", False),
        ("bundle.attempts.sha256", "g" * 64),
        ("bundle.attempts.hash_chain_head", None),
        ("bundle.artifacts.0.path", "a\\b"),
        ("bundle.artifacts.0.bytes", True),
        ("bundle.artifacts.0.sha256", "0" * 63),
        ("lifecycle.state", "DONE"),
        ("evidence_class", "publication"),
        ("controls.protocol_ancestry", "passed"),
        ("claim_eligible", 1),
        ("publication_eligible", None),
    ],
)
def test_nested_schema_digest_path_size_and_status_faults_are_rejected(
    tmp_path: Path, path: str, replacement: object
) -> None:
    record, _ = _record(tmp_path)
    document = copy.deepcopy(record.to_dict())
    _set(document, path, replacement)

    with pytest.raises(SchemaError):
        VerificationRecordV1.from_dict(document)


@pytest.mark.parametrize(
    "path",
    [
        "verifier.source_sha256",
        "verifier.sources.0.bytes",
        "bundle.root.sha256",
        "bundle.protocol.study_id",
        "bundle.attempts.bytes",
        "bundle.artifacts",
        "lifecycle.outcome",
        "controls.offline_reproduction",
        "claim_eligible",
    ],
)
def test_missing_fields_at_every_level_are_rejected(tmp_path: Path, path: str) -> None:
    record, _ = _record(tmp_path)
    document = copy.deepcopy(record.to_dict())
    _delete(document, path)

    with pytest.raises(SchemaError):
        VerificationRecordV1.from_dict(document)


def test_unknown_and_duplicate_fields_fail_closed(tmp_path: Path) -> None:
    record, _ = _record(tmp_path)
    document = record.to_dict()
    document["unknown"] = True
    with pytest.raises(SchemaError, match="unknown"):
        VerificationRecordV1.from_dict(document)

    canonical = encode_verification_record_v1(record).decode("utf-8")
    duplicate = canonical.replace(
        '"schema": "heliostune.verification-record/1"',
        '"schema": "heliostune.verification-record/1",\n  "schema": "heliostune.verification-record/1"',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(SchemaError, match="duplicate"):
        load_verification_record_v1(path)


def test_artifacts_must_have_canonical_role_path_order(tmp_path: Path) -> None:
    record, _ = _record(tmp_path)
    document = record.to_dict()
    artifacts = document["bundle"]["artifacts"]
    assert isinstance(artifacts, list)
    artifacts.reverse()

    with pytest.raises(SchemaError, match="sorted"):
        VerificationRecordV1.from_dict(document)


def test_eligibility_values_must_equal_all_checked(tmp_path: Path) -> None:
    record, _ = _record(tmp_path)
    for field_name in ("claim_eligible", "publication_eligible"):
        document = record.to_dict()
        document[field_name] = True
        with pytest.raises(SchemaError, match="all-controls-checked"):
            VerificationRecordV1.from_dict(document)


def test_repeated_copies_in_different_directories_have_identical_bytes(tmp_path: Path) -> None:
    first = verify_bundle_v1(_write_closed_bundle(tmp_path / "one"))
    second = verify_bundle_v1(_write_closed_bundle(tmp_path / "two"))

    assert first.root_path != second.root_path
    assert encode_verification_record_v1(build_verification_record_v1(first)) == (
        encode_verification_record_v1(build_verification_record_v1(second))
    )


def test_build_does_not_reread_verified_bundle_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, verified = _record(tmp_path)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (verified.root_path, *verified.referenced_paths)
    }
    original_read_bytes = Path.read_bytes
    bundle_directory = verified.root_path.parent

    def guarded_read_bytes(path: Path) -> bytes:
        try:
            path.resolve().relative_to(bundle_directory)
        except ValueError:
            return original_read_bytes(path)
        raise AssertionError(f"reread verified input {path}")

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    rebuilt = build_verification_record_v1(verified)
    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)

    assert rebuilt == record
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (verified.root_path, *verified.referenced_paths)
    } == before


def test_source_roster_aggregate_and_historical_identity(tmp_path: Path) -> None:
    record, _ = _record(tmp_path)
    assert tuple(source.path for source in record.verifier.sources) == VERIFIER_SOURCE_PATHS_V1

    historical_sources = list(record.verifier.sources)
    first = historical_sources[0]
    historical_sources[0] = FileIdentityV1(
        path=first.path,
        bytes=first.bytes,
        sha256="0" * 64,
    )
    sources = tuple(historical_sources)
    historical_verifier = replace(
        record.verifier,
        source_sha256=verification._source_aggregate_sha256(sources),
        sources=sources,
    )
    historical = replace(record, verifier=historical_verifier)
    path = tmp_path / "historical.json"
    path.write_bytes(encode_verification_record_v1(historical))

    assert load_verification_record_v1(path) == historical


@pytest.mark.parametrize("fault", ["roster", "order", "aggregate"])
def test_source_identity_faults_are_rejected(tmp_path: Path, fault: str) -> None:
    record, _ = _record(tmp_path)
    document = record.to_dict()
    verifier = document["verifier"]
    sources = verifier["sources"]
    assert isinstance(sources, list)
    if fault == "roster":
        sources[0]["path"] = "heliostune/other.py"
    elif fault == "order":
        sources[0], sources[1] = sources[1], sources[0]
    else:
        verifier["source_sha256"] = "0" * 64

    with pytest.raises(SchemaError):
        VerificationRecordV1.from_dict(document)


@pytest.mark.parametrize("mutation", ["source", "version"])
def test_source_or_version_mutation_after_import_fails_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    _, verified = _record(tmp_path)
    imported = verification._IMPORTED_VERIFIER_IDENTITY_V1
    if mutation == "source":
        sources = list(imported.sources)
        sources[0] = replace(sources[0], sha256="f" * 64)
        changed_sources = tuple(sources)
        changed = replace(
            imported,
            sources=changed_sources,
            source_sha256=verification._source_aggregate_sha256(changed_sources),
        )
    else:
        changed = replace(imported, version=imported.version + ".changed")
    monkeypatch.setattr(verification, "_capture_verifier_identity_v1", lambda: changed)

    with pytest.raises(ArtifactError, match="changed after import"):
        build_verification_record_v1(verified)


def test_loader_requires_exact_canonical_bytes(tmp_path: Path) -> None:
    record, _ = _record(tmp_path)
    canonical = encode_verification_record_v1(record)
    path = tmp_path / "record.json"
    path.write_bytes(canonical)
    assert load_verification_record_v1(path) == record

    decoded = json.loads(canonical)
    path.write_text(json.dumps(decoded), encoding="utf-8")
    with pytest.raises(SchemaError, match="not canonical"):
        load_verification_record_v1(path)


@pytest.mark.parametrize("payload", [b"\xff", b"[" * 2000, b'{"schema":NaN}'])
def test_loader_rejects_malformed_recursive_or_non_strict_json(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "malformed.json"
    path.write_bytes(payload)

    with pytest.raises(SchemaError):
        load_verification_record_v1(path)


def test_write_requires_exact_match_and_sibling_noreplace_output(tmp_path: Path) -> None:
    record, verified = _record(tmp_path)
    output = tmp_path / "record.json"

    write_verification_record_v1(output, record, verified=verified)
    assert output.read_bytes() == encode_verification_record_v1(record)
    with pytest.raises(ArtifactError):
        write_verification_record_v1(output, record, verified=verified)

    mismatch = replace(record, lifecycle=Lifecycle(state="VERIFIED", outcome="completed"))
    other_output = tmp_path / "other.json"
    with pytest.raises(ArtifactError, match="does not exactly match"):
        write_verification_record_v1(other_output, mismatch, verified=verified)
    assert not other_output.exists()


@pytest.mark.parametrize("relative", ["record.json", "nested/record.json"])
def test_write_rejects_bundle_directory_and_descendants_before_creation(
    tmp_path: Path, relative: str
) -> None:
    record, verified = _record(tmp_path)
    destination = verified.root_path.parent / relative
    destination.parent.mkdir(exist_ok=True)

    with pytest.raises(ArtifactError, match="sibling"):
        write_verification_record_v1(destination, record, verified=verified)
    assert not destination.exists()


def test_write_rejects_intermediate_symlink_alias_into_bundle(tmp_path: Path) -> None:
    record, verified = _record(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "alias").symlink_to(verified.root_path.parent, target_is_directory=True)
    destination = outside / "alias/artifacts/record.json"

    with pytest.raises(ArtifactError, match="sibling"):
        write_verification_record_v1(destination, record, verified=verified)
    assert not destination.exists()


def test_write_tracks_renamed_bundle_directory_by_identity(tmp_path: Path) -> None:
    record, verified = _record(tmp_path)
    moved = tmp_path / "renamed-bundle"
    verified.root_path.parent.rename(moved)
    destination = moved / "record.json"

    with pytest.raises(ArtifactError, match="sibling"):
        write_verification_record_v1(destination, record, verified=verified)
    assert not destination.exists()


def test_write_rejects_fd_verified_bundle_with_untrusted_diagnostic_path(
    tmp_path: Path,
) -> None:
    root = _write_closed_bundle(tmp_path / "bundle")
    directory_fd = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        verified = verify_bundle_v1_from_directory_fd(
            directory_fd,
            diagnostic_directory=tmp_path / "untrusted-diagnostic",
        )
    finally:
        os.close(directory_fd)
    record = build_verification_record_v1(verified)
    destination = root.parent.parent / "record.json"

    with pytest.raises(ArtifactError, match="cannot write"):
        write_verification_record_v1(destination, record, verified=verified)
    assert not destination.exists()


def test_write_rejects_arbitrary_non_sibling_directory(tmp_path: Path) -> None:
    record, verified = _record(tmp_path)
    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    destination = arbitrary / "record.json"

    with pytest.raises(ArtifactError, match="sibling"):
        write_verification_record_v1(destination, record, verified=verified)
    assert not destination.exists()


def test_write_rejects_rebound_checked_parent_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, verified = _record(tmp_path / "container")
    output_parent = tmp_path / "container"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    moved_parent = tmp_path / "moved-container"
    destination = output_parent / "record.json"
    original_writer = artifact_io.write_bytes_atomic_noreplace_at

    def swap_then_write(
        directory_fd: int,
        name: str,
        payload: bytes,
        *,
        expected_parent_path: str | Path | None = None,
        bundle_directory_fd: int | None = None,
        bundle_directory_name: str | None = None,
        expected_bundle_identity: tuple[int, int] | None = None,
    ) -> None:
        output_parent.rename(moved_parent)
        replacement.rename(output_parent)
        original_writer(
            directory_fd,
            name,
            payload,
            expected_parent_path=expected_parent_path,
            bundle_directory_fd=bundle_directory_fd,
            bundle_directory_name=bundle_directory_name,
            expected_bundle_identity=expected_bundle_identity,
        )

    monkeypatch.setattr(artifact_io, "write_bytes_atomic_noreplace_at", swap_then_write)
    with pytest.raises(ArtifactError):
        write_verification_record_v1(destination, record, verified=verified)

    assert not destination.exists()
    assert not (moved_parent / "record.json").exists()


def test_bundle_path_rebind_before_write_creates_no_record(tmp_path: Path) -> None:
    record, verified = _record(tmp_path)
    moved_bundle = tmp_path / "original-bundle"
    verified.root_path.parent.rename(moved_bundle)
    verified.root_path.parent.mkdir()
    destination = tmp_path / "record.json"

    with pytest.raises(ArtifactError, match="changed after verification"):
        write_verification_record_v1(destination, record, verified=verified)
    assert not destination.exists()


def test_bundle_reparent_after_link_reports_irreversible_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, verified = _record(tmp_path)
    destination = tmp_path / "record.json"
    moved_parent = tmp_path / "elsewhere"
    moved_parent.mkdir()
    moved_bundle = moved_parent / "bundle"
    original_link = artifact_io._link_fd_noreplace

    def link_then_reparent_bundle(
        source_fd: int,
        directory_fd: int,
        name: str,
    ) -> None:
        original_link(source_fd, directory_fd, name)
        verified.root_path.parent.rename(moved_bundle)

    monkeypatch.setattr(artifact_io, "_link_fd_noreplace", link_then_reparent_bundle)
    with pytest.raises(ArtifactError, match="committed"):
        write_verification_record_v1(destination, record, verified=verified)

    assert destination.read_bytes() == encode_verification_record_v1(record)


def test_output_parent_close_failure_after_success_is_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, verified = _record(tmp_path)
    output_parent = tmp_path
    destination = output_parent / "record.json"
    original_open = os.open
    original_close = os.close
    opened_parent: list[int] = []

    def capture_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is None and Path(path) == output_parent:
            opened_parent.append(descriptor)
        return descriptor

    def fail_parent_close(descriptor: int) -> None:
        if opened_parent and descriptor == opened_parent[0]:
            raise OSError("simulated close failure")
        original_close(descriptor)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "close", fail_parent_close)
    try:
        write_verification_record_v1(destination, record, verified=verified)
        assert destination.read_bytes() == encode_verification_record_v1(record)
    finally:
        original_close(opened_parent[0])


def test_output_parent_close_failure_does_not_mask_containment_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, verified = _record(tmp_path)
    output_parent = verified.root_path.parent
    destination = output_parent / "record.json"
    original_open = os.open
    original_close = os.close
    opened_parent: list[int] = []

    def capture_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is None and Path(path) == output_parent:
            opened_parent.append(descriptor)
        return descriptor

    def fail_parent_close(descriptor: int) -> None:
        if opened_parent and descriptor == opened_parent[0]:
            raise OSError("simulated close failure")
        original_close(descriptor)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "close", fail_parent_close)
    try:
        with pytest.raises(ArtifactError, match="sibling"):
            write_verification_record_v1(destination, record, verified=verified)
        assert not destination.exists()
    finally:
        original_close(opened_parent[0])
