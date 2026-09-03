"""Canonical, CPU-only verification records over closed evidence bundles."""

from __future__ import annotations

import hashlib
import importlib.resources
import os
from contextlib import suppress
from dataclasses import dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from heliostune.artifacts import strict_json_dumps, strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.methodology import (
    Artifact,
    Lifecycle,
    VerificationLimitations,
    VerifiedBundle,
)
from heliostune.validation import exact_bool, exact_fields, exact_int, nonblank_string

ControlStatus = Literal["checked", "not_checked", "not_applicable", "failed"]

VERIFICATION_CONTROL_NAMES_V1 = (
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

VERIFIER_SOURCE_PATHS_V1 = (
    "heliostune/artifacts.py",
    "heliostune/errors.py",
    "heliostune/methodology.py",
    "heliostune/scope.py",
    "heliostune/validation.py",
    "heliostune/verification.py",
)

_CONTROL_STATUSES = {"checked", "not_checked", "not_applicable", "failed"}
_DIGEST_HEXDIGITS = frozenset("0123456789abcdef")
_EVIDENCE_CLASSES = {"exploratory", "engineering_gate", "confirmatory"}
_SOURCE_DOMAIN = b"heliostune.verification-sources/1\0"


def _digest(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    if len(result) != 64 or any(character not in _DIGEST_HEXDIGITS for character in result):
        raise SchemaError(f"{context} must be a lowercase SHA-256 digest")
    return result


def _relative_path(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    if "\\" in result or "\x00" in result:
        raise SchemaError(f"{context} must be a normalized POSIX relative path")
    path = PurePosixPath(result)
    if (
        path.is_absolute()
        or result != path.as_posix()
        or any(part in {"", ".", ".."} for part in result.split("/"))
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise SchemaError(f"{context} must be normalized, relative, and non-escaping")
    return result


def _byte_count(value: object, *, context: str) -> int:
    result = exact_int(value, context=context, minimum=0)
    if result >= 1 << 64:
        raise SchemaError(f"{context} must fit an unsigned 64-bit integer")
    return result


def _exact_tuple(value: object, expected: type[object], *, context: str) -> None:
    if type(value) is not tuple:
        raise SchemaError(f"{context} must be a tuple")
    for item in cast(tuple[object, ...], value):
        if type(item) is not expected:
            raise SchemaError(f"{context} entries must be {expected.__name__}")


def _object_array(value: object, *, context: str) -> list[object]:
    if type(value) is not list:
        raise SchemaError(f"{context} must be an array")
    return cast(list[object], value)


@dataclass(frozen=True, slots=True)
class ContentIdentityV1:
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _byte_count(self.bytes, context="content identity bytes")
        _digest(self.sha256, context="content identity sha256")

    @classmethod
    def from_dict(cls, value: object) -> ContentIdentityV1:
        data = exact_fields(value, required=("bytes", "sha256"), context="content identity")
        return cls(
            bytes=_byte_count(data["bytes"], context="content identity bytes"),
            sha256=_digest(data["sha256"], context="content identity sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class FileIdentityV1:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.path, context="file identity path")
        _byte_count(self.bytes, context="file identity bytes")
        _digest(self.sha256, context="file identity sha256")

    @classmethod
    def from_dict(cls, value: object) -> FileIdentityV1:
        data = exact_fields(value, required=("path", "bytes", "sha256"), context="file identity")
        return cls(
            path=_relative_path(data["path"], context="file identity path"),
            bytes=_byte_count(data["bytes"], context="file identity bytes"),
            sha256=_digest(data["sha256"], context="file identity sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ProtocolIdentityV1:
    path: str
    bytes: int
    sha256: str
    study_id: str
    revision: int

    def __post_init__(self) -> None:
        _relative_path(self.path, context="protocol identity path")
        _byte_count(self.bytes, context="protocol identity bytes")
        _digest(self.sha256, context="protocol identity sha256")
        nonblank_string(self.study_id, context="protocol identity study_id")
        exact_int(self.revision, context="protocol identity revision", minimum=1)

    @classmethod
    def from_dict(cls, value: object) -> ProtocolIdentityV1:
        data = exact_fields(
            value,
            required=("path", "bytes", "sha256", "study_id", "revision"),
            context="protocol identity",
        )
        return cls(
            path=_relative_path(data["path"], context="protocol identity path"),
            bytes=_byte_count(data["bytes"], context="protocol identity bytes"),
            sha256=_digest(data["sha256"], context="protocol identity sha256"),
            study_id=nonblank_string(data["study_id"], context="protocol identity study_id"),
            revision=exact_int(data["revision"], context="protocol identity revision", minimum=1),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "study_id": self.study_id,
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class AttemptIdentityV1:
    path: str
    bytes: int
    sha256: str
    hash_chain_head: str

    def __post_init__(self) -> None:
        _relative_path(self.path, context="attempt identity path")
        _byte_count(self.bytes, context="attempt identity bytes")
        _digest(self.sha256, context="attempt identity sha256")
        _digest(self.hash_chain_head, context="attempt identity hash_chain_head")

    @classmethod
    def from_dict(cls, value: object) -> AttemptIdentityV1:
        data = exact_fields(
            value,
            required=("path", "bytes", "sha256", "hash_chain_head"),
            context="attempt identity",
        )
        return cls(
            path=_relative_path(data["path"], context="attempt identity path"),
            bytes=_byte_count(data["bytes"], context="attempt identity bytes"),
            sha256=_digest(data["sha256"], context="attempt identity sha256"),
            hash_chain_head=_digest(
                data["hash_chain_head"], context="attempt identity hash_chain_head"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "hash_chain_head": self.hash_chain_head,
        }


def _source_aggregate_sha256(sources: tuple[FileIdentityV1, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_DOMAIN)
    for source in sources:
        path_bytes = source.path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(source.bytes.to_bytes(8, "big"))
        digest.update(bytes.fromhex(source.sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VerifierIdentityV1:
    package: str
    version: str
    source_sha256: str
    sources: tuple[FileIdentityV1, ...]

    def __post_init__(self) -> None:
        if type(self.package) is not str or self.package != "heliostune":
            raise SchemaError("verifier package must be 'heliostune'")
        nonblank_string(self.version, context="verifier version")
        _digest(self.source_sha256, context="verifier source_sha256")
        _exact_tuple(self.sources, FileIdentityV1, context="verifier sources")
        if tuple(source.path for source in self.sources) != VERIFIER_SOURCE_PATHS_V1:
            raise SchemaError("verifier sources must use the fixed v1 roster and order")
        if self.source_sha256 != _source_aggregate_sha256(self.sources):
            raise SchemaError("verifier source_sha256 does not match its framed sources")

    @classmethod
    def from_dict(cls, value: object) -> VerifierIdentityV1:
        data = exact_fields(
            value,
            required=("package", "version", "source_sha256", "sources"),
            context="verifier identity",
        )
        sources = tuple(
            FileIdentityV1.from_dict(item)
            for item in _object_array(data["sources"], context="verifier sources")
        )
        return cls(
            package=nonblank_string(data["package"], context="verifier package"),
            version=nonblank_string(data["version"], context="verifier version"),
            source_sha256=_digest(data["source_sha256"], context="verifier source_sha256"),
            sources=sources,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "package": self.package,
            "version": self.version,
            "source_sha256": self.source_sha256,
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class BundleIdentityV1:
    schema: Literal["heliostune.bundle/1"]
    bundle_id: str
    root: ContentIdentityV1
    protocol: ProtocolIdentityV1
    attempts: AttemptIdentityV1
    artifacts: tuple[Artifact, ...]

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != "heliostune.bundle/1":
            raise SchemaError("bundle identity schema must be 'heliostune.bundle/1'")
        nonblank_string(self.bundle_id, context="bundle identity bundle_id")
        if type(self.root) is not ContentIdentityV1:
            raise SchemaError("bundle identity root must be a ContentIdentityV1")
        if type(self.protocol) is not ProtocolIdentityV1:
            raise SchemaError("bundle identity protocol must be a ProtocolIdentityV1")
        if type(self.attempts) is not AttemptIdentityV1:
            raise SchemaError("bundle identity attempts must be an AttemptIdentityV1")
        _exact_tuple(self.artifacts, Artifact, context="bundle identity artifacts")
        if self.artifacts != tuple(sorted(self.artifacts, key=lambda item: (item.role, item.path))):
            raise SchemaError("bundle identity artifacts must be sorted by role and path")
        roles = tuple(artifact.role for artifact in self.artifacts)
        if len(roles) != len(set(roles)):
            raise SchemaError("bundle identity artifact roles must be unique")
        paths = (self.protocol.path, self.attempts.path, *(item.path for item in self.artifacts))
        if len(paths) != len(set(paths)):
            raise SchemaError("bundle identity paths must be unique")

    @classmethod
    def from_dict(cls, value: object) -> BundleIdentityV1:
        data = exact_fields(
            value,
            required=("schema", "bundle_id", "root", "protocol", "attempts", "artifacts"),
            context="bundle identity",
        )
        schema = nonblank_string(data["schema"], context="bundle identity schema")
        if schema != "heliostune.bundle/1":
            raise SchemaError("bundle identity schema must be 'heliostune.bundle/1'")
        return cls(
            schema="heliostune.bundle/1",
            bundle_id=nonblank_string(data["bundle_id"], context="bundle identity bundle_id"),
            root=ContentIdentityV1.from_dict(data["root"]),
            protocol=ProtocolIdentityV1.from_dict(data["protocol"]),
            attempts=AttemptIdentityV1.from_dict(data["attempts"]),
            artifacts=tuple(
                Artifact.from_dict(item)
                for item in _object_array(data["artifacts"], context="bundle identity artifacts")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "bundle_id": self.bundle_id,
            "root": self.root.to_dict(),
            "protocol": self.protocol.to_dict(),
            "attempts": self.attempts.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class VerificationControlsV1:
    protocol_ancestry: ControlStatus
    evidence_nonpromotion: ControlStatus
    semantic_content_beyond_digests: ControlStatus
    plugin_suite_custody: ControlStatus
    attempt_journal_hash_chain: ControlStatus
    attempt_reconciliation: ControlStatus
    claim_eligibility: ControlStatus
    analyzer_replay: ControlStatus
    provenance_tier_derivation: ControlStatus
    signature_cryptography: ControlStatus
    catalog_membership: ControlStatus
    offline_reproduction: ControlStatus

    def __post_init__(self) -> None:
        for name in VERIFICATION_CONTROL_NAMES_V1:
            value = getattr(self, name)
            if type(value) is not str or value not in _CONTROL_STATUSES:
                raise SchemaError(f"verification control {name} has an invalid status")

    @classmethod
    def from_dict(cls, value: object) -> VerificationControlsV1:
        data = exact_fields(
            value,
            required=VERIFICATION_CONTROL_NAMES_V1,
            context="verification controls",
        )
        statuses: dict[str, ControlStatus] = {}
        for name in VERIFICATION_CONTROL_NAMES_V1:
            status = nonblank_string(data[name], context=f"verification control {name}")
            if status not in _CONTROL_STATUSES:
                raise SchemaError(f"verification control {name} has an invalid status")
            statuses[name] = cast(ControlStatus, status)
        return cls(
            protocol_ancestry=statuses["protocol_ancestry"],
            evidence_nonpromotion=statuses["evidence_nonpromotion"],
            semantic_content_beyond_digests=statuses["semantic_content_beyond_digests"],
            plugin_suite_custody=statuses["plugin_suite_custody"],
            attempt_journal_hash_chain=statuses["attempt_journal_hash_chain"],
            attempt_reconciliation=statuses["attempt_reconciliation"],
            claim_eligibility=statuses["claim_eligibility"],
            analyzer_replay=statuses["analyzer_replay"],
            provenance_tier_derivation=statuses["provenance_tier_derivation"],
            signature_cryptography=statuses["signature_cryptography"],
            catalog_membership=statuses["catalog_membership"],
            offline_reproduction=statuses["offline_reproduction"],
        )

    @property
    def all_checked(self) -> bool:
        return all(getattr(self, name) == "checked" for name in VERIFICATION_CONTROL_NAMES_V1)

    @property
    def has_failed_controls(self) -> bool:
        return any(getattr(self, name) == "failed" for name in VERIFICATION_CONTROL_NAMES_V1)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in VERIFICATION_CONTROL_NAMES_V1}


assert (
    tuple(field.name for field in fields(VerificationControlsV1)) == VERIFICATION_CONTROL_NAMES_V1
)
assert (
    tuple(field.name for field in fields(VerificationLimitations)) == VERIFICATION_CONTROL_NAMES_V1
)


@dataclass(frozen=True, slots=True)
class VerificationRecordV1:
    schema: Literal["heliostune.verification-record/1"]
    verifier: VerifierIdentityV1
    bundle: BundleIdentityV1
    lifecycle: Lifecycle
    evidence_class: Literal["exploratory", "engineering_gate", "confirmatory"]
    controls: VerificationControlsV1
    claim_eligible: bool
    publication_eligible: bool

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != "heliostune.verification-record/1":
            raise SchemaError(
                "verification record schema must be 'heliostune.verification-record/1'"
            )
        if type(self.verifier) is not VerifierIdentityV1:
            raise SchemaError("verification record verifier must be a VerifierIdentityV1")
        if type(self.bundle) is not BundleIdentityV1:
            raise SchemaError("verification record bundle must be a BundleIdentityV1")
        if type(self.lifecycle) is not Lifecycle:
            raise SchemaError("verification record lifecycle must be a Lifecycle")
        if type(self.evidence_class) is not str or self.evidence_class not in _EVIDENCE_CLASSES:
            raise SchemaError("verification record evidence_class is invalid")
        if type(self.controls) is not VerificationControlsV1:
            raise SchemaError("verification record controls must be VerificationControlsV1")
        exact_bool(self.claim_eligible, context="verification record claim_eligible")
        exact_bool(self.publication_eligible, context="verification record publication_eligible")
        if self.claim_eligible != self.controls.all_checked:
            raise SchemaError("verification record claim_eligible must equal all-controls-checked")
        if self.publication_eligible != self.controls.all_checked:
            raise SchemaError(
                "verification record publication_eligible must equal all-controls-checked"
            )

    @classmethod
    def from_dict(cls, value: object) -> VerificationRecordV1:
        data = exact_fields(
            value,
            required=(
                "schema",
                "verifier",
                "bundle",
                "lifecycle",
                "evidence_class",
                "controls",
                "claim_eligible",
                "publication_eligible",
            ),
            context="verification record",
        )
        schema = nonblank_string(data["schema"], context="verification record schema")
        if schema != "heliostune.verification-record/1":
            raise SchemaError(
                "verification record schema must be 'heliostune.verification-record/1'"
            )
        evidence_class = nonblank_string(
            data["evidence_class"], context="verification record evidence_class"
        )
        if evidence_class not in _EVIDENCE_CLASSES:
            raise SchemaError("verification record evidence_class is invalid")
        return cls(
            schema="heliostune.verification-record/1",
            verifier=VerifierIdentityV1.from_dict(data["verifier"]),
            bundle=BundleIdentityV1.from_dict(data["bundle"]),
            lifecycle=Lifecycle.from_dict(data["lifecycle"]),
            evidence_class=cast(
                Literal["exploratory", "engineering_gate", "confirmatory"], evidence_class
            ),
            controls=VerificationControlsV1.from_dict(data["controls"]),
            claim_eligible=exact_bool(
                data["claim_eligible"], context="verification record claim_eligible"
            ),
            publication_eligible=exact_bool(
                data["publication_eligible"], context="verification record publication_eligible"
            ),
        )

    @property
    def has_failed_controls(self) -> bool:
        return self.controls.has_failed_controls

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "verifier": self.verifier.to_dict(),
            "bundle": self.bundle.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
            "evidence_class": self.evidence_class,
            "controls": self.controls.to_dict(),
            "claim_eligible": self.claim_eligible,
            "publication_eligible": self.publication_eligible,
        }


def _capture_verifier_identity_v1() -> VerifierIdentityV1:
    try:
        package_version = version("heliostune")
        package_root = importlib.resources.files("heliostune")
        identities: list[FileIdentityV1] = []
        for relative_path in VERIFIER_SOURCE_PATHS_V1:
            payload = package_root.joinpath(relative_path.removeprefix("heliostune/")).read_bytes()
            identities.append(
                FileIdentityV1(
                    path=relative_path,
                    bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
    except (OSError, PackageNotFoundError, UnicodeError) as exc:
        raise ArtifactError(
            f"cannot identify installed heliostune verifier sources: {exc}"
        ) from exc
    sources = tuple(identities)
    return VerifierIdentityV1(
        package="heliostune",
        version=package_version,
        source_sha256=_source_aggregate_sha256(sources),
        sources=sources,
    )


_IMPORTED_VERIFIER_IDENTITY_V1 = _capture_verifier_identity_v1()


def _controls_from_limitations(limitations: VerificationLimitations) -> VerificationControlsV1:
    if type(limitations) is not VerificationLimitations:
        raise SchemaError("verified bundle limitations must be VerificationLimitations")
    return VerificationControlsV1.from_dict(
        {name: getattr(limitations, name) for name in VERIFICATION_CONTROL_NAMES_V1}
    )


def build_verification_record_v1(verified: VerifiedBundle) -> VerificationRecordV1:
    """Build a deterministic record without rereading any verified bundle input."""

    if type(verified) is not VerifiedBundle:
        raise SchemaError("verified must be a VerifiedBundle")
    verifier = _capture_verifier_identity_v1()
    if verifier != _IMPORTED_VERIFIER_IDENTITY_V1:
        raise ArtifactError("installed heliostune verifier sources or version changed after import")

    bundle = verified.bundle
    protocol = verified.protocol
    if protocol.sha256 != bundle.protocol.sha256 or protocol.bytes != bundle.protocol.bytes:
        raise ArtifactError("verified protocol identity does not match the bundle binding")
    artifacts = tuple(sorted(bundle.artifacts, key=lambda item: (item.role, item.path)))
    controls = _controls_from_limitations(verified.limitations)
    all_checked = controls.all_checked
    return VerificationRecordV1(
        schema="heliostune.verification-record/1",
        verifier=verifier,
        bundle=BundleIdentityV1(
            schema=bundle.schema,
            bundle_id=bundle.bundle_id,
            root=ContentIdentityV1(bytes=verified.root_bytes, sha256=verified.root_sha256),
            protocol=ProtocolIdentityV1(
                path=bundle.protocol.path,
                bytes=protocol.bytes,
                sha256=protocol.sha256,
                study_id=protocol.protocol.study_id,
                revision=protocol.protocol.revision,
            ),
            attempts=AttemptIdentityV1(
                path=bundle.attempts.path,
                bytes=verified.attempts_bytes,
                sha256=bundle.attempts.sha256,
                hash_chain_head=bundle.attempts.hash_chain_head,
            ),
            artifacts=artifacts,
        ),
        lifecycle=bundle.lifecycle,
        evidence_class=protocol.protocol.evidence_class,
        controls=controls,
        claim_eligible=all_checked,
        publication_eligible=all_checked,
    )


def encode_verification_record_v1(record: VerificationRecordV1) -> bytes:
    """Encode one record as canonical UTF-8 strict JSON bytes."""

    if type(record) is not VerificationRecordV1:
        raise SchemaError("record must be a VerificationRecordV1")
    validated = VerificationRecordV1.from_dict(record.to_dict())
    if validated != record:
        raise SchemaError("record does not round-trip through its canonical model")
    return strict_json_dumps(validated.to_dict()).encode("utf-8")


def load_verification_record_v1(path: str | Path) -> VerificationRecordV1:
    """Load a byte-canonical historical verification record."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read verification record {source}: {exc}") from exc
    try:
        text = payload.decode("utf-8")
        decoded = strict_json_loads(text, source=source)
    except UnicodeError as exc:
        raise SchemaError(f"{source}: verification record must be UTF-8") from exc
    except RecursionError as exc:
        raise SchemaError(f"{source}: verification record nesting is too deep") from exc
    record = VerificationRecordV1.from_dict(decoded)
    if encode_verification_record_v1(record) != payload:
        raise SchemaError(f"{source}: verification record bytes are not canonical")
    return record


def _directory_is_at_or_below(
    directory_descriptor: int,
    ancestor_identity: tuple[int, int],
) -> bool:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    current = os.dup(directory_descriptor)
    try:
        while True:
            identity = os.fstat(current)
            current_identity = (identity.st_dev, identity.st_ino)
            if current_identity == ancestor_identity:
                return True
            parent = os.open("..", flags, dir_fd=current)
            parent_identity = os.fstat(parent)
            if (parent_identity.st_dev, parent_identity.st_ino) == current_identity:
                os.close(parent)
                return False
            os.close(current)
            current = parent
    finally:
        os.close(current)


def write_verification_record_v1(
    path: str | Path,
    record: VerificationRecordV1,
    *,
    verified: VerifiedBundle,
) -> None:
    """Safely write an exact record outside its verified bundle directory."""

    expected = build_verification_record_v1(verified)
    if type(record) is not VerificationRecordV1 or record != expected:
        raise ArtifactError("verification record does not exactly match the verified bundle")
    payload = encode_verification_record_v1(record)

    destination = Path(path)
    name = destination.name
    if not name or name in {".", ".."}:
        raise ArtifactError(f"invalid verification record output path {destination}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        output_parent_descriptor = os.open(destination.parent, flags)
    except OSError as exc:
        raise ArtifactError(
            f"cannot open verification record output parent {destination.parent}: {exc}"
        ) from exc
    try:
        if _directory_is_at_or_below(
            output_parent_descriptor,
            verified.root_directory_identity,
        ):
            raise ArtifactError(
                "verification record output must be outside the verified bundle directory"
            )

        from heliostune.artifacts import write_bytes_atomic_noreplace_at

        write_bytes_atomic_noreplace_at(
            output_parent_descriptor,
            name,
            payload,
            expected_parent_path=destination.parent,
        )
    except OSError as exc:
        raise ArtifactError(
            f"cannot inspect verification record output parent {destination.parent}: {exc}"
        ) from exc
    finally:
        with suppress(OSError):
            os.close(output_parent_descriptor)


__all__ = [
    "AttemptIdentityV1",
    "BundleIdentityV1",
    "ContentIdentityV1",
    "ControlStatus",
    "FileIdentityV1",
    "ProtocolIdentityV1",
    "VERIFICATION_CONTROL_NAMES_V1",
    "VERIFIER_SOURCE_PATHS_V1",
    "VerificationControlsV1",
    "VerificationRecordV1",
    "VerifierIdentityV1",
    "build_verification_record_v1",
    "encode_verification_record_v1",
    "load_verification_record_v1",
    "write_verification_record_v1",
]
