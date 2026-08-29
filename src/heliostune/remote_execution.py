"""Strict request, result, journal, and receipt contracts for remote execution."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from heliostune.artifacts import strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.hardware import expectation_for_gpu, validate_hardware
from heliostune.local_executor import LocalExecutionResult
from heliostune.schema import HardwareProfile
from heliostune.validation import exact_bool, exact_fields, exact_int, nonblank_string

INTENT_SCHEMA = "heliostune.remote-intent/1"
REQUEST_SCHEMA = "heliostune.remote-request/1"
JOURNAL_SCHEMA = "heliostune.remote-journal-record/1"
RESULT_SCHEMA = "heliostune.remote-result-envelope/1"
RECEIPT_SCHEMA = "heliostune.remote-receipt/1"
EXECUTOR_API = "heliostune.modal_fusion_executor/1"
GPU = "H100"
GPU_SELECTOR = "H100!"
SERVER_TIMEOUT_SECONDS = 3600
CLIENT_TIMEOUT_SECONDS = 3660
RECEIPT_ROOT = "receipt.json"
_RENAME_NOREPLACE = 1

ReceiptStatus = Literal["completed", "failed", "aborted", "unresolved"]
JournalState = Literal[
    "intent",
    "spawned",
    "spawn_acknowledgement_lost",
    "retrieval_started",
    "completed",
    "failed",
    "aborted",
    "cancellation_requested",
    "unresolved",
]

RECEIPT_LIMITATIONS = (
    "One client-authorized spawn is recorded; Modal provider physical starts and restarts are unobservable.",
    "The 3600 second timeout applies to each provider execution; total GPU time and cost across provider restarts are unknown and have no stated upper bound.",
    "There is no attestation, publication eligibility, fusion claim, or performance claim beyond returned local observations.",
)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SchemaError(f"value is not canonical strict JSON: {exc}") from exc


def canonical_json_line_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SchemaError(f"value is not canonical strict JSON: {exc}") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object, *, context: str, length: int = 64) -> str:
    result = nonblank_string(value, context=context)
    if len(result) != length or any(character not in "0123456789abcdef" for character in result):
        raise SchemaError(f"{context} must be a {length}-character lowercase hexadecimal digest")
    return result


def _strict_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise SchemaError(f"{context} must be an object")
    result = cast(dict[object, object], value)
    if any(type(key) is not str for key in result):
        raise SchemaError(f"{context} keys must be strings")
    return cast(dict[str, object], result)


@dataclass(frozen=True, slots=True)
class RemoteIntent:
    suite_path: str
    output_path: str
    suite_sha256: str
    plugin_path: str
    plugin_sha256: str
    wheel_filename: str
    wheel_sha256: str
    manifest_sha256: str
    head_commit: str
    source_sha256: str
    schema: str = INTENT_SCHEMA
    gpu: str = GPU
    gpu_selector: str = GPU_SELECTOR
    server_timeout_seconds: int = SERVER_TIMEOUT_SECONDS
    client_timeout_seconds: int = CLIENT_TIMEOUT_SECONDS
    retries: int = 0
    max_containers: int = 1
    single_use_containers: bool = True
    block_network: bool = True
    restrict_modal_access: bool = True
    one_suite_per_call: bool = True
    attestation: str = "none"

    def __post_init__(self) -> None:
        for name in ("suite_path", "output_path", "plugin_path", "wheel_filename"):
            nonblank_string(getattr(self, name), context=f"remote intent {name}")
        for name in (
            "suite_sha256",
            "plugin_sha256",
            "wheel_sha256",
            "manifest_sha256",
            "source_sha256",
        ):
            _digest(getattr(self, name), context=f"remote intent {name}")
        _digest(self.head_commit, context="remote intent head_commit", length=40)
        fixed: dict[str, object] = {
            "schema": INTENT_SCHEMA,
            "gpu": GPU,
            "gpu_selector": GPU_SELECTOR,
            "server_timeout_seconds": SERVER_TIMEOUT_SECONDS,
            "client_timeout_seconds": CLIENT_TIMEOUT_SECONDS,
            "retries": 0,
            "max_containers": 1,
            "single_use_containers": True,
            "block_network": True,
            "restrict_modal_access": True,
            "one_suite_per_call": True,
            "attestation": "none",
        }
        for field, expected in fixed.items():
            if getattr(self, field) != expected:
                raise SchemaError(f"remote intent {field} must be {expected!r}")

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in _INTENT_FIELDS}

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.to_bytes())

    @classmethod
    def from_dict(cls, value: object) -> RemoteIntent:
        data = exact_fields(value, required=_INTENT_FIELDS, context="remote intent")
        return cls(
            suite_path=nonblank_string(data["suite_path"], context="remote intent suite_path"),
            output_path=nonblank_string(data["output_path"], context="remote intent output_path"),
            suite_sha256=_digest(data["suite_sha256"], context="remote intent suite_sha256"),
            plugin_path=nonblank_string(data["plugin_path"], context="remote intent plugin_path"),
            plugin_sha256=_digest(data["plugin_sha256"], context="remote intent plugin_sha256"),
            wheel_filename=nonblank_string(
                data["wheel_filename"], context="remote intent wheel_filename"
            ),
            wheel_sha256=_digest(data["wheel_sha256"], context="remote intent wheel_sha256"),
            manifest_sha256=_digest(
                data["manifest_sha256"], context="remote intent manifest_sha256"
            ),
            head_commit=_digest(
                data["head_commit"], context="remote intent head_commit", length=40
            ),
            source_sha256=_digest(data["source_sha256"], context="remote intent source_sha256"),
            schema=nonblank_string(data["schema"], context="remote intent schema"),
            gpu=nonblank_string(data["gpu"], context="remote intent gpu"),
            gpu_selector=nonblank_string(
                data["gpu_selector"], context="remote intent gpu_selector"
            ),
            server_timeout_seconds=exact_int(
                data["server_timeout_seconds"],
                context="remote intent server_timeout_seconds",
                minimum=1,
            ),
            client_timeout_seconds=exact_int(
                data["client_timeout_seconds"],
                context="remote intent client_timeout_seconds",
                minimum=1,
            ),
            retries=exact_int(data["retries"], context="remote intent retries", minimum=0),
            max_containers=exact_int(
                data["max_containers"], context="remote intent max_containers", minimum=1
            ),
            single_use_containers=exact_bool(
                data["single_use_containers"], context="remote intent single_use_containers"
            ),
            block_network=exact_bool(data["block_network"], context="remote intent block_network"),
            restrict_modal_access=exact_bool(
                data["restrict_modal_access"], context="remote intent restrict_modal_access"
            ),
            one_suite_per_call=exact_bool(
                data["one_suite_per_call"], context="remote intent one_suite_per_call"
            ),
            attestation=nonblank_string(data["attestation"], context="remote intent attestation"),
        )


_INTENT_FIELDS = (
    "schema",
    "suite_path",
    "output_path",
    "suite_sha256",
    "plugin_path",
    "plugin_sha256",
    "wheel_filename",
    "wheel_sha256",
    "manifest_sha256",
    "head_commit",
    "source_sha256",
    "gpu",
    "gpu_selector",
    "server_timeout_seconds",
    "client_timeout_seconds",
    "retries",
    "max_containers",
    "single_use_containers",
    "block_network",
    "restrict_modal_access",
    "one_suite_per_call",
    "attestation",
)


def encode_remote_request(intent: RemoteIntent, suite_bytes: bytes) -> str:
    if type(suite_bytes) is not bytes:
        raise SchemaError("remote request suite bytes must be bytes")
    if sha256_bytes(suite_bytes) != intent.suite_sha256:
        raise SchemaError("remote request suite bytes do not match the intent digest")
    try:
        suite_utf8 = suite_bytes.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise SchemaError("remote request suite bytes are not canonical UTF-8") from exc
    body: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "intent": intent.to_dict(),
        "suite_sha256": intent.suite_sha256,
        "suite_utf8": suite_utf8,
    }
    request_digest = sha256_bytes(canonical_json_bytes(body))
    body["request_digest"] = request_digest
    return canonical_json_bytes(body).decode("utf-8")


def decode_remote_request(payload: str) -> tuple[RemoteIntent, bytes, str]:
    if type(payload) is not str:
        raise SchemaError("remote request must be a JSON string")
    value = strict_json_loads(payload, source="remote request")
    data = exact_fields(
        value,
        required={"schema", "intent", "suite_sha256", "suite_utf8", "request_digest"},
        context="remote request",
    )
    if data["schema"] != REQUEST_SCHEMA:
        raise SchemaError(f"remote request schema must be {REQUEST_SCHEMA!r}")
    if type(data["suite_utf8"]) is not str:
        raise SchemaError("remote request suite_utf8 must be a string")
    suite_bytes = data["suite_utf8"].encode("utf-8")
    intent = RemoteIntent.from_dict(data["intent"])
    suite_sha256 = _digest(data["suite_sha256"], context="remote request suite_sha256")
    if suite_sha256 != intent.suite_sha256 or sha256_bytes(suite_bytes) != suite_sha256:
        raise SchemaError("remote request suite digest binding failed")
    request_digest = _digest(data["request_digest"], context="remote request request_digest")
    unsigned = dict(data)
    del unsigned["request_digest"]
    if sha256_bytes(canonical_json_bytes(unsigned)) != request_digest:
        raise SchemaError("remote request digest binding failed")
    if canonical_json_bytes(data).decode("utf-8") != payload:
        raise SchemaError("remote request is not in canonical JSON form")
    return intent, suite_bytes, request_digest


_JOURNAL_STATES = {
    "intent",
    "spawned",
    "spawn_acknowledgement_lost",
    "retrieval_started",
    "completed",
    "failed",
    "aborted",
    "cancellation_requested",
    "unresolved",
}
_LEGAL_TRANSITIONS: Mapping[JournalState, frozenset[JournalState]] = {
    "intent": frozenset(
        {"spawned", "spawn_acknowledgement_lost", "cancellation_requested", "unresolved"}
    ),
    "spawned": frozenset({"retrieval_started", "cancellation_requested"}),
    "spawn_acknowledgement_lost": frozenset({"unresolved"}),
    "retrieval_started": frozenset({"completed", "failed", "aborted", "cancellation_requested"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "aborted": frozenset(),
    "cancellation_requested": frozenset({"unresolved"}),
    "unresolved": frozenset(),
}


@dataclass(frozen=True, slots=True)
class RemoteJournalRecord:
    request_digest: str
    sequence: int
    state: JournalState
    call_id: str | None
    detail: str | None = None
    schema: str = JOURNAL_SCHEMA

    def __post_init__(self) -> None:
        _digest(self.request_digest, context="remote journal request_digest")
        exact_int(self.sequence, context="remote journal sequence", minimum=0)
        if self.schema != JOURNAL_SCHEMA or self.state not in _JOURNAL_STATES:
            raise SchemaError("invalid remote journal schema or state")
        if self.call_id is not None:
            nonblank_string(self.call_id, context="remote journal call_id")
        if self.detail is not None:
            nonblank_string(self.detail, context="remote journal detail")
        if self.state in {"intent", "spawn_acknowledgement_lost"} and self.call_id is not None:
            raise SchemaError(f"remote journal state {self.state!r} cannot have a call ID")
        if (
            self.state in {"spawned", "retrieval_started", "completed", "failed", "aborted"}
            and self.call_id is None
        ):
            raise SchemaError(f"remote journal state {self.state!r} requires a call ID")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "request_digest": self.request_digest,
            "sequence": self.sequence,
            "state": self.state,
            "call_id": self.call_id,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, value: object) -> RemoteJournalRecord:
        data = exact_fields(
            value,
            required={"schema", "request_digest", "sequence", "state", "call_id", "detail"},
            context="remote journal record",
        )
        state = nonblank_string(data["state"], context="remote journal state")
        if state not in _JOURNAL_STATES:
            raise SchemaError(f"unknown remote journal state {state!r}")
        return cls(
            request_digest=_digest(data["request_digest"], context="remote journal request_digest"),
            sequence=exact_int(data["sequence"], context="remote journal sequence", minimum=0),
            state=cast(JournalState, state),
            call_id=None
            if data["call_id"] is None
            else nonblank_string(data["call_id"], context="remote journal call_id"),
            detail=None
            if data["detail"] is None
            else nonblank_string(data["detail"], context="remote journal detail"),
            schema=nonblank_string(data["schema"], context="remote journal schema"),
        )


class RemoteJournal:
    """Append-only state machine backed by one retained descriptor."""

    def __init__(self, path: Path, descriptor: int, request_digest: str) -> None:
        self.path = path
        self._descriptor = descriptor
        self.request_digest = _digest(request_digest, context="remote journal request_digest")
        self._sequence = 0
        self._state: JournalState | None = None
        self._call_id: str | None = None

    @classmethod
    def create(cls, path: str | Path, request_digest: str) -> RemoteJournal:
        destination = Path(path).absolute()
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            descriptor = _open_exclusive_at(parent_fd, destination.name)
        finally:
            os.close(parent_fd)
        journal = cls(destination, descriptor, request_digest)
        try:
            journal.append("intent")
        except BaseException:
            journal.close()
            raise
        return journal

    @property
    def state(self) -> JournalState | None:
        return self._state

    def append(
        self, state: JournalState, *, call_id: str | None = None, detail: str | None = None
    ) -> RemoteJournalRecord:
        if self._descriptor < 0:
            raise RuntimeError("remote journal is closed")
        if self._state is None:
            if state != "intent":
                raise SchemaError("remote journal must begin with intent")
        elif state not in _LEGAL_TRANSITIONS[self._state]:
            raise SchemaError(f"illegal remote journal transition {self._state!r} -> {state!r}")
        bound_call_id = self._call_id if call_id is None else call_id
        if self._call_id is not None and bound_call_id != self._call_id:
            raise SchemaError("remote journal call ID cannot change")
        record = RemoteJournalRecord(
            self.request_digest, self._sequence, state, bound_call_id, detail
        )
        _write_all(self._descriptor, canonical_json_line_bytes(record.to_dict()))
        os.fsync(self._descriptor)
        self._sequence += 1
        self._state = state
        self._call_id = bound_call_id
        return record

    def bytes(self) -> bytes:
        if self._descriptor < 0:
            raise RuntimeError("remote journal is closed")
        os.fsync(self._descriptor)
        size = os.fstat(self._descriptor).st_size
        return os.pread(self._descriptor, size, 0)

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> RemoteJournal:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while creating durable remote artifact")
        view = view[written:]


def _open_exclusive_at(parent_fd: int, name: str) -> int:
    return os.open(
        name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd
    )


def remote_artifact_paths(output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    return output.with_name(f"{output.name}.remote-intent.json"), output.with_name(
        f"{output.name}.remote-attempts.jsonl"
    )


def protect_remote_output(output_dir: str | Path) -> tuple[Path, Path, Path]:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ArtifactError("descriptor-pinned remote publication is unsupported")
    output = Path(output_dir).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    intent_path, journal_path = remote_artifact_paths(output)
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name, label in (
            (output.name, "output directory"),
            (intent_path.name, "remote intent"),
            (journal_path.name, "remote journal"),
        ):
            try:
                metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            kind = "symlink" if stat.S_ISLNK(metadata.st_mode) else "existing path"
            raise ArtifactError(f"refusing to overwrite {label} {output.parent / name} ({kind})")
    finally:
        os.close(parent_fd)
    return output, intent_path, journal_path


class RemoteRecords:
    """Pinned intent and journal tombstones for one output parent."""

    def __init__(
        self,
        output: Path,
        parent_fd: int,
        parent_identity: tuple[int, int],
        intent_fd: int,
        intent_identity: tuple[int, int],
        journal_identity: tuple[int, int],
        intent: RemoteIntent,
        journal: RemoteJournal,
    ) -> None:
        self.output = output
        self.parent_fd = parent_fd
        self.parent_identity = parent_identity
        self.intent_fd = intent_fd
        self.intent_identity = intent_identity
        self.journal_identity = journal_identity
        self.intent = intent
        self.journal = journal

    @property
    def intent_path(self) -> Path:
        return remote_artifact_paths(self.output)[0]

    @property
    def journal_path(self) -> Path:
        return remote_artifact_paths(self.output)[1]

    def intent_bytes(self) -> bytes:
        size = os.fstat(self.intent_fd).st_size
        return os.pread(self.intent_fd, size, 0)

    def assert_parent_identity(self) -> None:
        if not _same_directory(os.fstat(self.parent_fd), self.parent_identity):
            raise ArtifactError("remote output parent descriptor identity changed")
        try:
            observed = os.stat(self.output.parent, follow_symlinks=False)
        except OSError as exc:
            raise ArtifactError("remote output parent path disappeared or changed") from exc
        if not _same_directory(observed, self.parent_identity):
            raise ArtifactError("remote output parent identity changed")
        for name, expected, label in (
            (self.intent_path.name, self.intent_identity, "intent"),
            (self.journal_path.name, self.journal_identity, "journal"),
        ):
            try:
                record_stat = os.stat(
                    name,
                    dir_fd=self.parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ArtifactError(f"remote {label} tombstone disappeared or changed") from exc
            if not _same_regular(record_stat, expected):
                raise ArtifactError(f"remote {label} tombstone identity changed")

    def close(self) -> None:
        self.journal.close()
        if self.intent_fd >= 0:
            os.close(self.intent_fd)
            self.intent_fd = -1
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


def create_remote_records(
    output_dir: str | Path, intent: RemoteIntent, request_digest: str
) -> RemoteRecords:
    output, intent_path, journal_path = protect_remote_output(output_dir)
    if Path(intent.output_path).absolute() != output:
        raise SchemaError("remote intent output path does not match the protected output")
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    intent_fd = -1
    journal_fd = -1
    try:
        parent_stat = os.fstat(parent_fd)
        identity = (parent_stat.st_dev, parent_stat.st_ino)
        if not _same_directory(os.stat(output.parent, follow_symlinks=False), identity):
            raise ArtifactError("remote output parent identity changed while opening records")
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ArtifactError(f"remote receipt output directory already exists: {output}")
        intent_fd = _open_exclusive_at(parent_fd, intent_path.name)
        _write_all(intent_fd, intent.to_bytes())
        os.fsync(intent_fd)
        os.fsync(parent_fd)
        journal_fd = _open_exclusive_at(parent_fd, journal_path.name)
        journal = RemoteJournal(journal_path, journal_fd, request_digest)
        journal_fd = -1
        journal.append("intent")
        os.fsync(parent_fd)
        intent_stat = os.fstat(intent_fd)
        journal_stat = os.fstat(journal._descriptor)
        return RemoteRecords(
            output,
            parent_fd,
            identity,
            intent_fd,
            (intent_stat.st_dev, intent_stat.st_ino),
            (journal_stat.st_dev, journal_stat.st_ino),
            intent,
            journal,
        )
    except BaseException:
        if journal_fd >= 0:
            os.close(journal_fd)
        if intent_fd >= 0:
            os.close(intent_fd)
        os.close(parent_fd)
        raise


@dataclass(frozen=True, slots=True)
class RemoteResultEnvelope:
    request_digest: str
    suite_path: str
    suite_sha256: str
    plugin_path: str
    plugin_sha256: str
    wheel_filename: str
    wheel_sha256: str
    manifest_sha256: str
    head_commit: str
    source_sha256: str
    gpu: str
    gpu_selector: str
    hardware: HardwareProfile
    environment: Mapping[str, object]
    result: Mapping[str, object]
    schema: str = RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESULT_SCHEMA:
            raise SchemaError(f"remote result schema must be {RESULT_SCHEMA!r}")
        _digest(self.request_digest, context="remote result request_digest")
        for name in ("suite_path", "plugin_path", "wheel_filename", "gpu", "gpu_selector"):
            nonblank_string(getattr(self, name), context=f"remote result {name}")
        for name in (
            "suite_sha256",
            "plugin_sha256",
            "wheel_sha256",
            "manifest_sha256",
            "source_sha256",
        ):
            _digest(getattr(self, name), context=f"remote result {name}")
        _digest(self.head_commit, context="remote result head_commit", length=40)
        _strict_mapping(self.environment, context="remote result environment")
        _strict_mapping(self.result, context="remote result payload")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "request_digest": self.request_digest,
            "suite_path": self.suite_path,
            "suite_sha256": self.suite_sha256,
            "plugin_path": self.plugin_path,
            "plugin_sha256": self.plugin_sha256,
            "wheel_filename": self.wheel_filename,
            "wheel_sha256": self.wheel_sha256,
            "manifest_sha256": self.manifest_sha256,
            "head_commit": self.head_commit,
            "source_sha256": self.source_sha256,
            "gpu": self.gpu,
            "gpu_selector": self.gpu_selector,
            "hardware": self.hardware.to_dict(),
            "environment": dict(self.environment),
            "result": dict(self.result),
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str) -> RemoteResultEnvelope:
        if type(payload) is not str:
            raise SchemaError("remote function must return a JSON string")
        value = strict_json_loads(payload, source="remote result envelope")
        data = exact_fields(value, required=_RESULT_FIELDS, context="remote result envelope")
        envelope = cls(
            request_digest=_digest(data["request_digest"], context="remote result request_digest"),
            suite_path=nonblank_string(data["suite_path"], context="remote result suite_path"),
            suite_sha256=_digest(data["suite_sha256"], context="remote result suite_sha256"),
            plugin_path=nonblank_string(data["plugin_path"], context="remote result plugin_path"),
            plugin_sha256=_digest(data["plugin_sha256"], context="remote result plugin_sha256"),
            wheel_filename=nonblank_string(
                data["wheel_filename"], context="remote result wheel_filename"
            ),
            wheel_sha256=_digest(data["wheel_sha256"], context="remote result wheel_sha256"),
            manifest_sha256=_digest(
                data["manifest_sha256"], context="remote result manifest_sha256"
            ),
            head_commit=_digest(
                data["head_commit"], context="remote result head_commit", length=40
            ),
            source_sha256=_digest(data["source_sha256"], context="remote result source_sha256"),
            gpu=nonblank_string(data["gpu"], context="remote result gpu"),
            gpu_selector=nonblank_string(
                data["gpu_selector"], context="remote result gpu_selector"
            ),
            hardware=HardwareProfile.from_dict(data["hardware"]),
            environment=_strict_mapping(data["environment"], context="remote result environment"),
            result=_strict_mapping(data["result"], context="remote result payload"),
            schema=nonblank_string(data["schema"], context="remote result schema"),
        )
        if envelope.to_json() != payload:
            raise SchemaError("remote result envelope is not in canonical JSON form")
        return envelope


_RESULT_FIELDS = {
    "schema",
    "request_digest",
    "suite_path",
    "suite_sha256",
    "plugin_path",
    "plugin_sha256",
    "wheel_filename",
    "wheel_sha256",
    "manifest_sha256",
    "head_commit",
    "source_sha256",
    "gpu",
    "gpu_selector",
    "hardware",
    "environment",
    "result",
}


def validate_remote_result(
    payload: str, *, intent: RemoteIntent, request_digest: str, verified_suite_bytes: bytes
) -> tuple[RemoteResultEnvelope, LocalExecutionResult]:
    envelope = RemoteResultEnvelope.from_json(payload)
    expected = {
        "request_digest": request_digest,
        "suite_path": intent.suite_path,
        "suite_sha256": intent.suite_sha256,
        "plugin_path": intent.plugin_path,
        "plugin_sha256": intent.plugin_sha256,
        "wheel_filename": intent.wheel_filename,
        "wheel_sha256": intent.wheel_sha256,
        "manifest_sha256": intent.manifest_sha256,
        "head_commit": intent.head_commit,
        "source_sha256": intent.source_sha256,
        "gpu": intent.gpu,
        "gpu_selector": intent.gpu_selector,
    }
    for field, expected_value in expected.items():
        if getattr(envelope, field) != expected_value:
            raise SchemaError(f"remote result {field} does not match the request intent")
    if sha256_bytes(verified_suite_bytes) != intent.suite_sha256:
        raise SchemaError("verified local suite bytes no longer match the request intent")
    validate_hardware(envelope.hardware, expectation_for_gpu(GPU))
    embedded_environment = _strict_mapping(
        envelope.result.get("environment"), context="remote result embedded environment"
    )
    if dict(envelope.environment) != dict(embedded_environment):
        raise SchemaError("remote result outer and embedded environments differ")
    serialized_path = nonblank_string(
        envelope.result.get("verified_suite_path"), context="remote result verified_suite_path"
    )
    if serialized_path != intent.suite_path:
        raise SchemaError("remote result verified_suite_path does not match the request intent")
    local_result = LocalExecutionResult.from_dict(
        envelope.result,
        verified_suite_path=intent.suite_path,
        verified_suite_sha256=intent.suite_sha256,
        verified_suite_bytes=verified_suite_bytes,
    )
    capability = local_result.capability
    hardware_evidence = {
        "device_name": capability.device_name,
        "compute_capability": capability.compute_capability,
        "cuda_version": capability.cuda_version,
        "torch_version": capability.torch_version,
    }
    for field, evidence in hardware_evidence.items():
        if (capability.available or evidence is not None) and getattr(
            envelope.hardware, field
        ) != evidence:
            raise SchemaError(f"remote result hardware differs from capability field {field}")
    return envelope, local_result


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object, *, context: str) -> ArtifactBinding:
        data = exact_fields(value, required={"path", "bytes", "sha256"}, context=context)
        path = nonblank_string(data["path"], context=f"{context} path")
        if Path(path).name != path or path in {".", ".."}:
            raise SchemaError(f"{context} path must be one plain filename")
        return cls(
            path,
            exact_int(data["bytes"], context=f"{context} bytes", minimum=0),
            _digest(data["sha256"], context=f"{context} sha256"),
        )


_RECEIPT_FIELDS = {
    "schema",
    "receipt_id",
    "status",
    "intent",
    "journal",
    "result",
    "bindings",
    "provider_attempts_observable",
    "provider_physical_attempts",
    "client_spawn_count",
    "per_execution_timeout_s",
    "total_gpu_seconds_upper_bound",
    "actual_cost_usd",
    "publication_eligible",
    "attestation",
    "limitations",
}


@dataclass(frozen=True, slots=True)
class RemoteReceiptV1:
    receipt_id: str
    status: ReceiptStatus
    intent: ArtifactBinding
    journal: ArtifactBinding
    result: ArtifactBinding | None
    bindings: Mapping[str, object]
    client_spawn_count: int
    limitations: tuple[str, ...]
    schema: str = RECEIPT_SCHEMA
    provider_attempts_observable: bool = False
    provider_physical_attempts: None = None
    per_execution_timeout_s: int = SERVER_TIMEOUT_SECONDS
    total_gpu_seconds_upper_bound: None = None
    actual_cost_usd: None = None
    publication_eligible: bool = False
    attestation: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "status": self.status,
            "intent": self.intent.to_dict(),
            "journal": self.journal.to_dict(),
            "result": None if self.result is None else self.result.to_dict(),
            "bindings": dict(self.bindings),
            "provider_attempts_observable": self.provider_attempts_observable,
            "provider_physical_attempts": self.provider_physical_attempts,
            "client_spawn_count": self.client_spawn_count,
            "per_execution_timeout_s": self.per_execution_timeout_s,
            "total_gpu_seconds_upper_bound": self.total_gpu_seconds_upper_bound,
            "actual_cost_usd": self.actual_cost_usd,
            "publication_eligible": self.publication_eligible,
            "attestation": self.attestation,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, value: object) -> RemoteReceiptV1:
        data = exact_fields(value, required=_RECEIPT_FIELDS, context="remote receipt root")
        status_value = nonblank_string(data["status"], context="remote receipt status")
        if status_value not in {"completed", "failed", "aborted", "unresolved"}:
            raise SchemaError(f"unknown remote receipt status {status_value!r}")
        limitations_value = data["limitations"]
        if type(limitations_value) is not list or any(
            type(item) is not str for item in limitations_value
        ):
            raise SchemaError("remote receipt limitations must be an array of strings")
        client_spawn_count = exact_int(
            data["client_spawn_count"],
            context="remote receipt client_spawn_count",
            minimum=0,
        )
        if client_spawn_count > 1:
            raise SchemaError("remote receipt client_spawn_count must be zero or one")
        for null_field in (
            "provider_physical_attempts",
            "total_gpu_seconds_upper_bound",
            "actual_cost_usd",
        ):
            if data[null_field] is not None:
                raise SchemaError(f"remote receipt {null_field} must be null")
        receipt = cls(
            receipt_id=_digest(data["receipt_id"], context="remote receipt receipt_id"),
            status=cast(ReceiptStatus, status_value),
            intent=ArtifactBinding.from_dict(data["intent"], context="remote receipt intent"),
            journal=ArtifactBinding.from_dict(data["journal"], context="remote receipt journal"),
            result=None
            if data["result"] is None
            else ArtifactBinding.from_dict(data["result"], context="remote receipt result"),
            bindings=_strict_mapping(data["bindings"], context="remote receipt bindings"),
            client_spawn_count=client_spawn_count,
            limitations=tuple(cast(list[str], limitations_value)),
            schema=nonblank_string(data["schema"], context="remote receipt schema"),
            provider_attempts_observable=exact_bool(
                data["provider_attempts_observable"],
                context="remote receipt provider_attempts_observable",
            ),
            provider_physical_attempts=None,
            per_execution_timeout_s=exact_int(
                data["per_execution_timeout_s"],
                context="remote receipt per_execution_timeout_s",
                minimum=1,
            ),
            total_gpu_seconds_upper_bound=None,
            actual_cost_usd=None,
            publication_eligible=exact_bool(
                data["publication_eligible"], context="remote receipt publication_eligible"
            ),
            attestation=nonblank_string(data["attestation"], context="remote receipt attestation"),
        )
        fixed = {
            "schema": RECEIPT_SCHEMA,
            "provider_attempts_observable": False,
            "provider_physical_attempts": None,
            "per_execution_timeout_s": SERVER_TIMEOUT_SECONDS,
            "total_gpu_seconds_upper_bound": None,
            "actual_cost_usd": None,
            "publication_eligible": False,
            "attestation": "none",
            "limitations": RECEIPT_LIMITATIONS,
        }
        for field, expected in fixed.items():
            if getattr(receipt, field) != expected:
                raise SchemaError(f"remote receipt {field} must be {expected!r}")
        return receipt


@dataclass(frozen=True, slots=True)
class VerifiedRemoteReceipt:
    receipt: RemoteReceiptV1
    intent: RemoteIntent
    journal: tuple[RemoteJournalRecord, ...]
    envelope: RemoteResultEnvelope | None
    result: LocalExecutionResult | None
    root_path: Path


def _binding(path: str, payload: bytes) -> ArtifactBinding:
    return ArtifactBinding(path, len(payload), sha256_bytes(payload))


def _receipt_bindings(
    intent: RemoteIntent,
    request_digest: str,
    suite: ArtifactBinding,
    plugin: ArtifactBinding,
    manifest: ArtifactBinding,
) -> dict[str, object]:
    return {
        "executor_api": EXECUTOR_API,
        "request_digest": request_digest,
        "suite": {"logical_path": intent.suite_path, **suite.to_dict()},
        "plugin": {"logical_path": intent.plugin_path, **plugin.to_dict()},
        "wheel": {
            "filename": intent.wheel_filename,
            "sha256": intent.wheel_sha256,
            "manifest": manifest.to_dict(),
            "head_commit": intent.head_commit,
            "source_sha256": intent.source_sha256,
        },
        "gpu": intent.gpu,
        "gpu_selector": intent.gpu_selector,
        "retries": intent.retries,
        "max_containers": intent.max_containers,
        "single_use_containers": intent.single_use_containers,
        "block_network": intent.block_network,
        "restrict_modal_access": intent.restrict_modal_access,
        "one_suite_per_call": intent.one_suite_per_call,
    }


def _parse_journal(payload: bytes, request_digest: str) -> tuple[RemoteJournalRecord, ...]:
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines(keepends=True)
    except UnicodeError as exc:
        raise SchemaError("remote receipt journal must be UTF-8") from exc
    if not lines or any(not line.endswith("\n") for line in lines):
        raise SchemaError("remote receipt journal must be nonempty complete JSONL")
    records: list[RemoteJournalRecord] = []
    prior: JournalState | None = None
    call_id: str | None = None
    for sequence, line in enumerate(lines):
        value = strict_json_loads(line, source="remote receipt journal")
        record = RemoteJournalRecord.from_dict(value)
        if canonical_json_line_bytes(record.to_dict()).decode("utf-8") != line:
            raise SchemaError("remote receipt journal record is not canonical JSONL")
        if record.sequence != sequence or record.request_digest != request_digest:
            raise SchemaError("remote receipt journal sequence or request binding differs")
        if prior is None:
            if record.state != "intent":
                raise SchemaError("remote receipt journal must begin with intent")
        elif record.state not in _LEGAL_TRANSITIONS[prior]:
            raise SchemaError(
                f"illegal remote receipt journal transition {prior!r} -> {record.state!r}"
            )
        if call_id is not None and record.call_id != call_id:
            raise SchemaError("remote receipt journal call ID changed")
        if record.call_id is not None:
            call_id = record.call_id
        records.append(record)
        prior = record.state
    return tuple(records)


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactError(f"remote receipt artifact is not a regular file: {name}")
        remaining = metadata.st_size
        offset = 0
        blocks: list[bytes] = []
        while remaining:
            block = os.pread(descriptor, min(remaining, 1024 * 1024), offset)
            if not block:
                raise ArtifactError(f"short read from remote receipt artifact: {name}")
            blocks.append(block)
            offset += len(block)
            remaining -= len(block)
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def _verify_receipt_fd(directory_fd: int, root_path: Path) -> VerifiedRemoteReceipt:
    root_payload = _read_regular_at(directory_fd, RECEIPT_ROOT)
    try:
        root_text = root_payload.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise SchemaError("remote receipt root must be UTF-8") from exc
    value = strict_json_loads(root_text, source="remote receipt root")
    receipt = RemoteReceiptV1.from_dict(value)
    if canonical_json_bytes(receipt.to_dict()) != root_payload:
        raise SchemaError("remote receipt root is not canonical JSON")
    expected_paths = {RECEIPT_ROOT, receipt.intent.path, receipt.journal.path}
    if receipt.result is not None:
        expected_paths.add(receipt.result.path)
    bindings = exact_fields(
        receipt.bindings,
        required={
            "executor_api",
            "request_digest",
            "suite",
            "plugin",
            "wheel",
            "gpu",
            "gpu_selector",
            "retries",
            "max_containers",
            "single_use_containers",
            "block_network",
            "restrict_modal_access",
            "one_suite_per_call",
        },
        context="remote receipt bindings",
    )
    suite_data = exact_fields(
        bindings["suite"],
        required={"logical_path", "path", "bytes", "sha256"},
        context="remote receipt suite binding",
    )
    plugin_data = exact_fields(
        bindings["plugin"],
        required={"logical_path", "path", "bytes", "sha256"},
        context="remote receipt plugin binding",
    )
    wheel_data = exact_fields(
        bindings["wheel"],
        required={"filename", "sha256", "manifest", "head_commit", "source_sha256"},
        context="remote receipt wheel binding",
    )
    suite_binding = ArtifactBinding.from_dict(
        {key: suite_data[key] for key in ("path", "bytes", "sha256")},
        context="remote receipt suite artifact",
    )
    plugin_binding = ArtifactBinding.from_dict(
        {key: plugin_data[key] for key in ("path", "bytes", "sha256")},
        context="remote receipt plugin artifact",
    )
    manifest_binding = ArtifactBinding.from_dict(
        wheel_data["manifest"], context="remote receipt manifest artifact"
    )
    expected_paths.update({suite_binding.path, plugin_binding.path, manifest_binding.path})
    actual_paths = set(os.listdir(directory_fd))
    if actual_paths != expected_paths:
        raise ArtifactError(
            f"remote receipt directory inventory differs: missing={sorted(expected_paths - actual_paths)}, extra={sorted(actual_paths - expected_paths)}"
        )

    def checked(binding: ArtifactBinding) -> bytes:
        payload = _read_regular_at(directory_fd, binding.path)
        if len(payload) != binding.bytes or sha256_bytes(payload) != binding.sha256:
            raise ArtifactError(f"remote receipt artifact digest/size mismatch: {binding.path}")
        return payload

    intent_payload = checked(receipt.intent)
    journal_payload = checked(receipt.journal)
    suite_payload = checked(suite_binding)
    plugin_payload = checked(plugin_binding)
    manifest_payload = checked(manifest_binding)
    result_payload = None if receipt.result is None else checked(receipt.result)
    intent_value = strict_json_loads(
        intent_payload.decode("utf-8", errors="strict"), source="remote receipt intent"
    )
    intent = RemoteIntent.from_dict(intent_value)
    if canonical_json_bytes(intent.to_dict()) != intent_payload:
        raise SchemaError("remote receipt intent is not canonical JSON")
    if Path(intent.output_path).absolute() != root_path.parent.absolute():
        raise SchemaError("remote receipt location differs from intent output_path")
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_payload))
    if receipt.receipt_id != request_digest or bindings["request_digest"] != request_digest:
        raise SchemaError("remote receipt ID/request digest binding differs")
    if (
        sha256_bytes(plugin_payload) != intent.plugin_sha256
        or sha256_bytes(manifest_payload) != intent.manifest_sha256
    ):
        raise SchemaError("remote receipt plugin or manifest binding differs from intent")
    if (
        suite_data["logical_path"] != intent.suite_path
        or plugin_data["logical_path"] != intent.plugin_path
    ):
        raise SchemaError("remote receipt logical artifact paths differ from intent")
    expected_bindings = _receipt_bindings(
        intent, request_digest, suite_binding, plugin_binding, manifest_binding
    )
    if dict(receipt.bindings) != expected_bindings:
        raise SchemaError("remote receipt bindings differ from exact intent policy")
    records = _parse_journal(journal_payload, request_digest)
    terminal = records[-1].state
    spawned = any(
        record.state
        in {
            "spawned",
            "spawn_acknowledgement_lost",
            "cancellation_requested",
        }
        for record in records
    )
    if receipt.client_spawn_count != int(spawned):
        raise SchemaError("remote receipt client spawn count differs from journal")
    if terminal != receipt.status:
        raise SchemaError("remote receipt status differs from journal terminal state")
    envelope: RemoteResultEnvelope | None = None
    result: LocalExecutionResult | None = None
    if receipt.status == "unresolved":
        if result_payload is not None:
            raise SchemaError("unresolved remote receipt cannot contain a validated result")
    else:
        if result_payload is None:
            raise SchemaError("terminal remote receipt lacks its result envelope")
        try:
            result_text = result_payload.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise SchemaError("remote result envelope must be UTF-8") from exc
        envelope, result = validate_remote_result(
            result_text,
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_payload,
        )
        if result.outcome != receipt.status:
            raise SchemaError("remote receipt status differs from LocalExecutionResult outcome")
    return VerifiedRemoteReceipt(receipt, intent, records, envelope, result, root_path)


def verify_remote_receipt(path: str | Path) -> VerifiedRemoteReceipt:
    supplied = Path(path).absolute()
    directory = supplied if supplied.name != RECEIPT_ROOT else supplied.parent
    root_path = directory / RECEIPT_ROOT
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        identity = (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
        if not _same_directory(os.stat(directory, follow_symlinks=False), identity):
            raise ArtifactError("remote receipt directory identity changed while opening")
        verified = _verify_receipt_fd(descriptor, root_path)
        if not _same_directory(os.stat(directory, follow_symlinks=False), identity):
            raise ArtifactError("remote receipt directory identity changed during verification")
        return verified
    finally:
        os.close(descriptor)


def _same_directory(value: os.stat_result, expected: tuple[int, int]) -> bool:
    return stat.S_ISDIR(value.st_mode) and (value.st_dev, value.st_ino) == expected


def _same_regular(value: os.stat_result, expected: tuple[int, int]) -> bool:
    return stat.S_ISREG(value.st_mode) and (value.st_dev, value.st_ino) == expected


def _write_file_at(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = _open_exclusive_at(directory_fd, name)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _rename_directory_noreplace(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise ArtifactError("atomic no-replace remote receipt publication is unsupported") from exc
    result = renameat2(
        parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), _RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _remove_staging(parent_fd: int, staging_fd: int, name: str) -> None:
    for entry in os.listdir(staging_fd):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(entry, dir_fd=staging_fd)
    with contextlib.suppress(FileNotFoundError):
        os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def write_remote_receipt(
    records: RemoteRecords,
    *,
    status: ReceiptStatus,
    request_digest: str,
    suite_bytes: bytes,
    plugin_bytes: bytes,
    manifest_bytes: bytes,
    result_payload: str | None,
    client_spawn_count: int,
) -> VerifiedRemoteReceipt:
    """Publish a descriptor-pinned, root-last receipt from retained bytes only."""
    records.assert_parent_identity()
    intent = records.intent
    if records.intent_bytes() != intent.to_bytes():
        raise ArtifactError("retained remote intent bytes changed")
    result_bytes = None if result_payload is None else result_payload.encode("utf-8")
    if status == "unresolved" and result_bytes is not None:
        raise SchemaError("unresolved receipt cannot publish a validated result")
    if status != "unresolved" and result_bytes is None:
        raise SchemaError("terminal receipt requires a result envelope")
    if client_spawn_count not in {0, 1}:
        raise SchemaError("remote receipt client spawn count must be zero or one")
    suite_binding = _binding("suite.json", suite_bytes)
    plugin_binding = _binding("plugin.json", plugin_bytes)
    manifest_binding = _binding("wheel.manifest.json", manifest_bytes)
    if (
        suite_binding.sha256 != intent.suite_sha256
        or plugin_binding.sha256 != intent.plugin_sha256
        or manifest_binding.sha256 != intent.manifest_sha256
    ):
        raise SchemaError("retained suite/plugin/manifest bytes differ from intent")
    journal_bytes = records.journal.bytes()
    intent_binding = _binding("intent.json", records.intent_bytes())
    journal_binding = _binding("journal.jsonl", journal_bytes)
    result_binding = None if result_bytes is None else _binding("result.json", result_bytes)
    receipt = RemoteReceiptV1(
        receipt_id=request_digest,
        status=status,
        intent=intent_binding,
        journal=journal_binding,
        result=result_binding,
        bindings=_receipt_bindings(
            intent, request_digest, suite_binding, plugin_binding, manifest_binding
        ),
        client_spawn_count=client_spawn_count,
        limitations=RECEIPT_LIMITATIONS,
    )
    root_payload = canonical_json_bytes(receipt.to_dict())
    payloads = [
        (intent_binding.path, records.intent_bytes()),
        (journal_binding.path, journal_bytes),
        (suite_binding.path, suite_bytes),
        (plugin_binding.path, plugin_bytes),
        (manifest_binding.path, manifest_bytes),
    ]
    if result_binding is not None and result_bytes is not None:
        payloads.append((result_binding.path, result_bytes))
    staging_name = ""
    staging_fd = -1
    published = False
    renamed = False
    try:
        for _ in range(32):
            staging_name = f".heliostune-remote-receipt-{secrets.token_hex(16)}"
            try:
                os.mkdir(staging_name, 0o700, dir_fd=records.parent_fd)
            except FileExistsError:
                continue
            break
        else:
            raise ArtifactError("cannot allocate a unique remote receipt staging directory")
        staging_fd = os.open(
            staging_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=records.parent_fd
        )
        os.fchmod(staging_fd, 0o700)
        staging_stat = os.fstat(staging_fd)
        staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
        if not _same_directory(
            os.stat(staging_name, dir_fd=records.parent_fd, follow_symlinks=False),
            staging_identity,
        ):
            raise ArtifactError("remote receipt staging directory identity changed while opening")
        if stat.S_IMODE(staging_stat.st_mode) != 0o700:
            raise ArtifactError("remote receipt staging directory does not have mode 0700")
        for name, payload in payloads:
            _write_file_at(staging_fd, name, payload)
        _write_file_at(staging_fd, RECEIPT_ROOT, root_payload)
        verified = _verify_receipt_fd(staging_fd, records.output / RECEIPT_ROOT)
        records.assert_parent_identity()
        if not _same_directory(os.fstat(staging_fd), staging_identity) or not _same_directory(
            os.stat(staging_name, dir_fd=records.parent_fd, follow_symlinks=False),
            staging_identity,
        ):
            raise ArtifactError(
                "remote receipt staging directory identity changed before publication"
            )
        try:
            os.stat(records.output.name, dir_fd=records.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ArtifactError(f"remote receipt output directory already exists: {records.output}")
        _rename_directory_noreplace(records.parent_fd, staging_name, records.output.name)
        renamed = True
        os.fsync(records.parent_fd)
        records.assert_parent_identity()
        if not _same_directory(
            os.stat(records.output.name, dir_fd=records.parent_fd, follow_symlinks=False),
            staging_identity,
        ):
            raise ArtifactError("published remote receipt identity changed during publication")
        published = True
        return verified
    except OSError as exc:
        raise ArtifactError(f"cannot publish remote receipt {records.output}: {exc}") from exc
    finally:
        if staging_fd >= 0:
            if not published and not renamed:
                with contextlib.suppress(OSError):
                    _remove_staging(records.parent_fd, staging_fd, staging_name)
            os.close(staging_fd)
