"""Modal-free paid-call planning, journaling, retrieval, and commit logic."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, cast

from heliostune.artifacts import (
    read_json,
    strict_json_dumps,
    strict_json_loads,
    write_json_atomic,
    write_measurements_atomic,
)
from heliostune.errors import ArtifactError, ProtocolError, SchemaError
from heliostune.protocol import v3_seed
from heliostune.schema import HardwareProfile, Measurement
from heliostune.validation import exact_fields, exact_int, exact_object, nonblank_string

AttemptStatus = Literal["spawned", "completed", "failed"]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise SchemaError(f"{context} must be a lowercase SHA-256 digest")
    return result


def _validate_unique_strings(values: object, *, context: str) -> tuple[str, ...]:
    if type(values) not in {list, tuple}:
        raise SchemaError(f"{context} must be a string sequence")
    sequence = cast(list[object] | tuple[object, ...], values)
    result = tuple(
        nonblank_string(value, context=f"{context}[{index}]")
        for index, value in enumerate(sequence)
    )
    if not result:
        raise SchemaError(f"{context} must not be empty")
    if len(set(result)) != len(result):
        raise SchemaError(f"{context} must not contain duplicates")
    return result


def _validate_banks(values: object, *, context: str) -> tuple[int, ...]:
    if type(values) not in {list, tuple}:
        raise SchemaError(f"{context} must be an integer sequence")
    sequence = cast(list[object] | tuple[object, ...], values)
    result = tuple(
        exact_int(value, context=f"{context}[{index}]", minimum=0)
        for index, value in enumerate(sequence)
    )
    if not result:
        raise SchemaError(f"{context} must not be empty")
    if len(set(result)) != len(result):
        raise SchemaError(f"{context} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class CollectionRequest:
    """Exact logical matrix and benchmark protocol requested from Modal."""

    gpus: tuple[str, ...]
    banks: tuple[int, ...]
    workload_keys: tuple[str, ...]
    config_keys: tuple[str, ...]
    warmup_ms: float
    repetition_ms: float
    pilot: bool = False
    seed_protocol: str = "legacy-bank"

    def __post_init__(self) -> None:
        for name in ("gpus", "banks", "workload_keys", "config_keys"):
            if type(getattr(self, name)) is not tuple:
                raise SchemaError(f"collection {name} must be an immutable tuple")
        _validate_unique_strings(self.gpus, context="collection gpus")
        _validate_banks(self.banks, context="collection banks")
        _validate_unique_strings(self.workload_keys, context="collection workload_keys")
        _validate_unique_strings(self.config_keys, context="collection config_keys")
        if type(self.warmup_ms) not in {int, float} or not 0 <= self.warmup_ms < float("inf"):
            raise SchemaError("collection warmup_ms must be finite and non-negative")
        if type(self.repetition_ms) not in {int, float} or not 0 < self.repetition_ms < float(
            "inf"
        ):
            raise SchemaError("collection repetition_ms must be finite and positive")
        if type(self.pilot) is not bool:
            raise SchemaError("collection pilot must be a boolean")
        if self.seed_protocol not in {"legacy-bank", "parhelion-v3"}:
            raise SchemaError("collection seed_protocol must be legacy-bank or parhelion-v3")

    def to_dict(self) -> dict[str, object]:
        return {
            "gpus": list(self.gpus),
            "banks": list(self.banks),
            "workload_keys": list(self.workload_keys),
            "config_keys": list(self.config_keys),
            "warmup_ms": self.warmup_ms,
            "repetition_ms": self.repetition_ms,
            "pilot": self.pilot,
            "seed_protocol": self.seed_protocol,
        }

    @property
    def sha256(self) -> str:
        return _sha256_bytes(strict_json_dumps(self.to_dict(), compact=True).encode())


@dataclass(frozen=True, slots=True)
class CollectionBinding:
    """Immutable source and package digests shared by every remote call."""

    protocol_sha256: str
    config_manifest_sha256: str
    wheel_sha256: str
    head_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "config_manifest_sha256",
            "wheel_sha256",
            "head_sha256",
        ):
            _validate_sha256(getattr(self, name), context=f"collection {name}")

    def to_dict(self) -> dict[str, str]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "config_manifest_sha256": self.config_manifest_sha256,
            "wheel_sha256": self.wheel_sha256,
            "head_sha256": self.head_sha256,
        }


@dataclass(frozen=True, slots=True)
class CallPlanItem:
    """One canonical paid call and its deterministic collector order."""

    gpu: str
    bank: int
    seed: int
    workload_order: tuple[str, ...]
    config_orders: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        nonblank_string(self.gpu, context="call plan gpu")
        exact_int(self.bank, context="call plan bank", minimum=0)
        exact_int(self.seed, context="call plan seed", minimum=0)
        _validate_unique_strings(self.workload_order, context="call plan workload_order")
        if type(self.config_orders) is not tuple:
            raise SchemaError("call plan config_orders must be an immutable tuple")
        if tuple(workload for workload, _configs in self.config_orders) != self.workload_order:
            raise SchemaError("call plan config orders must follow the workload order")
        for workload, configs in self.config_orders:
            nonblank_string(workload, context="call plan config workload")
            _validate_unique_strings(configs, context=f"call plan configs for {workload}")

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu": self.gpu,
            "bank": self.bank,
            "seed": self.seed,
            "workload_order": list(self.workload_order),
            "config_orders": [
                {"workload": workload, "config_keys": list(configs)}
                for workload, configs in self.config_orders
            ],
        }


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One append-only state transition for a remote FunctionCall."""

    request_sha256: str
    protocol_sha256: str
    config_manifest_sha256: str
    wheel_sha256: str
    head_sha256: str
    gpu: str
    bank: int
    call_id: str
    status: AttemptStatus
    timestamp_utc: str
    chunk_sha256: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "protocol_sha256",
            "config_manifest_sha256",
            "wheel_sha256",
            "head_sha256",
        ):
            _validate_sha256(getattr(self, name), context=f"attempt {name}")
        nonblank_string(self.gpu, context="attempt gpu")
        exact_int(self.bank, context="attempt bank", minimum=0)
        nonblank_string(self.call_id, context="attempt call_id")
        nonblank_string(self.timestamp_utc, context="attempt timestamp_utc")
        if self.status not in {"spawned", "completed", "failed"}:
            raise SchemaError(f"unknown attempt status {self.status!r}")
        if self.status == "spawned":
            if self.chunk_sha256 is not None or self.error is not None:
                raise SchemaError("spawned attempts cannot contain a chunk digest or error")
        elif self.status == "completed":
            _validate_sha256(self.chunk_sha256, context="attempt chunk_sha256")
            if self.error is not None:
                raise SchemaError("completed attempts cannot contain an error")
        else:
            if self.chunk_sha256 is not None:
                raise SchemaError("failed attempts cannot contain a chunk digest")
            nonblank_string(self.error, context="attempt error")

    @property
    def key(self) -> tuple[str, int]:
        return self.gpu, self.bank

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "protocol_sha256": self.protocol_sha256,
            "config_manifest_sha256": self.config_manifest_sha256,
            "wheel_sha256": self.wheel_sha256,
            "head_sha256": self.head_sha256,
            "gpu": self.gpu,
            "bank": self.bank,
            "call_id": self.call_id,
            "status": self.status,
            "timestamp_utc": self.timestamp_utc,
            "chunk_sha256": self.chunk_sha256,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: object) -> AttemptRecord:
        data = exact_fields(
            value,
            required=(
                "schema_version",
                "request_sha256",
                "protocol_sha256",
                "config_manifest_sha256",
                "wheel_sha256",
                "head_sha256",
                "gpu",
                "bank",
                "call_id",
                "status",
                "timestamp_utc",
                "chunk_sha256",
                "error",
            ),
            context="attempt record",
        )
        if exact_int(data["schema_version"], context="attempt schema_version") != 1:
            raise SchemaError("unsupported attempt record schema version")
        status = nonblank_string(data["status"], context="attempt status")
        if status not in {"spawned", "completed", "failed"}:
            raise SchemaError(f"unknown attempt status {status!r}")
        return cls(
            request_sha256=_validate_sha256(
                data["request_sha256"], context="attempt request_sha256"
            ),
            protocol_sha256=_validate_sha256(
                data["protocol_sha256"], context="attempt protocol_sha256"
            ),
            config_manifest_sha256=_validate_sha256(
                data["config_manifest_sha256"], context="attempt config_manifest_sha256"
            ),
            wheel_sha256=_validate_sha256(data["wheel_sha256"], context="attempt wheel_sha256"),
            head_sha256=_validate_sha256(data["head_sha256"], context="attempt head_sha256"),
            gpu=nonblank_string(data["gpu"], context="attempt gpu"),
            bank=exact_int(data["bank"], context="attempt bank", minimum=0),
            call_id=nonblank_string(data["call_id"], context="attempt call_id"),
            status=cast(AttemptStatus, status),
            timestamp_utc=nonblank_string(data["timestamp_utc"], context="attempt timestamp_utc"),
            chunk_sha256=(
                None
                if data["chunk_sha256"] is None
                else _validate_sha256(data["chunk_sha256"], context="attempt chunk_sha256")
            ),
            error=(
                None
                if data["error"] is None
                else nonblank_string(data["error"], context="attempt error")
            ),
        )


@dataclass(frozen=True, slots=True)
class CollectionChunk:
    """One retrieved remote result bound to its call and content digest."""

    gpu: str
    bank: int
    call_id: str
    measurements: tuple[Measurement, ...]
    sha256: str

    def __post_init__(self) -> None:
        nonblank_string(self.gpu, context="collection chunk gpu")
        exact_int(self.bank, context="collection chunk bank", minimum=0)
        nonblank_string(self.call_id, context="collection chunk call_id")
        if type(self.measurements) is not tuple or not self.measurements:
            raise SchemaError("collection chunk measurements must be a nonempty tuple")
        if any(type(row) is not Measurement for row in self.measurements):
            raise SchemaError("collection chunk contains a non-measurement value")
        _validate_sha256(self.sha256, context="collection chunk sha256")

    @classmethod
    def from_measurements(
        cls,
        gpu: str,
        bank: int,
        call_id: str,
        measurements: Sequence[Measurement],
    ) -> CollectionChunk:
        rows = tuple(measurements)
        if not rows:
            raise ProtocolError(f"empty collection chunk for {gpu}/bank-{bank}")
        for row in rows:
            if row.hardware.gpu != gpu or row.bank != bank:
                raise ProtocolError(
                    f"remote chunk identity mismatch for {gpu}/bank-{bank}: "
                    f"got {row.hardware.gpu}/bank-{row.bank}"
                )
        payload = "".join(
            strict_json_dumps(row.to_dict(), compact=True) + "\n"
            for row in sorted(rows, key=measurement_sort_key)
        ).encode()
        return cls(gpu, bank, call_id, rows, _sha256_bytes(payload))


@dataclass(frozen=True, slots=True)
class CollectionManifest:
    """Final local data/sidecar binding after every remote call completes."""

    request: CollectionRequest
    binding: CollectionBinding
    data_path: str
    data_sha256: str
    rows: int
    failures: int
    attempt_journal_path: str
    attempt_journal_sha256: str
    plan: tuple[CallPlanItem, ...]
    hardware: tuple[Mapping[str, object], ...]
    calls: tuple[AttemptRecord, ...]
    facts: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            type(self.request) is not CollectionRequest
            or type(self.binding) is not CollectionBinding
        ):
            raise SchemaError("collection manifest requires frozen request and binding records")
        nonblank_string(self.data_path, context="collection manifest data_path")
        _validate_sha256(self.data_sha256, context="collection manifest data_sha256")
        exact_int(self.rows, context="collection manifest rows", minimum=1)
        failures = exact_int(self.failures, context="collection manifest failures", minimum=0)
        if failures > self.rows:
            raise SchemaError("collection manifest failures cannot exceed rows")
        nonblank_string(
            self.attempt_journal_path,
            context="collection manifest attempt_journal_path",
        )
        _validate_sha256(
            self.attempt_journal_sha256,
            context="collection manifest attempt_journal_sha256",
        )
        if type(self.calls) is not tuple or any(
            type(record) is not AttemptRecord for record in self.calls
        ):
            raise SchemaError("collection manifest calls must be an immutable attempt tuple")
        if type(self.plan) is not tuple or any(
            type(item) is not CallPlanItem for item in self.plan
        ):
            raise SchemaError("collection manifest plan must be an immutable call-plan tuple")
        if type(self.hardware) is not tuple:
            raise SchemaError("collection manifest hardware must be an immutable tuple")
        object.__setattr__(
            self,
            "hardware",
            tuple(MappingProxyType(dict(profile)) for profile in self.hardware),
        )
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request": self.request.to_dict(),
            "request_sha256": self.request.sha256,
            "binding": self.binding.to_dict(),
            "data": {
                "path": self.data_path,
                "sha256": self.data_sha256,
                "rows": self.rows,
                "failures": self.failures,
            },
            "attempt_journal": {
                "path": self.attempt_journal_path,
                "sha256": self.attempt_journal_sha256,
            },
            "call_plan": [item.to_dict() for item in self.plan],
            "hardware": [dict(profile) for profile in self.hardware],
            "calls": [record.to_dict() for record in self.calls],
            "facts": dict(self.facts),
        }


class RemoteCall(Protocol):
    @property
    def object_id(self) -> str: ...

    def get(self) -> object: ...


class AttemptJournal:
    """Strict append-only attempt journal with per-record durability."""

    def __init__(self, path: str | Path, records: Sequence[AttemptRecord] = ()) -> None:
        self.path = Path(path)
        self._records = list(records)
        self._validate_transitions()

    @property
    def records(self) -> tuple[AttemptRecord, ...]:
        return tuple(self._records)

    @classmethod
    def create(cls, path: str | Path) -> AttemptJournal:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except OSError as exc:
            raise ArtifactError(f"cannot create attempt journal {destination}: {exc}") from exc
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(destination.parent)
        return cls(destination)

    @classmethod
    def load(cls, path: str | Path) -> AttemptJournal:
        source = Path(path)
        records: list[AttemptRecord] = []
        try:
            with source.open(encoding="utf-8", newline="") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        raise SchemaError(
                            f"{source}:{line_number}: blank attempt records are not permitted"
                        )
                    decoded = strict_json_loads(
                        line,
                        source=source,
                        line_number=line_number,
                    )
                    try:
                        records.append(AttemptRecord.from_dict(decoded))
                    except SchemaError as exc:
                        raise SchemaError(
                            f"{source}:{line_number}: invalid attempt record: {exc}"
                        ) from exc
        except (OSError, UnicodeError) as exc:
            raise ArtifactError(f"cannot read attempt journal {source}: {exc}") from exc
        return cls(source, records)

    def _validate_transitions(self) -> None:
        states: dict[tuple[str, int], AttemptRecord] = {}
        request_sha256: str | None = None
        binding: tuple[str, str, str, str] | None = None
        for record in self._records:
            if request_sha256 is None:
                request_sha256 = record.request_sha256
                binding = (
                    record.protocol_sha256,
                    record.config_manifest_sha256,
                    record.wheel_sha256,
                    record.head_sha256,
                )
            elif record.request_sha256 != request_sha256 or binding != (
                record.protocol_sha256,
                record.config_manifest_sha256,
                record.wheel_sha256,
                record.head_sha256,
            ):
                raise ProtocolError(
                    "attempt journal contains conflicting request or source digests"
                )
            prior = states.get(record.key)
            if prior is None:
                if record.status != "spawned":
                    raise ProtocolError(f"attempt {record.key} does not begin with spawned")
                states[record.key] = record
            else:
                if prior.status != "spawned" or record.status == "spawned":
                    raise ProtocolError(f"attempt {record.key} has an invalid transition")
                if prior.call_id != record.call_id:
                    raise ProtocolError(f"attempt {record.key} changes FunctionCall ID")
                states[record.key] = record

    def require_binding(
        self,
        request: CollectionRequest,
        binding: CollectionBinding,
    ) -> None:
        if not self._records:
            raise ProtocolError("cannot resume an empty attempt journal")
        first = self._records[0]
        if first.request_sha256 != request.sha256 or any(
            getattr(first, name) != getattr(binding, name)
            for name in (
                "protocol_sha256",
                "config_manifest_sha256",
                "wheel_sha256",
                "head_sha256",
            )
        ):
            raise ProtocolError(
                "resume journal does not match the exact request and source digests"
            )

    def latest_by_key(self) -> Mapping[tuple[str, int], AttemptRecord]:
        latest: dict[tuple[str, int], AttemptRecord] = {}
        for record in self._records:
            latest[record.key] = record
        return MappingProxyType(latest)

    def spawned_by_key(self) -> Mapping[tuple[str, int], AttemptRecord]:
        spawned: dict[tuple[str, int], AttemptRecord] = {}
        for record in self._records:
            if record.status == "spawned":
                spawned[record.key] = record
        return MappingProxyType(spawned)

    def append(self, record: AttemptRecord) -> None:
        candidate = AttemptJournal(self.path, (*self._records, record))
        payload = (strict_json_dumps(record.to_dict(), compact=True) + "\n").encode()
        try:
            with self.path.open("ab") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(self.path.parent)
        except OSError as exc:
            raise ArtifactError(f"cannot append attempt journal {self.path}: {exc}") from exc
        self._records = list(candidate.records)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def attempt_journal_path(output: str | Path) -> Path:
    return Path(f"{output}.attempts.jsonl")


def manifest_path(output: str | Path) -> Path:
    return Path(f"{output}.manifest.json")


def _probe_destination(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".preflight",
    )
    temporary = Path(temporary_name)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def preflight_collection(
    output: str | Path,
    *,
    resume_attempts: str | Path | None = None,
) -> AttemptJournal:
    """Prove data/sidecar writability and create or strict-load the journal."""
    destination = Path(output)
    sidecar = manifest_path(destination)
    if destination.exists() or sidecar.exists():
        if destination.exists() != sidecar.exists():
            raise ArtifactError(f"collection output pair is incomplete: {destination} / {sidecar}")
        sidecar_data = exact_object(read_json(sidecar), context="collection sidecar")
        data_binding = exact_object(
            sidecar_data.get("data"),
            context="collection sidecar data",
        )
        recorded_sha256 = _validate_sha256(
            data_binding.get("sha256"),
            context="collection sidecar data sha256",
        )
        actual_sha256 = sha256_file(destination)
        if actual_sha256 != recorded_sha256:
            raise ArtifactError(
                f"collection output digest mismatch for {destination}: "
                f"sidecar has {recorded_sha256}, data has {actual_sha256}"
            )
        raise ArtifactError(f"collection output already exists and is digest-valid: {destination}")
    _probe_destination(destination)
    _probe_destination(sidecar)
    if resume_attempts is not None:
        return AttemptJournal.load(resume_attempts)
    return AttemptJournal.create(attempt_journal_path(destination))


def build_call_plan(request: CollectionRequest) -> tuple[CallPlanItem, ...]:
    """Return canonical sorted calls and the exact collector shuffle schedule."""
    plan: list[CallPlanItem] = []
    for gpu, bank in sorted((gpu, bank) for gpu in request.gpus for bank in request.banks):
        if request.seed_protocol == "parhelion-v3":
            workload_seed = v3_seed(
                purpose="collector-workload-order",
                gpu=gpu,
                bank=bank,
            )
            workload_randomizer = random.Random(workload_seed)
        else:
            workload_seed = bank
            workload_randomizer = random.Random(bank)
        workload_order = list(request.workload_keys)
        workload_randomizer.shuffle(workload_order)
        config_orders: list[tuple[str, tuple[str, ...]]] = []
        for workload_key in workload_order:
            if request.seed_protocol == "parhelion-v3":
                config_seed = v3_seed(
                    purpose="collector-config-order",
                    gpu=gpu,
                    bank=bank,
                    workload_key=workload_key,
                )
                config_randomizer = random.Random(config_seed)
            else:
                config_randomizer = workload_randomizer
            config_order = list(request.config_keys)
            config_randomizer.shuffle(config_order)
            config_orders.append((workload_key, tuple(config_order)))
        plan.append(
            CallPlanItem(
                gpu=gpu,
                bank=bank,
                seed=workload_seed,
                workload_order=tuple(workload_order),
                config_orders=tuple(config_orders),
            )
        )
    return tuple(plan)


def _timestamp(now: Callable[[], datetime]) -> str:
    value = now()
    if value.tzinfo is None:
        raise ProtocolError("attempt timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _attempt_record(
    request: CollectionRequest,
    binding: CollectionBinding,
    item: CallPlanItem,
    call_id: str,
    status: AttemptStatus,
    timestamp_utc: str,
    *,
    chunk_sha256: str | None = None,
    error: str | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        request_sha256=request.sha256,
        **binding.to_dict(),
        gpu=item.gpu,
        bank=item.bank,
        call_id=call_id,
        status=status,
        timestamp_utc=timestamp_utc,
        chunk_sha256=chunk_sha256,
        error=error,
    )


def _decode_remote_rows(value: object) -> tuple[Measurement, ...]:
    if type(value) not in {list, tuple}:
        raise ProtocolError("remote call result must be a measurement sequence")
    sequence = cast(list[object] | tuple[object, ...], value)
    rows: list[Measurement] = []
    for row in sequence:
        if type(row) is Measurement:
            rows.append(row)
        else:
            rows.append(Measurement.from_dict(row))
    return tuple(rows)


def execute_call_plan(
    request: CollectionRequest,
    binding: CollectionBinding,
    journal: AttemptJournal,
    *,
    spawn: Callable[[CallPlanItem], RemoteCall],
    restore: Callable[[str], RemoteCall] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[CollectionChunk, ...]:
    """Spawn all calls durably before retrieval, or resume with zero new spawns."""
    plan = build_call_plan(request)
    handles: dict[tuple[str, int], RemoteCall] = {}
    terminal = journal.latest_by_key()
    spawned = journal.spawned_by_key()
    if journal.records:
        journal.require_binding(request, binding)
        if restore is None:
            raise ProtocolError("resume requires a FunctionCall.from_id adapter")
        expected = {(item.gpu, item.bank) for item in plan}
        if set(spawned) != expected:
            raise ProtocolError(
                "resume journal does not contain exactly one call for every plan item"
            )
        failed = [key for key, record in terminal.items() if record.status == "failed"]
        if failed:
            raise ProtocolError(f"attempt journal contains failed calls and cannot retry: {failed}")
        for item in plan:
            call_id = spawned[(item.gpu, item.bank)].call_id
            handles[(item.gpu, item.bank)] = restore(call_id)
    else:
        for item in plan:
            handle = spawn(item)
            call_id = nonblank_string(handle.object_id, context="FunctionCall.object_id")
            journal.append(
                _attempt_record(
                    request,
                    binding,
                    item,
                    call_id,
                    "spawned",
                    _timestamp(now),
                )
            )
            handles[(item.gpu, item.bank)] = handle

    chunks: list[CollectionChunk] = []
    for item in plan:
        key = (item.gpu, item.bank)
        handle = handles[key]
        call_id = spawned.get(key, journal.spawned_by_key()[key]).call_id
        try:
            rows = _decode_remote_rows(handle.get())
            chunk = CollectionChunk.from_measurements(item.gpu, item.bank, call_id, rows)
            prior_terminal = terminal.get(key)
            if prior_terminal is not None and prior_terminal.status == "completed":
                if prior_terminal.chunk_sha256 != chunk.sha256:
                    raise ProtocolError(f"resumed chunk digest changed for {key}")
            else:
                journal.append(
                    _attempt_record(
                        request,
                        binding,
                        item,
                        call_id,
                        "completed",
                        _timestamp(now),
                        chunk_sha256=chunk.sha256,
                    )
                )
            chunks.append(chunk)
        except Exception as exc:
            prior_terminal = terminal.get(key)
            if prior_terminal is None or prior_terminal.status == "spawned":
                journal.append(
                    _attempt_record(
                        request,
                        binding,
                        item,
                        call_id,
                        "failed",
                        _timestamp(now),
                        error=f"{type(exc).__name__}: {exc}".strip(),
                    )
                )
            raise ProtocolError(
                f"remote collection failed for {item.gpu}/bank-{item.bank}"
            ) from exc
    return tuple(chunks)


def measurement_sort_key(measurement: Measurement) -> tuple[object, ...]:
    return (
        measurement.hardware.gpu,
        measurement.bank,
        measurement.workload.key,
        measurement.config.key,
    )


def commit_chunks(
    output: str | Path,
    request: CollectionRequest,
    binding: CollectionBinding,
    journal: AttemptJournal,
    chunks: Sequence[CollectionChunk],
    *,
    facts: Mapping[str, object] | None = None,
) -> CollectionManifest:
    """Validate every canonical chunk, then durably replace data and sidecar."""
    destination = Path(output)
    expected_keys = {(item.gpu, item.bank) for item in build_call_plan(request)}
    chunk_by_key: dict[tuple[str, int], CollectionChunk] = {}
    for chunk in chunks:
        key = (chunk.gpu, chunk.bank)
        if key in chunk_by_key:
            raise ProtocolError(f"duplicate retrieved chunk {key}")
        chunk_by_key[key] = chunk
    if set(chunk_by_key) != expected_keys:
        raise ProtocolError(
            f"retrieved chunks are {sorted(chunk_by_key)}, expected {sorted(expected_keys)}"
        )
    journal.require_binding(request, binding)
    latest = journal.latest_by_key()
    if any(latest[key].status != "completed" for key in expected_keys):
        raise ProtocolError("all attempt journal calls must be completed before commit")
    for key in expected_keys:
        chunk = chunk_by_key[key]
        record = latest[key]
        if record.call_id != chunk.call_id or record.chunk_sha256 != chunk.sha256:
            raise ProtocolError(f"attempt journal digest or call ID mismatch for {key}")

    rows = tuple(
        sorted(
            (
                row
                for item in build_call_plan(request)
                for row in chunk_by_key[(item.gpu, item.bank)].measurements
            ),
            key=measurement_sort_key,
        )
    )
    expected_cells = {
        (gpu, bank, workload_key, config_key)
        for gpu in request.gpus
        for bank in request.banks
        for workload_key in request.workload_keys
        for config_key in request.config_keys
    }
    actual_cells = {(row.hardware.gpu, row.bank, row.workload.key, row.config.key) for row in rows}
    if actual_cells != expected_cells or len(rows) != len(expected_cells):
        raise ProtocolError("retrieved measurements do not match the exact requested cross-product")
    profiles: dict[str, HardwareProfile] = {}
    torch_timings: dict[
        tuple[str, int, str],
        tuple[float, float | None, float | None, float | None],
    ] = {}
    for row in rows:
        known_profile = profiles.setdefault(row.hardware.gpu, row.hardware)
        if known_profile != row.hardware:
            raise ProtocolError(f"inconsistent hardware profile in chunk for {row.hardware.gpu}")
        timing_key = (row.hardware.gpu, row.bank, row.workload.key)
        timing = (
            row.torch_latency_ms,
            row.torch_latency_p20_ms,
            row.torch_latency_p80_ms,
            row.torch_benchmark_wall_ms,
        )
        known_timing = torch_timings.setdefault(timing_key, timing)
        if known_timing != timing:
            raise ProtocolError(
                f"inconsistent duplicated torch timing for "
                f"{row.hardware.gpu}/{row.workload.key}/bank-{row.bank}"
            )

    write_measurements_atomic(destination, rows)
    data_sha256 = sha256_file(destination)
    journal_sha256 = sha256_file(journal.path)
    manifest = CollectionManifest(
        request=request,
        binding=binding,
        data_path=str(destination),
        data_sha256=data_sha256,
        rows=len(rows),
        failures=sum(not row.usable for row in rows),
        attempt_journal_path=str(journal.path),
        attempt_journal_sha256=journal_sha256,
        plan=build_call_plan(request),
        hardware=tuple(profiles[gpu].to_dict() for gpu in sorted(profiles)),
        calls=journal.records,
        facts=MappingProxyType(dict(facts or {})),
    )
    write_json_atomic(manifest_path(destination), manifest.to_dict())
    return manifest
