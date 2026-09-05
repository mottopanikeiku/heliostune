"""Audited deterministic analyzer replay in an unprivileged offline sandbox."""

from __future__ import annotations

import base64
import hashlib
import importlib.resources
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from heliostune import _reference_analyzer
from heliostune.artifacts import strict_json_dumps, strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.methodology import (
    CapturedBundleArtifactV1,
    VerifiedBundle,
    capture_bundle_artifacts_v1_from_directory_fd,
    verify_bundle_v1,
    verify_bundle_v1_from_directory_fd,
)
from heliostune.validation import exact_fields, exact_int, nonblank_string
from heliostune.verification import (
    VerificationRecordV1,
    VerifierIdentityV1,
    _publish_exact_verification_record_v1,
    build_verification_record_v1,
)

ANALYZER_MANIFEST_ROLE_V1 = "analyzer"
ANALYZER_MANIFEST_MEDIA_TYPE_V1 = "application/json"
ANALYZER_SOURCE_ROLE_V1 = "analyzer_source"
REFERENCE_ANALYZER_ID_V1 = "heliostune.reference.integer-summary/1"
OFFLINE_REPLAY_SANDBOX_V1 = "Linux no-new-privileges user, network, mount, and PID namespaces; empty read-only nosuid,nodev,noexec tmpfs chroot; deny-latch Python audit hook"
_SET_PRIV = "/usr/bin/setpriv"
_UNSHARE = "/usr/bin/unshare"
_RUNNER_API = "heliostune.offline-replay/1"
_REQUEST_SCHEMA = "heliostune.offline-replay-request/1"
_RESULT_SCHEMA = "heliostune.offline-replay-result/1"
_SOURCE_DOMAIN = b"heliostune.analyzer-sources/1\0"
_MAX_FRAME_BYTES = 8 * 1024 * 1024
_MAX_RESULT_BYTES = 2 * 1024 * 1024
_DEFAULT_TIMEOUT_S = 30.0
_MAX_TIMEOUT_S = 300.0
_REAP_TIMEOUT_S = 5.0
_SANITIZED_ENV = {
    "HOME": "/",
    "LC_ALL": "C",
    "LANG": "C",
    "TZ": "UTC",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def _digest(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise SchemaError(f"{context} must be a lowercase SHA-256 digest")
    return result


def _byte_count(value: object, *, context: str) -> int:
    result = exact_int(value, context=context, minimum=0)
    if result >= 1 << 64:
        raise SchemaError(f"{context} must fit an unsigned 64-bit integer")
    return result


def _object_array(value: object, *, context: str) -> list[object]:
    if type(value) is not list:
        raise SchemaError(f"{context} must be an array")
    return cast(list[object], value)


@dataclass(frozen=True, slots=True)
class AnalyzerArtifactBindingV1:
    role: str
    media_type: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        nonblank_string(self.role, context="analyzer artifact role")
        nonblank_string(self.media_type, context="analyzer artifact media_type")
        _byte_count(self.bytes, context="analyzer artifact bytes")
        _digest(self.sha256, context="analyzer artifact sha256")

    @classmethod
    def from_dict(cls, value: object) -> AnalyzerArtifactBindingV1:
        data = exact_fields(
            value,
            required=("role", "media_type", "bytes", "sha256"),
            context="analyzer artifact binding",
        )
        return cls(
            nonblank_string(data["role"], context="analyzer artifact role"),
            nonblank_string(data["media_type"], context="analyzer artifact media_type"),
            _byte_count(data["bytes"], context="analyzer artifact bytes"),
            _digest(data["sha256"], context="analyzer artifact sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "media_type": self.media_type,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


def _source_aggregate_sha256(sources: tuple[AnalyzerArtifactBindingV1, ...]) -> str:
    digest = hashlib.sha256(_SOURCE_DOMAIN)
    for source in sources:
        try:
            role = source.role.encode("utf-8")
            media = source.media_type.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SchemaError("analyzer source role and media_type must be valid Unicode") from exc
        for framed in (role, media):
            digest.update(len(framed).to_bytes(8, "big"))
            digest.update(framed)
        digest.update(source.bytes.to_bytes(8, "big"))
        digest.update(bytes.fromhex(source.sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AnalyzerImplementationV1:
    source_sha256: str
    sources: tuple[AnalyzerArtifactBindingV1, ...]

    def __post_init__(self) -> None:
        _digest(self.source_sha256, context="analyzer implementation source_sha256")
        if (
            type(self.sources) is not tuple
            or not self.sources
            or any(type(x) is not AnalyzerArtifactBindingV1 for x in self.sources)
        ):
            raise SchemaError("analyzer implementation sources must be a nonempty binding tuple")
        roles = tuple(x.role for x in self.sources)
        if len(roles) != len(set(roles)):
            raise SchemaError("analyzer implementation source roles must be unique")
        if self.source_sha256 != _source_aggregate_sha256(self.sources):
            raise SchemaError("analyzer implementation source_sha256 does not match sources")

    @classmethod
    def from_dict(cls, value: object) -> AnalyzerImplementationV1:
        data = exact_fields(
            value, required=("source_sha256", "sources"), context="analyzer implementation"
        )
        sources = tuple(
            AnalyzerArtifactBindingV1.from_dict(x)
            for x in _object_array(data["sources"], context="analyzer implementation sources")
        )
        return cls(
            _digest(data["source_sha256"], context="analyzer implementation source_sha256"), sources
        )

    def to_dict(self) -> dict[str, object]:
        return {"source_sha256": self.source_sha256, "sources": [x.to_dict() for x in self.sources]}


@dataclass(frozen=True, slots=True)
class AnalyzerManifestV1:
    schema: Literal["heliostune.analyzer-manifest/1"]
    analyzer_id: str
    runner_api: Literal["heliostune.offline-replay/1"]
    implementation: AnalyzerImplementationV1
    inputs: tuple[AnalyzerArtifactBindingV1, ...]
    outputs: tuple[AnalyzerArtifactBindingV1, ...]
    representation: Literal["byte_exact"]

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != "heliostune.analyzer-manifest/1":
            raise SchemaError("analyzer manifest schema is invalid")
        nonblank_string(self.analyzer_id, context="analyzer manifest analyzer_id")
        if type(self.runner_api) is not str or self.runner_api != _RUNNER_API:
            raise SchemaError("analyzer manifest runner_api is invalid")
        if type(self.implementation) is not AnalyzerImplementationV1:
            raise SchemaError("analyzer manifest implementation is invalid")
        groups = [tuple(x.role for x in self.implementation.sources)]
        for name, bindings in (("inputs", self.inputs), ("outputs", self.outputs)):
            if (
                type(bindings) is not tuple
                or not bindings
                or any(type(x) is not AnalyzerArtifactBindingV1 for x in bindings)
            ):
                raise SchemaError(f"analyzer manifest {name} must be a nonempty binding tuple")
            roles = tuple(x.role for x in bindings)
            if len(roles) != len(set(roles)):
                raise SchemaError(f"analyzer manifest {name} roles must be unique")
            groups.append(roles)
        roles = tuple(role for group in groups for role in group)
        if len(roles) != len(set(roles)) or ANALYZER_MANIFEST_ROLE_V1 in roles:
            raise SchemaError("analyzer manifest roles must be disjoint")
        if type(self.representation) is not str or self.representation != "byte_exact":
            raise SchemaError("analyzer manifest representation is invalid")

    @classmethod
    def from_dict(cls, value: object) -> AnalyzerManifestV1:
        data = exact_fields(
            value,
            required=(
                "schema",
                "analyzer_id",
                "runner_api",
                "implementation",
                "inputs",
                "outputs",
                "representation",
            ),
            context="analyzer manifest",
        )
        schema = nonblank_string(data["schema"], context="analyzer manifest schema")
        api = nonblank_string(data["runner_api"], context="analyzer manifest runner_api")
        representation = nonblank_string(
            data["representation"], context="analyzer manifest representation"
        )
        if (schema, api, representation) != (
            "heliostune.analyzer-manifest/1",
            _RUNNER_API,
            "byte_exact",
        ):
            raise SchemaError("analyzer manifest fixed literal is invalid")
        return cls(
            "heliostune.analyzer-manifest/1",
            nonblank_string(data["analyzer_id"], context="analyzer manifest analyzer_id"),
            "heliostune.offline-replay/1",
            AnalyzerImplementationV1.from_dict(data["implementation"]),
            tuple(
                AnalyzerArtifactBindingV1.from_dict(x)
                for x in _object_array(data["inputs"], context="analyzer manifest inputs")
            ),
            tuple(
                AnalyzerArtifactBindingV1.from_dict(x)
                for x in _object_array(data["outputs"], context="analyzer manifest outputs")
            ),
            "byte_exact",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "analyzer_id": self.analyzer_id,
            "runner_api": self.runner_api,
            "implementation": self.implementation.to_dict(),
            "inputs": [x.to_dict() for x in self.inputs],
            "outputs": [x.to_dict() for x in self.outputs],
            "representation": self.representation,
        }


AnalyzerCallable = Callable[[tuple[tuple[str, bytes], ...]], tuple[tuple[str, bytes], ...]]


@dataclass(frozen=True, slots=True)
class _AnalyzerRegistryEntry:
    analyzer_id: str
    implementation: AnalyzerImplementationV1
    input_spec: tuple[tuple[str, str], ...]
    output_spec: tuple[tuple[str, str], ...]
    callable: AnalyzerCallable


def _reference_registry_entry() -> _AnalyzerRegistryEntry:
    try:
        payload = (
            importlib.resources.files("heliostune").joinpath("_reference_analyzer.py").read_bytes()
        )
    except OSError as exc:
        raise ArtifactError(f"cannot capture installed reference analyzer source: {exc}") from exc
    source = AnalyzerArtifactBindingV1(
        ANALYZER_SOURCE_ROLE_V1, "text/x-python", len(payload), hashlib.sha256(payload).hexdigest()
    )
    implementation = AnalyzerImplementationV1(_source_aggregate_sha256((source,)), (source,))
    return _AnalyzerRegistryEntry(
        REFERENCE_ANALYZER_ID_V1,
        implementation,
        (("analysis_input", "application/json"),),
        (("analysis_summary", "application/json"),),
        _reference_analyzer.analyze,
    )


_REGISTRY = MappingProxyType({REFERENCE_ANALYZER_ID_V1: _reference_registry_entry()})


def _registered_analyzer(analyzer_id: str) -> _AnalyzerRegistryEntry:
    parsed_id = nonblank_string(analyzer_id, context="registered analyzer_id")
    try:
        return _REGISTRY[parsed_id]
    except KeyError as exc:
        raise ArtifactError(f"analyzer_id {parsed_id!r} is not in the audited registry") from exc


def _parse_manifest_bytes(payload: bytes, *, source: str | Path) -> AnalyzerManifestV1:
    if len(payload) > _MAX_FRAME_BYTES:
        raise SchemaError(f"{source}: analyzer manifest exceeds the byte limit")
    try:
        decoded = strict_json_loads(payload.decode(), source=source)
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{source}: analyzer manifest must be UTF-8") from exc
    except RecursionError as exc:
        raise SchemaError(f"{source}: analyzer manifest nesting is too deep") from exc
    manifest = AnalyzerManifestV1.from_dict(decoded)
    if strict_json_dumps(manifest.to_dict()).encode() != payload:
        raise SchemaError(f"{source}: analyzer manifest bytes are not canonical")
    return manifest


def load_analyzer_manifest_v1(path: str | Path) -> AnalyzerManifestV1:
    source = Path(path)
    descriptor: int | None = None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags)
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise ArtifactError(f"analyzer manifest is not a regular file: {source}")
        if identity.st_size > _MAX_FRAME_BYTES:
            raise ArtifactError(f"analyzer manifest exceeds the byte limit: {source}")
        payload = bytearray()
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_FRAME_BYTES - len(payload) + 1),
            )
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > _MAX_FRAME_BYTES:
                raise ArtifactError(f"analyzer manifest exceeds the byte limit: {source}")
        final_identity = os.fstat(descriptor)
        if (
            final_identity.st_dev,
            final_identity.st_ino,
            final_identity.st_size,
            final_identity.st_mtime_ns,
            final_identity.st_ctime_ns,
        ) != (
            identity.st_dev,
            identity.st_ino,
            identity.st_size,
            identity.st_mtime_ns,
            identity.st_ctime_ns,
        ):
            raise ArtifactError(f"analyzer manifest changed while reading: {source}")
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactError(f"cannot read analyzer manifest {source}: {exc}") from exc
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
    return _parse_manifest_bytes(bytes(payload), source=source)


def _require_manifest_registry_match(
    manifest: AnalyzerManifestV1, entry: _AnalyzerRegistryEntry
) -> None:
    if manifest.implementation != entry.implementation:
        raise ArtifactError("analyzer manifest implementation does not match the audited registry")
    if tuple((x.role, x.media_type) for x in manifest.inputs) != entry.input_spec:
        raise ArtifactError("analyzer manifest inputs do not match the registry")
    if tuple((x.role, x.media_type) for x in manifest.outputs) != entry.output_spec:
        raise ArtifactError("analyzer manifest outputs do not match the registry")


def _binding_matches_capture(
    binding: AnalyzerArtifactBindingV1, capture: CapturedBundleArtifactV1
) -> None:
    artifact = capture.artifact
    if (binding.role, binding.media_type, binding.bytes, binding.sha256) != (
        artifact.role,
        artifact.media_type,
        artifact.bytes,
        artifact.sha256,
    ):
        raise ArtifactError(f"manifest binding does not match artifact role {binding.role!r}")
    if (
        len(capture.payload) != binding.bytes
        or hashlib.sha256(capture.payload).hexdigest() != binding.sha256
    ):
        raise ArtifactError(f"captured artifact role {binding.role!r} changed identity")


def _encode_worker_request(
    analyzer_id: str,
    implementation: AnalyzerImplementationV1,
    verifier: VerifierIdentityV1,
    inputs: tuple[tuple[str, bytes], ...],
) -> bytes:
    if type(implementation) is not AnalyzerImplementationV1:
        raise SchemaError("worker request implementation is invalid")
    if type(verifier) is not VerifierIdentityV1:
        raise SchemaError("worker request verifier identity is invalid")
    payload = strict_json_dumps(
        {
            "schema": _REQUEST_SCHEMA,
            "analyzer_id": analyzer_id,
            "implementation": implementation.to_dict(),
            "verifier": verifier.to_dict(),
            "inputs": [
                {"role": role, "base64": base64.b64encode(value).decode("ascii")}
                for role, value in inputs
            ],
        }
    ).encode()
    if len(payload) > _MAX_FRAME_BYTES:
        raise ArtifactError("worker request exceeds the byte limit")
    return payload


def _decode_base64(value: object, *, context: str) -> bytes:
    encoded = nonblank_string(value, context=context)
    try:
        raw = encoded.encode("ascii")
        payload = base64.b64decode(raw, validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise SchemaError(f"{context} must be canonical base64") from exc
    if base64.b64encode(payload) != raw:
        raise SchemaError(f"{context} must be canonical base64")
    return payload


def _decode_worker_request(
    payload: bytes,
) -> tuple[
    str,
    AnalyzerImplementationV1,
    VerifierIdentityV1,
    tuple[tuple[str, bytes], ...],
]:
    if len(payload) > _MAX_FRAME_BYTES:
        raise SchemaError("worker request exceeds the byte limit")
    try:
        value = strict_json_loads(payload.decode(), source="worker request")
    except UnicodeDecodeError as exc:
        raise SchemaError("worker request must be UTF-8") from exc
    data = exact_fields(
        value,
        required=("schema", "analyzer_id", "implementation", "verifier", "inputs"),
        context="worker request",
    )
    if nonblank_string(data["schema"], context="worker request schema") != _REQUEST_SCHEMA:
        raise SchemaError("worker request schema is invalid")
    analyzer_id = nonblank_string(data["analyzer_id"], context="worker request analyzer_id")
    implementation = AnalyzerImplementationV1.from_dict(data["implementation"])
    verifier = VerifierIdentityV1.from_dict(data["verifier"])
    items = []
    for raw in _object_array(data["inputs"], context="worker request inputs"):
        item = exact_fields(raw, required=("role", "base64"), context="worker request input")
        items.append(
            (
                nonblank_string(item["role"], context="worker request input role"),
                _decode_base64(item["base64"], context="worker request input base64"),
            )
        )
    result = analyzer_id, implementation, verifier, tuple(items)
    if _encode_worker_request(*result) != payload:
        raise SchemaError("worker request bytes are not canonical")
    return result


def _encode_worker_result(outputs: tuple[tuple[str, bytes], ...]) -> bytes:
    payload = strict_json_dumps(
        {
            "schema": _RESULT_SCHEMA,
            "outputs": [
                {"role": r, "base64": base64.b64encode(p).decode("ascii")} for r, p in outputs
            ],
        }
    ).encode()
    if len(payload) > _MAX_RESULT_BYTES:
        raise ArtifactError("worker result exceeds the byte limit")
    return payload


def _decode_worker_result(payload: bytes) -> tuple[tuple[str, bytes], ...]:
    if len(payload) > _MAX_RESULT_BYTES:
        raise ArtifactError("worker result exceeds the byte limit")
    try:
        value = strict_json_loads(payload.decode(), source="worker result")
    except (UnicodeDecodeError, RecursionError) as exc:
        raise ArtifactError("worker result is malformed") from exc
    data = exact_fields(value, required=("schema", "outputs"), context="worker result")
    if nonblank_string(data["schema"], context="worker result schema") != _RESULT_SCHEMA:
        raise ArtifactError("worker result schema is invalid")
    outputs = []
    for raw in _object_array(data["outputs"], context="worker result outputs"):
        item = exact_fields(raw, required=("role", "base64"), context="worker result output")
        outputs.append(
            (
                nonblank_string(item["role"], context="worker result output role"),
                _decode_base64(item["base64"], context="worker result output base64"),
            )
        )
    result = tuple(outputs)
    if _encode_worker_result(result) != payload:
        raise ArtifactError("worker result bytes are not canonical")
    return result


def _worker_argv() -> tuple[str, ...]:
    for executable in (_SET_PRIV, _UNSHARE):
        try:
            identity = os.stat(executable, follow_symlinks=False)
        except OSError as exc:
            raise ArtifactError(f"required sandbox executable unavailable: {executable}") from exc
        if (
            not stat.S_ISREG(identity.st_mode)
            or os.path.realpath(executable) != executable
            or not os.access(executable, os.X_OK)
        ):
            raise ArtifactError(f"sandbox executable is not fixed: {executable}")
    if not isinstance(sys.executable, str) or not os.path.isabs(sys.executable):
        raise ArtifactError("replay requires an absolute current Python executable")
    return (
        _SET_PRIV,
        "--no-new-privs",
        _UNSHARE,
        "--user",
        "--map-root-user",
        "--net",
        "--mount",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
        "--mount-proc",
        sys.executable,
        "-B",
        "-P",
        "-s",
        "-m",
        "heliostune._offline_worker",
    )


def _timeout(value: object) -> float:
    if type(value) not in {int, float}:
        raise SchemaError("timeout_s must be a number")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or result <= 0 or result > _MAX_TIMEOUT_S:
        raise SchemaError("timeout_s is outside the permitted range")
    return result


def _kill_and_reap_worker(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    try:
        process.communicate(timeout=_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise ArtifactError("replay worker could not be reaped after SIGKILL") from exc


def _run_worker(
    request: bytes, workspace: Path, *, timeout_s: float
) -> tuple[tuple[str, bytes], ...]:
    if stat.S_IMODE(workspace.stat().st_mode) != 0o555:
        raise ArtifactError("replay workspace is not read-only")
    with (
        tempfile.TemporaryFile("w+b") as stdout_file,
        tempfile.TemporaryFile("w+b") as stderr_file,
    ):
        try:
            process = subprocess.Popen(
                _worker_argv(),
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=workspace,
                env=dict(_SANITIZED_ENV),
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise ArtifactError(f"cannot start replay sandbox: {exc}") from exc
        try:
            process.communicate(input=request, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_and_reap_worker(process)
            raise ArtifactError("replay worker timed out") from None
        except (OSError, ValueError) as exc:
            try:
                _kill_and_reap_worker(process)
            except ArtifactError as cleanup_error:
                raise cleanup_error from exc
            raise ArtifactError(f"cannot communicate with replay worker: {exc}") from exc
        except BaseException:
            with suppress(ArtifactError):
                _kill_and_reap_worker(process)
            raise
        if process.poll() is None:
            _kill_and_reap_worker(process)
            raise ArtifactError("replay worker was not reaped")
        stdout_size = os.fstat(stdout_file.fileno()).st_size
        stderr_size = os.fstat(stderr_file.fileno()).st_size
        if stdout_size > _MAX_RESULT_BYTES or stderr_size > _MAX_RESULT_BYTES:
            raise ArtifactError("worker output exceeded the byte limit")
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(_MAX_RESULT_BYTES + 1)
        stderr = stderr_file.read(_MAX_RESULT_BYTES + 1)
        if process.returncode != 0:
            raise ArtifactError(f"replay worker exited with status {process.returncode}")
        if stderr:
            raise ArtifactError("replay worker emitted stderr")
        return _decode_worker_result(stdout)


def _same_verified_bundle(current: VerifiedBundle, expected: VerifiedBundle) -> None:
    if current != expected:
        raise ArtifactError("bundle changed during offline replay")


def _output_identities(
    outputs: tuple[tuple[str, bytes], ...],
    expected: tuple[AnalyzerArtifactBindingV1, ...],
) -> tuple[AnalyzerArtifactBindingV1, ...]:
    if len(outputs) != len(expected):
        raise ArtifactError("replay output count does not match manifest")
    identities: list[AnalyzerArtifactBindingV1] = []
    for (role, payload), binding in zip(outputs, expected, strict=True):
        if type(role) is not str or type(payload) is not bytes or role != binding.role:
            raise ArtifactError("replay output roles, order, or types do not match manifest")
        identity = AnalyzerArtifactBindingV1(
            role,
            binding.media_type,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        if identity != binding:
            raise ArtifactError(f"replay output role {role!r} identity differs")
        identities.append(identity)
    return tuple(identities)


def _require_replay_prerequisites(verified: VerifiedBundle) -> VerificationRecordV1:
    base = build_verification_record_v1(verified)
    for name in (
        "plugin_suite_custody",
        "attempt_journal_hash_chain",
        "attempt_reconciliation",
    ):
        if getattr(base.controls, name) != "checked":
            raise ArtifactError(f"replay requires control {name!r} checked")
    return base


def _build_replay_verification_record_v1(
    verified: VerifiedBundle,
) -> VerificationRecordV1:
    """Rebuild the base record and upgrade exactly the two replay controls."""

    base = _require_replay_prerequisites(verified)
    controls = base.controls
    upgraded = replace(
        controls,
        analyzer_replay="checked",
        offline_reproduction="checked",
    )
    all_checked = upgraded.all_checked
    return replace(
        base,
        controls=upgraded,
        claim_eligible=all_checked,
        publication_eligible=all_checked,
    )


_REPLAY_SUCCESS = object()


@dataclass(frozen=True, slots=True)
class OfflineReplayResult:
    """Success-only result minted after two exact isolated runs."""

    verified: VerifiedBundle
    manifest: AnalyzerManifestV1
    first_run_outputs: tuple[AnalyzerArtifactBindingV1, ...]
    second_run_outputs: tuple[AnalyzerArtifactBindingV1, ...]
    record: VerificationRecordV1
    _authorization: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authorization is not _REPLAY_SUCCESS:
            raise SchemaError("OfflineReplayResult requires a completed replay")
        if (
            type(self.verified) is not VerifiedBundle
            or type(self.manifest) is not AnalyzerManifestV1
            or type(self.record) is not VerificationRecordV1
        ):
            raise SchemaError("replay result has invalid fields")
        for outputs in (self.first_run_outputs, self.second_run_outputs):
            if type(outputs) is not tuple or any(
                type(item) is not AnalyzerArtifactBindingV1 for item in outputs
            ):
                raise SchemaError("replay result has invalid output identities")
        if (
            self.first_run_outputs != self.second_run_outputs
            or self.first_run_outputs != self.manifest.outputs
        ):
            raise SchemaError("replay result is not exact and deterministic")
        if self.record != _build_replay_verification_record_v1(self.verified):
            raise SchemaError("replay result record is not exact")


def build_replay_verification_record_v1(
    result: OfflineReplayResult,
) -> VerificationRecordV1:
    """Rebuild the exact upgrade authorized by a completed replay result."""

    if type(result) is not OfflineReplayResult or result._authorization is not _REPLAY_SUCCESS:
        raise ArtifactError("only completed replay can authorize a replay record")
    expected = _build_replay_verification_record_v1(result.verified)
    if (
        result.record != expected
        or result.first_run_outputs != result.second_run_outputs
        or result.first_run_outputs != result.manifest.outputs
    ):
        raise ArtifactError("replay result is not exact")
    return expected


def replay_bundle_v1(
    root_manifest_path: str | Path,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> OfflineReplayResult:
    """Verify and replay one registered analyzer twice in distinct sandboxes."""

    timeout = _timeout(timeout_s)
    verified = verify_bundle_v1(root_manifest_path)
    base_record = _require_replay_prerequisites(verified)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_fd: int | None = None
    try:
        directory_fd = os.open(verified.root_path.parent, flags)
        identity = os.fstat(directory_fd)
        if (identity.st_dev, identity.st_ino) != verified.root_directory_identity:
            raise ArtifactError("bundle directory changed before replay")
        parent = os.stat("..", dir_fd=directory_fd, follow_symlinks=False)
        if (parent.st_dev, parent.st_ino) != verified.root_parent_directory_identity:
            raise ArtifactError("bundle parent changed before replay")

        def reverify() -> None:
            _same_verified_bundle(
                verify_bundle_v1_from_directory_fd(
                    directory_fd,
                    verified.root_path.name,
                    diagnostic_directory=verified.root_path.parent,
                ),
                verified,
            )

        reverify()
        manifest_capture = capture_bundle_artifacts_v1_from_directory_fd(
            directory_fd,
            verified,
            (ANALYZER_MANIFEST_ROLE_V1,),
            diagnostic_directory=verified.root_path.parent,
        )[0]
        if manifest_capture.artifact.media_type != ANALYZER_MANIFEST_MEDIA_TYPE_V1:
            raise ArtifactError("analyzer artifact media type must be application/json")
        manifest = _parse_manifest_bytes(
            manifest_capture.payload,
            source=manifest_capture.artifact.path,
        )
        entry = _registered_analyzer(manifest.analyzer_id)
        _require_manifest_registry_match(manifest, entry)

        bindings = (*manifest.implementation.sources, *manifest.inputs, *manifest.outputs)
        roles = tuple(binding.role for binding in bindings)
        captures = capture_bundle_artifacts_v1_from_directory_fd(
            directory_fd,
            verified,
            roles,
            diagnostic_directory=verified.root_path.parent,
        )
        by_role = {capture.artifact.role: capture for capture in captures}
        for binding in bindings:
            _binding_matches_capture(binding, by_role[binding.role])
        try:
            installed_source = (
                importlib.resources.files("heliostune")
                .joinpath("_reference_analyzer.py")
                .read_bytes()
            )
        except OSError as exc:
            raise ArtifactError(f"cannot recapture installed analyzer source: {exc}") from exc
        if by_role[ANALYZER_SOURCE_ROLE_V1].payload != installed_source:
            raise ArtifactError("captured analyzer source differs from installed registry")

        reverify()
        inputs = tuple((binding.role, by_role[binding.role].payload) for binding in manifest.inputs)
        request = _encode_worker_request(
            manifest.analyzer_id,
            manifest.implementation,
            base_record.verifier,
            inputs,
        )
        try:
            with ExitStack() as stack:
                workspaces: list[Path] = []
                for _ in range(2):
                    workspace = Path(
                        stack.enter_context(
                            tempfile.TemporaryDirectory(prefix="heliostune-replay-")
                        )
                    )
                    os.chmod(workspace, 0o555)
                    workspaces.append(workspace)
                if workspaces[0] == workspaces[1]:
                    raise ArtifactError("replay workspaces are not distinct")
                first_outputs = _run_worker(request, workspaces[0], timeout_s=timeout)
                second_outputs = _run_worker(request, workspaces[1], timeout_s=timeout)
        finally:
            reverify()

        if first_outputs != second_outputs:
            raise ArtifactError("analyzer outputs are nondeterministic")
        first_ids = _output_identities(first_outputs, manifest.outputs)
        second_ids = _output_identities(second_outputs, manifest.outputs)
        for (role, payload), binding in zip(first_outputs, manifest.outputs, strict=True):
            if payload != by_role[role].payload:
                raise ArtifactError(
                    f"replay output role {binding.role!r} differs from committed bytes"
                )
    except (ArtifactError, SchemaError):
        raise
    except OSError as exc:
        raise ArtifactError(f"offline replay custody failed: {exc}") from exc
    finally:
        if directory_fd is not None:
            with suppress(OSError):
                os.close(directory_fd)

    record = _build_replay_verification_record_v1(verified)
    return OfflineReplayResult(
        verified,
        manifest,
        first_ids,
        second_ids,
        record,
        _REPLAY_SUCCESS,
    )


def _validate_offline_replay_result(
    result: OfflineReplayResult,
) -> VerificationRecordV1:
    return build_replay_verification_record_v1(result)


def write_offline_replay_record_v1(
    path: str | Path,
    result: OfflineReplayResult,
) -> None:
    """Publish only the exact record authorized by a completed replay."""

    record = _validate_offline_replay_result(result)
    _publish_exact_verification_record_v1(path, record, verified=result.verified)


__all__ = [
    "ANALYZER_MANIFEST_MEDIA_TYPE_V1",
    "ANALYZER_MANIFEST_ROLE_V1",
    "ANALYZER_SOURCE_ROLE_V1",
    "AnalyzerArtifactBindingV1",
    "AnalyzerImplementationV1",
    "AnalyzerManifestV1",
    "OFFLINE_REPLAY_SANDBOX_V1",
    "OfflineReplayResult",
    "REFERENCE_ANALYZER_ID_V1",
    "build_replay_verification_record_v1",
    "load_analyzer_manifest_v1",
    "replay_bundle_v1",
    "write_offline_replay_record_v1",
]
