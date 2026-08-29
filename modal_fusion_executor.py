"""Run one frozen fusion suite through one client-authorized Modal H100 spawn.

Invoke with::

    uv run --extra modal modal run modal_fusion_executor.py::main --suite ... --plugin ... --output ...

Modal may physically start or restart the input despite ``retries=0``. Those
provider attempts are unobservable, so this publishes a remote receipt rather
than a methodology bundle and states no total GPU-time or cost upper bound.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import modal

_WHEEL_FILENAME = re.compile(
    r"[A-Za-z0-9_.]+-[A-Za-z0-9_.!+]+(-[0-9][A-Za-z0-9_.]*)?"
    r"-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\.whl"
)
_REPO = Path(__file__).resolve().parent
_WHEEL_DIRECTORY = "artifacts/modal-wheel"
_WHEEL_MANIFEST_SCHEMA_VERSION = 1
_PYTHON_VERSION = "3.11"
_PIP_DEPENDENCIES = (
    "numpy==2.4.6",
    "rich==14.3.4",
    "zstandard==0.25.0",
    "torch==2.8.0",
    "triton==3.4.0",
)
_BUILD_DEPENDENCIES = ("hatchling==1.32.0",)
_BUILD_TOOLS = {"uv": "0.12.5", "hatchling": "1.32.0"}
_MODAL_SELECTOR = "H100!"
_REMOTE_TIMEOUT_SECONDS = 3600
_CLIENT_TIMEOUT_SECONDS = 3660


@dataclass(frozen=True, slots=True)
class WheelProvenance:
    wheel: Path
    manifest: Path
    manifest_bytes: bytes
    wheel_sha256: str
    manifest_sha256: str
    head_commit: str
    source_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_digest(repository: Path) -> str:
    from heliostune.wheel_verifier import source_digest, source_entries

    return source_digest(source_entries(repository / "src/heliostune"))


def wheel_manifest_path(wheel: Path) -> Path:
    return wheel.with_name(f"{wheel.name}.manifest.json")


def remote_wheel_path(wheel: Path) -> str:
    if _WHEEL_FILENAME.fullmatch(wheel.name) is None:
        raise ValueError(
            f"Modal wheel is not a valid PEP 427 filename: {wheel.name}; expected "
            "distribution-version(-build)?-python-abi-platform.whl"
        )
    return f"/root/{wheel.name}"


def remote_wheel_manifest_path(wheel: Path) -> str:
    return f"{remote_wheel_path(wheel)}.manifest.json"


def _git_head(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("Modal wheel use requires a clean Git HEAD")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("Git HEAD is not a full lowercase hexadecimal commit")
    return head


def _manifest_object(payload: bytes, path: Path) -> dict[str, object]:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"wheel manifest contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RuntimeError(f"wheel manifest contains non-finite constant {item!r}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read Modal wheel manifest {path}: {exc}") from exc
    if type(value) is not dict:
        raise RuntimeError(f"Modal wheel manifest must be a JSON object: {path}")
    return cast(dict[str, object], value)


def validate_wheel_manifest(
    wheel: Path,
    *,
    repository: Path | None = None,
    remote: bool = False,
) -> WheelProvenance:
    """Validate supplemental manifest and, locally, wheel/RECORD/source custody."""
    configured_manifest = os.environ.get("HELIOSTUNE_MODAL_WHEEL_MANIFEST")
    if remote:
        if not configured_manifest:
            raise RuntimeError("HELIOSTUNE_MODAL_WHEEL_MANIFEST is required for a remote wheel")
        manifest = Path(configured_manifest)
        if manifest != wheel_manifest_path(wheel):
            raise RuntimeError("remote Modal wheel manifest must be adjacent to the wheel")
    else:
        manifest = wheel_manifest_path(wheel)
    if not wheel.is_file():
        raise RuntimeError(f"Modal wheel does not exist: {wheel}")
    if not manifest.is_file():
        raise RuntimeError(f"Modal wheel manifest does not exist: {manifest}")
    manifest_bytes = manifest.read_bytes()
    data = _manifest_object(manifest_bytes, manifest)
    expected_fields = {
        "schema_version",
        "head_commit",
        "source_sha256",
        "wheel_filename",
        "wheel_sha256",
        "python_version",
        "pip_dependencies",
        "build_dependencies",
        "build_tools",
        "wheel_install_args",
    }
    if set(data) != expected_fields:
        raise RuntimeError(
            "Modal wheel manifest fields differ: "
            f"missing={sorted(expected_fields - set(data))}, unknown={sorted(set(data) - expected_fields)}"
        )
    wheel_sha256 = _sha256_file(wheel)
    expected_values: dict[str, object] = {
        "schema_version": _WHEEL_MANIFEST_SCHEMA_VERSION,
        "wheel_filename": wheel.name,
        "wheel_sha256": wheel_sha256,
        "python_version": _PYTHON_VERSION,
        "pip_dependencies": list(_PIP_DEPENDENCIES),
        "build_dependencies": list(_BUILD_DEPENDENCIES),
        "build_tools": _BUILD_TOOLS,
        "wheel_install_args": ["--no-deps"],
    }
    for field, expected in expected_values.items():
        if data[field] != expected:
            raise RuntimeError(
                f"Modal wheel manifest {field} is {data[field]!r}, expected {expected!r}"
            )
    for field, length in (("head_commit", 40), ("source_sha256", 64)):
        value = data[field]
        if (
            type(value) is not str
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"Modal wheel manifest {field} is not a lowercase hex digest")
    if not remote:
        from heliostune.wheel_verifier import verify_wheel_against_source

        root = _REPO if repository is None else repository.resolve()
        head = _git_head(root)
        if data["head_commit"] != head:
            raise RuntimeError(
                f"Modal wheel was built at HEAD {data['head_commit']}, current HEAD is {head}"
            )
        verified = verify_wheel_against_source(wheel, root / "src/heliostune")
        if verified.wheel_sha256 != wheel_sha256:
            raise RuntimeError("Modal wheel digest changed during verification")
        if data["source_sha256"] != verified.source_sha256:
            raise RuntimeError("Modal wheel packaged sources do not match its source manifest")
    return WheelProvenance(
        wheel=wheel,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        wheel_sha256=wheel_sha256,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        head_commit=cast(str, data["head_commit"]),
        source_sha256=cast(str, data["source_sha256"]),
    )


def configured_modal_wheel(root: Path | None = None) -> Path:
    base = _REPO if root is None else root.resolve()
    configured = os.environ.get("HELIOSTUNE_MODAL_WHEEL")
    if configured:
        wheel = Path(configured)
        if not wheel.is_file():
            raise RuntimeError(f"HELIOSTUNE_MODAL_WHEEL does not exist: {wheel}")
        remote = wheel.is_absolute() and wheel.parent == Path("/root")
        validate_wheel_manifest(wheel, repository=base, remote=remote)
        return wheel
    directory = base / _WHEEL_DIRECTORY
    wheels = tuple(sorted(directory.glob("heliostune-*.whl")))
    if len(wheels) != 1:
        raise RuntimeError(
            "run `uv run python scripts/build_modal_wheel.py` after committing the final clean "
            f"HEAD; searched {directory}; found {[str(item) for item in wheels]}"
        )
    validate_wheel_manifest(wheels[0], repository=base)
    return wheels[0]


def build_image(wheel: Path) -> Any:
    remote = wheel.is_absolute() and wheel.parent == Path("/root")
    provenance = validate_wheel_manifest(wheel, remote=remote)
    wheel_remote = remote_wheel_path(wheel)
    manifest_remote = remote_wheel_manifest_path(wheel)
    return (
        modal.Image.debian_slim(python_version=_PYTHON_VERSION)
        .pip_install(*_PIP_DEPENDENCIES)
        .add_local_file(wheel, remote_path=wheel_remote, copy=True)
        .add_local_file(provenance.manifest, remote_path=manifest_remote, copy=True)
        .run_commands(f"python -m pip install --no-deps {wheel_remote}")
        .env(
            {
                "HELIOSTUNE_MODAL_WHEEL": wheel_remote,
                "HELIOSTUNE_MODAL_WHEEL_MANIFEST": manifest_remote,
            }
        )
    )


app = modal.App("heliostune-fusion-executor")
_MODAL_WHEEL = configured_modal_wheel()
image = build_image(_MODAL_WHEEL)


@app.function(
    image=image,
    gpu=_MODAL_SELECTOR,
    retries=0,
    timeout=_REMOTE_TIMEOUT_SECONDS,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    restrict_modal_access=True,
    _experimental_restrict_output=True,
)
def execute_fusion_suite(request_json: str) -> str:
    """Validate and execute one suite, returning one compressed transport wrapper."""
    from heliostune.hardware import expectation_for_gpu, validate_hardware
    from heliostune.kernel import get_hardware_profile
    from heliostune.local_executor import run_local_suite
    from heliostune.remote_execution import (
        RemoteResultEnvelope,
        decode_remote_request,
        sha256_bytes,
    )

    intent, suite_bytes, request_digest = decode_remote_request(request_json)
    wheel = configured_modal_wheel()
    provenance = validate_wheel_manifest(wheel, remote=True)
    observed_wheel = {
        "wheel_filename": wheel.name,
        "wheel_sha256": provenance.wheel_sha256,
        "manifest_sha256": provenance.manifest_sha256,
        "head_commit": provenance.head_commit,
        "source_sha256": provenance.source_sha256,
    }
    for field, observed in observed_wheel.items():
        if getattr(intent, field) != observed:
            raise RuntimeError(f"remote {field} does not match the request intent")
    if sha256_bytes(suite_bytes) != intent.suite_sha256:
        raise RuntimeError("remote suite bytes do not match the request intent")
    hardware = get_hardware_profile("H100")
    validate_hardware(hardware, expectation_for_gpu("H100"))
    with tempfile.TemporaryDirectory(prefix="heliostune-fusion-remote-") as temporary:
        os.chmod(temporary, 0o700)
        suite_path = Path(temporary) / "suite.json"
        descriptor = os.open(suite_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(suite_bytes)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while materializing private remote suite")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        result = run_local_suite(suite_path)
    if (
        result.verified_suite_sha256 != intent.suite_sha256
        or result.verified_suite_bytes != suite_bytes
    ):
        raise RuntimeError("local executor changed the verified remote suite byte binding")
    result = replace(result, verified_suite_path=intent.suite_path)
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
        hardware=hardware,
        environment=result.environment,
        result=result.to_dict(),
    ).to_transport_json()


@dataclass(frozen=True, slots=True)
class _LocalPlan:
    suite_path: Path
    suite_bytes: bytes
    plugin_path: Path
    plugin_bytes: bytes
    manifest_bytes: bytes
    intent: Any
    request_json: str
    request_digest: str


def _preflight(suite: str | Path, plugin: str | Path, output: str | Path) -> _LocalPlan:
    from heliostune.local_executor import GATED_MLP_SUITE_SHA256, RMSNORM_SUITE_SHA256
    from heliostune.remote_execution import (
        RemoteIntent,
        decode_remote_request,
        encode_remote_request,
        protect_remote_output,
        sha256_bytes,
    )
    from heliostune.scope import verify_plugin, verify_suite

    suite_path = Path(suite)
    plugin_path = Path(plugin)
    output_path, _, _ = protect_remote_output(output)
    verified_suite = verify_suite(suite_path)
    if verified_suite.sha256 not in {GATED_MLP_SUITE_SHA256, RMSNORM_SUITE_SHA256}:
        raise ValueError("--suite must be one of the two frozen fusion suites")
    verified_plugin = verify_plugin(plugin_path)
    if (
        verified_plugin.plugin.plugin_id != "fusion-reference-plugin"
        or verified_plugin.plugin.version != 1
    ):
        raise ValueError("--plugin must be the frozen fusion-reference-plugin version 1")
    matches = tuple(
        item
        for item in verified_plugin.suites
        if item.sha256 == verified_suite.sha256
        and item.suite.suite_id == verified_suite.suite.suite_id
        and item.suite.revision == verified_suite.suite.revision
    )
    if len(matches) != 1:
        raise ValueError("--plugin does not bind the selected frozen suite exactly once")
    provenance = validate_wheel_manifest(_MODAL_WHEEL)
    intent = RemoteIntent(
        suite_path=str(suite_path),
        output_path=str(output_path),
        suite_sha256=verified_suite.sha256,
        plugin_path=str(plugin_path),
        plugin_sha256=sha256_bytes(verified_plugin.bytes),
        wheel_filename=provenance.wheel.name,
        wheel_sha256=provenance.wheel_sha256,
        manifest_sha256=provenance.manifest_sha256,
        head_commit=provenance.head_commit,
        source_sha256=provenance.source_sha256,
    )
    request_json = encode_remote_request(intent, verified_suite.bytes)
    _, request_suite_bytes, request_digest = decode_remote_request(request_json)
    if request_suite_bytes != verified_suite.bytes:
        raise RuntimeError(
            "locally encoded request did not preserve the exact verified suite bytes"
        )
    return _LocalPlan(
        suite_path=suite_path,
        suite_bytes=verified_suite.bytes,
        plugin_path=plugin_path,
        plugin_bytes=verified_plugin.bytes,
        manifest_bytes=provenance.manifest_bytes,
        intent=intent,
        request_json=request_json,
        request_digest=request_digest,
    )


def _safe_detail(exc: BaseException) -> str:
    detail = f"{type(exc).__name__}: {exc}".strip()
    return detail if detail else type(exc).__name__


def _cancel_then_unresolve(
    call: Any, journal: Any, call_id: str | None, exc: BaseException
) -> None:
    """Cancellation is attempted before either fallible journal append."""
    with contextlib.suppress(BaseException):
        call.cancel(terminate_containers=True)
    detail = _safe_detail(exc)
    with contextlib.suppress(BaseException):
        journal.append("cancellation_requested", call_id=call_id, detail=detail)
    with contextlib.suppress(BaseException):
        journal.append("unresolved", call_id=call_id, detail=detail)


def _publish_unresolved(plan: _LocalPlan, records: Any, *, client_spawn_count: int) -> None:
    from heliostune.remote_execution import write_remote_receipt

    if records.journal.state != "unresolved":
        return
    with contextlib.suppress(BaseException):
        write_remote_receipt(
            records,
            status="unresolved",
            request_digest=plan.request_digest,
            suite_bytes=plan.suite_bytes,
            plugin_bytes=plan.plugin_bytes,
            manifest_bytes=plan.manifest_bytes,
            result_payload=None,
            client_spawn_count=client_spawn_count,
        )


def _execute_plan(plan: _LocalPlan, output: str | Path, remote_function: Any) -> int:
    from heliostune.remote_execution import (
        CLIENT_TIMEOUT_SECONDS,
        create_remote_records,
        validate_remote_result,
        write_remote_receipt,
    )
    from heliostune.validation import nonblank_string

    records = create_remote_records(output, plan.intent, plan.request_digest)
    call: Any | None = None
    call_id: str | None = None
    client_spawn_count = 0
    try:
        records.assert_parent_identity()
        try:
            client_spawn_count = 1
            call = remote_function.spawn(plan.request_json)
        except BaseException as exc:
            detail = _safe_detail(exc)
            with contextlib.suppress(BaseException):
                records.journal.append("spawn_acknowledgement_lost", detail=detail)
            with contextlib.suppress(BaseException):
                records.journal.append("unresolved", detail=detail)
            _publish_unresolved(plan, records, client_spawn_count=client_spawn_count)
            raise
        try:
            records.assert_parent_identity()
            call_id = nonblank_string(call.object_id, context="Modal function call ID")
            records.journal.append("spawned", call_id=call_id)
            records.journal.append("retrieval_started", call_id=call_id)
            remote_payload = call.get(timeout=CLIENT_TIMEOUT_SECONDS)
            envelope, result = validate_remote_result(
                remote_payload,
                intent=plan.intent,
                request_digest=plan.request_digest,
                verified_suite_bytes=plan.suite_bytes,
            )
        except BaseException as exc:
            _cancel_then_unresolve(call, records.journal, call_id, exc)
            _publish_unresolved(plan, records, client_spawn_count=client_spawn_count)
            raise
        status = cast(Any, result.outcome)
        records.journal.append(status, call_id=call_id)
        write_remote_receipt(
            records,
            status=status,
            request_digest=plan.request_digest,
            suite_bytes=plan.suite_bytes,
            plugin_bytes=plan.plugin_bytes,
            manifest_bytes=plan.manifest_bytes,
            result_payload=envelope.to_json(),
            client_spawn_count=client_spawn_count,
        )
        return 0 if result.outcome == "completed" else 1
    finally:
        records.close()


@app.local_entrypoint()
def main(suite: str, plugin: str, output: str) -> None:
    """Execute one frozen suite and publish a truthful remote receipt."""
    plan = _preflight(suite, plugin, output)
    exit_code = _execute_plan(plan, output, execute_fusion_suite)
    if exit_code:
        raise SystemExit(exit_code)
