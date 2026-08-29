from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from heliostune.artifacts import strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.remote_execution import (
    CLIENT_TIMEOUT_SECONDS,
    VerifiedRemoteReceipt,
    decode_remote_request,
    encode_remote_request,
    open_remote_records,
    sha256_bytes,
    validate_remote_result,
    write_remote_receipt,
)

_REPOSITORY = Path(__file__).resolve().parents[1]


def _retained_path(repository: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _read_bound(path: Path, expected_sha256: str, *, label: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SchemaError(f"cannot read retained {label}: {path}") from exc
    if sha256_bytes(payload) != expected_sha256:
        raise SchemaError(f"retained {label} digest differs from remote intent")
    return payload


def _validate_manifest(payload: bytes, intent: Any) -> None:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise SchemaError("retained wheel manifest must be UTF-8") from exc
    value = strict_json_loads(text, source="retained wheel manifest")
    if type(value) is not dict:
        raise SchemaError("retained wheel manifest must be an object")
    manifest = value
    expected = {
        "schema_version": 1,
        "head_commit": intent.head_commit,
        "source_sha256": intent.source_sha256,
        "wheel_filename": intent.wheel_filename,
        "wheel_sha256": intent.wheel_sha256,
    }
    for field, bound in expected.items():
        if manifest.get(field) != bound:
            raise SchemaError(f"retained wheel manifest {field} differs from remote intent")


def _load_modal() -> ModuleType:
    return importlib.import_module("modal")


def reconcile_remote_receipt(
    output: str | Path,
    *,
    timeout: int | None = None,
    repository: Path = _REPOSITORY,
    modal_module: Any | None = None,
) -> VerifiedRemoteReceipt:
    """Retrieve one completed FunctionCall and publish its missing receipt."""
    wait_seconds = CLIENT_TIMEOUT_SECONDS if timeout is None else timeout
    if type(wait_seconds) is not int or wait_seconds <= 0:
        raise SchemaError("reconciliation timeout must be a positive integer")

    records = open_remote_records(output)
    try:
        records.assert_parent_identity()
        intent_snapshot = records.intent_snapshot
        journal_snapshot = records.journal_snapshot
        if journal_snapshot is None:
            raise ArtifactError("existing remote journal has no retained snapshot")
        if records.journal.state != "completed":
            raise SchemaError("reconciliation requires journal terminal state exactly 'completed'")
        call_id = records.journal.call_id
        if call_id is None:
            raise SchemaError("completed remote journal has no FunctionCall ID")

        intent = records.intent
        suite_bytes = _read_bound(
            _retained_path(repository, intent.suite_path),
            intent.suite_sha256,
            label="suite",
        )
        plugin_bytes = _read_bound(
            _retained_path(repository, intent.plugin_path),
            intent.plugin_sha256,
            label="plugin",
        )
        manifest_path = (
            repository / "artifacts" / "modal-wheel" / f"{intent.wheel_filename}.manifest.json"
        )
        manifest_bytes = _read_bound(
            manifest_path,
            intent.manifest_sha256,
            label="wheel manifest",
        )
        _validate_manifest(manifest_bytes, intent)

        _, encoded_suite, request_digest = decode_remote_request(
            encode_remote_request(intent, suite_bytes)
        )
        if encoded_suite != suite_bytes or request_digest != records.journal.request_digest:
            raise SchemaError("retained request digest differs from remote journal")

        modal_api = _load_modal() if modal_module is None else modal_module
        remote_payload = modal_api.FunctionCall.from_id(call_id).get(timeout=wait_seconds)
        envelope, result = validate_remote_result(
            remote_payload,
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_bytes,
        )
        if result.outcome != "completed":
            raise SchemaError("completed remote journal result outcome is not completed")
        records.assert_parent_identity()
        return write_remote_receipt(
            records,
            status="completed",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=envelope.to_json(),
            client_spawn_count=1,
            expected_intent_snapshot=intent_snapshot,
            expected_journal_snapshot=journal_snapshot,
        )
    finally:
        records.close()


def _positive_timeout(value: str) -> int:
    try:
        timeout = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a positive integer") from exc
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive integer")
    return timeout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve a completed Modal FunctionCall and publish its missing receipt"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=_positive_timeout)
    args = parser.parse_args(argv)
    verified = reconcile_remote_receipt(args.output, timeout=args.timeout)
    print(verified.root_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
