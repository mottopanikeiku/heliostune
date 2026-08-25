"""Publish the terminal pre-H200 Parhelion v3 pilot failure evidence."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from heliostune.artifacts import read_json, write_bytes_atomic, write_json_atomic
from heliostune.collection import AttemptJournal, AttemptRecord
from heliostune.protocol import load_v3_protocol, require_v3_runtime
from heliostune.v3_artifacts import sha256_file

_REPO = Path(__file__).resolve().parents[1]
_PROTOCOL = _REPO / "benchmarks/parhelion-v3-development-protocol.json"
_JOURNAL = _REPO / "benchmarks/data/parhelion-v3-pilot-failure.attempts.jsonl"
_OUTPUT = _REPO / "benchmarks/parhelion-v3-validation-failure.json"
_FAILED_HEAD = "c0cdf0e87713aff09ee5a66b23cd366d4bae7817"
_FAILED_HEAD_SHA256 = "ab624c9fee5643031d8914ed47aaf74df637b9db6b4379fd8964a650ca763564"
_REQUEST_SHA256 = "6aa62ccd82de0c9fbd6d6d4c6b8f4a186445aef7b171b5eccdc118f84cd2e60f"
_PROTOCOL_SHA256 = "755ea87959edbeb1d50f1d9a5dea46ed6cd5e1aa5f8f964416767546109139cb"
_CONFIG_KEYS_SHA256 = "9111111e5b65c4abbfcde3c8e7573c10573cf964297e01b2955b47f56a891955"
_WHEEL_SHA256 = "4672d8c6edd57b8936b7d94a6a5049552a90c2f916e3c079b0d589ca19beaeb4"
_WHEEL_SOURCE_SHA256 = "9af18ff2005e0f784821ece4f761b3f29f8d99c21c98cddfd1348adbf98861a2"
_CALL_ID = "fc-01M0V2ZWYR8GKXNC0MB32YFPFF"
_PILOT_APP_ID = "ap-nWqf5qjkL9CdGVuL5lWcl6"
_IMAGE_FAILURE_APP_ID = "ap-23g6jX4qXrHvzDlbAPMa7z"
_RESUME_APP_ID = "ap-uFyQ3siR4lkNblqmdQBSIS"

_ABSENT_ARTIFACTS = (
    "benchmarks/data/parhelion-v3-pilot.jsonl.zst",
    "benchmarks/data/parhelion-v3-pilot.jsonl.zst.manifest.json",
    "benchmarks/data/parhelion-v3-candidate-bank0.jsonl.zst",
    "benchmarks/data/parhelion-v3-validation.jsonl.zst",
    "benchmarks/parhelion-v3-config-manifest.json",
    "benchmarks/results/parhelion-v3-a100-selection.json",
    "benchmarks/parhelion-v3-h200-freeze.json",
    "benchmarks/data/parhelion-v3-h200.jsonl.zst",
    "benchmarks/data/parhelion-v3-final.jsonl.zst",
    "benchmarks/results/parhelion-v3-final.json",
    "site/parhelion-v3.html",
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(_REPO.resolve()).as_posix()


def _same_call(left: AttemptRecord, right: AttemptRecord) -> bool:
    fields = (
        "request_sha256",
        "protocol_sha256",
        "config_manifest_sha256",
        "wheel_sha256",
        "head_sha256",
        "gpu",
        "bank",
        "call_id",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _load_evidence(
    protocol_path: Path,
    journal_path: Path,
) -> tuple[dict[str, object], tuple[AttemptRecord, AttemptRecord]]:
    protocol = load_v3_protocol(protocol_path)
    require_v3_runtime(protocol)
    journal = AttemptJournal.load(journal_path)
    if len(journal.records) != 2:
        raise ValueError("pilot journal must contain exactly spawned and failed records")
    spawned, failed = journal.records
    if spawned.status != "spawned" or failed.status != "failed":
        raise ValueError("pilot journal must terminate spawned -> failed")
    if not _same_call(spawned, failed):
        raise ValueError("pilot terminal record does not bind the spawned FunctionCall")
    expected = {
        "request_sha256": _REQUEST_SHA256,
        "protocol_sha256": _PROTOCOL_SHA256,
        "config_manifest_sha256": _CONFIG_KEYS_SHA256,
        "wheel_sha256": _WHEEL_SHA256,
        "head_sha256": _FAILED_HEAD_SHA256,
        "gpu": "L4",
        "bank": 0,
        "call_id": _CALL_ID,
    }
    for field, value in expected.items():
        if getattr(spawned, field) != value:
            raise ValueError(f"pilot {field} does not match the recorded campaign")
    if failed.error != "RemoteError:":
        raise ValueError("pilot terminal error must be the observed Modal RemoteError")
    if sha256_file(protocol_path) != _PROTOCOL_SHA256:
        raise ValueError("development protocol bytes changed after the paid pilot")
    if hashlib.sha256(_FAILED_HEAD.encode()).hexdigest() != _FAILED_HEAD_SHA256:
        raise AssertionError("failed campaign HEAD binding constant is invalid")
    for relative in _ABSENT_ARTIFACTS:
        if (_REPO / relative).exists():
            raise ValueError(f"pre-H200 failure outcome forbids artifact {relative}")
    return protocol, (spawned, failed)


def build_manifest(
    *,
    protocol_path: Path = _PROTOCOL,
    journal_path: Path = _JOURNAL,
) -> dict[str, object]:
    """Return the deterministic manifest after enforcing runtime and journal gates."""
    protocol, records = _load_evidence(protocol_path, journal_path)
    spawned, failed = records
    implementation = cast(Mapping[str, object], protocol["implementation_sha256"])
    failure_rule = cast(Mapping[str, object], protocol["failure_outcomes"])["pre_h200"]
    return {
        "schema_version": 1,
        "study_id": "parhelion-v3-validation-failure",
        "outcome": "pilot_failed_before_measurement_collection",
        "campaign_status": "terminated_pre_h200_under_predeclared_failure_rule",
        "protocol": {
            "path": _relative(protocol_path),
            "sha256": sha256_file(protocol_path),
            "status": protocol["protocol_status"],
            "pre_h200_failure_rule": failure_rule,
            "analysis_runtime": protocol["analysis_runtime"],
            "software": protocol["software"],
            "implementation_sha256": dict(implementation),
        },
        "failed_candidate": {
            "git_commit": _FAILED_HEAD,
            "git_commit_sha256": _FAILED_HEAD_SHA256,
            "wheel": {
                "filename": "heliostune-0.4.0.dev0-py3-none-any.whl",
                "sha256": _WHEEL_SHA256,
                "packaged_source_sha256": _WHEEL_SOURCE_SHA256,
            },
        },
        "collection_binding": {
            "request_sha256": spawned.request_sha256,
            "protocol_sha256": spawned.protocol_sha256,
            "pilot_config_key_manifest_sha256": spawned.config_manifest_sha256,
            "wheel_sha256": spawned.wheel_sha256,
            "head_sha256": spawned.head_sha256,
        },
        "pilot_function_call": {
            "app_id": _PILOT_APP_ID,
            "app_url": f"https://modal.com/apps/mottopanikeiku/main/{_PILOT_APP_ID}",
            "app_state": "stopped",
            "app_created_at_utc": "2026-08-24T23:51:38Z",
            "app_stopped_at_utc": "2026-08-25T00:03:51Z",
            "function_call_id": spawned.call_id,
            "gpu": spawned.gpu,
            "bank": spawned.bank,
            "spawned_at_utc": spawned.timestamp_utc,
            "terminal_status": failed.status,
            "terminal_recorded_at_utc": failed.timestamp_utc,
            "terminal_client_error": failed.error,
            "attempt_journal": {
                "path": _relative(journal_path),
                "sha256": sha256_file(journal_path),
                "bytes": journal_path.stat().st_size,
                "records": len(records),
            },
        },
        "failure": {
            "category": "remote_container_import_failure",
            "exception_type": "RuntimeError",
            "message": ("run `uv run python scripts/build_modal_wheel.py` before Modal; found []"),
            "source": "modal_bench.py:_configured_modal_wheel",
            "remote_path": "/root/modal_bench.py",
            "first_observed_at_utc": "2026-08-24T23:51:51Z",
            "crash_loop_observed_at_utc": "2026-08-24T23:56:49Z",
            "cause": (
                "the remote module imported before HELIOSTUNE_MODAL_WHEEL was present and "
                "searched the container-relative artifacts/modal-wheel directory"
            ),
        },
        "orchestration_attempts": [
            {
                "app_id": _IMAGE_FAILURE_APP_ID,
                "app_state": "stopped",
                "app_created_at_utc": "2026-08-24T23:26:55Z",
                "app_stopped_at_utc": "2026-08-24T23:28:43Z",
                "status": "image_build_failed_before_function_call_spawn",
                "function_calls_spawned": 0,
                "attempt_journal": None,
                "error": (
                    "pip 25 rejected the generic /root/heliostune.whl path before a paid "
                    "FunctionCall existed"
                ),
            },
            {
                "app_id": _PILOT_APP_ID,
                "app_state": "stopped",
                "app_created_at_utc": "2026-08-24T23:51:38Z",
                "app_stopped_at_utc": "2026-08-25T00:03:51Z",
                "status": "paid_function_call_failed_during_remote_import",
                "function_calls_spawned": 1,
                "function_call_id": _CALL_ID,
            },
            {
                "app_id": _RESUME_APP_ID,
                "app_state": "stopped",
                "app_created_at_utc": "2026-08-25T00:04:14Z",
                "app_stopped_at_utc": "2026-08-25T00:04:16Z",
                "status": "resume_only_restored_terminal_function_call",
                "function_calls_spawned": 0,
                "function_call_id": _CALL_ID,
                "result": "FunctionCall.from_id().get() returned RemoteError",
            },
        ],
        "terminal_record_provenance": {
            "retrieval": "zero-spawn resume of the existing FunctionCall ID",
            "serialization_note": (
                "the first resume observed RemoteError but failed to append it because the "
                "rendered exception carried surrounding whitespace; the terminal record was "
                "then appended from that observed result with the normalized value RemoteError:"
            ),
        },
        "policy_application": {
            "pilot_retried": False,
            "candidate_matrix_invoked": False,
            "a100_validation_invoked": False,
            "h200_invoked": False,
            "replacement_run_permitted": False,
            "release_scope": "0.4.0 software, causal addendum, protocol, and failure evidence",
            "performance_report": "not produced",
        },
        "artifacts_not_produced": list(_ABSENT_ARTIFACTS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-journal", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check and args.source_journal is not None:
        parser.error("--source-journal cannot be combined with --check")
    if args.check:
        expected = build_manifest()
        if read_json(_OUTPUT) != expected:
            raise ValueError(f"{_OUTPUT} is not the deterministic failure manifest")
        print(f"verified={_relative(_OUTPUT)} journal_sha256={sha256_file(_JOURNAL)}")
        return 0
    if args.source_journal is None:
        parser.error("--source-journal is required when publishing")
    _load_evidence(_PROTOCOL, args.source_journal)
    write_bytes_atomic(_JOURNAL, args.source_journal.read_bytes())
    write_json_atomic(_OUTPUT, build_manifest())
    print(f"wrote={_relative(_OUTPUT)} journal_sha256={sha256_file(_JOURNAL)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
