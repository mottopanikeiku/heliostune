"""Build and verify the deterministic native RMSNorm H100 stage-gate publication."""

from __future__ import annotations

import argparse
import base64
import binascii
import html
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

import zstandard

from heliostune.artifacts import strict_json_loads, write_bytes_atomic
from heliostune.native_fusion_analysis import analyze_native_fusion_result
from heliostune.native_fusion_executor import NativeFusionExecutionResult
from heliostune.remote_execution import (
    VerifiedRemoteReceipt,
    canonical_json_bytes,
    sha256_bytes,
    verify_remote_receipt,
    verify_remote_receipt_payloads,
)
from heliostune.validation import exact_fields, exact_object

_REPOSITORY = Path(__file__).resolve().parents[1]
_PUBLISHER_PATH = Path(__file__).resolve()
_RAW_PATH = _REPOSITORY / "benchmarks/data/native-rmsnorm-h100.json.zst"
_SUMMARY_PATH = _REPOSITORY / "benchmarks/results/native-rmsnorm-h100-summary.json"
_MANIFEST_PATH = _REPOSITORY / "benchmarks/native-rmsnorm-h100-manifest.json"
_REPORT_PATH = _REPOSITORY / "site/native-rmsnorm-h100.html"
_STUDY_ID = "native-rmsnorm-h100-stage-gate"
_RAW_SCHEMA = "heliostune.native-rmsnorm-h100.raw/1"
_SUMMARY_SCHEMA = "heliostune.native-rmsnorm-h100.summary/1"
_MANIFEST_SCHEMA = "heliostune.native-rmsnorm-h100.manifest/1"
_SUITE_PATH = "benchmarks/suites/residual-rmsnorm-triton-v1.json"
_SUITE_SHA256 = "23f7397f2adee93cd9f7919aaf075c0f8b5e92cd6d4257ce4c54197d3c98035f"
_PLUGIN_PATH = "benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json"
_PLUGIN_SHA256 = "ce4a497113adf1ee82ed995fb4ba671a8a1664d756321499d91187056ca0d815"
_ZSTD_DESCRIPTION = "zstd level=19, threads=1, checksum=true, content_size=false"
_EXPECTED_RECEIPT_COMMON = frozenset(
    {
        "intent.json",
        "journal.jsonl",
        "plugin.json",
        "receipt.json",
        "suite.json",
        "wheel.manifest.json",
    }
)
_LIMITATIONS = (
    "This is one authenticated exploratory H100 workload, not confirmatory evidence.",
    "The first attempt ended unresolved after inline result transport overflow; it establishes no execution result.",
    "Modal provider physical starts, restarts, total GPU time, time bounds, and actual cost are unknown.",
    "The completed attempt does not meet the predeclared 1.1x expansion threshold.",
)
_NONCLAIMS = (
    "No performance claim is made.",
    "No fusion claim is made.",
    "No publication claim is made.",
    "No expansion is authorized.",
)
_PUBLICATION_SOURCES = {
    "analysis": "src/heliostune/native_fusion_analysis.py",
    "executor": "src/heliostune/native_fusion_executor.py",
    "kernel_api": "src/heliostune/fusion_kernels.py",
    "kernel_runtime": "src/heliostune/_fusion_gpu.py",
    "remote_transport": "src/heliostune/remote_execution.py",
    "modal_transport": "modal_fusion_executor.py",
}


@dataclass(frozen=True, slots=True)
class AttemptTruth:
    attempt_id: str
    status: str
    call_id: str
    head_commit: str
    source_sha256: str
    wheel_sha256: str
    terminal_detail: str | None


_ATTEMPT_TRUTH = (
    AttemptTruth(
        "transport-overflow-unresolved",
        "unresolved",
        "fc-01M1B8V37FJP8J9NE0MQYDFEMA",
        "48ef2ea481bdf1ce35498a016f57cdf8602a89a7",
        "02b74382dce99559b93026c19a94ae1f3336ec3dd20244b33f75b208403b949a",
        "bea47623bd1baf18437386f997f3ed1276d62f6fac885c6e2943a3391dc5c2c0",
        "RemoteError: SchemaError('remote result transport exceeds 6144-byte inline limit')",
    ),
    AttemptTruth(
        "completed-inline-result",
        "completed",
        "fc-01M1D6SXDQJHYQY1HEKYB36PBW",
        "8c0429aed216e481def4d9fb1d3fcde30964517d",
        "0f91c248def6e64fa4d41ca8a9e0fcc55cb5e99b29f06e4f091fafe3f0141b93",
        "745b521d6a61ed7aae5f1b6b3311a931a4c9159f81d80f31476075348f0f19c5",
        None,
    ),
)


@dataclass(frozen=True, slots=True)
class AnalyzedAttempt:
    truth: AttemptTruth
    raw: Mapping[str, object]
    verified: VerifiedRemoteReceipt


@dataclass(frozen=True, slots=True)
class Publication:
    compressed: bytes
    raw: bytes
    summary: bytes
    report: bytes
    manifest: bytes


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _json_bytes(value: object) -> bytes:
    return canonical_json_bytes(value)


def _strict_json_bytes(payload: bytes, *, context: str) -> object:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError(f"{context} is not UTF-8") from exc
    value = strict_json_loads(text, source=context)
    if canonical_json_bytes(value) != payload:
        _fail(f"{context} is not canonical strict JSON")
    return value


def _binding(path: str, payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "path": path, "sha256": sha256_bytes(payload)}


def _compress(raw: bytes) -> bytes:
    return zstandard.ZstdCompressor(
        level=19, threads=1, write_checksum=True, write_content_size=False
    ).compress(raw)


def _decompress(compressed: bytes) -> bytes:
    try:
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
            return reader.read()
    except zstandard.ZstdError as exc:
        raise ValueError("published raw artifact is not a valid zstd frame") from exc


def _blob(name: str, payload: bytes) -> dict[str, object]:
    return {
        "base64": base64.b64encode(payload).decode("ascii"),
        "bytes": len(payload),
        "name": name,
        "sha256": sha256_bytes(payload),
    }


def _decode_blob(value: object, *, expected_name: str, context: str) -> bytes:
    data = exact_fields(value, required={"base64", "bytes", "name", "sha256"}, context=context)
    if data["name"] != expected_name or type(data["base64"]) is not str:
        _fail(f"{context} name or base64 payload differs")
    try:
        payload = base64.b64decode(data["base64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{context} has invalid base64") from exc
    if base64.b64encode(payload).decode("ascii") != data["base64"]:
        _fail(f"{context} base64 is not canonical")
    if data["bytes"] != len(payload) or data["sha256"] != sha256_bytes(payload):
        _fail(f"{context} byte count or digest differs")
    return payload


def _regular_inventory(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        _fail(f"receipt directory does not exist: {directory}")
    inventory: dict[str, bytes] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            _fail(f"receipt inventory contains a non-regular entry: {path}")
        inventory[path.name] = path.read_bytes()
    return inventory


def _sidecar_paths(directory: Path) -> tuple[Path, Path]:
    return (
        directory.with_name(f"{directory.name}.remote-intent.json"),
        directory.with_name(f"{directory.name}.remote-attempts.jsonl"),
    )


def _generation_attempt(directory: Path, truth: AttemptTruth) -> dict[str, object]:
    verified = verify_remote_receipt(directory)
    if verified.receipt.status != truth.status:
        _fail(f"{truth.attempt_id} receipt status differs")
    receipt_files = _regular_inventory(directory)
    expected = set(_EXPECTED_RECEIPT_COMMON)
    if truth.status == "completed":
        expected.add("result.json")
    if set(receipt_files) != expected:
        _fail(f"{truth.attempt_id} receipt file inventory differs")
    remote_intent_path, remote_attempts_path = _sidecar_paths(directory)
    remote_intent = remote_intent_path.read_bytes()
    remote_attempts = remote_attempts_path.read_bytes()
    if remote_intent != receipt_files["intent.json"]:
        _fail(f"{truth.attempt_id} remote intent sidecar differs from receipt intent")
    if remote_attempts != receipt_files["journal.jsonl"]:
        _fail(f"{truth.attempt_id} remote attempts sidecar differs from receipt journal")
    return {
        "attempt_id": truth.attempt_id,
        "logical_output_path": verified.intent.output_path,
        "receipt_files": [_blob(name, payload) for name, payload in receipt_files.items()],
        "source_attempt": {
            "remote_attempts": _blob(remote_attempts_path.name, remote_attempts),
            "remote_intent": _blob(remote_intent_path.name, remote_intent),
        },
        "status": truth.status,
    }


def _raw_value(unresolved: Path, completed: Path) -> dict[str, object]:
    return {
        "actual_cost_usd": None,
        "attempts": [
            _generation_attempt(unresolved.absolute(), _ATTEMPT_TRUTH[0]),
            _generation_attempt(completed.absolute(), _ATTEMPT_TRUTH[1]),
        ],
        "limitations": list(_LIMITATIONS),
        "nonclaims": list(_NONCLAIMS),
        "schema": _RAW_SCHEMA,
        "study_id": _STUDY_ID,
    }


def _validate_attempt(value: object, truth: AttemptTruth) -> AnalyzedAttempt:
    attempt = exact_fields(
        value,
        required={
            "attempt_id",
            "logical_output_path",
            "receipt_files",
            "source_attempt",
            "status",
        },
        context=f"raw attempt {truth.attempt_id}",
    )
    if attempt["attempt_id"] != truth.attempt_id or attempt["status"] != truth.status:
        _fail(f"{truth.attempt_id} identity or status differs")
    if type(attempt["logical_output_path"]) is not str or not attempt["logical_output_path"]:
        _fail(f"{truth.attempt_id} logical output path must be nonempty")
    receipt_values = attempt["receipt_files"]
    if type(receipt_values) is not list:
        _fail(f"{truth.attempt_id} receipt files must be an array")
    expected_names = set(_EXPECTED_RECEIPT_COMMON)
    if truth.status == "completed":
        expected_names.add("result.json")
    receipt_files: dict[str, bytes] = {}
    for item in cast(list[object], receipt_values):
        item_object = exact_object(item, context=f"{truth.attempt_id} receipt file")
        name = item_object.get("name")
        if type(name) is not str or name in receipt_files:
            _fail(f"{truth.attempt_id} receipt filename is invalid or duplicated")
        receipt_files[name] = _decode_blob(
            item_object, expected_name=name, context=f"{truth.attempt_id} {name}"
        )
    if set(receipt_files) != expected_names or list(receipt_files) != sorted(receipt_files):
        _fail(f"{truth.attempt_id} receipt file inventory differs")
    source_attempt = exact_fields(
        attempt["source_attempt"],
        required={"remote_attempts", "remote_intent"},
        context=f"{truth.attempt_id} source attempt",
    )
    basename = Path(attempt["logical_output_path"]).name
    remote_intent = _decode_blob(
        source_attempt["remote_intent"],
        expected_name=f"{basename}.remote-intent.json",
        context=f"{truth.attempt_id} remote intent",
    )
    remote_attempts = _decode_blob(
        source_attempt["remote_attempts"],
        expected_name=f"{basename}.remote-attempts.jsonl",
        context=f"{truth.attempt_id} remote attempts",
    )
    if (
        remote_intent != receipt_files["intent.json"]
        or remote_attempts != receipt_files["journal.jsonl"]
    ):
        _fail(f"{truth.attempt_id} source attempt sidecars differ from receipt files")
    verified = verify_remote_receipt_payloads(
        receipt_files, logical_output_path=attempt["logical_output_path"]
    )
    call_ids = {record.call_id for record in verified.journal if record.call_id is not None}
    terminal = verified.journal[-1]
    wheel = exact_object(verified.receipt.bindings["wheel"], context="receipt wheel")
    if (
        verified.receipt.status != truth.status
        or verified.intent.output_path != attempt["logical_output_path"]
        or verified.intent.suite_path != _SUITE_PATH
        or verified.intent.suite_sha256 != _SUITE_SHA256
        or verified.intent.plugin_path != _PLUGIN_PATH
        or verified.intent.plugin_sha256 != _PLUGIN_SHA256
        or verified.intent.head_commit != truth.head_commit
        or verified.intent.source_sha256 != truth.source_sha256
        or verified.intent.wheel_sha256 != truth.wheel_sha256
        or wheel.get("sha256") != truth.wheel_sha256
        or call_ids != {truth.call_id}
        or terminal.detail != truth.terminal_detail
    ):
        _fail(f"{truth.attempt_id} verified provenance differs from frozen truth")
    if truth.status == "completed":
        if (
            not isinstance(verified.result, NativeFusionExecutionResult)
            or verified.result.outcome != "completed"
            or verified.envelope is None
        ):
            _fail(
                "completed receipt did not strict-parse as a completed NativeFusionExecutionResult"
            )
    elif verified.result is not None or verified.envelope is not None:
        _fail("unresolved receipt must not contain a result")
    return AnalyzedAttempt(truth, attempt, verified)


def _validate_raw(raw: bytes) -> tuple[dict[str, object], tuple[AnalyzedAttempt, ...]]:
    value = exact_fields(
        _strict_json_bytes(raw, context="native RMSNorm H100 raw"),
        required={
            "actual_cost_usd",
            "attempts",
            "limitations",
            "nonclaims",
            "schema",
            "study_id",
        },
        context="native RMSNorm H100 raw",
    )
    if (
        value["schema"] != _RAW_SCHEMA
        or value["study_id"] != _STUDY_ID
        or value["actual_cost_usd"] is not None
        or value["limitations"] != list(_LIMITATIONS)
        or value["nonclaims"] != list(_NONCLAIMS)
    ):
        _fail("raw publication identity, cost, limitations, or nonclaims differ")
    attempts = value["attempts"]
    if type(attempts) is not list or len(attempts) != 2:
        _fail("raw publication must contain exactly two attempts")
    analyzed = tuple(
        _validate_attempt(item, truth)
        for item, truth in zip(cast(list[object], attempts), _ATTEMPT_TRUTH, strict=True)
    )
    return dict(value), analyzed


def _completed_result(analyzed: Sequence[AnalyzedAttempt]) -> NativeFusionExecutionResult:
    result = analyzed[1].verified.result
    if not isinstance(result, NativeFusionExecutionResult):
        _fail("completed native result is missing")
    return result


def _attempt_summary(item: AnalyzedAttempt) -> dict[str, object]:
    verified = item.verified
    wheel = exact_object(verified.receipt.bindings["wheel"], context="receipt wheel")
    call_ids = [record.call_id for record in verified.journal if record.call_id is not None]
    return {
        "actual_cost_usd": None,
        "attempt_id": item.truth.attempt_id,
        "call_id": call_ids[0],
        "client_spawn_count": verified.receipt.client_spawn_count,
        "head_commit": verified.intent.head_commit,
        "journal_states": [record.state for record in verified.journal],
        "manifest_sha256": verified.intent.manifest_sha256,
        "per_execution_timeout_s": verified.receipt.per_execution_timeout_s,
        "provider_attempts_observable": False,
        "provider_physical_attempts": None,
        "source_sha256": verified.intent.source_sha256,
        "status": verified.receipt.status,
        "terminal_detail": verified.journal[-1].detail,
        "total_gpu_seconds_upper_bound": None,
        "transport_error": verified.journal[-1].detail
        if item.truth.status == "unresolved"
        else None,
        "wheel_filename": cast(str, wheel["filename"]),
        "wheel_manifest_sha256": cast(Mapping[str, object], wheel["manifest"])["sha256"],
        "wheel_sha256": cast(str, wheel["sha256"]),
    }


def _observations_by_arm(
    result: Mapping[str, object],
) -> dict[str, dict[str, Mapping[str, object]]]:
    values = result["observations"]
    if type(values) is not list:
        _fail("native result observations must be an array")
    output: dict[str, dict[str, Mapping[str, object]]] = {}
    for value in cast(list[object], values):
        observation = exact_object(value, context="native result observation")
        arm_id = cast(str, observation["arm_id"])
        stage = cast(str, observation["stage"])
        output.setdefault(arm_id, {})[stage] = observation
    return output


def _evidence_for_arm(values: object, arm_id: str, *, context: str) -> Mapping[str, object]:
    evidence = exact_object(values, context=context)
    matches = [
        exact_object(value, context=f"{context} record")
        for value in evidence.values()
        if isinstance(value, Mapping) and value.get("arm_id") == arm_id
    ]
    if len(matches) != 1:
        _fail(f"{context} must contain exactly one record for {arm_id}")
    return matches[0]


def _summary(analyzed: Sequence[AnalyzedAttempt]) -> dict[str, object]:
    result = _completed_result(analyzed)
    result_value = result.to_dict()
    analysis = analyze_native_fusion_result(result)
    analysis_bytes = _json_bytes(analysis)
    observations = _observations_by_arm(result_value)
    candidates: list[dict[str, object]] = []
    for candidate in cast(list[dict[str, object]], analysis["candidates"]):
        arm_id = cast(str, candidate["arm_id"])
        candidates.append(
            {
                "analysis": candidate,
                "arm_id": arm_id,
                "one_kernel_evidence": _evidence_for_arm(
                    result_value["profile_evidence"], arm_id, context="profile evidence"
                ),
                "resource_evidence": _evidence_for_arm(
                    result_value["resource_evidence"], arm_id, context="resource evidence"
                ),
                "timing_observation": observations[arm_id]["timing"],
                "validation_evidence": _evidence_for_arm(
                    result_value["validation_evidence"], arm_id, context="validation evidence"
                ),
            }
        )
    baselines: list[dict[str, object]] = []
    for baseline in cast(list[dict[str, object]], analysis["baselines"]):
        arm_id = cast(str, baseline["arm_id"])
        baselines.append(
            {
                "analysis": baseline,
                "arm_id": arm_id,
                "correctness_observation": observations[arm_id]["correctness"],
                "timing_observation": observations[arm_id]["timing"],
            }
        )
    return {
        "actual_cost_usd": None,
        "analysis": analysis,
        "analysis_binding": {
            "bytes": len(analysis_bytes),
            "sha256": sha256_bytes(analysis_bytes),
        },
        "attempts": [_attempt_summary(item) for item in analyzed],
        "baselines": baselines,
        "candidates": candidates,
        "claims": [],
        "counts": {
            "attempts": 2,
            "completed": 1,
            "failed": 0,
            "unresolved": 1,
            "eligible_candidates": len(candidates),
            "baselines": len(baselines),
        },
        "decision_display": "STOP_BELOW_THRESHOLD",
        "expansion_authorized": False,
        "fusion_claim": False,
        "limitations": list(_LIMITATIONS),
        "nonclaims": list(_NONCLAIMS),
        "performance_claim": False,
        "provider_accounting": {
            "actual_cost_usd": None,
            "cost_status": "unknown",
            "provider_attempts_observable": False,
            "provider_physical_attempts": None,
            "total_gpu_seconds_upper_bound": None,
            "total_time_upper_bound_s": None,
        },
        "publication_eligible": False,
        "schema": _SUMMARY_SCHEMA,
        "study_id": _STUDY_ID,
    }


def _render_report(summary: Mapping[str, object]) -> bytes:
    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    candidate_rows: list[str] = []
    for item in cast(list[Mapping[str, object]], summary["candidates"]):
        analysis = cast(Mapping[str, object], item["analysis"])
        config = cast(Mapping[str, object], analysis["config"])
        resource = cast(Mapping[str, object], item["resource_evidence"])
        profile = cast(Mapping[str, object], item["one_kernel_evidence"])
        validation = cast(Mapping[str, object], item["validation_evidence"])
        errors = cast(Sequence[object], analysis["errors"])
        candidate_rows.append(
            "<tr>"
            f"<td><code>{esc(item['arm_id'])}</code></td>"
            f"<td>{esc(config['num_warps'])}</td><td>{esc(config['block_size'])}</td>"
            f"<td>{esc(analysis['median_ms'])}</td><td>{esc(resource['n_regs'])}</td>"
            f"<td>{esc(resource['n_spills'])}</td><td>{esc(profile['cuda_event_count'])}</td>"
            f"<td>{esc(profile['one_kernel_gate_passed'])}</td>"
            f"<td>{esc(validation['validation_gate_passed'])}</td>"
            f"<td>{esc('; '.join(map(str, errors)) if errors else 'none')}</td></tr>"
        )
    baseline_rows: list[str] = []
    for item in cast(list[Mapping[str, object]], summary["baselines"]):
        analysis = cast(Mapping[str, object], item["analysis"])
        errors = cast(Sequence[object], analysis["errors"])
        baseline_rows.append(
            "<tr>"
            f"<td><code>{esc(item['arm_id'])}</code></td><td>{esc(analysis['median_ms'])}</td>"
            f"<td>{esc(analysis['stages'])}</td>"
            f"<td>{esc('; '.join(map(str, errors)) if errors else 'none')}</td></tr>"
        )
    attempt_rows: list[str] = []
    failures: list[str] = []
    for item in cast(list[Mapping[str, object]], summary["attempts"]):
        detail = item["transport_error"]
        attempt_rows.append(
            "<tr>"
            f"<td>{esc(item['attempt_id'])}</td><td>{esc(item['status'])}</td>"
            f"<td><code>{esc(item['call_id'])}</code></td><td><code>{esc(item['wheel_sha256'])}</code></td>"
            f"<td>{esc(detail if detail is not None else 'none')}</td></tr>"
        )
        if detail is not None:
            failures.append(f"<li><strong>{esc(item['attempt_id'])}:</strong> {esc(detail)}</li>")
    nonclaims = "".join(
        f"<li>{esc(value)}</li>" for value in cast(Sequence[object], summary["nonclaims"])
    )
    limitations = "".join(
        f"<li>{esc(value)}</li>" for value in cast(Sequence[object], summary["limitations"])
    )
    analysis = cast(Mapping[str, object], summary["analysis"])
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Native RMSNorm H100 stage gate</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 78rem; padding: 2rem; line-height: 1.45; }}
h1, h2 {{ line-height: 1.15; }}
.decision {{ border: .2rem solid currentColor; font-size: 2rem; font-weight: 800; padding: 1rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #888; padding: .45rem; text-align: left; vertical-align: top; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<main>
<h1>Native RMSNorm on H100: authenticated stage gate</h1>
<p class="decision">STOP_BELOW_THRESHOLD</p>
<p>Winner <code>{esc(analysis["winner_id"])}</code> measured {esc(analysis["winner_median_ms"])} ms; best baseline <code>{esc(analysis["best_baseline_id"])}</code> measured {esc(analysis["best_baseline_median_ms"])} ms. Fair speedup {esc(analysis["speedup"])} versus the predeclared {esc(analysis["threshold"])} threshold.</p>
<h2>All native configurations</h2>
<table><thead><tr><th>Arm</th><th>Warps</th><th>Block</th><th>Median ms</th><th>Registers</th><th>Spills</th><th>CUDA kernels</th><th>One-kernel gate</th><th>Validation</th><th>Failures</th></tr></thead><tbody>{"".join(candidate_rows)}</tbody></table>
<h2>Both baselines</h2>
<table><thead><tr><th>Arm</th><th>Median ms</th><th>Stages</th><th>Failures</th></tr></thead><tbody>{"".join(baseline_rows)}</tbody></table>
<h2>Attempts and transport</h2>
<table><thead><tr><th>Attempt</th><th>Status</th><th>FunctionCall ID</th><th>Wheel SHA-256</th><th>Transport error</th></tr></thead><tbody>{"".join(attempt_rows)}</tbody></table>
<h2>Failures</h2><ul>{"".join(failures) if failures else "<li>None.</li>"}</ul>
<h2>Nonclaims</h2><ul>{nonclaims}</ul>
<h2>Limitations</h2><ul>{limitations}</ul>
<p>Actual cost: unknown (<code>null</code>). Publication eligible: false. Fusion claim: false.</p>
</main>
</body>
</html>
"""
    return document.encode("utf-8")


def _source_bindings(result: NativeFusionExecutionResult) -> dict[str, object]:
    current: dict[str, object] = {
        "suite": _binding(_SUITE_PATH, (_REPOSITORY / _SUITE_PATH).read_bytes()),
        "plugin": _binding(_PLUGIN_PATH, (_REPOSITORY / _PLUGIN_PATH).read_bytes()),
    }
    for role, relative in _PUBLICATION_SOURCES.items():
        current[role] = _binding(relative, (_REPOSITORY / relative).read_bytes())
    return {
        "current": current,
        "executed_remote_package": dict(result.executor_sources),
    }


def _manifest(
    analyzed: Sequence[AnalyzedAttempt],
    *,
    compressed: bytes,
    raw: bytes,
    summary: bytes,
    report: bytes,
) -> dict[str, object]:
    result = _completed_result(analyzed)
    attempts = [_attempt_summary(item) for item in analyzed]
    return {
        "actual_cost_usd": None,
        "artifacts": {
            "publisher": _binding(
                "scripts/publish_native_rmsnorm_h100.py", _PUBLISHER_PATH.read_bytes()
            ),
            "raw": {
                "compressed_bytes": len(compressed),
                "compressed_sha256": sha256_bytes(compressed),
                "decompressed_bytes": len(raw),
                "decompressed_sha256": sha256_bytes(raw),
                "format": _ZSTD_DESCRIPTION,
                "path": "benchmarks/data/native-rmsnorm-h100.json.zst",
            },
            "report": _binding("site/native-rmsnorm-h100.html", report),
            "summary": _binding("benchmarks/results/native-rmsnorm-h100-summary.json", summary),
        },
        "attempts": attempts,
        "commands": {
            "build": (
                "uv run python scripts/publish_native_rmsnorm_h100.py build "
                "--unresolved artifacts/fusion-remote/residual-rmsnorm-triton-v1-20260831T064037385754112 "
                "--completed artifacts/fusion-remote/residual-rmsnorm-triton-v1-20260901T004420761029839"
            ),
            "check": "uv run python scripts/publish_native_rmsnorm_h100.py --check",
        },
        "decision": "stop_below_threshold",
        "decision_display": "STOP_BELOW_THRESHOLD",
        "expansion_authorized": False,
        "fusion_claim": False,
        "limitations": list(_LIMITATIONS),
        "nonclaims": list(_NONCLAIMS),
        "performance_claim": False,
        "publication_eligible": False,
        "schema": _MANIFEST_SCHEMA,
        "sources": _source_bindings(result),
        "study_id": _STUDY_ID,
    }


def _derive(compressed: bytes) -> Publication:
    raw = _decompress(compressed)
    if _compress(raw) != compressed:
        _fail("published raw artifact is not the deterministic zstd encoding")
    _, analyzed = _validate_raw(raw)
    summary = _json_bytes(_summary(analyzed))
    report = _render_report(
        cast(Mapping[str, object], _strict_json_bytes(summary, context="summary"))
    )
    manifest = _json_bytes(
        _manifest(analyzed, compressed=compressed, raw=raw, summary=summary, report=report)
    )
    return Publication(compressed, raw, summary, report, manifest)


def build(*, unresolved: Path, completed: Path) -> None:
    raw = _json_bytes(_raw_value(unresolved, completed))
    compressed = _compress(raw)
    publication = _derive(compressed)
    write_bytes_atomic(_RAW_PATH, publication.compressed)
    write_bytes_atomic(_SUMMARY_PATH, publication.summary)
    write_bytes_atomic(_REPORT_PATH, publication.report)
    write_bytes_atomic(_MANIFEST_PATH, publication.manifest)
    print("published native RMSNorm H100 stage gate: STOP_BELOW_THRESHOLD")
    print(
        f"raw={sha256_bytes(publication.compressed)} summary={sha256_bytes(publication.summary)} "
        f"manifest={sha256_bytes(publication.manifest)} report={sha256_bytes(publication.report)}"
    )


def check() -> None:
    for path in (_RAW_PATH, _SUMMARY_PATH, _MANIFEST_PATH, _REPORT_PATH, _PUBLISHER_PATH):
        if not path.is_file():
            _fail(
                f"required committed publication file is missing: {path.relative_to(_REPOSITORY)}"
            )
    publication = _derive(_RAW_PATH.read_bytes())
    committed = {
        _SUMMARY_PATH: publication.summary,
        _MANIFEST_PATH: publication.manifest,
        _REPORT_PATH: publication.report,
    }
    for path, expected in committed.items():
        actual = path.read_bytes()
        if actual != expected:
            _fail(
                f"{path.relative_to(_REPOSITORY)} is not byte-derived from committed raw evidence"
            )
    print("checked native RMSNorm H100 publication from committed compressed raw evidence only")
    print(
        f"raw={sha256_bytes(publication.compressed)} summary={sha256_bytes(publication.summary)} "
        f"manifest={sha256_bytes(publication.manifest)} report={sha256_bytes(publication.report)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed outputs from raw")
    subparsers = parser.add_subparsers(dest="command")
    build_parser = subparsers.add_parser("build", help="build publication from two receipts")
    build_parser.add_argument("--unresolved", required=True, type=Path)
    build_parser.add_argument("--completed", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.check:
        if args.command is not None:
            parser.error("--check cannot be combined with a command")
        check()
    elif args.command == "build":
        build(unresolved=args.unresolved, completed=args.completed)
    else:
        parser.error("choose build or --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
