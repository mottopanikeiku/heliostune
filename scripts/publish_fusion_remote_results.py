"""Publish deterministic post-hoc exploratory evidence from four H100 fusion attempts.

Generation reads the operator-retained local attempts. ``--check`` reads only the
committed publication and other committed repository inputs; it never consults
those local attempts or their pointer files.
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast
from urllib.parse import urlparse

import zstandard

from heliostune.artifacts import strict_json_loads, write_bytes_atomic
from heliostune.hardware import expectation_for_gpu, validate_hardware
from heliostune.local_executor import LocalExecutionResult
from heliostune.remote_execution import (
    RemoteIntent,
    RemoteJournalRecord,
    RemoteResultEnvelope,
    canonical_json_bytes,
    canonical_json_line_bytes,
    encode_remote_request,
    open_remote_records,
    remote_artifact_paths,
    sha256_bytes,
    verify_remote_receipt,
    verify_remote_receipt_payloads,
)
from heliostune.validation import (
    exact_fields,
    exact_int,
    exact_object,
    finite_float,
    nonblank_string,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_RAW_PATH = _REPOSITORY / "benchmarks/data/fusion-remote-exploratory.json.zst"
_SUMMARY_PATH = _REPOSITORY / "benchmarks/results/fusion-remote-exploratory-summary.json"
_MANIFEST_PATH = _REPOSITORY / "benchmarks/fusion-remote-exploratory-manifest.json"
_PUBLISHER_PATH = Path(__file__).resolve()
_STUDY_ID = "fusion-remote-h100-exploratory"
_RAW_SCHEMA = "heliostune.fusion-remote-exploratory.raw/1"
_SUMMARY_SCHEMA = "heliostune.fusion-remote-exploratory.summary/1"
_MANIFEST_SCHEMA = "heliostune.fusion-remote-exploratory.manifest/1"
_MODAL_WORKSPACE = "mottopanikeiku"
_PLUGIN_PATH = "benchmarks/plugins/fusion-reference-plugin-v1.json"
_PLUGIN_SHA256 = "9d696f135a5e62ef622a88d85a7bb03e8fa76bddd0bf57ebf20b2eb4c1d1edc1"
_ZSTD_DESCRIPTION = "zstd level=19, threads=1, checksum=true, content_size=false"

_METHOD = {
    "analysis_status": "post_hoc_exploratory",
    "design": "four retained remote attempts analyzed after execution",
    "fusion_claim": False,
    "performance_inference": "not_tested",
    "publication_eligible": False,
    "report_status": "not_created",
    "superiority_claim": False,
}

_CLAIM_CLASSIFICATION = {
    "analysis": "exploratory",
    "completed_correctness_timing_compile_metrics": "supported_only_as_measured_fact",
    "fusion": "not_tested",
    "performance": "descriptive",
    "superiority": "not_tested",
}

_LIMITATIONS = (
    "Two gated-MLP calls ended unresolved after cancellation requests; no result receipts exist for them.",
    "The two completed calls are single returned observations collected without a prespecified comparative analysis plan.",
    "Reference/candidate ratios are descriptive for the returned medians only; no uncertainty or superiority test was performed.",
    "The candidate and reference arithmetic are not evidence of kernel fusion; every returned environment explicitly records fusion_claim=false.",
    "Modal provider physical starts and restarts are unobservable, so provider attempt count, total GPU time, and actual cost are unknown.",
    "The retained evidence has no attestation and is not publication eligible.",
)


@dataclass(frozen=True, slots=True)
class AttemptSpec:
    attempt_id: str
    pointer_path: Path
    output_relative: str
    app_id: str
    call_id: str
    status: str
    suite_path: str
    suite_id: str
    candidate_arm: str
    reference_arm: str

    @property
    def app_url(self) -> str:
        return f"https://modal.com/apps/{_MODAL_WORKSPACE}/main/{self.app_id}"

    @property
    def output_path(self) -> Path:
        return _REPOSITORY / self.output_relative


_SPECS = (
    AttemptSpec(
        "gated-mlp-01-unresolved",
        Path("/home/alp/gated-mlp-output-path.txt"),
        "artifacts/fusion-remote/gated-mlp-epilogue-v1-20260829T032317860136157",
        "ap-6A3Flv17wjZa7pSgL4qtJe",
        "fc-01M15RPHFH9ENEXWD1WYBX1KNW",
        "unresolved",
        "benchmarks/suites/gated-mlp-epilogue-v1.json",
        "gated-mlp-epilogue-reference",
        "mlp-candidate",
        "mlp-reference",
    ),
    AttemptSpec(
        "gated-mlp-02-unresolved",
        Path("/home/alp/gated-mlp-output-path-2.txt"),
        "artifacts/fusion-remote/gated-mlp-epilogue-v1-20260829T044455620412604",
        "ap-F06mCUOAA53kliXeB4K11V",
        "fc-01M15XBYSXW0JBYBRXVN9YEHSP",
        "unresolved",
        "benchmarks/suites/gated-mlp-epilogue-v1.json",
        "gated-mlp-epilogue-reference",
        "mlp-candidate",
        "mlp-reference",
    ),
    AttemptSpec(
        "gated-mlp-03-completed",
        Path("/home/alp/gated-mlp-output-path-3.txt"),
        "artifacts/fusion-remote/gated-mlp-epilogue-v1-20260829T063859723199271",
        "ap-0ybLik0aGa1sV9QlGbZGke",
        "fc-01M163X38NG8E1Q7X0VX53JKYP",
        "completed",
        "benchmarks/suites/gated-mlp-epilogue-v1.json",
        "gated-mlp-epilogue-reference",
        "mlp-candidate",
        "mlp-reference",
    ),
    AttemptSpec(
        "residual-rmsnorm-01-completed",
        Path("/home/alp/rmsnorm-output-path.txt"),
        "artifacts/fusion-remote/residual-rmsnorm-v1-20260829T075248920315545",
        "ap-jKSyIjohIzchDqd6kIu1kY",
        "fc-01M168458F6DDQ5887S83W6S46",
        "completed",
        "benchmarks/suites/residual-rmsnorm-v1.json",
        "residual-rmsnorm-reference",
        "rmsnorm-candidate",
        "rmsnorm-reference",
    ),
)

_JOURNAL_STATES = {
    "unresolved": (
        "intent",
        "spawned",
        "retrieval_started",
        "cancellation_requested",
        "unresolved",
    ),
    "completed": ("intent", "spawned", "retrieval_started", "completed"),
}


@dataclass(frozen=True, slots=True)
class AnalyzedAttempt:
    spec: AttemptSpec
    value: Mapping[str, object]
    intent: RemoteIntent
    journal: tuple[RemoteJournalRecord, ...]
    result: LocalExecutionResult | None
    envelope: RemoteResultEnvelope | None


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


def _validate_binding(value: object, payload: bytes, *, path: str, context: str) -> None:
    data = exact_fields(value, required={"bytes", "path", "sha256"}, context=context)
    if data != _binding(path, payload):
        _fail(f"{context} differs from reconstructed committed content")


def _compress(raw: bytes) -> bytes:
    return zstandard.ZstdCompressor(
        level=19,
        threads=1,
        write_checksum=True,
        write_content_size=False,
    ).compress(raw)


def _decompress(compressed: bytes) -> bytes:
    try:
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
            return reader.read()
    except zstandard.ZstdError as exc:
        raise ValueError("published raw artifact is not a valid zstd frame") from exc


def _read_canonical_object(path: Path, *, context: str) -> tuple[bytes, dict[str, object]]:
    payload = path.read_bytes()
    return payload, exact_object(_strict_json_bytes(payload, context=context), context=context)


def _journal_from_values(
    value: object,
    *,
    request_digest: str,
    call_id: str,
    status: str,
    context: str,
) -> tuple[tuple[RemoteJournalRecord, ...], bytes]:
    if type(value) is not list:
        _fail(f"{context} must be an array")
    items = cast(list[object], value)
    records = tuple(RemoteJournalRecord.from_dict(item) for item in items)
    payload = b"".join(canonical_json_line_bytes(record.to_dict()) for record in records)
    expected_states = _JOURNAL_STATES[status]
    if tuple(record.state for record in records) != expected_states:
        _fail(f"{context} does not have the exact {status} state sequence")
    for sequence, record in enumerate(records):
        if record.sequence != sequence or record.request_digest != request_digest:
            _fail(f"{context} sequence/request binding differs")
        expected_call = None if sequence == 0 else call_id
        if record.call_id != expected_call:
            _fail(f"{context} FunctionCall ID differs")
    return records, payload


def _journal_values(payload: bytes, *, context: str) -> list[object]:
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines(keepends=True)
    except UnicodeError as exc:
        raise ValueError(f"{context} is not UTF-8") from exc
    if not lines or any(not line.endswith("\n") for line in lines):
        _fail(f"{context} is not complete JSONL")
    values: list[object] = []
    for line in lines:
        value = strict_json_loads(line, source=context)
        record = RemoteJournalRecord.from_dict(value)
        if canonical_json_line_bytes(record.to_dict()).decode("utf-8") != line:
            _fail(f"{context} is not canonical JSONL")
        values.append(record.to_dict())
    return values


def _source_relative(path: Path) -> str:
    try:
        return path.relative_to(_REPOSITORY).as_posix()
    except ValueError:
        return str(path)


def _validate_app(spec: AttemptSpec, value: object) -> None:
    data = exact_fields(
        value,
        required={"app_id", "app_url", "artifact_binding", "identity_provenance"},
        context=f"{spec.attempt_id} app provenance",
    )
    expected = {
        "app_id": spec.app_id,
        "app_url": spec.app_url,
        "artifact_binding": "none",
        "identity_provenance": "operator_recorded",
    }
    if data != expected:
        _fail(f"{spec.attempt_id} operator-recorded app identity/URL differs")
    parsed = urlparse(spec.app_url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "modal.com"
        or parsed.path != f"/apps/{_MODAL_WORKSPACE}/main/{spec.app_id}"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        _fail(f"{spec.attempt_id} operator-recorded app URL is not canonical")


def _call_provenance(spec: AttemptSpec) -> dict[str, object]:
    return {
        "function_call_id": spec.call_id,
        "identity_provenance": "artifact_bound_remote_journal",
    }


def _app_provenance(spec: AttemptSpec) -> dict[str, object]:
    return {
        "app_id": spec.app_id,
        "app_url": spec.app_url,
        "artifact_binding": "none",
        "identity_provenance": "operator_recorded",
    }


def _read_pointer(spec: AttemptSpec) -> tuple[bytes, str]:
    payload = spec.pointer_path.read_bytes()
    expected = f"{spec.output_relative}\n".encode()
    if payload != expected:
        _fail(f"{spec.attempt_id} pointer is not the exact supplied output path")
    return payload, spec.output_relative


def _support_payloads(output: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    names = ("suite.json", "plugin.json", "wheel.manifest.json")
    values: dict[str, object] = {}
    payloads: dict[str, bytes] = {}
    for name in names:
        payload = (output / name).read_bytes()
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ValueError(f"receipt {name} is not UTF-8") from exc
        strict_json_loads(text, source=f"receipt {name}")
        key = f"{name.removesuffix('.json').replace('.', '_')}_utf8"
        values[key] = text
        payloads[name] = payload
    return values, payloads


def _generation_attempt(spec: AttemptSpec) -> dict[str, object]:
    pointer_payload, pointer_value = _read_pointer(spec)
    output = spec.output_path
    intent_path, journal_path = remote_artifact_paths(output)

    if spec.status == "unresolved":
        if output.exists() or (output / "receipt.json").exists():
            _fail(f"{spec.attempt_id} must have no receipt/output directory")
        records = open_remote_records(output)
        try:
            records.assert_parent_identity()
            if records.journal.state != "unresolved" or records.journal.call_id != spec.call_id:
                _fail(f"{spec.attempt_id} is not exactly unresolved for the supplied call")
            intent_payload = records.intent_bytes()
            journal_payload = records.journal.bytes()
            intent = records.intent
        finally:
            records.close()
        receipt_value: object = None
        result_value: object = None
        support_value: object = None
        source_files = {
            "operator_pointer": _binding(str(spec.pointer_path), pointer_payload),
            "remote_intent": _binding(_source_relative(intent_path), intent_payload),
            "remote_journal": _binding(_source_relative(journal_path), journal_payload),
        }
    else:
        verified = verify_remote_receipt(output)
        if (
            verified.receipt.status != "completed"
            or verified.result is None
            or verified.envelope is None
            or verified.result.outcome != "completed"
        ):
            _fail(f"{spec.attempt_id} is not an exact completed receipt/result")
        if verified.journal[-1].call_id != spec.call_id:
            _fail(f"{spec.attempt_id} receipt has the wrong FunctionCall ID")
        intent_payload = (output / "intent.json").read_bytes()
        journal_payload = (output / "journal.jsonl").read_bytes()
        if (
            intent_path.read_bytes() != intent_payload
            or journal_path.read_bytes() != journal_payload
        ):
            _fail(f"{spec.attempt_id} receipt intent/journal differ from retained sidecars")
        receipt_payload, receipt_object = _read_canonical_object(
            output / "receipt.json", context=f"{spec.attempt_id} receipt"
        )
        result_payload, result_object = _read_canonical_object(
            output / "result.json", context=f"{spec.attempt_id} result"
        )
        if receipt_object != verified.receipt.to_dict():
            _fail(f"{spec.attempt_id} receipt serialization differs from verification")
        if result_object != verified.envelope.to_dict():
            _fail(f"{spec.attempt_id} result serialization differs from verification")
        support, support_payloads = _support_payloads(output)
        intent = verified.intent
        receipt_value = receipt_object
        result_value = result_object
        support_value = support
        source_files = {
            "operator_pointer": _binding(str(spec.pointer_path), pointer_payload),
            "remote_intent": _binding(_source_relative(intent_path), intent_payload),
            "remote_journal": _binding(_source_relative(journal_path), journal_payload),
            "receipt_intent": _binding(_source_relative(output / "intent.json"), intent_payload),
            "receipt_journal": _binding(
                _source_relative(output / "journal.jsonl"), journal_payload
            ),
            "receipt_root": _binding(_source_relative(output / "receipt.json"), receipt_payload),
            "receipt_result": _binding(_source_relative(output / "result.json"), result_payload),
            "receipt_suite": _binding(
                _source_relative(output / "suite.json"), support_payloads["suite.json"]
            ),
            "receipt_plugin": _binding(
                _source_relative(output / "plugin.json"), support_payloads["plugin.json"]
            ),
            "receipt_wheel_manifest": _binding(
                _source_relative(output / "wheel.manifest.json"),
                support_payloads["wheel.manifest.json"],
            ),
        }

    if intent.suite_path != spec.suite_path or intent.plugin_path != _PLUGIN_PATH:
        _fail(f"{spec.attempt_id} intent suite/plugin differs")
    if intent.plugin_sha256 != _PLUGIN_SHA256:
        _fail(f"{spec.attempt_id} intent plugin digest differs")
    journal_values = _journal_values(journal_payload, context=f"{spec.attempt_id} journal")
    request_digest = cast(dict[str, object], journal_values[0])["request_digest"]
    if type(request_digest) is not str:
        _fail(f"{spec.attempt_id} journal request digest is not a string")
    _journal_from_values(
        journal_values,
        request_digest=request_digest,
        call_id=spec.call_id,
        status=spec.status,
        context=f"{spec.attempt_id} journal",
    )

    return {
        "app": _app_provenance(spec),
        "attempt_id": spec.attempt_id,
        "call": _call_provenance(spec),
        "intent": intent.to_dict(),
        "journal": journal_values,
        "pointer_value": pointer_value,
        "receipt": receipt_value,
        "receipt_support": support_value,
        "result_envelope": result_value,
        "source_files": source_files,
        "status": spec.status,
    }


def _suite_bytes(intent: RemoteIntent, *, context: str) -> bytes:
    if Path(intent.suite_path).is_absolute() or ".." in Path(intent.suite_path).parts:
        _fail(f"{context} suite path must be repository-relative")
    payload = (_REPOSITORY / intent.suite_path).read_bytes()
    if sha256_bytes(payload) != intent.suite_sha256:
        _fail(f"{context} committed suite digest differs from intent")
    return payload


def _plugin_bytes(intent: RemoteIntent, *, context: str) -> bytes:
    if Path(intent.plugin_path).is_absolute() or ".." in Path(intent.plugin_path).parts:
        _fail(f"{context} plugin path must be repository-relative")
    payload = (_REPOSITORY / intent.plugin_path).read_bytes()
    if sha256_bytes(payload) != intent.plugin_sha256:
        _fail(f"{context} committed plugin digest differs from intent")
    return payload


def _exact_result_metrics(result: LocalExecutionResult, spec: AttemptSpec) -> dict[str, object]:
    if result.suite_id != spec.suite_id or result.outcome != "completed":
        _fail(f"{spec.attempt_id} completed result suite/outcome differs")
    expected_cells = {
        f"{spec.candidate_arm}-correctness",
        f"{spec.candidate_arm}-timing",
        f"{spec.reference_arm}-correctness",
        f"{spec.reference_arm}-timing",
    }
    observations = {observation.cell_id: observation for observation in result.observations}
    if set(observations) != expected_cells or len(result.observations) != 4:
        _fail(f"{spec.attempt_id} result does not contain exactly the four expected cells")

    summary = exact_fields(
        result.summary,
        required={
            "all_cells_terminal",
            "blocked",
            "candidate_distinction",
            "candidate_reference_arithmetic",
            "expected_cell_ids",
            "failed",
            "fusion_claim",
            "outcome",
            "passed",
            "terminal_cell_ids",
        },
        context=f"{spec.attempt_id} local summary",
    )
    if (
        summary["all_cells_terminal"] is not True
        or summary["blocked"] != 0
        or summary["failed"] != 0
        or summary["fusion_claim"] is not False
        or summary["outcome"] != "completed"
        or summary["passed"] != 4
        or set(cast(list[object], summary["expected_cell_ids"])) != expected_cells
        or set(cast(list[object], summary["terminal_cell_ids"])) != expected_cells
    ):
        _fail(f"{spec.attempt_id} completed result summary is not an exact four-cell pass")
    environment = exact_object(result.environment, context=f"{spec.attempt_id} environment")
    if (
        environment.get("fusion_claim") is not False
        or environment.get("backend_invoked") is not True
    ):
        _fail(f"{spec.attempt_id} environment fusion/backend evidence differs")

    correctness: dict[str, object] = {}
    timing: dict[str, object] = {}
    for role, arm in (("candidate", spec.candidate_arm), ("reference", spec.reference_arm)):
        correctness_observation = observations[f"{arm}-correctness"]
        timing_observation = observations[f"{arm}-timing"]
        correctness_value = correctness_observation.correctness
        timing_value = timing_observation.timing
        if correctness_value is None or timing_value is None:
            _fail(f"{spec.attempt_id} {role} observations have the wrong stages")
        if (
            correctness_observation.status != "passed"
            or correctness_value.status != "passed"
            or not correctness_value.close
            or not correctness_value.finite
            or not correctness_value.input_storage_unchanged
            or not correctness_value.output_disjoint
        ):
            _fail(f"{spec.attempt_id} {role} correctness did not exactly pass")
        if timing_observation.status != "passed" or timing_value.status != "passed":
            _fail(f"{spec.attempt_id} {role} timing did not exactly pass")
        if timing_value.repetitions != 50 or timing_value.warmups != 10:
            _fail(f"{spec.attempt_id} {role} timing protocol differs")
        median_ms = finite_float(
            timing_value.median_ms,
            context=f"{spec.attempt_id} {role} median_ms",
            strictly_positive=True,
        )
        correctness[role] = {
            "close": correctness_value.close,
            "finite": correctness_value.finite,
            "input_storage_unchanged": correctness_value.input_storage_unchanged,
            "max_abs_error": correctness_value.max_abs_error,
            "output_disjoint": correctness_value.output_disjoint,
            "status": correctness_value.status,
        }
        timing[role] = {
            "median_ms": median_ms,
            "repetitions": timing_value.repetitions,
            "status": timing_value.status,
            "warmups": timing_value.warmups,
        }

    if set(result.compile_outcomes) != {spec.candidate_arm}:
        _fail(f"{spec.attempt_id} compile outcomes differ from the candidate-only contract")
    compile_value = exact_fields(
        result.compile_outcomes[spec.candidate_arm],
        required={
            "arm_id",
            "autocast_policy",
            "backend_invoked",
            "callable_distinct",
            "case_id",
            "eager_fallback",
            "entrypoint",
            "error",
            "first_call_ns",
            "status",
            "wrapper_create_ns",
        },
        context=f"{spec.attempt_id} compile outcome",
    )
    if (
        compile_value["arm_id"] != spec.candidate_arm
        or compile_value["backend_invoked"] is not True
        or compile_value["callable_distinct"] is not True
        or compile_value["eager_fallback"] is not False
        or compile_value["error"] is not None
        or compile_value["status"] != "compiled_and_first_call_completed"
    ):
        _fail(f"{spec.attempt_id} candidate compile outcome differs")
    wrapper_ns = exact_int(
        compile_value["wrapper_create_ns"],
        context=f"{spec.attempt_id} wrapper_create_ns",
        minimum=0,
    )
    first_call_ns = exact_int(
        compile_value["first_call_ns"],
        context=f"{spec.attempt_id} first_call_ns",
        minimum=1,
    )
    candidate_ms = cast(dict[str, object], timing["candidate"])["median_ms"]
    reference_ms = cast(dict[str, object], timing["reference"])["median_ms"]
    if type(candidate_ms) is not float or type(reference_ms) is not float:
        _fail(f"{spec.attempt_id} medians are not floats")
    return {
        "compile": {
            "arm_id": spec.candidate_arm,
            "backend_invoked": True,
            "callable_distinct": True,
            "eager_fallback": False,
            "first_call_ns": first_call_ns,
            "status": "compiled_and_first_call_completed",
            "wrapper_create_ns": wrapper_ns,
        },
        "candidate_distinction": nonblank_string(
            summary["candidate_distinction"],
            context=f"{spec.attempt_id} candidate distinction",
        ),
        "candidate_reference_arithmetic": nonblank_string(
            summary["candidate_reference_arithmetic"],
            context=f"{spec.attempt_id} candidate/reference arithmetic",
        ),
        "correctness": correctness,
        "descriptive_ratios": {
            "candidate_to_reference_median": candidate_ms / reference_ms,
            "interpretation": "ratio_of_returned_medians_only",
            "reference_to_candidate_median": reference_ms / candidate_ms,
            "superiority_tested": False,
        },
        "timing": timing,
    }


def _embedded_receipt_payloads(
    attempt: Mapping[str, object],
    *,
    intent_payload: bytes,
    journal_payload: bytes,
    source_files: Mapping[str, object],
    spec: AttemptSpec,
) -> dict[str, bytes]:
    support = exact_fields(
        attempt["receipt_support"],
        required={"suite_utf8", "plugin_utf8", "wheel_manifest_utf8"},
        context=f"{spec.attempt_id} receipt support",
    )
    files = {
        "intent.json": intent_payload,
        "journal.jsonl": journal_payload,
        "receipt.json": canonical_json_bytes(attempt["receipt"]),
        "result.json": canonical_json_bytes(attempt["result_envelope"]),
    }
    for field, filename in (
        ("suite_utf8", "suite.json"),
        ("plugin_utf8", "plugin.json"),
        ("wheel_manifest_utf8", "wheel.manifest.json"),
    ):
        text = support[field]
        if type(text) is not str or not text:
            _fail(f"{spec.attempt_id} {field} must be a nonempty string")
        strict_json_loads(text, source=f"{spec.attempt_id} {field}")
        files[filename] = text.encode("utf-8")

    source_keys = {
        "intent.json": "receipt_intent",
        "journal.jsonl": "receipt_journal",
        "receipt.json": "receipt_root",
        "result.json": "receipt_result",
        "suite.json": "receipt_suite",
        "plugin.json": "receipt_plugin",
        "wheel.manifest.json": "receipt_wheel_manifest",
    }
    for filename, source_key in source_keys.items():
        _validate_binding(
            source_files[source_key],
            files[filename],
            path=f"{spec.output_relative}/{filename}",
            context=f"{spec.attempt_id} {source_key}",
        )
    return files


def _validate_raw(raw: bytes) -> tuple[dict[str, object], tuple[AnalyzedAttempt, ...]]:
    value = exact_fields(
        _strict_json_bytes(raw, context="fusion remote exploratory raw"),
        required={"attempts", "methodology", "schema", "study_id"},
        context="fusion remote exploratory raw",
    )
    if value["schema"] != _RAW_SCHEMA or value["study_id"] != _STUDY_ID:
        _fail("raw artifact identity differs")
    if value["methodology"] != _METHOD:
        _fail("raw methodology differs from the post-hoc exploratory contract")
    attempts_value = value["attempts"]
    if type(attempts_value) is not list or len(attempts_value) != len(_SPECS):
        _fail("raw artifact must contain exactly four attempts")

    analyzed: list[AnalyzedAttempt] = []
    for spec, item in zip(_SPECS, cast(list[object], attempts_value), strict=True):
        attempt = exact_fields(
            item,
            required={
                "app",
                "attempt_id",
                "call",
                "intent",
                "journal",
                "pointer_value",
                "receipt",
                "receipt_support",
                "result_envelope",
                "source_files",
                "status",
            },
            context=f"raw attempt {spec.attempt_id}",
        )
        if (
            attempt["attempt_id"] != spec.attempt_id
            or attempt["status"] != spec.status
            or attempt["pointer_value"] != spec.output_relative
        ):
            _fail(f"{spec.attempt_id} identity/status/pointer differs")
        _validate_app(spec, attempt["app"])
        if attempt["call"] != _call_provenance(spec):
            _fail(f"{spec.attempt_id} call identity/provenance differs")
        intent = RemoteIntent.from_dict(attempt["intent"])
        if (
            intent.suite_path != spec.suite_path
            or intent.plugin_path != _PLUGIN_PATH
            or intent.plugin_sha256 != _PLUGIN_SHA256
            or not intent.output_path.endswith(f"/{spec.output_relative}")
        ):
            _fail(f"{spec.attempt_id} intent identity differs")
        suite_bytes = _suite_bytes(intent, context=spec.attempt_id)
        _plugin_bytes(intent, context=spec.attempt_id)
        _, _, request_digest = _request_components(intent, suite_bytes)
        journal, journal_payload = _journal_from_values(
            attempt["journal"],
            request_digest=request_digest,
            call_id=spec.call_id,
            status=spec.status,
            context=f"{spec.attempt_id} journal",
        )
        source_files = exact_object(
            attempt["source_files"], context=f"{spec.attempt_id} source files"
        )
        expected_source_keys = {
            "operator_pointer",
            "remote_intent",
            "remote_journal",
        }
        if spec.status == "completed":
            expected_source_keys.update(
                {
                    "receipt_intent",
                    "receipt_journal",
                    "receipt_plugin",
                    "receipt_result",
                    "receipt_root",
                    "receipt_suite",
                    "receipt_wheel_manifest",
                }
            )
        if set(source_files) != expected_source_keys:
            _fail(f"{spec.attempt_id} source file inventory differs")
        pointer_payload = f"{spec.output_relative}\n".encode()
        intent_payload = canonical_json_bytes(intent.to_dict())
        intent_path, journal_path = remote_artifact_paths(Path(spec.output_relative))
        _validate_binding(
            source_files["operator_pointer"],
            pointer_payload,
            path=str(spec.pointer_path),
            context=f"{spec.attempt_id} pointer binding",
        )
        _validate_binding(
            source_files["remote_intent"],
            intent_payload,
            path=intent_path.as_posix(),
            context=f"{spec.attempt_id} intent binding",
        )
        _validate_binding(
            source_files["remote_journal"],
            journal_payload,
            path=journal_path.as_posix(),
            context=f"{spec.attempt_id} journal binding",
        )

        envelope: RemoteResultEnvelope | None = None
        local_result: LocalExecutionResult | None = None
        if spec.status == "unresolved":
            if any(
                attempt[field] is not None
                for field in ("receipt", "receipt_support", "result_envelope")
            ):
                _fail(f"{spec.attempt_id} unresolved attempt must have no receipt/result")
        else:
            receipt_files = _embedded_receipt_payloads(
                attempt,
                intent_payload=intent_payload,
                journal_payload=journal_payload,
                source_files=source_files,
                spec=spec,
            )
            verified = verify_remote_receipt_payloads(
                receipt_files, logical_output_path=intent.output_path
            )
            if (
                verified.receipt.status != "completed"
                or verified.result is None
                or verified.envelope is None
                or verified.result.outcome != "completed"
            ):
                _fail(f"{spec.attempt_id} is not an exact completed receipt/result")
            if (
                verified.receipt.to_dict() != attempt["receipt"]
                or verified.intent.to_dict() != attempt["intent"]
                or verified.journal != journal
                or verified.envelope.to_dict() != attempt["result_envelope"]
            ):
                _fail(f"{spec.attempt_id} verified receipt/result differs from raw summary source")
            if verified.journal[-1].call_id != spec.call_id:
                _fail(f"{spec.attempt_id} verified receipt has the wrong FunctionCall ID")
            envelope = verified.envelope
            local_result = verified.result
            validate_hardware(envelope.hardware, expectation_for_gpu("H100"))
            _exact_result_metrics(local_result, spec)
        analyzed.append(AnalyzedAttempt(spec, attempt, intent, journal, local_result, envelope))

    statuses = [attempt.spec.status for attempt in analyzed]
    if statuses.count("completed") != 2 or statuses.count("unresolved") != 2:
        _fail("raw artifact must contain exactly two completed and two unresolved attempts")
    return value, tuple(analyzed)


def _request_components(
    intent: RemoteIntent, suite_bytes: bytes
) -> tuple[RemoteIntent, bytes, str]:
    encoded = encode_remote_request(intent, suite_bytes)
    value = exact_object(
        strict_json_loads(encoded, source="reconstructed remote request"), context="request"
    )
    digest = nonblank_string(value.get("request_digest"), context="request digest")
    return intent, suite_bytes, digest


def _summary(analyzed: Sequence[AnalyzedAttempt]) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    completed_results: list[dict[str, object]] = []
    for item in analyzed:
        states = [record.state for record in item.journal]
        attempts.append(
            {
                "app": dict(cast(Mapping[str, object], item.value["app"])),
                "attempt_id": item.spec.attempt_id,
                "call": dict(cast(Mapping[str, object], item.value["call"])),
                "head_commit": item.intent.head_commit,
                "journal_states": states,
                "source_sha256": item.intent.source_sha256,
                "status": item.spec.status,
                "suite_id": item.spec.suite_id,
                "suite_path": item.intent.suite_path,
                "terminal_detail": item.journal[-1].detail,
                "wheel_filename": item.intent.wheel_filename,
                "wheel_sha256": item.intent.wheel_sha256,
            }
        )
        if item.result is not None:
            completed_results.append(
                {
                    "attempt_id": item.spec.attempt_id,
                    "claim_scope": "measured_fact_only",
                    "fusion_claim": False,
                    "hardware": item.envelope.hardware.to_dict() if item.envelope else None,
                    "metrics": _exact_result_metrics(item.result, item.spec),
                    "publication_eligible": False,
                    "suite_id": item.spec.suite_id,
                }
            )
    return {
        "attempts": attempts,
        "claim_classification": dict(_CLAIM_CLASSIFICATION),
        "completed_results": completed_results,
        "counts": {"attempts": 4, "completed": 2, "failed": 0, "unresolved": 2},
        "limitations": list(_LIMITATIONS),
        "methodology": dict(_METHOD),
        "provider_accounting": {
            "actual_cost_usd": None,
            "client_authorized_spawns": 4,
            "cost_status": "unknown",
            "provider_attempts_observable": False,
            "provider_physical_attempts": None,
            "total_gpu_seconds": None,
        },
        "publication_eligible": False,
        "schema": _SUMMARY_SCHEMA,
        "study_id": _STUDY_ID,
    }


def _collection_command(spec: AttemptSpec) -> str:
    return (
        "uv run --extra modal modal run modal_fusion_executor.py::main "
        f"--suite {spec.suite_path} --plugin {_PLUGIN_PATH} --output {spec.output_relative}"
    )


def _manifest(
    analyzed: Sequence[AnalyzedAttempt],
    *,
    compressed: bytes,
    raw: bytes,
    summary: bytes,
) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    inputs: dict[str, object] = {}
    collection: dict[str, object] = {}
    for item in analyzed:
        attempts.append(
            {
                "app": dict(cast(Mapping[str, object], item.value["app"])),
                "attempt_id": item.spec.attempt_id,
                "call": dict(cast(Mapping[str, object], item.value["call"])),
                "head_commit": item.intent.head_commit,
                "manifest_sha256": item.intent.manifest_sha256,
                "source_sha256": item.intent.source_sha256,
                "status": item.spec.status,
                "suite_path": item.intent.suite_path,
                "suite_sha256": item.intent.suite_sha256,
                "wheel_filename": item.intent.wheel_filename,
                "wheel_sha256": item.intent.wheel_sha256,
            }
        )
        inputs[item.spec.attempt_id] = dict(cast(Mapping[str, object], item.value["source_files"]))
        collection[item.spec.attempt_id] = _collection_command(item.spec)
    return {
        "attempts": attempts,
        "commands": {
            "check": "uv run python scripts/publish_fusion_remote_results.py --check",
            "collection": collection,
            "generate": "uv run python scripts/publish_fusion_remote_results.py",
        },
        "inputs": inputs,
        "methodology": {
            **_METHOD,
            "claim_classification": dict(_CLAIM_CLASSIFICATION),
            "cost_status": "unknown",
            "provider_physical_attempts": None,
        },
        "publication": {
            "raw": {
                "compressed_bytes": len(compressed),
                "compressed_sha256": sha256_bytes(compressed),
                "decompressed_bytes": len(raw),
                "decompressed_sha256": sha256_bytes(raw),
                "format": _ZSTD_DESCRIPTION,
                "path": "benchmarks/data/fusion-remote-exploratory.json.zst",
            },
            "summary": {
                "bytes": len(summary),
                "path": "benchmarks/results/fusion-remote-exploratory-summary.json",
                "sha256": sha256_bytes(summary),
            },
        },
        "publisher": {
            "path": "scripts/publish_fusion_remote_results.py",
            "sha256": sha256_bytes(_PUBLISHER_PATH.read_bytes()),
        },
        "report": {"path": None, "status": "not_created"},
        "schema": _MANIFEST_SCHEMA,
        "study_id": _STUDY_ID,
    }


def generate() -> None:
    raw_value = {
        "attempts": [_generation_attempt(spec) for spec in _SPECS],
        "methodology": dict(_METHOD),
        "schema": _RAW_SCHEMA,
        "study_id": _STUDY_ID,
    }
    raw = _json_bytes(raw_value)
    _, analyzed = _validate_raw(raw)
    compressed = _compress(raw)
    summary = _json_bytes(_summary(analyzed))
    manifest = _json_bytes(_manifest(analyzed, compressed=compressed, raw=raw, summary=summary))
    write_bytes_atomic(_RAW_PATH, compressed)
    write_bytes_atomic(_SUMMARY_PATH, summary)
    write_bytes_atomic(_MANIFEST_PATH, manifest)
    print(
        "published four post-hoc exploratory H100 fusion remote attempts; no fusion or superiority claim"
    )
    print(
        f"raw={sha256_bytes(compressed)} summary={sha256_bytes(summary)} "
        f"manifest={sha256_bytes(manifest)}"
    )


def check() -> None:
    for path in (_RAW_PATH, _SUMMARY_PATH, _MANIFEST_PATH, _PUBLISHER_PATH):
        if not path.is_file():
            _fail(
                f"required committed publication file is missing: {path.relative_to(_REPOSITORY)}"
            )
    compressed = _RAW_PATH.read_bytes()
    raw = _decompress(compressed)
    if _compress(raw) != compressed:
        _fail("published raw artifact is not the deterministic zstd encoding")
    _, analyzed = _validate_raw(raw)
    expected_summary = _json_bytes(_summary(analyzed))
    actual_summary = _SUMMARY_PATH.read_bytes()
    _strict_json_bytes(actual_summary, context="fusion remote exploratory summary")
    if actual_summary != expected_summary:
        _fail("published summary is not the byte-exact analysis of committed raw evidence")
    expected_manifest = _json_bytes(
        _manifest(analyzed, compressed=compressed, raw=raw, summary=expected_summary)
    )
    actual_manifest = _MANIFEST_PATH.read_bytes()
    _strict_json_bytes(actual_manifest, context="fusion remote exploratory manifest")
    if actual_manifest != expected_manifest:
        _fail("published manifest is not the byte-exact binding of committed evidence")
    print("checked committed fusion remote exploratory publication from committed files only")
    print(
        f"raw={sha256_bytes(compressed)} summary={sha256_bytes(actual_summary)} "
        f"manifest={sha256_bytes(actual_manifest)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check committed publication only")
    args = parser.parse_args(argv)
    if cast(bool, args.check):
        check()
    else:
        generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
