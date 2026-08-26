"""Publish and byte-check the one-bank H100 engineering expansion-gate result.

Generation reads the gitignored local collection, correctness, and precision bundles. Check mode
uses only committed files and never contacts Git or Modal unless --compare-head is explicitly used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

import zstandard
from analyze_hopper_benchmark import (
    BenchmarkAnalysis,
    analyze_artifact,
    load_archive_ratios,
)
from analyze_precision_probe import load_summaries

from heliostune.artifacts import (
    read_json,
    strict_json_dumps,
    strict_json_loads,
    write_bytes_atomic,
)
from heliostune.collection import AttemptJournal, sha256_file
from heliostune.validation import (
    exact_bool,
    exact_fields,
    exact_int,
    exact_object,
    nonblank_string,
)

_REPO = Path(__file__).resolve().parents[1]
_LOCAL = _REPO / "artifacts"
_ENGINEERING = _LOCAL / "hopper-h100-engineering.json"
_CORRECTNESS = _LOCAL / "hopper-correctness.json"
_PRECISION = _LOCAL / "h100-precision-probe.json"
_ARCHIVE = _REPO / "benchmarks/data/parhelion-v2-measurements.jsonl.zst"
_RAW = _REPO / "benchmarks/data/hopper-h100-engineering.json.zst"
_JOURNAL = _REPO / "benchmarks/data/hopper-h100-engineering.attempts.jsonl"
_SUMMARY = _REPO / "benchmarks/results/hopper-h100-engineering-summary.json"
_MANIFEST = _REPO / "benchmarks/hopper-h100-engineering-manifest.json"
_ANALYZER = _REPO / "scripts/analyze_hopper_benchmark.py"
_PUBLISHER = Path(__file__).resolve()

_STUDY_ID = "hopper-h100-engineering-benchmark"
_ANALYSIS_STATUS = "post_hoc_exploratory"
_HEAD_COMMIT = "3e7734ada28e9ab6c83ea3b21895c311e8d492b7"
_HEAD_SHA256 = "5d709f6360fc95abba6c729c0952e01a8b1ab58809c794e1c68f5ccaeb1fc80b"
_SOURCE_SHA256 = "997dd421a8b1e2dea3c61de64e3239c46f64f0f8c95606efb10ce89e9782ee49"
_WHEEL_SHA256 = "d56791105a58f4b00259cd71624e3177071da96a025e468c3a3def5555728d69"
_WHEEL_MANIFEST_SHA256 = "c6c5b708a579be89ee0d92d9be23b8c2245eb35466f43303d79d80d9b73df02c"
_ARCHIVE_SHA256 = "ed6ec6ee8c3b61b451ea1276fc6f3925e82f70b5e208e9195c924ef6acc7343f"
_PRIOR_CONTEXTUAL_BASELINE = 0.627266
_GATE_SPEEDUP = 1.05
_GATE_WIN_FRACTION = 0.25

_ENGINEERING_CALL = "fc-01M0XX8KZ2WZQWPA4V2SYVGNWX"
_CORRECTNESS_CALL = "fc-01M0XX5MS2JFMYAH3XW98HJCFR"
_PRECISION_CALLS = (
    "fc-01M0XT9C4CS22SVVV2M2DV86HS",
    "fc-01M0XTATZKMR4BY3K6N2J813Z3",
    "fc-01M0XTBXK9X51JZJXCA6DWF3KY",
)
_ENGINEERING_APP = "ap-ryV3BXdW1g2TGp5LIg6MDH"
_CORRECTNESS_APP = "ap-yvSdUddrJrfljamxVa4CZI"
_PRECISION_APP = "ap-oxqdKZOLRVPqepVv8AL6R4"
_MODAL_WORKSPACE = "mottopanikeiku"

_INPUT_DIGESTS: dict[str, dict[str, str]] = {
    "engineering": {
        "artifact": "f59b0bb774f5ea54f7691cd2caa887543790e9685f93bfd5c5e4b05cf7b20d31",
        "manifest": "048b781f53d8fedb21991c517f5439f0eedbd7e0a80c649a210a8f38a381db05",
        "attempt_journal": "ecc9a2838ce57325a90799c6f3f08406bf0daddd3aa1b8b44f2b7d4aea569db4",
    },
    "correctness": {
        "artifact": "98116c12540102049871b08c09bdfb0b21540c514a1be16df2a9119017885123",
        "manifest": "cda8b732ef3aafbee8ce5f75a1d05d1a59b02d32ff62ecb9dd4733f40fc4dbff",
        "attempt_journal": "6901b380fd58683a29f6eedf0b8b95e954ae9ae04011e0b0dbdb4e67477ac230",
    },
    "precision": {
        "artifact": "015ec9773730169b4ccee4caa111103a6fb9d04d8b6978dbb61f0334df8e1829",
        "manifest": "2a7c8dec20a5e055a19802cd64a201129460fc6f0fa9e4c400d0ad578d9bfb92",
        "attempt_journal": "4771219bcedda851f09a5c8de6206ce5957c5a47e3c0e1e6b41971f6d5d4c633",
    },
}


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sidecar(path: Path) -> Path:
    return Path(f"{path}.manifest.json")


def _attempts(path: Path) -> Path:
    return Path(f"{path}.attempts.jsonl")


def _require_digest(path: Path, expected: str, *, context: str) -> None:
    if not path.is_file():
        _fail(f"{context} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        _fail(f"{context} SHA-256 is {actual}, expected {expected}")


def _strict_object(path: Path, *, context: str) -> dict[str, object]:
    return exact_object(read_json(path), context=context)


def _binding(sidecar: Mapping[str, object], *, context: str) -> dict[str, object]:
    return exact_fields(
        sidecar.get("binding"),
        required=(
            "protocol_sha256",
            "config_manifest_sha256",
            "wheel_sha256",
            "head_sha256",
        ),
        context=f"{context} binding",
    )


def _validate_journal_binding(
    path: Path,
    sidecar: Mapping[str, object],
    *,
    expected_calls: Sequence[str],
    expected_banks: Sequence[int],
    expected_request_sha256: str,
    context: str,
) -> tuple[str, ...]:
    journal_binding = exact_fields(
        sidecar.get("attempt_journal"),
        required=("path", "sha256"),
        context=f"{context} attempt_journal",
    )
    recorded_path = Path(
        nonblank_string(journal_binding["path"], context=f"{context} journal path")
    )
    if recorded_path.resolve() != path.resolve() or journal_binding["sha256"] != sha256_file(path):
        _fail(f"{context} attempt journal path/digest differs from its sidecar")
    records = AttemptJournal.load(path).records
    binding = _binding(sidecar, context=context)
    if len(records) != 2 * len(expected_calls):
        _fail(f"{context} attempt journal has the wrong record count")
    completed: list[str] = []
    for index, (call_id, bank) in enumerate(zip(expected_calls, expected_banks, strict=True)):
        spawned = records[2 * index]
        finished = records[2 * index + 1]
        if (
            spawned.status != "spawned"
            or finished.status != "completed"
            or spawned.call_id != call_id
            or finished.call_id != call_id
            or spawned.bank != bank
            or finished.bank != bank
            or spawned.gpu != "H100"
            or finished.gpu != "H100"
        ):
            _fail(f"{context} attempt journal does not contain the expected call transitions")
        for record in (spawned, finished):
            if record.request_sha256 != expected_request_sha256 or any(
                getattr(record, field) != binding[field]
                for field in (
                    "protocol_sha256",
                    "config_manifest_sha256",
                    "wheel_sha256",
                    "head_sha256",
                )
            ):
                _fail(f"{context} journal request/source binding differs from its sidecar")
        completed.append(call_id)
    return tuple(completed)


def _validate_wheel_binding(engineering_sidecar: Mapping[str, object]) -> dict[str, str]:
    inputs = exact_object(engineering_sidecar.get("inputs"), context="engineering inputs")
    wheel = exact_fields(inputs.get("wheel"), required=("path", "sha256"), context="wheel input")
    wheel_manifest = exact_fields(
        inputs.get("wheel_manifest"),
        required=("path", "sha256"),
        context="wheel manifest input",
    )
    source = exact_fields(inputs.get("source"), required=("sha256",), context="source input")
    wheel_path = Path(nonblank_string(wheel["path"], context="wheel path"))
    wheel_manifest_path = Path(
        nonblank_string(wheel_manifest["path"], context="wheel manifest path")
    )
    _require_digest(wheel_path, _WHEEL_SHA256, context="engineering wheel")
    _require_digest(
        wheel_manifest_path,
        _WHEEL_MANIFEST_SHA256,
        context="engineering wheel manifest",
    )
    manifest = exact_object(read_json(wheel_manifest_path), context="wheel manifest")
    expected = {
        "head_commit": _HEAD_COMMIT,
        "source_sha256": _SOURCE_SHA256,
        "wheel_sha256": _WHEEL_SHA256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        _fail("wheel manifest does not bind the engineering HEAD/source/wheel")
    if wheel["sha256"] != _WHEEL_SHA256 or wheel_manifest["sha256"] != _WHEEL_MANIFEST_SHA256:
        _fail("engineering sidecar wheel bindings differ")
    if source["sha256"] != _SOURCE_SHA256:
        _fail("engineering sidecar source digest differs")
    return {
        "manifest_path": str(wheel_manifest_path),
        "manifest_sha256": _WHEEL_MANIFEST_SHA256,
        "head_commit": _HEAD_COMMIT,
        "source_sha256": _SOURCE_SHA256,
        "wheel_sha256": _WHEEL_SHA256,
    }


def _validate_correctness_bundle(wheel_provenance: dict[str, str]) -> None:
    expected = _INPUT_DIGESTS["correctness"]
    _require_digest(_CORRECTNESS, expected["artifact"], context="correctness artifact")
    _require_digest(_sidecar(_CORRECTNESS), expected["manifest"], context="correctness manifest")
    _require_digest(
        _attempts(_CORRECTNESS),
        expected["attempt_journal"],
        context="correctness attempt journal",
    )
    artifact = exact_fields(
        read_json(_CORRECTNESS),
        required=(
            "schema_version",
            "gate",
            "study_status",
            "analysis_status",
            "verified",
            "correctness_only",
            "performance_validated",
            "protocol",
            "gpu",
            "gpu_selector",
            "hardware",
            "config_counts",
            "config_manifest",
            "validation_workload_count",
            "validation_check_count",
            "candidate_summaries",
            "validation_results",
            "remote_call",
        ),
        context="correctness artifact",
    )
    if (
        exact_int(artifact["schema_version"], context="correctness schema version") != 1
        or artifact["gate"] != "hopper-candidate-correctness"
        or artifact["analysis_status"] != _ANALYSIS_STATUS
        or artifact["study_status"] != _ANALYSIS_STATUS
        or artifact["gpu"] != "H100"
        or artifact["gpu_selector"] != "H100!"
        or not exact_bool(artifact["verified"], context="correctness verified")
        or not exact_bool(artifact["correctness_only"], context="correctness_only")
        or exact_bool(artifact["performance_validated"], context="performance_validated")
    ):
        _fail("correctness artifact status/scope differs from the strict gate")
    summaries = artifact["candidate_summaries"]
    results = artifact["validation_results"]
    if type(summaries) is not list or len(summaries) != 71:
        _fail("correctness artifact must contain 71 candidate summaries")
    if type(results) is not list or len(results) != 633:
        _fail("correctness artifact must contain 633 validation checks")
    config = exact_fields(
        artifact["config_manifest"],
        required=("sha256", "hopper_gemm", "skinny_gemv"),
        context="correctness config manifest",
    )
    if config["sha256"] != "7cc6bf55a00dfb10570481fd95a96694aa3f4d4085ea76881fd860683f10d134":
        _fail("correctness config manifest digest differs")
    sidecar = _strict_object(_sidecar(_CORRECTNESS), context="correctness sidecar")
    if (
        sidecar.get("gate") != artifact["gate"]
        or sidecar.get("protocol") != artifact["protocol"]
        or not exact_bool(sidecar.get("verified"), context="correctness sidecar verified")
    ):
        _fail("correctness sidecar differs from its artifact")
    data = exact_fields(
        sidecar.get("data"),
        required=("path", "sha256", "candidate_summaries", "validation_results"),
        context="correctness data",
    )
    if (
        Path(nonblank_string(data["path"], context="correctness data path")).resolve()
        != _CORRECTNESS.resolve()
        or data["sha256"] != expected["artifact"]
        or exact_int(data["candidate_summaries"], context="correctness summaries") != 71
        or exact_int(data["validation_results"], context="correctness checks") != 633
    ):
        _fail("correctness sidecar data binding differs")
    calls = _validate_journal_binding(
        _attempts(_CORRECTNESS),
        sidecar,
        expected_calls=(_CORRECTNESS_CALL,),
        expected_banks=(0,),
        expected_request_sha256="8e3da7aa1f2037c1b2cc7c4e233087e3b7342239b22decbb98e23d0dec40cf18",
        context="correctness",
    )
    remote = exact_fields(
        artifact["remote_call"],
        required=("call_id", "payload_sha256"),
        context="correctness remote call",
    )
    completed = AttemptJournal.load(_attempts(_CORRECTNESS)).records[-1]
    if (
        remote["call_id"] != calls[0]
        or remote["payload_sha256"] != completed.chunk_sha256
        or exact_int(artifact["validation_workload_count"], context="correctness workloads") != 15
        or exact_int(artifact["validation_check_count"], context="correctness checks") != 633
    ):
        _fail("correctness remote call/count binding differs")
    inputs = exact_object(sidecar.get("inputs"), context="correctness inputs")
    source = exact_fields(inputs.get("source"), required=("sha256",), context="correctness source")
    wheel = exact_fields(
        inputs.get("wheel"), required=("path", "sha256"), context="correctness wheel"
    )
    wheel_manifest = exact_fields(
        inputs.get("wheel_manifest"),
        required=("path", "sha256"),
        context="correctness wheel manifest",
    )
    if (
        source["sha256"] != wheel_provenance["source_sha256"]
        or wheel["sha256"] != wheel_provenance["wheel_sha256"]
        or wheel_manifest["sha256"] != wheel_provenance["manifest_sha256"]
    ):
        _fail("correctness source/wheel binding differs from engineering provenance")


def _validate_precision_bundle() -> float:
    expected = _INPUT_DIGESTS["precision"]
    _require_digest(_PRECISION, expected["artifact"], context="precision artifact")
    _require_digest(_sidecar(_PRECISION), expected["manifest"], context="precision manifest")
    _require_digest(
        _attempts(_PRECISION),
        expected["attempt_journal"],
        context="precision attempt journal",
    )
    _data, summaries = load_summaries(_PRECISION)
    sidecar = _strict_object(_sidecar(_PRECISION), context="precision sidecar")
    data_binding = exact_fields(
        sidecar.get("data"), required=("path", "rows", "sha256"), context="precision data"
    )
    if (
        Path(nonblank_string(data_binding["path"], context="precision data path")).resolve()
        != _PRECISION.resolve()
        or data_binding["sha256"] != expected["artifact"]
        or exact_int(data_binding["rows"], context="precision rows") != 288
    ):
        _fail("precision sidecar data path/digest/count differs")
    calls = _validate_journal_binding(
        _attempts(_PRECISION),
        sidecar,
        expected_calls=_PRECISION_CALLS,
        expected_banks=(0, 1, 2),
        expected_request_sha256="00c7c9f8c50eb1f336d20e9aa3a6389e4796a04bd42f627503383b21385addf0",
        context="precision",
    )
    facts = exact_object(sidecar.get("facts"), context="precision facts")
    fact_attempts = facts.get("attempts")
    if type(fact_attempts) is not list:
        _fail("precision sidecar attempt facts are missing")
    fact_calls = tuple(
        nonblank_string(
            exact_object(value, context="precision attempt fact").get("call_id"),
            context="precision fact call ID",
        )
        for value in cast(list[object], fact_attempts)
    )
    binding = _binding(sidecar, context="precision")
    precision_inputs = exact_object(sidecar.get("inputs"), context="precision inputs")
    precision_wheel = exact_fields(
        precision_inputs.get("wheel"),
        required=("path", "sha256"),
        context="precision wheel input",
    )
    if (
        fact_calls != calls
        or facts.get("head_commit") != "d411d4537ba63e1f5b9f353853836276400f87db"
        or binding["head_sha256"]
        != "eae395a8e20aa83afe1483a646b809c48710766897c1a81430e85ba5a4aba337"
        or binding["wheel_sha256"] != precision_wheel["sha256"]
    ):
        _fail("precision sidecar facts/source binding differs from its journal")
    archive = precision_inputs.get("archive")
    archive_binding = exact_fields(
        archive, required=("path", "sha256"), context="precision archive input"
    )
    archive_path = _REPO / nonblank_string(
        archive_binding["path"], context="precision archive path"
    )
    if archive_path.resolve() != _ARCHIVE.resolve() or archive_binding["sha256"] != _ARCHIVE_SHA256:
        _fail("precision sidecar archive path/digest differs")
    _require_digest(_ARCHIVE, _ARCHIVE_SHA256, context="context archive")
    return statistics.median(item.ratio("torch_reduced", "torch_strict") for item in summaries)


def _validate_engineering_bundle() -> tuple[object, dict[str, object], bytes]:
    expected = _INPUT_DIGESTS["engineering"]
    _require_digest(_ENGINEERING, expected["artifact"], context="engineering artifact")
    _require_digest(_sidecar(_ENGINEERING), expected["manifest"], context="engineering manifest")
    _require_digest(
        _attempts(_ENGINEERING),
        expected["attempt_journal"],
        context="engineering attempt journal",
    )
    artifact = read_json(_ENGINEERING)
    analyze_artifact(artifact, archive_ratios=load_archive_ratios(_ARCHIVE))
    sidecar = _strict_object(_sidecar(_ENGINEERING), context="engineering sidecar")
    artifact_object = exact_object(artifact, context="engineering artifact")
    if (
        sidecar.get("study_id") != _STUDY_ID
        or sidecar.get("analysis_status") != _ANALYSIS_STATUS
        or sidecar.get("protocol") != artifact_object.get("protocol")
    ):
        _fail("engineering sidecar has the wrong study identity/status/protocol")
    data = exact_fields(
        sidecar.get("data"),
        required=("path", "rows", "sha256", "workloads"),
        context="engineering data",
    )
    if (
        Path(nonblank_string(data["path"], context="engineering data path")).resolve()
        != _ENGINEERING.resolve()
        or data["sha256"] != expected["artifact"]
        or exact_int(data["rows"], context="engineering rows") != 3008
        or exact_int(data["workloads"], context="engineering workloads") != 96
    ):
        _fail("engineering data binding differs")
    calls = _validate_journal_binding(
        _attempts(_ENGINEERING),
        sidecar,
        expected_calls=(_ENGINEERING_CALL,),
        expected_banks=(0,),
        expected_request_sha256="63fcd3e683b81bb4815d1770ef36a8c19d90d988f0e2c3b00d267863d916f44a",
        context="engineering",
    )
    remote = exact_fields(
        sidecar.get("remote_call"),
        required=("call_id", "payload_sha256"),
        context="engineering remote call",
    )
    completed = AttemptJournal.load(_attempts(_ENGINEERING)).records[-1]
    if remote["call_id"] != calls[0] or remote["payload_sha256"] != completed.chunk_sha256:
        _fail("engineering sidecar remote call differs from its journal")
    facts = exact_object(sidecar.get("facts"), context="engineering facts")
    if facts.get("head_commit") != _HEAD_COMMIT:
        _fail("engineering sidecar HEAD differs")
    binding = _binding(sidecar, context="engineering")
    if binding["head_sha256"] != _HEAD_SHA256 or binding["wheel_sha256"] != _WHEEL_SHA256:
        _fail("engineering HEAD/wheel binding differs")
    correctness = exact_object(sidecar.get("inputs"), context="engineering inputs").get(
        "correctness_gate"
    )
    correctness_binding = exact_fields(
        correctness,
        required=("artifact", "artifact_sha256", "manifest", "manifest_sha256"),
        context="engineering correctness input",
    )
    if (
        Path(
            nonblank_string(
                correctness_binding["artifact"], context="engineering correctness artifact path"
            )
        ).resolve()
        != _CORRECTNESS.resolve()
        or Path(
            nonblank_string(
                correctness_binding["manifest"], context="engineering correctness manifest path"
            )
        ).resolve()
        != _sidecar(_CORRECTNESS).resolve()
        or correctness_binding["artifact_sha256"] != _INPUT_DIGESTS["correctness"]["artifact"]
        or correctness_binding["manifest_sha256"] != _INPUT_DIGESTS["correctness"]["manifest"]
    ):
        _fail("engineering sidecar correctness-gate binding differs")
    wheel_provenance = _validate_wheel_binding(sidecar)
    _validate_correctness_bundle(wheel_provenance)
    precision_ratio = _validate_precision_bundle()
    if precision_ratio != 1.0:
        _fail(f"precision reduced/strict median is {precision_ratio}, expected 1.0")
    return artifact, sidecar, _ENGINEERING.read_bytes()


def _compress(raw: bytes) -> bytes:
    compressor = zstandard.ZstdCompressor(
        level=19,
        threads=1,
        write_checksum=True,
        write_content_size=False,
    )
    return compressor.compress(raw)


def _decompress(compressed: bytes) -> bytes:
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
        return reader.read()


def _summary(analysis: BenchmarkAnalysis) -> dict[str, object]:
    regimes: dict[str, object] = {}
    for regime in analysis.regimes:
        regimes[regime.regime] = {
            "all_selected_correct": regime.all_selected_correct,
            "decision": "PROCEED" if regime.passes_gate else "STOP",
            "geometric_mean_speedup": regime.geometric_mean_speedup,
            "maximum_speedup": regime.maximum_speedup,
            "median_speedup": regime.median_speedup,
            "minimum_speedup": regime.minimum_speedup,
            "percent_at_least_five_percent_faster": (regime.percent_at_least_five_percent_faster),
            "workload_count": regime.workload_count,
            "workloads_at_least_five_percent_faster": (
                regime.workloads_at_least_five_percent_faster
            ),
        }
    candidates = [
        {
            "archive_torch_over_bank1_selected_bank2_scored_triton": (
                workload.archive_torch_over_best_triton
            ),
            "best_candidate_ms": workload.best_candidate_ms,
            "best_config": workload.best_config,
            "best_config_key": workload.best_config_key,
            "correct": workload.correct,
            "regime": workload.regime,
            "torch_ms": workload.torch_ms,
            "torch_over_best_candidate": workload.torch_over_best_candidate,
            "workload_key": workload.workload_key,
        }
        for workload in analysis.workloads
    ]
    return {
        "analysis_status": _ANALYSIS_STATUS,
        "candidate_selection": candidates,
        "claim": "No superiority claim is made.",
        "contextual_baseline": {
            "display_value": "0.627",
            "frozen_value": _PRIOR_CONTEXTUAL_BASELINE,
            "role": (
                "prior published bank-1-selected, bank-2-scored torch/best-Triton "
                "context only; it is not a threshold or a fresh confirmatory comparison"
            ),
        },
        "cost_screen": {
            "evaluated_independently_by_regime": True,
            "geometric_mean_speedup_threshold": _GATE_SPEEDUP,
            "required_fraction_at_least_five_percent_faster": _GATE_WIN_FRACTION,
            "speedup_threshold_for_win": _GATE_SPEEDUP,
        },
        "evidence_scope": "one H100 bank-0 engineering screen",
        "global_decision": "PROCEED" if analysis.proceed else "STOP",
        "limitations": [
            "This is post-hoc exploratory engineering evidence, not a confirmatory endpoint.",
            "Candidate selection and scoring reuse one bank, so selection optimism is possible.",
            "Only one H100 hardware instance and one fixed 96-workload corpus were measured.",
            "The frozen STOP rule prevented three-bank selection/scoring collection.",
            "The prior 0.627266 value is contextual and comes from a different frozen protocol.",
        ],
        "precision_context": {
            "conclusion": "FP16 reduction mode does not explain the old gap.",
            "torch_reduced_over_torch_strict_median": 1.0,
        },
        "protocol": {
            "bank": analysis.bank,
            "gpu": analysis.gpu,
            "ratio": "torch median milliseconds / selected candidate median milliseconds",
            "row_count": analysis.row_count,
            "workload_count": len(analysis.workloads),
        },
        "regimes": regimes,
        "schema_version": 1,
        "study_id": analysis.study_id,
        "three_bank_collection_performed": False,
    }


def _input_manifest_entries() -> dict[str, object]:
    names = {
        "engineering": "hopper-h100-engineering.json",
        "correctness": "hopper-correctness.json",
        "precision": "h100-precision-probe.json",
    }
    result: dict[str, object] = {}
    for bundle, basename in names.items():
        digests = _INPUT_DIGESTS[bundle]
        result[bundle] = {
            "artifact": {"local_path": f"artifacts/{basename}", "sha256": digests["artifact"]},
            "attempt_journal": {
                "local_path": f"artifacts/{basename}.attempts.jsonl",
                "sha256": digests["attempt_journal"],
            },
            "manifest": {
                "local_path": f"artifacts/{basename}.manifest.json",
                "sha256": digests["manifest"],
            },
        }
    return result


def _modal_run(app_id: str, call_ids: Sequence[str]) -> dict[str, object]:
    return {
        "app_id": app_id,
        "app_url": f"https://modal.com/apps/{_MODAL_WORKSPACE}/main/{app_id}",
        "app_url_binding": "operator-recorded; app ID is not present in the local artifact bundle",
        "function_call_ids": list(call_ids),
    }


def _manifest(
    *,
    analysis: BenchmarkAnalysis,
    artifact: Mapping[str, object],
    compressed: bytes,
    raw: bytes,
    journal: bytes,
    summary: bytes,
) -> dict[str, object]:
    return {
        "analysis": {
            "analysis_status": _ANALYSIS_STATUS,
            "analyzer": {
                "path": "scripts/analyze_hopper_benchmark.py",
                "sha256": sha256_file(_ANALYZER),
            },
            "cost_screen": {
                "geometric_mean_speedup_threshold": _GATE_SPEEDUP,
                "required_fraction_at_least_five_percent_faster": _GATE_WIN_FRACTION,
                "speedup_threshold_for_win": _GATE_SPEEDUP,
            },
            "global_decision": "PROCEED" if analysis.proceed else "STOP",
            "no_superiority_claim": True,
            "three_bank_collection_performed": False,
        },
        "commands": {
            "check": "uv run python scripts/publish_hopper_engineering_result.py --check",
            "engineering_collection": "modal run modal_precision_probe.py::hopper_benchmark",
            "generate": "uv run python scripts/publish_hopper_engineering_result.py",
            "strict_analysis": "uv run python scripts/analyze_hopper_benchmark.py artifacts/hopper-h100-engineering.json",
        },
        "hardware": artifact["hardware"],
        "inputs": {
            "bundles": _input_manifest_entries(),
            "context_archive": {
                "path": "benchmarks/data/parhelion-v2-measurements.jsonl.zst",
                "sha256": _ARCHIVE_SHA256,
            },
        },
        "modal": {
            "correctness": _modal_run(_CORRECTNESS_APP, (_CORRECTNESS_CALL,)),
            "engineering": _modal_run(_ENGINEERING_APP, (_ENGINEERING_CALL,)),
            "precision": _modal_run(_PRECISION_APP, _PRECISION_CALLS),
        },
        "protocol": artifact["protocol"],
        "provenance": {
            "head_commit": _HEAD_COMMIT,
            "head_sha256": _HEAD_SHA256,
            "source_sha256": _SOURCE_SHA256,
            "wheel_manifest_sha256": _WHEEL_MANIFEST_SHA256,
            "wheel_sha256": _WHEEL_SHA256,
        },
        "publication": {
            "attempt_journal": {
                "bytes": len(journal),
                "path": "benchmarks/data/hopper-h100-engineering.attempts.jsonl",
                "sha256": _sha256_bytes(journal),
            },
            "raw": {
                "compressed_bytes": len(compressed),
                "compressed_sha256": _sha256_bytes(compressed),
                "decompressed_bytes": len(raw),
                "decompressed_sha256": _sha256_bytes(raw),
                "format": "zstd level=19, threads=1, checksum=true, content_size=false",
                "path": "benchmarks/data/hopper-h100-engineering.json.zst",
            },
            "summary": {
                "bytes": len(summary),
                "path": "benchmarks/results/hopper-h100-engineering-summary.json",
                "sha256": _sha256_bytes(summary),
            },
        },
        "publisher": {
            "path": "scripts/publish_hopper_engineering_result.py",
            "sha256": sha256_file(_PUBLISHER),
        },
        "schema_version": 1,
        "study_id": _STUDY_ID,
    }


def _json_bytes(value: object) -> bytes:
    return strict_json_dumps(value).encode("utf-8")


def _analyze(raw: bytes) -> tuple[Mapping[str, object], BenchmarkAnalysis]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("published raw artifact is not UTF-8") from exc
    artifact = exact_object(
        strict_json_loads(text, source="benchmarks/data/hopper-h100-engineering.json.zst"),
        context="published engineering artifact",
    )
    analysis = analyze_artifact(artifact, archive_ratios=load_archive_ratios(_ARCHIVE))
    return artifact, analysis


def generate() -> None:
    artifact_value, _sidecar_value, raw = _validate_engineering_bundle()
    artifact = exact_object(artifact_value, context="engineering artifact")
    analysis = analyze_artifact(artifact, archive_ratios=load_archive_ratios(_ARCHIVE))
    summary = _json_bytes(_summary(analysis))
    compressed = _compress(raw)
    journal = _attempts(_ENGINEERING).read_bytes()
    manifest = _json_bytes(
        _manifest(
            analysis=analysis,
            artifact=artifact,
            compressed=compressed,
            raw=raw,
            journal=journal,
            summary=summary,
        )
    )
    write_bytes_atomic(_RAW, compressed)
    write_bytes_atomic(_JOURNAL, journal)
    write_bytes_atomic(_SUMMARY, summary)
    write_bytes_atomic(_MANIFEST, manifest)
    print("published post-hoc exploratory H100 engineering evidence: global STOP")


def _validate_published_journal(journal: bytes) -> None:
    if _sha256_bytes(journal) != _INPUT_DIGESTS["engineering"]["attempt_journal"]:
        _fail("published engineering journal differs from the collected input journal")
    records = AttemptJournal.load(_JOURNAL).records
    if len(records) != 2 or records[0].status != "spawned" or records[1].status != "completed":
        _fail("published engineering journal does not contain one completed transition")
    if any(record.call_id != _ENGINEERING_CALL for record in records):
        _fail("published engineering journal has the wrong FunctionCall ID")


def check(*, compare_head: bool = False) -> None:
    for path in (_RAW, _JOURNAL, _SUMMARY, _MANIFEST, _ARCHIVE, _ANALYZER, _PUBLISHER):
        if not path.is_file():
            _fail(f"required committed evidence file is missing: {path.relative_to(_REPO)}")
    _require_digest(_ARCHIVE, _ARCHIVE_SHA256, context="context archive")
    compressed = _RAW.read_bytes()
    raw = _decompress(compressed)
    if _sha256_bytes(raw) != _INPUT_DIGESTS["engineering"]["artifact"]:
        _fail("decompressed raw artifact digest differs from the collected input")
    if _compress(raw) != compressed:
        _fail("published raw artifact is not the deterministic zstd encoding")
    artifact, analysis = _analyze(raw)
    journal = _JOURNAL.read_bytes()
    _validate_published_journal(journal)
    expected_summary = _json_bytes(_summary(analysis))
    actual_summary = _SUMMARY.read_bytes()
    if actual_summary != expected_summary:
        _fail("published summary is not the byte-exact strict-analysis result")
    manifest_value = _strict_object(_MANIFEST, context="publication manifest")
    publication = exact_object(manifest_value.get("publication"), context="published bindings")
    raw_binding = exact_object(publication.get("raw"), context="published raw binding")
    journal_binding = exact_object(
        publication.get("attempt_journal"), context="published journal binding"
    )
    summary_binding = exact_object(publication.get("summary"), context="published summary binding")
    if (
        raw_binding.get("compressed_sha256") != _sha256_bytes(compressed)
        or raw_binding.get("decompressed_sha256") != _sha256_bytes(raw)
        or journal_binding.get("sha256") != _sha256_bytes(journal)
        or summary_binding.get("sha256") != _sha256_bytes(actual_summary)
    ):
        _fail("publication manifest has a raw/journal/summary digest mismatch")
    expected_manifest = _json_bytes(
        _manifest(
            analysis=analysis,
            artifact=artifact,
            compressed=compressed,
            raw=raw,
            journal=journal,
            summary=expected_summary,
        )
    )
    if _MANIFEST.read_bytes() != expected_manifest:
        _fail("publication manifest is not the byte-exact regenerated manifest")
    if compare_head:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != _HEAD_COMMIT:
            _fail(f"reported collection HEAD is {_HEAD_COMMIT}, current checkout is {head}")
    print("hopper-h100-engineering committed evidence: OK (post-hoc exploratory; global STOP)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check only committed evidence bytes")
    parser.add_argument(
        "--compare-head",
        action="store_true",
        help="also compare the reported collection HEAD with the current Git checkout",
    )
    args = parser.parse_args(argv)
    if args.compare_head and not args.check:
        parser.error("--compare-head requires --check")
    if args.check:
        check(compare_head=cast(bool, args.compare_head))
    else:
        generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
