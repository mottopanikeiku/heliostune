"""Publish and byte-check the H100 engineering and precision-probe evidence.

Generation reads the gitignored local collection, correctness, and precision bundles. Check mode
uses only committed files and never contacts Git or Modal unless --compare-head is explicitly used.
Both modes derive the precision finding with the precision analyzer's pure verdict function.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn, cast

import zstandard
from analyze_hopper_benchmark import (
    BenchmarkAnalysis,
    analyze_artifact,
    load_archive_ratios,
)
from analyze_precision_probe import (
    ExplanationVerdict,
    WorkloadSummary,
    evaluate_explanation,
    load_summaries,
)

from heliostune.artifacts import (
    read_json,
    strict_json_dumps,
    strict_json_loads,
    write_bytes_atomic,
)
from heliostune.collection import AttemptJournal, sha256_file
from heliostune.schema import HardwareProfile
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
_PRECISION_PROVENANCE_MANIFEST = _LOCAL / "history/d411d45/hopper-correctness.json.manifest.json"
_ARCHIVE = _REPO / "benchmarks/data/parhelion-v2-measurements.jsonl.zst"
_PRICING_SOURCE = _REPO / "benchmarks/parhelion-v2-h100-freeze.json"
_RAW = _REPO / "benchmarks/data/hopper-h100-engineering.json.zst"
_JOURNAL = _REPO / "benchmarks/data/hopper-h100-engineering.attempts.jsonl"
_IMMUTABLE_SUMMARY = _REPO / "benchmarks/results/hopper-h100-engineering-summary.json"
_IMMUTABLE_MANIFEST = _REPO / "benchmarks/hopper-h100-engineering-manifest.json"
_SUMMARY = _REPO / "benchmarks/results/hopper-h100-engineering-summary-v2.json"
_MANIFEST = _REPO / "benchmarks/hopper-h100-engineering-manifest-v2.json"
_PRECISION_RAW = _REPO / "benchmarks/data/h100-precision-probe.json.zst"
_PRECISION_JOURNAL = _REPO / "benchmarks/data/h100-precision-probe.attempts.jsonl"
_PRECISION_SUMMARY = _REPO / "benchmarks/results/h100-precision-probe-summary.json"
_PRECISION_MANIFEST = _REPO / "benchmarks/h100-precision-probe-manifest.json"
_ANALYZER = _REPO / "scripts/analyze_hopper_benchmark.py"
_PRECISION_ANALYZER = _REPO / "scripts/analyze_precision_probe.py"
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
_PRECISION_HEAD_COMMIT = "d411d4537ba63e1f5b9f353853836276400f87db"
_PRECISION_HEAD_SHA256 = "eae395a8e20aa83afe1483a646b809c48710766897c1a81430e85ba5a4aba337"
_PRECISION_SOURCE_SHA256 = "e9c96a912b24dac451f0771034ac73fbb25753f74dd4055871896e560b67427b"
_PRECISION_WHEEL_SHA256 = "86be6fecadca24d5e24468f06a0fb9cbb8cc297cdd39437f134b7a7cda77e9bc"
_PRECISION_REQUEST_SHA256 = "00c7c9f8c50eb1f336d20e9aa3a6389e4796a04bd42f627503383b21385addf0"
_PRECISION_PROTOCOL_SHA256 = "6deeba1ca79128903bbc8f59f20e49a0ca25a284a5bc09631cdcc8e32ef5b9f4"
_PRECISION_PROVENANCE_MANIFEST_SHA256 = (
    "1d53b318a3b2fd348304751d0e5ab4aebb496b7d76aea4785e006cb590088b56"
)
_PRECISION_CONFIG_SHA256 = "c64e7908e334541ebba47d51780c2c4c06f48bf809014bca2bb68cbea0973f08"
_MEANINGFUL_PAIRED_EFFECT = 0.05
_BASELINE_RELATIVE_TOLERANCE = 0.05
_H100_RATE_USD_PER_SECOND = 0.001097
_PRICING_CHECKED_AT_UTC = "2026-08-23"
_PRICING_SOURCE_SHA256 = "c9c7138ef812166756746687463f81b88b63f905fb6998b9f468f1b0dadb0b4a"
_IMMUTABLE_SUMMARY_SHA256 = "5d14a49c416d8cd4282c1d382002b2b33f7146a83f606bb7f2c70c3b6382eea5"
_IMMUTABLE_MANIFEST_SHA256 = "c6a65d8216de0bc736afdabab5d6cc5c0aa26fe9f23b3c837b71f5d86571ae99"
_PRECISION_RAW_SCHEMA = "h100-precision-probe-raw-v2"
_ACTUAL_COST_UNKNOWN_REASON = (
    "The actual Modal bill and per-call billable GPU duration were unavailable."
)

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


def _digest_string(value: object, *, context: str) -> str:
    digest = nonblank_string(value, context=context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(f"{context} must be a 64-character lowercase hexadecimal SHA-256")
    return digest


def _head_string(value: object, *, context: str) -> str:
    head = nonblank_string(value, context=context)
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        _fail(f"{context} must be a 40-character lowercase hexadecimal commit")
    return head


def _hardware_dict(value: object, *, context: str) -> dict[str, object]:
    try:
        return HardwareProfile.from_dict(value).to_dict()
    except ValueError as exc:
        raise ValueError(f"{context} is not a HardwareProfile: {exc}") from exc


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


def _precision_local_provenance(sidecar: Mapping[str, object]) -> dict[str, str]:
    _require_digest(
        _PRECISION_PROVENANCE_MANIFEST,
        _PRECISION_PROVENANCE_MANIFEST_SHA256,
        context="precision source provenance manifest",
    )
    source_manifest = _strict_object(
        _PRECISION_PROVENANCE_MANIFEST,
        context="precision source provenance manifest",
    )
    source_facts = exact_object(
        source_manifest.get("facts"), context="precision source provenance facts"
    )
    source_binding = _binding(source_manifest, context="precision source provenance")
    source_inputs = exact_object(
        source_manifest.get("inputs"), context="precision source provenance inputs"
    )
    source = exact_fields(
        source_inputs.get("source"),
        required=("sha256",),
        context="precision source provenance source",
    )
    source_wheel = exact_fields(
        source_inputs.get("wheel"),
        required=("path", "sha256"),
        context="precision source provenance wheel",
    )
    facts = exact_object(sidecar.get("facts"), context="precision facts")
    binding = _binding(sidecar, context="precision")
    provenance = {
        "head_commit": _head_string(facts.get("head_commit"), context="precision HEAD"),
        "source_sha256": _digest_string(source["sha256"], context="precision source SHA-256"),
        "wheel_sha256": _digest_string(binding["wheel_sha256"], context="precision wheel SHA-256"),
    }
    if (
        _head_string(source_facts.get("head_commit"), context="precision source provenance HEAD")
        != provenance["head_commit"]
        or _digest_string(
            source_binding["wheel_sha256"],
            context="precision source provenance wheel binding",
        )
        != provenance["wheel_sha256"]
        or _digest_string(source_wheel["sha256"], context="precision source provenance wheel")
        != provenance["wheel_sha256"]
        or provenance
        != {
            "head_commit": _PRECISION_HEAD_COMMIT,
            "source_sha256": _PRECISION_SOURCE_SHA256,
            "wheel_sha256": _PRECISION_WHEEL_SHA256,
        }
    ):
        _fail("precision provenance manifests disagree on HEAD/source/wheel")
    return provenance


def _engineering_local_provenance(
    wheel_provenance: Mapping[str, str],
) -> dict[str, str]:
    provenance = {
        "head_commit": _head_string(
            wheel_provenance.get("head_commit"), context="engineering HEAD"
        ),
        "source_sha256": _digest_string(
            wheel_provenance.get("source_sha256"), context="engineering source SHA-256"
        ),
        "wheel_sha256": _digest_string(
            wheel_provenance.get("wheel_sha256"), context="engineering wheel SHA-256"
        ),
    }
    if provenance != {
        "head_commit": _HEAD_COMMIT,
        "source_sha256": _SOURCE_SHA256,
        "wheel_sha256": _WHEEL_SHA256,
    }:
        _fail("engineering provenance manifest disagrees on HEAD/source/wheel")
    return provenance


def _validate_precision_bundle() -> tuple[float, dict[str, str]]:
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
    return (
        statistics.median(item.ratio("torch_reduced", "torch_strict") for item in summaries),
        _precision_local_provenance(sidecar),
    )


def _validate_engineering_bundle() -> tuple[
    object,
    bytes,
    dict[str, str],
    dict[str, str],
]:
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
    precision_ratio, precision_provenance = _validate_precision_bundle()
    if precision_ratio != 1.0:
        _fail(f"precision reduced/strict median is {precision_ratio}, expected 1.0")
    return (
        artifact,
        _ENGINEERING.read_bytes(),
        _engineering_local_provenance(wheel_provenance),
        precision_provenance,
    )


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


def _parse_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"attempt journal timestamp is not ISO-8601: {value!r}") from exc
    if timestamp.tzinfo is None:
        _fail(f"attempt journal timestamp lacks a UTC offset: {value!r}")
    return timestamp


def _engineering_cost_exclusions() -> tuple[dict[str, object], ...]:
    return (
        {
            "actual_h100_cost_usd": None,
            "actual_h100_cost_unknown_reason": _ACTUAL_COST_UNKNOWN_REASON,
            "call_ids": [_CORRECTNESS_CALL],
            "role": "prerequisite_correctness_call",
        },
        {
            "actual_h100_cost_usd": None,
            "actual_h100_cost_unknown_reason": _ACTUAL_COST_UNKNOWN_REASON,
            "call_ids": list(_PRECISION_CALLS),
            "role": "related_precision_probe_calls",
        },
    )


def _collection_accounting(
    journal_path: Path,
    *,
    expected_rows: int,
    published_rows: int,
    cost_scope: str,
    covered_call_ids: Sequence[str],
    excluded_calls: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    records = AttemptJournal.load(journal_path).records
    if not records:
        _fail(f"{journal_path} contains no attempt records")
    attempted = sum(record.status == "spawned" for record in records)
    completed = sum(record.status == "completed" for record in records)
    failed = sum(record.status == "failed" for record in records)
    spawned_call_ids = [record.call_id for record in records if record.status == "spawned"]
    if spawned_call_ids != list(covered_call_ids):
        _fail(f"{journal_path} call IDs differ from the declared cost scope")
    distinct_attempts = {
        (record.gpu, record.bank) for record in records if record.status == "spawned"
    }
    retried = attempted - len(distinct_attempts)
    if published_rows > expected_rows:
        _fail(f"published row count {published_rows} exceeds expected count {expected_rows}")
    timestamps = [_parse_timestamp(record.timestamp_utc) for record in records]
    elapsed_seconds = (max(timestamps) - min(timestamps)).total_seconds()
    return {
        "attempts": {
            "attempted": attempted,
            "completed": completed,
            "failed": failed,
            "retried": retried,
        },
        "cost": {
            "scope": nonblank_string(cost_scope, context="cost scope"),
            "covered_call_ids": list(covered_call_ids),
            "excluded_calls": [dict(item) for item in excluded_calls],
            "actual_h100_cost_usd": None,
            "actual_h100_cost_unknown_reason": _ACTUAL_COST_UNKNOWN_REASON,
            "published_rate_estimate": {
                "amount_usd": round(elapsed_seconds * _H100_RATE_USD_PER_SECOND, 12),
                "classification": "estimated",
                "gpu_rate_usd_per_second": _H100_RATE_USD_PER_SECOND,
                "limitations": (
                    "GPU line item only for the covered calls; excludes CPU, memory, "
                    "queueing/billing adjustments, discounts, taxes, and every excluded call."
                ),
                "rate_source": {
                    "checked_at_utc": _PRICING_CHECKED_AT_UTC,
                    "path": "benchmarks/parhelion-v2-h100-freeze.json",
                    "sha256": _PRICING_SOURCE_SHA256,
                    "url": "https://modal.com/pricing",
                },
                "time_basis": (
                    "Journal wall time for the covered calls from first transition through last "
                    "transition; this can differ from billable GPU-active seconds."
                ),
            },
        },
        "elapsed_seconds": elapsed_seconds,
        "rows": {
            "expected": expected_rows,
            "failed": 0,
            "omitted": expected_rows - published_rows,
            "published": published_rows,
        },
    }


def _precision_finding(
    verdict: ExplanationVerdict,
    summaries: Sequence[WorkloadSummary],
) -> dict[str, object]:
    conclusions = {
        "does not explain": (
            "Under the frozen descriptive thresholds, reduced FP16 reduction does not explain "
            "the archived gap in this fixed H100 probe."
        ),
        "inconclusive": (
            "The fixed H100 probe is inconclusive about whether reduced FP16 reduction explains "
            "the archived gap."
        ),
        "supports explanation": (
            "Under the frozen descriptive thresholds, the fixed H100 probe supports reduced FP16 "
            "reduction as an explanation of the archived gap."
        ),
    }
    try:
        conclusion = conclusions[verdict.classification]
    except KeyError as exc:
        raise ValueError(f"unknown precision verdict {verdict.classification!r}") from exc
    return {
        "accuracy_regression": verdict.accuracy_regression,
        "baseline_agrees": verdict.baseline_agrees,
        "classification": verdict.classification,
        "conclusion": conclusion,
        "metrics": {
            "archive_baseline_torch_over_best_triton": verdict.baseline_ratio,
            "paired_strict_slowdown": verdict.paired_strict_slowdown,
            "torch_reduced_over_torch_strict_median": statistics.median(
                item.ratio("torch_reduced", "torch_strict") for item in summaries
            ),
            "torch_reduced_over_triton_median": verdict.reduced_triton_ratio,
            "torch_strict_over_triton_median": statistics.median(
                item.ratio("torch_strict", "triton") for item in summaries
            ),
        },
        "parity_authorized": verdict.parity_authorized,
        "thresholds": {
            "baseline_absolute_tolerance": verdict.baseline_absolute_tolerance,
            "baseline_relative_tolerance": _BASELINE_RELATIVE_TOLERANCE,
            "meaningful_paired_effect": _MEANINGFUL_PAIRED_EFFECT,
        },
    }


def _precision_summary(
    artifact: Mapping[str, object],
    summaries: Sequence[WorkloadSummary],
    verdict: ExplanationVerdict,
    accounting: Mapping[str, object],
    publication: Mapping[str, object],
) -> dict[str, object]:
    rows = artifact.get("rows")
    if type(rows) is not list or len(rows) != 288 or len(summaries) != 96:
        _fail("precision publication must contain exactly 288 rows and 96 workloads")
    limitations = [
        "This is post-hoc exploratory descriptive evidence, not a confirmatory endpoint.",
        "The thresholds were written for this diagnostic and do not establish causality.",
        "Three calls reported one matching observed H100 profile/SKU and measured one fixed "
        "96-workload corpus.",
        "The archived comparator selected Triton configurations on bank 1 and scored bank 2, while "
        "the probe reports three-bank medians.",
        "The probe cannot revise the frozen Hopper engineering STOP or any Parhelion claim.",
    ]
    return {
        "analysis_status": _ANALYSIS_STATUS,
        "claim_classification": {
            "candidate_role": "torch.matmul with FP16 reduced-precision reduction enabled",
            "claim_kind": "descriptive",
            "comparator_role": "paired torch.matmul with FP16 reduced-precision reduction disabled",
            "decision": "supported",
            "evidence_class": "exploratory",
            "inferential": False,
            "limitations": limitations,
            "reference_role": (
                "FP32-output torch.mm numerical reference with TF32 disabled and the contextual "
                "archived bank-1-selected, bank-2-scored Triton result"
            ),
            "scope": (
                "one observed H100 profile/SKU across three calls, three banks, and the fixed "
                "96-workload corpus"
            ),
        },
        "collection_accounting": dict(accounting),
        "evidence_scope": (
            "one observed H100 profile/SKU across three calls, three banks, 96 workloads, "
            "and 288 paired rows"
        ),
        "limitations": limitations,
        "precision_finding": _precision_finding(verdict, summaries),
        "publication": dict(publication),
        "protocol": artifact["protocol"],
        "row_count": 288,
        "schema_version": 1,
        "study_id": "h100-fp16-reduction-probe",
        "workload_count": 96,
        "workloads": [
            {
                "archive_torch_over_best_triton": item.archive_ratio,
                "banks": list(item.banks),
                "errors": item.errors,
                "k": item.k,
                "latencies_ms": item.latencies,
                "m": item.m,
                "n": item.n,
                "ratios": {
                    "torch_reduced_over_torch_strict": item.ratio("torch_reduced", "torch_strict"),
                    "torch_reduced_over_triton": item.ratio("torch_reduced", "triton"),
                    "torch_strict_over_triton": item.ratio("torch_strict", "triton"),
                },
                "workload_key": item.key,
            }
            for item in summaries
        ],
    }


def _summary(
    analysis: BenchmarkAnalysis,
    *,
    accounting: Mapping[str, object],
    precision_summaries: Sequence[WorkloadSummary],
    precision_verdict: ExplanationVerdict,
    precision_publication: Mapping[str, object],
    publication: Mapping[str, object],
) -> dict[str, object]:
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
    limitations = [
        "This is post-hoc exploratory engineering evidence, not a confirmatory endpoint.",
        "Candidate selection and scoring reuse one bank, so selection optimism is possible.",
        "One timing call reported one observed H100 profile/SKU and measured one fixed "
        "96-workload corpus.",
        "The frozen STOP rule prevented three-bank selection/scoring collection.",
        "The prior 0.627266 value is contextual and comes from a different frozen protocol.",
    ]
    precision_finding = _precision_finding(precision_verdict, precision_summaries)
    precision_finding["committed_evidence"] = dict(precision_publication)
    return {
        "analysis_status": _ANALYSIS_STATUS,
        "candidate_selection": candidates,
        "claim_classification": {
            "candidate_role": "best post-hoc selected Triton candidate in each engineering regime",
            "claim_kind": "descriptive",
            "comparator_role": "torch.matmul measured on the same bank",
            "decision": "stopped",
            "evidence_class": "exploratory",
            "inferential": False,
            "limitations": limitations,
            "reference_role": (
                "FP32-output torch.mm correctness reference with TF32 disabled and the contextual "
                "frozen v2 archive"
            ),
            "scope": (
                "one observed H100 profile/SKU in one bank-0 timing call and the fixed "
                "96-workload corpus"
            ),
        },
        "collection_accounting": dict(accounting),
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
        "evidence_scope": (
            "one observed H100 profile/SKU in one H100 bank-0 engineering timing call"
        ),
        "global_decision": "PROCEED" if analysis.proceed else "STOP",
        "limitations": limitations,
        "precision_finding": precision_finding,
        "publication": dict(publication),
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


def _precision_hardware(artifact: Mapping[str, object]) -> dict[str, object]:
    banks_value = artifact.get("banks")
    if type(banks_value) is not list or len(banks_value) != 3:
        _fail("precision artifact must contain three bank records")
    profiles = [
        _hardware_dict(
            exact_fields(
                value,
                required=("bank", "call_id", "chunk_sha256", "hardware"),
                context="precision bank record",
            )["hardware"],
            context="precision bank hardware",
        )
        for value in cast(list[object], banks_value)
    ]
    if any(profile != profiles[0] for profile in profiles[1:]):
        _fail("precision bank records disagree on the observed hardware profile/SKU")
    return profiles[0]


def _runtime_from_hardware(hardware: Mapping[str, object]) -> dict[str, str]:
    runtime: dict[str, str] = {}
    for field in ("cuda_version", "torch_version", "triton_version"):
        value = hardware.get(field)
        if value is not None:
            runtime[field] = nonblank_string(value, context=f"hardware {field}")
    return runtime


def _summary_publication(
    *,
    artifact: Mapping[str, object],
    compressed: bytes,
    raw: bytes,
    journal: bytes,
    journal_path: Path,
    raw_path: str,
    journal_publication_path: str,
    manifest_path: str,
    hardware: Mapping[str, object],
    provenance: Mapping[str, str],
    app_id: str,
    expected_call_ids: Sequence[str],
    raw_schema: str | None = None,
) -> dict[str, object]:
    rows = artifact.get("rows")
    if type(rows) is not list:
        _fail("publication raw artifact rows must be an array")
    records = AttemptJournal.load(journal_path).records
    completed_call_ids = [record.call_id for record in records if record.status == "completed"]
    if completed_call_ids != list(expected_call_ids):
        _fail("publication journal completed calls differ from the validated call IDs")
    hardware_value = _hardware_dict(hardware, context="publication hardware")
    raw_publication: dict[str, object] = {
        "path": raw_path,
        "sha256": _sha256_bytes(compressed),
        "uncompressed_sha256": _sha256_bytes(raw),
        "bytes": len(compressed),
        "rows": len(rows),
    }
    if raw_schema is not None:
        schema_version = exact_int(
            artifact.get("schema_version"), context="publication raw schema version", minimum=1
        )
        raw_publication["schema"] = nonblank_string(raw_schema, context="publication raw schema")
        raw_publication["schema_version"] = schema_version
    return {
        "raw": raw_publication,
        "journal": {
            "path": journal_publication_path,
            "sha256": _sha256_bytes(journal),
            "records": len(records),
        },
        "manifest_path": manifest_path,
        "head_commit": _head_string(provenance.get("head_commit"), context="publication HEAD"),
        "source_sha256": _digest_string(
            provenance.get("source_sha256"), context="publication source SHA-256"
        ),
        "wheel_sha256": _digest_string(
            provenance.get("wheel_sha256"), context="publication wheel SHA-256"
        ),
        "hardware": hardware_value,
        "runtime": _runtime_from_hardware(hardware_value),
        "modal": {
            "app_url": f"https://modal.com/apps/{_MODAL_WORKSPACE}/main/{app_id}",
            "app_id": app_id,
            "app_url_provenance": "operator_recorded",
            "call_ids": completed_call_ids,
        },
    }


def _validate_summary_publication(
    value: object,
    *,
    expected: Mapping[str, object],
    context: str,
) -> None:
    publication = exact_fields(
        value,
        required=(
            "raw",
            "journal",
            "manifest_path",
            "head_commit",
            "source_sha256",
            "wheel_sha256",
            "hardware",
            "runtime",
            "modal",
        ),
        context=f"{context} publication",
    )
    expected_raw = exact_object(expected.get("raw"), context=f"{context} expected publication raw")
    raw_required: tuple[str, ...] = ("path", "sha256", "uncompressed_sha256", "bytes", "rows")
    if "schema" in expected_raw or "schema_version" in expected_raw:
        raw_required += ("schema", "schema_version")
    raw = exact_fields(
        publication["raw"],
        required=raw_required,
        context=f"{context} publication raw",
    )
    nonblank_string(raw["path"], context=f"{context} publication raw path")
    _digest_string(raw["sha256"], context=f"{context} publication raw SHA-256")
    _digest_string(
        raw["uncompressed_sha256"],
        context=f"{context} publication uncompressed SHA-256",
    )
    exact_int(raw["bytes"], context=f"{context} publication raw bytes", minimum=1)
    exact_int(raw["rows"], context=f"{context} publication raw rows", minimum=1)
    if "schema" in raw:
        nonblank_string(raw["schema"], context=f"{context} publication raw schema")
        exact_int(
            raw["schema_version"],
            context=f"{context} publication raw schema version",
            minimum=1,
        )
    journal = exact_fields(
        publication["journal"],
        required=("path", "sha256", "records"),
        context=f"{context} publication journal",
    )
    nonblank_string(journal["path"], context=f"{context} publication journal path")
    _digest_string(journal["sha256"], context=f"{context} publication journal SHA-256")
    exact_int(journal["records"], context=f"{context} publication records", minimum=1)
    nonblank_string(publication["manifest_path"], context=f"{context} publication manifest path")
    _head_string(publication["head_commit"], context=f"{context} publication HEAD")
    _digest_string(publication["source_sha256"], context=f"{context} publication source SHA-256")
    _digest_string(publication["wheel_sha256"], context=f"{context} publication wheel SHA-256")
    _hardware_dict(publication["hardware"], context=f"{context} publication hardware")
    runtime = exact_object(publication["runtime"], context=f"{context} publication runtime")
    for name, version in runtime.items():
        nonblank_string(name, context=f"{context} publication runtime name")
        nonblank_string(version, context=f"{context} publication runtime {name}")
    modal = exact_fields(
        publication["modal"],
        required=("app_url", "app_id", "app_url_provenance", "call_ids"),
        context=f"{context} publication modal",
    )
    app_url = nonblank_string(modal["app_url"], context=f"{context} publication app URL")
    if not app_url.startswith("https://"):
        _fail(f"{context} publication app URL must use HTTPS")
    nonblank_string(modal["app_id"], context=f"{context} publication app ID")
    if modal["app_url_provenance"] != "operator_recorded":
        _fail(f"{context} publication app URL provenance is not operator_recorded")
    call_ids = modal["call_ids"]
    if type(call_ids) is not list:
        _fail(f"{context} publication call IDs must be an array")
    for index, call_id in enumerate(cast(list[object], call_ids)):
        nonblank_string(call_id, context=f"{context} publication call ID {index}")
    if publication != expected:
        _fail(f"{context} publication differs from validated committed evidence")


def _committed_provenance(
    manifest: Mapping[str, object],
    *,
    expected: Mapping[str, str],
    context: str,
) -> dict[str, str]:
    source = exact_object(manifest.get("provenance"), context=f"{context} provenance")
    provenance = {
        "head_commit": _head_string(source.get("head_commit"), context=f"{context} HEAD"),
        "source_sha256": _digest_string(
            source.get("source_sha256"), context=f"{context} source SHA-256"
        ),
        "wheel_sha256": _digest_string(
            source.get("wheel_sha256"), context=f"{context} wheel SHA-256"
        ),
    }
    if provenance != expected:
        _fail(f"{context} manifest provenance differs from validated collection provenance")
    return provenance


def _precision_manifest(
    *,
    accounting: Mapping[str, object],
    artifact: Mapping[str, object],
    compressed: bytes,
    journal: bytes,
    raw: bytes,
    summary: bytes,
    verdict: ExplanationVerdict,
) -> dict[str, object]:
    hardware = _precision_hardware(artifact)
    return {
        "analysis": {
            "analysis_status": _ANALYSIS_STATUS,
            "analyzer": {
                "path": "scripts/analyze_precision_probe.py",
                "sha256": sha256_file(_PRECISION_ANALYZER),
            },
            "pure_verdict_function": "analyze_precision_probe.evaluate_explanation",
            "verdict": {
                "classification": verdict.classification,
                "inferential": False,
                "parity_authorized": verdict.parity_authorized,
            },
        },
        "collection_accounting": dict(accounting),
        "commands": {
            "check": "uv run python scripts/publish_hopper_engineering_result.py --check",
            "collection": "modal run modal_precision_probe.py::precision_probe",
            "generate": "uv run python scripts/publish_hopper_engineering_result.py",
            "strict_analysis": (
                "uv run python scripts/analyze_precision_probe.py "
                "artifacts/h100-precision-probe.json"
            ),
        },
        "evidence_class": "exploratory",
        "hardware": hardware,
        "inputs": {
            "context_archive": {
                "path": "benchmarks/data/parhelion-v2-measurements.jsonl.zst",
                "sha256": _ARCHIVE_SHA256,
            },
            "local_sidecar": {
                "local_path": "artifacts/h100-precision-probe.json.manifest.json",
                "sha256": _INPUT_DIGESTS["precision"]["manifest"],
            },
        },
        "modal": _modal_run(_PRECISION_APP, _PRECISION_CALLS),
        "protocol": artifact["protocol"],
        "provenance": {
            "config_manifest_sha256": _PRECISION_CONFIG_SHA256,
            "head_commit": _PRECISION_HEAD_COMMIT,
            "head_sha256": _PRECISION_HEAD_SHA256,
            "protocol_sha256": _PRECISION_PROTOCOL_SHA256,
            "request_sha256": _PRECISION_REQUEST_SHA256,
            "source_sha256": _PRECISION_SOURCE_SHA256,
            "wheel_sha256": _PRECISION_WHEEL_SHA256,
        },
        "publication": {
            "attempt_journal": {
                "bytes": len(journal),
                "path": "benchmarks/data/h100-precision-probe.attempts.jsonl",
                "sha256": _sha256_bytes(journal),
            },
            "raw": {
                "compressed_bytes": len(compressed),
                "compressed_sha256": _sha256_bytes(compressed),
                "decompressed_bytes": len(raw),
                "decompressed_sha256": _sha256_bytes(raw),
                "format": "zstd level=19, threads=1, checksum=true, content_size=false",
                "path": "benchmarks/data/h100-precision-probe.json.zst",
                "rows": 288,
                "schema": _PRECISION_RAW_SCHEMA,
                "schema_version": 2,
            },
            "summary": {
                "bytes": len(summary),
                "path": "benchmarks/results/h100-precision-probe-summary.json",
                "sha256": _sha256_bytes(summary),
            },
        },
        "publisher": {
            "path": "scripts/publish_hopper_engineering_result.py",
            "sha256": sha256_file(_PUBLISHER),
        },
        "schema_version": 1,
        "study_id": "h100-fp16-reduction-probe",
    }


def _manifest(
    *,
    accounting: Mapping[str, object],
    analysis: BenchmarkAnalysis,
    artifact: Mapping[str, object],
    compressed: bytes,
    raw: bytes,
    journal: bytes,
    precision_publication: Mapping[str, object],
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
        "collection_accounting": dict(accounting),
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
            "committed_precision_evidence": dict(precision_publication),
            "immutable_v1_publication": {
                "manifest": {
                    "path": "benchmarks/hopper-h100-engineering-manifest.json",
                    "sha256": _IMMUTABLE_MANIFEST_SHA256,
                },
                "summary": {
                    "path": "benchmarks/results/hopper-h100-engineering-summary.json",
                    "sha256": _IMMUTABLE_SUMMARY_SHA256,
                },
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
                "path": "benchmarks/results/hopper-h100-engineering-summary-v2.json",
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


def _precision_publication_binding(
    *,
    compressed: bytes,
    journal: bytes,
    manifest: bytes,
    summary: bytes,
) -> dict[str, object]:
    return {
        "attempt_journal": {
            "path": "benchmarks/data/h100-precision-probe.attempts.jsonl",
            "sha256": _sha256_bytes(journal),
        },
        "manifest": {
            "path": "benchmarks/h100-precision-probe-manifest.json",
            "sha256": _sha256_bytes(manifest),
        },
        "raw": {
            "compressed_sha256": _sha256_bytes(compressed),
            "decompressed_sha256": _INPUT_DIGESTS["precision"]["artifact"],
            "path": "benchmarks/data/h100-precision-probe.json.zst",
            "schema": _PRECISION_RAW_SCHEMA,
            "schema_version": 2,
        },
        "summary": {
            "path": "benchmarks/results/h100-precision-probe-summary.json",
            "sha256": _sha256_bytes(summary),
        },
    }


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


def _analyze_precision(
    raw: bytes,
) -> tuple[Mapping[str, object], list[WorkloadSummary], ExplanationVerdict]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("published precision raw artifact is not UTF-8") from exc
    artifact = exact_object(
        strict_json_loads(text, source="benchmarks/data/h100-precision-probe.json.zst"),
        context="published precision artifact",
    )
    if exact_int(artifact.get("schema_version"), context="precision raw schema version") != 2:
        _fail("precision raw schema must be h100-precision-probe-raw-v2 (schema_version 2)")
    with TemporaryDirectory(prefix="heliostune-precision-check-") as directory:
        path = Path(directory) / "h100-precision-probe.json"
        path.write_bytes(raw)
        loaded, summaries = load_summaries(path)
    if artifact != loaded:
        _fail("precision analyzer decoded a different artifact value")
    verdict = evaluate_explanation(
        summaries,
        baseline_ratio=_PRIOR_CONTEXTUAL_BASELINE,
        meaningful_effect=_MEANINGFUL_PAIRED_EFFECT,
        baseline_relative_tolerance=_BASELINE_RELATIVE_TOLERANCE,
    )
    return artifact, summaries, verdict


def _validate_immutable_v1() -> None:
    _require_digest(
        _IMMUTABLE_SUMMARY,
        _IMMUTABLE_SUMMARY_SHA256,
        context="immutable Hopper engineering v1 summary",
    )
    _require_digest(
        _IMMUTABLE_MANIFEST,
        _IMMUTABLE_MANIFEST_SHA256,
        context="immutable Hopper engineering v1 manifest",
    )


def generate() -> None:
    _validate_immutable_v1()
    artifact_value, raw, engineering_provenance, precision_provenance = (
        _validate_engineering_bundle()
    )
    _require_digest(_PRICING_SOURCE, _PRICING_SOURCE_SHA256, context="pricing source")

    precision_raw = _PRECISION.read_bytes()
    precision_artifact, precision_summaries, precision_verdict = _analyze_precision(precision_raw)
    precision_compressed = _compress(precision_raw)
    precision_journal = _attempts(_PRECISION).read_bytes()
    precision_accounting = _collection_accounting(
        _attempts(_PRECISION),
        expected_rows=288,
        published_rows=288,
        cost_scope="precision_probe_calls_only",
        covered_call_ids=_PRECISION_CALLS,
    )
    precision_summary_publication = _summary_publication(
        artifact=precision_artifact,
        compressed=precision_compressed,
        raw=precision_raw,
        journal=precision_journal,
        journal_path=_attempts(_PRECISION),
        raw_path="benchmarks/data/h100-precision-probe.json.zst",
        journal_publication_path="benchmarks/data/h100-precision-probe.attempts.jsonl",
        manifest_path="benchmarks/h100-precision-probe-manifest.json",
        hardware=_precision_hardware(precision_artifact),
        provenance=precision_provenance,
        app_id=_PRECISION_APP,
        expected_call_ids=_PRECISION_CALLS,
        raw_schema=_PRECISION_RAW_SCHEMA,
    )
    precision_summary = _json_bytes(
        _precision_summary(
            precision_artifact,
            precision_summaries,
            precision_verdict,
            precision_accounting,
            precision_summary_publication,
        )
    )
    precision_manifest = _json_bytes(
        _precision_manifest(
            accounting=precision_accounting,
            artifact=precision_artifact,
            compressed=precision_compressed,
            journal=precision_journal,
            raw=precision_raw,
            summary=precision_summary,
            verdict=precision_verdict,
        )
    )
    precision_publication = _precision_publication_binding(
        compressed=precision_compressed,
        journal=precision_journal,
        manifest=precision_manifest,
        summary=precision_summary,
    )

    artifact = exact_object(artifact_value, context="engineering artifact")
    analysis = analyze_artifact(artifact, archive_ratios=load_archive_ratios(_ARCHIVE))
    compressed = _compress(raw)
    journal = _attempts(_ENGINEERING).read_bytes()
    accounting = _collection_accounting(
        _attempts(_ENGINEERING),
        expected_rows=3008,
        published_rows=analysis.row_count,
        cost_scope="engineering_timing_call_only",
        covered_call_ids=(_ENGINEERING_CALL,),
        excluded_calls=_engineering_cost_exclusions(),
    )
    summary_publication = _summary_publication(
        artifact=artifact,
        compressed=compressed,
        raw=raw,
        journal=journal,
        journal_path=_attempts(_ENGINEERING),
        raw_path="benchmarks/data/hopper-h100-engineering.json.zst",
        journal_publication_path="benchmarks/data/hopper-h100-engineering.attempts.jsonl",
        manifest_path="benchmarks/hopper-h100-engineering-manifest-v2.json",
        hardware=_hardware_dict(artifact.get("hardware"), context="engineering hardware"),
        provenance=engineering_provenance,
        app_id=_ENGINEERING_APP,
        expected_call_ids=(_ENGINEERING_CALL,),
    )
    summary = _json_bytes(
        _summary(
            analysis,
            accounting=accounting,
            precision_summaries=precision_summaries,
            precision_verdict=precision_verdict,
            precision_publication=precision_publication,
            publication=summary_publication,
        )
    )
    manifest = _json_bytes(
        _manifest(
            accounting=accounting,
            analysis=analysis,
            artifact=artifact,
            compressed=compressed,
            raw=raw,
            journal=journal,
            precision_publication=precision_publication,
            summary=summary,
        )
    )

    _require_digest(
        _PRECISION_RAW,
        _sha256_bytes(precision_compressed),
        context="committed precision raw publication",
    )
    _require_digest(
        _PRECISION_JOURNAL,
        _sha256_bytes(precision_journal),
        context="committed precision journal publication",
    )
    _require_digest(
        _RAW,
        _sha256_bytes(compressed),
        context="committed engineering raw publication",
    )
    _require_digest(
        _JOURNAL,
        _sha256_bytes(journal),
        context="committed engineering journal publication",
    )
    write_bytes_atomic(_PRECISION_SUMMARY, precision_summary)
    write_bytes_atomic(_PRECISION_MANIFEST, precision_manifest)
    write_bytes_atomic(_SUMMARY, summary)
    write_bytes_atomic(_MANIFEST, manifest)
    print(
        "published post-hoc exploratory H100 precision evidence and Hopper engineering v2 "
        "evidence; immutable v1 retained and global STOP unchanged"
    )


def _validate_published_journal(journal: bytes) -> None:
    if _sha256_bytes(journal) != _INPUT_DIGESTS["engineering"]["attempt_journal"]:
        _fail("published engineering journal differs from the collected input journal")
    records = AttemptJournal.load(_JOURNAL).records
    if len(records) != 2 or records[0].status != "spawned" or records[1].status != "completed":
        _fail("published engineering journal does not contain one completed transition")
    if any(record.call_id != _ENGINEERING_CALL for record in records):
        _fail("published engineering journal has the wrong FunctionCall ID")


def _validate_published_precision_journal(journal: bytes) -> None:
    if _sha256_bytes(journal) != _INPUT_DIGESTS["precision"]["attempt_journal"]:
        _fail("published precision journal differs from the collected input journal")
    records = AttemptJournal.load(_PRECISION_JOURNAL).records
    if len(records) != 2 * len(_PRECISION_CALLS):
        _fail("published precision journal does not contain three completed transitions")
    for index, call_id in enumerate(_PRECISION_CALLS):
        spawned = records[2 * index]
        completed = records[2 * index + 1]
        if (
            spawned.status != "spawned"
            or completed.status != "completed"
            or spawned.call_id != call_id
            or completed.call_id != call_id
            or spawned.bank != index
            or completed.bank != index
        ):
            _fail("published precision journal has the wrong bank/call transitions")


def _validate_publication_bindings(
    manifest_path: Path,
    *,
    compressed: bytes,
    journal: bytes,
    raw: bytes,
    summary: bytes,
    context: str,
) -> None:
    manifest_value = _strict_object(manifest_path, context=f"{context} publication manifest")
    publication = exact_object(
        manifest_value.get("publication"), context=f"{context} published bindings"
    )
    raw_binding = exact_object(publication.get("raw"), context=f"{context} raw binding")
    journal_binding = exact_object(
        publication.get("attempt_journal"), context=f"{context} journal binding"
    )
    summary_binding = exact_object(publication.get("summary"), context=f"{context} summary binding")
    if (
        raw_binding.get("compressed_sha256") != _sha256_bytes(compressed)
        or raw_binding.get("decompressed_sha256") != _sha256_bytes(raw)
        or journal_binding.get("sha256") != _sha256_bytes(journal)
        or summary_binding.get("sha256") != _sha256_bytes(summary)
    ):
        _fail(f"{context} publication manifest has a raw/journal/summary digest mismatch")


def check(*, compare_head: bool = False) -> None:
    _validate_immutable_v1()
    required = (
        _RAW,
        _JOURNAL,
        _SUMMARY,
        _MANIFEST,
        _PRECISION_RAW,
        _PRECISION_JOURNAL,
        _PRECISION_SUMMARY,
        _PRECISION_MANIFEST,
        _ARCHIVE,
        _PRICING_SOURCE,
        _ANALYZER,
        _PRECISION_ANALYZER,
        _PUBLISHER,
    )
    for path in required:
        if not path.is_file():
            _fail(f"required committed evidence file is missing: {path.relative_to(_REPO)}")
    _require_digest(_ARCHIVE, _ARCHIVE_SHA256, context="context archive")
    _require_digest(_PRICING_SOURCE, _PRICING_SOURCE_SHA256, context="pricing source")

    precision_compressed = _PRECISION_RAW.read_bytes()
    precision_raw = _decompress(precision_compressed)
    if _sha256_bytes(precision_raw) != _INPUT_DIGESTS["precision"]["artifact"]:
        _fail("decompressed precision raw artifact digest differs from the collected input")
    if _compress(precision_raw) != precision_compressed:
        _fail("published precision raw artifact is not the deterministic zstd encoding")
    precision_artifact, precision_summaries, precision_verdict = _analyze_precision(precision_raw)
    precision_journal = _PRECISION_JOURNAL.read_bytes()
    _validate_published_precision_journal(precision_journal)
    precision_accounting = _collection_accounting(
        _PRECISION_JOURNAL,
        expected_rows=288,
        published_rows=288,
        cost_scope="precision_probe_calls_only",
        covered_call_ids=_PRECISION_CALLS,
    )
    precision_manifest_value = _strict_object(
        _PRECISION_MANIFEST, context="precision publication manifest"
    )
    precision_provenance = _committed_provenance(
        precision_manifest_value,
        expected={
            "head_commit": _PRECISION_HEAD_COMMIT,
            "source_sha256": _PRECISION_SOURCE_SHA256,
            "wheel_sha256": _PRECISION_WHEEL_SHA256,
        },
        context="precision",
    )
    precision_summary_publication = _summary_publication(
        artifact=precision_artifact,
        compressed=precision_compressed,
        raw=precision_raw,
        journal=precision_journal,
        journal_path=_PRECISION_JOURNAL,
        raw_path="benchmarks/data/h100-precision-probe.json.zst",
        journal_publication_path="benchmarks/data/h100-precision-probe.attempts.jsonl",
        manifest_path="benchmarks/h100-precision-probe-manifest.json",
        hardware=_precision_hardware(precision_artifact),
        provenance=precision_provenance,
        app_id=_PRECISION_APP,
        expected_call_ids=_PRECISION_CALLS,
        raw_schema=_PRECISION_RAW_SCHEMA,
    )
    expected_precision_summary = _json_bytes(
        _precision_summary(
            precision_artifact,
            precision_summaries,
            precision_verdict,
            precision_accounting,
            precision_summary_publication,
        )
    )
    actual_precision_summary = _PRECISION_SUMMARY.read_bytes()
    actual_precision_summary_value = exact_object(
        strict_json_loads(
            actual_precision_summary.decode("utf-8"),
            source="benchmarks/results/h100-precision-probe-summary.json",
        ),
        context="precision published summary",
    )
    _validate_summary_publication(
        actual_precision_summary_value.get("publication"),
        expected=precision_summary_publication,
        context="precision",
    )
    if actual_precision_summary != expected_precision_summary:
        _fail("published precision summary is not the byte-exact strict-analysis result")
    _validate_publication_bindings(
        _PRECISION_MANIFEST,
        compressed=precision_compressed,
        journal=precision_journal,
        raw=precision_raw,
        summary=actual_precision_summary,
        context="precision",
    )
    expected_precision_manifest = _json_bytes(
        _precision_manifest(
            accounting=precision_accounting,
            artifact=precision_artifact,
            compressed=precision_compressed,
            journal=precision_journal,
            raw=precision_raw,
            summary=expected_precision_summary,
            verdict=precision_verdict,
        )
    )
    actual_precision_manifest = _PRECISION_MANIFEST.read_bytes()
    if actual_precision_manifest != expected_precision_manifest:
        _fail("published precision manifest is not the byte-exact regenerated manifest")
    precision_publication = _precision_publication_binding(
        compressed=precision_compressed,
        journal=precision_journal,
        manifest=actual_precision_manifest,
        summary=actual_precision_summary,
    )

    compressed = _RAW.read_bytes()
    raw = _decompress(compressed)
    if _sha256_bytes(raw) != _INPUT_DIGESTS["engineering"]["artifact"]:
        _fail("decompressed engineering raw artifact digest differs from the collected input")
    if _compress(raw) != compressed:
        _fail("published engineering raw artifact is not the deterministic zstd encoding")
    artifact, analysis = _analyze(raw)
    journal = _JOURNAL.read_bytes()
    _validate_published_journal(journal)
    accounting = _collection_accounting(
        _JOURNAL,
        expected_rows=3008,
        published_rows=analysis.row_count,
        cost_scope="engineering_timing_call_only",
        covered_call_ids=(_ENGINEERING_CALL,),
        excluded_calls=_engineering_cost_exclusions(),
    )
    manifest_value = _strict_object(_MANIFEST, context="engineering publication manifest")
    engineering_provenance = _committed_provenance(
        manifest_value,
        expected={
            "head_commit": _HEAD_COMMIT,
            "source_sha256": _SOURCE_SHA256,
            "wheel_sha256": _WHEEL_SHA256,
        },
        context="engineering",
    )
    summary_publication = _summary_publication(
        artifact=artifact,
        compressed=compressed,
        raw=raw,
        journal=journal,
        journal_path=_JOURNAL,
        raw_path="benchmarks/data/hopper-h100-engineering.json.zst",
        journal_publication_path="benchmarks/data/hopper-h100-engineering.attempts.jsonl",
        manifest_path="benchmarks/hopper-h100-engineering-manifest-v2.json",
        hardware=_hardware_dict(artifact.get("hardware"), context="engineering hardware"),
        provenance=engineering_provenance,
        app_id=_ENGINEERING_APP,
        expected_call_ids=(_ENGINEERING_CALL,),
    )
    expected_summary = _json_bytes(
        _summary(
            analysis,
            accounting=accounting,
            precision_summaries=precision_summaries,
            precision_verdict=precision_verdict,
            precision_publication=precision_publication,
            publication=summary_publication,
        )
    )
    actual_summary = _SUMMARY.read_bytes()
    actual_summary_value = exact_object(
        strict_json_loads(
            actual_summary.decode("utf-8"),
            source="benchmarks/results/hopper-h100-engineering-summary-v2.json",
        ),
        context="engineering published summary",
    )
    _validate_summary_publication(
        actual_summary_value.get("publication"),
        expected=summary_publication,
        context="engineering",
    )
    if actual_summary != expected_summary:
        _fail("published engineering summary is not the byte-exact strict-analysis result")
    _validate_publication_bindings(
        _MANIFEST,
        compressed=compressed,
        journal=journal,
        raw=raw,
        summary=actual_summary,
        context="engineering",
    )
    expected_manifest = _json_bytes(
        _manifest(
            accounting=accounting,
            analysis=analysis,
            artifact=artifact,
            compressed=compressed,
            raw=raw,
            journal=journal,
            precision_publication=precision_publication,
            summary=expected_summary,
        )
    )
    if _MANIFEST.read_bytes() != expected_manifest:
        _fail("published engineering manifest is not the byte-exact regenerated manifest")
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
    print(
        "H100 precision and Hopper engineering v2 committed evidence: OK "
        "(immutable v1 retained; post-hoc exploratory; global STOP unchanged)"
    )


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
