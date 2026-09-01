from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from heliostune.errors import ArtifactError, SchemaError
from heliostune.remote_execution import RemoteIntent, verify_remote_receipt_payloads

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/publish_native_rmsnorm_h100.py"


def _load_publisher() -> ModuleType:
    name = "_test_publish_native_rmsnorm_h100"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PUBLISHER = _load_publisher()
_EXPECTED_ATTEMPTS = (
    (
        "transport-overflow-unresolved",
        "unresolved",
        (
            (
                "intent.json",
                1183,
                "9c6722aceb74422f55a958625d3ebb0a6be5a2de33a5a132d20ea2ab6356b263",
            ),
            (
                "journal.jsonl",
                1255,
                "5afd9b0f9f8a086fc8b979c534a50fe99d3a5e34cbaa0782a178e54bc2d4d010",
            ),
            (
                "plugin.json",
                638,
                "ce4a497113adf1ee82ed995fb4ba671a8a1664d756321499d91187056ca0d815",
            ),
            (
                "receipt.json",
                2502,
                "d8f3a0bade1dd39ad607bf4eaab4508c88af8386435ad1b947a72aac864aa8cc",
            ),
            (
                "suite.json",
                11136,
                "23f7397f2adee93cd9f7919aaf075c0f8b5e92cd6d4257ce4c54197d3c98035f",
            ),
            (
                "wheel.manifest.json",
                651,
                "117a7a0c52b77e552777dbc5ec0bae62c66bbaa9a7e07a76b3b853082a9e1f4f",
            ),
        ),
        (
            (
                "remote_attempts",
                "residual-rmsnorm-triton-v1-20260831T064037385754112.remote-attempts.jsonl",
                1255,
                "5afd9b0f9f8a086fc8b979c534a50fe99d3a5e34cbaa0782a178e54bc2d4d010",
                "journal.jsonl",
            ),
            (
                "remote_intent",
                "residual-rmsnorm-triton-v1-20260831T064037385754112.remote-intent.json",
                1183,
                "9c6722aceb74422f55a958625d3ebb0a6be5a2de33a5a132d20ea2ab6356b263",
                "intent.json",
            ),
        ),
    ),
    (
        "completed-inline-result",
        "completed",
        (
            (
                "intent.json",
                1183,
                "cc1de45e0d42b9bf896d5a1ae23c6a046617d5c35dc4e0f248a6beefcb4214b9",
            ),
            (
                "journal.jsonl",
                860,
                "60c158050fbf3296002468a8dc30638942edf680bd5b034d13625e6e137180ca",
            ),
            (
                "plugin.json",
                638,
                "ce4a497113adf1ee82ed995fb4ba671a8a1664d756321499d91187056ca0d815",
            ),
            (
                "receipt.json",
                2629,
                "9f0c705723ad81826fa2b4a7893fd03b2bf0b96926e434fc7094351fe7edca03",
            ),
            (
                "result.json",
                66756,
                "790f6d9335f6c7384c6651058ac775ed4dd4a553063f5742c61f7441072e9f9d",
            ),
            (
                "suite.json",
                11136,
                "23f7397f2adee93cd9f7919aaf075c0f8b5e92cd6d4257ce4c54197d3c98035f",
            ),
            (
                "wheel.manifest.json",
                651,
                "6f09348b025848a5b9e819620de5e3968173eef872f21d218442f1152193fbcf",
            ),
        ),
        (
            (
                "remote_attempts",
                "residual-rmsnorm-triton-v1-20260901T004420761029839.remote-attempts.jsonl",
                860,
                "60c158050fbf3296002468a8dc30638942edf680bd5b034d13625e6e137180ca",
                "journal.jsonl",
            ),
            (
                "remote_intent",
                "residual-rmsnorm-triton-v1-20260901T004420761029839.remote-intent.json",
                1183,
                "cc1de45e0d42b9bf896d5a1ae23c6a046617d5c35dc4e0f248a6beefcb4214b9",
                "intent.json",
            ),
        ),
    ),
)


def _raw() -> dict[str, Any]:
    value = PUBLISHER._strict_json_bytes(
        PUBLISHER._decompress(PUBLISHER._RAW_PATH.read_bytes()),
        context="test native RMSNorm raw",
    )
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_committed_publication_is_offline_byte_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_source_read(*_args: object, **_kwargs: object) -> None:
        pytest.fail("check must not read source receipt directories or sidecars")

    for source_reader in (
        "verify_remote_receipt",
        "_regular_inventory",
        "_sidecar_paths",
        "_generation_attempt",
    ):
        monkeypatch.setattr(PUBLISHER, source_reader, fail_source_read)
    PUBLISHER.check()


def test_raw_losslessly_embeds_both_exact_receipts_and_attempt_sidecars() -> None:
    raw = _raw()
    for attempt, expected in zip(raw["attempts"], _EXPECTED_ATTEMPTS, strict=True):
        expected_id, expected_status, expected_receipts, expected_sidecars = expected
        assert attempt["attempt_id"] == expected_id
        assert attempt["status"] == expected_status
        assert [
            (item["name"], item["bytes"], item["sha256"]) for item in attempt["receipt_files"]
        ] == list(expected_receipts)

        receipt_files = {
            item["name"]: PUBLISHER._decode_blob(
                item, expected_name=item["name"], context=f"test {expected_id} receipt"
            )
            for item in attempt["receipt_files"]
        }
        for sidecar_key, name, size, digest, receipt_name in expected_sidecars:
            sidecar = attempt["source_attempt"][sidecar_key]
            assert (sidecar["name"], sidecar["bytes"], sidecar["sha256"]) == (
                name,
                size,
                digest,
            )
            assert (
                PUBLISHER._decode_blob(
                    sidecar, expected_name=name, context=f"test {expected_id} {sidecar_key}"
                )
                == receipt_files[receipt_name]
            )

        intent_value = PUBLISHER._strict_json_bytes(
            receipt_files["intent.json"], context=f"test {expected_id} intent"
        )
        intent = RemoteIntent.from_dict(intent_value)
        assert attempt["logical_output_path"] == intent.output_path
        verified = verify_remote_receipt_payloads(
            receipt_files, logical_output_path=intent.output_path
        )
        assert verified.intent == intent
        assert verified.receipt.status == expected_status


def test_summary_exposes_exact_stage_gate_evidence_and_nonclaims() -> None:
    summary = PUBLISHER._strict_json_bytes(
        PUBLISHER._SUMMARY_PATH.read_bytes(), context="test summary"
    )
    assert isinstance(summary, dict)
    analysis = summary["analysis"]
    assert analysis["decision"] == "stop_below_threshold"
    assert summary["decision_display"] == "STOP_BELOW_THRESHOLD"
    assert analysis["winner_id"] == "rmsnorm-triton-w8"
    assert analysis["winner_median_ms"] == 0.050592001527547836
    assert analysis["best_baseline_median_ms"] == 0.045951999723911285
    assert analysis["speedup"] == 0.9082858621217027
    assert analysis["threshold"] == 1.1
    assert summary["actual_cost_usd"] is None
    assert summary["claims"] == []
    assert summary["expansion_authorized"] is False
    assert summary["fusion_claim"] is False
    assert summary["performance_claim"] is False
    assert summary["publication_eligible"] is False
    assert summary["provider_accounting"]["total_time_upper_bound_s"] is None
    assert len(summary["candidates"]) == 4
    assert len(summary["baselines"]) == 2
    for candidate in summary["candidates"]:
        assert candidate["resource_evidence"]["n_spills"] == 0
        assert candidate["one_kernel_evidence"]["one_kernel_gate_passed"] is True
        assert candidate["validation_evidence"]["validation_gate_passed"] is True
        assert candidate["timing_observation"]["status"] == "passed"
    assert summary["attempts"][0]["transport_error"] == (
        "RemoteError: SchemaError('remote result transport exceeds 6144-byte inline limit')"
    )
    report = PUBLISHER._REPORT_PATH.read_text(encoding="utf-8")
    assert "STOP_BELOW_THRESHOLD" in report
    assert all(candidate["arm_id"] in report for candidate in summary["candidates"])
    assert all(nonclaim in report for nonclaim in summary["nonclaims"])


def test_embedded_receipt_mutation_is_rejected() -> None:
    raw = copy.deepcopy(_raw())
    blob = raw["attempts"][1]["receipt_files"][0]
    blob["sha256"] = "0" * 64
    payload = PUBLISHER._json_bytes(raw)
    with pytest.raises((ValueError, ArtifactError, SchemaError)):
        PUBLISHER._validate_raw(payload)
