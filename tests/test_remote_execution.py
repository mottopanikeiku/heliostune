from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from heliostune.errors import ArtifactError, SchemaError
from heliostune.local_executor import CapabilityProbe, LocalExecutionResult
from heliostune.remote_execution import (
    RECEIPT_LIMITATIONS,
    RECEIPT_SCHEMA,
    SERVER_TIMEOUT_SECONDS,
    RemoteIntent,
    RemoteJournal,
    RemoteResultEnvelope,
    canonical_json_bytes,
    create_remote_records,
    decode_remote_request,
    encode_remote_request,
    protect_remote_output,
    remote_artifact_paths,
    sha256_bytes,
    validate_remote_result,
    verify_remote_receipt,
    write_remote_receipt,
)
from heliostune.schema import HardwareProfile
from heliostune.scope import verify_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/suites/gated-mlp-epilogue-v1.json"
PLUGIN = ROOT / "benchmarks/plugins/fusion-reference-plugin-v1.json"


def _intent(tmp_path: Path) -> tuple[RemoteIntent, bytes, bytes, bytes]:
    suite = verify_suite(SUITE)
    plugin = PLUGIN.read_bytes()
    manifest = b'{"supplemental":true}\n'
    return (
        RemoteIntent(
            suite_path=str(SUITE),
            output_path=str((tmp_path / "receipt").absolute()),
            suite_sha256=suite.sha256,
            plugin_path=str(PLUGIN),
            plugin_sha256=sha256_bytes(plugin),
            wheel_filename="heliostune-1.0.0-py3-none-any.whl",
            wheel_sha256="2" * 64,
            manifest_sha256=sha256_bytes(manifest),
            head_commit="4" * 40,
            source_sha256="5" * 64,
        ),
        suite.bytes,
        plugin,
        manifest,
    )


def _hardware(
    *,
    cuda_version: str | None = "12.8",
    torch_version: str = "2.8.0",
) -> HardwareProfile:
    return HardwareProfile(
        gpu="H100",
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability=(9, 0),
        multiprocessor_count=120,
        total_memory_gb=79.1,
        cuda_version=cuda_version,
        torch_version=torch_version,
        triton_version="3.4.0",
    )


def _aborted_result(
    intent: RemoteIntent,
    suite_bytes: bytes,
    *,
    capability: CapabilityProbe | None = None,
) -> LocalExecutionResult:
    hardware = _hardware()
    if capability is None:
        capability = CapabilityProbe(
            available=False,
            reasons=("inductor_unavailable",),
            torch_version=hardware.torch_version,
            cuda_version=hardware.cuda_version,
            rocm_version=None,
            device_index=0,
            device_name=hardware.device_name,
            compute_capability=hardware.compute_capability,
            native_bf16=True,
            inductor_available=False,
            allocation_succeeded=False,
            detail=None,
        )
    environment = {
        "schema": "heliostune.local-environment/1",
        "python": "3.11.14",
        "implementation": "CPython",
        "platform": "Linux",
        "torch_version": capability.torch_version,
        "cuda_version": capability.cuda_version,
        "rocm_version": capability.rocm_version,
        "device_index": capability.device_index,
        "device_name": capability.device_name,
        "compute_capability": (
            None if capability.compute_capability is None else list(capability.compute_capability)
        ),
        "precision_policy": {
            "float32_matmul_precision": "highest",
            "allow_tf32": False,
            "allow_bf16_reduced_precision_reduction": False,
            "allow_fp16_reduced_precision_reduction": False,
            "allow_fp16_accumulation": False,
        },
        "autocast_policy": {
            "device_type": "cuda",
            "enabled": False,
            "restore_ambient_state": True,
        },
        "backend_invoked": None,
        "fusion_claim": False,
    }
    expected_ids = [cell.id for cell in verify_suite(SUITE).suite.expected_cells]
    return LocalExecutionResult(
        verified_suite_path=intent.suite_path,
        verified_suite_sha256=intent.suite_sha256,
        verified_suite_bytes=suite_bytes,
        suite_id="gated-mlp-epilogue-reference",
        capability=capability,
        materialization=(),
        observations=(),
        attempts=(),
        environment=environment,
        compile_outcomes={},
        summary={
            "expected_cell_ids": expected_ids,
            "terminal_cell_ids": [],
            "passed": 0,
            "failed": 0,
            "blocked": 0,
            "all_cells_terminal": False,
            "outcome": "aborted",
            "fusion_claim": False,
            "capability_reasons": list(capability.reasons),
        },
        outcome="aborted",
    )


def _envelope(
    intent: RemoteIntent,
    result: LocalExecutionResult,
    request_digest: str,
    *,
    hardware: HardwareProfile | None = None,
) -> RemoteResultEnvelope:
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
        hardware=_hardware() if hardware is None else hardware,
        environment=result.environment,
        result=result.to_dict(),
    )


def test_request_is_canonical_and_binds_exact_suite_bytes(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    payload = encode_remote_request(intent, suite_bytes)
    decoded_intent, decoded_bytes, request_digest = decode_remote_request(payload)
    assert decoded_intent == intent
    assert decoded_bytes == suite_bytes
    assert len(request_digest) == 64
    forged = json.loads(payload)
    forged["suite_utf8"] += " "
    with pytest.raises(SchemaError, match="suite digest"):
        decode_remote_request(canonical_json_bytes(forged).decode())
    with pytest.raises(SchemaError, match="canonical"):
        decode_remote_request(" " + payload)


def test_pinned_records_are_exclusive_and_spawn_acknowledgement_is_terminal(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        assert records.intent_bytes() == intent.to_bytes()
        records.journal.append("spawn_acknowledgement_lost", detail="RuntimeError: transport")
        records.journal.append("unresolved", detail="RuntimeError: transport")
        rows = [json.loads(line) for line in records.journal.bytes().splitlines()]
        assert [row["state"] for row in rows] == [
            "intent",
            "spawn_acknowledgement_lost",
            "unresolved",
        ]
        with pytest.raises(ArtifactError, match="remote intent"):
            protect_remote_output(intent.output_path)
    finally:
        records.close()


def test_replaced_tombstone_path_cannot_hide_or_forge_retained_intent(
    tmp_path: Path,
) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        intent_path, _ = remote_artifact_paths(intent.output_path)
        intent_path.unlink()
        intent_path.write_bytes(canonical_json_bytes({"forged": True}))
        assert records.intent_bytes() == intent.to_bytes()
        with pytest.raises(ArtifactError, match="tombstone identity changed"):
            records.assert_parent_identity()
    finally:
        records.close()


def test_journal_rejects_illegal_transition_and_duplicate_path(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    journal = RemoteJournal.create(path, "a" * 64)
    with pytest.raises(SchemaError, match="illegal"):
        journal.append("completed", call_id="fc-1")
    journal.close()
    with pytest.raises(FileExistsError):
        RemoteJournal.create(path, "a" * 64)


def test_result_envelope_uses_strict_local_parser_and_rejects_contradiction(tmp_path: Path) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    envelope = _envelope(intent, _aborted_result(intent, suite_bytes), request_digest)
    parsed, result = validate_remote_result(
        envelope.to_json(),
        intent=intent,
        request_digest=request_digest,
        verified_suite_bytes=suite_bytes,
    )
    assert parsed == envelope
    assert result.outcome == "aborted"
    forged = envelope.to_dict()
    assert isinstance(forged["result"], dict)
    assert isinstance(forged["result"]["capability"], dict)
    forged["result"]["capability"]["inductor_available"] = True
    with pytest.raises(SchemaError, match="inconsistent probe evidence"):
        validate_remote_result(
            canonical_json_bytes(forged).decode(),
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_bytes,
        )


@pytest.mark.parametrize(
    ("capability", "hardware"),
    [
        (
            CapabilityProbe(
                False,
                ("torch_missing",),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                "ModuleNotFoundError: No module named 'torch'",
            ),
            _hardware(),
        ),
        (
            CapabilityProbe(
                False,
                ("torch_version_mismatch",),
                "2.7.1",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                None,
            ),
            _hardware(torch_version="2.7.1"),
        ),
        (
            CapabilityProbe(
                False,
                ("cuda_unavailable",),
                "2.8.0",
                "12.8",
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                None,
            ),
            _hardware(),
        ),
    ],
    ids=("torch-missing", "torch-version-mismatch", "cuda-unavailable"),
)
def test_valid_early_capability_aborts_preserve_null_stage_evidence(
    tmp_path: Path,
    capability: CapabilityProbe,
    hardware: HardwareProfile,
) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    envelope = _envelope(
        intent,
        _aborted_result(intent, suite_bytes, capability=capability),
        request_digest,
        hardware=hardware,
    )
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawned", call_id="fc-early-abort")
        records.journal.append("retrieval_started", call_id="fc-early-abort")
        records.journal.append("aborted", call_id="fc-early-abort")
        write_remote_receipt(
            records,
            status="aborted",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=envelope.to_json(),
            client_spawn_count=1,
        )
    finally:
        records.close()
    verified = verify_remote_receipt(intent.output_path)
    assert verified.receipt.status == "aborted"
    assert verified.result is not None and verified.result.capability == capability
    _, result = validate_remote_result(
        envelope.to_json(),
        intent=intent,
        request_digest=request_digest,
        verified_suite_bytes=suite_bytes,
    )
    assert result.outcome == "aborted"
    assert result.capability == capability


def test_early_capability_abort_rejects_hardware_and_environment_contradictions(
    tmp_path: Path,
) -> None:
    intent, suite_bytes, _, _ = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    capability = CapabilityProbe(
        False,
        ("torch_version_mismatch",),
        "2.7.1",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        None,
    )
    result = _aborted_result(intent, suite_bytes, capability=capability)
    mismatched = _envelope(intent, result, request_digest)
    with pytest.raises(SchemaError, match="capability field torch_version"):
        validate_remote_result(
            mismatched.to_json(),
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_bytes,
        )

    fabricated = _envelope(
        intent,
        _aborted_result(
            intent,
            suite_bytes,
            capability=CapabilityProbe(
                False,
                ("torch_missing",),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                "ModuleNotFoundError: No module named 'torch'",
            ),
        ),
        request_digest,
    ).to_dict()
    assert isinstance(fabricated["environment"], dict)
    assert isinstance(fabricated["result"], dict)
    assert isinstance(fabricated["result"]["environment"], dict)
    fabricated["environment"]["device_name"] = _hardware().device_name
    fabricated["result"]["environment"]["device_name"] = _hardware().device_name
    with pytest.raises(SchemaError, match="environment does not match its capability"):
        validate_remote_result(
            canonical_json_bytes(fabricated).decode(),
            intent=intent,
            request_digest=request_digest,
            verified_suite_bytes=suite_bytes,
        )


def test_aborted_receipt_is_root_last_strict_and_tamper_evident(tmp_path: Path) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    result_payload = _envelope(
        intent, _aborted_result(intent, suite_bytes), request_digest
    ).to_json()
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawned", call_id="fc-one")
        records.journal.append("retrieval_started", call_id="fc-one")
        records.journal.append("aborted", call_id="fc-one")
        published = write_remote_receipt(
            records,
            status="aborted",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=result_payload,
            client_spawn_count=1,
        )
    finally:
        records.close()
    assert published.root_path == Path(intent.output_path) / "receipt.json"
    verified = verify_remote_receipt(intent.output_path)
    assert verified.receipt.schema == RECEIPT_SCHEMA
    assert verified.receipt.status == "aborted"
    assert verified.result is not None and verified.result.outcome == "aborted"
    root = json.loads((Path(intent.output_path) / "receipt.json").read_text())
    assert root["provider_attempts_observable"] is False
    assert root["provider_physical_attempts"] is None
    assert root["client_spawn_count"] == 1
    assert root["per_execution_timeout_s"] == SERVER_TIMEOUT_SECONDS
    assert root["total_gpu_seconds_upper_bound"] is None
    assert root["actual_cost_usd"] is None
    assert root["publication_eligible"] is False
    assert root["attestation"] == "none"
    assert tuple(root["limitations"]) == RECEIPT_LIMITATIONS
    assert "heliostune.bundle/1" not in (Path(intent.output_path) / "receipt.json").read_text()

    result_path = Path(intent.output_path) / root["result"]["path"]
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(ArtifactError, match="digest/size"):
        verify_remote_receipt(intent.output_path)


def test_receipt_rejects_semantic_status_and_path_tampering(tmp_path: Path) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    payload = _envelope(intent, _aborted_result(intent, suite_bytes), request_digest).to_json()
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawned", call_id="fc-one")
        records.journal.append("retrieval_started", call_id="fc-one")
        records.journal.append("aborted", call_id="fc-one")
        write_remote_receipt(
            records,
            status="aborted",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=payload,
            client_spawn_count=1,
        )
    finally:
        records.close()
    root_path = Path(intent.output_path) / "receipt.json"
    root = json.loads(root_path.read_text())
    root["status"] = "failed"
    root_path.write_bytes(canonical_json_bytes(root))
    with pytest.raises(SchemaError, match="status differs"):
        verify_remote_receipt(intent.output_path)
    root["status"] = "aborted"
    root["bindings"]["suite"]["path"] = "../suite.json"
    root_path.write_bytes(canonical_json_bytes(root))
    with pytest.raises(SchemaError, match="plain filename"):
        verify_remote_receipt(intent.output_path)


def test_unresolved_receipt_has_no_result_or_false_cancellation_claim(tmp_path: Path) -> None:
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(tmp_path)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    records = create_remote_records(intent.output_path, intent, request_digest)
    try:
        records.journal.append("spawn_acknowledgement_lost", detail="transport lost")
        records.journal.append("unresolved", detail="transport lost")
        write_remote_receipt(
            records,
            status="unresolved",
            request_digest=request_digest,
            suite_bytes=suite_bytes,
            plugin_bytes=plugin_bytes,
            manifest_bytes=manifest_bytes,
            result_payload=None,
            client_spawn_count=1,
        )
    finally:
        records.close()
    verified = verify_remote_receipt(intent.output_path)
    assert verified.receipt.result is None
    assert [record.state for record in verified.journal] == [
        "intent",
        "spawn_acknowledgement_lost",
        "unresolved",
    ]


def test_parent_substitution_cannot_publish_into_forged_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    intent, suite_bytes, plugin_bytes, manifest_bytes = _intent(parent)
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    records = create_remote_records(intent.output_path, intent, request_digest)
    moved = tmp_path / "moved-parent"
    os.rename(parent, moved)
    parent.mkdir()
    try:
        with pytest.raises(ArtifactError, match="parent identity changed"):
            records.assert_parent_identity()
        assert not (parent / "receipt").exists()
        assert (moved / "receipt.remote-intent.json").read_bytes() == intent.to_bytes()
        records.journal.append("spawn_acknowledgement_lost", detail="parent changed")
        records.journal.append("unresolved", detail="parent changed")
        with pytest.raises(ArtifactError, match="parent identity changed"):
            write_remote_receipt(
                records,
                status="unresolved",
                request_digest=request_digest,
                suite_bytes=suite_bytes,
                plugin_bytes=plugin_bytes,
                manifest_bytes=manifest_bytes,
                result_payload=None,
                client_spawn_count=1,
            )
    finally:
        records.close()
