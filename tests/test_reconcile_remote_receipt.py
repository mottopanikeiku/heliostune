from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from test_remote_execution import _aborted_result, _envelope, _failed_compile_result, _intent

import heliostune.local_executor as local
from heliostune.errors import ArtifactError, SchemaError
from heliostune.remote_execution import (
    RemoteIntent,
    RemoteJournalRecord,
    canonical_json_bytes,
    canonical_json_line_bytes,
    decode_remote_request,
    encode_remote_request,
    remote_artifact_paths,
    sha256_bytes,
    verify_remote_receipt,
)
from heliostune.scope import verify_suite

_REPOSITORY = Path(__file__).resolve().parents[1]
_SCRIPT = _REPOSITORY / "scripts/reconcile_remote_receipt.py"


def _load_script() -> ModuleType:
    name = "_test_reconcile_remote_receipt"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _completed_transport(intent: RemoteIntent, suite_bytes: bytes, request_digest: str) -> str:
    result = _failed_compile_result(intent, suite_bytes, "representative compile failure")
    base = _envelope(intent, result, request_digest)
    suite = verify_suite(intent.suite_path).suite
    cells = suite.expected_cells
    cases = {case.id: case for case in suite.cases}
    observations: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        correctness_key = local._correctness_gate_key(
            intent.suite_sha256, cases[cell.case_id], cell
        )
        samples = [float(sample) / 1000 for sample in range(1, 51)]
        observations.append(
            {
                "cell_id": cell.id,
                "case_id": cell.case_id,
                "arm_id": cell.arm_id,
                "stage": cell.stage,
                "status": "passed",
                "correctness": (
                    {
                        "status": "passed",
                        "correctness_key": correctness_key,
                        "failure_kind": None,
                        "message": None,
                        "output": {
                            "shape": [8, 11008],
                            "device": "cuda:0",
                            "dtype": "torch.bfloat16",
                            "layout": "torch.strided",
                            "contiguous": True,
                        },
                        "input_storage_unchanged": True,
                        "output_disjoint": True,
                        "finite": True,
                        "close": True,
                        "max_abs_error": 0.0,
                    }
                    if cell.stage == "correctness"
                    else None
                ),
                "timing": (
                    {
                        "status": "passed",
                        "correctness_key": correctness_key,
                        "failure_kind": None,
                        "message": None,
                        "warmups": 10,
                        "repetitions": 50,
                        "samples_ms": samples,
                        "median_ms": 0.0255,
                    }
                    if cell.stage == "timing"
                    else None
                ),
            }
        )
        attempts.extend(
            (
                {
                    "attempt_id": 2 * index + 1,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": "running",
                    "from_state": "pending",
                    "to_state": "running",
                    "reason": None,
                },
                {
                    "attempt_id": 2 * index + 2,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": "success",
                    "from_state": "running",
                    "to_state": "passed",
                    "reason": None,
                },
            )
        )
    completed = result.to_dict()
    completed["observations"] = observations
    completed["attempts"] = attempts
    completed_environment = dict(result.environment)
    completed_environment["backend_invoked"] = True
    specs = {item.id: item for item in suite.tensors if item.role != "output"}
    completed["materialization"] = [
        local.TensorMaterialization(
            suite_sha256=intent.suite_sha256,
            case_id=cell.case_id,
            arm_id=cell.arm_id,
            input_seed=cases[cell.case_id].input_seed,
            tensor_order=tuple(
                draw.tensor_id for draw in local._resolve_draw_schedule(suite, cases[cell.case_id])
            ),
            tensors=tuple(
                {
                    "tensor_id": draw.tensor_id,
                    "role": draw.role,
                    "shape": list(draw.shape),
                    "draw": "normal_0_1_fp32_cpu",
                    "normal_scale": draw.normal_scale,
                    "normal_offset": draw.normal_offset,
                    "cpu_dtype": "float32",
                    "storage_dtype": "bfloat16",
                    "device": "cuda:0",
                    "contiguous": True,
                    "alignment_bytes": specs[draw.tensor_id].alignment,
                    "alignment_satisfied": True,
                    "storage_sha256": "0" * 64,
                }
                for draw in local._resolve_draw_schedule(suite, cases[cell.case_id])
            ),
        ).to_dict()
        for cell in cells
        if cell.stage == "correctness"
    ]
    completed["environment"] = completed_environment
    completed["compile_outcomes"] = {
        "mlp-candidate": {
            **dict(result.compile_outcomes["mlp-candidate"]),
            "status": "compiled_and_first_call_completed",
            "error": None,
            "first_call_ns": 2,
            "backend_invoked": True,
            "callable_distinct": True,
        }
    }
    completed["summary"] = {
        **dict(result.summary),
        "expected_cell_ids": [cell.id for cell in cells],
        "terminal_cell_ids": [cell.id for cell in cells],
        "passed": len(cells),
        "failed": 0,
        "blocked": 0,
        "all_cells_terminal": True,
        "outcome": "completed",
        "fusion_claim": False,
    }
    completed["outcome"] = "completed"
    return replace(
        base,
        environment=completed_environment,
        result=completed,
    ).to_transport_json()


def _write_old_records(
    output: Path,
    intent: RemoteIntent,
    request_digest: str,
    states: tuple[str, ...] = ("intent", "spawned", "retrieval_started", "completed"),
    *,
    call_id: str = "fc-existing",
) -> Path:
    intent_path, journal_path = remote_artifact_paths(output)
    intent_path.parent.mkdir(parents=True, exist_ok=True)
    intent_path.write_bytes(intent.to_bytes())
    rows = []
    for sequence, state in enumerate(states):
        rows.append(
            RemoteJournalRecord(
                request_digest=request_digest,
                sequence=sequence,
                state=state,  # type: ignore[arg-type]
                call_id=None if state == "intent" else call_id,
            )
        )
    journal_path.write_bytes(b"".join(canonical_json_line_bytes(row.to_dict()) for row in rows))
    return journal_path


def _replace_same_length_same_inode(path: Path, old: bytes, new: bytes) -> tuple[bytes, bytes]:
    assert len(old) == len(new)
    identity = path.stat().st_ino
    original = path.read_bytes()
    assert old in original
    changed = original.replace(old, new)
    assert len(changed) == len(original)
    with path.open("r+b") as stream:
        stream.write(changed)
        stream.flush()
        os.fsync(stream.fileno())
    assert path.stat().st_ino == identity
    return original, changed


def _case(tmp_path: Path) -> SimpleNamespace:
    base_intent, suite_bytes, plugin_bytes, _ = _intent(tmp_path)
    repository = tmp_path / "repository"
    suite_path = Path("benchmarks/suites/gated-mlp-epilogue-v1.json")
    plugin_path = Path("benchmarks/plugins/fusion-reference-plugin-v1.json")
    (repository / suite_path).parent.mkdir(parents=True)
    (repository / plugin_path).parent.mkdir(parents=True)
    (repository / suite_path).write_bytes(suite_bytes)
    (repository / plugin_path).write_bytes(plugin_bytes)
    manifest = canonical_json_bytes(
        {
            "schema_version": 1,
            "head_commit": base_intent.head_commit,
            "source_sha256": base_intent.source_sha256,
            "wheel_filename": base_intent.wheel_filename,
            "wheel_sha256": base_intent.wheel_sha256,
        }
    )
    intent = replace(
        base_intent,
        suite_path=str(suite_path),
        plugin_path=str(plugin_path),
        manifest_sha256=sha256_bytes(manifest),
    )
    _, _, request_digest = decode_remote_request(encode_remote_request(intent, suite_bytes))
    manifest_path = (
        repository / "artifacts" / "modal-wheel" / f"{intent.wheel_filename}.manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(manifest)
    journal_path = _write_old_records(Path(intent.output_path), intent, request_digest)
    return SimpleNamespace(
        output=Path(intent.output_path),
        intent=intent,
        suite_bytes=suite_bytes,
        plugin_bytes=plugin_bytes,
        manifest_bytes=manifest,
        manifest_path=manifest_path,
        repository=repository,
        request_digest=request_digest,
        journal_path=journal_path,
        transport=_completed_transport(intent, suite_bytes, request_digest),
    )


def _modal(payload: str | BaseException, calls: list[tuple[str, object]]) -> Any:
    class Handle:
        def get(self, *, timeout: int) -> str:
            calls.append(("get", timeout))
            if isinstance(payload, BaseException):
                raise payload
            return payload

    class FunctionCall:
        @classmethod
        def from_id(cls, call_id: str) -> Handle:
            calls.append(("from_id", call_id))
            return Handle()

    return SimpleNamespace(FunctionCall=FunctionCall)


def test_reconcile_completed_call_publishes_receipt_without_spawn(tmp_path: Path) -> None:
    script = _load_script()
    case = _case(tmp_path)
    calls: list[tuple[str, object]] = []
    journal_before = case.journal_path.read_bytes()

    published = script.reconcile_remote_receipt(
        case.output,
        timeout=19,
        repository=case.repository,
        modal_module=_modal(case.transport, calls),
    )

    assert calls == [("from_id", "fc-existing"), ("get", 19)]
    assert case.journal_path.read_bytes() == journal_before
    assert published.receipt.status == "completed"
    verified = verify_remote_receipt(case.output)
    assert verified.result is not None
    assert verified.result.outcome == "completed"


def test_reconcile_uses_opened_journal_snapshot_for_selection_and_rejects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    case = _case(tmp_path)
    calls: list[tuple[str, object]] = []
    original_open = script.open_remote_records

    def open_then_mutate(output: str | Path) -> Any:
        records = original_open(output)
        _replace_same_length_same_inode(case.journal_path, b"fc-existing", b"fc-forged00")
        return records

    monkeypatch.setattr(script, "open_remote_records", open_then_mutate)
    with pytest.raises(ArtifactError, match="journal changed before receipt publication"):
        script.reconcile_remote_receipt(
            case.output,
            repository=case.repository,
            modal_module=_modal(case.transport, calls),
        )

    assert calls == [("from_id", "fc-existing"), ("get", 3660)]
    assert not case.output.exists()


@pytest.mark.parametrize("artifact", ["journal", "intent"])
def test_reconcile_rejects_same_inode_source_mutation_after_get_before_writer(
    tmp_path: Path, artifact: str
) -> None:
    script = _load_script()
    case = _case(tmp_path)
    calls: list[tuple[str, object]] = []
    intent_path, _ = remote_artifact_paths(case.output)

    class Handle:
        def get(self, *, timeout: int) -> str:
            calls.append(("get", timeout))
            if artifact == "journal":
                _replace_same_length_same_inode(case.journal_path, b"fc-existing", b"fc-forged00")
            else:
                _replace_same_length_same_inode(intent_path, b'"gpu": "H100"', b'"gpu": "A100"')
            return cast(str, case.transport)

    class FunctionCall:
        @classmethod
        def from_id(cls, call_id: str) -> Handle:
            calls.append(("from_id", call_id))
            return Handle()

    with pytest.raises(
        ArtifactError, match=rf"remote {artifact} changed before receipt publication"
    ):
        script.reconcile_remote_receipt(
            case.output,
            repository=case.repository,
            modal_module=SimpleNamespace(FunctionCall=FunctionCall),
        )

    assert calls == [("from_id", "fc-existing"), ("get", 3660)]
    assert not case.output.exists()


@pytest.mark.parametrize("artifact", ["journal", "intent"])
def test_reconcile_publishes_authorized_snapshot_when_source_mutates_after_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    script = _load_script()
    case = _case(tmp_path)
    calls: list[tuple[str, object]] = []
    intent_path, _ = remote_artifact_paths(case.output)
    source = case.journal_path if artifact == "journal" else intent_path
    authorized = source.read_bytes()
    original_mkdir = os.mkdir
    mutated: list[bytes] = []

    def mkdir_after_boundary(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        if not mutated and isinstance(path, str) and path.startswith(".heliostune-remote-receipt-"):
            if artifact == "journal":
                _, changed = _replace_same_length_same_inode(source, b"fc-existing", b"fc-forged00")
            else:
                _, changed = _replace_same_length_same_inode(
                    source, b'"gpu": "H100"', b'"gpu": "A100"'
                )
            mutated.append(changed)
        original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr("heliostune.remote_execution.os.mkdir", mkdir_after_boundary)
    published = script.reconcile_remote_receipt(
        case.output,
        repository=case.repository,
        modal_module=_modal(case.transport, calls),
    )

    artifact_name = "journal.jsonl" if artifact == "journal" else "intent.json"
    assert mutated and source.read_bytes() == mutated[0]
    assert (case.output / artifact_name).read_bytes() == authorized
    if artifact == "journal":
        assert b"fc-forged00" not in (case.output / artifact_name).read_bytes()
    assert calls == [("from_id", "fc-existing"), ("get", 3660)]
    assert published.receipt.status == "completed"
    assert verify_remote_receipt(case.output).receipt.status == "completed"


def test_reconcile_refuses_existing_receipt_before_modal_lookup(tmp_path: Path) -> None:
    script = _load_script()
    case = _case(tmp_path)
    script.reconcile_remote_receipt(
        case.output,
        repository=case.repository,
        modal_module=_modal(case.transport, []),
    )
    calls: list[tuple[str, object]] = []
    with pytest.raises(ArtifactError, match="already exists"):
        script.reconcile_remote_receipt(
            case.output,
            repository=case.repository,
            modal_module=_modal(case.transport, calls),
        )
    assert calls == []


@pytest.mark.parametrize(
    "states",
    [
        ("intent", "spawned"),
        ("intent", "spawned", "retrieval_started"),
        ("intent", "spawned", "retrieval_started", "failed"),
        ("intent", "spawned", "retrieval_started", "aborted"),
        ("intent", "spawned", "retrieval_started", "cancellation_requested"),
        (
            "intent",
            "spawned",
            "retrieval_started",
            "cancellation_requested",
            "unresolved",
        ),
    ],
)
def test_reconcile_refuses_every_noncompleted_terminal_or_intermediate_state(
    tmp_path: Path, states: tuple[str, ...]
) -> None:
    script = _load_script()
    case = _case(tmp_path)
    case.journal_path.unlink()
    _write_old_records(case.output, case.intent, case.request_digest, states)
    calls: list[tuple[str, object]] = []

    with pytest.raises(SchemaError, match="terminal state exactly 'completed'"):
        script.reconcile_remote_receipt(
            case.output,
            repository=case.repository,
            modal_module=_modal(case.transport, calls),
        )
    assert calls == []
    assert not case.output.exists()


def test_reconcile_rejects_changed_call_id_in_strict_journal(tmp_path: Path) -> None:
    script = _load_script()
    case = _case(tmp_path)
    rows = [json.loads(line) for line in case.journal_path.read_bytes().splitlines()]
    rows[2]["call_id"] = "fc-other"
    case.journal_path.write_bytes(b"".join(canonical_json_line_bytes(row) for row in rows))

    with pytest.raises(SchemaError, match="call ID changed"):
        script.reconcile_remote_receipt(
            case.output,
            repository=case.repository,
            modal_module=_modal(case.transport, []),
        )
    assert not case.output.exists()


def test_reconcile_rejects_journal_request_digest_not_bound_to_suite(tmp_path: Path) -> None:
    script = _load_script()
    case = _case(tmp_path)
    rows = [json.loads(line) for line in case.journal_path.read_bytes().splitlines()]
    for row in rows:
        row["request_digest"] = "f" * 64
    case.journal_path.write_bytes(b"".join(canonical_json_line_bytes(row) for row in rows))
    calls: list[tuple[str, object]] = []

    with pytest.raises(SchemaError, match="request digest differs"):
        script.reconcile_remote_receipt(
            case.output,
            repository=case.repository,
            modal_module=_modal(case.transport, calls),
        )
    assert calls == []
    assert not case.output.exists()


@pytest.mark.parametrize("artifact", ["suite", "plugin", "manifest"])
def test_reconcile_rejects_changed_retained_digest_before_lookup(
    tmp_path: Path, artifact: str
) -> None:
    script = _load_script()
    case = _case(tmp_path)
    path = {
        "suite": case.repository / case.intent.suite_path,
        "plugin": case.repository / case.intent.plugin_path,
        "manifest": case.manifest_path,
    }[artifact]
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    calls: list[tuple[str, object]] = []
    try:
        with pytest.raises(SchemaError, match="digest differs"):
            script.reconcile_remote_receipt(
                case.output,
                repository=case.repository,
                modal_module=_modal(case.transport, calls),
            )
    finally:
        path.write_bytes(original)
    assert calls == []
    assert not case.output.exists()


def test_reconcile_rejects_result_not_matching_completed_journal(tmp_path: Path) -> None:
    script = _load_script()
    case = _case(tmp_path)
    aborted = _envelope(
        case.intent,
        _aborted_result(case.intent, case.suite_bytes),
        case.request_digest,
    ).to_transport_json()

    with pytest.raises(SchemaError, match="outcome is not completed"):
        script.reconcile_remote_receipt(
            case.output,
            repository=case.repository,
            modal_module=_modal(aborted, []),
        )
    assert not case.output.exists()


def test_reconcile_get_failure_leaves_journal_and_receipt_unchanged(tmp_path: Path) -> None:
    script = _load_script()
    case = _case(tmp_path)
    journal_before = case.journal_path.read_bytes()
    calls: list[tuple[str, object]] = []

    with pytest.raises(RuntimeError, match="retrieval failed"):
        script.reconcile_remote_receipt(
            case.output,
            repository=case.repository,
            modal_module=_modal(RuntimeError("retrieval failed"), calls),
        )
    assert calls == [("from_id", "fc-existing"), ("get", 3660)]
    assert case.journal_path.read_bytes() == journal_before
    assert not case.output.exists()


def test_reconcile_blocks_parent_substitution_during_retrieval(tmp_path: Path) -> None:
    script = _load_script()
    parent = tmp_path / "parent"
    parent.mkdir()
    case = _case(parent)
    moved = tmp_path / "moved-parent"

    class Handle:
        def get(self, *, timeout: int) -> str:
            assert timeout == 3660
            os.rename(parent, moved)
            parent.mkdir()
            return cast(str, case.transport)

    class FunctionCall:
        @classmethod
        def from_id(cls, call_id: str) -> Handle:
            assert call_id == "fc-existing"
            return Handle()

    with pytest.raises(ArtifactError, match="parent identity changed"):
        script.reconcile_remote_receipt(
            case.output,
            repository=case.repository,
            modal_module=SimpleNamespace(FunctionCall=FunctionCall),
        )
    assert not case.output.exists()
    assert not (moved / case.output.name).exists()
