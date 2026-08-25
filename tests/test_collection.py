from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from heliostune.artifacts import (
    read_json,
    read_measurements,
    write_json_atomic,
    write_text_atomic,
)
from heliostune.collection import (
    AttemptJournal,
    CallPlanItem,
    CollectionBinding,
    CollectionRequest,
    build_call_plan,
    commit_chunks,
    execute_call_plan,
    manifest_path,
    preflight_collection,
)
from heliostune.configs import KernelConfig, Workload
from heliostune.errors import ArtifactError, ProtocolError, SchemaError
from heliostune.protocol import v3_seed
from heliostune.schema import HardwareProfile, Measurement

_WORKLOADS = (
    Workload(1, 32, 32, "alpha", "attention", "decode"),
    Workload(2, 32, 32, "beta", "feedforward", "decode"),
)
_CONFIGS = (
    KernelConfig(16, 32, 32, 4, 3),
    KernelConfig(32, 32, 32, 4, 3),
)
_BINDING = CollectionBinding("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _request(
    *,
    gpus: tuple[str, ...] = ("A", "B"),
    banks: tuple[int, ...] = (0, 1),
) -> CollectionRequest:
    return CollectionRequest(
        gpus=gpus,
        banks=banks,
        workload_keys=tuple(workload.key for workload in _WORKLOADS),
        config_keys=tuple(config.key for config in _CONFIGS),
        warmup_ms=25.0,
        repetition_ms=100.0,
    )


def _rows(item: CallPlanItem, *, failed: bool = False) -> list[dict[str, object]]:
    hardware = HardwareProfile(
        item.gpu,
        f"Synthetic {item.gpu}",
        (8, 0),
        64,
        24.0,
    )
    rows: list[dict[str, object]] = []
    for workload_index, workload in enumerate(_WORKLOADS):
        torch_latency = 3.0 + workload_index + item.bank * 0.1
        for config_index, config in enumerate(_CONFIGS):
            is_failed = failed and workload_index == 0 and config_index == 0
            row = Measurement(
                hardware=hardware,
                workload=workload,
                config=config,
                bank=item.bank,
                latency_ms=None if is_failed else 1.0 + config_index + workload_index * 0.1,
                torch_latency_ms=torch_latency,
                correct=not is_failed,
                max_abs_error=None if is_failed else 0.0,
                error="compile failed" if is_failed else None,
                compile_ms=None if is_failed else 1.0,
                torch_benchmark_wall_ms=101.0 + workload_index,
                failure_stage="compile" if is_failed else None,
            )
            rows.append(row.to_dict())
    return rows


class SimulatedCrash(BaseException):
    pass


class FakeCall:
    def __init__(
        self,
        object_id: str,
        result: object,
        events: list[tuple[str, str]],
        *,
        crash: bool = False,
        failure: Exception | None = None,
    ) -> None:
        self.object_id = object_id
        self._result = result
        self._events = events
        self._crash = crash
        self._failure = failure

    def get(self) -> object:
        self._events.append(("get", self.object_id))
        if self._crash:
            raise SimulatedCrash
        if self._failure is not None:
            raise self._failure
        return self._result


def _clock() -> Any:
    value = datetime(2026, 8, 24, tzinfo=UTC)

    def now() -> datetime:
        nonlocal value
        current = value
        value += timedelta(seconds=1)
        return current

    return now


def test_build_call_plan_is_canonical_and_matches_collector_shuffle() -> None:
    request = _request(gpus=("B", "A"), banks=(2, 0))

    plan = build_call_plan(request)

    assert [(item.gpu, item.bank) for item in plan] == [
        ("A", 0),
        ("A", 2),
        ("B", 0),
        ("B", 2),
    ]
    assert all(item.seed == item.bank for item in plan)
    assert plan[0].workload_order == plan[2].workload_order
    assert plan[0].config_orders == plan[2].config_orders


def test_v3_call_plan_uses_protocol_seed_preimages() -> None:
    request = CollectionRequest(
        gpus=("L4", "A10"),
        banks=(0,),
        workload_keys=tuple(workload.key for workload in _WORKLOADS),
        config_keys=tuple(config.key for config in _CONFIGS),
        warmup_ms=25.0,
        repetition_ms=100.0,
        seed_protocol="parhelion-v3",
    )

    plan = build_call_plan(request)

    for item in plan:
        expected_seed = v3_seed(
            purpose="collector-workload-order",
            gpu=item.gpu,
            bank=item.bank,
        )
        assert item.seed == expected_seed
        expected_workloads = list(request.workload_keys)
        random.Random(expected_seed).shuffle(expected_workloads)
        assert item.workload_order == tuple(expected_workloads)
        for workload_key, config_order in item.config_orders:
            expected_configs = list(request.config_keys)
            random.Random(
                v3_seed(
                    purpose="collector-config-order",
                    gpu=item.gpu,
                    bank=item.bank,
                    workload_key=workload_key,
                )
            ).shuffle(expected_configs)
            assert config_order == tuple(expected_configs)
    assert plan[0].seed != plan[1].seed


@pytest.mark.parametrize("banks", [(), (0, 0), (True,), (-1,)])
def test_collection_request_rejects_invalid_banks(banks: tuple[object, ...]) -> None:
    with pytest.raises(SchemaError):
        CollectionRequest(
            gpus=("A",),
            banks=banks,
            workload_keys=("workload",),
            config_keys=("config",),
            warmup_ms=25.0,
            repetition_ms=100.0,
        )


def test_crash_after_all_spawns_resumes_by_id_without_new_calls_and_commits(
    tmp_path: Path,
) -> None:
    request = _request()
    output = tmp_path / "measurements.jsonl.zst"
    journal = preflight_collection(output)
    events: list[tuple[str, str]] = []
    remote_results: dict[str, object] = {}
    spawn_count = 0

    def spawn(item: CallPlanItem) -> FakeCall:
        nonlocal spawn_count
        spawn_count += 1
        call_id = f"fc-{item.gpu}-{item.bank}"
        events.append(("spawn", call_id))
        remote_results[call_id] = _rows(item)
        return FakeCall(
            call_id,
            remote_results[call_id],
            events,
            crash=spawn_count == 1,
        )

    with pytest.raises(SimulatedCrash):
        execute_call_plan(
            request,
            _BINDING,
            journal,
            spawn=spawn,
            now=_clock(),
        )

    assert spawn_count == 4
    assert [event[0] for event in events[:4]] == ["spawn"] * 4
    assert [record.status for record in journal.records] == ["spawned"] * 4
    loaded = AttemptJournal.load(journal.path)
    restored: list[str] = []

    def no_spawn(_item: CallPlanItem) -> FakeCall:
        raise AssertionError("resume must spawn zero replacement calls")

    def restore(call_id: str) -> FakeCall:
        restored.append(call_id)
        return FakeCall(call_id, remote_results[call_id], events)

    chunks = execute_call_plan(
        request,
        _BINDING,
        loaded,
        spawn=no_spawn,
        restore=restore,
        now=_clock(),
    )

    assert restored == [f"fc-{item.gpu}-{item.bank}" for item in build_call_plan(request)]
    assert [record.status for record in loaded.records] == ["spawned"] * 4 + ["completed"] * 4
    manifest = commit_chunks(
        output,
        request,
        _BINDING,
        loaded,
        chunks,
        facts={"python": "3.11", "nvidia_smi": None},
    )

    assert output.is_file()
    assert manifest_path(output).is_file()
    assert len(read_measurements(output)) == 16
    sidecar = read_json(manifest_path(output))
    assert sidecar["data"]["sha256"] == manifest.data_sha256
    assert sidecar["attempt_journal"]["sha256"] == manifest.attempt_journal_sha256
    assert {record["call_id"] for record in sidecar["calls"]} == set(restored)
    assert [(item["gpu"], item["bank"]) for item in sidecar["call_plan"]] == [
        (item.gpu, item.bank) for item in build_call_plan(request)
    ]
    assert [profile["gpu"] for profile in sidecar["hardware"]] == ["A", "B"]


def test_retrieval_failure_is_journaled_and_never_retryable(tmp_path: Path) -> None:
    request = _request(gpus=("A",), banks=(0,))
    output = tmp_path / "failed.jsonl.zst"
    journal = preflight_collection(output)

    def spawn(item: CallPlanItem) -> FakeCall:
        return FakeCall(
            "fc-failed",
            _rows(item),
            [],
            failure=RuntimeError("\nremote exploded\n"),
        )

    with pytest.raises(ProtocolError, match="remote collection failed"):
        execute_call_plan(request, _BINDING, journal, spawn=spawn, now=_clock())

    assert [record.status for record in journal.records] == ["spawned", "failed"]
    assert journal.records[-1].error == "RuntimeError: \nremote exploded"
    loaded = AttemptJournal.load(journal.path)
    with pytest.raises(ProtocolError, match="cannot retry"):
        execute_call_plan(
            request,
            _BINDING,
            loaded,
            spawn=lambda _item: pytest.fail("must not spawn"),
            restore=lambda _call_id: pytest.fail("must not restore failed calls"),
            now=_clock(),
        )


def test_commit_accepts_structured_failure_rows(tmp_path: Path) -> None:
    request = _request(gpus=("A",), banks=(0,))
    output = tmp_path / "with-failure.jsonl.zst"
    journal = preflight_collection(output)

    def spawn(item: CallPlanItem) -> FakeCall:
        return FakeCall("fc-failure-row", _rows(item, failed=True), [])

    chunks = execute_call_plan(request, _BINDING, journal, spawn=spawn, now=_clock())
    manifest = commit_chunks(output, request, _BINDING, journal, chunks)

    assert manifest.failures == 1
    rows = read_measurements(output)
    failure = next(row for row in rows if not row.correct)
    assert failure.failure_stage == "compile"


def test_invalid_commit_preserves_existing_output(tmp_path: Path) -> None:
    request = _request(gpus=("A",), banks=(0,))
    output = tmp_path / "existing.jsonl.zst"
    journal = preflight_collection(output)
    write_text_atomic(output, "existing")

    with pytest.raises(ProtocolError, match="retrieved chunks"):
        commit_chunks(output, request, _BINDING, journal, ())

    assert output.read_text(encoding="utf-8") == "existing"


def test_preflight_detects_incomplete_output_pair(tmp_path: Path) -> None:
    output = tmp_path / "orphan.jsonl.zst"
    write_text_atomic(output, "orphan")
    with pytest.raises(ArtifactError, match="output pair is incomplete"):
        preflight_collection(output)


def test_preflight_detects_data_sidecar_digest_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "mismatch.jsonl.zst"
    write_text_atomic(output, "changed")
    write_json_atomic(manifest_path(output), {"data": {"sha256": "0" * 64}})

    with pytest.raises(ArtifactError, match="digest mismatch"):
        preflight_collection(output)


def test_resume_requires_exact_request_and_digests(tmp_path: Path) -> None:
    request = _request(gpus=("A",), banks=(0,))
    output = tmp_path / "binding.jsonl.zst"
    journal = preflight_collection(output)

    def spawn(item: CallPlanItem) -> FakeCall:
        return FakeCall("fc-binding", _rows(item), [])

    execute_call_plan(request, _BINDING, journal, spawn=spawn, now=_clock())
    changed = CollectionBinding("f" * 64, "b" * 64, "c" * 64, "d" * 64)
    with pytest.raises(ProtocolError, match="does not match"):
        journal.require_binding(request, changed)
