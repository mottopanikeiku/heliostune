"""Run identity, checkpoint isolation and resume rules for the action collector, on CPU."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from heliostune.action_collection import (
    BANKS,
    IDENTITY_FIELDS,
    RunSpec,
    bank_order,
    begin_session,
    checkpoint_path,
    collection_workloads,
    default_output,
    git_identity,
    is_complete,
    new_result,
    package_source_sha256,
    plan_manifest,
    protocol,
    run_collection,
    save_checkpoint,
    select_finalists,
    start_or_resume,
    timing_order,
    validate_run_id,
)
from heliostune.action_configs import action_set_manifest, candidates_for
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, Workload
from heliostune.errors import ArtifactError, ProtocolError
from heliostune.tooling.artifacts import strict_json_dumps

_REPO = Path(__file__).resolve().parents[1]
_HARDWARE = {
    "gpu": "H100",
    "device_name": "NVIDIA H100 80GB HBM3",
    "compute_capability": [9, 0],
    "multiprocessor_count": 132,
    "total_memory_gb": 79.18,
    "cuda_version": "12.8",
    "torch_version": "2.8.0+cu128",
    "triton_version": "3.4.0",
}
_VERSIONS = {"torch": "2.8.0", "triton": "3.4.0", "numpy": "2.4.6"}
_PRECISION = {
    "allow_tf32": False,
    "allow_fp16_reduced_precision_reduction": True,
    "float32_matmul_precision": "highest",
}


def _spec(**changes: Any) -> RunSpec:
    fields: dict[str, Any] = {
        "run_id": "h100-repeat-1",
        "runner": "local",
        "gpu": "H100",
        "pilot": True,
        "source_commit": "a" * 40,
        "plan_commit": "b" * 40,
        "collector_sha256": "c" * 64,
        "source_sha256": "d" * 64,
    }
    fields.update(changes)
    return RunSpec(**fields)


def _median(bank: int, workload: Workload, key: str) -> float:
    # Deterministic fake medians: distinct per unit and action, so selection is unique.
    return 0.01 + (sum(map(ord, f"{bank}{workload.key}{key}")) % 997) / 1e5


class _FakeMeasure:
    """Stand-in for the GPU unit body: same row fields, deterministic medians."""

    def __init__(self, fail_after: int | None = None, failing_keys: frozenset[str] = frozenset()):
        self.calls: list[tuple[int, str]] = []
        self.fail_after = fail_after
        self.failing_keys = failing_keys

    def __call__(
        self, bank: int, index: int, workload: Workload, finalists: frozenset[str] | None
    ) -> list[dict[str, Any]]:
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError("simulated preemption")
        self.calls.append((bank, workload.key))
        rows = []
        for key in timing_order(bank, index, workload, finalists):
            failed = key in self.failing_keys
            family = "torch" if key == "torch" else ("old" if ":" not in key else key.split(":")[0])
            median = None if failed else _median(bank, workload, key)
            rows.append(
                {
                    "workload_key": workload.key,
                    "bank": bank,
                    "family": family,
                    "config_key": key,
                    "correct": not failed,
                    "error": "compile: RuntimeError: boom" if failed else None,
                    "median_ms": median,
                }
            )
        return rows


def _clock() -> Any:
    ticks = iter(range(1, 10_000))
    return lambda: float(next(ticks))


def _run(
    spec: RunSpec, path: Path, measure: _FakeMeasure, session_id: str = "s1"
) -> tuple[dict[str, Any], set[tuple[int, str]]]:
    workloads = collection_workloads(spec.pilot)
    fresh = new_result(spec, _HARDWARE, _VERSIONS, _PRECISION)
    result, completed = start_or_resume(path, fresh, workloads)
    session = begin_session(result, spec, _HARDWARE, session_id)
    run_collection(
        result,
        completed,
        workloads,
        measure,
        lambda data, units: save_checkpoint(path, data, units),
        session=session,
        started=0.0,
        clock=_clock(),
        log=lambda _message: None,
    )
    return result, completed


@pytest.mark.parametrize("run_id", ["h100-repeat-1", "a100.2026-11-02", "r", "x" * 64, "0_a"])
def test_run_ids_are_directory_safe_names(run_id: str) -> None:
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["", "../escape", "A100", "-x", "a/b", "x" * 65, " x", "a b"])
def test_unsafe_run_ids_are_rejected(run_id: str) -> None:
    with pytest.raises(ProtocolError, match="run ID"):
        validate_run_id(run_id)


def test_run_spec_validates_identity_and_records_hardware_validation() -> None:
    assert _spec().run_record() == {
        "run_id": "h100-repeat-1",
        "runner": "local",
        "gpu": "H100",
        "hardware_validation": "registry expectation",
    }
    with pytest.raises(ProtocolError, match="no hardware expectation"):
        _spec(gpu="A100-40GB")
    unlisted = _spec(gpu="A100-40GB", allow_unlisted_gpu=True)
    assert not unlisted.validates_hardware
    assert unlisted.run_record()["hardware_validation"] == "unlisted GPU label"
    for changes in (
        {"runner": "slurm"},
        {"source_commit": "abc"},
        {"plan_commit": "B" * 40},
        {"collector_sha256": "c" * 63},
        {"source_sha256": "z" * 64},
        {"gpu": " "},
        {"pilot": 1},
    ):
        with pytest.raises(ProtocolError):
            _spec(**changes)


def test_every_run_id_and_collection_kind_gets_its_own_checkpoint(tmp_path: Path) -> None:
    paths = {
        checkpoint_path(tmp_path, _spec(run_id=run_id, pilot=pilot))
        for run_id in ("one", "two")
        for pilot in (True, False)
    }
    assert len(paths) == 4
    assert all(path.is_relative_to(tmp_path) for path in paths)
    assert checkpoint_path(tmp_path, _spec(run_id="one", pilot=False)) == (
        tmp_path / "one/full/checkpoint.json"
    )
    assert default_output(_spec(pilot=True)).as_posix() == (
        "results/action-expansion-runs/h100-repeat-1-pilot.json"
    )


@pytest.mark.parametrize(
    ("artifact", "pilot"),
    [("results/action-expansion-raw.json", False), ("results/action-expansion-pilot.json", True)],
)
def test_the_planner_reproduces_the_published_collection_order(artifact: str, pilot: bool) -> None:
    raw = json.loads((_REPO / artifact).read_text(encoding="utf-8"))
    assert raw["protocol"] == protocol()
    assert raw["action_set"] == action_set_manifest()
    workloads = collection_workloads(pilot)
    assert sorted(raw["selected"]) == sorted(w.key for w in workloads)
    assert len(raw["sessions"]) == 1
    expected_units = []
    for bank in BANKS:
        for index, workload in enumerate(bank_order(workloads, bank)):
            expected_units.append([bank, workload.key])
            rows = [
                r for r in raw["rows"] if r["bank"] == bank and r["workload_key"] == workload.key
            ]
            finalists = (
                None
                if bank == 1
                else frozenset(k for k in raw["selected"][workload.key].values() if k)
            )
            assert [r["config_key"] for r in rows] == timing_order(bank, index, workload, finalists)
            if bank == 1:
                assert select_finalists(rows) == raw["selected"][workload.key]
    assert raw["sessions"][0]["units"] == expected_units


def test_pilot_and_full_rosters() -> None:
    assert collection_workloads(False) == DEFAULT_WORKLOADS
    pilot = collection_workloads(True)
    assert [w.m for w in pilot] == [1, 7, 31, 1024]
    assert {(w.model, w.projection) for w in pilot} == {("qwen2.5-7b", "ffn-down")}


def test_bank1_order_puts_torch_first_and_times_every_applicable_action() -> None:
    workload = collection_workloads(True)[0]
    order = timing_order(1, 0, workload, None)
    assert order[0] == "torch"
    assert Counter(order[1:]) == Counter(
        [c.key for c in DEFAULT_CONFIGS] + [c.key for c in candidates_for(workload)]
    )
    finalists = frozenset({DEFAULT_CONFIGS[3].key, candidates_for(workload)[0].key})
    assert sorted(timing_order(2, 0, workload, finalists)[1:]) == sorted(finalists)


def test_selection_ignores_failures_and_breaks_ties_by_config_key() -> None:
    def row(family: str, key: str, median: float | None, *, correct: bool = True) -> dict[str, Any]:
        error = None if correct and median is not None else "correctness: ValueError"
        return {
            "family": family,
            "config_key": key,
            "median_ms": median,
            "correct": correct,
            "error": error,
        }

    rows = [
        row("torch", "torch", 0.001),
        row("old", "m64-b", 0.02),
        row("old", "m64-a", 0.02),
        row("old", "m16", 0.01, correct=False),
        row("split_k", "split_k:x", 0.03),
        row("tile", "tile:y", None),
    ]
    assert select_finalists(rows) == {"old": "m64-a", "new": "split_k:x"}
    assert select_finalists([row("tile", "tile:y", None)]) == {"old": None, "new": None}


def test_bank2_times_only_bank1_finalists_and_torch(tmp_path: Path) -> None:
    spec = _spec()
    result, completed = _run(spec, checkpoint_path(tmp_path, spec), _FakeMeasure())
    workloads = collection_workloads(True)
    assert is_complete(completed, workloads)
    for workload in workloads:
        bank1 = [r for r in result["rows"] if r["bank"] == 1 and r["workload_key"] == workload.key]
        bank2 = {
            r["config_key"]
            for r in result["rows"]
            if r["bank"] == 2 and r["workload_key"] == workload.key
        }
        selected = select_finalists(bank1)
        assert result["selected"][workload.key] == selected
        assert bank2 == {"torch", *(k for k in selected.values() if k)}


def test_an_interrupted_run_resumes_without_repeating_or_losing_units(tmp_path: Path) -> None:
    spec = _spec()
    path = checkpoint_path(tmp_path, spec)
    reference, _ = _run(spec, tmp_path / "reference.json", _FakeMeasure())

    crashing = _FakeMeasure(fail_after=3)
    with pytest.raises(RuntimeError, match="preemption"):
        _run(spec, path, crashing, session_id="first")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["completed_units"]) == 3

    resumed_measure = _FakeMeasure()
    resumed, completed = _run(spec, path, resumed_measure, session_id="second")
    assert not set(crashing.calls) & set(resumed_measure.calls)
    assert len(crashing.calls) + len(resumed_measure.calls) == 2 * len(collection_workloads(True))
    assert resumed["rows"] == reference["rows"]
    assert resumed["selected"] == reference["selected"]
    assert [s["id"] for s in resumed["sessions"]] == ["first", "second"]
    assert sum(len(s["units"]) for s in resumed["sessions"]) == len(completed)
    assert resumed["duration_seconds"] == sum(s["duration_seconds"] for s in resumed["sessions"])


def _changed(fresh: dict[str, Any], field: str) -> dict[str, Any]:
    changed = copy.deepcopy(fresh)
    if field == "run":
        changed["run"]["run_id"] = "other-run"
    elif field == "pilot":
        changed["pilot"] = not changed["pilot"]
    elif field == "plan_commit":
        changed["plan_commit"] = "e" * 40
    elif field == "hardware":
        changed["hardware"]["device_name"] = "NVIDIA H100 PCIe"
    elif field == "software":
        changed["software"]["source_sha256"] = "f" * 64
    elif field == "precision":
        changed["precision"]["allow_tf32"] = True
    elif field == "protocol":
        changed["protocol"]["rep_ms"] = 200
    else:
        changed["action_set"]["new"] = changed["action_set"]["new"][:-1]
    return changed


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_resume_rejects_any_identity_change(tmp_path: Path, field: str) -> None:
    spec = _spec()
    path = checkpoint_path(tmp_path, spec)
    fresh = new_result(spec, _HARDWARE, _VERSIONS, _PRECISION)
    save_checkpoint(path, fresh, set())
    assert start_or_resume(path, fresh, collection_workloads(True))[1] == set()
    with pytest.raises(ProtocolError, match=f"differs in {field}"):
        start_or_resume(path, _changed(fresh, field), collection_workloads(True))


def test_resume_rejects_units_outside_the_plan_or_without_rows(tmp_path: Path) -> None:
    spec = _spec()
    path = checkpoint_path(tmp_path, spec)
    fresh = new_result(spec, _HARDWARE, _VERSIONS, _PRECISION)
    full_only = DEFAULT_WORKLOADS[0]
    assert full_only not in collection_workloads(True)
    save_checkpoint(path, fresh, {(1, full_only.key)})
    with pytest.raises(ProtocolError, match="outside this run's plan"):
        start_or_resume(path, fresh, collection_workloads(True))
    pilot_workload = collection_workloads(True)[0]
    save_checkpoint(path, fresh, {(1, pilot_workload.key)})
    with pytest.raises(ProtocolError, match="do not match its completed units"):
        start_or_resume(path, fresh, collection_workloads(True))
    path.write_text('{"data": {}}', encoding="utf-8")
    with pytest.raises(ProtocolError, match="not an action-expansion checkpoint"):
        start_or_resume(path, fresh, collection_workloads(True))


def test_failed_candidates_stay_unselected_rows(tmp_path: Path) -> None:
    workload = collection_workloads(True)[0]
    failing = frozenset({c.key for c in candidates_for(workload)})
    spec = _spec()
    result, _ = _run(spec, checkpoint_path(tmp_path, spec), _FakeMeasure(failing_keys=failing))
    assert result["selected"][workload.key]["new"] is None
    bank2 = {
        r["config_key"]
        for r in result["rows"]
        if r["bank"] == 2 and r["workload_key"] == workload.key
    }
    assert bank2 == {"torch", result["selected"][workload.key]["old"]}
    # Every new action fails on bank 1 for every pilot workload, and none reaches bank 2.
    expected_failures = sum(
        len({c.key for c in candidates_for(w)} & failing) for w in collection_workloads(True)
    )
    assert sum(r["error"] is not None for r in result["rows"]) == expected_failures == 36
    assert all(selection["new"] is None for selection in result["selected"].values())


def test_plan_is_deterministic_and_matches_the_collection_schedule(tmp_path: Path) -> None:
    spec = _spec(pilot=False)
    checkpoint = checkpoint_path(tmp_path, spec)
    plan = plan_manifest(spec, checkpoint, clean_tree=True)
    assert strict_json_dumps(plan) == strict_json_dumps(
        plan_manifest(spec, checkpoint, clean_tree=True)
    )
    assert plan["expected_units"] == 2 * len(DEFAULT_WORKLOADS) == 192
    assert plan["run"] == spec.run_record() and plan["protocol"] == protocol()
    bank1, bank2 = plan["banks"]
    assert [u["workload_key"] for u in bank1["units"]] == [
        w.key for w in bank_order(DEFAULT_WORKLOADS, 1)
    ]
    assert [u["workload_key"] for u in bank2["units"]] == [
        w.key for w in bank_order(DEFAULT_WORKLOADS, 2)
    ]
    assert all("timing_order" not in u for u in bank2["units"])
    assert bank1["units"][5]["tensor_seed"] == 10005
    assert bank2["units"][5]["config_order_seed"] == 2 * 10000 + 5 + 70000


def _git(repository: Path, *arguments: str) -> str:
    # Ignore the machine's global Git config (signing, hooks, default branch).
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "PATH": "/usr/bin:/bin",
    }
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    (repository / "docs").mkdir(parents=True)
    (repository / "src/heliostune").mkdir(parents=True)
    (repository / "docs/action-expansion.md").write_text("plan\n", encoding="utf-8")
    (repository / "src/heliostune/__init__.py").write_text("", encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(repository, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "plan")
    return repository


def test_git_identity_reports_head_plan_commit_and_cleanliness(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    plan_commit = _git(repository, "rev-parse", "HEAD")
    (repository / "notes.txt").write_text("x", encoding="utf-8")
    _git(repository, "add", "notes.txt")
    _git(
        repository, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "notes"
    )
    head = _git(repository, "rev-parse", "HEAD")
    assert git_identity(repository) == (head, plan_commit, True)
    (repository / "notes.txt").write_text("changed", encoding="utf-8")
    assert git_identity(repository) == (head, plan_commit, False)
    empty = tmp_path / "empty"
    empty.mkdir()
    _git(empty, "init", "-q")
    (empty / "x").write_text("x", encoding="utf-8")
    _git(empty, "add", "x")
    _git(empty, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "x")
    with pytest.raises(ProtocolError, match="has no commit"):
        git_identity(empty)


def test_source_digest_matches_the_modal_wheel_manifest_digest() -> None:
    spec = importlib.util.spec_from_file_location(
        "_test_build_modal_wheel", _REPO / "scripts/build_modal_wheel.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module._source_digest(module._source_entries())
    assert package_source_sha256(_REPO / "src/heliostune") == expected
    with pytest.raises(ProtocolError, match="does not exist"):
        package_source_sha256(_REPO / "missing")


def _runner() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_test_collect_action_expansion", _REPO / "scripts/collect_action_expansion.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_runner_dry_run_writes_the_plan_without_a_gpu(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    plan_path = tmp_path / "plan.json"
    arguments = [
        "--run-id",
        "a100-first",
        "--gpu",
        "A100-80GB",
        "--pilot",
        "--dry-run",
        "--checkpoint-root",
        str(tmp_path / "runs"),
        "--plan-output",
        str(plan_path),
    ]
    assert _runner().main(arguments, repository=repository) == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["run"]["run_id"] == "a100-first" and plan["run"]["gpu"] == "A100-80GB"
    assert plan["clean_tree"] is True and plan["expected_units"] == 8
    assert plan["checkpoint"] == (tmp_path / "runs/a100-first/pilot/checkpoint.json").as_posix()
    with pytest.raises(ArtifactError, match="cannot commit artifact"):
        _runner().main(arguments, repository=repository)


def test_local_runner_refuses_dirty_trees_existing_outputs_and_missing_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    runner = _runner()
    base = ["--run-id", "r1", "--gpu", "H100", "--checkpoint-root", str(tmp_path / "runs")]
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        runner.main([*base, "--output", str(existing)], repository=repository)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit, match="Torch is not installed"):
        runner.main([*base, "--output", str(tmp_path / "new.json")], repository=repository)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(SystemExit, match="no CUDA device"):
        runner.main([*base, "--output", str(tmp_path / "new.json")], repository=repository)
    (repository / "docs/action-expansion.md").write_text("edited\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="commit every tracked change"):
        runner.main(base, repository=repository)
    with pytest.raises(ProtocolError, match="no hardware expectation"):
        runner.main(["--run-id", "r2", "--gpu", "RTX-5090", "--dry-run"], repository=repository)
