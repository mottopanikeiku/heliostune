"""Plan, checkpoint and schedule the frozen action-expansion collection.

Everything in this module runs on CPU. Torch and Triton enter only through the
measurement function in ``heliostune.action_expansion``, so CPU tests can check run
identity, checkpoint isolation, resume rules and bank-1 selection directly.

A run is named by an explicit run ID. Its checkpoint lives under its own directory,
and a resume must match every identity field of the saved run, including the exact
hardware profile observed in its first session. Repeats therefore never share a
checkpoint, and two experiments cannot be mixed in one artifact.
"""

from __future__ import annotations

import hashlib
import random
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from heliostune.action_configs import action_set_manifest, candidates_for
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, Workload
from heliostune.errors import ProtocolError
from heliostune.hardware import HARDWARE_EXPECTATIONS
from heliostune.tooling.artifacts import strict_json_dumps, strict_json_loads, write_bytes_atomic

RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
RUNNERS = ("modal", "local")
BANKS = (1, 2)
PILOT_M = (1, 7, 31, 1024)
WARMUP_MS = 25
REP_MS = 100
QUANTILES = (0.2, 0.5, 0.8)
ATOL = 1e-2
RTOL = 1e-2
PLAN_PATH = "docs/action-expansion.md"
# Fields a resumed checkpoint must match exactly. Sessions, rows, selections and
# durations accumulate; everything here identifies one experiment.
IDENTITY_FIELDS = (
    "run",
    "pilot",
    "plan_commit",
    "hardware",
    "software",
    "precision",
    "protocol",
    "action_set",
)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")

Measure = Callable[[int, int, Workload, frozenset[str] | None], list[dict[str, Any]]]


def validate_run_id(run_id: str) -> str:
    """Accept a lowercase directory-safe name, so a run ID can never escape its root."""
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ProtocolError(
            "run ID must be 1-64 characters of lowercase letters, digits, '.', '_' or '-', "
            f"starting with a letter or digit; got {run_id!r}"
        )
    return run_id


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything that names one collection before any GPU is touched."""

    run_id: str
    runner: str
    gpu: str
    pilot: bool
    source_commit: str
    plan_commit: str
    collector_sha256: str
    source_sha256: str
    allow_unlisted_gpu: bool = False

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if self.runner not in RUNNERS:
            raise ProtocolError(f"runner must be one of {RUNNERS}, got {self.runner!r}")
        if not isinstance(self.gpu, str) or not self.gpu.strip():
            raise ProtocolError("GPU label must be a nonblank string")
        if type(self.pilot) is not bool or type(self.allow_unlisted_gpu) is not bool:
            raise ProtocolError("pilot and allow_unlisted_gpu must be booleans")
        if self.gpu not in HARDWARE_EXPECTATIONS and not self.allow_unlisted_gpu:
            choices = ", ".join(HARDWARE_EXPECTATIONS)
            raise ProtocolError(
                f"GPU {self.gpu!r} has no hardware expectation ({choices}); pass "
                "allow_unlisted_gpu to bind the run to its exact observed profile instead"
            )
        for name in ("source_commit", "plan_commit"):
            if _HEX40.fullmatch(getattr(self, name)) is None:
                raise ProtocolError(f"{name} must be a full 40-character commit hash")
        for name in ("collector_sha256", "source_sha256"):
            if _HEX64.fullmatch(getattr(self, name)) is None:
                raise ProtocolError(f"{name} must be a 64-character SHA-256 digest")

    @property
    def validates_hardware(self) -> bool:
        return self.gpu in HARDWARE_EXPECTATIONS

    def run_record(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "runner": self.runner,
            "gpu": self.gpu,
            "hardware_validation": (
                "registry expectation" if self.validates_hardware else "unlisted GPU label"
            ),
        }


def protocol() -> dict[str, Any]:
    """The frozen measurement protocol, unchanged from the published H100 run."""
    return {
        "warmup_ms": WARMUP_MS,
        "rep_ms": REP_MS,
        "quantiles": list(QUANTILES),
        "atol": ATOL,
        "rtol": RTOL,
        "selection_bank": 1,
        "scoring_bank": 2,
        "selection": "minimum correct bank-1 median, then config-key tie-break",
        "reference": "FP16 operands; torch.mm out_dtype=float32, TF32 disabled only for reference",
        "timed_calls": "allocating CUDA FP16 calls; split-K workspace/reduction included",
        "tensor_seed": "bank * 10000 + index in that bank's shuffled workload order",
        "workload_order_seed": "bank",
        "config_order_seed": "bank * 10000 + workload index + 70000",
    }


def collection_workloads(pilot: bool) -> tuple[Workload, ...]:
    """All 96 named workloads, or the pilot's largest-K workload at each pilot M."""
    if not pilot:
        return DEFAULT_WORKLOADS
    return tuple(
        max((w for w in DEFAULT_WORKLOADS if w.m == m), key=lambda w: (w.k, w.n)) for m in PILOT_M
    )


def bank_order(workloads: Sequence[Workload], bank: int) -> list[Workload]:
    order = list(workloads)
    random.Random(bank).shuffle(order)
    return order


def tensor_seed(bank: int, index: int) -> int:
    return bank * 10000 + index


def timing_order(
    bank: int, index: int, workload: Workload, finalists: frozenset[str] | None
) -> list[str]:
    """Return config keys in timing order: torch first, then the seeded candidate shuffle.

    Bank 2 keeps only the bank-1 finalists before shuffling, exactly as the published
    collector did, so a CPU plan reproduces the GPU order.
    """
    keys = [c.key for c in DEFAULT_CONFIGS] + [c.key for c in candidates_for(workload)]
    if finalists is not None:
        keys = [key for key in keys if key in finalists]
    random.Random(bank * 10000 + index + 70000).shuffle(keys)
    return ["torch", *keys]


def select_finalists(rows: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    """Pick the minimum valid bank-1 median separately for old and new actions."""
    selection: dict[str, str | None] = {}
    for group in ("old", "new"):
        valid = [
            r
            for r in rows
            if r["family"] != "torch"
            and (r["family"] == "old") == (group == "old")
            and r["correct"]
            and r["error"] is None
            and r["median_ms"] is not None
        ]
        selection[group] = (
            min(valid, key=lambda r: (r["median_ms"], r["config_key"]))["config_key"]
            if valid
            else None
        )
    return selection


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    """Give every run ID, and its pilot and full collections, a separate checkpoint."""
    return root / spec.run_id / ("pilot" if spec.pilot else "full") / "checkpoint.json"


def default_output(spec: RunSpec) -> Path:
    suffix = "-pilot" if spec.pilot else ""
    return Path("results/action-expansion-runs") / f"{spec.run_id}{suffix}.json"


def new_result(
    spec: RunSpec,
    hardware: Mapping[str, Any],
    versions: Mapping[str, str],
    precision: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": spec.run_record(),
        "plan_commit": spec.plan_commit,
        "source_commit": spec.source_commit,
        "pilot": spec.pilot,
        "hardware": dict(hardware),
        "software": {
            **versions,
            "source_sha256": spec.source_sha256,
            "collector_sha256": spec.collector_sha256,
        },
        "precision": dict(precision),
        "protocol": protocol(),
        "action_set": action_set_manifest(),
        "rows": [],
        "selected": {},
        "duration_seconds": 0.0,
        "sessions": [],
    }


def start_or_resume(
    path: Path, fresh: Mapping[str, Any], workloads: Sequence[Workload]
) -> tuple[dict[str, Any], set[tuple[int, str]]]:
    """Load a matching checkpoint, or start from ``fresh`` when none exists.

    Any difference in an identity field is an error: a changed run, plan, device,
    software, precision setting, protocol or action set needs a new run ID.
    """
    if not path.exists():
        return dict(fresh), set()
    checkpoint = strict_json_loads(path.read_text(encoding="utf-8"), source=path)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"data", "completed_units"}:
        raise ProtocolError(f"{path} is not an action-expansion checkpoint")
    saved = checkpoint["data"]
    if not isinstance(saved, dict):
        raise ProtocolError(f"{path} has no checkpoint data")
    for field in IDENTITY_FIELDS:
        if saved.get(field) != fresh[field]:
            raise ProtocolError(
                f"checkpoint {path} differs in {field}; use a new run ID instead of "
                "mixing experiments"
            )
    planned = {(bank, w.key) for bank in BANKS for w in workloads}
    completed = {(int(bank), str(key)) for bank, key in checkpoint["completed_units"]}
    if not completed <= planned:
        raise ProtocolError(f"checkpoint {path} records units outside this run's plan")
    measured = {(int(r["bank"]), str(r["workload_key"])) for r in saved["rows"]}
    if measured != completed:
        raise ProtocolError(f"checkpoint {path} rows do not match its completed units")
    return saved, completed


def save_checkpoint(path: Path, result: Mapping[str, Any], completed: set[tuple[int, str]]) -> None:
    payload = {"data": result, "completed_units": sorted([bank, key] for bank, key in completed)}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(path, strict_json_dumps(payload, compact=True).encode("utf-8"))


def begin_session(
    result: dict[str, Any], spec: RunSpec, hardware: Mapping[str, Any], session_id: str
) -> dict[str, Any]:
    session: dict[str, Any] = {
        "id": session_id,
        "runner": spec.runner,
        "source_commit": spec.source_commit,
        "hardware": dict(hardware),
        "units": [],
        "duration_seconds": 0.0,
    }
    result["sessions"].append(session)
    return session


def is_complete(completed: set[tuple[int, str]], workloads: Sequence[Workload]) -> bool:
    return all((bank, w.key) in completed for bank in BANKS for w in workloads)


def run_collection(
    result: dict[str, Any],
    completed: set[tuple[int, str]],
    workloads: Sequence[Workload],
    measure: Measure,
    save: Callable[[dict[str, Any], set[tuple[int, str]]], None],
    *,
    session: dict[str, Any],
    started: float,
    clock: Callable[[], float],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure every unfinished bank/workload unit and checkpoint after each one.

    Bank 1 times every applicable action and selects finalists. Bank 2 times only
    those finalists and torch. Completed units are skipped, so an interrupted run
    resumes where it stopped.
    """
    prior_duration = result["duration_seconds"]
    for bank in BANKS:
        order = bank_order(workloads, bank)
        for index, workload in enumerate(order):
            if (bank, workload.key) in completed:
                continue
            finalists = (
                None
                if bank == 1
                else frozenset(k for k in result["selected"][workload.key].values() if k)
            )
            rows = measure(bank, index, workload, finalists)
            result["rows"].extend(rows)
            if bank == 1:
                result["selected"][workload.key] = select_finalists(rows)
            session["units"].append([bank, workload.key])
            session["duration_seconds"] = clock() - started
            result["duration_seconds"] = prior_duration + session["duration_seconds"]
            completed.add((bank, workload.key))
            save(result, completed)
            log(f"bank={bank} completed={index + 1}/{len(order)} workload={workload.key}")
    return result


def plan_manifest(spec: RunSpec, checkpoint: Path, *, clean_tree: bool) -> dict[str, Any]:
    """Describe a run without a GPU: identity, protocol, unit order, seeds and paths."""
    workloads = collection_workloads(spec.pilot)
    banks = []
    for bank in BANKS:
        units = []
        for index, workload in enumerate(bank_order(workloads, bank)):
            unit: dict[str, Any] = {
                "index": index,
                "workload_key": workload.key,
                "tensor_seed": tensor_seed(bank, index),
                "config_order_seed": bank * 10000 + index + 70000,
            }
            if bank == 1:
                unit["timing_order"] = timing_order(bank, index, workload, None)
            units.append(unit)
        banks.append({"bank": bank, "units": units})
    return {
        "schema_version": 1,
        "kind": "action-expansion run plan (no measurements)",
        "run": spec.run_record(),
        "pilot": spec.pilot,
        "source_commit": spec.source_commit,
        "plan_commit": spec.plan_commit,
        "source_sha256": spec.source_sha256,
        "collector_sha256": spec.collector_sha256,
        "clean_tree": clean_tree,
        "checkpoint": checkpoint.as_posix(),
        "default_output": default_output(spec).as_posix(),
        "protocol": protocol(),
        "action_set": action_set_manifest(),
        "workload_count": len(workloads),
        "expected_units": len(BANKS) * len(workloads),
        "bank2_rule": (
            "bank 2 times torch and only the old/new finalists selected on bank 1, "
            "in the config-order seed's shuffle"
        ),
        "banks": banks,
    }


def git_identity(repository: Path) -> tuple[str, str, bool]:
    """Return HEAD, the last commit touching the analysis plan, and whether the tree is clean."""

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()

    clean = not git("status", "--porcelain", "--untracked-files=no")
    plan_commit = git("log", "-1", "--format=%H", "--", PLAN_PATH)
    if not plan_commit:
        raise ProtocolError(f"{PLAN_PATH} has no commit; commit the plan before collecting")
    return git("rev-parse", "HEAD"), plan_commit, clean


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_source_sha256(package: Path) -> str:
    """Digest the package source exactly as the Modal wheel manifest does."""
    if not package.is_dir():
        raise ProtocolError(f"package source directory does not exist: {package}")
    digest = hashlib.sha256()
    paths = sorted(
        path for path in package.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    )
    for path in paths:
        digest.update(f"heliostune/{path.relative_to(package).as_posix()}".encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
