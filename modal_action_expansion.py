"""Compare fixed old/new action sets on H100 with bank-1 selection only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import modal

from modal_bench import build_image, configured_modal_wheel

app = modal.App("heliostune-day-actions")
volume = modal.Volume.from_name("heliostune-day-actions", create_if_missing=True)
image = (
    build_image(configured_modal_wheel())
    .env({"TRITON_CACHE_DIR": "/data/triton-cache"})
    .add_local_file("modal_bench.py", remote_path="/root/modal_bench.py")
)


def collect(
    source_commit: str, plan_commit: str, collector_sha256: str, pilot: bool
) -> dict[str, Any]:
    import importlib.metadata
    import math
    import os
    import random
    import time
    import uuid
    from functools import partial

    import torch

    from heliostune.action_configs import ALL_CANDIDATES, candidates_for
    from heliostune.action_expansion import launch
    from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS
    from heliostune.hardware import expectation_for_gpu, validate_hardware
    from heliostune.kernel import _timed_do_bench, _within_tolerance, get_hardware_profile, matmul

    started = time.monotonic()
    device = torch.device("cuda", torch.cuda.current_device())
    hardware = get_hardware_profile("H100", device)
    validate_hardware(hardware, expectation_for_gpu("H100"))
    workloads = list(DEFAULT_WORKLOADS)
    if pilot:
        workloads = [
            max((w for w in workloads if w.m == m), key=lambda w: (w.k, w.n))
            for m in (1, 7, 31, 1024)
        ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "plan_commit": plan_commit,
        "source_commit": source_commit,
        "pilot": pilot,
        "hardware": hardware.to_dict(),
        "software": {
            package: importlib.metadata.version(package) for package in ("torch", "triton", "numpy")
        },
        "precision": {
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "protocol": {
            "warmup_ms": 25,
            "rep_ms": 100,
            "quantiles": [0.2, 0.5, 0.8],
            "atol": 1e-2,
            "rtol": 1e-2,
            "selection_bank": 1,
            "scoring_bank": 2,
            "selection": "minimum correct bank-1 median, then config-key tie-break",
            "reference": "FP16 operands; torch.mm out_dtype=float32, TF32 disabled only for reference",
            "timed_calls": "allocating CUDA FP16 calls; split-K workspace/reduction included",
            "tensor_seed": "bank * 10000 + index in that bank's shuffled workload order",
            "workload_order_seed": "bank",
            "config_order_seed": "bank * 10000 + workload index + 70000",
        },
        "action_set": {
            "old": [
                {"family": "old", "config_key": c.key, "config": c.to_dict()}
                for c in DEFAULT_CONFIGS
            ],
            "new": [
                {"family": c.family, "config_key": c.key, "config": c.to_dict()}
                for c in ALL_CANDIDATES
            ],
        },
        "rows": [],
        "selected": {},
        "duration_seconds": 0.0,
        "sessions": [],
    }
    wheel_manifest = json.loads(
        Path(os.environ["HELIOSTUNE_MODAL_WHEEL_MANIFEST"]).read_text(encoding="utf-8")
    )
    result["software"]["source_sha256"] = wheel_manifest["source_sha256"]
    result["software"]["collector_sha256"] = collector_sha256
    target = Path("/data/pilot-checkpoint.json" if pilot else "/data/full-checkpoint.json")
    completed: set[tuple[int, str]] = set()
    if target.exists():
        checkpoint = json.loads(target.read_text(encoding="utf-8"))
        saved = checkpoint["data"]
        for field in ("pilot", "plan_commit", "software", "precision", "protocol", "action_set"):
            if saved[field] != result[field]:
                raise ValueError(f"checkpoint differs in {field}; do not mix experiments")
        result = saved
        completed = {(int(bank), str(key)) for bank, key in checkpoint["completed_units"]}
    if all((bank, w.key) in completed for bank in (1, 2) for w in workloads):
        return result
    prior_duration = result["duration_seconds"]
    session: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "source_commit": source_commit,
        "hardware": hardware.to_dict(),
        "units": [],
        "duration_seconds": 0.0,
    }
    result["sessions"].append(session)
    row_list: list[dict[str, Any]] = result["rows"]
    for bank in (1, 2):
        order = list(workloads)
        random.Random(bank).shuffle(order)
        for index, workload in enumerate(order):
            if (bank, workload.key) in completed:
                continue
            torch.manual_seed(bank * 10000 + index)
            a = torch.empty((workload.m, workload.k), device=device, dtype=torch.float16)
            b = torch.empty((workload.k, workload.n), device=device, dtype=torch.float16)
            a.uniform_(-1.0, 1.0)
            b.uniform_(-1.0, 1.0)
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                expected = torch.mm(a, b, out_dtype=torch.float32)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            difference = torch.empty_like(expected)
            candidates: list[tuple[str, str, Any, Any]] = [
                ("old", c.key, c.to_dict(), partial(matmul, a, b, c)) for c in DEFAULT_CONFIGS
            ] + [
                (c.family, c.key, c.to_dict(), partial(launch, a, b, c))
                for c in candidates_for(workload)
            ]
            if bank == 2:
                selected = result["selected"][workload.key]
                keys = {key for key in selected.values() if key is not None}
                candidates = [c for c in candidates if c[1] in keys]
            random.Random(bank * 10000 + index + 70000).shuffle(candidates)
            candidates.insert(0, ("torch", "torch", None, partial(torch.matmul, a, b)))
            local_rows: list[dict[str, Any]] = []
            for family, key, config, function in candidates:
                row: dict[str, Any] = {
                    "workload_key": workload.key,
                    "workload": workload.to_dict(),
                    "bank": bank,
                    "family": family,
                    "config_key": key,
                    "config": config,
                    "correct": False,
                    "error": None,
                    "p20_ms": None,
                    "median_ms": None,
                    "p80_ms": None,
                    "max_abs_error": None,
                    "compile_seconds": None,
                    "timing_seconds": None,
                }
                stage = "compile"
                try:
                    torch.cuda.synchronize()
                    compile_started = time.monotonic()
                    output = function()
                    torch.cuda.synchronize()
                    row["compile_seconds"] = time.monotonic() - compile_started
                    stage = "correctness"
                    if not bool(torch.isfinite(output).all().item()):
                        raise ValueError("non-finite output")
                    torch.sub(output, expected, out=difference)
                    difference.abs_()
                    row["max_abs_error"] = float(difference.max().item())
                    if not _within_tolerance(difference, expected, atol=1e-2, rtol=1e-2):
                        raise ValueError("outside unchanged atol=rtol=1e-2 tolerance")
                    row["correct"] = True
                    del output
                    stage = "timing"
                    quantiles, wall_ms = _timed_do_bench(function, warmup_ms=25, rep_ms=100)
                    if not all(math.isfinite(v) and v > 0 for v in quantiles):
                        raise ValueError("non-positive or non-finite timing")
                    row.update(
                        p20_ms=quantiles[0],
                        median_ms=quantiles[1],
                        p80_ms=quantiles[2],
                        timing_seconds=wall_ms / 1000,
                    )
                except Exception as exc:
                    row["error"] = f"{stage}: {type(exc).__name__}: {exc}"
                local_rows.append(row)
                row_list.append(row)
            if bank == 1:
                selection: dict[str, str | None] = {}
                for group in ("old", "new"):
                    valid = [
                        r
                        for r in local_rows
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
                result["selected"][workload.key] = selection
            del difference, expected, b, a, candidates
            session["units"].append([bank, workload.key])
            session["duration_seconds"] = time.monotonic() - started
            result["duration_seconds"] = prior_duration + session["duration_seconds"]
            completed.add((bank, workload.key))
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"data": result, "completed_units": sorted(completed)}, allow_nan=False),
                encoding="utf-8",
            )
            temporary.replace(target)
            volume.commit()
            print(
                f"bank={bank} completed={index + 1}/{len(order)} workload={workload.key}",
                flush=True,
            )
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=1,
    memory=4096,
    timeout=240,
    max_containers=1,
    volumes={"/data": volume},
)
def pilot_collect(source_commit: str, plan_commit: str, collector_sha256: str) -> dict[str, Any]:
    return collect(source_commit, plan_commit, collector_sha256, True)


@app.function(
    image=image,
    gpu="H100!",
    cpu=1,
    memory=4096,
    timeout=1620,
    max_containers=1,
    volumes={"/data": volume},
)
def full_collect(source_commit: str, plan_commit: str, collector_sha256: str) -> dict[str, Any]:
    return collect(source_commit, plan_commit, collector_sha256, False)


@app.local_entrypoint()
def main(pilot: bool = False, output: str = "results/action-expansion-raw.json") -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("commit the action set and analysis plan before collecting measurements")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    plan_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "docs/action-expansion-plan.md"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    collector_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = (
        pilot_collect.remote(commit, plan_commit, collector_sha256)
        if pilot
        else full_collect.remote(commit, plan_commit, collector_sha256)
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['rows'])} rows to {destination}")
