"""Collect the HeliosTune benchmark matrix concurrently on Modal GPUs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

app = modal.App("heliostune-bench")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26,<3", "torch==2.8.0", "triton==3.4.0")
    .add_local_dir("src/heliostune", remote_path="/root/heliostune")
)


@app.function(image=image, gpu="L4", timeout=60 * 60)
def benchmark_l4(
    replicate: int = 0,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    pilot: bool = False,
) -> list[dict[str, Any]]:
    from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS
    from heliostune.kernel import collect_benchmarks

    return collect_benchmarks(
        "L4",
        replicate=replicate,
        configs=DEFAULT_CONFIGS[:3] if pilot else DEFAULT_CONFIGS,
        workloads=DEFAULT_WORKLOADS[:2] if pilot else DEFAULT_WORKLOADS,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )


@app.function(image=image, gpu="A10", timeout=60 * 60)
def benchmark_a10(
    replicate: int = 0,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    pilot: bool = False,
) -> list[dict[str, Any]]:
    from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS
    from heliostune.kernel import collect_benchmarks

    return collect_benchmarks(
        "A10",
        replicate=replicate,
        configs=DEFAULT_CONFIGS[:3] if pilot else DEFAULT_CONFIGS,
        workloads=DEFAULT_WORKLOADS[:2] if pilot else DEFAULT_WORKLOADS,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )


@app.function(image=image, gpu="T4", timeout=60 * 60)
def benchmark_t4(
    replicate: int = 0,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    pilot: bool = False,
) -> list[dict[str, Any]]:
    from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS
    from heliostune.kernel import collect_benchmarks

    return collect_benchmarks(
        "T4",
        replicate=replicate,
        configs=DEFAULT_CONFIGS[:3] if pilot else DEFAULT_CONFIGS,
        workloads=DEFAULT_WORKLOADS[:2] if pilot else DEFAULT_WORKLOADS,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )


@app.function(image=image, gpu="H100!", timeout=60 * 60)
def benchmark_h100(
    replicate: int = 0,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    pilot: bool = False,
) -> list[dict[str, Any]]:
    from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS
    from heliostune.kernel import collect_benchmarks

    return collect_benchmarks(
        "H100",
        replicate=replicate,
        configs=DEFAULT_CONFIGS[:3] if pilot else DEFAULT_CONFIGS,
        workloads=DEFAULT_WORKLOADS[:2] if pilot else DEFAULT_WORKLOADS,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )


def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    hardware = record["hardware"]
    workload = record["workload"]
    config = record["config"]
    return (
        hardware["gpu"],
        record["replicate"],
        workload["model"],
        workload["projection"],
        workload["regime"],
        workload["m"],
        workload["n"],
        workload["k"],
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["num_warps"],
        config["num_stages"],
        config["group_m"],
    )


@app.local_entrypoint()
def main(
    output: str = "measurements.jsonl",
    warmup_ms: int = 25,
    rep_ms: int = 100,
    replicates: int = 3,
    pilot: bool = False,
    gpus: str = "L4,A10",
) -> None:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    gpu_names = [gpu.strip() for gpu in gpus.split(",")]
    if any(not gpu for gpu in gpu_names):
        raise ValueError("gpus must be a non-empty comma-separated list")
    if len(set(gpu_names)) != len(gpu_names):
        raise ValueError("gpus must not contain duplicate selectors")

    benchmarks = {
        "L4": benchmark_l4,
        "A10": benchmark_a10,
        "T4": benchmark_t4,
        "H100": benchmark_h100,
    }
    unknown = [gpu for gpu in gpu_names if gpu not in benchmarks]
    if unknown:
        raise ValueError(
            f"unknown GPU selector(s): {', '.join(unknown)}; choose from {', '.join(benchmarks)}"
        )

    calls = [
        benchmarks[gpu].spawn(
            replicate=replicate,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
            pilot=pilot,
        )
        for replicate in range(replicates)
        for gpu in gpu_names
    ]
    records = sorted(
        (record for call in calls for record in call.get()),
        key=_record_sort_key,
    )
    payload = "".join(
        f"{json.dumps(record, sort_keys=True, separators=(',', ':'))}\n" for record in records
    )
    Path(output).write_text(payload, encoding="utf-8")
