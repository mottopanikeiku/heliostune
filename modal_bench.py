"""Collect the HeliosTune benchmark matrix concurrently on Modal GPUs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal


app = modal.App("heliostune-bench")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "triton==3.4.0")
    .add_local_dir("src/heliostune", remote_path="/root/heliostune")
)


@app.function(image=image, gpu="L4", timeout=60 * 60)
def benchmark_l4(
    replicate: int = 0,
    warmup_ms: int = 25,
    rep_ms: int = 100,
) -> list[dict[str, Any]]:
    from heliostune.kernel import collect_benchmarks

    return collect_benchmarks(
        "L4",
        replicate=replicate,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )


@app.function(image=image, gpu="A10", timeout=60 * 60)
def benchmark_a10(
    replicate: int = 0,
    warmup_ms: int = 25,
    rep_ms: int = 100,
) -> list[dict[str, Any]]:
    from heliostune.kernel import collect_benchmarks

    return collect_benchmarks(
        "A10",
        replicate=replicate,
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
) -> None:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    calls = [
        benchmark.spawn(
            replicate=replicate,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
        for replicate in range(replicates)
        for benchmark in (benchmark_l4, benchmark_a10)
    ]
    records = sorted(
        (record for call in calls for record in call.get()),
        key=_record_sort_key,
    )
    payload = "".join(
        f"{json.dumps(record, sort_keys=True, separators=(',', ':'))}\n" for record in records
    )
    Path(output).write_text(payload, encoding="utf-8")
