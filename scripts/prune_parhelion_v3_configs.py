"""Apply the frozen global compile/correctness pruning and v3 rank gates."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from heliostune.artifacts import read_json, read_measurements, write_json_atomic
from heliostune.configs import (
    DEFAULT_WORKLOADS,
    PARHELION_V3_CANDIDATE_CONFIGS,
    PARHELION_V3_OFFICIAL_CONFIG_KEYS,
    KernelConfig,
)
from heliostune.features import v3_feature_rank
from heliostune.hardware import expectation_for_gpu, validate_hardware
from heliostune.protocol import load_v3_protocol, require_v3_runtime, runtime_manifest
from heliostune.schema import HardwareProfile

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_PROTOCOL = _REPO / "benchmarks/parhelion-v3-development-protocol.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _rank_folds(
    retained: tuple[KernelConfig, ...],
    profiles: tuple[HardwareProfile, ...],
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for heldout_model in sorted({workload.model for workload in DEFAULT_WORKLOADS}):
        heldout_shapes = {
            (workload.m, workload.n, workload.k)
            for workload in DEFAULT_WORKLOADS
            if workload.model == heldout_model
        }
        eligible = tuple(
            workload
            for workload in DEFAULT_WORKLOADS
            if workload.model != heldout_model
            and (workload.m, workload.n, workload.k) not in heldout_shapes
        )
        ranks[heldout_model] = v3_feature_rank(eligible, retained, profiles)
    return ranks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=_DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)

    protocol = load_v3_protocol(args.protocol)
    require_v3_runtime(protocol)
    rows = read_measurements(args.input)
    sidecar_path = Path(f"{args.input}.manifest.json")
    sidecar = cast(Mapping[str, object], read_json(sidecar_path))
    facts = cast(Mapping[str, object], sidecar.get("facts", {}))
    if facts.get("head_commit") != _head():
        raise ValueError("candidate sidecar HEAD does not match the pruning checkout")
    binding = cast(Mapping[str, object], sidecar.get("binding", {}))
    if binding.get("protocol_sha256") != _sha256(args.protocol):
        raise ValueError("candidate sidecar protocol digest does not match")

    expected_gpus = ("L4", "A10", "A100-80GB")
    expected_config_keys = {config.key for config in PARHELION_V3_CANDIDATE_CONFIGS}
    expected_workload_keys = {workload.key for workload in DEFAULT_WORKLOADS}
    expected_cells = {
        (gpu, workload, config)
        for gpu in expected_gpus
        for workload in expected_workload_keys
        for config in expected_config_keys
    }
    actual_cells = {(row.hardware.gpu, row.workload.key, row.config.key) for row in rows}
    if len(rows) != len(expected_cells) or actual_cells != expected_cells:
        raise ValueError("candidate bank-0 rows do not match the exact 3×96×52 grid")
    if any(row.bank != 0 for row in rows):
        raise ValueError("candidate pruning may inspect bank 0 only")

    profiles: dict[str, HardwareProfile] = {}
    torch_timings: dict[tuple[str, str], tuple[float, ...]] = {}
    prune_reasons: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        gpu = row.hardware.gpu
        known = profiles.setdefault(gpu, row.hardware)
        if known != row.hardware:
            raise ValueError(f"candidate hardware profile changes within {gpu}")
        validate_hardware(row.hardware, expectation_for_gpu(gpu))
        torch_p20 = row.torch_latency_p20_ms
        torch_p80 = row.torch_latency_p80_ms
        torch_wall = row.torch_benchmark_wall_ms
        if torch_p20 is None or torch_p80 is None or torch_wall is None:
            raise ValueError(f"candidate torch quantiles/wall are missing on {gpu}")
        timing = (row.torch_latency_ms, torch_p20, torch_p80, torch_wall)
        if any(value <= 0 for value in timing):
            raise ValueError(f"candidate torch timings/wall must be positive on {gpu}")
        key = (gpu, row.workload.key)
        previous = torch_timings.setdefault(key, timing)
        if previous != timing:
            raise ValueError(f"candidate repeated torch timing differs at {key}")
        if row.usable:
            if row.failure_stage is not None or row.benchmark_wall_ms is None:
                raise ValueError("usable candidate row lacks a positive benchmark wall")
        else:
            if row.failure_stage == "benchmark":
                raise ValueError(
                    f"benchmark failure aborts pruning: {gpu}/{row.workload.key}/{row.config.key}"
                )
            if row.failure_stage not in {"compile", "correctness"}:
                raise ValueError("candidate failure is not explicitly prunable")
            prune_reasons[row.config.key].append(
                {
                    "gpu": gpu,
                    "workload_key": row.workload.key,
                    "failure_stage": row.failure_stage,
                    "error": row.error or "unreported",
                }
            )

    retained = tuple(
        config for config in PARHELION_V3_CANDIDATE_CONFIGS if config.key not in prune_reasons
    )
    retained_official = sum(config.key in PARHELION_V3_OFFICIAL_CONFIG_KEYS for config in retained)
    if len(retained) < 36 or retained_official == 0:
        raise ValueError(
            f"pruning gate failed: retained={len(retained)}, official={retained_official}"
        )
    l4_a10_ranks = _rank_folds(retained, (profiles["L4"], profiles["A10"]))
    a100_ranks = _rank_folds(
        retained,
        (profiles["L4"], profiles["A10"], profiles["A100-80GB"]),
    )
    if any(rank < 18 for rank in l4_a10_ranks.values()):
        raise ValueError(f"L4+A10 v3 rank gate failed: {l4_a10_ranks}")
    if any(rank < 19 for rank in a100_ranks.values()):
        raise ValueError(f"A100 v3 rank gate failed: {a100_ranks}")

    payload = {
        "schema_version": 1,
        "study_id": "parhelion-v3-retained-configs",
        "protocol": {
            "path": str(args.protocol),
            "sha256": _sha256(args.protocol),
        },
        "candidate_data": {
            "path": str(args.input),
            "sha256": _sha256(args.input),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": _sha256(sidecar_path),
            "rows": len(rows),
        },
        "candidate_count": len(PARHELION_V3_CANDIDATE_CONFIGS),
        "retained_config_keys": [config.key for config in retained],
        "retained_count": len(retained),
        "retained_official_config_keys": [
            config.key for config in retained if config.key in PARHELION_V3_OFFICIAL_CONFIG_KEYS
        ],
        "retained_official_count": retained_official,
        "pruned": [
            {"config_key": key, "reasons": reasons}
            for key, reasons in sorted(prune_reasons.items())
        ],
        "rank_gates": {
            "l4_a10": l4_a10_ranks,
            "l4_a10_a100_80gb": a100_ranks,
        },
        "hardware": [profiles[gpu].to_dict() for gpu in expected_gpus],
        "runtime": runtime_manifest(),
        "head_commit": _head(),
    }
    write_json_atomic(args.output, payload)
    print(f"retained={len(retained)} official={retained_official} pruned={len(prune_reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
