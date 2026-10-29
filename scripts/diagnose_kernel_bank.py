"""Regenerate the CPU-only diagnosis of every stored matmul action matrix.

Reads committed GPU measurements only; it collects no timings. Each file and exact
hardware profile stays a separate view because several archives repeat rows.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import zstandard
from analyze_action_expansion import analyze
from analyze_hopper_benchmark import analyze_artifact

from heliostune.configs import DEFAULT_WORKLOADS, Workload
from heliostune.tooling.artifacts import (
    read_json,
    read_measurements,
    strict_json_dumps,
    strict_json_loads,
)
from heliostune.tooling.bank_diagnosis import Cell, diagnose, expansion_view, shared_cells
from heliostune.tooling.kernel_dossier import build_dossier

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _REPO / "results/kernel-bank-diagnosis.json"
_PILOT_WORKLOADS = tuple(
    w
    for w in DEFAULT_WORKLOADS
    if w.model == "qwen2.5-7b" and w.projection == "ffn-down" and w.m in {1, 7, 31, 1024}
)
DEFINITIONS = {
    "scope": (
        "All benchmarks/data/*.jsonl.zst action matrices, both action-expansion artifacts, "
        "and the single-bank Hopper engineering screen. Each file and exact hardware "
        "profile stays a separate view."
    ),
    "overlap": (
        "Combined archives, raw/canonical validation and component acquisitions repeat "
        "identical normalized cells (workload, bank, action, family, failure, candidate and "
        "torch medians in microseconds; counted in shared_cells). They are alternative views, "
        "never independent replicates, so I do not sum their rows or pool their statistics."
    ),
    "policy": (
        "Exhaustive earlier-bank selection, then fixed-action later-bank scoring. This is "
        "a descriptive selector diagnosis, not a rerun of Thompson/transfer policies or a "
        "probe-budget result."
    ),
    "oracle": (
        "Retrospective minimum valid later-bank median over the compared actions; an "
        "optimistic in-sample diagnostic, not the hardware optimum. Failed candidates "
        "cannot win."
    ),
    "regret": (
        "policy/oracle ratio (1 is best); absolute regret = policy_us - oracle_us. "
        "Separate from the policy/torch and oracle/torch ratios."
    ),
    "weighting": (
        "Named: equal workload weight. Unique shape: equal (M,N,K) weight, averaging all "
        "aliases within a shape first. Ratios use geometric means; microseconds and "
        "indicator rates use arithmetic means. A shape is covered only if every alias has "
        "the metric."
    ),
    "stability": (
        "winner_same compares config-key-tiebroken winners; winner_still_optimal also "
        "accepts exactly tied target medians. Descriptive agreement, not independent-session "
        "reliability."
    ),
    "coverage": (
        "Observed and failed cells by bank/config/regime. Missing cells are not successes. "
        "Unequal bank action rosters appear only in common_actions_only, never full_roster; "
        "workloads absent from either bank appear only in status_counts. No bank-pair claim "
        "for single-bank artifacts."
    ),
    "non_action_artifacts": (
        "The precision probe measures fixed preselected configs under arithmetic settings, "
        "not an action bank; fusion/RMSNorm artifacts are different operators. Their reports "
        "remain linked in the historical archive."
    ),
}


def _compact_diagnosis(cells: list[Cell]) -> dict[str, Any]:
    result = diagnose(cells)
    # Per-regime sufficient statistics retain coverage; avoid repeating aliases
    # and combined-archive rows in a large second copy of the raw measurements.
    for row in (*result["banks"], *result["bank_pairs"]):
        del row["workloads"]
    return result


def _archive_cells(path: Path) -> dict[str, list[Cell]]:
    profiles: dict[str, list[Cell]] = defaultdict(list)
    for row in read_measurements(path):
        profile = strict_json_dumps(row.hardware.to_dict(), compact=True)
        usable = row.usable and row.latency_ms is not None
        profiles[profile].append(
            Cell(
                row.workload,
                row.bank,
                row.config.key,
                "tile",
                cast(float, row.latency_ms) * 1000 if usable else None,
                row.torch_latency_ms * 1000,
                None if usable else row.failure_stage or "unusable",
            )
        )
    return profiles


def _expansion_cells(raw: dict[str, Any]) -> list[Cell]:
    torch = {
        (r["workload_key"], r["bank"]): r["median_ms"]
        if r["correct"] and r["error"] is None
        else None
        for r in raw["rows"]
        if r["family"] == "torch"
    }
    cells = []
    for r in raw["rows"]:
        if r["family"] == "torch":
            continue
        valid = r["correct"] and r["error"] is None and r["median_ms"] is not None
        torch_ms = torch[r["workload_key"], r["bank"]]
        cells.append(
            Cell(
                Workload.from_dict(r["workload"]),
                r["bank"],
                r["config_key"],
                r["family"],
                r["median_ms"] * 1000 if valid else None,
                torch_ms * 1000 if torch_ms is not None else None,
                None if valid else r["error"] or "correctness",
            )
        )
    return cells


def _hopper_cells(raw: dict[str, Any]) -> list[Cell]:
    return [
        Cell(
            Workload.from_dict(r["workload"]),
            r["bank"],
            r["config_key"],
            r["config_kind"],
            r["latency"]["median_ms"] * 1000 if r["correct"] else None,
            r["torch"]["median_ms"] * 1000,
            None if r["correct"] else "correctness",
        )
        for r in raw["rows"]
    ]


def build(repo: Path = _REPO) -> dict[str, Any]:
    inputs: list[dict[str, str]] = []
    datasets: list[dict[str, Any]] = []
    views: dict[str, list[Cell]] = {}

    def bind(path: Path) -> str:
        relative = path.relative_to(repo).as_posix()
        inputs.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return relative

    def add(relative: str, hardware: dict[str, Any], cells: list[Cell], **extra: Any) -> None:
        view = f"{relative} [{hardware['device_name']}]"
        if view in views:
            raise ValueError(f"duplicate view {view}")
        views[view] = cells
        datasets.append(
            {
                "view": view,
                "path": relative,
                "hardware": hardware,
                **extra,
                **_compact_diagnosis(cells),
            }
        )

    # Includes original acquisitions, combined archives, failed/pruned candidate
    # bank, raw mixed-hardware validation, canonical validation and pilot retry.
    for path in sorted((repo / "benchmarks/data").glob("*.jsonl.zst")):
        relative = bind(path)
        for profile, cells in sorted(_archive_cells(path).items()):
            add(relative, cast(dict[str, Any], strict_json_loads(profile)), cells)

    full_summary: dict[str, Any] | None = None
    for name, roster in (
        ("action-expansion-raw.json", DEFAULT_WORKLOADS),
        ("action-expansion-pilot.json", _PILOT_WORKLOADS),
    ):
        path = repo / "results" / name
        relative = bind(path)
        raw = cast(dict[str, Any], read_json(path))
        summary = analyze(raw, expected_workloads=roster)
        view = expansion_view(summary)
        del view["workloads"]
        add(
            relative,
            raw["hardware"],
            _expansion_cells(raw),
            pilot=bool(raw.get("pilot")),
            heldout_expansion=view,
        )
        if not raw.get("pilot"):
            full_summary = summary
    if full_summary is None:
        raise ValueError("the full action-expansion artifact is missing")

    path = repo / "benchmarks/data/hopper-h100-engineering.json.zst"
    relative = bind(path)
    with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as decoded:
        raw = cast(dict[str, Any], strict_json_loads(decoded.read().decode("utf-8")))
    analyze_artifact(raw)
    add(relative, raw["hardware"], _hopper_cells(raw))

    return {
        "schema_version": 1,
        "evidence_label": "MEAS: CPU reanalysis of committed GPU measurements; timings unchanged",
        "inputs": inputs,
        "definitions": DEFINITIONS,
        "expansion": expansion_view(full_summary),
        "datasets": datasets,
        "shared_cells": shared_cells(views),
        "dossier": build_dossier(full_summary),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="fail unless --output is byte-identical"
    )
    args = parser.parse_args(argv)
    result = build()
    payload = strict_json_dumps(result).encode("utf-8")
    if args.check:
        # Compare raw bytes: text mode would hide CRLF or other newline rewrites.
        if not args.output.is_file() or args.output.read_bytes() != payload:
            raise SystemExit(f"{args.output} is stale; rerun scripts/diagnose_kernel_bank.py")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(
        f"Kernel-bank diagnosis: {len(result['inputs'])} input artifacts; "
        f"{len(result['datasets'])} separate matrix/profile views; "
        f"{len(result['shared_cells'])} overlapping view pairs; "
        f"{'byte-exact' if args.check else 'written'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
