"""Regenerate the CPU-only diagnosis of every stored matmul action matrix."""

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
from heliostune.tooling.artifacts import read_json, read_measurements, strict_json_dumps, strict_json_loads
from heliostune.tooling.bank_diagnosis import Cell, diagnose, expansion_view
from heliostune.tooling.kernel_dossier import build_dossier

_REPO = Path(__file__).resolve().parents[1]


def _compact_diagnosis(cells: list[Cell]) -> dict[str, Any]:
    result = diagnose(cells)
    # Per-regime sufficient statistics retain coverage; avoid repeating aliases
    # and combined-archive rows in a large second copy of the raw measurements.
    for row in (*result["banks"], *result["bank_pairs"]):
        del row["workloads"]
    return result


def build(repo: Path = _REPO) -> dict[str, Any]:
    datasets = []
    inputs = []

    def bind(path: Path) -> str:
        relative = path.relative_to(repo).as_posix()
        inputs.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return relative

    # Includes original acquisitions, combined archives, failed/pruned candidate
    # bank, raw mixed-hardware validation, canonical validation and pilot retry.
    for path in sorted((repo / "benchmarks/data").glob("*.jsonl.zst")):
        relative = bind(path)
        profiles: dict[str, list[Cell]] = defaultdict(list)
        for row in read_measurements(path):
            profile = strict_json_dumps(row.hardware.to_dict(), compact=True)
            profiles[profile].append(Cell(
                row.workload, row.bank, row.config.key, "tile",
                row.latency_ms * 1000 if row.usable and row.latency_ms is not None else None,
                row.torch_latency_ms * 1000,
                row.failure_stage if not row.usable else None,
            ))
        for profile, cells in sorted(profiles.items()):
            datasets.append({"path": relative, "hardware": strict_json_loads(profile), **_compact_diagnosis(cells)})

    expansion = None
    for name in ("action-expansion-raw.json", "action-expansion-pilot.json"):
        path = repo / "results" / name
        relative = bind(path)
        raw = cast(dict[str, Any], read_json(path))
        pilot = name.endswith("pilot.json")
        roster = tuple(w for w in DEFAULT_WORKLOADS if w.model == "qwen2.5-7b" and w.projection == "ffn-down" and w.m in {1, 7, 31, 1024}) if pilot else DEFAULT_WORKLOADS
        summary = analyze(raw, expected_workloads=roster)
        torch = {(r["workload_key"], r["bank"]): r["median_ms"] if r["correct"] and r["error"] is None else None for r in raw["rows"] if r["family"] == "torch"}
        cells = []
        for r in raw["rows"]:
            if r["family"] == "torch":
                continue
            valid = r["correct"] and r["error"] is None and r["median_ms"] is not None
            t = torch[r["workload_key"], r["bank"]]
            cells.append(Cell(Workload.from_dict(r["workload"]), r["bank"], r["config_key"], r["family"], r["median_ms"] * 1000 if valid else None, t * 1000 if t is not None else None, None if valid else r["error"] or "correctness"))
        view = expansion_view(summary)
        datasets.append({"path": relative, "hardware": raw["hardware"], "pilot": pilot, **_compact_diagnosis(cells), "heldout_expansion": view})
        if not pilot:
            expansion = view
            sms = raw["hardware"]["multiprocessor_count"]

    path = repo / "benchmarks/data/hopper-h100-engineering.json.zst"
    relative = bind(path)
    with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as decoded:
        raw = cast(dict[str, Any], strict_json_loads(decoded.read().decode("utf-8")))
    analyze_artifact(raw)
    cells = [Cell(Workload.from_dict(r["workload"]), r["bank"], r["config_key"], r["config_kind"], r["latency"]["median_ms"] * 1000 if r["correct"] else None, r["torch"]["median_ms"] * 1000, None if r["correct"] else "correctness") for r in raw["rows"]]
    datasets.append({"path": relative, "hardware": raw["hardware"], **_compact_diagnosis(cells)})
    assert expansion is not None
    return {
        "schema_version": 1,
        "evidence_label": "MEAS: CPU reanalysis of committed GPU measurements; timings unchanged",
        "inputs": inputs,
        "definitions": {
            "scope": "All benchmarks/data/*.jsonl.zst action matrices, both action-expansion artifacts, and the single-bank Hopper engineering screen. Each file and exact hardware profile stays separate.",
            "overlap": "Combined archives, raw/canonical validation and component acquisitions overlap. They are alternative views, never independent replicates; do not sum rows or pool their statistics.",
            "policy": "Exhaustive earlier-bank selection, then fixed-action later-bank scoring. This is a descriptive selector diagnosis, not a rerun of Thompson/transfer policies or a probe-budget result.",
            "oracle": "Retrospective minimum valid later-bank median over the compared actions; optimistic in-sample diagnostic, not hardware optimum. Failed candidates cannot win.",
            "regret": "policy/oracle ratio (1 is best); absolute regret = policy_us-oracle_us. Separate from policy/torch ratio.",
            "weighting": "Named: equal workload weight. Unique shape: equal (M,N,K) weight, averaging all aliases within shape first. Ratios use geometric means; microseconds and indicator rates use arithmetic means. A shape is covered only if every alias has the metric.",
            "stability": "winner_same compares config-key-tiebroken winners; winner_still_optimal also accepts exact tied target medians. Descriptive agreement, not independent-session reliability.",
            "coverage": "Observed and failed cells by bank/config/regime. Missing cells are not successes. Unequal bank action rosters appear only in common_actions_only, never full_roster. No bank-pair claim for single-bank artifacts.",
            "non_action_artifacts": "Precision probe measures fixed preselected configs under arithmetic settings, not an action bank; fusion/RMSNorm artifacts are different operators. Their existing reports remain linked in the historical archive.",
        },
        "expansion": expansion,
        "datasets": datasets,
        "dossier": build_dossier(sms=sms),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_REPO / "results/kernel-bank-diagnosis.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    result = build()
    rendered = strict_json_dumps(result)
    if args.check:
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("kernel-bank diagnosis is stale")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(f"Kernel-bank diagnosis: {len(result['inputs'])} input artifacts; {len(result['datasets'])} separate matrix/profile views; {'byte-exact' if args.check else 'written'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
