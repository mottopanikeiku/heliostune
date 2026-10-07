"""Score frozen bank-1 action winners on bank 2, entirely on CPU.

I retain named model projections even when their matrix shapes coincide. Ratios
are candidate / same-session torch (smaller is better); failures remain unscored.
The single-device comparison and timing quantiles are descriptive, not evidence
of statistical significance or a hardware-specific cause.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any, cast

from heliostune.action_configs import ALL_CANDIDATES, candidates_for
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, Workload
from heliostune.tooling.artifacts import read_json, strict_json_dumps
from heliostune.validation import (
    exact_bool,
    exact_fields,
    exact_int,
    exact_object,
    finite_float,
    nonblank_string,
    optional_finite_float,
    optional_nonblank_string,
)

_REPO = Path(__file__).resolve().parents[1]
_ROW_FIELDS = (
    "workload_key",
    "workload",
    "bank",
    "family",
    "config_key",
    "config",
    "correct",
    "error",
    "p20_ms",
    "median_ms",
    "p80_ms",
    "max_abs_error",
    "compile_seconds",
    "timing_seconds",
)
_ARMS = ("old", "new", "union")
_NOTE = (
    "Selection uses only bank 1 (minimum valid median, then config-key tie-break); "
    "scoring uses only the fixed winners and same-workload torch from bank 2. "
    "Expanded union is selected on bank 1, never the faster bank-2 arm. "
    "Quantiles describe timing spread, not confidence intervals. This single-device "
    "session establishes neither statistical significance nor a hardware-cause claim. "
    "Named workloads are counted separately even when shapes coincide."
)


def action_manifest() -> dict[str, list[dict[str, Any]]]:
    """Return the exact frozen, CPU-safe action-set identity."""
    return {
        "old": [
            {"family": "old", "config_key": c.key, "config": c.to_dict()} for c in DEFAULT_CONFIGS
        ],
        "new": [
            {"family": c.family, "config_key": c.key, "config": c.to_dict()} for c in ALL_CANDIDATES
        ],
    }


def _valid(row: dict[str, Any]) -> bool:
    return bool(row["correct"] and row["error"] is None and row["median_ms"] is not None)


def _winner(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if _valid(row)]
    return min(valid, key=lambda row: (row["median_ms"], row["config_key"])) if valid else None


def _geomean(values: Sequence[float]) -> float | None:
    return (
        math.exp(math.fsum(math.log(value) for value in values) / len(values)) if values else None
    )


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    result = numerator / denominator
    if not math.isfinite(result) or result <= 0:
        raise ValueError("ratio must be finite and positive")
    return result


def _aggregate(workloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    comparable = [w for w in workloads if all(w[f"{arm}_to_torch"] is not None for arm in _ARMS)]
    arms: dict[str, Any] = {}
    for arm in _ARMS:
        ratios = [w[f"{arm}_to_torch"] for w in comparable]
        scored = sum(w[f"{arm}_to_torch"] is not None for w in workloads)
        arms[arm] = {
            "scored_count": scored,
            "failure_count": len(workloads) - scored,
            "comparable_win_count": sum(r < 1.0 for r in ratios),
            "comparable_tie_count": sum(r == 1.0 for r in ratios),
            "comparable_loss_count": sum(r > 1.0 for r in ratios),
            "geometric_mean_to_torch": _geomean(ratios),
        }
    return {
        "named_workload_count": len(workloads),
        "unique_shape_count": len({tuple(w["shape"]) for w in workloads}),
        "comparable_count": len(comparable),
        "noncomparable_count": len(workloads) - len(comparable),
        "arms": arms,
        "old_new_comparison": {
            "old_faster_count": sum(w["old_ms"] < w["new_ms"] for w in comparable),
            "new_faster_count": sum(w["new_ms"] < w["old_ms"] for w in comparable),
            "tie_count": sum(w["old_ms"] == w["new_ms"] for w in comparable),
            "geometric_mean_new_to_old": _geomean(
                [cast(float, _ratio(w["new_ms"], w["old_ms"])) for w in comparable]
            ),
        },
        "torch_scored_count": sum(w["torch_ms"] is not None for w in workloads),
        "torch_failure_count": sum(w["torch_ms"] is None for w in workloads),
    }


def _arm_result(row: dict[str, Any] | None, selected: dict[str, Any] | None) -> dict[str, Any]:
    if selected is None:
        return {
            "config_key": None,
            "family": None,
            "config": None,
            "status": "no_valid_bank1_candidate",
            "error": None,
            "p20_ms": None,
            "median_ms": None,
            "p80_ms": None,
        }
    if row is None:
        raise ValueError("selected bank-2 candidate is absent")
    return {
        "config_key": row["config_key"],
        "family": row["family"],
        "config": row["config"],
        "status": "scored" if _valid(row) else "bank2_failure",
        "error": row["error"]
        if row["error"] is not None
        else (None if _valid(row) else "correctness check failed"),
        "p20_ms": row["p20_ms"],
        "median_ms": row["median_ms"],
        "p80_ms": row["p80_ms"],
    }


def analyze(
    data: object,
    *,
    expected_workloads: Sequence[Workload] = DEFAULT_WORKLOADS,
) -> dict[str, Any]:
    """Validate full coverage and provenance of selection before calculating ratios.

    The default corpus is the original 96 named workloads / 84 unique shapes.
    An explicitly supplied roster supports separate pilot data and CPU fixtures;
    the CLI never infers or silently accepts a reduced full-run corpus.
    """
    raw = exact_object(data, context="action expansion artifact")
    metadata: dict[str, Any] = {
        field: raw[field]
        for field in ("pilot", "precision", "source_commit", "cloud")
        if field in raw
    }
    raw = exact_fields(
        raw,
        required=(
            "schema_version",
            "plan_commit",
            "hardware",
            "software",
            "protocol",
            "action_set",
            "rows",
            "selected",
            "duration_seconds",
            *metadata,
        ),
        context="action expansion artifact",
    )
    if "pilot" in metadata:
        exact_bool(metadata["pilot"], context="pilot")
    if "precision" in metadata:
        exact_object(metadata["precision"], context="precision")
    if "source_commit" in metadata:
        nonblank_string(metadata["source_commit"], context="source_commit")
    if "cloud" in metadata:
        cloud = exact_object(metadata["cloud"], context="cloud")
        for field in ("gpu", "note"):
            if field in cloud:
                nonblank_string(cloud[field], context=f"cloud.{field}")
        if "cpu_cores" in cloud:
            exact_int(cloud["cpu_cores"], context="cloud.cpu_cores", minimum=1)
        for field in ("host_memory_gib", "gpu_rate_usd_per_hour"):
            if field in cloud:
                finite_float(cloud[field], context=f"cloud.{field}", strictly_positive=True)
        for field in ("wall_minutes", "cost_upper_usd"):
            if field in cloud:
                finite_float(cloud[field], context=f"cloud.{field}", minimum=0)
    if exact_int(raw["schema_version"], context="schema_version") != 1:
        raise ValueError("unsupported action expansion schema_version")
    nonblank_string(raw["plan_commit"], context="plan_commit")
    for field in ("hardware", "software", "protocol"):
        exact_object(raw[field], context=field)
    finite_float(raw["duration_seconds"], context="duration_seconds", minimum=0)
    manifest = action_manifest()
    saved_manifest = exact_fields(raw["action_set"], required=("old", "new"), context="action_set")
    for arm in ("old", "new"):
        records = saved_manifest[arm]
        if type(records) is not list:
            raise ValueError(f"action_set.{arm} must be a list")
        parsed: list[dict[str, Any]] = [
            exact_fields(
                r, required=("family", "config_key", "config"), context=f"action_set.{arm}"
            )
            for r in records
        ]
        keys = [nonblank_string(r["config_key"], context="manifest config_key") for r in parsed]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate action manifest config")
        if sorted(parsed, key=lambda r: r["config_key"]) != sorted(
            manifest[arm], key=lambda r: r["config_key"]
        ):
            raise ValueError(f"action_set.{arm} differs from the frozen action manifest")
    roster = {w.key: w for w in expected_workloads}
    if not roster or len(roster) != len(expected_workloads):
        raise ValueError("expected workload roster must be nonempty and unique")
    if metadata.get("pilot") and set(roster) == {w.key for w in DEFAULT_WORKLOADS}:
        raise ValueError("pilot artifacts require an explicitly reduced expected workload roster")
    identities = {r["config_key"]: r for arm in ("old", "new") for r in manifest[arm]}
    identities["torch"] = {"family": "torch", "config_key": "torch", "config": None}
    expected_bank1 = {
        key: {"torch", *(c.key for c in DEFAULT_CONFIGS), *(c.key for c in candidates_for(w))}
        for key, w in roster.items()
    }
    if type(raw["rows"]) is not list:
        raise ValueError("rows must be a list")
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for value in raw["rows"]:
        row: dict[str, Any] = dict(
            exact_fields(value, required=_ROW_FIELDS, context="measurement row")
        )
        key = nonblank_string(row["workload_key"], context="workload_key")
        workload = Workload.from_dict(row["workload"])
        if key not in roster or workload != roster[key] or workload.key != key:
            raise ValueError("unexpected workload or inconsistent workload identity")
        bank = exact_int(row["bank"], context="bank")
        if bank not in (1, 2):
            raise ValueError("bank must be 1 or 2")
        config_key = nonblank_string(row["config_key"], context="config_key")
        if config_key not in expected_bank1[key]:
            raise ValueError("unexpected or inapplicable config")
        identity = identities[config_key]
        if row["family"] != identity["family"] or row["config"] != identity["config"]:
            raise ValueError("row config identity differs from the frozen manifest")
        row["correct"] = exact_bool(row["correct"], context="correct")
        row["error"] = optional_nonblank_string(row["error"], context="error")
        for field in ("p20_ms", "median_ms", "p80_ms"):
            row[field] = optional_finite_float(row[field], context=field, strictly_positive=True)
        for field in ("max_abs_error", "compile_seconds", "timing_seconds"):
            row[field] = optional_finite_float(row[field], context=field, minimum=0)
        times = [row[f] for f in ("p20_ms", "median_ms", "p80_ms")]
        if any(t is None for t in times) and not all(t is None for t in times):
            raise ValueError("timing quantiles must all be present or all null")
        if all(t is not None for t in times) and not times[0] <= times[1] <= times[2]:
            raise ValueError("timing quantiles must be ordered p20 <= median <= p80")
        if row["correct"] and row["error"] is None and times[1] is None:
            raise ValueError("successful row has absent timing quantiles")
        index = (key, bank, config_key)
        if index in rows:
            raise ValueError("duplicate measurement row")
        rows[index] = row
        if not _valid(row):
            failures.append(row)
    selected = exact_object(raw["selected"], context="selected")
    if set(selected) != set(roster):
        raise ValueError("selected workload coverage differs from expected roster")
    summaries: list[dict[str, Any]] = []
    for key, workload in sorted(roster.items(), key=lambda pair: (pair[1].m, pair[0])):
        bank1 = [r for (w, b, _), r in rows.items() if w == key and b == 1]
        if {r["config_key"] for r in bank1} != expected_bank1[key]:
            raise ValueError(f"bank-1 workload/config coverage is incomplete for {key}")
        choices = exact_fields(selected[key], required=("old", "new"), context="selected workload")
        old = _winner([r for r in bank1 if r["family"] == "old"])
        new = _winner([r for r in bank1 if r["family"] in {"tile", "split_k", "persistent"}])
        winners = {"old": old, "new": new}
        for arm, winner in winners.items():
            expected = winner["config_key"] if winner is not None else None
            if choices[arm] != expected:
                raise ValueError(f"selected {arm} differs from bank-1 minimum for {key}")
        bank2 = {c: r for (w, b, c), r in rows.items() if w == key and b == 2}
        expected_bank2 = {"torch", *(r["config_key"] for r in winners.values() if r is not None)}
        if set(bank2) != expected_bank2:
            raise ValueError(
                f"bank-2 coverage must contain only fixed bank-1 winners and torch for {key}"
            )
        union = _winner([r for r in winners.values() if r is not None])
        results = {
            arm: _arm_result(bank2.get(r["config_key"]) if r is not None else None, r)
            for arm, r in {**winners, "union": union}.items()
        }
        torch_row = bank2["torch"]
        results["torch"] = _arm_result(torch_row, torch_row)
        torch_ms = torch_row["median_ms"] if _valid(torch_row) else None
        item: dict[str, Any] = {
            "workload_key": key,
            "workload": workload.to_dict(),
            "shape": [workload.m, workload.n, workload.k],
            "model": workload.model,
            "projection": workload.projection,
            "regime": workload.regime,
            "label": f"{len(summaries) + 1:03d}",
            "torch_ms": torch_ms,
            "selected": {
                arm: {"config_key": r["config_key"], "family": r["family"]}
                for arm, r in results.items()
                if arm != "torch"
            },
            "bank1_selected_medians_ms": {
                arm: r["median_ms"] if r is not None else None
                for arm, r in {**winners, "union": union}.items()
            },
            "bank2": results,
        }
        for arm in _ARMS:
            result = results[arm]
            latency = result["median_ms"] if result["status"] == "scored" else None
            item[f"{arm}_ms"] = latency
            item[f"{arm}_to_torch"] = _ratio(latency, torch_ms)
        summaries.append(item)
    overall = _aggregate(summaries)
    comparable = [w for w in summaries if all(w[f"{a}_to_torch"] is not None for a in _ARMS)]
    largest = {}
    for arm in _ARMS:
        wins = sorted(
            (w for w in comparable if w[f"{arm}_to_torch"] < 1.0),
            key=lambda w: (w[f"{arm}_to_torch"], w["workload_key"]),
        )[:10]
        largest[arm] = [
            {
                "workload_key": w["workload_key"],
                "label": w["label"],
                "ratio_to_torch": w[f"{arm}_to_torch"],
                "selected": w["selected"][arm],
            }
            for w in wins
        ]
    return {
        **metadata,
        "schema_version": 1,
        "plan_commit": raw["plan_commit"],
        "hardware": raw["hardware"],
        "software": raw["software"],
        "protocol": raw["protocol"],
        "action_set": manifest,
        "duration_seconds": raw["duration_seconds"],
        "selection_bank": 1,
        "scoring_bank": 2,
        "ratio_definition": "candidate bank-2 median / same-workload torch bank-2 median",
        "aggregation_population": "intersection of finite, scored old/new/union-to-torch pairs",
        "interpretation": _NOTE,
        "overall": overall,
        "by_m": [
            {"m": m, **_aggregate([w for w in summaries if w["shape"][0] == m])}
            for m in sorted({w["shape"][0] for w in summaries})
        ],
        "largest_wins": largest,
        "workloads": summaries,
        "failure_row_count": len(failures),
        "failures": sorted(failures, key=lambda r: (r["workload_key"], r["bank"], r["config_key"])),
        "figure_labels": {w["label"]: w["workload_key"] for w in summaries},
    }


def render_svg(
    summary: dict[str, Any], *, result_href: str = "action-expansion-summary.json"
) -> str:
    """Render deterministic horizontal ratios, grouped by M with explicit failures."""
    workloads = summary["workloads"]
    ratios = [
        w[f"{a}_to_torch"]
        for w in workloads
        for a in ("old", "new")
        if w[f"{a}_to_torch"] is not None
    ]
    low = min(-1, math.floor(math.log2(min([1.0, *ratios]))))
    high = max(1, math.ceil(math.log2(max([1.0, *ratios]))))
    left, right, width = 700, 1140, 1320

    def x(ratio: float) -> float:
        return left + (math.log2(ratio) - low) / (high - low) * (right - left)

    groups = len(summary["by_m"])
    height = 160 + len(workloads) * 28 + groups * 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Action expansion: fixed bank-1 winners scored on bank 2</title>',
        '<desc id="desc">Horizontal log-scale candidate-to-torch ratios. Blue circles show old actions; orange squares show new actions. Lower is faster. Missing ratios are marked unscored. Labels map to named workloads in the linked summary.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#17212b">',
        '<text x="20" y="28" font-size="19">Action expansion — bank-1 selection / bank-2 scoring</text>',
        '<text x="20" y="50">Candidate / same-session torch; smaller is better. Named workloads retained; no bank-2 oracle selection.</text>',
        '<circle cx="26" cy="72" r="4" fill="#1261a0"/><text x="38" y="76">Old 36-action winner</text>',
        '<rect x="235" y="68" width="8" height="8" fill="#b54c00"/><text x="251" y="76">New-action winner</text>',
        f'<a href="{escape(result_href, quote=True)}"><text x="470" y="76" fill="#1261a0" text-decoration="underline">JSON results / full label mapping</text></a>',
        '<text x="20" y="97">Descriptive single-device timing; quantile spread is in JSON, not statistical uncertainty. Union winner and failures are in JSON.</text>',
    ]
    tick_step = max(1, math.ceil((high - low) / 8))
    ticks = sorted({low, high, 0, *range(low, high + 1, tick_step)})
    for tick in ticks:
        position = left + (tick - low) / (high - low) * (right - left)
        color = "#444" if tick == 0 else "#ddd"
        parts.append(
            f'<line x1="{position:.2f}" y1="125" x2="{position:.2f}" y2="{height - 26}" stroke="{color}"/>'
        )
        label = f"{2.0**tick:.3g}" if -1074 <= tick <= 1023 else f"2^{tick}"
        parts.append(f'<text x="{position:.2f}" y="119" text-anchor="middle">{label}×</text>')
    parts.append('<text x="1215" y="119" text-anchor="middle">Unscored</text>')
    y, previous_m = 140, None
    for w in workloads:
        m = w["shape"][0]
        if m != previous_m:
            parts.append(f'<text x="20" y="{y}" font-weight="bold">M = {m}</text>')
            y += 30
            previous_m = m
        name = f"{w['label']}  {w['model']} / {w['projection']}  [{m}, {w['shape'][1]}, {w['shape'][2]}]"
        parts.append(f'<g id="w{w["label"]}"><title>{escape(w["workload_key"])}</title>')
        parts.append(
            f'<a href="{escape(result_href, quote=True)}"><text x="20" y="{y + 4}">{escape(name)}</text></a>'
        )
        for arm, offset, color in (("old", -4, "#1261a0"), ("new", 5, "#b54c00")):
            ratio = w[f"{arm}_to_torch"]
            cy = y + offset
            if ratio is None:
                status = w["bank2"][arm]["status"]
                if w["torch_ms"] is None:
                    status += "; torch failure"
                parts.append(
                    f'<text x="1160" y="{cy + 3}" font-size="10" fill="{color}">{arm}: unscored<title>{escape(status)}</title></text>'
                )
                continue
            cx = x(ratio)
            parts.append(
                f'<line x1="{x(1):.2f}" y1="{cy}" x2="{cx:.2f}" y2="{cy}" stroke="{color}" stroke-opacity="0.5"/>'
            )
            tooltip = escape(f"{arm}: {ratio:.6g}× torch; {w['selected'][arm]['config_key']}")
            if arm == "old":
                parts.append(
                    f'<circle cx="{cx:.2f}" cy="{cy}" r="3.5" fill="{color}"><title>{tooltip}</title></circle>'
                )
            else:
                parts.append(
                    f'<rect x="{cx - 3.5:.2f}" y="{cy - 3.5}" width="7" height="7" fill="{color}"><title>{tooltip}</title></rect>'
                )
        parts.append("</g>")
        y += 28
    parts.extend(["</g>", "</svg>"])
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_REPO / "results/action-expansion-raw.json")
    parser.add_argument(
        "--output", type=Path, default=_REPO / "results/action-expansion-summary.json"
    )
    parser.add_argument("--figure", type=Path, default=_REPO / "results/action-expansion.svg")
    args = parser.parse_args(argv)
    if len({args.input.resolve(), args.output.resolve(), args.figure.resolve()}) != 3:
        raise ValueError("input, output, and figure paths must be distinct")
    summary = analyze(read_json(args.input))
    summary["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    href = Path(os.path.relpath(args.output.resolve(), args.figure.resolve().parent)).as_posix()
    figure = render_svg(summary, result_href=href)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strict_json_dumps(summary), encoding="utf-8")
    args.figure.write_text(figure, encoding="utf-8")
    print(strict_json_dumps(summary["overall"]), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
