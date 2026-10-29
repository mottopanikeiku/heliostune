"""Descriptive CPU replay of stored action matrices, keeping failures and banks apart."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from heliostune.configs import Workload
from heliostune.tooling.matmul_audit import geometric_mean
from heliostune.validation import exact_int, finite_float

PAIR_METRICS = (
    "policy_us",
    "oracle_us",
    "torch_us",
    "policy_oracle_ratio",
    "oracle_torch_ratio",
    "regret_us",
    "policy_torch_ratio",
    "winner_same",
    "winner_still_optimal",
)
EXPANSION_METRICS = (
    "old_us",
    "new_us",
    "union_us",
    "torch_us",
    "old_torch_ratio",
    "new_torch_ratio",
    "union_torch_ratio",
)


@dataclass(frozen=True, slots=True)
class Cell:
    """One stored candidate median; a failed cell keeps its reason and no latency."""

    workload: Workload
    bank: int
    config: str
    family: str
    us: float | None
    torch_us: float | None
    failure: str | None = None

    def __post_init__(self) -> None:
        exact_int(self.bank, context="bank", minimum=0)
        for value in (self.us, self.torch_us):
            if value is not None:
                finite_float(value, context="latency us", strictly_positive=True)
        if (self.us is None) != (self.failure is not None):
            raise ValueError("failed cells must have a reason and no usable latency")


def _winner(cells: Sequence[Cell]) -> Cell | None:
    valid = [c for c in cells if c.us is not None]
    return min(valid, key=lambda c: (c.us, c.config)) if valid else None


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def aggregate(records: Sequence[dict[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    """Named weights vs equal shapes; never select the fastest duplicate alias.

    For a shape-weighted metric every alias of that shape must have the metric.
    Ratios use geometric means within and across shapes; microseconds and
    indicator rates use arithmetic means. Coverage is reported per metric.
    """
    shapes: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        shapes[tuple(row["shape"])].append(row)
    result: dict[str, Any] = {"named_count": len(records), "shape_count": len(shapes)}
    for metric in metrics:
        reducer = geometric_mean if metric.endswith("_ratio") else _mean
        named = [r[metric] for r in records if r.get(metric) is not None]
        unique = [
            reducer([r[metric] for r in aliases])
            for aliases in shapes.values()
            if all(r.get(metric) is not None for r in aliases)
        ]
        result[metric] = {
            "named": reducer(named) if named else None,
            "unique_shape": reducer(unique) if unique else None,
            "named_covered": len(named),
            "shapes_covered": len(unique),
        }
    return result


def regimes(records: Sequence[dict[str, Any]]) -> list[str]:
    """Order regimes by their smallest M, so prefill-257 precedes prefill-1024."""
    smallest: dict[str, int] = {}
    for row in records:
        smallest[row["regime"]] = min(row["shape"][0], smallest.get(row["regime"], row["shape"][0]))
    return sorted(smallest, key=lambda regime: (smallest[regime], regime))


def grouped(records: Sequence[dict[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    return {
        "overall": aggregate(records, metrics),
        "by_regime": [
            {"regime": regime, **aggregate([r for r in records if r["regime"] == regime], metrics)}
            for regime in regimes(records)
        ],
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    return numerator / denominator if numerator is not None and denominator is not None else None


def _pair_record(
    workload: Workload, left: Mapping[str, Cell], right: Mapping[str, Cell]
) -> dict[str, Any]:
    common = sorted(left.keys() & right.keys())
    selected = _winner([left[c] for c in common])
    oracle = _winner([right[c] for c in common])
    policy_us = right[selected.config].us if selected else None
    oracle_us = oracle.us if oracle else None
    torch = next(iter(right.values())).torch_us if right else None
    if not left:
        status = "selection_bank_unobserved"
    elif not right:
        status = "scoring_bank_unobserved"
    elif not common:
        status = "no_common_actions"
    elif selected is None:
        status = "no_valid_selection"
    elif policy_us is None:
        status = "scoring_failure"
    else:
        status = "scored"
    return {
        "workload_key": workload.key,
        "shape": [workload.m, workload.n, workload.k],
        "regime": workload.regime,
        "status": status,
        "same_action_roster": bool(left) and left.keys() == right.keys(),
        "selection_actions": len(left),
        "scoring_actions": len(right),
        "common_actions": len(common),
        "selected_config": selected.config if selected else None,
        "oracle_config": oracle.config if oracle else None,
        "policy_us": policy_us,
        "oracle_us": oracle_us,
        "torch_us": torch,
        "policy_oracle_ratio": _ratio(policy_us, oracle_us),
        "oracle_torch_ratio": _ratio(oracle_us, torch),
        "regret_us": policy_us - oracle_us
        if policy_us is not None and oracle_us is not None
        else None,
        "policy_torch_ratio": _ratio(policy_us, torch),
        "winner_same": int(selected.config == oracle.config) if selected and oracle else None,
        "winner_still_optimal": int(policy_us == oracle_us)
        if policy_us is not None and oracle_us is not None
        else None,
    }


def diagnose(cells: Sequence[Cell]) -> dict[str, Any]:
    """Compare all bank pairs, without treating a partial matrix as full coverage.

    The policy here is exhaustive selection on the earlier bank, not Thompson or
    transfer replay. Target minima are deliberately labelled in-sample oracles.
    With unequal rosters only the common-action diagnostic is identifiable.
    Workloads absent from either bank stay in the status counts, never a mean.
    """
    if not cells:
        raise ValueError("empty action matrix")
    index: dict[tuple[str, int], dict[str, Cell]] = defaultdict(dict)
    workloads: dict[str, Workload] = {}
    for cell in cells:
        # Workload keys encode all six workload fields, so equal keys are equal workloads.
        key = cell.workload.key
        workloads[key] = cell.workload
        bucket = index[key, cell.bank]
        if cell.config in bucket:
            raise ValueError("duplicate action cell")
        if bucket and next(iter(bucket.values())).torch_us != cell.torch_us:
            raise ValueError("inconsistent duplicated torch timing")
        bucket[cell.config] = cell
    banks = sorted({c.bank for c in cells})
    bank_results = []
    for bank in banks:
        observed = [c for c in cells if c.bank == bank]
        records: list[dict[str, Any]] = []
        for key, w in sorted(workloads.items()):
            bucket = index.get((key, bank), {})
            win = _winner(list(bucket.values()))
            torch = next(iter(bucket.values())).torch_us if bucket else None
            us = win.us if win else None
            records.append(
                {
                    "workload_key": key,
                    "shape": [w.m, w.n, w.k],
                    "regime": w.regime,
                    "oracle_config": win.config if win else None,
                    "oracle_family": win.family if win else None,
                    "oracle_us": us,
                    "torch_us": torch,
                    "oracle_torch_ratio": _ratio(us, torch),
                    "observed_actions": len(bucket),
                    "failed_actions": sum(c.us is None for c in bucket.values()),
                }
            )
        bank_results.append(
            {
                "bank": bank,
                "observed_cells": len(observed),
                "observed_workloads": sum(r["observed_actions"] > 0 for r in records),
                "failure_cells": sum(c.us is None for c in observed),
                "failures_by_stage": dict(
                    sorted(Counter(c.failure for c in observed if c.failure).items())
                ),
                "failures_by_config": dict(
                    sorted(Counter(c.config for c in observed if c.failure).items())
                ),
                "failures_by_regime": dict(
                    sorted(Counter(c.workload.regime for c in observed if c.failure).items())
                ),
                "winner_families": dict(
                    sorted(
                        Counter(r["oracle_family"] for r in records if r["oracle_family"]).items()
                    )
                ),
                **grouped(records, ("oracle_us", "torch_us", "oracle_torch_ratio")),
                "workloads": records,
            }
        )
    pairs = []
    for selection, scoring in combinations(banks, 2):
        rows = [
            _pair_record(w, index.get((key, selection), {}), index.get((key, scoring), {}))
            for key, w in sorted(workloads.items())
        ]
        both = [r for r in rows if r["selection_actions"] and r["scoring_actions"]]
        pairs.append(
            {
                "selection_bank": selection,
                "scoring_bank": scoring,
                "policy": "exhaustive earlier-bank minimum; deterministic config-key tie break",
                "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
                "full_roster": grouped([r for r in both if r["same_action_roster"]], PAIR_METRICS),
                "common_actions_only": grouped(
                    [r for r in both if not r["same_action_roster"]], PAIR_METRICS
                ),
                "workloads": rows,
            }
        )
    return {"banks": bank_results, "bank_pairs": pairs}


def shared_cells(views: Mapping[str, Sequence[Cell]]) -> list[dict[str, Any]]:
    """Count identical normalized cells between views; any overlap forbids pooling them.

    Cells compare their normalized fields (workload, bank, action, family, microsecond
    medians and failure), not the bytes of the source rows.
    """
    sets = {name: frozenset(cells) for name, cells in views.items()}
    names = sorted(sets)
    return [
        {"views": [a, b], "shared_cells": len(sets[a] & sets[b])}
        for a, b in combinations(names, 2)
        if sets[a] & sets[b]
    ]


def expansion_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Re-aggregate the held-out finalists by named workload, unique shape and regime."""
    records = []
    for w in summary["workloads"]:
        record: dict[str, Any] = {
            "workload_key": w["workload_key"],
            "shape": w["shape"],
            "regime": w["regime"],
            "new_family": w["selected"]["new"]["family"],
        }
        for arm in ("old", "new", "union", "torch"):
            ms = w[f"{arm}_ms"]
            record[f"{arm}_us"] = ms * 1000 if ms is not None else None
            if arm != "torch":
                record[f"{arm}_torch_ratio"] = w[f"{arm}_to_torch"]
        records.append(record)
    view = grouped(records, EXPANSION_METRICS)
    for row in view["by_regime"]:
        members = [r for r in records if r["regime"] == row["regime"]]
        row["m_values"] = sorted({r["shape"][0] for r in members})
        row["new_winner_families"] = dict(
            sorted(Counter(r["new_family"] for r in members if r["new_family"]).items())
        )
        ratios = [r["new_torch_ratio"] for r in members if r["new_torch_ratio"] is not None]
        row["new_vs_torch"] = {
            "wins": sum(r < 1 for r in ratios),
            "ties": sum(r == 1 for r in ratios),
            "losses": sum(r > 1 for r in ratios),
        }
    return {
        **view,
        "new_winner_families": dict(
            sorted(Counter(r["new_family"] for r in records if r["new_family"]).items())
        ),
        "failure_row_count": summary["failure_row_count"],
        "full_bank2_oracle": None,
        "full_winner_stability": None,
        "missing_reason": (
            "Bank 2 re-times only the fixed old and new finalists and torch, not the whole "
            "action set. Full-bank oracle regret and per-family winner stability are therefore "
            "unidentifiable. The common-action bank-pair view compares only these finalists."
        ),
        "workloads": records,
    }
