"""Descriptive CPU replay of stored action matrices, keeping failures and banks apart."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from heliostune.configs import Workload
from heliostune.tooling.matmul_audit import geometric_mean
from heliostune.validation import exact_int, finite_float


@dataclass(frozen=True, slots=True)
class Cell:
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


def aggregate(records: Sequence[dict[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    """Named weights vs equal shapes; never select the fastest duplicate alias.

    For a shape-weighted metric every alias of that shape must have the metric.
    Ratios use geometric means within and across shapes; us use arithmetic means
    within and across shapes. Coverage is reported for each metric separately.
    """
    shapes: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        shapes[tuple(row["shape"])].append(row)
    result: dict[str, Any] = {"named_count": len(records), "shape_count": len(shapes)}
    for metric in metrics:
        reducer = geometric_mean if metric.endswith("_ratio") else lambda v: math.fsum(v) / len(v)
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


def grouped(records: Sequence[dict[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    return {
        "overall": aggregate(records, metrics),
        "by_regime": [
            {"regime": regime, **aggregate([r for r in records if r["regime"] == regime], metrics)}
            for regime in sorted({r["regime"] for r in records})
        ],
    }


def diagnose(cells: Sequence[Cell]) -> dict[str, Any]:
    """Compare all bank pairs, without treating a partial matrix as full coverage.

    The policy here is exhaustive selection on the earlier bank, not Thompson or
    transfer replay. Target minima are deliberately labelled in-sample oracles.
    With unequal rosters only the common-action diagnostic is identifiable.
    """
    if not cells:
        raise ValueError("empty action matrix")
    index: dict[tuple[str, int], dict[str, Cell]] = defaultdict(dict)
    workloads: dict[str, Workload] = {}
    for cell in cells:
        key = cell.workload.key
        if workloads.setdefault(key, cell.workload) != cell.workload:
            raise ValueError("inconsistent workload identity")
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
        records = []
        for key, w in sorted(workloads.items()):
            bucket = index.get((key, bank), {})
            win = _winner(list(bucket.values()))
            torch = next(iter(bucket.values())).torch_us if bucket else None
            us = win.us if win else None
            records.append({
                "workload_key": key, "shape": [w.m, w.n, w.k], "regime": w.regime,
                "oracle_config": win.config if win else None,
                "oracle_family": win.family if win else None,
                "oracle_us": us, "torch_us": torch,
                "oracle_torch_ratio": us / torch if us is not None and torch is not None else None,
                "observed_actions": len(bucket),
                "failed_actions": sum(c.us is None for c in bucket.values()),
            })
        bank_results.append({
            "bank": bank,
            "observed_cells": len(observed),
            "failure_cells": sum(c.us is None for c in observed),
            "failures_by_stage": dict(sorted(Counter(c.failure for c in observed if c.failure).items())),
            "failures_by_config": dict(sorted(Counter(c.config for c in observed if c.failure).items())),
            "failures_by_regime": dict(sorted(Counter(c.workload.regime for c in observed if c.failure).items())),
            "winner_families": dict(sorted(Counter(r["oracle_family"] for r in records if r["oracle_family"]).items())),
            **grouped(records, ("oracle_us", "torch_us", "oracle_torch_ratio")),
            "workloads": records,
        })
    pairs = []
    for selection, scoring in combinations(banks, 2):
        records = []
        for key, w in sorted(workloads.items()):
            left, right = index.get((key, selection), {}), index.get((key, scoring), {})
            common = sorted(left.keys() & right.keys())
            selected = _winner([left[c] for c in common])
            oracle = _winner([right[c] for c in common])
            evaluated = right[selected.config] if selected else None
            policy_us = evaluated.us if evaluated else None
            oracle_us = oracle.us if oracle else None
            torch = next(iter(right.values())).torch_us if right else None
            records.append({
                "workload_key": key, "shape": [w.m, w.n, w.k], "regime": w.regime,
                "same_action_roster": bool(left) and left.keys() == right.keys(),
                "selection_actions": len(left), "scoring_actions": len(right), "common_actions": len(common),
                "selected_config": selected.config if selected else None,
                "oracle_config": oracle.config if oracle else None,
                "status": "no_valid_selection" if not selected else "scoring_failure" if policy_us is None else "scored",
                "policy_us": policy_us, "oracle_us": oracle_us, "torch_us": torch,
                "policy_oracle_ratio": policy_us / oracle_us if policy_us is not None and oracle_us is not None else None,
                "regret_us": policy_us - oracle_us if policy_us is not None and oracle_us is not None else None,
                "policy_torch_ratio": policy_us / torch if policy_us is not None and torch is not None else None,
                "winner_same": int(selected.config == oracle.config) if selected and oracle else None,
                "winner_still_optimal": int(policy_us == oracle_us) if policy_us is not None and oracle_us is not None else None,
            })
        metrics = ("policy_us", "oracle_us", "torch_us", "policy_oracle_ratio", "regret_us", "policy_torch_ratio", "winner_same", "winner_still_optimal")
        full = [r for r in records if r["same_action_roster"]]
        restricted = [r for r in records if not r["same_action_roster"]]
        pairs.append({
            "selection_bank": selection, "scoring_bank": scoring,
            "policy": "exhaustive earlier-bank minimum; deterministic config-key tie break",
            "full_roster": grouped(full, metrics),
            "common_actions_only": grouped(restricted, metrics),
            "unscored_count": sum(r["status"] != "scored" for r in records),
            "workloads": records,
        })
    return {"banks": bank_results, "bank_pairs": pairs}


def expansion_view(summary: dict[str, Any]) -> dict[str, Any]:
    records = []
    for w in summary["workloads"]:
        record = {"workload_key": w["workload_key"], "shape": w["shape"], "regime": w["regime"]}
        for arm in ("old", "new", "union", "torch"):
            ms = w[f"{arm}_ms"]
            record[f"{arm}_us"] = ms * 1000 if ms is not None else None
            if arm != "torch":
                record[f"{arm}_torch_ratio"] = w[f"{arm}_to_torch"]
        records.append(record)
    metrics = ("old_us", "new_us", "union_us", "torch_us", "old_torch_ratio", "new_torch_ratio", "union_torch_ratio")
    return {
        **grouped(records, metrics),
        "new_winner_families": dict(sorted(Counter(w["selected"]["new"]["family"] for w in summary["workloads"] if w["selected"]["new"]["family"]).items())),
        "failure_row_count": summary["failure_row_count"],
        "full_bank2_oracle": None,
        "full_winner_stability": None,
        "missing_reason": "Bank 2 measures only the fixed old/new finalists and torch, not the entire action set. Full-bank oracle regret and per-family winner stability are unidentifiable. The common-action bank-pair view compares only these finalists.",
        "workloads": records,
    }
