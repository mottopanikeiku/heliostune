"""Keep the README's result tables and linked figure tied to the measurements."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_README = (_REPO / "README.md").read_text(encoding="utf-8")
_SUMMARY: dict[str, Any] = json.loads(
    (_REPO / "results/action-expansion-summary.json").read_text(encoding="utf-8")
)
_DIAGNOSIS: dict[str, Any] = json.loads(
    (_REPO / "results/kernel-bank-diagnosis.json").read_text(encoding="utf-8")
)


def _table_rows(header: str) -> list[list[str]]:
    lines = _README.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {header} |"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def test_original_four_gpu_table_matches_separately_scored_results() -> None:
    sources = (
        ("T4", "benchmarks/data/t4-measurements.jsonl.zst", "Tesla T4"),
        ("A10", "benchmarks/data/measurements.jsonl.zst", "NVIDIA A10"),
        ("L4", "benchmarks/data/measurements.jsonl.zst", "NVIDIA L4"),
        ("H100", "benchmarks/data/h100-measurements.jsonl.zst", "NVIDIA H100 80GB HBM3"),
    )
    expected = []
    for label, path, hardware in sources:
        (dataset,) = [
            d
            for d in _DIAGNOSIS["datasets"]
            if d["path"] == path and d["hardware"]["device_name"] == hardware
        ]
        (pair,) = [
            p for p in dataset["bank_pairs"] if p["selection_bank"] == 1 and p["scoring_bank"] == 2
        ]
        overall = pair["full_roster"]["overall"]
        assert (overall["named_count"], overall["shape_count"]) == (96, 84)
        overhead = overall["policy_oracle_ratio"]["named"] - 1
        assert overhead < 0.007
        ratio = overall["policy_torch_ratio"]["named"]
        if label in {"A10", "L4"}:
            assert abs(ratio - 1) < 0.03
        expected.append([label, f"{ratio:.3f}×", f"{overhead * 100:.2f}%"])
    assert _table_rows("GPU") == expected


def test_h100_expansion_table_matches_summary() -> None:
    arms = _SUMMARY["overall"]["arms"]
    labels = (("old", "Original 36"), ("new", "Ten new configurations"), ("union", "Combined set"))
    expected = [
        [
            label,
            f"{arms[key]['geometric_mean_to_torch']:.3f}×",
            str(arms[key]["comparable_win_count"]),
        ]
        for key, label in labels
    ]
    assert _table_rows("H100 kernel set") == expected
    comparison = _SUMMARY["overall"]["old_new_comparison"]
    assert f"{(1 - comparison['geometric_mean_new_to_old']) * 100:.0f}%" in _README
    assert f"{comparison['new_faster_count']} of 96" in _README


def test_largest_h100_win_matches_measured_latencies() -> None:
    wins = _SUMMARY["largest_wins"]["new"]
    (best,) = [
        w
        for w in _SUMMARY["workloads"]
        if w["workload_key"] == min(wins, key=lambda w: w["ratio_to_torch"])["workload_key"]
    ]
    assert best["shape"] == [1, 3584, 18944]
    assert f"{(1 - best['new_to_torch']) * 100:.2f}%" in _README
    assert f"{best['new_ms'] * 1000:.3f} µs" in _README
    assert f"{best['torch_ms'] * 1000:.3f} µs" in _README
    families = [w["selected"]["family"] for w in wins]
    assert families.count("split_k") == 6
    assert families.count("tile") == 3
    assert families.count("persistent") == 0
    assert sum(w["ratio_to_torch"] > 0.99 for w in wins) == 5


def test_eight_probe_result_matches_h100_replay() -> None:
    result = json.loads((_REPO / "benchmarks/results/parhelion-h100-final.json").read_text())
    (point,) = [p for p in result["methods"]["cold_thompson"] if p["budget"] == 8]
    assert f"{point['mean_fraction_oracle'] * 100:.1f}%" in _README
    assert result["hyperparameters"]["parhelion"]["transfer_strength"] == 0


def test_readme_relative_links_exist() -> None:
    links = re.findall(r"\]\(([^)]+)\)", _README)
    for link in links:
        if "://" not in link and not link.startswith("#"):
            assert (_REPO / link.split("#", 1)[0]).is_file(), link
    assert "site/h100-by-m.svg" in links
