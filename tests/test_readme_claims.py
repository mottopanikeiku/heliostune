"""Check that the README's action-expansion numbers match committed results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_README = (_REPO / "README.md").read_text(encoding="utf-8")
_NEXT = (_REPO / "docs/NEXT.md").read_text(encoding="utf-8")
_SUMMARY: dict[str, Any] = json.loads(
    (_REPO / "results/action-expansion-summary.json").read_text(encoding="utf-8")
)
_SHORT_MODEL = {
    "granite-3.1-8b": "Granite",
    "mistral-7b": "Mistral",
    "phi-3-mini": "Phi",
    "qwen2.5-7b": "Qwen",
}
_PROJECTION = {
    "attention-out": "attention-out",
    "attention-qkv": "attention-QKV",
    "ffn-down": "FFN-down",
    "ffn-up": "FFN-up",
}


def _table_rows(text: str, header: str) -> list[list[str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {header} |"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def test_headline_counts_match_summary() -> None:
    overall = _SUMMARY["overall"]
    new = overall["arms"]["new"]
    old = overall["arms"]["old"]
    named = overall["named_workload_count"]
    assert f"**{new['comparable_win_count']}/{named} named workloads**" in _README
    assert f"**{old['comparable_win_count']}/{named}**" in _README
    assert new["comparable_tie_count"] == 1
    assert f"one exact median tie and {new['comparable_loss_count']} losses" in _README
    assert f"**{overall['old_new_comparison']['new_faster_count']}/{named}**" in _README
    assert f"**{named} workloads / {overall['unique_shape_count']} unique shapes**" in _README


def test_bank2_table_matches_summary() -> None:
    arms = _SUMMARY["overall"]["arms"]
    rows = {row[0]: row[1:] for row in _table_rows(_README, "Bank-2 result")}
    order = ("old", "new", "union")
    assert rows["Wins against torch"] == [str(arms[a]["comparable_win_count"]) for a in order]
    assert rows["Exact ties"] == [str(arms[a]["comparable_tie_count"]) for a in order]
    assert rows["Geometric-mean latency / torch"] == [
        f"{arms[a]['geometric_mean_to_torch']:.6f}" for a in order
    ]


def test_old_new_ratio_row_count_and_cost_match_committed_files() -> None:
    ratio = _SUMMARY["overall"]["old_new_comparison"]["geometric_mean_new_to_old"]
    assert f"**new / old latency is {ratio:.6f}**" in _README
    assert f"**{(1 - ratio) * 100:.1f}% reduction**" in _README
    raw = json.loads((_REPO / "results/action-expansion-raw.json").read_text(encoding="utf-8"))
    assert f"**{len(raw['rows']):,} measurement rows**" in _README
    assert _SUMMARY["failure_row_count"] == 0
    cost = json.loads((_REPO / "results/action-expansion-cost.json").read_text(encoding="utf-8"))
    assert f"**${cost['total_cost_upper_usd']}**" in _README


def test_win_table_lists_exactly_the_summary_wins() -> None:
    workloads = {w["workload_key"]: w for w in _SUMMARY["workloads"]}
    wins = [workloads[w["workload_key"]] for w in _SUMMARY["largest_wins"]["new"]]
    assert len(wins) == _SUMMARY["overall"]["arms"]["new"]["comparable_win_count"]
    expected = sorted(
        [
            f"{_SHORT_MODEL[w['model']]} / {_PROJECTION[w['projection']]}",
            "({},{},{})".format(*w["shape"]),
            f"{w['new_ms'] * 1000:.3f} / {w['torch_ms'] * 1000:.3f}",
            f"{w['new_to_torch']:.6f}",
        ]
        for w in wins
    )
    assert sorted(_table_rows(_README, "Model / projection")) == expected


def test_win_breakdown_sentences_match_summary() -> None:
    wins = _SUMMARY["largest_wins"]["new"]
    families = [w["selected"]["family"] for w in wins]
    split_k, tile = families.count("split_k"), families.count("tile")
    assert families.count("persistent") == 0
    words = {3: "three", 5: "Five", 6: "Six"}
    assert f"{words[split_k]} wins use split-K; {words[tile]} use larger ordinary tiles" in _README
    below_one_percent = sum(w["ratio_to_torch"] > 0.99 for w in wins)
    assert f"{words[below_one_percent]} margins are below 1%" in _README
    best = min(w["ratio_to_torch"] for w in wins)
    assert f"**{(1 - best) * 100:.2f}%**" in _README
    assert f"{(1 - best) * 100:.2f}%" in _NEXT
    winning_m = {m["m"] for m in _SUMMARY["by_m"] if m["arms"]["new"]["comparable_win_count"] > 0}
    losing_m = sorted({m["m"] for m in _SUMMARY["by_m"]} - winning_m)
    assert f"at M={','.join(map(str, losing_m))}." in _README
    assert f"wins at M={','.join(map(str, losing_m))}." in _NEXT


def test_next_document_repeats_the_same_aggregates() -> None:
    overall = _SUMMARY["overall"]
    arms = overall["arms"]
    new = arms["new"]
    assert (
        f"beat torch on {new['comparable_win_count']} workloads, tie once and lose on "
        f"{new['comparable_loss_count']}" in _NEXT
    )
    assert f"loses on all {overall['named_workload_count']}" in _NEXT
    assert f"New/torch geometric-mean latency is {new['geometric_mean_to_torch']:.6f}" in _NEXT
    assert f"old/torch {arms['old']['geometric_mean_to_torch']:.6f}" in _NEXT
    ratio = overall["old_new_comparison"]["geometric_mean_new_to_old"]
    assert f"new/old is {ratio:.6f}" in _NEXT
    persistent = sum(w["selected"]["new"]["family"] == "persistent" for w in _SUMMARY["workloads"])
    assert f"Persistent TMA is selected for {persistent} workloads" in _NEXT
