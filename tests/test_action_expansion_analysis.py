from __future__ import annotations

import builtins
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from xml.etree import ElementTree

import pytest

from heliostune.action_configs import candidates_for
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, Workload

_REPO = Path(__file__).resolve().parents[1]
_PATH = _REPO / "scripts/analyze_action_expansion.py"
_WORKLOADS = (
    Workload(7, 64, 32, "model-a", "attention-out", "decode-7"),
    Workload(7, 64, 32, "model-b", "attention-out", "decode-7"),
)


def _load_analyzer(name: str = "_test_action_expansion_analysis") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ANALYZER = _load_analyzer()


def _row(
    workload: Workload,
    bank: int,
    family: str,
    key: str,
    config: dict[str, Any] | None,
    median: float,
) -> dict[str, Any]:
    return {
        "workload_key": workload.key,
        "workload": workload.to_dict(),
        "bank": bank,
        "family": family,
        "config_key": key,
        "config": config,
        "correct": True,
        "error": None,
        "p20_ms": median * 0.9,
        "median_ms": median,
        "p80_ms": median * 1.1,
        "max_abs_error": 0.001,
        "compile_seconds": 0.0,
        "timing_seconds": 0.125,
    }


def _artifact(workloads: tuple[Workload, ...] = _WORKLOADS) -> dict[str, Any]:
    rows = []
    selected = {}
    for w in workloads:
        old = sorted(DEFAULT_CONFIGS, key=lambda c: c.key)
        new = sorted(candidates_for(w), key=lambda c: c.key)
        for index, c in enumerate(old):
            rows.append(_row(w, 1, "old", c.key, c.to_dict(), 1.0 + index))
        for index, candidate in enumerate(new):
            rows.append(
                _row(w, 1, candidate.family, candidate.key, candidate.to_dict(), 2.0 + index)
            )
        rows.append(_row(w, 1, "torch", "torch", None, 3.0))
        rows.append(_row(w, 2, "old", old[0].key, old[0].to_dict(), 1.0))
        rows.append(_row(w, 2, new[0].family, new[0].key, new[0].to_dict(), 1.5))
        rows.append(_row(w, 2, "torch", "torch", None, 2.0))
        selected[w.key] = {"old": old[0].key, "new": new[0].key}
    return {
        "schema_version": 1,
        "plan_commit": "a" * 40,
        "hardware": {"device_name": "NVIDIA H100"},
        "software": {"torch": "2.8.0", "triton": "3.4.0"},
        "protocol": {"selection_bank": 1, "scoring_bank": 2},
        "action_set": _ANALYZER.action_manifest(),
        "rows": rows,
        "selected": selected,
        "duration_seconds": 12.0,
    }


def _analyze(data: dict[str, Any]) -> dict[str, Any]:
    return _ANALYZER.analyze(data, expected_workloads=_WORKLOADS)


def _find(data: dict[str, Any], workload: Workload, bank: int, arm: str) -> dict[str, Any]:
    key = "torch" if arm == "torch" else data["selected"][workload.key][arm]
    return next(
        r
        for r in data["rows"]
        if r["workload_key"] == workload.key and r["bank"] == bank and r["config_key"] == key
    )


def _set_timing(row: dict[str, Any], median: float) -> None:
    row.update(p20_ms=median * 0.9, median_ms=median, p80_ms=median * 1.1)


def _fail(row: dict[str, Any], error: str = "candidate failed") -> None:
    row.update(correct=False, error=error, p20_ms=None, median_ms=None, p80_ms=None)


def test_ratios_counts_named_duplicates_and_descriptive_quantiles() -> None:
    summary = _analyze(_artifact())
    assert summary["overall"]["named_workload_count"] == 2
    assert summary["overall"]["unique_shape_count"] == 1
    assert summary["overall"]["comparable_count"] == 2
    assert summary["by_m"][0]["m"] == 7
    assert summary["by_m"][0]["named_workload_count"] == 2
    for w in summary["workloads"]:
        assert w["shape"] == [7, 64, 32]
        assert w["projection"] == "attention-out"
        assert w["torch_ms"] == 2.0
        assert w["old_to_torch"] == 0.5
        assert w["new_to_torch"] == 0.75
        assert w["union_to_torch"] == 0.5
        assert w["selected"]["old"]["family"] == "old"
        assert w["selected"]["new"]["family"] in {"tile", "persistent", "split_k"}
        assert w["bank2"]["old"]["p20_ms"] == 0.9
    for arm in ("old", "new", "union"):
        counts = summary["overall"]["arms"][arm]
        assert counts["scored_count"] == counts["comparable_win_count"] == 2
        assert counts["failure_count"] == counts["comparable_tie_count"] == 0
    assert summary["overall"]["arms"]["old"]["geometric_mean_to_torch"] == 0.5
    assert summary["overall"]["old_new_comparison"]["geometric_mean_new_to_old"] == 1.5
    assert len(summary["largest_wins"]["new"]) == 2
    assert "not confidence intervals" in summary["interpretation"]
    assert "neither statistical significance" in summary["interpretation"]


def test_union_is_bank1_winner_not_bank2_oracle() -> None:
    data = _artifact()
    _set_timing(_find(data, _WORKLOADS[0], 2, "new"), 0.1)
    summary = _analyze(data)
    w = summary["workloads"][0]
    assert w["new_to_torch"] == 0.05
    assert w["union_ms"] == w["old_ms"] == 1.0
    assert w["selected"]["union"] == w["selected"]["old"]
    assert w["bank1_selected_medians_ms"] == {"old": 1.0, "new": 2.0, "union": 1.0}
    _set_timing(_find(data, _WORKLOADS[0], 1, "new"), 0.5)
    w = _analyze(data)["workloads"][0]
    assert w["selected"]["union"] == w["selected"]["new"]
    assert w["union_ms"] == 0.1


def test_bad_bank2_selection_is_rejected() -> None:
    data = _artifact()
    old = sorted(DEFAULT_CONFIGS, key=lambda c: c.key)
    selected_row = _find(data, _WORKLOADS[0], 2, "old")
    selected_row.update(config_key=old[1].key, config=old[1].to_dict())
    _set_timing(selected_row, 0.01)
    data["selected"][_WORKLOADS[0].key]["old"] = old[1].key
    with pytest.raises(ValueError, match="bank-1 minimum"):
        _analyze(data)


def test_bank1_ties_use_config_key_not_row_order() -> None:
    data = _artifact()
    w = _WORKLOADS[0]
    old_rows = [
        r
        for r in data["rows"]
        if r["workload_key"] == w.key and r["bank"] == 1 and r["family"] == "old"
    ]
    _set_timing(old_rows[1], 1.0)
    _set_timing(_find(data, w, 1, "new"), 1.0)
    data["rows"].reverse()
    summary = _analyze(data)
    result = summary["workloads"][0]
    assert result["selected"]["old"]["config_key"] == min(r["config_key"] for r in old_rows)
    assert result["selected"]["union"]["config_key"] == min(data["selected"][w.key].values())
    assert summary == _analyze({**data, "rows": list(reversed(data["rows"]))})


def test_bank2_failure_is_visible_and_never_replaced_by_scored_arm() -> None:
    data = _artifact()
    _fail(_find(data, _WORKLOADS[0], 2, "old"), "compile failed")
    summary = _analyze(data)
    w = summary["workloads"][0]
    assert w["old_ms"] is w["old_to_torch"] is w["union_to_torch"] is None
    assert w["new_to_torch"] == 0.75
    assert w["bank2"]["old"]["status"] == "bank2_failure"
    assert w["bank2"]["union"]["error"] == "compile failed"
    assert summary["failure_row_count"] == 1
    assert summary["failures"][0]["error"] == "compile failed"
    overall = summary["overall"]
    assert overall["comparable_count"] == 1
    assert overall["arms"]["new"]["scored_count"] == 2
    assert overall["arms"]["new"]["comparable_win_count"] == 1
    assert overall["arms"]["old"]["failure_count"] == 1
    assert overall["arms"]["old"]["geometric_mean_to_torch"] == 0.5


def test_torch_failure_blocks_all_ratios_but_preserves_candidate_medians() -> None:
    data = _artifact()
    _fail(_find(data, _WORKLOADS[0], 2, "torch"), "torch failed")
    summary = _analyze(data)
    w = summary["workloads"][0]
    assert w["torch_ms"] is None
    assert w["old_ms"] == 1.0 and w["new_ms"] == 1.5
    assert all(w[f"{arm}_to_torch"] is None for arm in ("old", "new", "union"))
    assert summary["overall"]["torch_failure_count"] == 1


def test_no_valid_bank1_candidates_are_explicit_unscored_failures() -> None:
    data = _artifact()
    for row in data["rows"]:
        if row["bank"] == 1 and row["family"] == "old":
            _fail(row)
    data["rows"] = [r for r in data["rows"] if not (r["bank"] == 2 and r["family"] == "old")]
    for choices in data["selected"].values():
        choices["old"] = None
    summary = _analyze(data)
    assert summary["overall"]["comparable_count"] == 0
    assert summary["overall"]["arms"]["old"]["failure_count"] == 2
    assert summary["overall"]["arms"]["new"]["geometric_mean_to_torch"] is None
    assert summary["overall"]["arms"]["new"]["comparable_win_count"] == 0
    for w in summary["workloads"]:
        assert w["bank2"]["old"]["status"] == "no_valid_bank1_candidate"
        assert w["selected"]["union"] == w["selected"]["new"]
        assert w["union_to_torch"] == 0.75
    assert summary["failure_row_count"] == 2 * len(DEFAULT_CONFIGS)


def test_strict_wins_ties_and_geometric_means_use_common_pairs() -> None:
    data = _artifact()
    _set_timing(_find(data, _WORKLOADS[0], 2, "old"), 2.0)
    _set_timing(_find(data, _WORKLOADS[1], 2, "old"), 8.0)
    _set_timing(_find(data, _WORKLOADS[0], 2, "new"), 2.0)
    summary = _analyze(data)
    old = summary["overall"]["arms"]["old"]
    assert old["comparable_win_count"] == 0
    assert old["comparable_tie_count"] == old["comparable_loss_count"] == 1
    assert old["geometric_mean_to_torch"] == 2.0
    assert summary["overall"]["old_new_comparison"]["tie_count"] == 1
    assert summary["largest_wins"]["old"] == []


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "missing_bank1",
        "missing_bank2",
        "extra_bank2",
        "missing_workload",
        "extra_workload",
        "missing_selection",
    ],
)
def test_rejects_duplicate_or_incomplete_coverage(kind: str) -> None:
    data = _artifact()
    if kind == "duplicate":
        data["rows"].append(copy.deepcopy(data["rows"][0]))
    elif kind == "missing_bank1":
        data["rows"].pop(0)
    elif kind == "missing_bank2":
        data["rows"].remove(_find(data, _WORKLOADS[0], 2, "old"))
    elif kind == "extra_bank2":
        extra = copy.deepcopy(data["rows"][1])
        extra["bank"] = 2
        data["rows"].append(extra)
    elif kind == "missing_workload":
        data["rows"] = [r for r in data["rows"] if r["workload_key"] != _WORKLOADS[0].key]
    elif kind == "extra_workload":
        extra = copy.deepcopy(data["rows"][0])
        extra["workload"]["model"] = "unexpected"
        extra["workload_key"] = Workload.from_dict(extra["workload"]).key
        data["rows"].append(extra)
    else:
        del data["selected"][_WORKLOADS[0].key]
    with pytest.raises(ValueError):
        _analyze(data)


@pytest.mark.parametrize("failure", ["incorrect", "error"])
def test_failed_bank1_candidate_with_fast_timing_cannot_be_selected(failure: str) -> None:
    data = _artifact()
    row = _find(data, _WORKLOADS[0], 1, "old")
    if failure == "incorrect":
        row["correct"] = False
    else:
        row["error"] = "timing failed after correctness"
    with pytest.raises(ValueError, match="bank-1 minimum"):
        _analyze(data)


def test_split_k_is_rejected_outside_predeclared_applicability() -> None:
    workload = Workload(96, 64, 32, "model-a", "attention-out", "mixed-96")
    data = _artifact((workload,))
    summary = _ANALYZER.analyze(data, expected_workloads=(workload,))
    assert summary["overall"]["named_workload_count"] == 1
    split = next(r for r in data["action_set"]["new"] if r["family"] == "split_k")
    data["rows"].append(
        _row(workload, 1, split["family"], split["config_key"], split["config"], 0.001)
    )
    with pytest.raises(ValueError, match="inapplicable"):
        _ANALYZER.analyze(data, expected_workloads=(workload,))


@pytest.mark.parametrize("field", ["p20_ms", "median_ms", "p80_ms"])
@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, True])
def test_rejects_bad_timings_even_for_failed_candidates(field: str, value: object) -> None:
    data = _artifact()
    data["rows"][1].update(correct=False, error="failure")
    data["rows"][1][field] = value
    with pytest.raises(ValueError):
        _analyze(data)


@pytest.mark.parametrize(
    "kind",
    [
        "manifest",
        "config",
        "family",
        "partial_quantiles",
        "unordered_quantiles",
        "missing_success_timing",
    ],
)
def test_rejects_inconsistent_row_or_manifest(kind: str) -> None:
    data = _artifact()
    row = data["rows"][0]
    if kind == "manifest":
        data["action_set"]["old"].pop()
    elif kind == "config":
        row["config"]["block_m"] *= 2
    elif kind == "family":
        row["family"] = "tile"
    elif kind == "partial_quantiles":
        row["p80_ms"] = None
    elif kind == "unordered_quantiles":
        row["p20_ms"] = 5.0
    else:
        row.update(p20_ms=None, median_ms=None, p80_ms=None)
    with pytest.raises(ValueError):
        _analyze(data)


def test_default_roster_requires_96_named_and_retains_84_shapes() -> None:
    with pytest.raises(ValueError, match="workload"):
        _ANALYZER.analyze(_artifact())
    summary = _ANALYZER.analyze(_artifact(DEFAULT_WORKLOADS))
    assert summary["overall"]["named_workload_count"] == 96
    assert summary["overall"]["unique_shape_count"] == 84
    assert len(summary["workloads"]) == len(summary["figure_labels"]) == 96


def test_figure_is_deterministic_valid_svg_with_failure_and_label_mapping() -> None:
    data = _artifact()
    _fail(_find(data, _WORKLOADS[0], 2, "new"))
    summary = _analyze(data)
    svg = _ANALYZER.render_svg(summary, result_href="summary.json")
    assert svg == _ANALYZER.render_svg(summary, result_href="summary.json")
    root = ElementTree.fromstring(svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert 'href="summary.json"' in svg
    assert "M = 7" in svg and "1×" in svg
    assert "new: unscored" in svg
    assert 'id="w001"' in svg and 'id="w002"' in svg
    for w in summary["workloads"]:
        assert w["model"] in svg and w["workload_key"] in svg
        assert summary["figure_labels"][w["label"]] == w["workload_key"]


def test_import_and_analysis_never_import_gpu_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"torch", "triton", "modal"}:
            raise AssertionError(f"GPU dependency imported: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = _load_analyzer("_test_action_expansion_cpu_only")
    summary = module.analyze(_artifact(), expected_workloads=_WORKLOADS)
    assert summary["overall"]["comparable_count"] == 2
    assert "<svg" in module.render_svg(summary)


def test_cli_reproduces_summary_and_figure_without_machine_local_paths(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    output = tmp_path / "output/summary.json"
    figure = tmp_path / "figure/chart.svg"
    source.write_text(json.dumps(_artifact(DEFAULT_WORKLOADS)), encoding="utf-8")
    args = ["--input", str(source), "--output", str(output), "--figure", str(figure)]
    assert _ANALYZER.main(args) == 0
    saved_json, saved_svg = output.read_bytes(), figure.read_bytes()
    assert _ANALYZER.main(args) == 0
    assert output.read_bytes() == saved_json and figure.read_bytes() == saved_svg
    summary = json.loads(saved_json)
    assert summary["overall"]["named_workload_count"] == 96
    assert len(summary["input_sha256"]) == 64
    assert str(tmp_path) not in saved_json.decode() and str(tmp_path) not in saved_svg.decode()
    assert "../output/summary.json" in saved_svg.decode()


def test_cli_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    source.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        _ANALYZER.main(
            [
                "--input",
                str(source),
                "--output",
                str(tmp_path / "out.json"),
                "--figure",
                str(tmp_path / "out.svg"),
            ]
        )


def test_timing_error_after_correctness_is_unscored_not_malformed() -> None:
    data = _artifact()
    row = _find(data, _WORKLOADS[0], 2, "old")
    row.update(
        correct=True, error="timing: launch failed", p20_ms=None, median_ms=None, p80_ms=None
    )
    summary = _analyze(data)
    assert summary["workloads"][0]["old_to_torch"] is None
    assert summary["workloads"][0]["bank2"]["old"]["error"] == "timing: launch failed"
    assert summary["failure_row_count"] == 1


def test_collector_metadata_is_preserved_and_pilot_cannot_enter_full_analysis() -> None:
    data = _artifact()
    data.update(
        pilot=True,
        source_commit="b" * 40,
        precision={
            "allow_tf32": False,
            "allow_fp16_reduced_precision_reduction": True,
            "float32_matmul_precision": "highest",
        },
    )
    summary = _analyze(data)
    for field in ("pilot", "source_commit", "precision"):
        assert summary[field] == data[field]
    with pytest.raises(ValueError, match="pilot"):
        _ANALYZER.analyze(data)
    data["pilot"] = False
    with pytest.raises(ValueError, match="workload"):
        _ANALYZER.analyze(data)


def test_cloud_estimate_is_preserved_without_changing_selection() -> None:
    data = _artifact()
    baseline = _analyze(data)
    data["cloud"] = {
        "gpu": "H100",
        "cpu_cores": 1,
        "host_memory_gib": 4,
        "wall_minutes": 7.5,
        "cost_upper_usd": 0.5,
        "gpu_rate_usd_per_hour": 3.9492,
        "note": "Conservative wall-time estimate, not a provider invoice",
        "extra_metadata": {"billing_status": "estimated"},
    }
    summary = _analyze(data)
    assert summary["cloud"] == data["cloud"]
    assert {k: v for k, v in summary.items() if k != "cloud"} == baseline


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_cores", 0),
        ("host_memory_gib", 0),
        ("wall_minutes", -1),
        ("cost_upper_usd", math.inf),
        ("gpu_rate_usd_per_hour", math.nan),
    ],
)
def test_cloud_known_scalars_are_validated(field: str, value: object) -> None:
    data = _artifact()
    data["cloud"] = {field: value}
    with pytest.raises(ValueError, match="cloud"):
        _analyze(data)
