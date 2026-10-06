from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts/summarize_tuner_results.py"


@pytest.fixture
def summarizer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_test_summarize_tuner_results", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_is_deterministic_and_matches_generated_table(summarizer: ModuleType) -> None:
    first = summarizer.build_summary(_REPO)
    assert summarizer.build_summary(_REPO) == first
    assert first.endswith("\n")
    assert (_REPO / "results/tuner-comparison.md").read_text(encoding="utf-8") == first


def test_summary_extracts_every_displayed_method_from_committed_inputs(
    summarizer: ModuleType,
) -> None:
    results = summarizer.load_results(_REPO)
    summary = summarizer.build_summary(_REPO)
    h100 = results["parhelion-h100-final.json"]
    for method in summarizer._H100_METHODS:
        points = {point["budget"]: point for point in h100["methods"][method]}
        assert set(points) == set(range(1, 9))
        assert h100["auc"][method] == pytest.approx(
            sum(points[b]["mean_fraction_oracle"] for b in range(1, 9)) / 8
        )
        endpoint = points[8]
        row = (
            f"| {h100['method_labels'][method]} (`{method}`) | "
            f"{h100['auc'][method]:.6f} | {endpoint['mean_fraction_oracle']:.6f} | "
            f"[{endpoint['ci95_low']:.6f}, {endpoint['ci95_high']:.6f}] |"
        )
        assert row in summary
        curve = " | ".join(f"{points[b]['mean_fraction_oracle']:.6f}" for b in range(1, 9))
        assert f"| {h100['method_labels'][method]} | {curve} |" in summary
    for name in ("l4-to-a10.json", "a10-to-l4.json"):
        result = results[name]
        for method in summarizer._V1_METHODS:
            endpoint = next(p for p in result["methods"][method] if p["budget"] == 8)
            auc = result["primary_metrics"]["fraction_reference_auc"][method]
            row = (
                f"| {result['source_gpu']} → {result['target_gpu']} | "
                f"{result['method_labels'][method]} (`{method}`) | "
                f"{auc:.6f} | {endpoint['mean_fraction_oracle']:.6f} |"
            )
            assert row in summary
    for name in summarizer._INPUTS:
        assert f"../benchmarks/results/{name}" in summary


def test_summary_names_denominator_action_set_and_protocol_limits(
    summarizer: ModuleType,
) -> None:
    summary = summarizer.build_summary(_REPO)
    assert "bank-1-selected, bank-2-scored curated 36-config Triton reference" in summary
    assert "not the hardware optimum" in summary
    assert "reference latency / recommendation latency" in summary
    assert "evaluation-only, outside the tuner's action set" in summary
    assert "not a trapezoidal integral" in summary
    assert "separate protocol" in summary
    assert "Do not pool the rows across protocols or GPUs" in summary
    assert "does not claim a physical collection cost reduction" in summary


def test_selection_provenance_uses_real_committed_parameters(summarizer: ModuleType) -> None:
    results = summarizer.load_results(_REPO)
    h100 = results["parhelion-h100-final.json"]
    selected = results["parhelion-t4-selection.json"]["selected"]
    assert h100["primary_comparator"] == selected["primary_comparator"]
    for method, fields in (
        ("multisource_retrieval", ("k", "temperature")),
        ("parhelion", ("k", "temperature", "transfer_strength")),
    ):
        for field in fields:
            assert h100["hyperparameters"][method][field] == selected[method][field]
    assert (
        h100["hyperparameters"]["single_source_nearest"]["source_gpu"]
        == selected["single_source_nearest"]["source_gpu"]
    )
    summary = summarizer.build_summary(_REPO)
    for method in ("single_source_nearest", "multisource_retrieval", "parhelion"):
        settings = json.dumps(h100["hyperparameters"][method], sort_keys=True)
        assert f"| `{method}` | `{settings}` |" in summary


def test_values_and_budget_order_come_from_inputs_not_constants(
    summarizer: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = copy.deepcopy(summarizer.load_results(_REPO))
    h100 = results["parhelion-h100-final.json"]
    h100["auc"]["cold_thompson"] = 0.123456
    endpoint = next(p for p in h100["methods"]["cold_thompson"] if p["budget"] == 8)
    endpoint["mean_fraction_oracle"] = 0.654321
    monkeypatch.setattr(summarizer, "load_results", lambda _repo: results)
    original = summarizer.build_summary(_REPO)
    assert "| 0.123456 | 0.654321 |" in original
    for result in results.values():
        if "methods" in result:
            for points in result["methods"].values():
                points.reverse()
    assert summarizer.build_summary(_REPO) == original


def test_command_writes_only_requested_output(
    summarizer: ModuleType,
    tmp_path: Path,
) -> None:
    inputs = [_REPO / "benchmarks/results" / name for name in summarizer._INPUTS]
    before = [path.read_bytes() for path in inputs]
    output = tmp_path / "results/tuner-comparison.md"
    assert summarizer.main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == summarizer.build_summary(_REPO)
    assert [path.read_bytes() for path in inputs] == before
