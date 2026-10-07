from __future__ import annotations

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from heliostune.configs import KernelConfig, Workload
from heliostune.replay import BenchmarkTable
from heliostune.schema import HardwareProfile, Measurement
from heliostune.tooling.matmul_audit import (
    audit_rows,
    build_audit,
    geometric_mean,
    gpu_overview,
    render_by_m,
    render_shapes,
    summarize_rows,
)

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = (KernelConfig(16, 32, 32, 4, 3), KernelConfig(32, 32, 32, 4, 3))
_WORKLOADS = (
    Workload(1, 32, 32, "model-a", "projection", "decode"),
    Workload(7, 32, 32, "model-a", "projection", "decode"),
    Workload(1, 32, 32, "model-b<&", "projection", "decode"),
)


def _matrix(*, quantiles: bool = True) -> tuple[Measurement, ...]:
    hardware = HardwareProfile("H100", "Test GPU", (9, 0), 132, 80.0)
    rows = []
    for i, workload in enumerate(_WORKLOADS):
        for j, config in enumerate(_CONFIGS):
            for bank in (0, 1, 2):
                latency = ((100.0, 0.1), (2.0, 3.0), ((4.0, 0.5, 1.0)[i], 0.25))[bank][j]
                rows.append(
                    Measurement(
                        hardware=hardware,
                        workload=workload,
                        config=config,
                        bank=bank,
                        latency_ms=latency,
                        torch_latency_ms=1.0,
                        correct=True,
                        latency_p20_ms=0.9 * latency if quantiles else None,
                        latency_p80_ms=1.1 * latency if quantiles else None,
                        torch_latency_p20_ms=0.9 if quantiles else None,
                        torch_latency_p80_ms=1.1 if quantiles else None,
                    )
                )
    return tuple(rows)


def test_reference_selection_is_independent_of_scoring_and_observation() -> None:
    rows = audit_rows(BenchmarkTable(_matrix()), "H100")
    assert [r.reference_config for r in rows] == [_CONFIGS[0].key] * 3
    assert [r.best_bank2_config for r in rows] == [_CONFIGS[1].key] * 3
    assert [r.ratio for r in rows] == [4.0, 0.5, 1.0]
    assert [r.saved_us for r in rows] == [3000.0, -500.0, 0.0]
    original = build_audit(BenchmarkTable(_matrix()))
    poisoned = [
        replace(r, latency_ms=900.0, latency_p20_ms=800.0, latency_p80_ms=1000.0)
        if r.bank == 0
        else r
        for r in _matrix()
    ]
    assert build_audit(BenchmarkTable(poisoned)) == original


def test_reference_ties_break_by_config_key_not_input_order() -> None:
    tied = [
        replace(r, latency_ms=2.0, latency_p20_ms=1.8, latency_p80_ms=2.2) if r.bank == 1 else r
        for r in reversed(_matrix())
    ]
    assert all(
        r.reference_config == _CONFIGS[0].key for r in audit_rows(BenchmarkTable(tied), "H100")
    )
    assert build_audit(BenchmarkTable(_matrix())) == build_audit(
        BenchmarkTable(reversed(_matrix()))
    )


def test_wins_geomeans_spread_and_duplicate_shapes_have_distinct_units() -> None:
    audit = build_audit(BenchmarkTable(_matrix()))
    summary = audit["overall"]
    assert summary["workloads"] == 3  # Not 18 config/bank rows.
    assert summary["unique_shapes"] == 2
    assert summary["torch_wins_reference"] == 1
    assert summary["triton_wins_reference"] == 1
    assert summary["ties_reference"] == 1
    assert summary["torch_wins_best_bank2"] == 0
    assert summary["triton_wins_best_bank2"] == 3
    assert summary["reference_over_torch_geomean"] == pytest.approx(2 ** (1 / 3))
    assert summary["reference_over_torch_equal_fold_mean"] == pytest.approx((math.sqrt(2) + 1) / 2)
    assert summary["reference_over_torch_median"] == 1.0
    assert summary["torch_saved_us_median"] == 0.0
    assert summary["spread_comparisons_available"] == 3
    assert summary["torch_p80_below_reference_p20"] == 1
    assert audit["by_m"][0]["reference_over_torch_geomean"] == pytest.approx(2.0)
    assert audit["by_shape"][0]["workloads"] == 2
    assert audit["by_shape"][0]["torch_wins_reference"] == 1
    assert "optimistic" in audit["definitions"]["best_bank2"]
    assert "not a significance test" in audit["definitions"]["wins"]
    assert "not confidence intervals" in audit["definitions"]["spread"]


def test_missing_quantiles_are_not_reported_as_separated_spreads() -> None:
    summary = summarize_rows(audit_rows(BenchmarkTable(_matrix(quantiles=False)), "H100"))
    assert summary["spread_comparisons_available"] == 0
    assert summary["torch_p80_below_reference_p20"] == 0


def test_gpu_overview_keeps_devices_separate() -> None:
    original = _matrix()
    other = [
        replace(
            r,
            hardware=replace(r.hardware, gpu="L4"),
            latency_ms=0.25 * float(r.latency_ms),
            latency_p20_ms=None,
            latency_p80_ms=None,
        )
        for r in original
    ]
    overview = gpu_overview(BenchmarkTable((*original, *other)))
    assert [item["gpu"] for item in overview] == ["H100", "L4"]
    assert [item["workloads"] for item in overview] == [3, 3]
    assert [item["torch_wins_reference"] for item in overview] == [1, 0]
    assert [item["ties_reference"] for item in overview] == [1, 1]
    assert overview[0]["reference_over_torch_geomean"] == pytest.approx(2 ** (1 / 3))
    assert overview[1]["reference_over_torch_geomean"] == pytest.approx(0.25 * 2 ** (1 / 3))


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [math.nan], [math.inf]])
def test_geometric_mean_rejects_invalid_values(values: list[float]) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        geometric_mean(values)


def test_empty_summary_and_incomplete_or_inconsistent_matrix_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one workload"):
        summarize_rows(())
    with pytest.raises(ValueError, match="missing matrix cell"):
        BenchmarkTable(_matrix()[:-1])
    bad = list(_matrix())
    bad[0] = replace(bad[0], torch_latency_ms=1.01)
    with pytest.raises(ValueError, match="inconsistent duplicated torch timing"):
        BenchmarkTable(bad)
    no_bank2 = BenchmarkTable(r for r in _matrix() if r.bank != 2)
    with pytest.raises(ValueError, match="requires exactly banks"):
        build_audit(no_bank2)


def test_figures_are_deterministic_valid_svg_with_comparison_labels() -> None:
    audit = build_audit(BenchmarkTable(_matrix()))
    for render in (render_by_m, render_shapes):
        svg = render(audit)
        assert svg == render(audit)
        root = ET.fromstring(svg)
        assert root.attrib["role"] == "img"
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        assert root.find("{http://www.w3.org/2000/svg}desc") is not None
        assert "Bank-2" in svg or "bank-2" in svg
        assert svg.endswith("\n")
    assert "model-b&lt;&amp;" in render_by_m(audit)
    assert "2.00×" in render_shapes(audit)
    assert "0.50×" in render_shapes(audit)
    assert "torch wins" in render_shapes(audit)


@pytest.fixture(scope="module")
def script() -> ModuleType:
    path = _REPO / "scripts/audit_matmul_action_set.py"
    spec = importlib.util.spec_from_file_location("_test_audit_matmul_action_set", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed_audit(script: ModuleType) -> dict[str, Any]:
    return script.analyze(_REPO)


def test_committed_analysis_matches_json_and_figures(committed_audit: dict[str, Any]) -> None:
    stored = (_REPO / "results/action-set-audit.json").read_text(encoding="utf-8")
    encoded = json.dumps(
        committed_audit, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    assert encoded + "\n" == stored
    for name, render in (
        ("action-set-by-m.svg", render_by_m),
        ("action-set-shapes.svg", render_shapes),
    ):
        assert (_REPO / "results" / name).read_text(encoding="utf-8") == render(committed_audit)
    published = json.loads(
        (_REPO / "benchmarks/results/parhelion-h100-final.json").read_text(encoding="utf-8")
    )
    assert committed_audit["overall"]["reference_over_torch_equal_fold_mean"] == pytest.approx(
        published["auc"]["torch"], abs=1e-12
    )
    assert len(committed_audit["workloads"]) == published["workloads"]
    assert committed_audit["configs"] == published["configs"]
    assert committed_audit["published_policy_context"]["parhelion_transfer_strength"] == 0
    assert committed_audit["published_policy_context"]["pooled_transfer_strength"] == 0
    frozen = json.loads(
        (_REPO / "benchmarks/parhelion-v2-h100-freeze.json").read_text(encoding="utf-8")
    )
    contract = committed_audit["timing_contract"]
    assert contract["source"]["commit"] == frozen["collector_commit"]
    assert "FP32-input" in contract["correctness"]
    assert "timed FP16 torch comparator" in contract["correctness"]
    assert "torch median only" in contract["timing"]
    assert committed_audit["overall"]["spread_comparisons_available"] == 0
    overview = committed_audit["by_gpu"]
    assert [item["gpu"] for item in overview] == ["A10", "H100", "L4", "T4"]
    assert all(item["workloads"] == 96 and item["configs"] == 36 for item in overview)
    assert all(item["banks"] == [0, 1, 2] for item in overview)
    assert {item["collection"]["stage"] for item in overview} == {
        "earlier L4/A10",
        "T4 validation",
        "H100 final",
    }
    assert "not a pooled" in committed_audit["overview_note"]


def test_script_writes_only_new_outputs(
    script: ModuleType,
    committed_audit: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = [_REPO / item["path"] for item in committed_audit["inputs"]]
    before = [path.read_bytes() for path in paths]
    monkeypatch.setattr(script, "analyze", lambda: committed_audit)
    assert script.main(["--output-dir", str(tmp_path / "audit")]) == 0
    output = tmp_path / "audit"
    assert sorted(p.name for p in output.iterdir()) == [
        "action-set-audit.json",
        "action-set-by-m.svg",
        "action-set-shapes.svg",
    ]
    assert json.loads(capsys.readouterr().out)["overall"] == committed_audit["overall"]
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("field", ["workload_keys", "config_keys"])
def test_script_refuses_roster_changes(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    published = json.loads((_REPO / script._PUBLISHED).read_text(encoding="utf-8"))
    table = BenchmarkTable(_matrix())
    published["experiment"]["workload_keys"] = [w.key for w in table.workloads("H100")]
    published["experiment"]["config_keys"] = [c.key for c in table.configs("H100")]
    published["experiment"][field] = []
    path = tmp_path / script._PUBLISHED
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(published), encoding="utf-8")
    monkeypatch.setattr(script, "read_measurements", lambda _path: _matrix())
    with pytest.raises(ValueError, match="roster differs"):
        script.analyze(tmp_path)
