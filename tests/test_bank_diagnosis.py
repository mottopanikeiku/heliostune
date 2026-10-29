from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from heliostune.configs import Workload
from heliostune.tooling.bank_diagnosis import (
    PAIR_METRICS,
    Cell,
    aggregate,
    diagnose,
    expansion_view,
    grouped,
    shared_cells,
)

_REPO = Path(__file__).resolve().parents[1]
_A1 = Workload(1, 64, 32, "a", "ffn-down", "decode-1")
_A2 = Workload(1, 64, 32, "b", "ffn-down", "decode-1")  # same shape as _A1
_SCORING_FAILURE = Workload(1, 128, 32, "c", "attention-out", "decode-1")
_UNSCORED = Workload(8, 64, 32, "a", "ffn-up", "decode-8")
_UNSELECTED = Workload(8, 128, 32, "c", "ffn-up", "decode-8")
_NO_VALID = Workload(16, 64, 32, "d", "ffn-down", "prefill-16")


def _cell(
    workload: Workload, bank: int, config: str, us: float | None, torch: float, family: str = "tile"
) -> Cell:
    return Cell(workload, bank, config, family, us, torch, None if us is not None else "compile")


def _matrix() -> list[Cell]:
    return [
        # Full roster: bank 0 picks a (10), bank 1 scores it at 11; c failed on bank 0 but
        # is the bank-1 oracle at 9.5. Failed cells never win but stay counted.
        _cell(_A1, 0, "a", 10.0, 9.0),
        _cell(_A1, 0, "b", 12.0, 9.0),
        _cell(_A1, 0, "c", None, 9.0),
        _cell(_A1, 1, "a", 11.0, 9.5),
        _cell(_A1, 1, "b", 10.5, 9.5),
        _cell(_A1, 1, "c", 9.5, 9.5, family="split_k"),
        # Unequal roster: only finalist b was scored, so this is common-action only.
        _cell(_A2, 0, "a", 5.0, 4.0),
        _cell(_A2, 0, "b", 4.0, 4.0),
        _cell(_A2, 1, "b", 5.0, 5.0),
        # The selected action fails on the scoring bank.
        _cell(_SCORING_FAILURE, 0, "a", 3.0, 2.0),
        _cell(_SCORING_FAILURE, 0, "b", 4.0, 2.0),
        _cell(_SCORING_FAILURE, 1, "a", None, 2.5),
        _cell(_SCORING_FAILURE, 1, "b", 4.5, 2.5),
        # Present in one bank only.
        _cell(_UNSCORED, 0, "a", 20.0, 10.0),
        _cell(_UNSELECTED, 1, "a", 7.0, 7.0),
        # Every selection-bank candidate failed.
        _cell(_NO_VALID, 0, "a", None, 1.0),
        _cell(_NO_VALID, 1, "a", 2.0, 1.0),
    ]


def _pair_rows(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["workload_key"]: r for r in result["bank_pairs"][0]["workloads"]}


def test_aggregate_separates_named_and_unique_shape_weights() -> None:
    records: list[dict[str, Any]] = [
        {"shape": [1, 64, 32], "x_ratio": 2.0, "y_us": 10.0},
        {"shape": [1, 64, 32], "x_ratio": 8.0, "y_us": 30.0},
        {"shape": [8, 64, 32], "x_ratio": 1.0, "y_us": 50.0},
    ]
    result = aggregate(records, ("x_ratio", "y_us"))
    assert result["named_count"] == 3 and result["shape_count"] == 2
    # Ratios: geometric means. Named (2*8*1)^(1/3); shapes first sqrt(2*8)=4, then sqrt(4*1).
    assert result["x_ratio"]["named"] == pytest.approx(16 ** (1 / 3))
    assert result["x_ratio"]["unique_shape"] == pytest.approx(2.0)
    # Microseconds: arithmetic means. Named 90/3; shapes (20 + 50) / 2.
    assert result["y_us"]["named"] == pytest.approx(30.0)
    assert result["y_us"]["unique_shape"] == pytest.approx(35.0)
    assert result["x_ratio"]["named_covered"] == 3 and result["x_ratio"]["shapes_covered"] == 2


def test_aggregate_never_keeps_the_covered_alias_of_a_partial_shape() -> None:
    records: list[dict[str, Any]] = [
        {"shape": [1, 64, 32], "x_ratio": 2.0},
        {"shape": [1, 64, 32], "x_ratio": None},
        {"shape": [8, 64, 32], "x_ratio": 1.0},
    ]
    result = aggregate(records, ("x_ratio", "absent_us"))["x_ratio"]
    assert result["named"] == pytest.approx(math.sqrt(2.0))
    assert result["named_covered"] == 2
    # The (1,64,32) shape is uncovered because one alias lacks the metric.
    assert result["unique_shape"] == 1.0 and result["shapes_covered"] == 1
    empty = aggregate(records, ("absent_us",))["absent_us"]
    assert empty == {"named": None, "unique_shape": None, "named_covered": 0, "shapes_covered": 0}


def test_grouped_orders_regimes_by_smallest_m() -> None:
    records: list[dict[str, Any]] = [
        {"shape": [1024, 8, 8], "regime": "prefill-1024", "v_us": 1.0},
        {"shape": [257, 8, 8], "regime": "prefill-257", "v_us": 2.0},
        {"shape": [1, 8, 8], "regime": "decode-1", "v_us": 3.0},
    ]
    result = grouped(records, ("v_us",))
    assert [r["regime"] for r in result["by_regime"]] == ["decode-1", "prefill-257", "prefill-1024"]
    assert result["by_regime"][1]["v_us"]["named"] == 2.0


def test_full_roster_regret_and_winner_agreement_are_hand_calculated() -> None:
    row = _pair_rows(diagnose(_matrix()))[_A1.key]
    assert row["status"] == "scored" and row["same_action_roster"]
    assert (row["selected_config"], row["oracle_config"]) == ("a", "c")
    assert row["policy_us"] == 11.0 and row["oracle_us"] == 9.5 and row["torch_us"] == 9.5
    assert row["policy_oracle_ratio"] == pytest.approx(11.0 / 9.5)
    assert row["regret_us"] == pytest.approx(1.5)
    assert row["policy_torch_ratio"] == pytest.approx(11.0 / 9.5)
    assert row["oracle_torch_ratio"] == pytest.approx(1.0)
    assert row["winner_same"] == 0 and row["winner_still_optimal"] == 0


def test_unequal_rosters_only_enter_the_common_action_view() -> None:
    result = diagnose(_matrix())
    pair = result["bank_pairs"][0]
    row = _pair_rows(result)[_A2.key]
    assert not row["same_action_roster"]
    assert (row["selection_actions"], row["scoring_actions"], row["common_actions"]) == (2, 1, 1)
    assert row["policy_oracle_ratio"] == 1.0 and row["winner_same"] == 1
    common = pair["common_actions_only"]["overall"]
    assert common["named_count"] == 1 and common["policy_us"]["named"] == 5.0
    full = pair["full_roster"]["overall"]
    # _A1, the scoring-failure workload and the no-valid-selection workload share full
    # rosters; _A2 does not.
    assert full["named_count"] == 3 and full["policy_us"]["named_covered"] == 1
    # _A1 and _A2 share a shape, but _A2 lies in the other view, so _A1 alone covers it.
    assert full["policy_oracle_ratio"]["shapes_covered"] == 1


def test_partial_coverage_and_failures_keep_explicit_statuses() -> None:
    result = diagnose(_matrix())
    rows = _pair_rows(result)
    assert rows[_SCORING_FAILURE.key]["status"] == "scoring_failure"
    assert rows[_SCORING_FAILURE.key]["policy_us"] is None
    assert rows[_SCORING_FAILURE.key]["regret_us"] is None
    assert rows[_SCORING_FAILURE.key]["oracle_us"] == 4.5
    assert rows[_UNSCORED.key]["status"] == "scoring_bank_unobserved"
    assert rows[_UNSELECTED.key]["status"] == "selection_bank_unobserved"
    assert rows[_NO_VALID.key]["status"] == "no_valid_selection"
    assert result["bank_pairs"][0]["status_counts"] == {
        "no_valid_selection": 1,
        "scored": 2,
        "scoring_bank_unobserved": 1,
        "scoring_failure": 1,
        "selection_bank_unobserved": 1,
    }
    # Unobserved workloads appear in neither aggregate.
    pair = result["bank_pairs"][0]
    assert pair["full_roster"]["overall"]["named_count"] == 3
    assert pair["common_actions_only"]["overall"]["named_count"] == 1


def test_bank_summaries_count_failures_and_unobserved_workloads() -> None:
    banks = {b["bank"]: b for b in diagnose(_matrix())["banks"]}
    zero, one = banks[0], banks[1]
    assert zero["observed_cells"] == 9 and zero["failure_cells"] == 2
    assert zero["failures_by_stage"] == {"compile": 2}
    assert zero["failures_by_config"] == {"a": 1, "c": 1}
    assert zero["failures_by_regime"] == {"decode-1": 1, "prefill-16": 1}
    assert zero["observed_workloads"] == 5
    assert one["observed_cells"] == 8 and one["failure_cells"] == 1
    assert one["observed_workloads"] == 5
    assert one["winner_families"] == {"split_k": 1, "tile": 4}
    by_key = {r["workload_key"]: r for r in one["workloads"]}
    assert by_key[_UNSCORED.key]["oracle_us"] is None
    assert by_key[_A1.key]["oracle_torch_ratio"] == pytest.approx(1.0)
    # A bank with no valid candidates contributes no oracle.
    assert {r["workload_key"]: r for r in zero["workloads"]}[_NO_VALID.key]["oracle_us"] is None


def test_exact_ties_use_config_order_and_count_as_still_optimal() -> None:
    cells = [
        _cell(_A1, 0, "b", 1.0, 1.0),
        _cell(_A1, 0, "a", 2.0, 1.0),
        _cell(_A1, 1, "a", 3.0, 1.0),
        _cell(_A1, 1, "b", 3.0, 1.0),
    ]
    row = _pair_rows(diagnose(cells))[_A1.key]
    assert (row["selected_config"], row["oracle_config"]) == ("b", "a")
    assert row["winner_same"] == 0 and row["winner_still_optimal"] == 1
    assert row["regret_us"] == 0.0


def test_single_bank_matrices_make_no_pair_claim() -> None:
    result = diagnose([_cell(_A1, 0, "a", 1.0, 1.0)])
    assert result["bank_pairs"] == [] and len(result["banks"]) == 1


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        ([], "empty action matrix"),
        ([_cell(_A1, 0, "a", 1.0, 1.0), _cell(_A1, 0, "a", 2.0, 1.0)], "duplicate action cell"),
        (
            [_cell(_A1, 0, "a", 1.0, 1.0), _cell(_A1, 0, "b", 2.0, 1.5)],
            "inconsistent duplicated torch timing",
        ),
    ],
)
def test_diagnose_rejects_ambiguous_matrices(cells: list[Cell], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        diagnose(cells)


@pytest.mark.parametrize(
    ("us", "failure"),
    [(None, None), (1.0, "compile"), (-1.0, None), (math.inf, None)],
)
def test_cells_require_a_reason_exactly_when_latency_is_missing(
    us: float | None, failure: str | None
) -> None:
    with pytest.raises(ValueError):
        Cell(_A1, 0, "a", "tile", us, 1.0, failure)


def test_shared_cells_count_identical_rows_between_views() -> None:
    shared = _cell(_A1, 0, "a", 1.0, 1.0)
    views = {
        "combined": [shared, _cell(_A2, 0, "a", 2.0, 2.0)],
        "component": [shared],
        "other": [_cell(_A1, 0, "a", 1.5, 1.0)],
    }
    assert shared_cells(views) == [{"views": ["combined", "component"], "shared_cells": 1}]


def _summary_workload(
    workload: Workload, family: str | None, old: float, new: float, torch: float
) -> dict[str, Any]:
    return {
        "workload_key": workload.key,
        "shape": [workload.m, workload.n, workload.k],
        "regime": workload.regime,
        "selected": {"new": {"family": family}},
        "old_ms": old,
        "new_ms": new,
        "union_ms": min(old, new),
        "torch_ms": torch,
        "old_to_torch": old / torch,
        "new_to_torch": new / torch,
        "union_to_torch": min(old, new) / torch,
    }


def test_expansion_view_reports_regimes_families_and_unidentifiable_fields() -> None:
    summary = {
        "failure_row_count": 0,
        "workloads": [
            _summary_workload(_A1, "split_k", 0.004, 0.001, 0.002),
            _summary_workload(_A2, "tile", 0.004, 0.002, 0.002),
            _summary_workload(_UNSCORED, "persistent", 0.004, 0.004, 0.002),
        ],
    }
    view = expansion_view(summary)
    assert view["full_bank2_oracle"] is None and view["full_winner_stability"] is None
    assert "unidentifiable" in view["missing_reason"]
    assert view["new_winner_families"] == {"persistent": 1, "split_k": 1, "tile": 1}
    decode1, decode8 = view["by_regime"]
    assert decode1["regime"] == "decode-1" and decode1["m_values"] == [1]
    assert decode1["new_vs_torch"] == {"wins": 1, "ties": 1, "losses": 0}
    assert decode8["new_vs_torch"] == {"wins": 0, "ties": 0, "losses": 1}
    # Microseconds from milliseconds; named arithmetic mean of 1, 2 and 4 us.
    assert view["overall"]["new_us"]["named"] == pytest.approx(7 / 3)
    # Unique shapes: sqrt(0.5 * 1) for the aliased shape, then with 2.
    assert view["overall"]["new_torch_ratio"]["unique_shape"] == pytest.approx(
        math.sqrt(math.sqrt(0.5) * 2)
    )


def test_pair_metrics_include_both_regret_and_bank_floor() -> None:
    assert {"policy_oracle_ratio", "oracle_torch_ratio", "regret_us"} <= set(PAIR_METRICS)


def _load_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(_REPO / "scripts"))
    name = "_test_diagnose_kernel_bank"
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts/diagnose_kernel_bank.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_diagnosis_is_reproducible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script(monkeypatch)
    output = tmp_path / "diagnosis.json"
    assert script.main(["--output", str(output)]) == 0
    assert output.read_bytes() == (_REPO / "results/kernel-bank-diagnosis.json").read_bytes()
    assert "13 input artifacts; 27 separate matrix/profile views" in capsys.readouterr().out


def test_check_mode_rejects_a_stale_diagnosis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script(monkeypatch)
    built: dict[str, list[Any]] = {"inputs": [], "datasets": [], "shared_cells": []}
    monkeypatch.setattr(script, "build", lambda: built)
    output = tmp_path / "diagnosis.json"
    output.write_text(script.strict_json_dumps(built), encoding="utf-8")
    assert script.main(["--output", str(output), "--check"]) == 0
    assert "byte-exact" in capsys.readouterr().out
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="stale"):
        script.main(["--output", str(output), "--check"])
