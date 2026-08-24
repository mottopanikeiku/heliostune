from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import heliostune.selection as selection_module
from heliostune.artifacts import read_measurements
from heliostune.selection import ParhelionCandidate, parhelion_grid, select_parhelion

EXPECTED_GRID = tuple(
    ParhelionCandidate(k, temperature, transfer_strength)
    for k in (1, 3, 8, 16)
    for temperature in (0.2, 0.7, 2.0)
    for transfer_strength in (0.0, 0.02, 0.08, 0.2)
)


def test_parhelion_grid_is_the_exact_frozen_lexicographic_grid() -> None:
    assert len(EXPECTED_GRID) == 4 * 3 * 4 == 48
    assert parhelion_grid() == EXPECTED_GRID


def test_selection_prepares_once_and_uses_method_local_evaluator_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = object()
    calls = {"prepare": 0, "retrieval": 0, "pooled": 0, "parhelion": 0, "assemble": 0}
    tied_retrieval = {(1, 2.0), (8, 0.2)}
    tied_pooled = {0.02, 0.2}
    tied_parhelion = {
        ParhelionCandidate(3, 0.7, 0.08),
        ParhelionCandidate(8, 0.2, 0.0),
    }

    def fake_prepare(_context: object) -> object:
        calls["prepare"] += 1
        return prepared

    def fake_retrieval(actual: object, *, k: int, temperature: float) -> dict[str, object]:
        assert actual is prepared
        calls["retrieval"] += 1
        return {
            "method": "retrieval",
            "auc": 0.85 if (k, temperature) in tied_retrieval else 0.20,
            "k": k,
            "temperature": temperature,
        }

    def fake_pooled(actual: object, *, transfer_strength: float) -> dict[str, object]:
        assert actual is prepared
        calls["pooled"] += 1
        return {
            "method": "pooled",
            "auc": 0.80 if transfer_strength in tied_pooled else 0.15,
            "transfer_strength": transfer_strength,
        }

    def fake_parhelion(
        actual: object,
        *,
        k: int,
        temperature: float,
        transfer_strength: float,
        retrieval: object,
    ) -> dict[str, object]:
        assert actual is prepared
        assert retrieval["k"] == 1
        assert retrieval["temperature"] == 2.0
        calls["parhelion"] += 1
        candidate = ParhelionCandidate(k, temperature, transfer_strength)
        return {
            "method": "parhelion",
            "auc": 0.95 if candidate in tied_parhelion else 0.25,
            "candidate": candidate,
        }

    def fake_auc(actual: object, evaluation: dict[str, object]) -> float:
        assert actual is prepared
        return float(evaluation["auc"])

    comparator_auc = {
        "static_multisource": 0.40,
        "torch": 0.90,
        "random": 0.10,
        "single_source_nearest": 0.50,
        "cold_thompson": 0.90,
    }

    def fake_assemble(
        actual: object,
        *,
        retrieval: dict[str, object],
        pooled: dict[str, object],
        parhelion: dict[str, object],
        primary_comparator: str | None,
        **_parameters: object,
    ) -> dict[str, Any]:
        assert actual is prepared
        calls["assemble"] += 1
        return {
            "auc": {
                **comparator_auc,
                "multisource_retrieval": float(retrieval["auc"]),
                "pooled_source_thompson": float(pooled["auc"]),
                "parhelion_thompson": float(parhelion["auc"]),
            },
            "primary_comparator": primary_comparator,
        }

    monkeypatch.setattr(selection_module, "_prepare_context", fake_prepare)
    monkeypatch.setattr(selection_module, "evaluate_multisource_retrieval", fake_retrieval)
    monkeypatch.setattr(selection_module, "evaluate_pooled_source", fake_pooled)
    monkeypatch.setattr(selection_module, "evaluate_parhelion", fake_parhelion)
    monkeypatch.setattr(selection_module, "evaluation_auc", fake_auc)
    monkeypatch.setattr(selection_module, "assemble_multisource_summary", fake_assemble)

    selection, summary = select_parhelion((object(),), jobs=1)

    assert calls == {
        "prepare": 1,
        "retrieval": 12,
        "pooled": 4,
        "parhelion": 48,
        "assemble": 2,
    }
    assert selection["evaluator_counts"] == {
        "prepare_per_process": 1,
        "multisource_retrieval": 12,
        "pooled_source_thompson": 4,
        "parhelion_thompson": 48,
        "parameter_independent_baselines": 1,
    }
    assert selection["selected"] == {
        "parhelion": {
            "k": 3,
            "temperature": 0.7,
            "transfer_strength": 0.08,
            "auc": 0.95,
        },
        "multisource_retrieval": {"k": 1, "temperature": 2.0, "auc": 0.85},
        "pooled_source_thompson": {"transfer_strength": 0.02, "auc": 0.80},
        "single_source_nearest": {"source_gpu": "L4", "neighbors": 1},
        "primary_comparator": "cold_thompson",
        "primary_comparator_auc": 0.90,
    }
    assert summary["primary_comparator"] == "cold_thompson"
    assert len(selection["baseline_candidate_scores"]) == 16
    assert len(selection["candidate_scores"]) == 48


def test_frozen_v2_method_local_candidate_values_and_winners() -> None:
    repository = Path(__file__).resolve().parents[1]
    historical = json.loads(
        (repository / "benchmarks/results/parhelion-t4-selection.json").read_text(encoding="utf-8")
    )
    rows = read_measurements(repository / "benchmarks/data/parhelion-v2-measurements.jsonl.zst")

    selection, summary = select_parhelion(rows, jobs=1)

    expected_retrieval = {
        (row["k"], row["temperature"]): row["multisource_retrieval"]
        for row in historical["baseline_candidate_scores"]
        if row["transfer_strength"] == 0.0
    }
    actual_retrieval = {
        (row["k"], row["temperature"]): row["auc"]
        for row in selection["baseline_candidate_scores"]
        if row["method"] == "multisource_retrieval"
    }
    expected_pooled = {
        row["transfer_strength"]: row["pooled_source_thompson"]
        for row in historical["baseline_candidate_scores"]
        if row["k"] == 1 and row["temperature"] == 0.2
    }
    actual_pooled = {
        row["transfer_strength"]: row["auc"]
        for row in selection["baseline_candidate_scores"]
        if row["method"] == "pooled_source_thompson"
    }
    expected_parhelion = {
        (row["k"], row["temperature"], row["transfer_strength"]): row["parhelion_thompson"]
        for row in historical["candidate_scores"]
    }
    actual_parhelion = {
        (row["k"], row["temperature"], row["transfer_strength"]): row["parhelion_thompson"]
        for row in selection["candidate_scores"]
    }

    assert actual_retrieval == expected_retrieval
    assert actual_pooled == expected_pooled
    assert actual_parhelion == expected_parhelion
    assert selection["selected"] == historical["selected"]
    assert summary["auc"]["parhelion_thompson"] == pytest.approx(0.8518046329737359)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_gpus": ("A10", "L4")}, "requires source_gpus"),
        ({"target_gpu": "H100"}, "requires target_gpu"),
        ({"seeds": 11}, "requires exactly 12 seeds"),
        ({"seeds": 13}, "requires exactly 12 seeds"),
        ({"max_budget": 7}, "requires max_budget=8"),
        ({"max_budget": 9}, "requires max_budget=8"),
    ],
)
def test_select_parhelion_rejects_non_frozen_protocol_values_before_prepare(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    def unexpected_prepare(_context: object) -> object:
        raise AssertionError("validation data must not prepare for an invalid protocol")

    monkeypatch.setattr(selection_module, "_prepare_context", unexpected_prepare)
    with pytest.raises(ValueError, match=message):
        select_parhelion((), jobs=1, **overrides)
