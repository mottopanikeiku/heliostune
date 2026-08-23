from __future__ import annotations

from typing import Any

import pytest

import heliostune.selection as selection_module
from heliostune.selection import ParhelionCandidate, parhelion_grid, select_parhelion

EXPECTED_GRID = tuple(
    ParhelionCandidate(k, temperature, transfer_strength)
    for k in (1, 3, 8, 16)
    for temperature in (0.2, 0.7, 2.0)
    for transfer_strength in (0.0, 0.02, 0.08, 0.2)
)

EXPECTED_BASELINE_GRID = tuple(
    candidate
    for candidate in EXPECTED_GRID
    if candidate.transfer_strength == 0.0
    or (candidate.k == 1 and candidate.temperature == 0.2)
)


def test_parhelion_grid_is_the_exact_frozen_lexicographic_grid() -> None:
    assert len(EXPECTED_GRID) == 4 * 3 * 4 == 48
    assert len(EXPECTED_BASELINE_GRID) == 12 + 4 - 1 == 15
    assert parhelion_grid() == EXPECTED_GRID


def test_select_parhelion_freezes_validation_choice_and_discloses_final_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (object(),)
    sources = ("L4", "A10")
    tied_parhelion_candidates = {
        ParhelionCandidate(3, 0.7, 0.08),
        ParhelionCandidate(8, 0.2, 0.0),
    }
    tied_retrieval_candidates = {(1, 2.0), (8, 0.2)}
    tied_pooled_strengths = {0.02, 0.2}
    comparator_auc = {
        "static_multisource": 0.40,
        "torch": 0.90,
        "random": 0.10,
        "single_source_nearest": 0.50,
        "multisource_retrieval": -1.0,
        "cold_thompson": 0.90,
        "pooled_source_thompson": -1.0,
    }
    calls: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    def fake_compare_multisource(
        measurements: tuple[object, ...],
        *,
        source_gpus: tuple[str, ...],
        target_gpu: str,
        max_budget: int,
        seeds: int,
        k: int,
        temperature: float,
        transfer_strength: float,
        protocol_role: str,
        retrieval_k: int | None = None,
        retrieval_temperature: float | None = None,
        pooled_transfer_strength: float | None = None,
    ) -> dict[str, Any]:
        candidate = ParhelionCandidate(k, temperature, transfer_strength)
        effective_retrieval = (
            retrieval_k if retrieval_k is not None else k,
            retrieval_temperature if retrieval_temperature is not None else temperature,
        )
        effective_pooled_strength = (
            pooled_transfer_strength
            if pooled_transfer_strength is not None
            else transfer_strength
        )
        calls.append(
            {
                "measurements": measurements,
                "source_gpus": source_gpus,
                "target_gpu": target_gpu,
                "max_budget": max_budget,
                "seeds": seeds,
                "candidate": candidate,
                "protocol_role": protocol_role,
                "retrieval_k": retrieval_k,
                "retrieval_temperature": retrieval_temperature,
                "pooled_transfer_strength": pooled_transfer_strength,
            }
        )
        parhelion_auc = 0.95 if candidate in tied_parhelion_candidates else 0.25
        retrieval_auc = 0.85 if effective_retrieval in tied_retrieval_candidates else 0.20
        pooled_auc = 0.80 if effective_pooled_strength in tied_pooled_strengths else 0.15
        summary = {
            "auc": {
                "parhelion_thompson": parhelion_auc,
                **comparator_auc,
                "multisource_retrieval": retrieval_auc,
                "pooled_source_thompson": pooled_auc,
            },
            "fake_call": len(calls),
        }
        summaries.append(summary)
        return summary

    monkeypatch.setattr(selection_module, "compare_multisource", fake_compare_multisource)

    selection, selected_summary = select_parhelion(
        records,
        source_gpus=sources,
        target_gpu="T4",
        seeds=12,
        max_budget=8,
        jobs=1,
    )

    assert set(selection) == {
        "schema_version",
        "protocol",
        "source_gpus",
        "validation_gpu",
        "final_gpu",
        "h100_invoked",
        "selection_stages",
        "parhelion_grid",
        "baseline_grids",
        "selection_seeds",
        "final_evaluation_seeds",
        "budgets",
        "selection_metric",
        "tie_breaks",
        "comparator_candidates",
        "baseline_candidate_scores",
        "candidate_scores",
        "selected",
        "budget_one_invariant",
        "final_source_archive",
        "final_evaluation_rule",
        "disclosures",
    }
    assert selection["schema_version"] == 1
    assert selection["protocol"] == "parhelion-v2-validation-selection"
    assert selection["selection_stages"] == [
        "independent baseline grids",
        "Parhelion grid with frozen retrieval anchor and pooled baseline",
        "primary comparator selection",
    ]
    assert selection["parhelion_grid"] == {
        "k": [1, 3, 8, 16],
        "temperature": [0.2, 0.7, 2.0],
        "transfer_strength": [0.0, 0.02, 0.08, 0.2],
        "candidate_count": 48,
    }
    assert selection["baseline_grids"] == {
        "single_source_nearest": {
            "source_gpu": "L4",
            "neighbors": 1,
            "parameter_free": True,
        },
        "multisource_retrieval": {
            "k": [1, 3, 8, 16],
            "temperature": [0.2, 0.7, 2.0],
            "candidate_count": 12,
        },
        "pooled_source_thompson": {
            "transfer_strength": [0.0, 0.02, 0.08, 0.2],
            "candidate_count": 4,
        },
    }
    assert selection["selection_seeds"] == list(range(12))
    assert selection["final_evaluation_seeds"] == list(range(30))
    assert selection["budgets"] == list(range(1, 9))
    assert selection["tie_breaks"] == {
        "parhelion": "ascending (k, temperature, transfer_strength)",
        "multisource_retrieval": "ascending (k, temperature)",
        "pooled_source_thompson": "ascending transfer_strength",
        "primary_comparator": "ascending method name",
    }
    assert selection["comparator_candidates"] == list(selection_module.LEGACY_COMPARATORS)

    parhelion_winner = ParhelionCandidate(3, 0.7, 0.08)
    assert selection["selected"] == {
        "parhelion": {
            "k": parhelion_winner.k,
            "temperature": parhelion_winner.temperature,
            "transfer_strength": parhelion_winner.transfer_strength,
            "auc": 0.95,
        },
        "multisource_retrieval": {
            "k": 1,
            "temperature": 2.0,
            "auc": 0.85,
        },
        "pooled_source_thompson": {
            "transfer_strength": 0.02,
            "auc": 0.80,
        },
        "single_source_nearest": {
            "source_gpu": "L4",
            "neighbors": 1,
        },
        "primary_comparator": "cold_thompson",
        "primary_comparator_auc": 0.90,
    }
    assert selection["baseline_candidate_scores"] == [
        {
            "k": candidate.k,
            "temperature": candidate.temperature,
            "transfer_strength": candidate.transfer_strength,
            "parhelion_thompson": (
                0.95 if candidate in tied_parhelion_candidates else 0.25
            ),
            "multisource_retrieval": (
                0.85
                if (candidate.k, candidate.temperature) in tied_retrieval_candidates
                else 0.20
            ),
            "pooled_source_thompson": (
                0.80 if candidate.transfer_strength in tied_pooled_strengths else 0.15
            ),
        }
        for candidate in EXPECTED_BASELINE_GRID
    ]
    assert selection["candidate_scores"] == [
        {
            "k": candidate.k,
            "temperature": candidate.temperature,
            "transfer_strength": candidate.transfer_strength,
            "parhelion_thompson": (
                0.95 if candidate in tied_parhelion_candidates else 0.25
            ),
            "multisource_retrieval": 0.85,
            "pooled_source_thompson": 0.80,
        }
        for candidate in EXPECTED_GRID
    ]
    assert selected_summary is summaries[-1]
    assert selected_summary == {
        "auc": {
            "parhelion_thompson": 0.95,
            **comparator_auc,
            "multisource_retrieval": 0.85,
            "pooled_source_thompson": 0.80,
        },
        "fake_call": 64,
    }
    assert all(
        set(summary["auc"])
        == {"parhelion_thompson", *selection_module.LEGACY_COMPARATORS}
        for summary in summaries
    )

    assert len(calls) == 15 + 48 + 1 == 64
    baseline_calls = calls[:15]
    parhelion_calls = calls[15:63]
    assert [call["candidate"] for call in baseline_calls] == list(EXPECTED_BASELINE_GRID)
    assert [
        (
            call["retrieval_k"],
            call["retrieval_temperature"],
            call["pooled_transfer_strength"],
        )
        for call in baseline_calls
    ] == [
        (candidate.k, candidate.temperature, candidate.transfer_strength)
        for candidate in EXPECTED_BASELINE_GRID
    ]
    assert [call["candidate"] for call in parhelion_calls] == list(EXPECTED_GRID)
    assert all(call["retrieval_k"] == 1 for call in parhelion_calls)
    assert all(call["retrieval_temperature"] == 2.0 for call in parhelion_calls)
    assert all(call["pooled_transfer_strength"] == 0.02 for call in parhelion_calls)
    assert calls[-1]["candidate"] == parhelion_winner
    assert calls[-1]["retrieval_k"] == 1
    assert calls[-1]["retrieval_temperature"] == 2.0
    assert calls[-1]["pooled_transfer_strength"] == 0.02
    assert all(call["measurements"] == records for call in calls)
    assert all(call["source_gpus"] == sources for call in calls)
    assert all(call["target_gpu"] == "T4" for call in calls)
    assert all(call["max_budget"] == 8 and call["seeds"] == 12 for call in calls)
    assert all(call["protocol_role"] == "validation" for call in calls)

    assert selection["budget_one_invariant"] == (
        "Parhelion and multi-source retrieval query and recommend the same frozen retrieval "
        "anchor at budget 1; the query is charged to both methods."
    )
    assert selection["source_gpus"] == ["L4", "A10"]
    assert selection["validation_gpu"] == "T4"
    assert selection["final_source_archive"] == ["L4", "A10", "T4"]
    assert selection["final_gpu"] == "H100"
    assert selection["h100_invoked"] is False
    assert selection["final_evaluation_rule"] == (
        "Hash a freeze artifact before one H100 collection; use 30 fixed seeds, every "
        "independently selected parameter, the frozen comparator, and no rerun or grid expansion."
    )
    assert (
        "The final archive adds T4 to the two-source validation archive, changing source cost "
        "but not method logic."
        in selection["disclosures"]
    )


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
def test_select_parhelion_rejects_non_frozen_protocol_values_before_replay(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    def unexpected_replay(*args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("validation replay must not run for an invalid frozen protocol")

    monkeypatch.setattr(selection_module, "compare_multisource", unexpected_replay)

    with pytest.raises(ValueError, match=message):
        select_parhelion((), jobs=1, **overrides)
