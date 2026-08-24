from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from heliostune.artifacts import read_json, read_measurements
from heliostune.v2_addendum import build_v2_addendum_summary

_REPO = Path(__file__).resolve().parents[1]


def test_v2_addendum_reproduces_frozen_values_and_causal_contracts() -> None:
    rows = read_measurements(_REPO / "benchmarks/data/parhelion-v2-measurements.jsonl.zst")
    historical = read_json(_REPO / "benchmarks/results/parhelion-h100-final.json")

    summary = build_v2_addendum_summary(rows, historical)

    assert summary["analysis_status"] == "post_hoc_exploratory"
    assert summary["auc"]["parhelion_thompson"]["mean"] == 0.9502592348438624
    assert (
        summary["methods"]["parhelion_thompson"][-1]["mean_fraction_oracle"] == 0.9965225333288278
    )
    historical_endpoint = summary["historical_confirmatory_endpoint"]
    assert historical_endpoint["analysis_status"] == "historical_confirmatory_unchanged"
    assert historical_endpoint["evidence"]["mean_auc_delta"] == -0.6600001975593753
    assert historical_endpoint["evidence"]["comparator"] == "torch"

    assert summary["budget_one_invariant"]["verified"] is True
    assert set(summary["algorithmic_contrasts"]) == {
        "anchored_cold_thompson",
        "cold_thompson",
        "multisource_retrieval",
        "parhelion_no_forced_anchor",
    }
    for contrast in summary["algorithmic_contrasts"].values():
        assert contrast["analysis_status"] == "post_hoc_exploratory"
        assert len(contrast["by_seed_and_fold"]) == 30
        assert all(len(seed["folds"]) == 4 for seed in contrast["by_seed_and_fold"])
        for endpoint in (contrast["auc1_8"], contrast["budget8"]):
            assert endpoint["superiority_supported"] is None
            assert endpoint["claim"] is None
            assert endpoint["uncertainty"]["sampling_unit"] == "paired policy seed"
            assert endpoint["uncertainty"]["n"] == 30
            assert "conditional_on" in endpoint["uncertainty"]

    endpoints = summary["policy_seed_workload_endpoints_budget8"]
    assert len(endpoints) == 6 * 30 * 96
    counts = Counter(record["method"] for record in endpoints)
    assert set(counts.values()) == {30 * 96}
    assert all(record["bank2_evaluation_latency_ms"] > 0 for record in endpoints)
    assert all(record["tflops"] > 0 for record in endpoints)
    assert all(record["fraction_reference"] > 0 for record in endpoints)
    workload_summary = summary["workload_endpoint_summary"]
    assert workload_summary["quantile_method"] == "numpy.quantile(method='linear')"
    assert workload_summary["seed_averaging"].startswith("arithmetic mean")

    pooled = summary["methods"]["pooled_source_thompson"]
    cold = summary["methods"]["cold_thompson"]
    assert [point["mean_fraction_oracle"] for point in pooled] == [
        point["mean_fraction_oracle"] for point in cold
    ]
    assert [point["uncertainty"]["low"] for point in pooled] == pytest.approx(
        [point["uncertainty"]["low"] for point in cold]
    )
    assert "ci95_low" not in str(summary["methods"])
    assert "ci95_high" not in str(summary["methods"])
