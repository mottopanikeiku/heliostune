from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from heliostune.artifacts import read_json
from heliostune.errors import SchemaError
from heliostune.report_model import ReportData, normalize_report_summary


def _summary() -> dict[str, object]:
    def point(mean: float) -> dict[str, object]:
        return {
            "budget": 8,
            "mean_fraction_oracle": mean,
            "uncertainty": {
                "estimand": "mean fraction of held-out reference",
                "sampling_unit": "paired policy seed",
                "n": 30,
                "conditional_on": "fixed matrix and corpus",
                "interval_method": "Student-t 95%",
                "low": mean - 0.01,
                "high": mean + 0.01,
            },
        }

    return {
        "source_gpu": "L4 + A10",
        "target_gpu": "H100",
        "max_budget": 8,
        "seeds": 30,
        "workloads": 96,
        "configs": 36,
        "methods": {
            "static_multisource": [point(0.8)],
            "parhelion_thompson": [point(0.95)],
            "torch": [point(0.9)],
            "official_triton_config_exhaustive": [point(0.99)],
            "heldout_reference": [point(1.0)],
        },
        "method_roles": {
            "static_multisource": "zero_query",
            "parhelion_thompson": "sequential",
            "torch": "external",
            "official_triton_config_exhaustive": "exhaustive",
            "heldout_reference": "reference",
        },
        "method_labels": {"parhelion_thompson": "Parhelion"},
        "transfer_method": "parhelion_thompson",
        "primary_comparator": "torch",
        "primary_metrics": {
            "paired_parhelion_vs_primary_auc_delta": {
                "comparator": "torch",
                "mean_auc_delta": 0.05,
                "ci95_low": 0.01,
                "ci95_high": 0.09,
                "paired_seeds": 30,
                "degrees_of_freedom": 29,
                "superiority_supported": True,
                "claim": "Parhelion has higher AUC.",
            }
        },
        "hardware": [
            {"gpu": "L4", "device_name": "NVIDIA L4"},
            {"gpu": "H100", "device_name": "NVIDIA H100"},
        ],
        "folds": [{"heldout_model": "model-a", "target_workloads": 24}],
        "source_cost": {"queries": 0},
        "target_collection_cost": {"queries": 768},
        "provenance": {"protocol": "frozen"},
        "limitations": ["Fixed corpus."],
    }


def test_normalized_report_is_frozen_and_role_partitioned() -> None:
    data = normalize_report_summary(_summary())

    assert isinstance(data, ReportData)
    assert data.source_label == "L4 + A10"
    assert data.max_budget == 8
    assert {method.key: method.role for method in data.methods} == {
        "static_multisource": "zero_query",
        "parhelion_thompson": "sequential",
        "torch": "external",
        "official_triton_config_exhaustive": "exhaustive",
        "heldout_reference": "reference",
    }
    assert data.primary[0].degrees_of_freedom == 29
    assert data.primary[0].uncertainty.sampling_unit == "paired policy seed"
    with pytest.raises(FrozenInstanceError):
        data.max_budget = 16
    with pytest.raises(TypeError):
        data.raw_summary["max_budget"] = 16
    assert isinstance(data.raw_summary["methods"]["parhelion_thompson"], tuple)


def test_unknown_new_method_requires_declared_role() -> None:
    summary = _summary()
    summary["methods"]["new_policy"] = summary["methods"]["parhelion_thompson"]
    with pytest.raises(SchemaError, match="must declare one role"):
        normalize_report_summary(summary)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean_auc_delta", float("nan"), "must be finite"),
        ("paired_seeds", 0, "positive integer"),
        ("degrees_of_freedom", 28, "paired_seeds - 1"),
        ("superiority_supported", False, "must equal"),
        ("comparator", "missing", "not a rendered method"),
    ],
)
def test_primary_evidence_invariants(field: str, value: object, message: str) -> None:
    summary = _summary()
    metric = summary["primary_metrics"]["paired_parhelion_vs_primary_auc_delta"]
    metric[field] = value
    with pytest.raises(SchemaError, match=message):
        normalize_report_summary(summary)


def test_curve_interval_must_contain_mean() -> None:
    summary = _summary()
    uncertainty = summary["methods"]["parhelion_thompson"][0]["uncertainty"]
    uncertainty["low"] = 0.96
    with pytest.raises(SchemaError, match="low <= mean <= high"):
        normalize_report_summary(summary)


def test_historical_v1_and_v2_summaries_are_explicitly_supported() -> None:
    repository = Path(__file__).resolve().parents[1]
    v1 = read_json(repository / "benchmarks/results/l4-to-a10.json")
    v2 = read_json(repository / "benchmarks/results/parhelion-h100-final.json")

    v1_data = normalize_report_summary(v1)
    v2_data = normalize_report_summary(v2)

    assert v1_data.max_budget == 8
    assert v2_data.max_budget == 8
    assert any(method.key == "transfer_thompson" for method in v1_data.methods)
    assert any(method.key == "parhelion_thompson" for method in v2_data.methods)
