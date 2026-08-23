from pathlib import Path

from heliostune.report import render_report


def test_report_is_offline_and_escapes_dynamic_text(tmp_path: Path) -> None:
    output = tmp_path / "index.html"
    summary = {
        "source_gpu": "source<script>",
        "target_gpu": "target",
        "workloads": 2,
        "configs": 3,
        "methods": {
            "transfer_thompson": [
                {
                    "budget": 1,
                    "mean_fraction_oracle": 0.9,
                    "ci95_low": 0.8,
                    "ci95_high": 0.95,
                }
            ],
            "cold_thompson": [
                {
                    "budget": 1,
                    "mean_fraction_oracle": 0.7,
                    "ci95_low": 0.6,
                    "ci95_high": 0.8,
                }
            ],
        },
    }

    render_report(summary, output)
    document = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in document
    assert "source&lt;script&gt;" in document
    assert "source<script>" not in document
    assert "https://" not in document
    assert "<svg" in document
    assert 'type="application/json"' in document


def _parhelion_summary(superiority_supported: bool) -> dict:
    frozen_comparator = "frozen<baseline>"
    descriptive_comparator = "target_selected<winner>"
    return {
        "source_gpu": "legacy aggregate source",
        "source_gpus": ["L4<source>", "A10&source", 'T4"source'],
        "source_hardware": {
            "profiles": [
                {
                    "gpu": "L4<source>",
                    "device_name": "L4 <profile>",
                    "compute_capability": [8, 9],
                },
                {
                    "gpu": "A10&source",
                    "device_name": "A10 & profile",
                    "compute_capability": [8, 6],
                },
                {
                    "gpu": 'T4"source',
                    "device_name": 'T4 "profile"',
                    "compute_capability": [7, 5],
                },
            ]
        },
        "target_gpu": "H100<target>",
        "target_hardware": {
            "gpu": "H100<target>",
            "device_name": "H100 <evaluation>",
            "compute_capability": [9, 0],
        },
        "workloads": 24,
        "configs": 36,
        "methods": {
            "parhelion_thompson": [
                {
                    "budget": 8,
                    "mean_fraction_oracle": 0.94,
                    "ci95_low": 0.92,
                    "ci95_high": 0.96,
                }
            ],
            frozen_comparator: [
                {
                    "budget": 8,
                    "mean_fraction_oracle": 0.91,
                    "ci95_low": 0.89,
                    "ci95_high": 0.93,
                }
            ],
            descriptive_comparator: [
                {
                    "budget": 8,
                    "mean_fraction_oracle": 0.92,
                    "ci95_low": 0.9,
                    "ci95_high": 0.94,
                }
            ],
            "cold_thompson": [
                {
                    "budget": 8,
                    "mean_fraction_oracle": 0.5,
                    "ci95_low": 0.45,
                    "ci95_high": 0.55,
                }
            ],
        },
        "method_labels": {
            "parhelion_thompson": "Parhelion <adaptive> & shared",
            frozen_comparator: "Frozen <baseline> & fixed",
            descriptive_comparator: "H100 <descriptive winner>",
            "cold_thompson": "Cold <endpoint>",
        },
        "transfer_method": "parhelion_thompson",
        "cold_method": "cold_thompson",
        "primary_comparator": frozen_comparator,
        "headline": {
            "descriptive_target_strongest_legacy_method": descriptive_comparator,
        },
        "primary_metrics": {
            "paired_parhelion_vs_primary_auc_delta": {
                "comparator": frozen_comparator,
                "mean_auc_delta": 0.031 if superiority_supported else 0.002,
                "ci95_low": 0.01 if superiority_supported else -0.01,
                "ci95_high": 0.052 if superiority_supported else 0.014,
                "paired_seeds": 30,
                "degrees_of_freedom": 29,
                "superiority_supported": superiority_supported,
                "claim": (
                    "Parhelion <wins> & remains frozen."
                    if superiority_supported
                    else "MUST NOT RENDER <unsupported claim>"
                ),
            }
        },
        "experiment": {
            "target_budget_unit": "probes per held-out <workload>",
            "adaptation_scope": "one shared <posterior> & paired updates",
        },
        "source_cost": {
            "disclosure": "source <archive> is paid before target tuning",
        },
        "target_collection_cost": {
            "simulated_online_queries_per_live_method_per_workload": 8,
            "budget_b_formula": "24*b probes per <fold>",
            "adaptation_scope": "shared <posterior> within each fold",
        },
    }


def test_parhelion_report_renders_supported_frozen_primary_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "supported.html"

    render_report(_parhelion_summary(True), output)

    document = output.read_text(encoding="utf-8")
    assert "Frozen primary-comparison evidence" in document
    assert "Parhelion &lt;adaptive&gt; &amp; shared" in document
    assert "Frozen &lt;baseline&gt; &amp; fixed" in document
    assert "+0.0310 AUC (+3.10 pp)" in document
    assert "[+0.0100, +0.0520] AUC ([+1.00, +5.20] pp)" in document
    assert "30 paired seeds / 29 df" in document
    assert "superiority_supported" in document
    assert ">true</td>" in document
    assert "<strong>Superiority supported.</strong>" in document
    assert "Parhelion &lt;wins&gt; &amp; remains frozen." in document
    assert "H100 &lt;descriptive winner&gt;" in document
    assert "Descriptive target-side strongest baseline only" in document
    assert "Source contexts · 3" in document
    assert "L4 &lt;profile&gt;" in document
    assert "Target context · evaluation domain" in document
    assert "H100 &lt;evaluation&gt;" in document
    assert "probes per held-out &lt;workload&gt;" in document
    assert "one shared &lt;posterior&gt; &amp; paired updates" in document
    assert "shared &lt;posterior&gt; within each fold" in document
    assert "<adaptive>" not in document
    assert "<baseline>" not in document
    assert "<profile>" not in document


def test_parhelion_report_does_not_claim_unsupported_superiority_or_use_cold_fallback(
    tmp_path: Path,
) -> None:
    output = tmp_path / "unsupported.html"

    render_report(_parhelion_summary(False), output)

    document = output.read_text(encoding="utf-8")
    assert "Frozen primary-comparison evidence" in document
    assert "+0.0020 AUC (+0.20 pp)" in document
    assert "[-0.0100, +0.0140] AUC ([-1.00, +1.40] pp)" in document
    assert ">false</td>" in document
    assert "<strong>Superiority was not demonstrated.</strong>" in document
    assert "does not support a superiority claim" in document
    assert "MUST NOT RENDER" not in document
    assert "at their largest shared budget" not in document
    assert "Cold &lt;endpoint&gt; at shared budget" not in document
    assert (
        "versus frozen primary comparator "
        "<strong>Frozen &lt;baseline&gt; &amp; fixed</strong>"
    ) in document
    assert "H100 &lt;descriptive winner&gt; is reported as target-selected descriptive context" in document
    assert "does not replace the frozen primary comparator" in document
    assert "<unsupported claim>" not in document
