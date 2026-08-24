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
    assert "Fold-level evidence audit" not in document
    assert "Release chain of custody" not in document


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


def test_report_audits_fold_results_release_chain_and_online_budget(
    tmp_path: Path,
) -> None:
    def point(
        budget: int,
        mean: float,
        low: float | None = None,
        high: float | None = None,
    ) -> dict:
        return {
            "budget": budget,
            "mean_fraction_oracle": mean,
            "ci95_low": mean if low is None else low,
            "ci95_high": mean if high is None else high,
        }

    summary = _parhelion_summary(True)
    summary["max_budget"] = 8
    summary["primary_metrics"]["primary_budget"] = 8
    summary["methods"]["exhaustive"] = [point(36, 1.0)]
    summary["method_labels"]["exhaustive"] = "Exhaustive <reference>"

    fold_families = (
        "bert<encoder>",
        "gpt&decoder",
        'llama"family"',
        "t5/family",
    )
    fold_means = (0.8111, 0.8222, 0.8333, 0.8444)
    summary["fold_results"] = [
        {
            "heldout_model": family,
            "target_workloads": 20 + index,
            "visible_bank0_source_observations_by_gpu": {
                "L4<visible>": 100 + index,
                "A10&visible": 200 + index,
            },
            "excluded_exact_target_shapes_by_gpu": {
                "L4<visible>": 10 + index,
                "A10&visible": 20 + index,
            },
            "methods": {
                "parhelion_thompson": [
                    point(1, mean, mean - 0.01, mean + 0.01),
                    point(8, mean + 0.1, mean + 0.08, mean + 0.12),
                ],
                "frozen<baseline>": [
                    point(1, mean - 0.1),
                    point(8, mean + 0.05),
                ],
                "exhaustive": [point(36, 1.0)],
            },
        }
        for index, (family, mean) in enumerate(
            zip(fold_families, fold_means, strict=True),
            start=1,
        )
    ]
    summary["release_provenance"] = {
        "sole_h100_run": 'https://modal.com/apps/<owner>/runs/"sole"&1',
        "algorithm_commit": "algorithm<commit>",
        "freeze_commit": "freeze&commit",
        "freeze_sha256": "sha256:<freeze>",
        "raw_h100_sha256": "sha256:raw&h100",
        "final_archive_sha256": 'sha256:"final"',
        "post_run_manifest_path": "artifacts/<post>&manifest.json",
    }
    output = tmp_path / "fold-evidence.html"

    render_report(summary, output)

    document = output.read_text(encoding="utf-8")
    assert document.count('class="panel fold-audit"') == 4
    assert document.count("Complete method-by-budget numeric table · 5 rows") == 4
    assert document.count(
        "Per-source fold visibility and exact-target-shape exclusion audit"
    ) == 4
    assert "bert&lt;encoder&gt;" in document
    assert "gpt&amp;decoder" in document
    assert "llama&quot;family&quot;" in document
    assert "t5/family" in document
    assert "L4&lt;visible&gt;" in document
    assert "A10&amp;visible" in document
    escaped_families = (
        "bert&lt;encoder&gt;",
        "gpt&amp;decoder",
        "llama&quot;family&quot;",
        "t5/family",
    )
    for index, (family, mean) in enumerate(
        zip(escaped_families, fold_means, strict=True),
        start=1,
    ):
        fold_start = document.index(f'id="fold-result-{index}"')
        fold_end = document.index("</article>", fold_start)
        fold_document = document[fold_start:fold_end]
        assert family in fold_document
        assert (
            f'<strong>{20 + index}</strong><span>Target workloads</span>'
        ) in fold_document
        assert (
            "L4&lt;visible&gt;</td>"
            f'<td class="numeric">{100 + index}</td>'
            f'<td class="numeric">{10 + index}</td>'
        ) in fold_document
        assert (
            "A10&amp;visible</td>"
            f'<td class="numeric">{200 + index}</td>'
            f'<td class="numeric">{20 + index}</td>'
        ) in fold_document
        assert f'<td class="numeric">{mean - 0.01:.4f}</td>' in fold_document
        assert f'<td class="numeric">{mean + 0.01:.4f}</td>' in fold_document
        assert fold_document.count("Parhelion &lt;adaptive&gt; &amp; shared</td>") == 2
        assert fold_document.count("Frozen &lt;baseline&gt; &amp; fixed</td>") == 2
        assert fold_document.count("Exhaustive &lt;reference&gt;</td>") == 1
        assert f'<td class="numeric">{mean:.4f}</td>' in fold_document
        assert (
            "Exhaustive &lt;reference&gt;</td><td class=\"numeric\">36</td>"
        ) in fold_document

    assert "Release chain of custody" in document
    assert "Sole H100 run" in document
    assert "Algorithm commit" in document
    assert "Freeze commit" in document
    assert "Freeze SHA-256" in document
    assert "Raw H100 SHA-256" in document
    assert "Final archive SHA-256" in document
    assert "Post-run manifest path" in document
    assert document.index("01 · Freeze") < document.index("02 · Collection")
    assert document.index("02 · Collection") < document.index("03 · Release")
    assert (
        "https://modal.com/apps/&lt;owner&gt;/runs/&quot;sole&quot;&amp;1"
        in document
    )
    assert "algorithm&lt;commit&gt;" in document
    assert "freeze&amp;commit" in document
    assert "sha256:&lt;freeze&gt;" in document
    assert "sha256:raw&amp;h100" in document
    assert "sha256:&quot;final&quot;" in document
    assert "artifacts/&lt;post&gt;&amp;manifest.json" in document
    assert "<owner>" not in document
    assert "<post>" not in document

    assert (
        '<span class="result-value">8</span>'
        '<span class="result-label">Declared online target budget</span>'
    ) in document
    assert (
        '<span class="result-value">36</span>'
        '<span class="result-label">Declared online target budget</span>'
    ) not in document
    assert document.index("Frozen primary-comparison evidence") < document.index(
        "Fold-level evidence audit"
    )
    assert "the target-selected strongest legacy method remains descriptive only" in document
