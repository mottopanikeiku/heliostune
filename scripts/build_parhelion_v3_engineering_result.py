"""Evaluate and render the operator-authorized Parhelion v3 H200 engineering run."""

from __future__ import annotations

import html
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from heliostune.artifacts import read_json, read_measurements, write_json_atomic, write_text_atomic
from heliostune.protocol import load_v3_protocol, require_v3_runtime, runtime_manifest
from heliostune.uncertainty import paired_contrast
from heliostune.v3_artifacts import sha256_file
from heliostune.v3_engine import (
    V3Evaluation,
    V3Prepared,
    evaluate_v3_anchored_cold,
    evaluate_v3_cold,
    evaluate_v3_nearest,
    evaluate_v3_no_anchor,
    evaluate_v3_no_transfer,
    evaluate_v3_parhelion,
    evaluate_v3_pooled,
    evaluate_v3_random,
    evaluate_v3_retrieval,
    evaluation_seed_curves,
    prepare_v3,
)
from heliostune.validation import exact_int, exact_object, finite_float

_REPO = Path(__file__).resolve().parents[1]
_FREEZE = _REPO / "benchmarks/parhelion-v3-h200-freeze.json"
_CONFIG = _REPO / "benchmarks/parhelion-v3-config-manifest.json"
_SELECTION = _REPO / "benchmarks/results/parhelion-v3-a100-selection.json"
_FINAL = _REPO / "benchmarks/data/parhelion-v3-final.jsonl.zst"
_VALIDATION_MANIFEST = _REPO / "benchmarks/data/parhelion-v3-validation.jsonl.zst.manifest.json"
_H200_MANIFEST = _REPO / "benchmarks/data/parhelion-v3-h200.jsonl.zst.manifest.json"
_FINAL_MANIFEST = _REPO / "benchmarks/data/parhelion-v3-final.jsonl.zst.manifest.json"
_OUTPUT = _REPO / "benchmarks/results/parhelion-v3-h200-engineering.json"
_REPORT = _REPO / "site/parhelion-v3-engineering.html"
_CONDITIONAL_ON = (
    "fixed mixed-A100 engineering archive, fixed H200 matrix, retained action set, "
    "policy seeds, and operator-authorized protocol deviation"
)


def _queries_to_95(curve: Sequence[float]) -> int | None:
    for budget, value in enumerate(curve, start=1):
        if value >= 0.95:
            return budget
    return None


def summarize_evaluation(
    prepared: V3Prepared,
    evaluation: V3Evaluation,
) -> tuple[dict[str, object], dict[int, tuple[float, ...]]]:
    """Serialize primary/sensitivity curves and return AUC vectors by scoring bank."""
    banks: dict[str, object] = {}
    auc_vectors: dict[int, tuple[float, ...]] = {}
    for bank in (2, 3, 4):
        curves = evaluation_seed_curves(prepared, evaluation, bank=bank)
        matrix = np.asarray(curves, dtype=np.float64)
        auc = tuple(float(value) for value in np.mean(matrix[:, :8], axis=1))
        auc_vectors[bank] = auc
        queries = tuple(_queries_to_95(curve) for curve in curves)
        successful_queries = [query for query in queries if query is not None]
        banks[str(bank)] = {
            "seed_count": len(curves),
            "seed_curves": [[float(value) for value in curve] for curve in curves],
            "mean_curve": [float(value) for value in np.mean(matrix, axis=0)],
            "auc1_8_by_seed": list(auc),
            "auc1_8": float(np.mean(auc)),
            "fraction_at_budget_8": float(np.mean(matrix[:, 7])),
            "queries_to_95_by_seed": list(queries),
            "queries_to_95_median_low_successes": (
                statistics.median_low(successful_queries) if successful_queries else None
            ),
            "queries_to_95_successes": len(successful_queries),
        }
    return {
        "method": evaluation.method,
        "deterministic": evaluation.deterministic,
        "banks": banks,
    }, auc_vectors


def _selected_parameters(selection: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    selected = exact_object(selection.get("selected"), context="v3 selected parameters")
    retrieval = exact_object(selected.get("multisource_retrieval"), context="retrieval winner")
    pooled = exact_object(selected.get("pooled_source_thompson"), context="pooled winner")
    parhelion = exact_object(selected.get("parhelion_thompson"), context="Parhelion winner")
    return retrieval, pooled, parhelion


def build_result() -> dict[str, object]:
    freeze = load_v3_protocol(_FREEZE)
    require_v3_runtime(freeze)
    config = exact_object(read_json(_CONFIG), context="v3 config manifest")
    selection = exact_object(read_json(_SELECTION), context="v3 A100 selection")
    retained = cast(list[str], config["retained_config_keys"])
    official = cast(list[str], config["retained_official_config_keys"])
    seeds = cast(list[int], freeze["final_seeds"])
    rows = read_measurements(_FINAL)
    prepared = prepare_v3(
        freeze,
        rows,
        source_gpus=("L4", "A10", "A100-80GB"),
        target_gpu="H200",
        retained_config_keys=retained,
        official_config_keys=official,
        seeds=seeds,
    )
    retrieval_parameters, pooled_parameters, parhelion_parameters = _selected_parameters(selection)
    retrieval = evaluate_v3_retrieval(
        prepared,
        k=cast(int, retrieval_parameters["k"]),
        temperature=cast(float, retrieval_parameters["temperature"]),
    )
    evaluations = (
        retrieval,
        evaluate_v3_nearest(prepared),
        evaluate_v3_random(prepared),
        evaluate_v3_cold(prepared),
        evaluate_v3_anchored_cold(prepared, retrieval),
        evaluate_v3_pooled(
            prepared,
            transfer_strength=cast(float, pooled_parameters["transfer_strength"]),
        ),
        evaluate_v3_parhelion(
            prepared,
            k=cast(int, parhelion_parameters["k"]),
            temperature=cast(float, parhelion_parameters["temperature"]),
            transfer_strength=cast(float, parhelion_parameters["transfer_strength"]),
            retrieval_evaluation=retrieval,
        ),
        evaluate_v3_no_anchor(
            prepared,
            k=cast(int, parhelion_parameters["k"]),
            temperature=cast(float, parhelion_parameters["temperature"]),
            transfer_strength=cast(float, parhelion_parameters["transfer_strength"]),
        ),
        evaluate_v3_no_transfer(
            prepared,
            k=cast(int, parhelion_parameters["k"]),
            temperature=cast(float, parhelion_parameters["temperature"]),
            retrieval_evaluation=retrieval,
        ),
    )
    methods: dict[str, object] = {}
    auc_vectors: dict[str, dict[int, tuple[float, ...]]] = {}
    for evaluation in evaluations:
        summary, vectors = summarize_evaluation(prepared, evaluation)
        methods[evaluation.method] = summary
        auc_vectors[evaluation.method] = vectors

    contrast = paired_contrast(
        auc_vectors["parhelion_thompson"][2],
        auc_vectors["anchored_cold_thompson"][2],
        estimand="H200 Parhelion minus anchored-cold AUC1-8",
        conditional_on=_CONDITIONAL_ON,
        analysis_status="operator_authorized_engineering_protocol_deviation",
    )
    uncertainty = exact_object(contrast["uncertainty"], context="primary uncertainty")
    contrast["frozen_rule_would_support_superiority"] = cast(float, uncertainty["low"]) > 0
    contrast["claim"] = None
    contrast["superiority_supported"] = None

    manifests = {
        "validation": _VALIDATION_MANIFEST,
        "h200": _H200_MANIFEST,
        "final": _FINAL_MANIFEST,
    }
    return {
        "schema_version": 1,
        "study_id": "parhelion-v3-h200-engineering",
        "analysis_status": "operator_authorized_engineering_protocol_deviation",
        "claim": None,
        "result_scope": "engineering benchmark; not a confirmatory protocol result",
        "limitations": [
            "the operator explicitly overrode the frozen pre-H200 no-retry rule",
            "A100 bank 0 used PCIe while banks 1-4 used SXM; only device_name was canonicalized in the derived archive",
            "the mixed A100 subvariant domain confounds parameter selection",
            "the fixed 96-workload FP16 corpus is not an unseen-workload or end-to-end serving study",
        ],
        "source_gpus": list(prepared.source_gpus),
        "target_gpu": prepared.target_gpu,
        "retained_config_count": len(prepared.configs),
        "official_config_count": len(official),
        "workload_count": sum(len(fold.target_workloads) for fold in prepared.folds),
        "seeds": list(prepared.seeds),
        "budgets": list(prepared.budgets),
        "selected": selection["selected"],
        "methods": methods,
        "primary_contrast": contrast,
        "sensitivity_contrasts": {
            str(bank): paired_contrast(
                auc_vectors["parhelion_thompson"][bank],
                auc_vectors["anchored_cold_thompson"][bank],
                estimand=f"bank-{bank} Parhelion minus anchored-cold AUC1-8",
                conditional_on=_CONDITIONAL_ON,
                analysis_status="operator_authorized_engineering_protocol_deviation",
            )
            for bank in (3, 4)
        },
        "collection_runs": {
            "failed_pilot": "ap-nWqf5qjkL9CdGVuL5lWcl6",
            "successful_pilot_retry": "ap-8TUfMQNoH4lzXJ1uOI5n2x",
            "candidate_bank0": "ap-a2tRHcReiUYfLwZ4iCN5Pu",
            "validation_banks1_4": "ap-Ba9FkS0Ax7i6cmjUIVh1Qi",
            "h200": "ap-dKaK5ML43S2EqbaW33OTtA",
        },
        "artifacts": {
            "freeze": {"path": str(_FREEZE.relative_to(_REPO)), "sha256": sha256_file(_FREEZE)},
            "config": {"path": str(_CONFIG.relative_to(_REPO)), "sha256": sha256_file(_CONFIG)},
            "selection": {
                "path": str(_SELECTION.relative_to(_REPO)),
                "sha256": sha256_file(_SELECTION),
            },
            "final_archive": {
                "path": str(_FINAL.relative_to(_REPO)),
                "sha256": sha256_file(_FINAL),
                "rows": len(rows),
            },
            "manifests": {
                name: {"path": str(path.relative_to(_REPO)), "sha256": sha256_file(path)}
                for name, path in manifests.items()
            },
        },
        "runtime": runtime_manifest(),
    }


def _percent(value: object) -> str:
    return f"{100 * finite_float(value, context='percentage'):.2f}%"


def render_html(result: Mapping[str, object]) -> str:
    methods = exact_object(result["methods"], context="v3 methods")
    rows = []
    curve_rows = []
    for name, raw_method in methods.items():
        method = exact_object(raw_method, context=f"method {name}")
        banks = exact_object(method["banks"], context=f"method {name} banks")
        primary = exact_object(banks["2"], context=f"method {name} bank 2")
        query = primary["queries_to_95_median_low_successes"]
        query_text = "—" if query is None else str(query)
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            f"<td>{_percent(primary['auc1_8'])}</td>"
            f"<td>{_percent(primary['fraction_at_budget_8'])}</td>"
            f"<td>{html.escape(query_text)}</td>"
            "</tr>"
        )
        curve = cast(Sequence[object], primary["mean_curve"])
        curve_rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            + "".join(f"<td>{_percent(value)}</td>" for value in curve)
            + "</tr>"
        )
    contrast = exact_object(result["primary_contrast"], context="primary contrast")
    interval = exact_object(contrast["uncertainty"], context="primary interval")
    limitations = cast(Sequence[object], result["limitations"])
    run_items = exact_object(result["collection_runs"], context="collection runs")
    artifacts = exact_object(result["artifacts"], context="artifacts")
    final_archive = exact_object(artifacts["final_archive"], context="final archive")
    raw_budgets = result["budgets"]
    if type(raw_budgets) is not list:
        raise TypeError("v3 budgets must be a list")
    budgets = cast(list[object], raw_budgets)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parhelion v3 H200 engineering benchmark</title>
<style>
:root{{--bg:#07111f;--panel:#0e1c2e;--ink:#edf6ff;--muted:#9db1c8;--accent:#59d4c7;--warn:#ffcc66;--line:#263d56}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:40px 20px 64px}}h1{{font-size:clamp(2rem,5vw,4rem);line-height:1.02;margin:.2em 0}}
.banner{{border:1px solid var(--warn);background:#2d2412;color:#ffe2a3;padding:14px 18px;border-radius:10px;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:24px 0}}.card{{background:var(--panel);border:1px solid var(--line);padding:18px;border-radius:12px}}
.card strong{{display:block;font-size:1.6rem;color:var(--accent)}}h2{{margin-top:38px}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}}
table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted)}}
code{{color:#b9f6ef}}a{{color:var(--accent)}}li{{margin:.45em 0}}.muted{{color:var(--muted)}}
</style></head><body><main>
<p class="muted">HeliosTune · Parhelion v3</p><h1>H200 engineering benchmark</h1>
<p class="banner">Protocol-deviation engineering result. The original failed campaign remains terminal; this page makes no confirmatory superiority claim.</p>
<div class="grid"><div class="card">Primary delta<strong>{finite_float(contrast["mean"], context="primary contrast mean"):+.4f}</strong>Parhelion − anchored cold AUC1–8</div>
<div class="card">95% seed interval<strong>[{finite_float(interval["low"], context="primary interval low"):+.4f}, {finite_float(interval["high"], context="primary interval high"):+.4f}]</strong>Conditional Monte Carlo interval</div>
<div class="card">H200 action set<strong>{exact_int(result["retained_config_count"], context="retained config count")}</strong>retained configurations</div>
<div class="card">Final archive<strong>{exact_int(final_archive["rows"], context="final archive rows"):,}</strong>four-GPU measurement rows</div></div>
<h2>Primary bank-2 results</h2><div class="table-wrap"><table><thead><tr><th>Method</th><th>AUC 1–8</th><th>Budget 8</th><th>Queries to 95%</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<h2>Mean curves</h2><div class="table-wrap"><table><thead><tr><th>Method</th>{"".join(f"<th>{budget}</th>" for budget in budgets)}</tr></thead><tbody>{"".join(curve_rows)}</tbody></table></div>
<h2>Validity limits</h2><ul>{"".join(f"<li>{html.escape(str(item))}</li>" for item in limitations)}</ul>
<h2>Collection chain</h2><ul>{"".join(f'<li><code>{html.escape(name)}</code>: <a href="https://modal.com/apps/mottopanikeiku/main/{html.escape(str(app_id))}">{html.escape(str(app_id))}</a></li>' for name, app_id in run_items.items())}</ul>
<p class="muted">All fractions use the bank-1-selected retained-config reference and score unchanged recommendations on bank 2. Banks 3 and 4 remain separate sensitivity matrices.</p>
</main></body></html>"""


def main() -> int:
    result = build_result()
    write_json_atomic(_OUTPUT, result)
    write_text_atomic(_REPORT, render_html(result))
    print(f"result={_OUTPUT.relative_to(_REPO)} report={_REPORT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
