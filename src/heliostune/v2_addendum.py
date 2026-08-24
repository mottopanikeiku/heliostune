"""Post-hoc Parhelion v2 causal ablations with explicit uncertainty semantics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from heliostune.errors import ProtocolError
from heliostune.multisource_engine import (
    MethodEvaluation,
    PreparedReplay,
    assemble_multisource_summary,
    evaluate_anchored_cold,
    evaluate_cold_thompson,
    evaluate_multisource_retrieval,
    evaluate_parhelion,
    evaluate_parhelion_no_forced_anchor,
    evaluate_pooled_source,
    evaluate_random,
    evaluation_auc,
    parameter_independent_evaluations,
    prepare_multisource,
    serialize_workload_endpoints,
)
from heliostune.schema import Measurement
from heliostune.uncertainty import (
    deterministic_fold_summary,
    paired_contrast,
    stochastic_interval,
)

_ANALYSIS_STATUS = "post_hoc_exploratory"
_CONDITIONAL = (
    "the fixed historical H100 timing matrix, 96-workload corpus, frozen source archive, "
    "selected v2 parameters, and 30-seed campaign"
)
_METHOD_LABELS = {
    "static_multisource": "Static multi-source best",
    "torch": "torch.matmul external control",
    "random": "Random search",
    "single_source_nearest": "Single-source nearest-shape reuse",
    "multisource_retrieval": "Multi-source retrieval",
    "cold_thompson": "Cold Thompson sampling",
    "anchored_cold_thompson": "Anchored cold Thompson sampling",
    "pooled_source_thompson": "Pooled-source Thompson sampling",
    "parhelion_thompson": "Parhelion",
    "parhelion_no_forced_anchor": "Parhelion without forced anchor",
    "exhaustive": "Exhaustive curated Triton reference",
    "heldout_reference": "Held-out reference parity",
}
_METHOD_ROLES = {
    "static_multisource": "zero_query",
    "torch": "external",
    "random": "sequential",
    "single_source_nearest": "sequential",
    "multisource_retrieval": "sequential",
    "cold_thompson": "sequential",
    "anchored_cold_thompson": "sequential",
    "pooled_source_thompson": "sequential",
    "parhelion_thompson": "sequential",
    "parhelion_no_forced_anchor": "sequential",
    "exhaustive": "exhaustive",
    "heldout_reference": "reference",
}


def _number(value: object, *, context: str) -> float:
    if type(value) not in {int, float}:
        raise ProtocolError(f"{context} must be numeric")
    result = float(cast(int | float, value))
    if not np.isfinite(result):
        raise ProtocolError(f"{context} must be finite")
    return result


def _seed_curves(evaluation: MethodEvaluation) -> tuple[tuple[float, ...], ...]:
    if not evaluation.stochastic_seed_fold_curves:
        raise ProtocolError(f"{evaluation.method} is not a stochastic evaluation")
    return tuple(
        tuple(float(value) for value in np.mean(np.asarray(folds), axis=0))
        for folds in evaluation.stochastic_seed_fold_curves
    )


def _stochastic_curve_points(
    evaluation: MethodEvaluation,
    budgets: Sequence[int],
) -> list[dict[str, object]]:
    seed_curves = _seed_curves(evaluation)
    points: list[dict[str, object]] = []
    for index, budget in enumerate(budgets):
        summary = stochastic_interval(
            [curve[index] for curve in seed_curves],
            estimand=(
                f"{evaluation.method} mean fraction of held-out reference at budget {budget}"
            ),
            conditional_on=_CONDITIONAL,
        )
        points.append(
            {
                "budget": budget,
                "mean_fraction_oracle": summary["mean"],
                "uncertainty": summary["uncertainty"],
            }
        )
    return points


def _deterministic_curve_points(
    method: str,
    fold_curves: Sequence[Sequence[float]],
    budgets: Sequence[int],
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for index, budget in enumerate(budgets):
        descriptive = deterministic_fold_summary(
            [curve[index] for curve in fold_curves],
            estimand=f"{method} mean fraction of held-out reference at budget {budget}",
            conditional_on=_CONDITIONAL,
        )
        details = cast(Mapping[str, object], descriptive["descriptive"])
        points.append(
            {
                "budget": budget,
                "mean_fraction_oracle": descriptive["mean"],
                "uncertainty": {
                    "estimand": details["estimand"],
                    "sampling_unit": details["sampling_unit"],
                    "n": 4,
                    "conditional_on": details["conditional_on"],
                    "interval_method": details["summary_method"],
                    "low": descriptive["min"],
                    "high": descriptive["max"],
                },
                "descriptive": descriptive,
            }
        )
    return points


def _auc_summary(
    prepared: PreparedReplay,
    evaluation: MethodEvaluation,
) -> dict[str, object]:
    if evaluation.stochastic_seed_fold_curves:
        values = [float(np.mean(curve)) for curve in _seed_curves(evaluation)]
        summary = stochastic_interval(
            values,
            estimand=f"{evaluation.method} equal-budget fraction-reference AUC1-8",
            conditional_on=_CONDITIONAL,
        )
    else:
        values = [float(np.mean(curve)) for curve in evaluation.deterministic_fold_curves]
        summary = deterministic_fold_summary(
            values,
            estimand=f"{evaluation.method} equal-budget fraction-reference AUC1-8",
            conditional_on=_CONDITIONAL,
        )
    curve_length = (
        len(evaluation.stochastic_seed_fold_curves[0][0])
        if evaluation.stochastic_seed_fold_curves
        else len(evaluation.deterministic_fold_curves[0])
    )
    if curve_length == len(prepared.budgets):
        summary["mean"] = evaluation_auc(prepared, evaluation)
    return summary


def _contrast(
    prepared: PreparedReplay,
    parhelion: MethodEvaluation,
    comparator: MethodEvaluation,
) -> dict[str, object]:
    by_seed_and_fold: list[dict[str, object]] = []
    auc_seed_values: list[float] = []
    budget8_seed_values: list[float] = []
    for seed in range(prepared.seeds):
        fold_values: list[dict[str, object]] = []
        for fold_index, fold in enumerate(prepared.folds):
            parhelion_curve = parhelion.stochastic_seed_fold_curves[seed][fold_index]
            if comparator.stochastic_seed_fold_curves:
                comparator_curve = comparator.stochastic_seed_fold_curves[seed][fold_index]
            else:
                comparator_curve = comparator.deterministic_fold_curves[fold_index]
            fold_values.append(
                {
                    "heldout_model": fold.heldout_model,
                    "auc1_8_delta": float(np.mean(parhelion_curve))
                    - float(np.mean(comparator_curve)),
                    "budget8_delta": parhelion_curve[-1] - comparator_curve[-1],
                }
            )
        equal_fold_auc = float(np.mean([row["auc1_8_delta"] for row in fold_values]))
        equal_fold_budget8 = float(np.mean([row["budget8_delta"] for row in fold_values]))
        auc_seed_values.append(equal_fold_auc)
        budget8_seed_values.append(equal_fold_budget8)
        by_seed_and_fold.append(
            {
                "seed": seed,
                "folds": fold_values,
                "equal_fold_auc1_8_delta": equal_fold_auc,
                "equal_fold_budget8_delta": equal_fold_budget8,
            }
        )
    return {
        "analysis_status": _ANALYSIS_STATUS,
        "left_method": "parhelion_thompson",
        "right_method": comparator.method,
        "by_seed_and_fold": by_seed_and_fold,
        "auc1_8": paired_contrast(
            auc_seed_values,
            [0.0] * prepared.seeds,
            estimand=f"Parhelion minus {comparator.method} AUC1-8",
            conditional_on=_CONDITIONAL,
            analysis_status=_ANALYSIS_STATUS,
        ),
        "budget8": paired_contrast(
            budget8_seed_values,
            [0.0] * prepared.seeds,
            estimand=f"Parhelion minus {comparator.method} budget-8 fraction",
            conditional_on=_CONDITIONAL,
            analysis_status=_ANALYSIS_STATUS,
        ),
    }


def _endpoint_summaries(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["method"]), str(record["workload_key"]))].append(record)
    per_workload: list[dict[str, object]] = []
    by_method: dict[str, list[float]] = defaultdict(list)
    for (method, workload), rows in sorted(grouped.items()):
        fractions = [
            _number(row["fraction_reference"], context="endpoint fraction") for row in rows
        ]
        latencies = [
            _number(
                row["bank2_evaluation_latency_ms"],
                context="endpoint latency",
            )
            for row in rows
        ]
        tflops = [_number(row["tflops"], context="endpoint TFLOP/s") for row in rows]
        mean_fraction = float(np.mean(fractions))
        by_method[method].append(mean_fraction)
        per_workload.append(
            {
                "method": method,
                "workload_key": workload,
                "policy_seeds": len(rows),
                "mean_fraction_reference": mean_fraction,
                "mean_bank2_evaluation_latency_ms": float(np.mean(latencies)),
                "mean_tflops": float(np.mean(tflops)),
            }
        )
    distributions = {
        method: {
            "workloads": len(values),
            "median_fraction_reference": float(np.quantile(values, 0.5, method="linear")),
            "p10_fraction_reference": float(np.quantile(values, 0.1, method="linear")),
            "worst_fraction_reference": float(np.min(values)),
        }
        for method, values in sorted(by_method.items())
    }
    return {
        "seed_averaging": "arithmetic mean within each workload before corpus quantiles",
        "quantile_method": "numpy.quantile(method='linear')",
        "per_workload": per_workload,
        "distributions": distributions,
    }


def build_v2_addendum_summary(
    measurements: Sequence[Measurement],
    historical_summary: Mapping[str, object],
) -> dict[str, object]:
    """Run frozen v2 ablations without selection or outcome-dependent branching."""
    prepared = prepare_multisource(
        measurements,
        source_gpus=("L4", "A10", "T4"),
        target_gpu="H100",
        max_budget=8,
        seeds=30,
        protocol_role="final",
    )
    retrieval = evaluate_multisource_retrieval(
        prepared,
        k=8,
        temperature=0.2,
        capture_endpoints=True,
    )
    random = evaluate_random(prepared, capture_endpoints=True)
    cold = evaluate_cold_thompson(prepared, capture_endpoints=True)
    pooled = evaluate_pooled_source(
        prepared,
        transfer_strength=0.0,
        capture_endpoints=True,
    )
    parhelion = evaluate_parhelion(
        prepared,
        k=16,
        temperature=2.0,
        transfer_strength=0.0,
        retrieval=retrieval,
        capture_endpoints=True,
    )
    anchored = evaluate_anchored_cold(
        prepared,
        retrieval=retrieval,
        capture_endpoints=True,
    )
    no_anchor = evaluate_parhelion_no_forced_anchor(
        prepared,
        k=16,
        temperature=2.0,
        transfer_strength=0.0,
        capture_endpoints=True,
    )
    if pooled.stochastic_seed_fold_curves != cold.stochastic_seed_fold_curves:
        raise ProtocolError("selected pooled strength zero must be byte-identical to cold")
    if not (parhelion.fold_metadata == anchored.fold_metadata == retrieval.fold_metadata):
        raise ProtocolError("Parhelion and anchored cold must share the paid retrieval anchor")
    for seed in range(prepared.seeds):
        for fold in range(len(prepared.folds)):
            expected = retrieval.deterministic_fold_curves[fold][0]
            if parhelion.stochastic_seed_fold_curves[seed][fold][0] != expected:
                raise ProtocolError("Parhelion budget-one value differs from retrieval anchor")
            if anchored.stochastic_seed_fold_curves[seed][fold][0] != expected:
                raise ProtocolError("anchored cold budget-one value differs from retrieval anchor")

    independent = parameter_independent_evaluations(prepared)
    evaluations = {
        "static_multisource": independent["static_multisource"],
        "torch": independent["torch"],
        "random": random,
        "single_source_nearest": independent["single_source_nearest"],
        "multisource_retrieval": retrieval,
        "cold_thompson": cold,
        "anchored_cold_thompson": anchored,
        "pooled_source_thompson": pooled,
        "parhelion_thompson": parhelion,
        "parhelion_no_forced_anchor": no_anchor,
        "exhaustive": independent["exhaustive"],
    }
    methods: dict[str, object] = {}
    auc: dict[str, object] = {}
    for key, evaluation in evaluations.items():
        if evaluation.stochastic_seed_fold_curves:
            methods[key] = _stochastic_curve_points(evaluation, prepared.budgets)
        else:
            budgets = (len(prepared.configs),) if key == "exhaustive" else prepared.budgets
            methods[key] = _deterministic_curve_points(
                key,
                evaluation.deterministic_fold_curves,
                budgets,
            )
        auc[key] = _auc_summary(prepared, evaluation)
    methods["heldout_reference"] = _deterministic_curve_points(
        "heldout_reference",
        ((1.0,), (1.0,), (1.0,), (1.0,)),
        (len(prepared.configs),),
    )

    base = assemble_multisource_summary(
        prepared,
        retrieval=retrieval,
        pooled=pooled,
        parhelion=parhelion,
        k=16,
        temperature=2.0,
        transfer_strength=0.0,
        retrieval_k=8,
        retrieval_temperature=0.2,
        pooled_transfer_strength=0.0,
        primary_comparator="torch",
    )
    endpoint_evaluations = (
        random,
        cold,
        anchored,
        pooled,
        parhelion,
        no_anchor,
    )
    endpoint_records = [
        record
        for evaluation in endpoint_evaluations
        for record in serialize_workload_endpoints(prepared, evaluation)
    ]
    historical_primary = cast(Mapping[str, object], historical_summary["primary_metrics"])[
        "paired_parhelion_vs_primary_auc_delta"
    ]
    contrasts = {
        comparator.method: _contrast(prepared, parhelion, comparator)
        for comparator in (anchored, cold, retrieval, no_anchor)
    }
    return {
        "schema_version": 1,
        "study_id": "parhelion-v2-post-hoc-causal-addendum",
        "analysis_status": _ANALYSIS_STATUS,
        "data_kind": "measured",
        "source_gpu": base["source_gpu"],
        "source_gpus": base["source_gpus"],
        "target_gpu": base["target_gpu"],
        "workloads": base["workloads"],
        "configs": base["configs"],
        "max_budget": 8,
        "seeds": 30,
        "methodology": (
            "Post-hoc exploratory causal ablations on the immutable Parhelion v2 matrix. "
            "No H100-side parameter selection or recollection is performed."
        ),
        "method_labels": _METHOD_LABELS,
        "method_roles": _METHOD_ROLES,
        "methods": methods,
        "auc": auc,
        "transfer_method": "parhelion_thompson",
        "cold_method": "anchored_cold_thompson",
        "primary_comparator": "torch",
        "primary_metrics": {
            "paired_parhelion_vs_primary_auc_delta": historical_primary,
            "historical_confirmatory_endpoint_status": "unchanged",
            "algorithmic_contrasts": contrasts,
        },
        "historical_confirmatory_endpoint": {
            "analysis_status": "historical_confirmatory_unchanged",
            "method": "parhelion_thompson",
            "comparator": "torch",
            "evidence": historical_primary,
        },
        "algorithmic_contrasts": contrasts,
        "budget_one_invariant": {
            "verified": True,
            "methods": [
                "multisource_retrieval",
                "parhelion_thompson",
                "anchored_cold_thompson",
            ],
            "anchor_configs_by_fold": [dict(values) for values in retrieval.fold_metadata],
        },
        "policy_seed_workload_endpoints_budget8": endpoint_records,
        "workload_endpoint_summary": _endpoint_summaries(endpoint_records),
        "hardware": base["hardware"],
        "source_hardware": base["source_hardware"],
        "target_hardware": base["target_hardware"],
        "folds": base["folds"],
        "fold_results": base["fold_results"],
        "source_cost": base["source_cost"],
        "target_collection_cost": base["target_collection_cost"],
        "experiment": {
            **cast(Mapping[str, object], base["experiment"]),
            "analysis_status": _ANALYSIS_STATUS,
            "sampling_unit": "paired policy seed after equal-fold averaging",
            "conditional_on": _CONDITIONAL,
        },
        "provenance": {
            **cast(Mapping[str, object], base["provenance"]),
            "historical_input": "benchmarks/data/parhelion-v2-measurements.jsonl.zst",
            "selection": "frozen T4 choices; no H100-side selection",
        },
        "limitations": [
            "Every new algorithmic contrast is post-hoc exploratory and has no multiplicity-adjusted superiority claim.",
            "Monte Carlo intervals condition on the fixed historical matrix, corpus, archive, and campaign.",
            *cast(Sequence[str], base["limitations"]),
        ],
    }


__all__ = ["build_v2_addendum_summary"]
