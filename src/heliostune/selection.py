"""Frozen T4 hyperparameter and comparator selection for Parhelion v2."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Literal

from heliostune.multisource_engine import (
    MethodEvaluation,
    PreparedReplay,
    assemble_multisource_summary,
    evaluate_multisource_retrieval,
    evaluate_parhelion,
    evaluate_pooled_source,
    evaluation_auc,
    prepare_multisource,
)
from heliostune.schema import Measurement

K_GRID: tuple[int, ...] = (1, 3, 8, 16)
TEMPERATURE_GRID: tuple[float, ...] = (0.2, 0.7, 2.0)
TRANSFER_STRENGTH_GRID: tuple[float, ...] = (0.0, 0.02, 0.08, 0.2)
SELECTION_SEEDS = 12
FINAL_SEEDS = 30
MAX_BUDGET = 8
LEGACY_COMPARATORS: tuple[str, ...] = (
    "static_multisource",
    "torch",
    "random",
    "single_source_nearest",
    "multisource_retrieval",
    "cold_thompson",
    "pooled_source_thompson",
)


@dataclass(frozen=True, order=True, slots=True)
class ParhelionCandidate:
    """One point in the protocol's global Parhelion hyperparameter grid."""

    k: int
    temperature: float
    transfer_strength: float

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("k must be a positive integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive")
        if (
            isinstance(self.transfer_strength, bool)
            or not isinstance(self.transfer_strength, (int, float))
            or not math.isfinite(self.transfer_strength)
            or self.transfer_strength < 0
        ):
            raise ValueError("transfer_strength must be finite and non-negative")


def parhelion_grid() -> tuple[ParhelionCandidate, ...]:
    """Return the 48 candidates in deterministic lexicographic order."""
    return tuple(
        ParhelionCandidate(k, temperature, transfer_strength)
        for k in K_GRID
        for temperature in TEMPERATURE_GRID
        for transfer_strength in TRANSFER_STRENGTH_GRID
    )


_RawReplayContext = tuple[tuple[Measurement, ...], tuple[str, ...], str, int, int]
_JobKind = Literal["retrieval", "pooled", "parhelion"]
_SelectionJob = tuple[_JobKind, ParhelionCandidate, int | None, float | None]
_ScoredJob = tuple[_JobKind, ParhelionCandidate, float, MethodEvaluation]
_WORKER_PREPARED: PreparedReplay | None = None
_WORKER_RETRIEVAL: dict[tuple[int, float], MethodEvaluation] = {}


def _prepare_context(context: _RawReplayContext) -> PreparedReplay:
    measurements, sources, target, max_budget, seeds = context
    return prepare_multisource(
        measurements,
        source_gpus=sources,
        target_gpu=target,
        max_budget=max_budget,
        seeds=seeds,
        protocol_role="validation",
    )


def _initialize_worker(context: _RawReplayContext) -> None:
    global _WORKER_PREPARED, _WORKER_RETRIEVAL
    _WORKER_PREPARED = _prepare_context(context)
    _WORKER_RETRIEVAL = {}


def _evaluate_job(prepared: PreparedReplay, job: _SelectionJob) -> _ScoredJob:
    kind, candidate, retrieval_k, retrieval_temperature = job
    if kind == "retrieval":
        evaluation = evaluate_multisource_retrieval(
            prepared,
            k=candidate.k,
            temperature=candidate.temperature,
        )
    elif kind == "pooled":
        evaluation = evaluate_pooled_source(
            prepared,
            transfer_strength=candidate.transfer_strength,
        )
    else:
        if retrieval_k is None or retrieval_temperature is None:
            raise RuntimeError("Parhelion selection job is missing its frozen retrieval anchor")
        retrieval = evaluate_multisource_retrieval(
            prepared,
            k=retrieval_k,
            temperature=retrieval_temperature,
        )
        evaluation = evaluate_parhelion(
            prepared,
            k=candidate.k,
            temperature=candidate.temperature,
            transfer_strength=candidate.transfer_strength,
            retrieval=retrieval,
        )
    return kind, candidate, evaluation_auc(prepared, evaluation), evaluation


def _evaluate_worker(job: _SelectionJob) -> _ScoredJob:
    if _WORKER_PREPARED is None:
        raise RuntimeError("selection worker was not initialized")
    kind, candidate, retrieval_k, retrieval_temperature = job
    if kind != "parhelion":
        return _evaluate_job(_WORKER_PREPARED, job)
    if retrieval_k is None or retrieval_temperature is None:
        raise RuntimeError("Parhelion selection job is missing its frozen retrieval anchor")
    retrieval_key = (retrieval_k, retrieval_temperature)
    retrieval = _WORKER_RETRIEVAL.get(retrieval_key)
    if retrieval is None:
        retrieval = evaluate_multisource_retrieval(
            _WORKER_PREPARED,
            k=retrieval_k,
            temperature=retrieval_temperature,
        )
        _WORKER_RETRIEVAL[retrieval_key] = retrieval
    evaluation = evaluate_parhelion(
        _WORKER_PREPARED,
        k=candidate.k,
        temperature=candidate.temperature,
        transfer_strength=candidate.transfer_strength,
        retrieval=retrieval,
    )
    return kind, candidate, evaluation_auc(_WORKER_PREPARED, evaluation), evaluation


def _select_comparator(summary: dict[str, Any]) -> str:
    auc = summary.get("auc")
    if not isinstance(auc, dict):
        raise ValueError("multi-source summary does not contain an AUC mapping")
    missing = [method for method in LEGACY_COMPARATORS if method not in auc]
    if missing:
        raise ValueError(f"multi-source summary is missing comparators: {', '.join(missing)}")
    return min(LEGACY_COMPARATORS, key=lambda method: (-float(auc[method]), method))


def select_parhelion(
    measurements: Iterable[Measurement],
    *,
    source_gpus: Sequence[str] = ("L4", "A10"),
    target_gpu: str = "T4",
    max_budget: int = MAX_BUDGET,
    seeds: int = SELECTION_SEEDS,
    jobs: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select each method on its own grid from one prepared validation replay."""
    records = tuple(measurements)
    sources = tuple(source_gpus)
    if sources != ("L4", "A10"):
        raise ValueError("the frozen selection protocol requires source_gpus=('L4', 'A10')")
    if target_gpu != "T4":
        raise ValueError("the frozen selection protocol requires target_gpu='T4'")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs <= 0:
        raise ValueError("jobs must be a positive integer")
    if seeds != SELECTION_SEEDS:
        raise ValueError(f"the frozen selection protocol requires exactly {SELECTION_SEEDS} seeds")
    if max_budget != MAX_BUDGET:
        raise ValueError(f"the frozen selection protocol requires max_budget={MAX_BUDGET}")

    context: _RawReplayContext = (records, sources, target_gpu, max_budget, seeds)
    prepared = _prepare_context(context)
    retrieval_candidates = tuple(
        ParhelionCandidate(k, temperature, 0.0) for k in K_GRID for temperature in TEMPERATURE_GRID
    )
    pooled_candidates = tuple(
        ParhelionCandidate(K_GRID[0], TEMPERATURE_GRID[0], transfer_strength)
        for transfer_strength in TRANSFER_STRENGTH_GRID
    )
    retrieval_jobs: tuple[_SelectionJob, ...] = tuple(
        ("retrieval", candidate, None, None) for candidate in retrieval_candidates
    )
    pooled_jobs: tuple[_SelectionJob, ...] = tuple(
        ("pooled", candidate, None, None) for candidate in pooled_candidates
    )

    retrieval_evaluations: dict[ParhelionCandidate, MethodEvaluation] = {}
    pooled_evaluations: dict[ParhelionCandidate, MethodEvaluation] = {}
    parhelion_evaluations: dict[ParhelionCandidate, MethodEvaluation] = {}
    if jobs == 1:
        retrieval_scores: dict[ParhelionCandidate, float] = {}
        for candidate in retrieval_candidates:
            evaluation = evaluate_multisource_retrieval(
                prepared,
                k=candidate.k,
                temperature=candidate.temperature,
            )
            retrieval_evaluations[candidate] = evaluation
            retrieval_scores[candidate] = evaluation_auc(prepared, evaluation)
        pooled_scores: dict[ParhelionCandidate, float] = {}
        for candidate in pooled_candidates:
            evaluation = evaluate_pooled_source(
                prepared,
                transfer_strength=candidate.transfer_strength,
            )
            pooled_evaluations[candidate] = evaluation
            pooled_scores[candidate] = evaluation_auc(prepared, evaluation)
    else:
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_initialize_worker,
            initargs=(context,),
        ) as executor:
            baseline_results = tuple(
                executor.map(_evaluate_worker, (*retrieval_jobs, *pooled_jobs))
            )
            retrieval_scores = {
                candidate: score
                for kind, candidate, score, _evaluation in baseline_results
                if kind == "retrieval"
            }
            retrieval_evaluations = {
                candidate: evaluation
                for kind, candidate, _score, evaluation in baseline_results
                if kind == "retrieval"
            }
            pooled_scores = {
                candidate: score
                for kind, candidate, score, _evaluation in baseline_results
                if kind == "pooled"
            }
            pooled_evaluations = {
                candidate: evaluation
                for kind, candidate, _score, evaluation in baseline_results
                if kind == "pooled"
            }
            retrieval_candidate = min(
                retrieval_candidates,
                key=lambda candidate: (
                    -retrieval_scores[candidate],
                    candidate.k,
                    candidate.temperature,
                ),
            )
            pooled_candidate = min(
                pooled_candidates,
                key=lambda candidate: (
                    -pooled_scores[candidate],
                    candidate.transfer_strength,
                ),
            )
            parhelion_jobs: tuple[_SelectionJob, ...] = tuple(
                (
                    "parhelion",
                    candidate,
                    retrieval_candidate.k,
                    retrieval_candidate.temperature,
                )
                for candidate in parhelion_grid()
            )
            parhelion_results = tuple(executor.map(_evaluate_worker, parhelion_jobs))
            parhelion_scores = {
                candidate: score for _kind, candidate, score, _evaluation in parhelion_results
            }
            parhelion_evaluations = {
                candidate: evaluation for _kind, candidate, _score, evaluation in parhelion_results
            }

    retrieval_candidate = min(
        retrieval_candidates,
        key=lambda candidate: (
            -retrieval_scores[candidate],
            candidate.k,
            candidate.temperature,
        ),
    )
    pooled_candidate = min(
        pooled_candidates,
        key=lambda candidate: (
            -pooled_scores[candidate],
            candidate.transfer_strength,
        ),
    )
    candidates = parhelion_grid()
    selected_retrieval = retrieval_evaluations.get(retrieval_candidate)
    if selected_retrieval is None:
        selected_retrieval = evaluate_multisource_retrieval(
            prepared,
            k=retrieval_candidate.k,
            temperature=retrieval_candidate.temperature,
        )
    selected_pooled = pooled_evaluations.get(pooled_candidate)
    if selected_pooled is None:
        selected_pooled = evaluate_pooled_source(
            prepared,
            transfer_strength=pooled_candidate.transfer_strength,
        )

    if jobs == 1:
        parhelion_scores = {}
        for candidate in candidates:
            evaluation = evaluate_parhelion(
                prepared,
                k=candidate.k,
                temperature=candidate.temperature,
                transfer_strength=candidate.transfer_strength,
                retrieval=selected_retrieval,
            )
            parhelion_evaluations[candidate] = evaluation
            parhelion_scores[candidate] = evaluation_auc(prepared, evaluation)
    parhelion_candidate = min(
        candidates,
        key=lambda candidate: (-parhelion_scores[candidate], candidate),
    )
    selected_parhelion = parhelion_evaluations.get(parhelion_candidate)
    if selected_parhelion is None:
        selected_parhelion = evaluate_parhelion(
            prepared,
            k=parhelion_candidate.k,
            temperature=parhelion_candidate.temperature,
            transfer_strength=parhelion_candidate.transfer_strength,
            retrieval=selected_retrieval,
        )
    selected_summary = assemble_multisource_summary(
        prepared,
        retrieval=selected_retrieval,
        pooled=selected_pooled,
        parhelion=selected_parhelion,
        k=parhelion_candidate.k,
        temperature=parhelion_candidate.temperature,
        transfer_strength=parhelion_candidate.transfer_strength,
        retrieval_k=retrieval_candidate.k,
        retrieval_temperature=retrieval_candidate.temperature,
        pooled_transfer_strength=pooled_candidate.transfer_strength,
        primary_comparator=None,
    )
    comparator = _select_comparator(selected_summary)
    selected_summary = assemble_multisource_summary(
        prepared,
        retrieval=selected_retrieval,
        pooled=selected_pooled,
        parhelion=selected_parhelion,
        k=parhelion_candidate.k,
        temperature=parhelion_candidate.temperature,
        transfer_strength=parhelion_candidate.transfer_strength,
        retrieval_k=retrieval_candidate.k,
        retrieval_temperature=retrieval_candidate.temperature,
        pooled_transfer_strength=pooled_candidate.transfer_strength,
        primary_comparator=comparator,
    )
    comparator_auc = float(selected_summary["auc"][comparator])

    selection = {
        "schema_version": 2,
        "protocol": "parhelion-v2-validation-selection",
        "source_gpus": list(sources),
        "validation_gpu": target_gpu,
        "final_gpu": "H100",
        "h100_invoked": False,
        "selection_stages": [
            "independent method-local baseline grids",
            "Parhelion grid with frozen retrieval anchor",
            "primary comparator selection",
        ],
        "parhelion_grid": {
            "k": list(K_GRID),
            "temperature": list(TEMPERATURE_GRID),
            "transfer_strength": list(TRANSFER_STRENGTH_GRID),
            "candidate_count": len(candidates),
        },
        "baseline_grids": {
            "single_source_nearest": {
                "source_gpu": sources[0],
                "neighbors": 1,
                "parameter_free": True,
            },
            "multisource_retrieval": {
                "k": list(K_GRID),
                "temperature": list(TEMPERATURE_GRID),
                "candidate_count": len(retrieval_candidates),
            },
            "pooled_source_thompson": {
                "transfer_strength": list(TRANSFER_STRENGTH_GRID),
                "candidate_count": len(pooled_candidates),
            },
        },
        "selection_seeds": list(range(seeds)),
        "final_evaluation_seeds": list(range(FINAL_SEEDS)),
        "budgets": list(range(1, max_budget + 1)),
        "selection_metric": (
            "mean fraction of held-out bank-1 reference, equally weighted over family-and-"
            "exact-shape-safe folds, seeds, and budgets 1 through 8"
        ),
        "tie_breaks": {
            "parhelion": "ascending (k, temperature, transfer_strength)",
            "multisource_retrieval": "ascending (k, temperature)",
            "pooled_source_thompson": "ascending transfer_strength",
            "primary_comparator": "ascending method name",
        },
        "comparator_candidates": list(LEGACY_COMPARATORS),
        "baseline_candidate_scores": [
            {
                "method": "multisource_retrieval",
                "k": candidate.k,
                "temperature": candidate.temperature,
                "auc": retrieval_scores[candidate],
            }
            for candidate in retrieval_candidates
        ]
        + [
            {
                "method": "pooled_source_thompson",
                "transfer_strength": candidate.transfer_strength,
                "auc": pooled_scores[candidate],
            }
            for candidate in pooled_candidates
        ],
        "candidate_scores": [
            {**asdict(candidate), "parhelion_thompson": parhelion_scores[candidate]}
            for candidate in candidates
        ],
        "selected": {
            "parhelion": {
                **asdict(parhelion_candidate),
                "auc": parhelion_scores[parhelion_candidate],
            },
            "multisource_retrieval": {
                "k": retrieval_candidate.k,
                "temperature": retrieval_candidate.temperature,
                "auc": retrieval_scores[retrieval_candidate],
            },
            "pooled_source_thompson": {
                "transfer_strength": pooled_candidate.transfer_strength,
                "auc": pooled_scores[pooled_candidate],
            },
            "single_source_nearest": {
                "source_gpu": sources[0],
                "neighbors": 1,
            },
            "primary_comparator": comparator,
            "primary_comparator_auc": comparator_auc,
        },
        "evaluator_counts": {
            "prepare_per_process": 1,
            "multisource_retrieval": len(retrieval_candidates),
            "pooled_source_thompson": len(pooled_candidates),
            "parhelion_thompson": len(candidates),
            "parameter_independent_baselines": 1,
        },
        "budget_one_invariant": (
            "Parhelion and multi-source retrieval query and recommend the same frozen retrieval "
            "anchor at budget 1; the query is charged to both methods."
        ),
        "final_source_archive": [*sources, target_gpu],
        "final_evaluation_rule": (
            "Hash a freeze artifact before one H100 collection; use 30 fixed seeds, every "
            "independently selected parameter, the frozen comparator, and no rerun or grid expansion."
        ),
        "disclosures": [
            "The T4 pilot is a collector smoke test and is excluded from selection.",
            "T4 selects global parameters and a comparator; it is not final-domain evidence.",
            "The final archive adds T4 to the two-source validation archive, changing source cost but not method logic.",
            "Budget b means b target configuration measurements per workload; posterior updates are shared across workloads within each held-out-family fold.",
            "The shape-disjoint L4 nearest baseline is v1-inspired, not an exact v1 replay.",
        ],
    }
    return selection, selected_summary


__all__ = [
    "FINAL_SEEDS",
    "K_GRID",
    "LEGACY_COMPARATORS",
    "MAX_BUDGET",
    "ParhelionCandidate",
    "SELECTION_SEEDS",
    "TEMPERATURE_GRID",
    "TRANSFER_STRENGTH_GRID",
    "parhelion_grid",
    "select_parhelion",
]
