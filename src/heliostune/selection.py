"""Frozen T4 hyperparameter and comparator selection for Parhelion v2."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from heliostune.multisource import compare_multisource
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
_SCORED_METHODS: tuple[str, ...] = (
    "parhelion_thompson",
    "multisource_retrieval",
    "pooled_source_thompson",
)


@dataclass(frozen=True, order=True, slots=True)
class ParhelionCandidate:
    """One point in the protocol's global Parhelion hyperparameter grid."""

    k: int
    temperature: float
    transfer_strength: float

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.transfer_strength) or self.transfer_strength < 0.0:
            raise ValueError("transfer_strength must be finite and non-negative")


def parhelion_grid() -> tuple[ParhelionCandidate, ...]:
    """Return the 48 candidates in deterministic lexicographic order."""

    return tuple(
        ParhelionCandidate(k, temperature, transfer_strength)
        for k in K_GRID
        for temperature in TEMPERATURE_GRID
        for transfer_strength in TRANSFER_STRENGTH_GRID
    )


_ReplayContext = tuple[tuple[Measurement, ...], tuple[str, ...], str, int, int]
_SelectionJob = tuple[ParhelionCandidate, int, float, float]
_ScoredCandidate = tuple[ParhelionCandidate, dict[str, float]]
_WORKER_CONTEXT: _ReplayContext | None = None


def _evaluate_candidate(context: _ReplayContext, job: _SelectionJob) -> _ScoredCandidate:
    measurements, source_gpus, target_gpu, max_budget, seeds = context
    candidate, retrieval_k, retrieval_temperature, pooled_transfer_strength = job
    summary = compare_multisource(
        measurements,
        source_gpus=source_gpus,
        target_gpu=target_gpu,
        max_budget=max_budget,
        seeds=seeds,
        k=candidate.k,
        temperature=candidate.temperature,
        transfer_strength=candidate.transfer_strength,
        retrieval_k=retrieval_k,
        retrieval_temperature=retrieval_temperature,
        pooled_transfer_strength=pooled_transfer_strength,
        protocol_role="validation",
    )
    auc = summary["auc"]
    return candidate, {method: float(auc[method]) for method in _SCORED_METHODS}


def _initialize_worker(context: _ReplayContext) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _evaluate_worker(job: _SelectionJob) -> _ScoredCandidate:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("selection worker was not initialized")
    return _evaluate_candidate(_WORKER_CONTEXT, job)


def _run_jobs(
    context: _ReplayContext,
    job_specs: Sequence[_SelectionJob],
    jobs: int,
) -> tuple[_ScoredCandidate, ...]:
    if jobs == 1:
        return tuple(_evaluate_candidate(context, job) for job in job_specs)
    with ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_initialize_worker,
        initargs=(context,),
    ) as executor:
        return tuple(executor.map(_evaluate_worker, job_specs))


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
    """Select baseline parameters, Parhelion parameters, and one comparator on T4.

    Stage one independently selects the 12-point multi-source retrieval grid and
    four pooled-source strengths. Stage two fixes both winners, then selects
    Parhelion over its 48-point grid. Thus retrieval-only and Parhelion pay the
    same frozen consensus anchor at budget one for every candidate and final
    seed. Every score equally weights family folds, seeds, and budgets 1--8.
    """

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
    context: _ReplayContext = (records, sources, target_gpu, max_budget, seeds)

    retrieval_candidates = tuple(
        ParhelionCandidate(k, temperature, TRANSFER_STRENGTH_GRID[0])
        for k in K_GRID
        for temperature in TEMPERATURE_GRID
    )
    pooled_candidates = tuple(
        ParhelionCandidate(K_GRID[0], TEMPERATURE_GRID[0], transfer_strength)
        for transfer_strength in TRANSFER_STRENGTH_GRID
    )
    baseline_candidates = tuple(sorted(set(retrieval_candidates) | set(pooled_candidates)))
    baseline_jobs: tuple[_SelectionJob, ...] = tuple(
        (
            candidate,
            candidate.k,
            candidate.temperature,
            candidate.transfer_strength,
        )
        for candidate in baseline_candidates
    )
    baseline_scores = _run_jobs(context, baseline_jobs, jobs)
    retrieval_candidate, retrieval_method_scores = min(
        (
            (candidate, method_scores)
            for candidate, method_scores in baseline_scores
            if candidate.transfer_strength == TRANSFER_STRENGTH_GRID[0]
        ),
        key=lambda item: (
            -item[1]["multisource_retrieval"],
            item[0].k,
            item[0].temperature,
        ),
    )
    pooled_candidate, pooled_method_scores = min(
        (
            (candidate, method_scores)
            for candidate, method_scores in baseline_scores
            if candidate.k == K_GRID[0] and candidate.temperature == TEMPERATURE_GRID[0]
        ),
        key=lambda item: (-item[1]["pooled_source_thompson"], item[0].transfer_strength),
    )

    candidates = parhelion_grid()
    parhelion_jobs: tuple[_SelectionJob, ...] = tuple(
        (
            candidate,
            retrieval_candidate.k,
            retrieval_candidate.temperature,
            pooled_candidate.transfer_strength,
        )
        for candidate in candidates
    )
    parhelion_scores = _run_jobs(context, parhelion_jobs, jobs)
    parhelion_candidate, parhelion_method_scores = min(
        parhelion_scores,
        key=lambda item: (-item[1]["parhelion_thompson"], item[0]),
    )

    selected_summary = compare_multisource(
        records,
        source_gpus=sources,
        target_gpu=target_gpu,
        max_budget=max_budget,
        seeds=seeds,
        k=parhelion_candidate.k,
        temperature=parhelion_candidate.temperature,
        transfer_strength=parhelion_candidate.transfer_strength,
        retrieval_k=retrieval_candidate.k,
        retrieval_temperature=retrieval_candidate.temperature,
        pooled_transfer_strength=pooled_candidate.transfer_strength,
        protocol_role="validation",
    )
    comparator = _select_comparator(selected_summary)
    comparator_auc = float(selected_summary["auc"][comparator])

    selection = {
        "schema_version": 1,
        "protocol": "parhelion-v2-validation-selection",
        "source_gpus": list(sources),
        "validation_gpu": target_gpu,
        "final_gpu": "H100",
        "h100_invoked": False,
        "selection_stages": [
            "independent baseline grids",
            "Parhelion grid with frozen retrieval anchor and pooled baseline",
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
            "mean fraction of held-out bank-1 reference, equally weighted over model-family "
            "folds, seeds, and budgets 1 through 8"
        ),
        "tie_breaks": {
            "parhelion": "ascending (k, temperature, transfer_strength)",
            "multisource_retrieval": "ascending (k, temperature)",
            "pooled_source_thompson": "ascending transfer_strength",
            "primary_comparator": "ascending method name",
        },
        "comparator_candidates": list(LEGACY_COMPARATORS),
        "baseline_candidate_scores": [
            {**asdict(candidate), **method_scores} for candidate, method_scores in baseline_scores
        ],
        "candidate_scores": [
            {**asdict(candidate), **method_scores} for candidate, method_scores in parhelion_scores
        ],
        "selected": {
            "parhelion": {
                **asdict(parhelion_candidate),
                "auc": parhelion_method_scores["parhelion_thompson"],
            },
            "multisource_retrieval": {
                "k": retrieval_candidate.k,
                "temperature": retrieval_candidate.temperature,
                "auc": retrieval_method_scores["multisource_retrieval"],
            },
            "pooled_source_thompson": {
                "transfer_strength": pooled_candidate.transfer_strength,
                "auc": pooled_method_scores["pooled_source_thompson"],
            },
            "single_source_nearest": {
                "source_gpu": sources[0],
                "neighbors": 1,
            },
            "primary_comparator": comparator,
            "primary_comparator_auc": comparator_auc,
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
