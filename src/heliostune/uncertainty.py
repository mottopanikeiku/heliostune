"""Explicit Monte Carlo and deterministic fold uncertainty summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from heliostune.errors import ProtocolError

_T_CRITICAL_95 = {
    29: 2.045229642,
    49: 2.009575237,
}


def _finite_values(values: Sequence[float], *, context: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.all(np.isfinite(result)):
        raise ProtocolError(f"{context} must be a nonempty finite vector")
    return result


def stochastic_interval(
    values: Sequence[float],
    *,
    estimand: str,
    conditional_on: str,
) -> dict[str, object]:
    """Summarize 30- or 50-seed Monte Carlo values with the frozen t critical value."""
    vector = _finite_values(values, context=estimand)
    degrees_of_freedom = len(vector) - 1
    try:
        critical = _T_CRITICAL_95[degrees_of_freedom]
    except KeyError as exc:
        raise ProtocolError(
            "confirmatory Monte Carlo intervals require exactly 30 or 50 policy seeds"
        ) from exc
    mean = float(np.mean(vector))
    half_width = float(critical * np.std(vector, ddof=1) / math.sqrt(len(vector)))
    return {
        "mean": mean,
        "values": [float(value) for value in vector],
        "uncertainty": {
            "estimand": estimand,
            "sampling_unit": "paired policy seed",
            "n": len(vector),
            "conditional_on": conditional_on,
            "interval_method": (
                "two-sided 95% Student-t Monte Carlo interval over paired policy seeds"
            ),
            "low": mean - half_width,
            "high": mean + half_width,
        },
    }


def paired_contrast(
    left: Sequence[float],
    right: Sequence[float],
    *,
    estimand: str,
    conditional_on: str,
    analysis_status: str,
) -> dict[str, object]:
    """Return a same-seed left-minus-right contrast without a superiority claim."""
    left_values = _finite_values(left, context=f"{estimand} left")
    right_values = _finite_values(right, context=f"{estimand} right")
    if left_values.shape != right_values.shape:
        raise ProtocolError(f"{estimand} paired vectors have different shapes")
    summary = stochastic_interval(
        [float(value) for value in left_values - right_values],
        estimand=estimand,
        conditional_on=conditional_on,
    )
    summary["analysis_status"] = analysis_status
    summary["superiority_supported"] = None
    summary["claim"] = None
    return summary


def deterministic_fold_summary(
    values: Sequence[float],
    *,
    estimand: str,
    conditional_on: str,
) -> dict[str, object]:
    """Describe exactly four equal-weight folds without an interval claim."""
    vector = _finite_values(values, context=estimand)
    if len(vector) != 4:
        raise ProtocolError("deterministic summaries require exactly four fold values")
    return {
        "mean": float(np.mean(vector)),
        "fold_values": [float(value) for value in vector],
        "min": float(np.min(vector)),
        "max": float(np.max(vector)),
        "sample_std": float(np.std(vector, ddof=1)),
        "descriptive": {
            "estimand": estimand,
            "sampling_unit": "equal-weight held-out model-family fold",
            "n": 4,
            "conditional_on": conditional_on,
            "summary_method": "minimum, maximum, and sample standard deviation; not an interval",
        },
    }


__all__ = ["deterministic_fold_summary", "paired_contrast", "stochastic_interval"]
