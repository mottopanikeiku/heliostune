"""Explicit Monte Carlo and deterministic fold uncertainty summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from heliostune.errors import ProtocolError

_T_CRITICAL_95 = {
    11: 2.200985160,
    29: 2.045229642,
    49: 2.009575237,
}
_CONFIRMATORY_DEGREES_OF_FREEDOM = frozenset({29, 49})


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (tiny if abs(d) < tiny else d)
    fraction = d
    for step in range(1, 301):
        even = step * (b - step) * x / ((a + 2.0 * step - 1.0) * (a + 2.0 * step))
        d = 1.0 + even * d
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = 1.0 + even / c
        c = tiny if abs(c) < tiny else c
        fraction *= d * c
        odd = -(a + step) * (a + b + step) * x / ((a + 2.0 * step) * (a + 2.0 * step + 1.0))
        d = 1.0 + odd * d
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = 1.0 + odd / c
        c = tiny if abs(c) < tiny else c
        term = d * c
        fraction *= term
        if abs(term - 1.0) < 1e-16:
            break
    return fraction


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    prefactor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return prefactor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - prefactor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _exact_student_t_critical_95(degrees_of_freedom: int) -> float:
    """Bisect the two-sided Student-t CDF for the 95% critical value."""
    half_dof = degrees_of_freedom / 2.0
    low = 0.0
    high = 1000.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        x = degrees_of_freedom / (degrees_of_freedom + middle * middle)
        two_sided = 1.0 - _regularized_incomplete_beta(half_dof, 0.5, x)
        if two_sided < 0.95:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def student_t_critical_95(degrees_of_freedom: int) -> float:
    """Return the two-sided 95% Student-t critical value for one degrees-of-freedom count."""
    if type(degrees_of_freedom) is not int or degrees_of_freedom < 1:
        raise ProtocolError("degrees_of_freedom must be a positive integer")
    frozen = _T_CRITICAL_95.get(degrees_of_freedom)
    if frozen is not None:
        return frozen
    return _exact_student_t_critical_95(degrees_of_freedom)


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
    if degrees_of_freedom not in _CONFIRMATORY_DEGREES_OF_FREEDOM:
        raise ProtocolError(
            "confirmatory Monte Carlo intervals require exactly 30 or 50 policy seeds"
        )
    critical = _T_CRITICAL_95[degrees_of_freedom]
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


__all__ = [
    "deterministic_fold_summary",
    "paired_contrast",
    "stochastic_interval",
    "student_t_critical_95",
]
