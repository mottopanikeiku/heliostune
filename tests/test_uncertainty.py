from __future__ import annotations

import math

import numpy as np
import pytest

from heliostune.errors import ProtocolError
from heliostune.uncertainty import (
    _exact_student_t_critical_95,
    deterministic_fold_summary,
    paired_contrast,
    stochastic_interval,
    student_t_critical_95,
)

_CONDITIONAL = "fixed H100 matrix, corpus, archive, and campaign"


def test_thirty_seed_interval_uses_frozen_student_t_critical_value() -> None:
    values = [float(index) / 100 for index in range(30)]

    summary = stochastic_interval(
        values,
        estimand="policy AUC1-8",
        conditional_on=_CONDITIONAL,
    )

    mean = float(np.mean(values))
    half_width = 2.045229642 * float(np.std(values, ddof=1)) / math.sqrt(30)
    assert summary["mean"] == pytest.approx(mean)
    assert summary["uncertainty"] == {
        "estimand": "policy AUC1-8",
        "sampling_unit": "paired policy seed",
        "n": 30,
        "conditional_on": _CONDITIONAL,
        "interval_method": (
            "two-sided 95% Student-t Monte Carlo interval over paired policy seeds"
        ),
        "low": pytest.approx(mean - half_width),
        "high": pytest.approx(mean + half_width),
    }


def test_frozen_table_values_are_returned_exactly() -> None:
    assert student_t_critical_95(11) == 2.200985160
    assert student_t_critical_95(29) == 2.045229642
    assert student_t_critical_95(49) == 2.009575237


def test_frozen_table_values_are_the_real_quantiles() -> None:
    for degrees_of_freedom in (11, 29, 49):
        assert _exact_student_t_critical_95(degrees_of_freedom) == pytest.approx(
            student_t_critical_95(degrees_of_freedom), abs=5e-10
        )


def test_offprotocol_dof_uses_the_exact_quantile_not_the_normal_approximation() -> None:
    assert student_t_critical_95(19) == pytest.approx(2.0930240544, abs=1e-9)


def test_quantile_extremes_and_normal_convergence() -> None:
    assert student_t_critical_95(1) == pytest.approx(12.7062047362, abs=1e-9)
    assert student_t_critical_95(100000) == pytest.approx(1.959988, abs=1e-5)


@pytest.mark.parametrize("degrees_of_freedom", [0, -1, True])
def test_invalid_degrees_of_freedom_are_rejected(degrees_of_freedom: int) -> None:
    with pytest.raises(ProtocolError, match="positive integer"):
        student_t_critical_95(degrees_of_freedom)


def test_nonprotocol_seed_count_cannot_emit_confirmatory_interval() -> None:
    with pytest.raises(ProtocolError, match="exactly 30 or 50"):
        stochastic_interval(
            [0.1, 0.2, 0.3],
            estimand="invalid endpoint",
            conditional_on=_CONDITIONAL,
        )


def test_deterministic_fold_summary_is_descriptive_not_interval() -> None:
    summary = deterministic_fold_summary(
        [0.8, 0.9, 1.0, 1.1],
        estimand="static budget-8 fraction",
        conditional_on=_CONDITIONAL,
    )

    assert summary["mean"] == pytest.approx(0.95)
    assert summary["min"] == 0.8
    assert summary["max"] == 1.1
    assert summary["sample_std"] == pytest.approx(np.std([0.8, 0.9, 1.0, 1.1], ddof=1))
    assert "uncertainty" not in summary
    assert summary["descriptive"]["summary_method"].endswith("not an interval")


def test_exploratory_paired_contrast_has_no_superiority_claim() -> None:
    contrast = paired_contrast(
        [0.9 + index / 1000 for index in range(30)],
        [0.8 + index / 1000 for index in range(30)],
        estimand="Parhelion minus anchored cold AUC1-8",
        conditional_on=_CONDITIONAL,
        analysis_status="post_hoc_exploratory",
    )

    assert contrast["mean"] == pytest.approx(0.1)
    assert contrast["analysis_status"] == "post_hoc_exploratory"
    assert contrast["superiority_supported"] is None
    assert contrast["claim"] is None
