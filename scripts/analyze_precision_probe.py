"""Answer whether FP16 reduced-precision reduction explains torch's H100 matmul lead.

Consumes the artifact written by ``modal_precision_probe.py`` and reports, per workload
and in aggregate, the ratios ``torch_reduced / torch_strict``, ``torch_strict / triton``
and ``torch_reduced / triton``, how many workloads Triton wins under each torch setting,
the same split by M bucket and by K, the accuracy of all three arms against the FP32
reference, and whether the archive's median torch/best-Triton ratio survives with
reduced-precision reduction disabled.

Status: post-hoc exploratory diagnostics. This is NOT a confirmatory Parhelion endpoint
and nothing it prints may be used to revise a published claim.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from heliostune.artifacts import read_json
from heliostune.validation import (
    exact_int,
    exact_object,
    finite_float,
    nonblank_string,
)

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_ARTIFACT = _REPO / "artifacts/h100-precision-probe.json"
_PROBE_NAME = "h100-fp16-reduction-probe"
_ARM_NAMES = ("torch_reduced", "torch_strict", "triton")
# Published endpoint: each config is selected on bank 1 and that selected config is
# scored on bank 2. The displayed endpoint is frozen at six decimal places.
_FROZEN_ARCHIVE_RATIO = 0.627266
_MEANINGFUL_PAIRED_EFFECT = 0.05
_BASELINE_RELATIVE_TOLERANCE = 0.05
_M_BUCKETS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("M = 1", lambda m: m == 1),
    ("M <= 8", lambda m: m <= 8),
    ("8 < M <= 128", lambda m: 8 < m <= 128),
    ("M > 128", lambda m: m > 128),
)


@dataclass(frozen=True, slots=True)
class WorkloadSummary:
    """One workload's per-arm medians over the probe's independent banks."""

    key: str
    m: int
    n: int
    k: int
    banks: tuple[int, ...]
    latencies: dict[str, float]
    errors: dict[str, float]
    archive_ratio: float | None

    def ratio(self, numerator: str, denominator: str) -> float:
        return self.latencies[numerator] / self.latencies[denominator]


@dataclass(frozen=True, slots=True)
class ExplanationVerdict:
    """Pure decision result for the exploratory speed and accuracy checks."""

    classification: str
    paired_strict_slowdown: float
    reduced_triton_ratio: float
    baseline_ratio: float
    baseline_absolute_tolerance: float
    baseline_agrees: bool
    accuracy_regression: bool
    parity_authorized: bool


def evaluate_explanation(
    summaries: Sequence[WorkloadSummary],
    *,
    baseline_ratio: float = _FROZEN_ARCHIVE_RATIO,
    meaningful_effect: float = _MEANINGFUL_PAIRED_EFFECT,
    baseline_relative_tolerance: float = _BASELINE_RELATIVE_TOLERANCE,
) -> ExplanationVerdict:
    """Classify the explanation using paired timing, baseline agreement, and accuracy."""
    if not summaries:
        raise ValueError("cannot evaluate an empty precision probe")
    if meaningful_effect <= 0.0 or baseline_relative_tolerance <= 0.0:
        raise ValueError("verdict tolerances must be positive")
    paired_effect = _median(
        [item.ratio("torch_strict", "torch_reduced") - 1.0 for item in summaries]
    )
    reduced_triton = _median([item.ratio("torch_reduced", "triton") for item in summaries])
    absolute_tolerance = baseline_ratio * baseline_relative_tolerance
    baseline_agrees = abs(reduced_triton - baseline_ratio) <= absolute_tolerance
    meaningful = paired_effect >= meaningful_effect
    if not meaningful:
        classification = "does not explain"
    elif not baseline_agrees:
        classification = "inconclusive"
    else:
        classification = "supports explanation"
    accuracy_regression = any(
        item.errors["torch_reduced"] > item.errors["torch_strict"] for item in summaries
    )
    return ExplanationVerdict(
        classification=classification,
        paired_strict_slowdown=paired_effect,
        reduced_triton_ratio=reduced_triton,
        baseline_ratio=baseline_ratio,
        baseline_absolute_tolerance=absolute_tolerance,
        baseline_agrees=baseline_agrees,
        accuracy_regression=accuracy_regression,
        parity_authorized=classification == "supports explanation" and not accuracy_regression,
    )


def load_summaries(artifact: Path) -> tuple[dict[str, Any], list[WorkloadSummary]]:
    """Validate the probe artifact and reduce its rows to one summary per workload."""
    data = exact_object(read_json(artifact), context="precision probe artifact")
    if nonblank_string(data.get("probe"), context="probe name") != _PROBE_NAME:
        raise ValueError(f"{artifact} was not written by {_PROBE_NAME}")
    if exact_int(data.get("schema_version"), context="probe schema_version", minimum=1) != 2:
        raise ValueError("precision probe artifact has an unsupported schema version")
    protocol = exact_object(data.get("protocol"), context="probe protocol")
    if protocol.get("gpu") != "H100" or protocol.get("banks") != [0, 1, 2]:
        raise ValueError("precision probe protocol is not the frozen H100 three-bank protocol")
    if protocol.get("warmup_ms") != 25 or protocol.get("rep_ms") != 100:
        raise ValueError("precision probe timing does not match the frozen protocol")
    if protocol.get("arms") != list(_ARM_NAMES):
        raise ValueError("precision probe protocol does not carry the exact arms")
    rows = data.get("rows")
    if type(rows) is not list or not rows:
        raise ValueError(f"{artifact} contains no probe rows")
    baseline = exact_object(data.get("archive_baseline"), context="probe archive_baseline")
    if baseline.get("selection_bank") != 1 or baseline.get("evaluation_bank") != 2:
        raise ValueError("archive baseline is not bank-1-selected and bank-2-scored")
    if baseline.get("published_endpoint_torch_over_best_triton") != _FROZEN_ARCHIVE_RATIO:
        raise ValueError("archive baseline does not label the frozen 0.627266 endpoint")
    recorded_baseline = finite_float(
        baseline.get("median_torch_over_best_triton"),
        context="archive baseline ratio",
        strictly_positive=True,
    )
    if abs(recorded_baseline - _FROZEN_ARCHIVE_RATIO) >= 0.000001:
        raise ValueError(
            f"archive baseline {recorded_baseline:.9f} does not match the frozen "
            f"{_FROZEN_ARCHIVE_RATIO:.6f} endpoint"
        )
    archive_workloads = exact_object(baseline.get("workloads"), context="archive workloads")
    latencies: dict[str, dict[str, list[float]]] = {}
    errors: dict[str, dict[str, list[float]]] = {}
    banks: dict[str, set[int]] = {}
    shapes: dict[str, tuple[int, int, int]] = {}
    seen: set[tuple[int, str]] = set()
    for row in rows:
        record = exact_object(row, context="probe row")
        key = nonblank_string(record.get("workload_key"), context="probe row workload_key")
        workload = exact_object(record.get("workload"), context="probe row workload")
        shape = (
            exact_int(workload.get("m"), context="workload m", minimum=1),
            exact_int(workload.get("n"), context="workload n", minimum=1),
            exact_int(workload.get("k"), context="workload k", minimum=1),
        )
        known_shape = shapes.setdefault(key, shape)
        if known_shape != shape:
            raise ValueError(f"probe rows disagree on the shape of {key}")
        row_bank = exact_int(record.get("bank"), context="probe row bank", minimum=0)
        pair = (row_bank, key)
        if pair in seen:
            raise ValueError(f"precision probe duplicates bank {row_bank} workload {key}")
        seen.add(pair)
        banks.setdefault(key, set()).add(row_bank)
        arms = exact_object(record.get("arms"), context="probe row arms")
        if set(arms) != set(_ARM_NAMES):
            raise ValueError(f"probe row {key} does not carry exactly the arms {_ARM_NAMES}")
        for arm in _ARM_NAMES:
            values = exact_object(arms[arm], context=f"probe arm {arm}")
            latencies.setdefault(key, {}).setdefault(arm, []).append(
                finite_float(
                    values.get("latency_ms"),
                    context=f"{key}/{arm} latency_ms",
                    strictly_positive=True,
                )
            )
            errors.setdefault(key, {}).setdefault(arm, []).append(
                finite_float(
                    values.get("max_abs_error"),
                    context=f"{key}/{arm} max_abs_error",
                    minimum=0,
                )
            )

    if any(values != {0, 1, 2} for values in banks.values()):
        raise ValueError("every precision-probe workload must have exactly banks 0, 1, and 2")
    if len(rows) != 3 * len(banks):
        raise ValueError("precision-probe rows are not exactly three banks × workloads")

    summaries: list[WorkloadSummary] = []
    for key in sorted(latencies):
        archive_row = archive_workloads.get(key)
        archive_ratio = (
            finite_float(
                exact_object(archive_row, context="archive workload").get("torch_over_best_triton"),
                context=f"archive {key} ratio",
                strictly_positive=True,
            )
            if archive_row is not None
            else None
        )
        m, n, k = shapes[key]
        summaries.append(
            WorkloadSummary(
                key,
                m=m,
                n=n,
                k=k,
                banks=tuple(sorted(banks[key])),
                latencies={
                    arm: statistics.median(values) for arm, values in latencies[key].items()
                },
                errors={arm: statistics.median(values) for arm, values in errors[key].items()},
                archive_ratio=archive_ratio,
            )
        )
    return data, summaries


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot take the median of an empty sample")
    return statistics.median(values)


def _triton_wins(summaries: Sequence[WorkloadSummary], torch_arm: str) -> int:
    return sum(1 for item in summaries if item.ratio(torch_arm, "triton") > 1.0)


def _print_group(label: str, summaries: Sequence[WorkloadSummary]) -> None:
    if not summaries:
        print(f"  {label:<14} (no workloads)")
        return
    reduced_strict = _median([item.ratio("torch_reduced", "torch_strict") for item in summaries])
    strict_triton = _median([item.ratio("torch_strict", "triton") for item in summaries])
    reduced_triton = _median([item.ratio("torch_reduced", "triton") for item in summaries])
    print(
        f"  {label:<14} n={len(summaries):>3}"
        f"  red/strict={reduced_strict:>7.4f}"
        f"  strict/triton={strict_triton:>7.4f}"
        f"  red/triton={reduced_triton:>7.4f}"
        f"  triton wins: {_triton_wins(summaries, 'torch_strict'):>3} strict /"
        f" {_triton_wins(summaries, 'torch_reduced'):>3} reduced"
    )


def _print_report(data: dict[str, Any], summaries: Sequence[WorkloadSummary]) -> None:
    protocol = exact_object(data.get("protocol"), context="probe protocol")
    print(f"{_PROBE_NAME}: post-hoc exploratory analysis, not a confirmatory endpoint")
    print(
        f"gpu={protocol.get('gpu')} banks={protocol.get('banks')} "
        f"warmup_ms={protocol.get('warmup_ms')} rep_ms={protocol.get('rep_ms')} "
        f"workloads={len(summaries)}"
    )
    print("ratios are latency quotients: > 1 means the denominator arm is faster")
    print()

    print("Per-workload medians over banks")
    # 59 is the longest key in heliostune.configs.DEFAULT_WORKLOADS.
    header = (
        f"  {'workload':<59} {'M':>5} {'K':>6} {'red/strict':>11} "
        f"{'strict/triton':>14} {'red/triton':>11} {'frozen b1→b2':>12}"
    )
    print(header)
    for item in summaries:
        archive = "     n/a" if item.archive_ratio is None else f"{item.archive_ratio:>8.4f}"
        print(
            f"  {item.key:<59} {item.m:>5} {item.k:>6}"
            f" {item.ratio('torch_reduced', 'torch_strict'):>11.4f}"
            f" {item.ratio('torch_strict', 'triton'):>14.4f}"
            f" {item.ratio('torch_reduced', 'triton'):>11.4f}"
            f" {archive}"
        )
    print()

    reduced_strict = _median([item.ratio("torch_reduced", "torch_strict") for item in summaries])
    strict_triton = _median([item.ratio("torch_strict", "triton") for item in summaries])
    reduced_triton = _median([item.ratio("torch_reduced", "triton") for item in summaries])
    print(f"Medians over {len(summaries)} workloads")
    print(f"  torch_reduced / torch_strict : {reduced_strict:.6f}")
    print(f"  torch_strict  / triton       : {strict_triton:.6f}")
    print(f"  torch_reduced / triton       : {reduced_triton:.6f}")
    print(
        "  frozen bank-1-selected / bank-2-scored torch / best triton"
        f" : {_FROZEN_ARCHIVE_RATIO:.6f}"
    )
    print()

    print("Triton wins (Triton faster than torch)")
    print(f"  vs torch_reduced : {_triton_wins(summaries, 'torch_reduced')} / {len(summaries)}")
    print(f"  vs torch_strict  : {_triton_wins(summaries, 'torch_strict')} / {len(summaries)}")
    print()

    print("By M bucket (M <= 8 includes M = 1 by construction)")
    for label, predicate in _M_BUCKETS:
        _print_group(label, [item for item in summaries if predicate(item.m)])
    print()

    print("By K")
    for k in sorted({item.k for item in summaries}):
        _print_group(f"K = {k}", [item for item in summaries if item.k == k])
    print()

    print("Accuracy against the FP32 reference (max absolute error, median over workloads)")
    for arm in _ARM_NAMES:
        values = [item.errors[arm] for item in summaries]
        print(
            f"  {arm:<14} median={_median(values):.6g}  max={max(values):.6g}"
            f"  min={min(values):.6g}"
        )
    less_accurate = sum(
        1 for item in summaries if item.errors["torch_reduced"] > item.errors["torch_strict"]
    )
    triton_worse = sum(
        1 for item in summaries if item.errors["triton"] > item.errors["torch_strict"]
    )
    print(f"  workloads where torch_reduced is less accurate than torch_strict: {less_accurate}")
    print(f"  workloads where triton is less accurate than torch_strict:        {triton_worse}")
    error_ratios = [
        item.errors["torch_reduced"] / item.errors["torch_strict"]
        for item in summaries
        if item.errors["torch_strict"] > 0
    ]
    if error_ratios:
        print(
            "  median torch_reduced / torch_strict max-abs-error ratio: "
            f"{_median(error_ratios):.6f}"
        )
    print()

    verdict = evaluate_explanation(summaries)
    print("Verdict")
    print(
        f"  paired strict-vs-reduced slowdown: {verdict.paired_strict_slowdown:.2%} "
        f"(meaningful threshold: {_MEANINGFUL_PAIRED_EFFECT:.2%})"
    )
    print(
        f"  reduced / Triton: {verdict.reduced_triton_ratio:.6f}; frozen "
        f"bank-1-selected/bank-2-scored baseline: {verdict.baseline_ratio:.6f}; "
        f"agreement tolerance: ±{verdict.baseline_absolute_tolerance:.6f} "
        f"({_BASELINE_RELATIVE_TOLERANCE:.1%} relative)"
    )
    print(f"  baseline agreement: {'yes' if verdict.baseline_agrees else 'no'}")
    print(f"  speed classification: {verdict.classification}")
    if verdict.accuracy_regression:
        print("  accuracy regression: yes; the speed finding cannot authorize numerical parity")
    else:
        print("  accuracy regression: no")
    print(
        f"  speed-and-accuracy parity authorization: {'yes' if verdict.parity_authorized else 'no'}"
    )
    print(
        "  This is exploratory evidence about a possible explanation, not a causal "
        "determination or a revision of any published Parhelion result."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="?",
        default=str(_DEFAULT_ARTIFACT),
        help="precision probe artifact written by modal_precision_probe.py",
    )
    arguments = parser.parse_args(argv)
    artifact = Path(arguments.artifact)
    if not artifact.is_file():
        raise SystemExit(f"precision probe artifact does not exist: {artifact}")
    data, summaries = load_summaries(artifact)
    if not summaries:
        raise SystemExit(f"precision probe artifact has no workloads: {artifact}")
    _print_report(data, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
