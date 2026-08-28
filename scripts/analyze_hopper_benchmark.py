"""Analyze the one-bank H100 engineering benchmark expansion gate.

This is post-hoc exploratory engineering evidence, not a confirmatory endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from heliostune.artifacts import read_json, read_measurements, strict_json_dumps
from heliostune.configs import (
    DEFAULT_CONFIGS,
    DEFAULT_WORKLOADS,
    HOPPER_GEMM_CONFIGS,
    SKINNY_GEMV_CONFIGS,
    HopperGemmConfig,
    SkinnyGemvConfig,
    Workload,
)
from heliostune.hardware import expectation_for_gpu, validate_hardware
from heliostune.schema import HardwareProfile, Measurement
from heliostune.validation import (
    exact_bool,
    exact_fields,
    exact_int,
    finite_float,
    nonblank_string,
)

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_ARTIFACT = _REPO / "artifacts/hopper-h100-engineering.json"
_ARCHIVE = _REPO / "benchmarks/data/parhelion-v2-measurements.jsonl.zst"
_STUDY_ID = "hopper-h100-engineering-benchmark"
_ANALYSIS_STATUS = "post_hoc_exploratory"
_REGIMES = ("skinny_gemv", "hopper_gemm")
_GATE_SPEEDUP = 1.05
_GATE_WIN_FRACTION = 0.25
_EXPECTED_WORKLOADS = 96
_EXPECTED_ROWS = 32 * 48 + 64 * 23
_TIMING_FIELDS = ("p20_ms", "median_ms", "p80_ms", "wall_ms")
_TOP_FIELDS = (
    "schema_version",
    "study_id",
    "analysis_status",
    "gpu",
    "gpu_selector",
    "hardware",
    "bank",
    "protocol",
    "correctness_gate",
    "configs",
    "config_manifest_sha256",
    "workloads",
    "rows",
    "verified",
)
_ROW_FIELDS = (
    "workload_key",
    "workload",
    "regime",
    "config_kind",
    "config_key",
    "config",
    "bank",
    "seed",
    "latency",
    "torch",
    "correct",
    "max_abs_error",
)


@dataclass(frozen=True, slots=True)
class WorkloadAnalysis:
    """The selected bank-0 candidate and latency ratio for one workload."""

    workload_key: str
    regime: str
    best_config_key: str
    best_config: dict[str, int | bool]
    best_candidate_ms: float
    torch_ms: float
    torch_over_best_candidate: float
    archive_torch_over_best_triton: float | None
    correct: bool


@dataclass(frozen=True, slots=True)
class RegimeAnalysis:
    """Aggregate expansion-gate statistics for one candidate regime."""

    regime: str
    workload_count: int
    geometric_mean_speedup: float
    median_speedup: float
    minimum_speedup: float
    maximum_speedup: float
    workloads_at_least_five_percent_faster: int
    percent_at_least_five_percent_faster: float
    all_selected_correct: bool
    passes_gate: bool


@dataclass(frozen=True, slots=True)
class BenchmarkAnalysis:
    """Pure structured result of validating and analyzing an artifact."""

    study_id: str
    analysis_status: str
    gpu: str
    bank: int
    row_count: int
    workloads: tuple[WorkloadAnalysis, ...]
    regimes: tuple[RegimeAnalysis, ...]
    proceed: bool

    def regime(self, name: str) -> RegimeAnalysis:
        """Return one named regime result."""
        for result in self.regimes:
            if result.regime == name:
                return result
        raise KeyError(name)


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _validate_sha256(value: object, *, context: str) -> str:
    digest = nonblank_string(value, context=context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _validate_timing(value: object, *, context: str) -> dict[str, float]:
    timing = exact_fields(value, required=_TIMING_FIELDS, context=context)
    result = {
        field: finite_float(
            timing[field],
            context=f"{context} {field}",
            strictly_positive=True,
        )
        for field in _TIMING_FIELDS
    }
    if not result["p20_ms"] <= result["median_ms"] <= result["p80_ms"]:
        _fail(f"{context} quantiles are not ordered p20 <= median <= p80")
    return result


def _expected_seed_by_workload() -> dict[str, int]:
    ordered = list(DEFAULT_WORKLOADS)
    random.Random(0).shuffle(ordered)
    return {workload.key: index for index, workload in enumerate(ordered)}


def _validate_protocol(value: object) -> None:
    fields = (
        "warmup_ms",
        "rep_ms",
        "quantiles",
        "candidate_policy",
        "expected_workloads",
        "expected_skinny_workloads",
        "expected_hopper_workloads",
        "expected_skinny_rows",
        "expected_hopper_rows",
        "expected_candidate_rows",
        "torch_measurements",
    )
    protocol = exact_fields(value, required=fields, context="benchmark protocol")
    expected_counts = {
        "warmup_ms": 25,
        "rep_ms": 100,
        "expected_workloads": 96,
        "expected_skinny_workloads": 32,
        "expected_hopper_workloads": 64,
        "expected_skinny_rows": 1536,
        "expected_hopper_rows": 1472,
        "expected_candidate_rows": 3008,
        "torch_measurements": 96,
    }
    for field, expected in expected_counts.items():
        actual = exact_int(protocol[field], context=f"benchmark protocol {field}", minimum=0)
        if actual != expected:
            _fail(f"benchmark protocol {field} must be exactly {expected}")
    if protocol["quantiles"] != [0.2, 0.5, 0.8]:
        _fail("benchmark protocol must use exactly the p20/median/p80 quantiles")
    policy = exact_fields(
        protocol["candidate_policy"],
        required=_REGIMES,
        context="candidate policy",
    )
    expected_policy = {
        "skinny_gemv": {
            "condition": "m <= 8",
            "config_set": "SKINNY_GEMV_CONFIGS",
            "config_count": 48,
        },
        "hopper_gemm": {
            "condition": "m > 8",
            "config_set": "HOPPER_GEMM_CONFIGS",
            "config_count": 23,
        },
    }
    for regime, policy_expected in expected_policy.items():
        entry = exact_fields(
            policy[regime],
            required=("condition", "config_set", "config_count"),
            context=f"{regime} candidate policy",
        )
        condition = nonblank_string(
            entry["condition"],
            context=f"{regime} candidate policy condition",
        )
        config_set = nonblank_string(
            entry["config_set"],
            context=f"{regime} candidate policy config_set",
        )
        config_count = exact_int(
            entry["config_count"],
            context=f"{regime} candidate policy config_count",
            minimum=1,
        )
        if (
            condition != policy_expected["condition"]
            or config_set != policy_expected["config_set"]
            or config_count != policy_expected["config_count"]
        ):
            _fail(f"candidate policy for {regime} does not match the frozen policy")


def _validate_correctness_gate(value: object) -> None:
    gate = exact_fields(
        value,
        required=("artifact", "artifact_sha256", "manifest", "manifest_sha256"),
        context="correctness gate",
    )
    nonblank_string(gate["artifact"], context="correctness gate artifact")
    nonblank_string(gate["manifest"], context="correctness gate manifest")
    _validate_sha256(gate["artifact_sha256"], context="correctness gate artifact_sha256")
    _validate_sha256(gate["manifest_sha256"], context="correctness gate manifest_sha256")


def _config_catalog() -> dict[str, tuple[HopperGemmConfig | SkinnyGemvConfig, ...]]:
    return {
        "hopper_gemm": tuple(HOPPER_GEMM_CONFIGS),
        "skinny_gemv": tuple(SKINNY_GEMV_CONFIGS),
    }


def _parse_config(
    regime: str,
    value: object,
) -> HopperGemmConfig | SkinnyGemvConfig:
    if regime == "hopper_gemm":
        return HopperGemmConfig.from_dict(value)
    if regime == "skinny_gemv":
        return SkinnyGemvConfig.from_dict(value)
    _fail(f"unsupported config regime {regime!r}")


def _validate_configs(
    value: object,
    digest_value: object,
) -> dict[str, dict[str, HopperGemmConfig | SkinnyGemvConfig]]:
    manifest = exact_fields(value, required=_REGIMES, context="config manifest")
    canonical_manifest: dict[str, list[dict[str, int | bool]]] = {}
    result: dict[str, dict[str, HopperGemmConfig | SkinnyGemvConfig]] = {}
    for regime, expected_configs in _config_catalog().items():
        raw_configs = manifest[regime]
        if type(raw_configs) is not list:
            _fail(f"config manifest {regime} must be a list")
        parsed_configs = tuple(
            _parse_config(regime, raw_config) for raw_config in cast(list[object], raw_configs)
        )
        if parsed_configs != expected_configs:
            _fail(f"config manifest {regime} does not match the frozen config order")
        canonical_manifest[regime] = [config.to_dict() for config in parsed_configs]
        result[regime] = {config.key: config for config in expected_configs}

    recorded_digest = _validate_sha256(
        digest_value,
        context="config_manifest_sha256",
    )
    actual_digest = hashlib.sha256(
        strict_json_dumps(canonical_manifest, compact=True).encode("utf-8")
    ).hexdigest()
    if recorded_digest != actual_digest:
        _fail("config_manifest_sha256 does not match canonical configs")
    return result


def _validate_workload_manifest(value: object) -> None:
    if type(value) is not list:
        _fail("workloads must be a list")
    workloads = tuple(
        Workload.from_dict(raw_workload) for raw_workload in cast(list[object], value)
    )
    if workloads != tuple(DEFAULT_WORKLOADS):
        _fail("workloads do not match the exact 96 DEFAULT_WORKLOADS in frozen order")


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        _fail("cannot aggregate an empty regime")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def analyze_artifact(
    artifact: object,
    *,
    archive_ratios: Mapping[str, float] | None = None,
) -> BenchmarkAnalysis:
    """Validate and analyze a decoded artifact without performing any I/O."""
    data = exact_fields(artifact, required=_TOP_FIELDS, context="H100 benchmark artifact")
    if exact_int(data["schema_version"], context="schema_version", minimum=1) != 1:
        _fail("unsupported H100 benchmark schema_version")
    if nonblank_string(data["study_id"], context="study_id") != _STUDY_ID:
        _fail(f"artifact study_id must be {_STUDY_ID!r}")
    if nonblank_string(data["analysis_status"], context="analysis_status") != _ANALYSIS_STATUS:
        _fail("artifact analysis_status must be post_hoc_exploratory")
    if data["gpu"] != "H100" or data["gpu_selector"] != "H100!":
        _fail("artifact must be the bounded H100/H100! collection")
    if exact_int(data["bank"], context="bank", minimum=0) != 0:
        _fail("artifact must contain bank 0 only")
    if exact_bool(data["verified"], context="verified") is not True:
        _fail("artifact must be locally validated and verified=true")
    hardware = HardwareProfile.from_dict(data["hardware"])
    validate_hardware(hardware, expectation_for_gpu("H100"))
    runtime_fields = {
        "cuda_version": hardware.cuda_version,
        "torch_version": hardware.torch_version,
        "triton_version": hardware.triton_version,
    }
    missing_runtime = [name for name, runtime in runtime_fields.items() if runtime is None]
    if missing_runtime:
        _fail(f"hardware profile has incomplete runtime fields {missing_runtime!r}")
    _validate_protocol(data["protocol"])
    _validate_correctness_gate(data["correctness_gate"])
    configs = _validate_configs(data["configs"], data["config_manifest_sha256"])
    _validate_workload_manifest(data["workloads"])

    raw_rows = data["rows"]
    if type(raw_rows) is not list:
        _fail("rows must be a list")
    rows = cast(list[object], raw_rows)
    if len(rows) != _EXPECTED_ROWS:
        _fail(f"artifact must contain exactly {_EXPECTED_ROWS} candidate rows")

    expected_workloads = {workload.key: workload for workload in DEFAULT_WORKLOADS}
    expected_seeds = _expected_seed_by_workload()
    seen: set[tuple[str, str]] = set()
    rows_by_workload: dict[
        str,
        list[
            tuple[
                str,
                HopperGemmConfig | SkinnyGemvConfig,
                float,
                float,
                bool,
            ]
        ],
    ] = {}
    torch_by_workload: dict[str, dict[str, float]] = {}
    seed_by_workload: dict[str, int] = {}
    regime_workloads: dict[str, set[str]] = {regime: set() for regime in _REGIMES}

    for index, raw_row in enumerate(rows):
        context = f"row {index}"
        row = exact_fields(raw_row, required=_ROW_FIELDS, context=context)
        workload_key = nonblank_string(row["workload_key"], context=f"{context} workload_key")
        workload = expected_workloads.get(workload_key)
        if workload is None:
            _fail(f"{context} has unexpected workload {workload_key!r}")
        row_workload = Workload.from_dict(row["workload"])
        if row_workload != workload:
            _fail(f"{context} workload payload does not match {workload_key}")
        expected_regime = "skinny_gemv" if workload.m <= 8 else "hopper_gemm"
        regime = nonblank_string(row["regime"], context=f"{context} regime")
        config_kind = nonblank_string(row["config_kind"], context=f"{context} config_kind")
        if regime != expected_regime or config_kind != expected_regime:
            _fail(f"{context} has wrong regime/config_kind for {workload_key}")
        config_key = nonblank_string(row["config_key"], context=f"{context} config_key")
        config = configs[regime].get(config_key)
        if config is None:
            _fail(f"{context} has unexpected {regime} config {config_key!r}")
        row_config = _parse_config(regime, row["config"])
        if row_config != config:
            _fail(f"{context} config payload does not match {config_key}")
        pair = (workload_key, config_key)
        if pair in seen:
            _fail(f"artifact duplicates workload/config row {workload_key}/{config_key}")
        seen.add(pair)
        if exact_int(row["bank"], context=f"{context} bank", minimum=0) != 0:
            _fail(f"{context} is not bank 0")
        seed = exact_int(row["seed"], context=f"{context} seed", minimum=0)
        if seed != expected_seeds[workload_key]:
            _fail(f"{context} seed does not match frozen bank-0 tensor seeding")
        previous_seed = seed_by_workload.setdefault(workload_key, seed)
        if previous_seed != seed:
            _fail(f"rows for {workload_key} disagree on seed")
        latency = _validate_timing(row["latency"], context=f"{context} latency")
        torch = _validate_timing(row["torch"], context=f"{context} torch")
        previous_torch = torch_by_workload.setdefault(workload_key, torch)
        if previous_torch != torch:
            _fail(f"rows for {workload_key} disagree on the once-per-workload torch baseline")
        correct = exact_bool(row["correct"], context=f"{context} correct")
        if not correct:
            _fail(f"{context} is incorrect; fail-closed analysis refuses partial correctness")
        finite_float(
            row["max_abs_error"],
            context=f"{context} max_abs_error",
            minimum=0.0,
        )
        regime_workloads[regime].add(workload_key)
        rows_by_workload.setdefault(workload_key, []).append(
            (config_key, row_config, latency["median_ms"], torch["median_ms"], correct)
        )

    expected_pairs = {
        (workload.key, config.key)
        for workload in DEFAULT_WORKLOADS
        for config in (SKINNY_GEMV_CONFIGS if workload.m <= 8 else HOPPER_GEMM_CONFIGS)
    }
    if seen != expected_pairs:
        missing = sorted(expected_pairs - seen)
        extra = sorted(seen - expected_pairs)
        _fail(f"workload/config coverage mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}")
    if len(regime_workloads["skinny_gemv"]) != 32 or len(regime_workloads["hopper_gemm"]) != 64:
        _fail("artifact must contain exactly 32 skinny_gemv and 64 hopper_gemm workloads")
    for workload_key, candidates in rows_by_workload.items():
        expected_count = 48 if expected_workloads[workload_key].m <= 8 else 23
        if len(candidates) != expected_count:
            _fail(f"{workload_key} must contain exactly {expected_count} candidates")

    normalized_archive: dict[str, float] = {}
    if archive_ratios is not None:
        for key, value in archive_ratios.items():
            if key in expected_workloads:
                normalized_archive[key] = finite_float(
                    value,
                    context=f"archive ratio {key}",
                    strictly_positive=True,
                )

    workload_results: list[WorkloadAnalysis] = []
    for workload in DEFAULT_WORKLOADS:
        candidates = rows_by_workload[workload.key]
        best_key, best_config, best_ms, torch_ms, correct = min(
            candidates,
            key=lambda candidate: (candidate[2], candidate[0]),
        )
        workload_results.append(
            WorkloadAnalysis(
                workload_key=workload.key,
                regime="skinny_gemv" if workload.m <= 8 else "hopper_gemm",
                best_config_key=best_key,
                best_config=dict(best_config.to_dict()),
                best_candidate_ms=best_ms,
                torch_ms=torch_ms,
                torch_over_best_candidate=torch_ms / best_ms,
                archive_torch_over_best_triton=normalized_archive.get(workload.key),
                correct=correct,
            )
        )

    regime_results: list[RegimeAnalysis] = []
    for regime in _REGIMES:
        selected = [result for result in workload_results if result.regime == regime]
        speedups = [result.torch_over_best_candidate for result in selected]
        wins = sum(speedup >= _GATE_SPEEDUP for speedup in speedups)
        all_correct = all(result.correct for result in selected)
        geometric_mean = _geometric_mean(speedups)
        passes = (
            geometric_mean >= _GATE_SPEEDUP
            and wins / len(selected) >= _GATE_WIN_FRACTION
            and all_correct
        )
        regime_results.append(
            RegimeAnalysis(
                regime=regime,
                workload_count=len(selected),
                geometric_mean_speedup=geometric_mean,
                median_speedup=statistics.median(speedups),
                minimum_speedup=min(speedups),
                maximum_speedup=max(speedups),
                workloads_at_least_five_percent_faster=wins,
                percent_at_least_five_percent_faster=100.0 * wins / len(selected),
                all_selected_correct=all_correct,
                passes_gate=passes,
            )
        )

    return BenchmarkAnalysis(
        study_id=_STUDY_ID,
        analysis_status=_ANALYSIS_STATUS,
        gpu="H100",
        bank=0,
        row_count=len(rows),
        workloads=tuple(workload_results),
        regimes=tuple(regime_results),
        proceed=any(result.passes_gate for result in regime_results),
    )


def load_archive_ratios(path: Path = _ARCHIVE) -> dict[str, float]:
    """Recreate the frozen bank-1-selected/bank-2-scored contextual ratios."""
    expected_workloads = {workload.key for workload in DEFAULT_WORKLOADS}
    expected_configs = {config.key for config in DEFAULT_CONFIGS}
    cells: dict[tuple[str, str, int], Measurement] = {}
    for row in read_measurements(path):
        if row.hardware.gpu != "H100" or row.bank not in {1, 2}:
            continue
        if row.workload.key not in expected_workloads or row.config.key not in expected_configs:
            _fail(f"archive contains unexpected H100 cell {row.workload.key}/{row.config.key}")
        key = (row.workload.key, row.config.key, row.bank)
        if key in cells:
            _fail(f"archive duplicates H100 cell {key!r}")
        if not row.usable or row.latency_ms is None or row.torch_latency_ms is None:
            _fail(f"archive contains unusable H100 cell {key!r}")
        finite_float(row.latency_ms, context=f"archive latency {key!r}", strictly_positive=True)
        finite_float(
            row.torch_latency_ms,
            context=f"archive torch latency {key!r}",
            strictly_positive=True,
        )
        cells[key] = row
    expected_cells = {
        (workload, config, bank)
        for workload in expected_workloads
        for config in expected_configs
        for bank in (1, 2)
    }
    if set(cells) != expected_cells:
        _fail("archive does not contain the exact frozen H100 bank-1/bank-2 cells")
    ratios: dict[str, float] = {}
    for workload_key in sorted(expected_workloads):
        best_key = min(
            expected_configs,
            key=lambda config_key: (
                finite_float(
                    cells[(workload_key, config_key, 1)].latency_ms,
                    context=f"archive selection latency {workload_key}/{config_key}",
                    strictly_positive=True,
                ),
                config_key,
            ),
        )
        selected = cells[(workload_key, best_key, 2)]
        baseline = cells[(workload_key, DEFAULT_CONFIGS[0].key, 2)]
        ratios[workload_key] = finite_float(
            baseline.torch_latency_ms,
            context=f"archive torch latency {workload_key}",
            strictly_positive=True,
        ) / finite_float(
            selected.latency_ms,
            context=f"archive selected latency {workload_key}/{best_key}",
            strictly_positive=True,
        )
    return ratios


def format_report(analysis: BenchmarkAnalysis) -> str:
    """Render the structured exploratory analysis as deterministic CLI text."""
    lines = [
        f"{analysis.study_id}: post-hoc exploratory analysis, not confirmatory",
        f"gpu={analysis.gpu} bank={analysis.bank} workloads={len(analysis.workloads)} rows={analysis.row_count}",
        "Ratios are torch_ms / best_candidate_ms; > 1 means the candidate is faster.",
        "",
        "Per workload",
    ]
    for workload in analysis.workloads:
        archive = (
            "n/a"
            if workload.archive_torch_over_best_triton is None
            else f"{workload.archive_torch_over_best_triton:.6f}"
        )
        lines.append(
            f"  {workload.workload_key} regime={workload.regime} "
            f"best={workload.best_config_key} torch/best={workload.torch_over_best_candidate:.6f} "
            f"frozen-bank1-selected-bank2-scored={archive}"
        )
    lines.extend(("", "Expansion gate (evaluated independently by regime)"))
    for regime in analysis.regimes:
        decision = "PROCEED" if regime.passes_gate else "STOP"
        lines.append(
            f"  {regime.regime}: {decision}; n={regime.workload_count} "
            f"geomean={regime.geometric_mean_speedup:.6f} "
            f"median={regime.median_speedup:.6f} min={regime.minimum_speedup:.6f} "
            f"max={regime.maximum_speedup:.6f} "
            f">=5%-faster={regime.workloads_at_least_five_percent_faster}/"
            f"{regime.workload_count} ({regime.percent_at_least_five_percent_faster:.2f}%) "
            f"all-correct={'yes' if regime.all_selected_correct else 'no'}"
        )
    lines.extend(
        (
            f"Global: {'PROCEED' if analysis.proceed else 'STOP'}",
            "This stage gate is exploratory engineering evidence, not a confirmatory scientific endpoint.",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", default=str(_DEFAULT_ARTIFACT))
    parser.add_argument("--archive", default=str(_ARCHIVE))
    arguments = parser.parse_args(argv)
    artifact_path = Path(arguments.artifact)
    if not artifact_path.is_file():
        raise SystemExit(f"benchmark artifact does not exist: {artifact_path}")
    archive_path = Path(arguments.archive)
    ratios = load_archive_ratios(archive_path) if archive_path.is_file() else None
    analysis = analyze_artifact(read_json(artifact_path), archive_ratios=ratios)
    print(format_report(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
