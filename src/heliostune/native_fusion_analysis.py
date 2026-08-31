"""Deterministic exploratory stage-gate analysis for the frozen native fusion suite."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .artifacts import strict_json_loads
from .errors import SchemaError
from .native_fusion_executor import NativeFusionExecutionResult
from .scope import Suite

_SCHEMA = "heliostune.native-fusion-stage-gate/1"
_THRESHOLD = 1.1
_NATIVE_BACKEND = "native_triton"
_BASELINE_BACKENDS = ("eager", "inductor")
_VALIDATION_PROBES = ("zeros", "cancellation", "overflow")

_LIMITATIONS = [
    "A single authenticated one-bank workload is exploratory and not confirmatory.",
    "Expansion authorizes only additional exploratory workloads.",
]
_NONCLAIMS = [
    "No performance claim is made.",
    "No fusion claim is made.",
    "The result is not publication eligible.",
]


def _strict_result(
    result: NativeFusionExecutionResult,
) -> tuple[NativeFusionExecutionResult, Suite]:
    if not isinstance(result, NativeFusionExecutionResult):
        raise SchemaError("native fusion analysis requires a NativeFusionExecutionResult")
    parsed = NativeFusionExecutionResult.from_dict(
        result.to_dict(),
        verified_suite_path=result.verified_suite_path,
        verified_suite_sha256=result.verified_suite_sha256,
        verified_suite_bytes=result.verified_suite_bytes,
    )
    try:
        suite = Suite.from_dict(
            strict_json_loads(
                parsed.verified_suite_bytes.decode("utf-8"),
                source=parsed.verified_suite_path,
            )
        )
    except UnicodeError as exc:
        raise SchemaError("verified native suite bytes must be UTF-8") from exc
    return parsed, suite


def _observation_by_arm(result: NativeFusionExecutionResult, stage: str) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for item in result.observations:
        if item.stage == stage:
            observations[item.arm_id] = item
    return observations


def _error(
    errors: list[dict[str, object]],
    *,
    stage: str,
    failure_kind: object = None,
    message: object = None,
) -> None:
    if failure_kind is not None or message is not None:
        errors.append({"stage": stage, "failure_kind": failure_kind, "message": message})


def _native_errors(
    stage: Mapping[str, object],
    compile_evidence: Mapping[str, object],
    resource: Mapping[str, object],
    validation: Mapping[str, object],
    profile: Mapping[str, object],
    correctness: Any,
    timing: Any,
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    _error(
        errors,
        stage="stage_gate",
        failure_kind=stage["failure_kind"],
        message=stage["error"],
    )
    _error(errors, stage="compile", message=compile_evidence["error"])
    _error(errors, stage="resource", message=resource["error"])
    _error(errors, stage="validation", message=validation["error"])
    _error(errors, stage="profile", message=profile["error"])
    nested_correctness = correctness.correctness
    nested_timing = timing.timing
    _error(
        errors,
        stage="correctness",
        failure_kind=nested_correctness.failure_kind,
        message=nested_correctness.message,
    )
    _error(
        errors,
        stage="timing",
        failure_kind=nested_timing.failure_kind,
        message=nested_timing.message,
    )
    return errors


def _baseline_errors(
    stage: Mapping[str, object],
    compile_evidence: Mapping[str, object] | None,
    correctness: Any,
    timing: Any,
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    _error(
        errors,
        stage="stage_gate",
        failure_kind=stage["failure_kind"],
        message=stage["error"],
    )
    if compile_evidence is not None:
        _error(errors, stage="compile", message=compile_evidence["error"])
    nested_correctness = correctness.correctness
    nested_timing = timing.timing
    _error(
        errors,
        stage="correctness",
        failure_kind=nested_correctness.failure_kind,
        message=nested_correctness.message,
    )
    _error(
        errors,
        stage="timing",
        failure_kind=nested_timing.failure_kind,
        message=nested_timing.message,
    )
    return errors


def _passing_correctness(observation: Any) -> bool:
    nested = observation.correctness
    return (
        observation.status == "passed"
        and nested.status == "passed"
        and nested.failure_kind is None
        and nested.message is None
        and nested.output is not None
        and nested.max_abs_error is not None
        and nested.input_storage_unchanged is True
        and nested.output_disjoint is True
        and nested.finite is True
        and nested.close is True
    )


def _passing_timing(observation: Any) -> tuple[bool, float | None]:
    nested = observation.timing
    median = nested.median_ms
    passed = (
        observation.status == "passed"
        and nested.status == "passed"
        and nested.failure_kind is None
        and nested.message is None
        and nested.warmups == 10
        and nested.repetitions == 50
        and len(nested.samples_ms) == 50
        and type(median) is float
        and math.isfinite(median)
        and median > 0.0
        and all(
            type(sample) is float and math.isfinite(sample) and sample > 0.0
            for sample in nested.samples_ms
        )
    )
    return passed, median if passed else None


def _passing_compile(evidence: Mapping[str, object]) -> bool:
    return (
        evidence["status"] == "compiled"
        and evidence["error"] is None
        and evidence["compile_ns"] is not None
        and evidence["backend_invoked"] is True
        and evidence["callable_distinct"] is True
        and evidence["eager_fallback"] is False
        and evidence["kernel_name"] is not None
        and evidence["kernel_hash"] is not None
        and evidence["config"] is not None
    )


def _passing_resource(evidence: Mapping[str, object]) -> bool:
    metadata = evidence["metadata"]
    return (
        evidence["status"] == "compiled"
        and evidence["error"] is None
        and evidence["resource_gate_passed"] is True
        and evidence["n_spills"] == 0
        and evidence["kernel_name"] is not None
        and evidence["kernel_hash"] is not None
        and evidence["target"] is not None
        and type(metadata) is dict
        and evidence["n_regs"] is not None
        and evidence["n_max_threads"] is not None
        and bool(evidence["asm_stages"])
    )


def _passing_probe(probe: Mapping[str, object]) -> bool:
    return (
        probe["passed"] is True
        and probe["deterministic"] is True
        and probe["inputs_unchanged"] is True
        and probe["output_disjoint"] is True
        and probe["value_class_match"] is True
        and probe["sign_match"] is True
        and probe["finite_close"] is True
    )


def _passing_validation(evidence: Mapping[str, object]) -> bool:
    probes = cast(Sequence[Mapping[str, object]], evidence["probes"])
    return (
        evidence["status"] == "validated"
        and evidence["error"] is None
        and evidence["validation_gate_passed"] is True
        and tuple(probe["id"] for probe in probes) == _VALIDATION_PROBES
        and all(_passing_probe(probe) for probe in probes)
    )


def _passing_profile(evidence: Mapping[str, object]) -> bool:
    expected_name = evidence["expected_kernel_name"]
    return (
        evidence["status"] == "profiled"
        and evidence["error"] is None
        and evidence["one_kernel_gate_passed"] is True
        and evidence["warmed"] is True
        and expected_name is not None
        and evidence["expected_kernel_hash"] is not None
        and evidence["invocation_count"] == 1
        and evidence["cuda_event_count"] == 1
        and evidence["cuda_event_names_sample"] == [expected_name]
        and evidence["exact_name_match_count"] == 1
        and evidence["output_revalidated"] is True
        and evidence["inputs_revalidated"] is True
    )


def analyze_native_fusion_result(result: NativeFusionExecutionResult) -> dict[str, object]:
    """Strictly validate and deterministically analyze one native fusion execution result."""
    parsed, suite = _strict_result(result)
    case_id = suite.cases[0].id
    correctness_by_arm = _observation_by_arm(parsed, "correctness")
    timing_by_arm = _observation_by_arm(parsed, "timing")

    candidates: list[dict[str, object]] = []
    baselines: list[dict[str, object]] = []
    eligible_ids: list[str] = []
    eligible_medians: dict[str, float] = {}
    baseline_medians: dict[str, float] = {}
    winner_configs: dict[str, dict[str, object]] = {}

    for arm in suite.arms:
        cell_id = f"{arm.id}-correctness"
        stage = parsed.stage_outcomes[cell_id]
        correctness = correctness_by_arm[arm.id]
        timing = timing_by_arm[arm.id]
        correctness_passed = _passing_correctness(correctness)
        timing_passed, median = _passing_timing(timing)
        backend = cast(str, stage["backend_kind"])

        if backend == _NATIVE_BACKEND:
            compile_evidence = parsed.compile_evidence[cell_id]
            resource = parsed.resource_evidence[cell_id]
            validation = parsed.validation_evidence[cell_id]
            profile = parsed.profile_evidence[cell_id]
            compile_passed = _passing_compile(compile_evidence)
            resource_passed = _passing_resource(resource)
            validation_passed = _passing_validation(validation)
            probes = cast(Sequence[Mapping[str, object]], validation["probes"])
            probe_passed = {cast(str, probe["id"]): _passing_probe(probe) for probe in probes}
            profile_passed = _passing_profile(profile)
            eligible = all(
                (
                    compile_passed,
                    resource_passed,
                    correctness_passed,
                    validation_passed,
                    profile_passed,
                    timing_passed,
                )
            )
            config = dict(cast(Mapping[str, object], compile_evidence["config"]))
            candidates.append(
                {
                    "arm_id": arm.id,
                    "entrypoint": arm.entrypoint,
                    "config": config,
                    "stages": {
                        "compile": compile_passed,
                        "resource": resource_passed,
                        "correctness": correctness_passed,
                        "zeros": probe_passed["zeros"],
                        "cancellation": probe_passed["cancellation"],
                        "overflow": probe_passed["overflow"],
                        "profile": profile_passed,
                        "timing": timing_passed,
                    },
                    "median_ms": median,
                    "resource": dict(resource),
                    "profile": dict(profile),
                    "errors": _native_errors(
                        stage,
                        compile_evidence,
                        resource,
                        validation,
                        profile,
                        correctness,
                        timing,
                    ),
                }
            )
            if eligible:
                assert median is not None
                eligible_ids.append(arm.id)
                eligible_medians[arm.id] = median
                winner_configs[arm.id] = config
            continue

        if backend not in _BASELINE_BACKENDS:
            raise SchemaError(f"unsupported native analysis backend {backend!r}")
        baseline_compile = parsed.compile_evidence[cell_id] if backend == "inductor" else None
        baseline_passed = correctness_passed and timing_passed
        if baseline_compile is not None:
            baseline_passed = baseline_passed and (
                baseline_compile["status"] == "compiled"
                and baseline_compile["error"] is None
                and baseline_compile["backend_invoked"] is True
                and baseline_compile["callable_distinct"] is True
                and baseline_compile["eager_fallback"] is False
            )
        baselines.append(
            {
                "arm_id": arm.id,
                "entrypoint": arm.entrypoint,
                "stages": {
                    "correctness": correctness_passed,
                    "timing": timing_passed,
                },
                "median_ms": median,
                "errors": _baseline_errors(stage, baseline_compile, correctness, timing),
            }
        )
        if baseline_passed:
            assert median is not None
            baseline_medians[arm.id] = median

    winner_id = (
        min(eligible_medians, key=lambda arm_id: (eligible_medians[arm_id], arm_id))
        if eligible_medians
        else None
    )
    best_baseline_id = (
        min(baseline_medians, key=lambda arm_id: (baseline_medians[arm_id], arm_id))
        if len(baseline_medians) == 2
        else None
    )
    winner_median = None if winner_id is None else eligible_medians[winner_id]
    best_baseline_median = None if best_baseline_id is None else baseline_medians[best_baseline_id]
    speedup: float | None = None

    if len(baseline_medians) != 2:
        decision = "stop_incomplete_baseline"
    elif winner_id is None:
        decision = "stop_no_eligible_candidate"
    else:
        assert winner_median is not None and best_baseline_median is not None
        speedup = best_baseline_median / winner_median
        if not math.isfinite(speedup):
            raise SchemaError("native fusion speedup is not finite")
        decision = "expand_exploratory" if speedup >= _THRESHOLD else "stop_below_threshold"

    return {
        "schema": _SCHEMA,
        "suite_id": suite.suite_id,
        "suite_sha256": parsed.verified_suite_sha256,
        "case_id": case_id,
        "candidates": candidates,
        "baselines": baselines,
        "eligible_candidate_ids": eligible_ids,
        "winner_id": winner_id,
        "winner_config": None if winner_id is None else winner_configs[winner_id],
        "winner_median_ms": winner_median,
        "best_baseline_id": best_baseline_id,
        "best_baseline_median_ms": best_baseline_median,
        "speedup": speedup,
        "threshold": _THRESHOLD,
        "decision": decision,
        "confirmatory": False,
        "fusion_claim": False,
        "publication_eligible": False,
        "claims": [],
        "limitations": list(_LIMITATIONS),
        "nonclaims": list(_NONCLAIMS),
    }


__all__ = ["analyze_native_fusion_result"]
