from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import heliostune.native_fusion_analysis as analysis
from heliostune.errors import SchemaError
from heliostune.local_executor import CapabilityProbe
from heliostune.native_fusion_analysis import analyze_native_fusion_result
from heliostune.native_fusion_executor import NativeFusionExecutionResult, run_native_fusion_suite
from heliostune.scope import verify_suite

_ROOT = Path(__file__).resolve().parents[1]
_SUITE_PATH = _ROOT / "benchmarks/suites/residual-rmsnorm-triton-v1.json"
_SUITE = verify_suite(_SUITE_PATH).suite
_NATIVE = (
    "rmsnorm-triton-w4",
    "rmsnorm-triton-w8",
    "rmsnorm-triton-w16",
    "rmsnorm-triton-w32",
)
_EAGER = "rmsnorm-eager-reference"
_INDUCTOR = "rmsnorm-inductor-comparator"


def _correctness(arm_id: str) -> SimpleNamespace:
    nested = SimpleNamespace(
        status="passed",
        failure_kind=None,
        message=None,
        output=SimpleNamespace(),
        max_abs_error=0.0,
        input_storage_unchanged=True,
        output_disjoint=True,
        finite=True,
        close=True,
    )
    return SimpleNamespace(
        arm_id=arm_id, stage="correctness", status="passed", correctness=nested, timing=None
    )


def _timing(arm_id: str, median_ms: float) -> SimpleNamespace:
    nested = SimpleNamespace(
        status="passed",
        failure_kind=None,
        message=None,
        warmups=10,
        repetitions=50,
        samples_ms=(median_ms,) * 50,
        median_ms=median_ms,
    )
    return SimpleNamespace(
        arm_id=arm_id, stage="timing", status="passed", correctness=None, timing=nested
    )


def _compile(arm_id: str, num_warps: int, *, native: bool) -> dict[str, object]:
    return {
        "status": "compiled",
        "error": None,
        "compile_ns": 1,
        "backend_invoked": True,
        "callable_distinct": True,
        "eager_fallback": False,
        "kernel_name": f"kernel_w{num_warps}" if native else None,
        "kernel_hash": "a" * 64 if native else None,
        "config": {
            "block_size": 4096,
            "num_warps": num_warps,
            "num_stages": 1,
        }
        if native
        else None,
    }


def _resource(arm_id: str, num_warps: int) -> dict[str, object]:
    return {
        "status": "compiled",
        "error": None,
        "kernel_name": f"kernel_w{num_warps}",
        "kernel_hash": "a" * 64,
        "target": {"backend": "cuda", "arch": "90", "warp_size": 32},
        "metadata": {
            "shared": 0,
            "num_warps": num_warps,
            "num_ctas": 1,
            "num_stages": 1,
        },
        "n_regs": 32,
        "n_spills": 0,
        "n_max_threads": 128,
        "asm_stages": [{"stage": "cubin", "bytes": 1, "sha256": "b" * 64}],
        "resource_gate_passed": True,
    }


def _validation() -> dict[str, object]:
    probes = []
    for probe_id in ("zeros", "cancellation", "overflow"):
        probes.append(
            {
                "id": probe_id,
                "passed": True,
                "deterministic": True,
                "inputs_unchanged": True,
                "output_disjoint": True,
                "value_class_match": True,
                "sign_match": True,
                "finite_close": True,
                "max_abs_error": 0.0,
            }
        )
    return {
        "status": "validated",
        "error": None,
        "probes": probes,
        "validation_gate_passed": True,
    }


def _profile(num_warps: int) -> dict[str, object]:
    name = f"kernel_w{num_warps}"
    return {
        "status": "profiled",
        "error": None,
        "warmed": True,
        "expected_kernel_name": name,
        "expected_kernel_hash": "a" * 64,
        "config": {"block_size": 4096, "num_warps": num_warps, "num_stages": 1},
        "invocation_count": 1,
        "cuda_event_count": 1,
        "cuda_event_names_sample": [name],
        "exact_name_match_count": 1,
        "output_revalidated": True,
        "inputs_revalidated": True,
        "one_kernel_gate_passed": True,
    }


def _passing_result(
    *,
    candidate_medians: tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0),
    eager_median: float = 1.2,
    inductor_median: float = 1.3,
) -> SimpleNamespace:
    medians = dict(zip(_NATIVE, candidate_medians, strict=True))
    medians.update({_EAGER: eager_median, _INDUCTOR: inductor_median})
    observations: list[SimpleNamespace] = []
    stages: dict[str, dict[str, object]] = {}
    compile_evidence: dict[str, dict[str, object]] = {}
    resources: dict[str, dict[str, object]] = {}
    validations: dict[str, dict[str, object]] = {}
    profiles: dict[str, dict[str, object]] = {}

    for arm in _SUITE.arms:
        observations.extend((_correctness(arm.id), _timing(arm.id, medians[arm.id])))
        backend = (
            "native_triton" if arm.id in _NATIVE else "eager" if arm.id == _EAGER else "inductor"
        )
        stages[f"{arm.id}-correctness"] = {
            "backend_kind": backend,
            "status": "completed",
            "failure_kind": None,
            "error": None,
        }
        if arm.id in _NATIVE:
            warps = int(arm.id.removeprefix("rmsnorm-triton-w"))
            cell_id = f"{arm.id}-correctness"
            compile_evidence[cell_id] = _compile(arm.id, warps, native=True)
            resources[cell_id] = _resource(arm.id, warps)
            validations[cell_id] = _validation()
            profiles[cell_id] = _profile(warps)
        elif arm.id == _INDUCTOR:
            compile_evidence[f"{arm.id}-correctness"] = _compile(arm.id, 0, native=False)

    return SimpleNamespace(
        verified_suite_sha256=verify_suite(_SUITE_PATH).sha256,
        observations=tuple(observations),
        stage_outcomes=stages,
        compile_evidence=compile_evidence,
        resource_evidence=resources,
        validation_evidence=validations,
        profile_evidence=profiles,
    )


def _analyze_fake(monkeypatch: pytest.MonkeyPatch, result: SimpleNamespace) -> dict[str, object]:
    monkeypatch.setattr(analysis, "_strict_result", lambda _value: (result, _SUITE))
    return analyze_native_fusion_result(cast(NativeFusionExecutionResult, result))


def _candidate(output: dict[str, object], arm_id: str) -> dict[str, object]:
    candidates = cast(list[dict[str, object]], output["candidates"])
    return next(item for item in candidates if item["arm_id"] == arm_id)


def _fail_gate(result: SimpleNamespace, gate: str) -> None:
    arm_id = _NATIVE[0]
    cell_id = f"{arm_id}-correctness"
    observations = {f"{item.arm_id}:{item.stage}": item for item in result.observations}
    if gate == "compile":
        result.compile_evidence[cell_id]["backend_invoked"] = False
    elif gate == "resource":
        result.resource_evidence[cell_id]["n_spills"] = 1
    elif gate == "correctness":
        observations[f"{arm_id}:correctness"].correctness.close = False
    elif gate in {"zeros", "cancellation", "overflow"}:
        probes = result.validation_evidence[cell_id]["probes"]
        cast(list[dict[str, object]], probes)[("zeros", "cancellation", "overflow").index(gate)][
            "passed"
        ] = False
    elif gate == "profile":
        result.profile_evidence[cell_id]["cuda_event_count"] = 2
    elif gate == "timing":
        observations[f"{arm_id}:timing"].timing.repetitions = 49
    else:
        raise AssertionError(gate)


@pytest.mark.parametrize(
    "gate",
    [
        "compile",
        "resource",
        "correctness",
        "zeros",
        "cancellation",
        "overflow",
        "profile",
        "timing",
    ],
)
def test_each_native_gate_is_derived_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    result = _passing_result()
    _fail_gate(result, gate)
    output = _analyze_fake(monkeypatch, result)
    assert _NATIVE[0] not in output["eligible_candidate_ids"]
    stages = cast(dict[str, bool], _candidate(output, _NATIVE[0])["stages"])
    assert stages[gate] is False


def test_output_is_stable_tie_broken_and_claimless(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _passing_result(
        candidate_medians=(1.0, 1.0, 2.0, 3.0),
        eager_median=1.2,
        inductor_median=1.2,
    )
    output = _analyze_fake(monkeypatch, result)
    repeated = _analyze_fake(monkeypatch, result)

    assert output == repeated
    assert "evaluated_at" not in output
    assert output["schema"] == "heliostune.native-fusion-stage-gate/1"
    assert [item["arm_id"] for item in cast(list[dict[str, object]], output["candidates"])] == list(
        _NATIVE
    )
    assert [item["arm_id"] for item in cast(list[dict[str, object]], output["baselines"])] == [
        _EAGER,
        _INDUCTOR,
    ]
    assert output["winner_id"] == _NATIVE[0]
    assert output["best_baseline_id"] == _EAGER
    assert output["decision"] == "expand_exploratory"
    assert output["confirmatory"] is False
    assert output["fusion_claim"] is False
    assert output["publication_eligible"] is False
    assert output["claims"] == []
    assert _candidate(output, _NATIVE[0])["errors"] == []


@pytest.mark.parametrize(
    ("baseline_median", "decision"),
    [(1.09, "stop_below_threshold"), (1.1, "expand_exploratory"), (1.11, "expand_exploratory")],
)
def test_expansion_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch, baseline_median: float, decision: str
) -> None:
    output = _analyze_fake(
        monkeypatch,
        _passing_result(eager_median=baseline_median, inductor_median=baseline_median),
    )
    assert output["speedup"] == baseline_median
    assert output["decision"] == decision


def test_missing_baseline_stops_before_candidate_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _passing_result()
    eager_timing = next(
        item for item in result.observations if item.arm_id == _EAGER and item.stage == "timing"
    )
    eager_timing.timing.repetitions = 0
    output = _analyze_fake(monkeypatch, result)

    assert output["decision"] == "stop_incomplete_baseline"
    assert output["best_baseline_id"] is None
    assert output["best_baseline_median_ms"] is None
    assert output["speedup"] is None


def test_no_eligible_candidate_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _passing_result()
    for arm_id in _NATIVE:
        result.compile_evidence[f"{arm_id}-correctness"]["backend_invoked"] = False
    output = _analyze_fake(monkeypatch, result)
    assert output["decision"] == "stop_no_eligible_candidate"
    assert output["winner_id"] is None
    assert output["winner_config"] is None
    assert output["winner_median_ms"] is None
    assert output["speedup"] is None


def _unavailable() -> CapabilityProbe:
    return CapabilityProbe(
        False,
        ("torch_missing",),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        "strict CPU fake",
    )


def test_result_is_strictly_reparsed_and_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import heliostune.native_fusion_executor as executor

    monkeypatch.setattr(executor, "_probe_capability", lambda: (_unavailable(), None, None))
    result = run_native_fusion_suite(_SUITE_PATH)
    closed = analyze_native_fusion_result(result)
    assert closed["decision"] == "stop_incomplete_baseline"
    assert closed["eligible_candidate_ids"] == []
    assert closed["claims"] == []
    tampered_summary = dict(result.summary)
    tampered_summary["fusion_claim"] = True
    tampered = replace(result, summary=tampered_summary)

    with pytest.raises(SchemaError, match="summary"):
        analyze_native_fusion_result(tampered)
    with pytest.raises(SchemaError, match="NativeFusionExecutionResult"):
        analyze_native_fusion_result(cast(Any, SimpleNamespace()))


def test_invalid_passing_float_is_not_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _passing_result()
    timing = next(
        item for item in result.observations if item.arm_id == _NATIVE[0] and item.stage == "timing"
    )
    timing.timing.median_ms = float("nan")
    output = _analyze_fake(monkeypatch, result)
    assert _NATIVE[0] not in output["eligible_candidate_ids"]
    assert _candidate(output, _NATIVE[0])["median_ms"] is None
