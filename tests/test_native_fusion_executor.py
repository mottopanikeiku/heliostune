from __future__ import annotations

import hashlib
import importlib
import json
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from heliostune.errors import SchemaError
from heliostune.local_executor import CapabilityProbe, TensorMaterialization
from heliostune.native_fusion_executor import (
    _CONFIG,
    _ENTRYPOINT,
    _RUNTIME_ARMS,
    NATIVE_RMSNORM_SUITE_SHA256,
    NativeFusionExecutionResult,
    _blocked_validation,
    _capture_executor_sources,
    _compile_comparator,
    _correctness_key,
    _names_digest,
    _parse_profile,
    _parse_resource,
    _parse_validation,
    _profile_once,
    _safe_error,
    run_native_fusion_suite,
)
from heliostune.scope import verify_suite

_ROOT = Path(__file__).resolve().parents[1]
_SUITE = _ROOT / "benchmarks/suites/residual-rmsnorm-triton-v1.json"


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


def _available() -> CapabilityProbe:
    return CapabilityProbe(
        True,
        (),
        "2.8.0",
        "12.8",
        None,
        0,
        "NVIDIA H100 80GB HBM3",
        (9, 0),
        True,
        True,
        True,
        None,
    )


@pytest.fixture
def aborted(monkeypatch: pytest.MonkeyPatch) -> NativeFusionExecutionResult:
    import heliostune.native_fusion_executor as executor

    monkeypatch.setattr(executor, "_probe_capability", lambda: (_unavailable(), None, None))
    return run_native_fusion_suite(_SUITE)


def _parse(result: NativeFusionExecutionResult, value: object) -> NativeFusionExecutionResult:
    return NativeFusionExecutionResult.from_dict(
        value,
        verified_suite_path=result.suite_path,
        verified_suite_sha256=result.suite_sha256,
        verified_suite_bytes=result.suite_bytes,
    )


def test_frozen_digest_and_cpu_import_safety() -> None:
    payload = _SUITE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == NATIVE_RMSNORM_SUITE_SHA256

    # Reloading the executor itself must not ask importlib for either optional GPU package.
    module = importlib.import_module("heliostune.native_fusion_executor")
    assert "torch" not in module.__dict__
    assert "triton" not in module.__dict__
    inventory = _capture_executor_sources()
    assert set(inventory) == {
        "schema",
        "package_source_sha256",
        "package_source_count",
        "sources",
    }
    package_source_count = inventory["package_source_count"]
    assert type(package_source_count) is int
    assert package_source_count >= len(cast(list[object], inventory["sources"]))


def test_safe_error_is_utf8_bounded_typed_and_binds_truncated_bytes() -> None:
    message = "compiler exploded: " + "界" * 500
    original = f"RuntimeError: {message}".encode()
    error = _safe_error(RuntimeError(message))

    assert error.startswith("RuntimeError: compiler exploded:")
    assert len(error.encode("utf-8")) <= 384
    assert error.endswith(f"...[truncated sha256={hashlib.sha256(original).hexdigest()}]")
    assert _safe_error(ValueError("short")) == "ValueError: short"


def test_recording_inductor_backend_requires_successful_backend_return() -> None:
    state: dict[str, object] = {}

    def backend(_graph: object, _inputs: object) -> object:
        raise RuntimeError("inductor backend failed")

    def compile_callable(
        _kernel: object,
        *,
        backend: Any,
        fullgraph: bool,
        dynamic: bool,
        mode: str,
    ) -> Any:
        assert (fullgraph, dynamic, mode) == (True, False, "default")

        def compiled() -> object:
            return backend(object(), ())

        return compiled

    registry = SimpleNamespace(lookup_backend=lambda name: backend if name == "inductor" else None)
    torch = SimpleNamespace(
        _dynamo=SimpleNamespace(
            config=SimpleNamespace(disable=False, suppress_errors=False),
            backends=SimpleNamespace(registry=registry),
        ),
        compile=compile_callable,
    )
    compiled = _compile_comparator(torch, lambda: None, state)
    with pytest.raises(RuntimeError, match="inductor backend failed"):
        compiled()
    assert state["invoked"] is True
    assert state["completed"] is False
    assert state["callable_distinct"] is True
    assert "inductor backend failed" in cast(str, state["error"])


def test_cpu_capability_abort_is_strict_and_makes_no_cuda_claims(
    aborted: NativeFusionExecutionResult,
) -> None:
    assert aborted.schema == "heliostune.local_executor/2"
    assert aborted.outcome == "aborted"
    assert aborted.materialization == ()
    assert len(aborted.observations) == 12
    assert tuple(aborted.stage_outcomes) == tuple(f"{arm}-correctness" for arm in _RUNTIME_ARMS)
    assert all(item["status"] == "blocked" for item in aborted.stage_outcomes.values())
    assert all(item["status"] == "blocked" for item in aborted.compile_evidence.values())
    assert all(item["status"] == "blocked" for item in aborted.resource_evidence.values())
    assert all(item["status"] == "blocked" for item in aborted.profile_evidence.values())
    assert all(item["status"] == "blocked" for item in aborted.validation_evidence.values())
    assert aborted.environment["backend_invoked"] is False
    assert aborted.environment["fusion_claim"] is False
    assert aborted.environment["device_name"] is None
    assert aborted.summary["fusion_claim"] is False
    assert aborted.summary["counts"] == {
        "stage_completed": 0,
        "stage_failed": 0,
        "stage_blocked": 6,
        "compile_compiled": 0,
        "compile_failed": 0,
        "compile_blocked": 5,
        "resource_passed": 0,
        "resource_failed": 0,
        "resource_blocked": 4,
        "profile_passed": 0,
        "profile_failed": 0,
        "profile_blocked": 4,
        "validation_passed": 0,
        "validation_failed": 0,
        "validation_blocked": 4,
    }


def test_unavailable_capability_rejects_tampered_observations(
    aborted: NativeFusionExecutionResult,
) -> None:
    failed = deepcopy(aborted.to_dict())
    observations = cast(list[dict[str, object]], failed["observations"])
    observations[0]["status"] = "failed"
    cast(dict[str, object], observations[0]["correctness"])["status"] = "failed"
    summary = cast(dict[str, object], failed["summary"])
    summary["failed"] = 1
    summary["blocked"] = 11
    with pytest.raises(SchemaError, match="capability-rejected observation"):
        _parse(aborted, failed)

    for field, replacement in (
        (
            "output",
            {
                "shape": [128, 4096],
                "device": "cuda:0",
                "dtype": "torch.bfloat16",
                "layout": "torch.strided",
                "contiguous": True,
            },
        ),
        ("input_storage_unchanged", True),
        ("max_abs_error", 0.0),
        ("message", "different capability failure"),
    ):
        value = deepcopy(aborted.to_dict())
        observations = cast(list[dict[str, object]], value["observations"])
        cast(dict[str, object], observations[0]["correctness"])[field] = replacement
        with pytest.raises(SchemaError, match="blocked correctness"):
            _parse(aborted, value)

    wrong_kind = deepcopy(aborted.to_dict())
    observations = cast(list[dict[str, object]], wrong_kind["observations"])
    cast(dict[str, object], observations[1]["timing"])["failure_kind"] = "executor"
    attempts = cast(list[dict[str, object]], wrong_kind["attempts"])
    attempts[3]["reason"] = "executor"
    with pytest.raises(SchemaError, match="blocked timing"):
        _parse(aborted, wrong_kind)

    wrong_message = deepcopy(aborted.to_dict())
    observations = cast(list[dict[str, object]], wrong_message["observations"])
    cast(dict[str, object], observations[1]["timing"])["message"] = "different capability failure"
    with pytest.raises(SchemaError, match="blocked timing"):
        _parse(aborted, wrong_message)

    wrong_stage_message = deepcopy(aborted.to_dict())
    stages = cast(dict[str, dict[str, object]], wrong_stage_message["stage_outcomes"])
    stages["rmsnorm-triton-w4-correctness"]["error"] = "different capability failure"
    with pytest.raises(SchemaError, match="stage failure linkage"):
        _parse(aborted, wrong_stage_message)


def test_strict_round_trip_uses_exact_v1_custody_kwargs(
    aborted: NativeFusionExecutionResult,
) -> None:
    parsed = _parse(aborted, aborted.to_dict())
    assert parsed.to_dict() == aborted.to_dict()
    assert parsed.suite_path == aborted.suite_path
    assert parsed.suite_sha256 == aborted.suite_sha256
    assert parsed.suite_bytes == aborted.suite_bytes
    assert aborted.to_dict(include_suite_bytes=True)["verified_suite_bytes"] == aborted.suite_bytes


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda value: value.__setitem__("extra", None), "unknown fields"),
        (lambda value: value.__setitem__("schema", "heliostune.local_executor/1"), "schema"),
        (lambda value: value["environment"].__setitem__("fusion_claim", True), "environment"),
        (lambda value: value["summary"].__setitem__("fusion_claim", True), "summary"),
        (
            lambda value: value["stage_outcomes"].pop("rmsnorm-triton-w4-correctness"),
            "exact frozen cell IDs",
        ),
        (
            lambda value: value["compile_evidence"].pop("rmsnorm-inductor-comparator-correctness"),
            "exact frozen cell IDs",
        ),
        (
            lambda value: value["resource_evidence"].pop("rmsnorm-triton-w32-correctness"),
            "exact frozen cell IDs",
        ),
        (
            lambda value: value["profile_evidence"].pop("rmsnorm-triton-w8-correctness"),
            "exact frozen cell IDs",
        ),
        (
            lambda value: value["validation_evidence"].pop("rmsnorm-triton-w16-correctness"),
            "exact frozen cell IDs",
        ),
        (
            lambda value: value["executor_sources"]["sources"].pop(),
            "exact source inventory",
        ),
        (
            lambda value: value["executor_sources"].__setitem__("extra", None),
            "unknown fields",
        ),
        (
            lambda value: value["executor_sources"].pop("package_source_sha256"),
            "missing fields",
        ),
        (
            lambda value: value["executor_sources"].__setitem__("package_source_sha256", "A" * 64),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["executor_sources"].__setitem__("package_source_count", True),
            "must be an integer",
        ),
    ],
)
def test_parser_rejects_schema_and_mapping_shape_mutations(
    aborted: NativeFusionExecutionResult,
    mutation: Any,
    match: str,
) -> None:
    value = deepcopy(aborted.to_dict())
    mutation(value)
    with pytest.raises(SchemaError, match=match):
        _parse(aborted, value)


def test_parser_rejects_cross_linkage_and_gate_mutations(
    aborted: NativeFusionExecutionResult,
) -> None:
    mutations = []

    def wrong_arm(value: dict[str, Any]) -> None:
        value["stage_outcomes"]["rmsnorm-triton-w4-correctness"]["arm_id"] = "rmsnorm-triton-w8"

    mutations.append(wrong_arm)

    def open_timing(value: dict[str, Any]) -> None:
        value["stage_outcomes"]["rmsnorm-triton-w4-correctness"]["timing_allowed"] = True

    mutations.append(open_timing)

    def resource_link(value: dict[str, Any]) -> None:
        value["resource_evidence"]["rmsnorm-triton-w4-correctness"]["entrypoint"] = _ENTRYPOINT[
            "rmsnorm-triton-w8"
        ]

    mutations.append(resource_link)

    def profile_hash(value: dict[str, Any]) -> None:
        value["profile_evidence"]["rmsnorm-triton-w4-correctness"]["expected_kernel_hash"] = (
            "a" * 64
        )

    mutations.append(profile_hash)

    for mutation in mutations:
        value = deepcopy(aborted.to_dict())
        mutation(value)
        with pytest.raises(SchemaError):
            _parse(aborted, value)


def test_result_rejects_wrong_custody_digest_and_bytes(
    aborted: NativeFusionExecutionResult,
) -> None:
    with pytest.raises(SchemaError, match="bytes"):
        NativeFusionExecutionResult.from_dict(
            aborted.to_dict(),
            verified_suite_path=aborted.suite_path,
            verified_suite_sha256="0" * 64,
            verified_suite_bytes=aborted.suite_bytes,
        )
    with pytest.raises(SchemaError, match="bytes"):
        NativeFusionExecutionResult.from_dict(
            aborted.to_dict(),
            verified_suite_path=aborted.suite_path,
            verified_suite_sha256=aborted.suite_sha256,
            verified_suite_bytes=aborted.suite_bytes + b" ",
        )


def test_frozen_validation_precedes_capability_or_torch_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = json.loads(_SUITE.read_text())
    value["suite_id"] = "not-the-frozen-suite"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value))
    called = False

    def forbidden() -> tuple[object, None, None]:
        nonlocal called
        called = True
        raise AssertionError("capability probe must not run")

    import heliostune.native_fusion_executor as executor

    monkeypatch.setattr(executor, "_probe_capability", forbidden)
    with pytest.raises(SchemaError, match="digest"):
        run_native_fusion_suite(changed)
    assert called is False


def test_closed_entrypoint_dispatch_is_role_independent() -> None:
    suite = verify_suite(_SUITE).suite
    by_id = {arm.id: arm for arm in suite.arms}
    assert set(_ENTRYPOINT) == set(by_id)
    assert all(_ENTRYPOINT[arm_id] == by_id[arm_id].entrypoint for arm_id in by_id)
    assert _ENTRYPOINT["rmsnorm-triton-w4"].endswith("residual_rmsnorm_w4")
    assert _ENTRYPOINT["rmsnorm-inductor-comparator"].startswith("reference_template.")


def test_serialization_order_and_correctness_keys_match_suite_and_v1_formula() -> None:
    from heliostune.local_executor import _correctness_gate_key

    suite = verify_suite(_SUITE).suite
    assert tuple(arm.id for arm in suite.arms) == _RUNTIME_ARMS
    case = suite.cases[0]
    for cell in suite.expected_cells:
        if cell.stage == "correctness":
            assert _correctness_key(cell.id) == _correctness_gate_key(
                NATIVE_RMSNORM_SUITE_SHA256, case, cell
            )


def _resource() -> dict[str, object]:
    arm = "rmsnorm-triton-w4"
    return {
        "case_id": "rmsnorm-case-001",
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": "compiled",
        "error": None,
        "kernel_name": "residual_rmsnorm_kernel",
        "kernel_hash": "a" * 64,
        "target": {"backend": "cuda", "arch": "90", "warp_size": 32},
        "metadata": {"shared": 0, "num_warps": 4, "num_ctas": 1, "num_stages": 1},
        "n_regs": 32,
        "n_spills": 0,
        "n_max_threads": 128,
        "asm_stages": [
            {"stage": "cubin", "bytes": 7, "sha256": "b" * 64},
            {"stage": "ptx", "bytes": 11, "sha256": "c" * 64},
        ],
        "resource_gate_passed": True,
    }


def test_resource_evidence_requires_zero_spills_and_sorted_exact_asm() -> None:
    cell = "rmsnorm-triton-w4-correctness"
    parsed = _parse_resource(_resource(), cell)
    assert parsed["n_spills"] == 0
    assert parsed["resource_gate_passed"] is True

    spill = deepcopy(_resource())
    spill["n_spills"] = 1
    with pytest.raises(SchemaError, match="zero-spill"):
        _parse_resource(spill, cell)

    multiple_ctas = deepcopy(_resource())
    metadata = cast(dict[str, object], multiple_ctas["metadata"])
    metadata["num_ctas"] = 2
    with pytest.raises(SchemaError, match="exact-config"):
        _parse_resource(multiple_ctas, cell)

    unsorted = deepcopy(_resource())
    cast(list[dict[str, object]], unsorted["asm_stages"]).reverse()
    with pytest.raises(SchemaError, match="sorted"):
        _parse_resource(unsorted, cell)

    extra = deepcopy(_resource())
    cast(list[dict[str, object]], extra["asm_stages"])[0]["format"] = "binary"
    with pytest.raises(SchemaError, match="unknown fields"):
        _parse_resource(extra, cell)
    for target in (
        {"backend": "hip", "arch": "90", "warp_size": 32},
        {"backend": "cuda", "arch": "89", "warp_size": 32},
        {"backend": "cuda", "arch": "90", "warp_size": 64},
    ):
        wrong_target = deepcopy(_resource())
        wrong_target["target"] = target
        with pytest.raises(SchemaError, match="H100"):
            _parse_resource(wrong_target, cell)


def _profile(names: list[str]) -> dict[str, object]:
    arm = "rmsnorm-triton-w4"
    expected = "residual_rmsnorm_kernel"
    return {
        "case_id": "rmsnorm-case-001",
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": "profiled",
        "error": None,
        "method": "torch.profiler.cuda_events",
        "warmed": True,
        "expected_kernel_name": expected,
        "expected_kernel_hash": "a" * 64,
        "config": dict(_CONFIG[arm]),
        "invocation_count": 1,
        "cuda_event_count": len(names),
        "cuda_event_names_sample": names,
        "cuda_event_names_sha256": _names_digest(names),
        "exact_name_match_count": names.count(expected),
        "output_revalidated": True,
        "inputs_revalidated": True,
        "one_kernel_gate_passed": True,
    }


def test_profile_evidence_requires_one_total_exact_cuda_kernel() -> None:
    cell = "rmsnorm-triton-w4-correctness"
    names = ["residual_rmsnorm_kernel"]
    parsed = _parse_profile(_profile(names), cell)
    assert parsed["cuda_event_names_sample"] == names
    assert parsed["cuda_event_count"] == 1
    assert parsed["exact_name_match_count"] == 1

    extra = _profile(["residual_rmsnorm_kernel", "cuda_runtime_helper"])
    with pytest.raises(SchemaError, match="one-kernel"):
        _parse_profile(extra, cell)

    duplicate = _profile(["residual_rmsnorm_kernel", "residual_rmsnorm_kernel"])
    with pytest.raises(SchemaError, match="one-kernel"):
        _parse_profile(duplicate, cell)

    wrong_digest = _profile(names)
    wrong_digest["cuda_event_names_sha256"] = "f" * 64
    with pytest.raises(SchemaError, match="digest"):
        _parse_profile(wrong_digest, cell)

    too_many = _profile([f"kernel-{index}" for index in range(33)])
    too_many["exact_name_match_count"] = 1
    with pytest.raises(SchemaError, match="exceeds 32"):
        _parse_profile(too_many, cell)


def test_profile_runtime_filters_non_cuda_events_with_strict_cpu_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    import heliostune.local_executor as legacy

    cuda_type = object()
    cpu_type = object()
    events = (
        SimpleNamespace(name="python_wrapper", device_type=cpu_type),
        SimpleNamespace(name="residual_rmsnorm_kernel", device_type=cuda_type),
    )

    class FakeProfile:
        def __enter__(self) -> FakeProfile:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def events(self) -> tuple[SimpleNamespace, ...]:
            return events

    fake_torch = SimpleNamespace(
        autograd=SimpleNamespace(DeviceType=SimpleNamespace(CUDA=cuda_type)),
        cuda=SimpleNamespace(synchronize=lambda _device: None),
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CUDA=object()),
            profile=lambda **_kwargs: FakeProfile(),
        ),
    )
    monkeypatch.setattr(legacy, "_tensor_hash", lambda _torch, _tensor: "d" * 64)
    monkeypatch.setattr(
        legacy,
        "_validate_correctness",
        lambda *_args, **_kwargs: SimpleNamespace(status="passed"),
    )
    invocations = 0

    def kernel(*_arguments: object) -> object:
        nonlocal invocations
        invocations += 1
        return object()

    evidence = _profile_once(
        fake_torch,
        "rmsnorm-triton-w4",
        kernel,
        (object(), object(), object()),
        object(),
        {"input": object()},
        0,
        "residual_rmsnorm_kernel",
        "a" * 64,
    )
    assert invocations == 2  # one warm invocation and exactly one profiled invocation
    assert evidence["status"] == "profiled"
    assert evidence["cuda_event_names_sample"] == ["residual_rmsnorm_kernel"]
    assert evidence["cuda_event_count"] == 1
    assert evidence["exact_name_match_count"] == 1
    assert evidence["cuda_event_names_sha256"] == _names_digest(["residual_rmsnorm_kernel"])


def _validation() -> dict[str, object]:
    arm = "rmsnorm-triton-w4"
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
                "max_abs_error": None if probe_id == "overflow" else 0.0,
            }
        )
    return {
        "case_id": "rmsnorm-case-001",
        "arm_id": arm,
        "entrypoint": _ENTRYPOINT[arm],
        "status": "validated",
        "error": None,
        "probes": probes,
        "validation_gate_passed": True,
    }


def test_validation_evidence_is_exact_and_overflow_error_is_nullable() -> None:
    cell = "rmsnorm-triton-w4-correctness"
    parsed = _parse_validation(_validation(), cell)
    probes = cast(list[dict[str, object]], parsed["probes"])
    assert [probe["id"] for probe in probes] == [
        "zeros",
        "cancellation",
        "overflow",
    ]
    assert probes[-1]["max_abs_error"] is None

    nondeterministic = deepcopy(_validation())
    cast(list[dict[str, object]], nondeterministic["probes"])[0]["deterministic"] = False
    with pytest.raises(SchemaError, match="passed flag"):
        _parse_validation(nondeterministic, cell)

    reordered = deepcopy(_validation())
    cast(list[dict[str, object]], reordered["probes"]).reverse()
    with pytest.raises(SchemaError, match="in order"):
        _parse_validation(reordered, cell)

    blocked = _blocked_validation("rmsnorm-triton-w4", "blocked")
    cast(list[dict[str, object]], blocked["probes"])[2]["value_class_match"] = True
    with pytest.raises(SchemaError, match="blocked validation"):
        _parse_validation(blocked, cell)


def test_gate_transition_requires_all_native_gates_but_not_baseline_gates(
    aborted: NativeFusionExecutionResult,
) -> None:
    value = deepcopy(aborted.to_dict())
    stage_outcomes = cast(dict[str, dict[str, object]], value["stage_outcomes"])
    native = stage_outcomes["rmsnorm-triton-w4-correctness"]
    native.update(
        status="completed",
        failure_kind=None,
        error=None,
        correctness_passed=True,
        resource_passed=True,
        profile_passed=True,
        validation_passed=True,
        timing_allowed=True,
    )
    # Cross linkage still rejects this fabricated result because its correctness,
    # resource and profile records remain blocked.
    with pytest.raises(SchemaError, match="summary|linkage|gate"):
        _parse(aborted, value)

    baseline = stage_outcomes["rmsnorm-eager-reference-correctness"]
    assert baseline["resource_passed"] is None
    assert baseline["profile_passed"] is None
    assert baseline["validation_passed"] is None


def test_parser_binds_observation_keys_timing_shape_and_attempt_reason(
    aborted: NativeFusionExecutionResult,
) -> None:
    wrong_key = deepcopy(aborted.to_dict())
    observations = cast(list[dict[str, object]], wrong_key["observations"])
    correctness = cast(dict[str, object], observations[0]["correctness"])
    correctness["correctness_key"] = "a" * 64
    with pytest.raises(SchemaError, match="correctness key"):
        _parse(aborted, wrong_key)

    timing_claim = deepcopy(aborted.to_dict())
    observations = cast(list[dict[str, object]], timing_claim["observations"])
    timing = cast(dict[str, object], observations[1]["timing"])
    timing["warmups"] = 1
    with pytest.raises(SchemaError, match="blocked timing"):
        _parse(aborted, timing_claim)

    wrong_reason = deepcopy(aborted.to_dict())
    attempts = cast(list[dict[str, object]], wrong_reason["attempts"])
    attempts[1]["reason"] = "different"
    with pytest.raises(SchemaError, match="reason"):
        _parse(aborted, wrong_reason)


def test_materialization_failure_returns_terminal_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import heliostune.local_executor as legacy
    import heliostune.native_fusion_executor as executor

    monkeypatch.setattr(
        executor, "_probe_capability", lambda: (_available(), SimpleNamespace(), "3.4.0")
    )
    monkeypatch.setattr(legacy, "_precision_flags", lambda _torch: nullcontext())
    monkeypatch.setattr(legacy, "_cuda_autocast_disabled", lambda _torch: nullcontext())

    def fail_materialization(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(legacy, "_materialize_arm", fail_materialization)
    result = run_native_fusion_suite(_SUITE)
    assert result.outcome == "failed"
    assert result.materialization == ()
    assert len(result.observations) == 12
    assert all(item.status == "blocked" for item in result.observations)
    assert _parse(result, result.to_dict()).outcome == "failed"

    tampered = deepcopy(result.to_dict())
    stage = cast(dict[str, dict[str, object]], tampered["stage_outcomes"])
    stage["rmsnorm-triton-w4-correctness"]["failure_kind"] = "executor"
    stage["rmsnorm-triton-w4-correctness"]["error"] = "materialization failed"
    with pytest.raises(SchemaError, match="underlying evidence"):
        _parse(result, tampered)


def _fake_materialization(arm: str) -> TensorMaterialization:
    descriptors: list[dict[str, object]] = []
    for tensor_id, role, shape, scale, offset, digest in (
        ("input", "input", [128, 4096], 1.0, 0.0, "1" * 64),
        ("residual", "input", [128, 4096], 1.0, 0.0, "2" * 64),
        ("gamma", "parameter", [4096], 0.02, 1.0, "3" * 64),
    ):
        descriptors.append(
            {
                "tensor_id": tensor_id,
                "role": role,
                "shape": shape,
                "draw": "normal_0_1_fp32_cpu",
                "normal_scale": scale,
                "normal_offset": offset,
                "cpu_dtype": "float32",
                "storage_dtype": "bfloat16",
                "device": "cuda:0",
                "contiguous": True,
                "alignment_bytes": 16,
                "alignment_satisfied": True,
                "storage_sha256": digest,
            }
        )
    return TensorMaterialization(
        NATIVE_RMSNORM_SUITE_SHA256,
        "rmsnorm-case-001",
        arm,
        17,
        ("input", "residual", "gamma"),
        tuple(descriptors),
    )


def test_oracle_failure_retains_compile_resource_and_terminalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import heliostune.fusion_kernels as fusion
    import heliostune.local_executor as legacy
    import heliostune.native_fusion_executor as executor

    fake_torch = SimpleNamespace()
    monkeypatch.setattr(executor, "_probe_capability", lambda: (_available(), fake_torch, "3.4.0"))
    monkeypatch.setattr(legacy, "_precision_flags", lambda _torch: nullcontext())
    monkeypatch.setattr(legacy, "_cuda_autocast_disabled", lambda _torch: nullcontext())

    def materialize(
        _torch: object,
        _suite: object,
        _case: object,
        arm: str,
        _digest: str,
        _device: int,
    ) -> tuple[dict[str, object], TensorMaterialization]:
        return (
            {"input": object(), "residual": object(), "gamma": object()},
            _fake_materialization(arm),
        )

    monkeypatch.setattr(legacy, "_materialize_arm", materialize)
    monkeypatch.setattr(fusion, "compile_residual_rmsnorm", lambda *_args: object())

    def compiled_evidence(_compiled: object, config: object) -> dict[str, object]:
        warps = cast(Any, config).num_warps
        name = f"kernel_w{warps}"
        target = {"backend": "cuda", "arch": 90, "warp_size": 32}
        return {
            "status": "compiled",
            "error": None,
            "kernel_name": name,
            "kernel_hash": hashlib.sha256(name.encode()).hexdigest(),
            "target": target,
            "metadata": {
                "target": target,
                "shared": 0,
                "num_warps": warps,
                "num_ctas": 1,
                "num_stages": 1,
            },
            "n_regs": 32,
            "n_spills": 0,
            "n_max_threads": 128,
            "asm_stages": [{"stage": "cubin", "bytes": 1, "sha256": "4" * 64}],
            "resource_gate_passed": True,
        }

    monkeypatch.setattr(fusion, "compiled_kernel_evidence", compiled_evidence)
    monkeypatch.setattr(fusion, "load_residual_rmsnorm", lambda _entrypoint: object())

    def fail_oracle(*_args: object) -> object:
        raise RuntimeError("oracle failed")

    monkeypatch.setattr(legacy, "_residual_rmsnorm", fail_oracle)
    result = run_native_fusion_suite(_SUITE)
    assert result.outcome == "failed"
    assert all(
        item["status"] == "compiled"
        for item in result.compile_evidence.values()
        if item["arm_id"] in _RUNTIME_ARMS[:4]
    )
    assert all(item["resource_gate_passed"] is True for item in result.resource_evidence.values())
    assert _parse(result, result.to_dict()).outcome == "failed"

    wrong_hash = deepcopy(result.to_dict())
    materialization = cast(list[dict[str, object]], wrong_hash["materialization"])
    tensors = cast(list[dict[str, object]], materialization[0]["tensors"])
    tensors[0]["storage_sha256"] = "f" * 64
    with pytest.raises(SchemaError, match="cross-arm"):
        _parse(result, wrong_hash)

    native_prefix = deepcopy(result.to_dict())
    materialization = cast(list[dict[str, object]], native_prefix["materialization"])
    del materialization[2:]
    _parse(result, native_prefix)
    tensors = cast(list[dict[str, object]], materialization[1]["tensors"])
    tensors[0]["storage_sha256"] = "f" * 64
    with pytest.raises(SchemaError, match="cross-arm"):
        _parse(result, native_prefix)

    wrong_name = deepcopy(result.to_dict())
    profiles = cast(dict[str, dict[str, object]], wrong_name["profile_evidence"])
    profiles["rmsnorm-triton-w4-correctness"]["expected_kernel_name"] = "other"
    with pytest.raises(SchemaError, match="kernel identity"):
        _parse(result, wrong_name)
