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
    _force_eager_reason,
    _names_digest,
    _parse_compile,
    _parse_executor_sources,
    _parse_profile,
    _parse_resource,
    _parse_stage,
    _parse_validation,
    _probe_capability,
    _profile_once,
    _run_correctness,
    _run_validation_battery,
    _safe_error,
    _timing,
    _validate_frozen_suite,
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


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("schema", "schema differs"),
        ("package_source_count", "at least"),
        ("source_path", "order/path"),
        ("source_bytes", "at least"),
    ],
)
def test_executor_source_inventory_rejects_substitution_and_loss(
    mutation: str, match: str
) -> None:
    inventory = _capture_executor_sources()
    if mutation == "source_path":
        cast(list[dict[str, object]], inventory["sources"])[0]["path"] = "substitute.py"
    elif mutation == "source_bytes":
        cast(list[dict[str, object]], inventory["sources"])[0]["bytes"] = 0
    elif mutation == "schema":
        inventory["schema"] = "heliostune.executor-sources/2"
    else:
        inventory["package_source_count"] = 0

    with pytest.raises(SchemaError, match=match):
        _parse_executor_sources(inventory)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("identity", "identity/template"),
        ("case_count", "one case"),
        ("case", "case differs"),
        ("arm_order", "arm order"),
        ("entrypoint", "entrypoint"),
        ("roles", "arm roles"),
        ("correctness", "correctness policy"),
        ("timing", "timing policy"),
        ("cells", "expected cells"),
        ("cell_link", "cell policy/linkage"),
        ("gate", "timing gate"),
    ],
)
def test_frozen_suite_validator_rejects_every_execution_selector_drift(
    mutation: str, match: str
) -> None:
    suite = verify_suite(_SUITE).suite

    def replaced(value: Any, **changes: object) -> SimpleNamespace:
        names = value.__dataclass_fields__
        fields = {name: getattr(value, name) for name in names}
        fields.update(changes)
        return SimpleNamespace(**fields)

    frozen = replaced(suite)
    if mutation == "identity":
        frozen.suite_id = "residual-rmsnorm-triton-substitute"
    elif mutation == "case_count":
        frozen.cases = ()
    elif mutation == "case":
        frozen.cases = (replaced(suite.cases[0], input_seed=18),)
    elif mutation == "arm_order":
        frozen.arms = tuple(reversed(suite.arms))
    elif mutation == "entrypoint":
        frozen.arms = (replaced(suite.arms[0], entrypoint="substitute.kernel"), *suite.arms[1:])
    elif mutation == "roles":
        frozen.arms = (replaced(suite.arms[0], role="reference"), *suite.arms[1:])
    elif mutation == "correctness":
        frozen.correctness_policies = ()
    elif mutation == "timing":
        frozen.timing_policies = ()
    elif mutation == "cells":
        frozen.expected_cells = tuple(reversed(suite.expected_cells))
    elif mutation == "cell_link":
        frozen.expected_cells = (
            replaced(suite.expected_cells[0], input_seed=18),
            *suite.expected_cells[1:],
        )
    else:
        frozen.executor_rule = "timing_without_retained_correctness"

    with pytest.raises(SchemaError, match=match):
        _validate_frozen_suite(cast(Any, frozen), NATIVE_RMSNORM_SUITE_SHA256)


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


@pytest.mark.parametrize(
    ("guard", "expected"),
    [
        ("environment", "TORCHDYNAMO_DISABLE"),
        ("disabled", "config.disable"),
        ("suppressed", "suppress_errors"),
    ],
)
def test_comparator_rejects_every_eager_fallback_guard(
    monkeypatch: pytest.MonkeyPatch, guard: str, expected: str
) -> None:
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    config = SimpleNamespace(disable=guard == "disabled", suppress_errors=guard == "suppressed")
    if guard == "environment":
        monkeypatch.setenv("TORCHDYNAMO_DISABLE", "yes")
    assert expected in cast(str, _force_eager_reason(SimpleNamespace(_dynamo=SimpleNamespace(config=config))))


@pytest.mark.parametrize("compile_return", ["noncallable", "original"])
def test_comparator_rejects_compile_results_without_distinct_callable_custody(
    compile_return: str,
) -> None:
    def kernel() -> None:
        return None

    registry = SimpleNamespace(lookup_backend=lambda _name: lambda _graph, _inputs: object())

    def compile_callable(candidate: Any, **_kwargs: object) -> object:
        return object() if compile_return == "noncallable" else candidate

    torch = SimpleNamespace(
        _dynamo=SimpleNamespace(
            config=SimpleNamespace(disable=False, suppress_errors=False),
            backends=SimpleNamespace(registry=registry),
        ),
        compile=compile_callable,
    )
    with pytest.raises(RuntimeError, match="callable|original eager"):
        _compile_comparator(torch, kernel, {})


@pytest.mark.parametrize(
    ("scenario", "available", "triton_version"),
    [
        ("torch_missing", False, None),
        ("torch_unavailable", False, None),
        ("triton_missing", False, None),
        ("wrong_triton", False, "3.3.0"),
        ("available", True, "3.4.0"),
    ],
)
def test_native_capability_probe_preserves_exact_failure_shape(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    available: bool,
    triton_version: str | None,
) -> None:
    import heliostune.local_executor as legacy

    fake_torch = SimpleNamespace()

    def import_module(name: str) -> object:
        if name == "torch":
            if scenario == "torch_missing":
                raise RuntimeError("torch import failed")
            return fake_torch
        if name == "triton":
            if scenario == "triton_missing":
                raise RuntimeError("triton import failed")
            return SimpleNamespace(__version__="3.3.0" if scenario == "wrong_triton" else "3.4.0")
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(
        legacy,
        "_probe_torch",
        lambda _torch, _suite: _unavailable() if scenario == "torch_unavailable" else _available(),
    )

    capability, torch, version = _probe_capability()

    assert capability.available is available
    assert version == triton_version
    assert torch is (None if scenario == "torch_missing" else fake_torch)
    if not available:
        assert capability.reasons


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


def test_failed_compile_parser_retains_only_consistent_native_progress(
    aborted: NativeFusionExecutionResult,
) -> None:
    cell = "rmsnorm-triton-w4-correctness"
    failed = dict(aborted.compile_evidence[cell])
    failed.update(
        status="failed",
        error="loader failed",
        compile_ns=17,
        backend_invoked=True,
        callable_distinct=False,
        kernel_name="kernel_w4",
        kernel_hash="a" * 64,
    )
    parsed = _parse_compile(failed, cell)
    assert parsed["backend_invoked"] is True
    assert parsed["callable_distinct"] is False
    assert parsed["kernel_name"] == "kernel_w4"

    for mutation in (
        {"backend_invoked": False},
        {"backend_invoked": False, "callable_distinct": False},
        {"callable_distinct": True},
    ):
        tampered = {**failed, **mutation}
        with pytest.raises(SchemaError, match="inconsistent"):
            _parse_compile(tampered, cell)

    blocked = dict(failed)
    blocked.update(status="blocked", compile_ns=None)
    with pytest.raises(SchemaError, match="blocked compile"):
        _parse_compile(blocked, cell)

    completed = dict(failed)
    completed.update(status="compiled", error=None, callable_distinct=False)
    with pytest.raises(SchemaError, match="passing evidence"):
        _parse_compile(completed, cell)


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


def test_profile_retains_launch_progress_when_profiler_events_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import heliostune.local_executor as legacy

    cuda_type = object()

    class FailingEventsProfile:
        def __enter__(self) -> FailingEventsProfile:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def events(self) -> tuple[object, ...]:
            raise RuntimeError("events failed")

    fake_torch = SimpleNamespace(
        autograd=SimpleNamespace(DeviceType=SimpleNamespace(CUDA=cuda_type)),
        cuda=SimpleNamespace(synchronize=lambda _device: None),
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CUDA=object()),
            profile=lambda **_kwargs: FailingEventsProfile(),
        ),
    )
    monkeypatch.setattr(legacy, "_tensor_hash", lambda _torch, _tensor: "d" * 64)
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
    assert invocations == 2
    assert evidence["status"] == "failed"
    assert evidence["warmed"] is True
    assert evidence["invocation_count"] == 1
    assert evidence["cuda_event_count"] == 0
    assert evidence["cuda_event_names_sha256"] is None
    assert "events failed" in cast(str, evidence["error"])
    _parse_profile(evidence, "rmsnorm-triton-w4-correctness")


def test_profile_retains_event_and_output_evidence_when_input_revalidation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import heliostune.local_executor as legacy

    cuda_type = object()
    event_name = "residual_rmsnorm_kernel"

    class FakeProfile:
        def __enter__(self) -> FakeProfile:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def events(self) -> tuple[SimpleNamespace, ...]:
            return (SimpleNamespace(name=event_name, device_type=cuda_type),)

    fake_torch = SimpleNamespace(
        autograd=SimpleNamespace(DeviceType=SimpleNamespace(CUDA=cuda_type)),
        cuda=SimpleNamespace(synchronize=lambda _device: None),
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CUDA=object()),
            profile=lambda **_kwargs: FakeProfile(),
        ),
    )
    hash_calls = 0

    def tensor_hash(_torch: object, _tensor: object) -> str:
        nonlocal hash_calls
        hash_calls += 1
        if hash_calls > 1:
            raise RuntimeError("input revalidation failed")
        return "d" * 64

    monkeypatch.setattr(legacy, "_tensor_hash", tensor_hash)
    monkeypatch.setattr(
        legacy,
        "_validate_correctness",
        lambda *_args, **_kwargs: SimpleNamespace(status="passed"),
    )
    evidence = _profile_once(
        fake_torch,
        "rmsnorm-triton-w4",
        lambda *_arguments: object(),
        (object(), object(), object()),
        object(),
        {"input": object()},
        0,
        event_name,
        "a" * 64,
    )
    assert evidence["status"] == "failed"
    assert evidence["warmed"] is True
    assert evidence["invocation_count"] == 1
    assert evidence["cuda_event_names_sample"] == [event_name]
    assert evidence["cuda_event_names_sha256"] == _names_digest([event_name])
    assert evidence["exact_name_match_count"] == 1
    assert evidence["output_revalidated"] is True
    assert evidence["inputs_revalidated"] is False
    assert "input revalidation failed" in cast(str, evidence["error"])
    _parse_profile(evidence, "rmsnorm-triton-w4-correctness")


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


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("backend", "backend differs"),
        ("native_null", "nullability"),
        ("completed_failure", "passing evidence"),
        ("completed_closed", "passing evidence"),
        ("failed_open", "failure evidence"),
        ("failed_missing", "failure evidence"),
    ],
)
def test_stage_status_matrix_rejects_paid_gate_claim_tampering(
    aborted: NativeFusionExecutionResult, mutation: str, match: str
) -> None:
    cell = "rmsnorm-triton-w4-correctness"
    record = dict(aborted.stage_outcomes[cell])
    if mutation == "backend":
        record["backend_kind"] = "eager"
    elif mutation == "native_null":
        record["resource_passed"] = None
    elif mutation.startswith("completed"):
        record.update(
            status="completed",
            failure_kind=None,
            error=None,
            correctness_passed=True,
            resource_passed=True,
            profile_passed=True,
            validation_passed=True,
            timing_allowed=True,
        )
        if mutation == "completed_failure":
            record["error"] = "hidden failure"
        else:
            record["validation_passed"] = False
            record["timing_allowed"] = False
    elif mutation == "failed_open":
        record.update(
            status="failed",
            correctness_passed=True,
            resource_passed=True,
            profile_passed=True,
            validation_passed=True,
            timing_allowed=True,
        )
    else:
        record.update(status="failed", failure_kind=None)

    with pytest.raises(SchemaError, match=match):
        _parse_stage(record, cell)


@pytest.mark.parametrize(
    "mutation",
    [
        "dynamic",
        "eager_fallback",
        "fullgraph",
        "mode",
        "config",
        "blocked_error",
        "blocked_duration",
        "blocked_backend",
        "blocked_callable",
        "blocked_name",
        "blocked_hash",
    ],
)
def test_compile_policy_and_blocked_claim_matrix_is_fail_closed(
    aborted: NativeFusionExecutionResult, mutation: str
) -> None:
    cell = "rmsnorm-triton-w4-correctness"
    record = dict(aborted.compile_evidence[cell])
    if mutation == "dynamic":
        record["dynamic"] = True
    elif mutation == "eager_fallback":
        record["eager_fallback"] = True
    elif mutation == "fullgraph":
        record["fullgraph"] = True
    elif mutation == "mode":
        record["mode"] = "default"
    elif mutation == "config":
        config = cast(dict[str, object], deepcopy(record["config"]))
        config["num_warps"] = 8
        record["config"] = config
    elif mutation == "blocked_error":
        record["error"] = None
    elif mutation == "blocked_duration":
        record["compile_ns"] = 1
    elif mutation == "blocked_backend":
        record["backend_invoked"] = True
    elif mutation == "blocked_callable":
        record["callable_distinct"] = True
    elif mutation == "blocked_name":
        record["kernel_name"] = "paid_kernel"
    else:
        record["kernel_hash"] = "a" * 64

    with pytest.raises(SchemaError, match="policy/config|blocked compile"):
        _parse_compile(record, cell)


@pytest.mark.parametrize("mutation", ["error", "gate", "complete"])
def test_failed_resource_status_cannot_hide_success_or_drop_its_error(mutation: str) -> None:
    cell = "rmsnorm-triton-w4-correctness"
    record = _resource()
    record.update(status="failed", error="resource extraction failed", n_spills=1)
    record["resource_gate_passed"] = False
    _parse_resource(record, cell)
    if mutation == "error":
        record["error"] = None
    elif mutation == "gate":
        record["resource_gate_passed"] = True
    else:
        record["n_spills"] = 0

    with pytest.raises(SchemaError, match="non-completed"):
        _parse_resource(record, cell)


@pytest.mark.parametrize("claim", ["kernel_name", "target", "asm_stages"])
def test_blocked_resource_status_rejects_partial_paid_evidence_claims(
    aborted: NativeFusionExecutionResult, claim: str
) -> None:
    cell = "rmsnorm-triton-w4-correctness"
    record = dict(aborted.resource_evidence[cell])
    if claim == "kernel_name":
        record[claim] = "paid_kernel"
    elif claim == "target":
        record[claim] = {"backend": "cuda", "arch": "90", "warp_size": 32}
    else:
        record[claim] = [{"stage": "cubin", "bytes": 1, "sha256": "a" * 64}]

    with pytest.raises(SchemaError, match="blocked resource"):
        _parse_resource(record, cell)


@pytest.mark.parametrize("mutation", ["error", "gate", "complete"])
def test_failed_profile_status_cannot_hide_success_or_drop_its_error(mutation: str) -> None:
    cell = "rmsnorm-triton-w4-correctness"
    record = _profile(["residual_rmsnorm_kernel"])
    record.update(
        status="failed",
        error="profile post-check failed",
        inputs_revalidated=False,
        one_kernel_gate_passed=False,
    )
    _parse_profile(record, cell)
    if mutation == "error":
        record["error"] = None
    elif mutation == "gate":
        record["one_kernel_gate_passed"] = True
    else:
        record["inputs_revalidated"] = True

    with pytest.raises(SchemaError, match="non-completed"):
        _parse_profile(record, cell)


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("warmed", True),
        ("invocation_count", 1),
        ("output_revalidated", True),
        ("inputs_revalidated", True),
    ],
)
def test_blocked_profile_status_rejects_partial_runtime_claims(
    aborted: NativeFusionExecutionResult, claim: str, value: object
) -> None:
    cell = "rmsnorm-triton-w4-correctness"
    record = dict(aborted.profile_evidence[cell])
    record[claim] = value
    with pytest.raises(SchemaError, match="blocked profile"):
        _parse_profile(record, cell)


@pytest.mark.parametrize("mutation", ["error", "gate", "complete"])
def test_failed_validation_status_cannot_hide_success_or_drop_its_error(mutation: str) -> None:
    cell = "rmsnorm-triton-w4-correctness"
    record = _validation()
    probes = cast(list[dict[str, object]], record["probes"])
    probes[0]["deterministic"] = False
    probes[0]["passed"] = False
    record.update(
        status="failed",
        error="structured probe failed",
        validation_gate_passed=False,
    )
    _parse_validation(record, cell)
    if mutation == "error":
        record["error"] = None
    elif mutation == "gate":
        record["validation_gate_passed"] = True
    else:
        probes[0]["deterministic"] = True
        probes[0]["passed"] = True

    with pytest.raises(SchemaError, match="non-completed"):
        _parse_validation(record, cell)



@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "heliostune.local-environment/1"),
        ("precision_policy", {}),
        ("autocast_policy", {}),
        ("torch_version", "2.8.0"),
        ("device_index", 0),
        ("fusion_claim", True),
    ],
)
def test_environment_rejects_policy_and_capability_shape_substitution(
    aborted: NativeFusionExecutionResult, field: str, value: object
) -> None:
    payload = deepcopy(aborted.to_dict())
    cast(dict[str, object], payload["environment"])[field] = value
    with pytest.raises(SchemaError, match="environment policy/capability"):
        _parse(aborted, payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "expected_cells",
        "terminal_cells",
        "passed",
        "failed",
        "blocked",
        "terminal_flag",
        "counts",
        "outcome",
        "fusion_claim",
    ],
)
def test_summary_rejects_cell_inventory_and_evidence_count_tampering(
    aborted: NativeFusionExecutionResult, mutation: str
) -> None:
    payload = deepcopy(aborted.to_dict())
    summary = cast(dict[str, Any], payload["summary"])
    if mutation == "expected_cells":
        cast(list[object], summary["expected_cell_ids"]).reverse()
    elif mutation == "terminal_cells":
        cast(list[object], summary["terminal_cell_ids"]).reverse()
    elif mutation in {"passed", "failed", "blocked"}:
        summary[mutation] += 1
    elif mutation == "terminal_flag":
        summary["all_cells_terminal"] = False
    elif mutation == "counts":
        cast(dict[str, int], summary["counts"])["stage_blocked"] -= 1
    elif mutation == "outcome":
        summary["outcome"] = "failed"
    else:
        summary["fusion_claim"] = True

    with pytest.raises(SchemaError, match="summary does not match"):
        _parse(aborted, payload)


@pytest.mark.parametrize("mutation", ["observation_order", "attempt_id", "outcome"])
def test_result_cross_links_reject_order_attempt_and_outcome_custody_tampering(
    aborted: NativeFusionExecutionResult, mutation: str
) -> None:
    payload = deepcopy(aborted.to_dict())
    if mutation == "observation_order":
        cast(list[object], payload["observations"]).reverse()
        cast(list[object], cast(dict[str, object], payload["summary"])["terminal_cell_ids"]).reverse()
    elif mutation == "attempt_id":
        cast(list[dict[str, object]], payload["attempts"])[0]["attempt_id"] = 2
    else:
        payload["outcome"] = "failed"
        cast(dict[str, object], payload["summary"])["outcome"] = "failed"

    with pytest.raises(SchemaError, match="runtime order|attempt IDs|failed outcome"):
        _parse(aborted, payload)


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


def _install_native_pipeline_fakes(monkeypatch: pytest.MonkeyPatch) -> Any:
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

    def compile_kernel(entrypoint: str, *_args: object) -> SimpleNamespace:
        name = f"kernel_{entrypoint.rsplit('_', maxsplit=1)[-1]}"
        return SimpleNamespace(name=name, hash=hashlib.sha256(name.encode()).hexdigest())

    monkeypatch.setattr(legacy, "_materialize_arm", materialize)
    monkeypatch.setattr(fusion, "compile_residual_rmsnorm", compile_kernel)
    monkeypatch.setattr(
        legacy,
        "_residual_rmsnorm",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("oracle failed")),
    )
    return fusion


def _fake_compiled_resource(compiled: object, config: object) -> dict[str, object]:
    warps = cast(Any, config).num_warps
    target = {"backend": "cuda", "arch": 90, "warp_size": 32}
    return {
        "status": "compiled",
        "error": None,
        "kernel_name": cast(Any, compiled).name,
        "kernel_hash": cast(Any, compiled).hash,
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


@pytest.mark.parametrize(
    ("failing_gate", "failure_kind"),
    [("validation", "validation_gate"), ("profile", "profile_gate")],
)
def test_passing_correctness_cannot_bypass_paid_validation_or_profile_failure(
    monkeypatch: pytest.MonkeyPatch, failing_gate: str, failure_kind: str
) -> None:
    import heliostune.local_executor as legacy
    import heliostune.native_fusion_executor as executor

    fusion = _install_native_pipeline_fakes(monkeypatch)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _device: None))
    monkeypatch.setattr(
        executor, "_probe_capability", lambda: (_available(), fake_torch, "3.4.0")
    )
    monkeypatch.setattr(fusion, "compiled_kernel_evidence", _fake_compiled_resource)
    monkeypatch.setattr(fusion, "load_residual_rmsnorm", lambda _entrypoint: object())
    monkeypatch.setattr(legacy, "_residual_rmsnorm", lambda *_args: object())

    def passing_correctness(
        _torch: object,
        arm: str,
        _kernel: object,
        _arguments: object,
        _expected: object,
        _inputs: object,
        _device_index: int,
    ) -> tuple[object, object]:
        nested = legacy.CorrectnessObservation(
            "passed",
            _correctness_key(f"{arm}-correctness"),
            None,
            None,
            {
                "shape": [128, 4096],
                "device": "cuda:0",
                "dtype": "torch.bfloat16",
                "layout": "torch.strided",
                "contiguous": True,
            },
            True,
            True,
            True,
            True,
            0.0,
        )
        return (
            legacy.CellObservation(
                f"{arm}-correctness",
                "rmsnorm-case-001",
                arm,
                "correctness",
                "passed",
                nested,
                None,
            ),
            object(),
        )

    def passing_validation(
        _torch: object,
        arm: str,
        _kernel: object,
        _reference: object,
        _device_index: int,
    ) -> dict[str, object]:
        evidence = _validation()
        evidence["arm_id"] = arm
        evidence["entrypoint"] = _ENTRYPOINT[arm]
        return evidence

    monkeypatch.setattr(executor, "_run_correctness", passing_correctness)
    if failing_gate == "validation":
        monkeypatch.setattr(
            executor,
            "_run_validation_battery",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("validation runtime failed")),
        )
    else:
        monkeypatch.setattr(executor, "_run_validation_battery", passing_validation)
        monkeypatch.setattr(
            executor,
            "_profile_once",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("profile runtime failed")),
        )

    result = run_native_fusion_suite(_SUITE)

    assert result.outcome == "failed"
    for stage in result.stage_outcomes.values():
        if stage["arm_id"] not in _RUNTIME_ARMS[:4]:
            continue
        assert stage["correctness_passed"] is True
        assert stage["failure_kind"] == failure_kind
        assert stage["timing_allowed"] is False
        assert result.observations[_RUNTIME_ARMS.index(stage["arm_id"]) * 2].status == "passed"
    assert _parse(result, result.to_dict()).outcome == "failed"
    if failing_gate == "validation":
        for mutation, match in (
            ("materialization_order", "suite-order prefix"),
            ("materialization_link", "materialization linkage"),
            ("missing_materialization", "passing correctness lacks"),
            ("stage_correctness", "stage/correctness linkage"),
            ("stage_gate", "stage native gate linkage"),
            ("observation_link", "observation cell linkage"),
            ("output", "output descriptor"),
            ("backend_flag", "environment backend flag"),
        ):
            payload = deepcopy(result.to_dict())
            if mutation == "materialization_order":
                records = cast(list[object], payload["materialization"])
                records[0], records[1] = records[1], records[0]
            elif mutation == "materialization_link":
                cast(list[dict[str, object]], payload["materialization"])[0]["input_seed"] = 18
            elif mutation == "missing_materialization":
                cast(list[object], payload["materialization"]).clear()
            elif mutation == "stage_correctness":
                stages = cast(dict[str, dict[str, object]], payload["stage_outcomes"])
                stages["rmsnorm-triton-w4-correctness"]["correctness_passed"] = False
            elif mutation == "stage_gate":
                stages = cast(dict[str, dict[str, object]], payload["stage_outcomes"])
                stages["rmsnorm-triton-w4-correctness"]["profile_passed"] = True
            elif mutation == "observation_link":
                cast(list[dict[str, object]], payload["observations"])[0]["case_id"] = "other"
            elif mutation == "output":
                observation = cast(list[dict[str, Any]], payload["observations"])[0]
                cast(dict[str, object], observation["correctness"])["output"] = {
                    "shape": [128, 4096],
                    "device": "cuda:1",
                    "dtype": "torch.bfloat16",
                    "layout": "torch.strided",
                    "contiguous": True,
                }
            else:
                cast(dict[str, object], payload["environment"])["backend_invoked"] = False
            with pytest.raises(SchemaError, match=match):
                _parse(result, payload)


def test_resource_extractor_failure_retains_compile_return_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fusion = _install_native_pipeline_fakes(monkeypatch)
    monkeypatch.setattr(
        fusion,
        "compiled_kernel_evidence",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("resource extraction failed")),
    )
    result = run_native_fusion_suite(_SUITE)
    assert result.outcome == "failed"
    assert result.environment["backend_invoked"] is True
    for cell, compile_record in result.compile_evidence.items():
        if compile_record["arm_id"] not in _RUNTIME_ARMS[:4]:
            continue
        resource_record = result.resource_evidence[cell]
        stage = result.stage_outcomes[cell]
        assert compile_record["status"] == "failed"
        assert compile_record["backend_invoked"] is True
        assert compile_record["callable_distinct"] is False
        assert compile_record["kernel_name"] == resource_record["kernel_name"]
        assert compile_record["kernel_hash"] == resource_record["kernel_hash"]
        assert resource_record["status"] == "failed"
        assert "resource extraction failed" in cast(str, compile_record["error"])
        assert stage["failure_kind"] == "compile_failed"
        assert stage["error"] == compile_record["error"]
    assert _parse(result, result.to_dict()).outcome == "failed"


def test_loader_failure_retains_passing_resource_and_drives_compile_stage_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fusion = _install_native_pipeline_fakes(monkeypatch)
    monkeypatch.setattr(fusion, "compiled_kernel_evidence", _fake_compiled_resource)
    monkeypatch.setattr(
        fusion,
        "load_residual_rmsnorm",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("loader failed")),
    )
    result = run_native_fusion_suite(_SUITE)
    assert result.outcome == "failed"
    for cell, resource_record in result.resource_evidence.items():
        compile_record = result.compile_evidence[cell]
        stage = result.stage_outcomes[cell]
        assert resource_record["status"] == "compiled"
        assert resource_record["resource_gate_passed"] is True
        assert compile_record["status"] == "failed"
        assert compile_record["backend_invoked"] is True
        assert compile_record["callable_distinct"] is False
        assert compile_record["kernel_name"] == resource_record["kernel_name"]
        assert compile_record["kernel_hash"] == resource_record["kernel_hash"]
        assert "loader failed" in cast(str, compile_record["error"])
        assert stage["resource_passed"] is True
        assert stage["failure_kind"] == "compile_failed"
        assert stage["error"] == compile_record["error"]
    assert _parse(result, result.to_dict()).outcome == "failed"


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


@pytest.mark.parametrize(
    ("mode", "failure_kind"),
    [
        ("passing", None),
        ("mutation", "mutation"),
        ("value_class", "value_class"),
        ("sign", "sign"),
        ("determinism", "determinism"),
    ],
)
def test_correctness_runtime_defends_repeatability_and_value_custody(
    monkeypatch: pytest.MonkeyPatch, mode: str, failure_kind: str | None
) -> None:
    import heliostune.local_executor as legacy

    class Tensor:
        def __init__(
            self,
            token: str,
            pointer: int,
            classes: tuple[bool, bool, bool] = (False, False, False),
            sign: bool = False,
        ) -> None:
            self.token = token
            self.pointer = pointer
            self.classes = classes
            self.sign = sign
            self.digest = hashlib.sha256(token.encode()).hexdigest()

    actual = Tensor(
        "actual",
        10,
        classes=(True, False, False) if mode == "value_class" else (False, False, False),
        sign=mode == "sign",
    )
    expected = Tensor("expected", 20)
    repeated = Tensor("different" if mode == "determinism" else "actual", 11)
    input_tensor = Tensor("input", 11 if mode == "mutation" else 1)
    outputs = iter((actual, repeated))

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda _device: None),
        equal=lambda left, right: (
            left.token == right.token
            if isinstance(left, Tensor) and isinstance(right, Tensor)
            else left == right
        ),
        isnan=lambda tensor: tensor.classes[0],
        isposinf=lambda tensor: tensor.classes[1],
        isneginf=lambda tensor: tensor.classes[2],
        signbit=lambda tensor: tensor.sign,
    )
    monkeypatch.setattr(legacy, "_storage_pointer", lambda tensor: tensor.pointer)
    monkeypatch.setattr(legacy, "_tensor_hash", lambda _torch, tensor: tensor.digest)
    monkeypatch.setattr(
        legacy,
        "_validate_correctness",
        lambda *_args, **_kwargs: legacy.CorrectnessObservation(
            "passed",
            _correctness_key("rmsnorm-triton-w4-correctness"),
            None,
            None,
            {
                "shape": [128, 4096],
                "device": "cuda:0",
                "dtype": "torch.bfloat16",
                "layout": "torch.strided",
                "contiguous": True,
            },
            True,
            True,
            True,
            True,
            0.0,
        ),
    )

    observation, returned = _run_correctness(
        fake_torch,
        "rmsnorm-triton-w4",
        lambda *_args: next(outputs),
        (),
        expected,
        {"input": input_tensor},
        0,
    )

    assert observation.correctness.failure_kind == failure_kind
    assert observation.status == ("passed" if failure_kind is None else "failed")
    assert returned is actual


def test_validation_runtime_exceptions_fail_closed_for_every_probe() -> None:
    def fail_zeros(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("CUDA allocation failed")

    fake_torch = SimpleNamespace(bfloat16="bfloat16", zeros=fail_zeros)
    evidence = _run_validation_battery(
        fake_torch,
        "rmsnorm-triton-w4",
        lambda *_args: pytest.fail("kernel must not run after probe setup failure"),
        lambda *_args: pytest.fail("reference must not run after probe setup failure"),
        0,
    )

    assert evidence["status"] == "failed"
    assert evidence["validation_gate_passed"] is False
    probes = cast(list[dict[str, object]], evidence["probes"])
    assert [probe["id"] for probe in probes] == ["zeros", "cancellation", "overflow"]
    assert all(probe["passed"] is False for probe in probes)
    _parse_validation(evidence, "rmsnorm-triton-w4-correctness")


@pytest.mark.parametrize("close", [True, False])
def test_validation_runtime_binds_all_structured_probe_checks(
    monkeypatch: pytest.MonkeyPatch, close: bool
) -> None:
    import heliostune.local_executor as legacy

    class Tensor:
        next_pointer = 1

        def __init__(
            self,
            token: str,
            *,
            shape: tuple[int, ...] = (128, 4096),
            device: str = "cuda:0",
            dtype: str = "bfloat16",
        ) -> None:
            self.token = token
            self.shape = shape
            self.device = device
            self.dtype = dtype
            self.layout = "strided"
            self.pointer = Tensor.next_pointer
            Tensor.next_pointer += 1
            self.digest = hashlib.sha256(token.encode()).hexdigest()

        def remainder(self, _divisor: int) -> Tensor:
            return self

        def sub(self, _offset: int) -> Tensor:
            return self

        def to(self, *, dtype: str) -> Tensor:
            self.dtype = dtype
            return self

        def expand(self, *_shape: int) -> Tensor:
            self.shape = (128, 4096)
            return self

        def contiguous(self) -> Tensor:
            return self

        def is_contiguous(self) -> bool:
            return True

        def float(self) -> Tensor:
            return self

        def numel(self) -> int:
            return 1

        def __neg__(self) -> Tensor:
            return Tensor(f"negative-{self.token}")

        def __getitem__(self, _key: object) -> Tensor:
            return self

        def __setitem__(self, _key: object, _value: object) -> None:
            return None

        def __sub__(self, _other: object) -> Tensor:
            return Tensor("difference")

    def output() -> Tensor:
        return Tensor("output")

    def assert_close(*_args: object, **_kwargs: object) -> None:
        if not close:
            raise AssertionError("finite values differ")

    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        strided="strided",
        cuda=SimpleNamespace(synchronize=lambda _device: None),
        testing=SimpleNamespace(assert_close=assert_close),
        zeros=lambda shape, *, device, dtype: Tensor(
            "zeros", shape=cast(tuple[int, ...], shape), device=device, dtype=dtype
        ),
        zeros_like=lambda tensor: Tensor(
            "zeros-like", shape=tensor.shape, device=tensor.device, dtype=tensor.dtype
        ),
        arange=lambda _size, *, device, dtype: Tensor(
            "arange", shape=(4096,), device=device, dtype=dtype
        ),
        ones=lambda shape, *, device, dtype: Tensor(
            "ones", shape=cast(tuple[int, ...], shape), device=device, dtype=dtype
        ),
        finfo=lambda _dtype: SimpleNamespace(max=3.0),
        equal=lambda left, right: left == right,
        isnan=lambda _tensor: False,
        isposinf=lambda _tensor: False,
        isneginf=lambda _tensor: False,
        isfinite=lambda _tensor: True,
        signbit=lambda _tensor: False,
        abs=lambda tensor: tensor,
        max=lambda _tensor: SimpleNamespace(item=lambda: 0.0),
    )
    monkeypatch.setattr(legacy, "_tensor_hash", lambda _torch, tensor: tensor.digest)
    monkeypatch.setattr(legacy, "_storage_pointer", lambda tensor: tensor.pointer)

    evidence = _run_validation_battery(
        fake_torch,
        "rmsnorm-triton-w4",
        lambda *_args: output(),
        lambda *_args: output(),
        0,
    )

    assert evidence["status"] == ("validated" if close else "failed")
    assert evidence["validation_gate_passed"] is close
    probes = cast(list[dict[str, object]], evidence["probes"])
    assert all(probe["passed"] is close for probe in probes)
    _parse_validation(evidence, "rmsnorm-triton-w4-correctness")


@pytest.mark.parametrize(
    ("failure_mode", "failure_kind"),
    [("mutation", "mutation"), ("execution", "execution")],
)
def test_timing_runtime_never_preserves_passed_status_after_failure(
    monkeypatch: pytest.MonkeyPatch, failure_mode: str, failure_kind: str
) -> None:
    import heliostune.local_executor as legacy

    hashes = iter(("before", "after" if failure_mode == "mutation" else "before"))
    monkeypatch.setattr(legacy, "_tensor_hash", lambda _torch, _tensor: next(hashes))
    if failure_mode == "execution":
        monkeypatch.setattr(
            legacy,
            "_timing_observation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("timer failed")),
        )
    else:
        monkeypatch.setattr(
            legacy,
            "_timing_observation",
            lambda *_args, **_kwargs: legacy.TimingObservation(
                "passed",
                _correctness_key("rmsnorm-triton-w4-correctness"),
                None,
                None,
                10,
                50,
                (1.0,) * 50,
                1.0,
            ),
        )

    observation = _timing(
        SimpleNamespace(),
        "rmsnorm-triton-w4",
        lambda: None,
        (),
        {"input": object()},
        0,
    )

    assert observation.status == "failed"
    assert observation.timing.failure_kind == failure_kind
