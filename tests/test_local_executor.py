from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import heliostune.local_executor as local
from heliostune.errors import SchemaError
from heliostune.scope import Suite, load_suite

ROOT = Path(__file__).parents[1]
MLP = ROOT / "benchmarks/suites/gated-mlp-epilogue-v1.json"
RMS = ROOT / "benchmarks/suites/residual-rmsnorm-v1.json"


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        cc: tuple[int, int] = (8, 0),
        bf16: bool = True,
        device_error: bool = False,
    ) -> None:
        self.available = available
        self.cc = cc
        self.bf16 = bf16
        self.device_error = device_error
        self.syncs = 0

    def is_available(self) -> bool:
        return self.available

    def current_device(self) -> int:
        if self.device_error:
            raise RuntimeError("device probe exploded")
        return 0

    def get_device_capability(self, _index: int) -> tuple[int, int]:
        return self.cc

    def get_device_name(self, _index: int) -> str:
        return "strict fake accelerator"

    def is_bf16_supported(self, *, including_emulation: bool = False) -> bool:
        assert including_emulation is False
        return self.bf16

    def synchronize(self, _index: int) -> None:
        self.syncs += 1


class FakeTorchProbe:
    bfloat16 = "bfloat16"

    def __init__(
        self,
        *,
        version: str = "2.8.0+cu128",
        cuda: FakeCuda | None = None,
        hip: str | None = None,
        backends: tuple[str, ...] = ("inductor",),
        allocation_error: bool = False,
    ) -> None:
        self.__version__ = version
        self.cuda = cuda or FakeCuda()
        self.version = SimpleNamespace(cuda="12.8", hip=hip)
        self.compiler = SimpleNamespace(list_backends=lambda: backends)
        self.allocation_error = allocation_error
        self.compile = lambda fn, **_kwargs: fn

    def empty(self, *_args: object, **_kwargs: object) -> object:
        if self.allocation_error:
            raise RuntimeError("allocation exploded")
        return object()


def test_torch_absence_is_structured_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        assert name == "torch"
        raise ModuleNotFoundError("No module named torch")

    monkeypatch.setattr(importlib, "import_module", missing)
    probe = local.probe_local_capability(load_suite(MLP))
    assert probe.reasons == ("torch_missing",)
    assert probe.available is False
    assert "Traceback" not in (probe.detail or "")


def test_unavailable_run_is_zero_prefix_structured_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("absent")),
    )
    result = local.run_local_suite(MLP)
    assert result.outcome == "aborted"
    assert result.verified_suite_bytes == MLP.read_bytes()
    assert result.observations == ()
    assert result.attempts == ()
    assert result.materialization == ()
    assert result.summary["terminal_cell_ids"] == []
    assert result.summary["all_cells_terminal"] is False
    assert result.summary["capability_reasons"] == ["torch_missing"]


@pytest.mark.parametrize(
    ("torch", "reason"),
    [
        (FakeTorchProbe(version="2.7.1"), "torch_version_mismatch"),
        (FakeTorchProbe(cuda=FakeCuda(available=False)), "cuda_unavailable"),
        (FakeTorchProbe(hip="6.3"), "rocm_unsupported"),
        (FakeTorchProbe(cuda=FakeCuda(cc=(7, 5))), "compute_capability_too_low"),
        (FakeTorchProbe(cuda=FakeCuda(bf16=False)), "bf16_unsupported"),
        (FakeTorchProbe(backends=()), "inductor_unavailable"),
        (FakeTorchProbe(allocation_error=True), "allocation_failed"),
        (FakeTorchProbe(cuda=FakeCuda(device_error=True)), "device_probe_failed"),
    ],
)
def test_every_capability_rejection_reason(torch: Any, reason: str) -> None:
    probe = local._probe_torch(torch, load_suite(MLP))
    assert probe.available is False
    assert probe.reason == reason
    assert probe.allocation_succeeded is False


def test_complete_fake_capability_is_available() -> None:
    probe = local._probe_torch(FakeTorchProbe(), load_suite(RMS))
    assert probe.available is True
    assert probe.reasons == ()
    assert probe.compute_capability == (8, 0)
    assert probe.native_bf16 is probe.inductor_available is probe.allocation_succeeded is True


def suite_dict(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def test_frozen_hash_registry_matches_current_committed_suite_bytes() -> None:
    assert hashlib.sha256(MLP.read_bytes()).hexdigest() == local.GATED_MLP_SUITE_SHA256
    assert hashlib.sha256(RMS.read_bytes()).hexdigest() == local.RMSNORM_SUITE_SHA256


def test_exact_copied_suite_bytes_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / MLP.name
    copied.write_bytes(MLP.read_bytes())
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("absent")),
    )
    result = local.run_local_suite(copied)
    assert result.outcome == "aborted"
    assert result.verified_suite_sha256 == local.GATED_MLP_SUITE_SHA256


def test_exact_frozen_acceptance_rejects_valid_requirement_policy_and_semantic_drift(
    tmp_path: Path,
) -> None:
    changed_values: list[dict[str, Any]] = []

    requirements = suite_dict(MLP)
    requirements["arms"][0]["requirements"]["min_compute_capability"] = "8.1"
    changed_values.append(requirements)

    timing = suite_dict(MLP)
    timing["timing_policies"][0]["warmups"] = 11
    changed_values.append(timing)

    semantics = suite_dict(MLP)
    semantics["cases"][0]["semantics"]["activation"] = "gelu"
    semantics["cases"][0]["semantics"]["fusion_boundary"][2] = "gelu"
    changed_values.append(semantics)

    for index, value in enumerate(changed_values):
        path = tmp_path / f"drift-{index}.json"
        path.write_text(json.dumps(value))
        with pytest.raises(SchemaError, match="exact committed frozen suite bytes"):
            local.run_local_suite(path)


def test_registry_is_closed_to_four_exact_declared_strings() -> None:
    assert set(local._ENTRYPOINTS) == {
        "reference_template.gated_mlp_candidate",
        "reference_template.gated_mlp_reference",
        "reference_template.residual_rmsnorm_candidate",
        "reference_template.residual_rmsnorm_reference",
    }
    with pytest.raises(SchemaError, match="closed local registry"):
        local._make_kernel(object(), "os.system", load_suite(MLP).cases[0].semantics)


def test_frozen_validator_rejects_valid_nonfrozen_semantics_and_contract() -> None:
    value = suite_dict(MLP)
    value["cases"][0]["semantics"]["activation"] = "gelu"
    value["cases"][0]["semantics"]["fusion_boundary"][2] = "gelu"
    with pytest.raises(SchemaError, match="outside the frozen"):
        local._validate_frozen_suite(Suite.from_dict(value))

    value = suite_dict(MLP)
    contract = value["numeric_contracts"][0]
    for field in ("input", "storage", "output"):
        contract[field]["name"] = "fp16"
    for tensor in value["tensors"]:
        tensor["storage_dtype"] = tensor["logical_dtype"] = "fp16"
    with pytest.raises(SchemaError, match="numeric contract"):
        local._validate_frozen_suite(Suite.from_dict(value))


def test_scope_parser_rejects_template_status_and_domain() -> None:
    for key, changed in (("template_status", "execution_freeze"), ("domain", "rmsnorm_residual")):
        value = suite_dict(MLP)
        value[key] = changed
        with pytest.raises(SchemaError):
            Suite.from_dict(value)


def test_draw_schedules_follow_declared_order_and_distributions() -> None:
    mlp = load_suite(MLP)
    draws = local._resolve_draw_schedule(mlp, mlp.cases[0])
    assert [item.tensor_id for item in draws] == ["input", "gate_weight", "up_weight"]
    assert draws[0].shape == (8, 4096)
    assert draws[1].normal_scale == 1 / math.sqrt(4096)
    rms = load_suite(RMS)
    draws = local._resolve_draw_schedule(rms, rms.cases[0])
    assert [item.tensor_id for item in draws] == ["input", "residual", "gamma"]
    assert (draws[-1].normal_scale, draws[-1].normal_offset) == (0.02, 1.0)


class FakeStorage:
    def __init__(self, pointer: int) -> None:
        self.pointer = pointer

    def data_ptr(self) -> int:
        return self.pointer


class FakeTensor:
    next_pointer = 96

    def __init__(self, array: np.ndarray, dtype: str = "float32", device: str = "cpu") -> None:
        self.array = np.ascontiguousarray(array)
        self.dtype = dtype
        self.device = device
        self.layout = "strided"
        self.pointer = FakeTensor.next_pointer
        FakeTensor.next_pointer += 16

    def __mul__(self, value: float) -> FakeTensor:
        return FakeTensor(self.array * value, self.dtype, self.device)

    def __add__(self, value: float) -> FakeTensor:
        return FakeTensor(self.array + value, self.dtype, self.device)

    def to(
        self, *, dtype: str, device: str | None = None, non_blocking: bool = False
    ) -> FakeTensor:
        assert non_blocking is False
        array = self.array.astype(np.float16 if dtype == "bfloat16" else np.float32)
        return FakeTensor(array, dtype, device or self.device)

    def contiguous(self) -> FakeTensor:
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def is_contiguous(self) -> bool:
        return bool(self.array.flags.c_contiguous)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def view(self, dtype: str) -> FakeTensor:
        assert dtype == "uint8"
        return FakeTensor(self.array.view(np.uint8), dtype)

    def numpy(self) -> np.ndarray:
        return self.array

    def untyped_storage(self) -> FakeStorage:
        return FakeStorage(self.pointer)


class FakeGenerator:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def manual_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)


class FakeMaterializeTorch:
    float32 = "float32"
    bfloat16 = "bfloat16"
    uint8 = "uint8"
    strided = "strided"

    def __init__(self) -> None:
        self.cuda = SimpleNamespace(synchronize=lambda _index: None)

    def Generator(self, *, device: str) -> FakeGenerator:  # noqa: N802
        assert device == "cpu"
        return FakeGenerator()

    def randn(
        self, shape: tuple[int, ...], *, generator: FakeGenerator, dtype: str, device: str
    ) -> FakeTensor:
        assert (dtype, device) == ("float32", "cpu")
        return FakeTensor(generator.rng.standard_normal(shape).astype(np.float32))


def test_materialization_hashes_repeat_per_arm_but_storages_are_disjoint() -> None:
    suite = load_suite(MLP)
    tiny = replace(suite.cases[0], shape=(("batch", 2), ("hidden", 4), ("intermediate", 3)))
    torch = FakeMaterializeTorch()
    tensors_a, record_a = local._materialize_arm(torch, suite, tiny, "a", "a" * 64, 0)
    tensors_b, record_b = local._materialize_arm(torch, suite, tiny, "b", "a" * 64, 0)
    assert record_a.tensor_order == ("input", "gate_weight", "up_weight")
    assert [item["storage_sha256"] for item in record_a.tensors] == [
        item["storage_sha256"] for item in record_b.tensors
    ]
    assert [item["shape"] for item in record_a.tensors] == [[2, 4], [3, 4], [3, 4]]
    assert {local._storage_pointer(x) for x in tensors_a.values()}.isdisjoint(
        {local._storage_pointer(x) for x in tensors_b.values()}
    )


def test_cell_state_machine_and_attempt_statuses() -> None:
    assert local._advance_cell_state("pending", "running") == "running"
    assert local._advance_cell_state("running", "passed") == "passed"
    with pytest.raises(ValueError):
        local._advance_cell_state("pending", "passed")
    cell = load_suite(MLP).expected_cells[0]
    attempts: list[Mapping[str, object]] = []
    states = {cell.id: "pending"}
    local._record_transition(attempts, states, cell, "running")
    local._record_transition(attempts, states, cell, "failed", "compile")
    assert [x["status"] for x in attempts] == ["running", "failure"]
    assert [(x["from_state"], x["to_state"]) for x in attempts] == [
        ("pending", "running"),
        ("running", "failed"),
    ]


def test_timing_gate_key_requires_exact_arm_seed_and_contract() -> None:
    suite = load_suite(MLP)
    correctness, timing = suite.expected_cells[:2]
    key = local._correctness_gate_key("a" * 64, suite.cases[0], correctness)
    assert key == local._correctness_gate_key("a" * 64, suite.cases[0], timing)
    assert key != local._correctness_gate_key(
        "a" * 64, suite.cases[0], replace(timing, arm_id="mlp-reference")
    )
    assert key != local._correctness_gate_key(
        "a" * 64, suite.cases[0], replace(timing, input_seed=18)
    )
    assert local._timing_gate_allows(key, {key})
    assert not local._timing_gate_allows(key, set())
    assert not local._timing_gate_allows(key, {"different-key"})


def test_environment_and_output_records_have_explicit_schema() -> None:
    environment = local._environment_schema(local._probe_torch(FakeTorchProbe(), load_suite(MLP)))
    assert environment["schema"] == "heliostune.local-environment/1"
    assert environment["fusion_claim"] is False
    assert environment["precision_policy"] == {
        "float32_matmul_precision": "highest",
        "allow_tf32": False,
        "allow_bf16_reduced_precision_reduction": False,
        "allow_fp16_reduced_precision_reduction": False,
        "allow_fp16_accumulation": False,
    }
    assert environment["autocast_policy"] == {
        "device_type": "cuda",
        "enabled": False,
        "restore_ambient_state": True,
    }
    assert environment["backend_invoked"] is None
    record = local.TensorMaterialization("a" * 64, "case", "arm", 17, ("x",), ({"tensor_id": "x"},))
    assert record.to_dict()["tensor_order"] == ["x"]


def test_cell_records_and_summary_preserve_expected_suite_order() -> None:
    suite = load_suite(RMS)
    observations = tuple(
        local._failed_cell(cell, f"key-{index}", "correctness_gate", "denied")
        for index, cell in enumerate(suite.expected_cells)
    )
    summary = local._summary(suite, observations, "failed")
    expected_ids = [cell.id for cell in suite.expected_cells]
    assert [item.cell_id for item in observations] == expected_ids
    assert summary["terminal_cell_ids"] == expected_ids
    assert summary["all_cells_terminal"] is True
    assert summary["failed"] == 4


class MatmulFlags:
    allow_tf32 = True
    allow_bf16_reduced_precision_reduction = True
    allow_fp16_reduced_precision_reduction = True
    allow_fp16_accumulation = True


class FakeAutocast:
    def __init__(self, torch: PrecisionTorch, enabled: bool) -> None:
        self.torch = torch
        self.enabled = enabled
        self.previous = torch.autocast_active

    def __enter__(self) -> None:
        self.torch.autocast_active = self.enabled

    def __exit__(self, *_exc: object) -> None:
        self.torch.autocast_active = self.previous


class PrecisionTorch:
    def __init__(self) -> None:
        self.precision = "medium"
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=MatmulFlags()))
        self.autocast_active = True

    def get_float32_matmul_precision(self) -> str:
        return self.precision

    def set_float32_matmul_precision(self, value: str) -> None:
        self.precision = value

    def autocast(self, *, device_type: str, enabled: bool) -> FakeAutocast:
        assert device_type == "cuda"
        return FakeAutocast(self, enabled)


def test_precision_and_autocast_flags_restore_ambient_state_on_error() -> None:
    torch = PrecisionTorch()
    flags = torch.backends.cuda.matmul
    calls: list[bool] = []
    with (
        pytest.raises(RuntimeError, match="kernel failed"),
        local._precision_flags(torch),
        local._cuda_autocast_disabled(torch),
    ):
        calls.append(torch.autocast_active)
        assert torch.precision == "highest"
        assert flags.allow_tf32 is flags.allow_bf16_reduced_precision_reduction is False
        assert (
            flags.allow_fp16_reduced_precision_reduction is flags.allow_fp16_accumulation is False
        )
        raise RuntimeError("kernel failed")
    assert calls == [False]
    assert torch.autocast_active is True
    assert torch.precision == "medium"
    assert flags.allow_tf32 is flags.allow_bf16_reduced_precision_reduction is True
    assert flags.allow_fp16_reduced_precision_reduction is flags.allow_fp16_accumulation is True


class CompileCuda:
    def synchronize(self, index: int) -> None:
        assert index == 0


def test_compile_error_never_calls_eager_fallback() -> None:
    calls: list[str] = []

    def eager() -> None:
        calls.append("eager")

    def fail_compile(_kernel: Any, **kwargs: object) -> Any:
        assert callable(kwargs.pop("backend"))
        assert kwargs == {
            "fullgraph": True,
            "dynamic": False,
            "mode": "default",
        }
        raise RuntimeError("compile exploded")

    with pytest.raises(RuntimeError, match="compile exploded"):
        local._compile_candidate(SimpleNamespace(compile=fail_compile), eager)
    assert calls == []


def test_lazy_first_call_compile_error_is_compile_failed() -> None:
    def lazy_compile(_kernel: Any, **_kwargs: object) -> Any:
        def compiled() -> None:
            raise RuntimeError("lazy compile exploded")

        return compiled

    state: dict[str, bool] = {}
    compiled = local._compile_candidate(SimpleNamespace(compile=lazy_compile), lambda: None, state)
    with pytest.raises(local._ExecutionValidationError, match="lazy compile exploded") as caught:
        local._first_candidate_call(SimpleNamespace(cuda=CompileCuda()), compiled, (), 0, state)
    assert caught.value.kind == "compile_failed"
    assert state == {"invoked": False}


def test_noop_compile_original_and_never_invoked_backend_are_compile_failed() -> None:
    eager_calls: list[str] = []

    def eager() -> None:
        eager_calls.append("eager")

    with pytest.raises(RuntimeError, match="original eager callable"):
        local._compile_candidate(SimpleNamespace(compile=lambda kernel, **_kwargs: kernel), eager)
    assert eager_calls == []

    state: dict[str, bool] = {}
    compiled = local._compile_candidate(
        SimpleNamespace(compile=lambda kernel, **_kwargs: lambda: kernel()), eager, state
    )
    with pytest.raises(local._ExecutionValidationError, match="without invoking") as caught:
        local._first_candidate_call(SimpleNamespace(cuda=CompileCuda()), compiled, (), 0, state)
    assert caught.value.kind == "compile_failed"
    assert state == {"invoked": False}


def test_recording_backend_wraps_pinned_inductor_and_proves_invocation() -> None:
    def eager(value: int) -> int:
        return value + 1

    def lookup_backend(name: str) -> Any:
        assert name == "inductor"
        return lambda _graph, _inputs: eager

    def compile_with_backend(_kernel: Any, **kwargs: object) -> Any:
        backend = cast(Any, kwargs["backend"])

        def compiled(value: int) -> int:
            lowered = backend("graph", (value,))
            return cast(int, lowered(value))

        return compiled

    torch = SimpleNamespace(
        compile=compile_with_backend,
        cuda=CompileCuda(),
        _dynamo=SimpleNamespace(
            config=SimpleNamespace(disable=False, suppress_errors=False),
            backends=SimpleNamespace(registry=SimpleNamespace(lookup_backend=lookup_backend)),
        ),
    )
    state: dict[str, bool] = {}
    compiled = local._compile_candidate(torch, eager, state)
    assert compiled is not eager
    assert local._first_candidate_call(torch, compiled, (3,), 0, state) == 4
    assert state == {"invoked": True}


@pytest.mark.parametrize("field", ["disable", "suppress_errors"])
def test_compile_rejects_force_eager_config(field: str) -> None:
    config = SimpleNamespace(disable=False, suppress_errors=False)
    setattr(config, field, True)
    torch = SimpleNamespace(
        compile=lambda *_args, **_kwargs: pytest.fail("compile must not be called"),
        _dynamo=SimpleNamespace(config=config),
    )
    with pytest.raises(RuntimeError, match="eager"):
        local._compile_candidate(torch, lambda: None)


def test_compile_rejects_torchdynamo_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    torch = SimpleNamespace(
        compile=lambda *_args, **_kwargs: pytest.fail("compile must not be called")
    )
    with pytest.raises(RuntimeError, match="TORCHDYNAMO_DISABLE"):
        local._compile_candidate(torch, lambda: None)


class FakeEvent:
    def record(self) -> None:
        pass

    def elapsed_time(self, _other: FakeEvent) -> float:
        return 2.5


class TimingCuda:
    def __init__(self) -> None:
        self.syncs = 0
        self.Event = lambda *, enable_timing: FakeEvent()

    def synchronize(self, _index: int) -> None:
        self.syncs += 1


def test_timing_retains_exact_raw_samples_and_median() -> None:
    cuda = TimingCuda()
    calls = 0

    def kernel() -> None:
        nonlocal calls
        calls += 1

    observation = local._timing_observation(
        SimpleNamespace(cuda=cuda),
        kernel=kernel,
        arguments=(),
        device_index=0,
        correctness_key="key",
        warmups=10,
        repetitions=50,
    )
    assert observation.status == "passed"
    assert calls == 60
    assert observation.samples_ms == (2.5,) * 50
    assert observation.median_ms == 2.5
    assert cuda.syncs == 51


def test_timing_failure_after_several_samples_discards_positive_payload() -> None:
    class FailingTimingCuda(TimingCuda):
        def synchronize(self, _index: int) -> None:
            self.syncs += 1
            if self.syncs == 5:
                raise RuntimeError("timing synchronize failed")

    cuda = FailingTimingCuda()
    calls = 0

    def kernel() -> None:
        nonlocal calls
        calls += 1

    observation = local._timing_observation(
        SimpleNamespace(cuda=cuda),
        kernel=kernel,
        arguments=(),
        device_index=0,
        correctness_key="key",
        warmups=10,
        repetitions=50,
    )

    assert calls == 14
    assert observation.to_dict() == {
        "status": "failed",
        "correctness_key": "key",
        "failure_kind": "timing",
        "message": "RuntimeError: timing synchronize failed",
        "warmups": 10,
        "repetitions": 0,
        "samples_ms": [],
        "median_ms": None,
    }


def test_mutation_after_completed_timing_discards_positive_payload() -> None:
    completed = local._timing_observation(
        SimpleNamespace(cuda=TimingCuda()),
        kernel=lambda: None,
        arguments=(),
        device_index=0,
        correctness_key="key",
        warmups=10,
        repetitions=50,
    )
    observation = local._with_timing_failure(
        completed,
        failure_kind="mutation",
        message="an input tensor was mutated during timing",
    )

    assert completed.status == "passed"
    assert observation.to_dict() == {
        "status": "failed",
        "correctness_key": "key",
        "failure_kind": "mutation",
        "message": "an input tensor was mutated during timing",
        "warmups": 10,
        "repetitions": 0,
        "samples_ms": [],
        "median_ms": None,
    }


def torch_280_or_skip() -> Any:
    torch = pytest.importorskip("torch")
    if str(torch.__version__).partition("+")[0] != "2.8.0":
        pytest.skip("requires frozen torch 2.8.0 API")
    return torch


def test_tiny_cpu_reference_equations() -> None:
    torch = torch_280_or_skip()
    x = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    gate_weight = torch.tensor([[0.5, 1.0], [-1.0, 0.25]], dtype=torch.bfloat16)
    up_weight = torch.tensor([[2.0, -0.5], [0.75, 1.25]], dtype=torch.bfloat16)
    actual = local._gated_mlp_reference(torch, x, gate_weight, up_weight)
    gate, up = x.float() @ gate_weight.float().T, x.float() @ up_weight.float().T
    torch.testing.assert_close(
        actual, (gate * torch.sigmoid(gate) * up).to(torch.bfloat16), atol=0, rtol=0
    )
    residual = torch.tensor([[0.5, 1.0]], dtype=torch.bfloat16)
    gamma = torch.tensor([1.0, 1.25], dtype=torch.bfloat16)
    actual = local._residual_rmsnorm(torch, x, residual, gamma, 1e-5)
    z = x.float() + residual.float()
    expected = (
        z
        * torch.rsqrt(torch.mean(z * z, dim=-1, keepdim=True, dtype=torch.float32) + 1e-5)
        * gamma.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def validation_inputs(torch: Any) -> tuple[dict[str, Any], dict[str, str], Any]:
    inputs = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}
    return (
        inputs,
        {"x": local._tensor_hash(torch, inputs["x"])},
        torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
    )


def validate(
    torch: Any, actual: Any, expected: Any, inputs: dict[str, Any], hashes: dict[str, str]
) -> local.CorrectnessObservation:
    return local._validate_correctness(
        torch,
        actual=actual,
        expected=expected,
        inputs=inputs,
        before_hashes=hashes,
        expected_shape=(1, 2),
        atol=0.02,
        rtol=0.02,
        correctness_key="key",
    )


def test_correctness_rejects_alias_mutation_nonfinite_and_tolerance() -> None:
    torch = torch_280_or_skip()
    inputs, hashes, expected = validation_inputs(torch)
    assert validate(torch, inputs["x"], expected, inputs, hashes).failure_kind == "alias"
    inputs, hashes, expected = validation_inputs(torch)
    inputs["x"].add_(1)
    assert validate(torch, expected.clone(), expected, inputs, hashes).failure_kind == "mutation"
    inputs, hashes, expected = validation_inputs(torch)
    actual = torch.tensor([[float("inf"), 4.0]], dtype=torch.bfloat16)
    assert validate(torch, actual, expected, inputs, hashes).failure_kind == "nonfinite"
    inputs, hashes, expected = validation_inputs(torch)
    assert (
        validate(torch, torch.zeros_like(expected), expected, inputs, hashes).failure_kind
        == "tolerance"
    )
