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
TRITON_RMS = ROOT / "benchmarks/suites/residual-rmsnorm-triton-v1.json"


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        cc: tuple[int, int] = (8, 0),
        bf16: bool = True,
        device_error: bool = False,
        bf16_error: bool = False,
    ) -> None:
        self.available = available
        self.cc = cc
        self.bf16 = bf16
        self.device_error = device_error
        self.bf16_error = bf16_error
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
        if self.bf16_error:
            raise RuntimeError("BF16 probe exploded")
        return self.bf16

    def synchronize(self, _index: int) -> None:
        self.syncs += 1


class FakeTorchProbe:
    bfloat16 = "bfloat16"

    def __init__(
        self,
        *,
        version: object = "2.8.0+cu128",
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
    assert local.CapabilityProbe.from_dict(probe.to_dict()) == probe


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
        (FakeTorchProbe(version=object()), "torch_version_mismatch"),
        (FakeTorchProbe(version="2.7.1"), "torch_version_mismatch"),
        (FakeTorchProbe(cuda=FakeCuda(available=False)), "cuda_unavailable"),
        (FakeTorchProbe(hip="6.3"), "rocm_unsupported"),
        (FakeTorchProbe(cuda=FakeCuda(cc=(7, 5))), "compute_capability_too_low"),
        (FakeTorchProbe(cuda=FakeCuda(bf16=False)), "bf16_unsupported"),
        (FakeTorchProbe(backends=()), "inductor_unavailable"),
        (FakeTorchProbe(allocation_error=True), "allocation_failed"),
        (FakeTorchProbe(cuda=FakeCuda(device_error=True)), "device_probe_failed"),
        (FakeTorchProbe(cuda=FakeCuda(bf16_error=True)), "device_probe_failed"),
    ],
)
def test_every_capability_rejection_reason(torch: Any, reason: str) -> None:
    probe = local._probe_torch(torch, load_suite(MLP))
    assert probe.available is False
    assert probe.reason == reason
    assert probe.allocation_succeeded is False
    assert local.CapabilityProbe.from_dict(probe.to_dict()) == probe


def test_complete_fake_capability_is_available() -> None:
    probe = local._probe_torch(FakeTorchProbe(), load_suite(RMS))
    assert probe.available is True
    assert probe.reasons == ()
    assert probe.compute_capability == (8, 0)
    assert probe.native_bf16 is probe.inductor_available is probe.allocation_succeeded is True
    assert local.CapabilityProbe.from_dict(probe.to_dict()) == probe


def suite_dict(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def test_frozen_hash_registry_matches_current_committed_suite_bytes() -> None:
    assert hashlib.sha256(MLP.read_bytes()).hexdigest() == local.GATED_MLP_SUITE_SHA256
    assert hashlib.sha256(RMS.read_bytes()).hexdigest() == local.RMSNORM_SUITE_SHA256
    assert (
        hashlib.sha256(TRITON_RMS.read_bytes()).hexdigest()
        == local.NATIVE_RMSNORM_SUITE_SHA256
    )


def test_execute_local_suite_dispatches_legacy_without_native_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[Path] = []

    def legacy(path: str | Path) -> object:
        calls.append(Path(path))
        return sentinel

    def unexpected_import(name: str) -> object:
        raise AssertionError(f"legacy dispatch imported {name}")

    monkeypatch.setattr(local, "run_local_suite", legacy)
    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    assert local.execute_local_suite(MLP) is sentinel
    assert local.execute_local_suite(RMS) is sentinel
    assert calls == [MLP, RMS]


def test_execute_and_parse_native_suite_import_lazily_by_exact_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "renamed-native-suite.json"
    copied.write_bytes(TRITON_RMS.read_bytes())
    executed = object()
    parsed = object()
    imports: list[str] = []
    parser_calls: list[tuple[object, dict[str, object]]] = []

    class FakeNativeResult:
        @classmethod
        def from_dict(cls, value: object, **kwargs: object) -> object:
            parser_calls.append((value, kwargs))
            return parsed

    fake_module = SimpleNamespace(
        NativeFusionExecutionResult=FakeNativeResult,
        run_native_fusion_suite=lambda path: executed if Path(path) == copied else None,
    )

    def load(name: str) -> object:
        imports.append(name)
        assert name == "heliostune.native_fusion_executor"
        return fake_module

    monkeypatch.setattr(importlib, "import_module", load)
    assert imports == []
    assert local.execute_local_suite(copied) is executed

    payload = copied.read_bytes()
    serialized = {"schema": "heliostune.local_executor/2"}
    assert (
        local.parse_local_execution_result(
            serialized,
            verified_suite_path="logical/native-suite.json",
            verified_suite_sha256=local.NATIVE_RMSNORM_SUITE_SHA256,
            verified_suite_bytes=payload,
        )
        is parsed
    )
    assert imports == [
        "heliostune.native_fusion_executor",
        "heliostune.native_fusion_executor",
    ]
    assert parser_calls == [
        (
            serialized,
            {
                "verified_suite_path": "logical/native-suite.json",
                "verified_suite_sha256": local.NATIVE_RMSNORM_SUITE_SHA256,
                "verified_suite_bytes": payload,
            },
        )
    ]


def test_dispatchers_fail_closed_for_unknown_digest_without_native_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unknown = tmp_path / "unknown.json"
    unknown_suite = suite_dict(MLP)
    unknown_suite["timing_policies"][0]["warmups"] = 11
    unknown.write_text(json.dumps(unknown_suite))
    payload = unknown.read_bytes()
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"unexpected import {name}")),
    )
    with pytest.raises(SchemaError, match="unsupported suite SHA-256"):
        local.execute_local_suite(unknown)
    with pytest.raises(SchemaError, match="unsupported suite SHA-256"):
        local.parse_local_execution_result(
            {},
            verified_suite_path="unknown.json",
            verified_suite_sha256=hashlib.sha256(payload).hexdigest(),
            verified_suite_bytes=payload,
        )


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
    for path in (RMS, TRITON_RMS):
        rms = load_suite(path)
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


class EquationTensor:
    def __init__(self, torch: EquationTorch, expression: str) -> None:
        self.torch = torch
        self.expression = expression

    def float(self) -> EquationTensor:
        expression = f"{self.expression}.float()"
        self.torch.trace.append(expression)
        return EquationTensor(self.torch, expression)

    @property
    def T(self) -> EquationTensor:
        expression = f"{self.expression}.T"
        self.torch.trace.append(expression)
        return EquationTensor(self.torch, expression)

    def __mul__(self, other: EquationTensor) -> EquationTensor:
        expression = f"({self.expression} * {other.expression})"
        self.torch.trace.append(expression)
        return EquationTensor(self.torch, expression)

    def to(self, *, dtype: object) -> EquationTensor:
        expression = f"{self.expression}.to(dtype={dtype})"
        self.torch.trace.append(expression)
        return EquationTensor(self.torch, expression)


class EquationTorch:
    bfloat16 = "bfloat16"

    def __init__(self) -> None:
        self.trace: list[str] = []
        self.cuda = CompileCuda()
        self.nn = SimpleNamespace(
            functional=SimpleNamespace(silu=self._silu),
        )
        self._dynamo = SimpleNamespace(
            config=SimpleNamespace(disable=False, suppress_errors=False),
            backends=SimpleNamespace(registry=SimpleNamespace(lookup_backend=self._lookup_backend)),
        )

    def tensor(self, name: str) -> EquationTensor:
        return EquationTensor(self, name)

    def mm(self, left: EquationTensor, right: EquationTensor, **kwargs: object) -> EquationTensor:
        if kwargs:
            raise TypeError("meta_mm() takes 2 positional arguments but 3 were given")
        expression = f"torch.mm({left.expression}, {right.expression})"
        self.trace.append(expression)
        return EquationTensor(self, expression)

    def _silu(self, value: EquationTensor, *, inplace: bool) -> EquationTensor:
        expression = f"silu({value.expression}, inplace={inplace})"
        self.trace.append(expression)
        return EquationTensor(self, expression)

    def _lookup_backend(self, name: str) -> Any:
        assert name == "inductor"
        return lambda graph, _inputs: graph

    def compile(self, kernel: Any, **kwargs: object) -> Any:
        backend = cast(Any, kwargs.pop("backend"))
        assert kwargs == {"fullgraph": True, "dynamic": False, "mode": "default"}

        def compiled(*arguments: EquationTensor) -> EquationTensor:
            lowered = backend(kernel, arguments)
            return cast(EquationTensor, lowered(*arguments))

        return compiled


def _equation_arguments(torch: EquationTorch) -> tuple[EquationTensor, ...]:
    return (
        torch.tensor("x"),
        torch.tensor("gate_weight"),
        torch.tensor("up_weight"),
    )


def test_gated_candidate_is_exact_reference_arithmetic_compiled_fullgraph() -> None:
    reference_torch = EquationTorch()
    reference = local._gated_mlp_reference(reference_torch, *_equation_arguments(reference_torch))
    candidate_torch = EquationTorch()
    eager_candidate = local._gated_mlp_candidate
    candidate = eager_candidate(candidate_torch, *_equation_arguments(candidate_torch))

    expected_trace = [
        "x.float()",
        "gate_weight.float()",
        "gate_weight.float().T",
        "torch.mm(x.float(), gate_weight.float().T)",
        "x.float()",
        "up_weight.float()",
        "up_weight.float().T",
        "torch.mm(x.float(), up_weight.float().T)",
        "silu(torch.mm(x.float(), gate_weight.float().T), inplace=False)",
        "(silu(torch.mm(x.float(), gate_weight.float().T), inplace=False) * torch.mm(x.float(), up_weight.float().T))",
        "(silu(torch.mm(x.float(), gate_weight.float().T), inplace=False) * torch.mm(x.float(), up_weight.float().T)).to(dtype=bfloat16)",
    ]
    assert candidate_torch.trace == reference_torch.trace == expected_trace
    assert candidate.expression == reference.expression

    compiled_torch = EquationTorch()

    def kernel(*arguments: EquationTensor) -> EquationTensor:
        return cast(EquationTensor, local._gated_mlp_candidate(compiled_torch, *arguments))

    backend_state: dict[str, bool] = {}
    compiled = local._compile_candidate(compiled_torch, kernel, backend_state)
    actual = local._first_candidate_call(
        compiled_torch, compiled, _equation_arguments(compiled_torch), 0, backend_state
    )
    assert actual.expression == reference.expression
    assert backend_state == {"invoked": True}


def test_safe_error_canonicalizes_and_bounds_multimegabyte_utf8() -> None:
    raw = "  observed\nFakeTensor\tfailure " + ("界" * 1_000_000)
    canonical = "RuntimeError: observed FakeTensor failure " + ("界" * 1_000_000)
    safe = local._safe_error(RuntimeError(raw))

    assert len(safe.encode("utf-8")) <= 4096
    assert safe.startswith("RuntimeError: observed FakeTensor failure 界")
    assert safe.endswith(
        f" [truncated sha256={hashlib.sha256(canonical.encode('utf-8')).hexdigest()}]"
    )
    assert "\n" not in safe and "\t" not in safe


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


def _parse_result(
    value: object, *, suite_path: Path = MLP, logical_path: str = "logical/suite.json"
) -> local.LocalExecutionResult:
    payload = suite_path.read_bytes()
    return local.LocalExecutionResult.from_dict(
        value,
        verified_suite_path=logical_path,
        verified_suite_sha256=hashlib.sha256(payload).hexdigest(),
        verified_suite_bytes=payload,
    )


@pytest.mark.parametrize("suite_path", [MLP, RMS])
def test_execution_result_from_dict_roundtrips_and_injects_verified_suite(
    suite_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> Any:
        raise ModuleNotFoundError("no torch")

    monkeypatch.setattr(importlib, "import_module", missing)
    result = local.run_local_suite(suite_path)
    serialized = result.to_dict()
    serialized["verified_suite_path"] = "/tmp/private-remote-suite.json"
    payload = suite_path.read_bytes()
    parsed = cast(
        local.LocalExecutionResult,
        local.parse_local_execution_result(
            serialized,
            verified_suite_path="logical/suite.json",
            verified_suite_sha256=hashlib.sha256(payload).hexdigest(),
            verified_suite_bytes=payload,
        ),
    )

    assert parsed.to_dict() == {
        **serialized,
        "verified_suite_path": "logical/suite.json",
    }
    assert parsed.verified_suite_bytes == suite_path.read_bytes()
    assert isinstance(parsed.capability.reasons, tuple)
    assert isinstance(parsed.materialization, tuple)
    assert isinstance(parsed.observations, tuple)
    assert isinstance(parsed.attempts, tuple)
    with pytest.raises(SchemaError, match="unknown fields"):
        _parse_result(result.to_dict(include_suite_bytes=True))


def _valid_nested_records() -> dict[str, tuple[Any, dict[str, Any]]]:
    digest = "a" * 64
    capability = {
        "available": False,
        "reasons": ["torch_missing"],
        "torch_version": None,
        "cuda_version": None,
        "rocm_version": None,
        "device_index": None,
        "device_name": None,
        "compute_capability": None,
        "native_bf16": None,
        "inductor_available": None,
        "allocation_succeeded": False,
        "detail": "missing",
    }
    descriptor = {
        "tensor_id": "input",
        "role": "input",
        "shape": [2, 4],
        "draw": "normal_0_1_fp32_cpu",
        "normal_scale": 1.0,
        "normal_offset": 0.0,
        "cpu_dtype": "float32",
        "storage_dtype": "bfloat16",
        "device": "cuda:0",
        "contiguous": True,
        "alignment_bytes": 16,
        "alignment_satisfied": True,
        "storage_sha256": digest,
    }
    materialization = {
        "suite_sha256": digest,
        "case_id": "case",
        "arm_id": "arm",
        "input_seed": 1,
        "tensor_order": ["input"],
        "tensors": [descriptor],
    }
    correctness = {
        "status": "passed",
        "correctness_key": digest,
        "failure_kind": None,
        "message": None,
        "output": {
            "shape": [2, 4],
            "device": "cuda:0",
            "dtype": "torch.bfloat16",
            "layout": "torch.strided",
            "contiguous": True,
        },
        "input_storage_unchanged": True,
        "output_disjoint": True,
        "finite": True,
        "close": True,
        "max_abs_error": 0.0,
    }
    timing = {
        "status": "passed",
        "correctness_key": digest,
        "failure_kind": None,
        "message": None,
        "warmups": 2,
        "repetitions": 3,
        "samples_ms": [1.0, 2.0, 3.0],
        "median_ms": 2.0,
    }
    cell = {
        "cell_id": "cell",
        "case_id": "case",
        "arm_id": "arm",
        "stage": "correctness",
        "status": "passed",
        "correctness": correctness,
        "timing": None,
    }
    return {
        "capability": (local.CapabilityProbe.from_dict, capability),
        "materialization": (local.TensorMaterialization.from_dict, materialization),
        "correctness": (local.CorrectnessObservation.from_dict, correctness),
        "timing": (local.TimingObservation.from_dict, timing),
        "cell": (local.CellObservation.from_dict, cell),
    }


@pytest.mark.parametrize(
    ("record", "field", "bad"),
    [
        ("capability", "available", 1),
        ("capability", "reasons", "torch_missing"),
        ("capability", "torch_version", 1),
        ("capability", "cuda_version", 1),
        ("capability", "rocm_version", 1),
        ("capability", "device_index", True),
        ("capability", "device_name", 1),
        ("capability", "compute_capability", [True, 0]),
        ("capability", "native_bf16", 1),
        ("capability", "inductor_available", 1),
        ("capability", "allocation_succeeded", 1),
        ("capability", "detail", 1),
        ("materialization", "suite_sha256", "A" * 64),
        ("materialization", "case_id", 1),
        ("materialization", "arm_id", 1),
        ("materialization", "input_seed", True),
        ("materialization", "tensor_order", ("input",)),
        ("materialization", "tensors", ()),
        ("correctness", "status", "unknown"),
        ("correctness", "correctness_key", "A" * 64),
        ("correctness", "failure_kind", 1),
        ("correctness", "message", 1),
        ("correctness", "output", []),
        ("correctness", "input_storage_unchanged", 1),
        ("correctness", "output_disjoint", 1),
        ("correctness", "finite", 1),
        ("correctness", "close", 1),
        ("correctness", "max_abs_error", math.inf),
        ("timing", "status", "unknown"),
        ("timing", "correctness_key", "A" * 64),
        ("timing", "failure_kind", 1),
        ("timing", "message", 1),
        ("timing", "warmups", True),
        ("timing", "repetitions", True),
        ("timing", "samples_ms", (1.0,)),
        ("timing", "median_ms", math.nan),
        ("cell", "cell_id", 1),
        ("cell", "case_id", 1),
        ("cell", "arm_id", 1),
        ("cell", "stage", "compile"),
        ("cell", "status", "unknown"),
        ("cell", "correctness", []),
        ("cell", "timing", []),
    ],
)
def test_nested_from_dict_rejects_every_malformed_field(
    record: str, field: str, bad: object
) -> None:
    parser, valid = _valid_nested_records()[record]
    changed = dict(valid)
    changed[field] = bad
    with pytest.raises(SchemaError):
        parser(changed)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("tensor_id", 1),
        ("role", 1),
        ("shape", [True]),
        ("draw", 1),
        ("normal_scale", math.inf),
        ("normal_offset", math.nan),
        ("cpu_dtype", 1),
        ("storage_dtype", 1),
        ("device", 1),
        ("contiguous", 1),
        ("alignment_bytes", True),
        ("alignment_satisfied", 1),
        ("storage_sha256", "A" * 64),
    ],
)
def test_materialization_descriptor_parser_rejects_every_malformed_field(
    field: str, bad: object
) -> None:
    parser, valid = _valid_nested_records()["materialization"]
    changed = dict(valid)
    descriptor = dict(cast(list[dict[str, Any]], changed["tensors"])[0])
    descriptor[field] = bad
    changed["tensors"] = [descriptor]
    with pytest.raises(SchemaError):
        parser(changed)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("shape", [True]),
        ("device", 1),
        ("dtype", 1),
        ("layout", 1),
        ("contiguous", 1),
    ],
)
def test_output_descriptor_parser_rejects_every_malformed_field(field: str, bad: object) -> None:
    parser, valid = _valid_nested_records()["correctness"]
    changed = dict(valid)
    output = dict(cast(dict[str, Any], changed["output"]))
    output[field] = bad
    changed["output"] = output
    with pytest.raises(SchemaError):
        parser(changed)


def _available_failed_payload(suite_path: Path = MLP) -> dict[str, Any]:
    suite = load_suite(suite_path)
    digest = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    capability = {
        "available": True,
        "reasons": [],
        "torch_version": "2.8.0+cu128",
        "cuda_version": "12.8",
        "rocm_version": None,
        "device_index": 0,
        "device_name": "NVIDIA H100",
        "compute_capability": [9, 0],
        "native_bf16": True,
        "inductor_available": True,
        "allocation_succeeded": True,
        "detail": None,
    }
    attempts: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    case_by_id = {case.id: case for case in suite.cases}
    for cell in suite.expected_cells:
        key = local._correctness_gate_key(digest, case_by_id[cell.case_id], cell)
        reason = "runtime" if cell.stage == "correctness" else "correctness_gate"
        attempts.extend(
            (
                {
                    "attempt_id": len(attempts) + 1,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": "running",
                    "from_state": "pending",
                    "to_state": "running",
                    "reason": None,
                },
                {
                    "attempt_id": len(attempts) + 2,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": "failure",
                    "from_state": "running",
                    "to_state": "failed",
                    "reason": reason,
                },
            )
        )
        correctness = (
            {
                "status": "failed",
                "correctness_key": key,
                "failure_kind": reason,
                "message": "failed",
                "output": None,
                "input_storage_unchanged": False,
                "output_disjoint": False,
                "finite": False,
                "close": False,
                "max_abs_error": None,
            }
            if cell.stage == "correctness"
            else None
        )
        timing = (
            {
                "status": "failed",
                "correctness_key": key,
                "failure_kind": reason,
                "message": "failed",
                "warmups": 0,
                "repetitions": 0,
                "samples_ms": [],
                "median_ms": None,
            }
            if cell.stage == "timing"
            else None
        )
        observations.append(
            {
                "cell_id": cell.id,
                "case_id": cell.case_id,
                "arm_id": cell.arm_id,
                "stage": cell.stage,
                "status": "failed",
                "correctness": correctness,
                "timing": timing,
            }
        )
    return {
        "verified_suite_path": "/tmp/private-suite.json",
        "verified_suite_sha256": digest,
        "suite_id": suite.suite_id,
        "capability": capability,
        "materialization": [],
        "observations": observations,
        "attempts": attempts,
        "environment": {
            "schema": "heliostune.local-environment/1",
            "python": "3.11.0",
            "implementation": "CPython",
            "platform": "Linux",
            "torch_version": capability["torch_version"],
            "cuda_version": capability["cuda_version"],
            "rocm_version": None,
            "device_index": 0,
            "device_name": capability["device_name"],
            "compute_capability": [9, 0],
            "precision_policy": dict(local._PRECISION_POLICY),
            "autocast_policy": dict(local._AUTOCAST_POLICY),
            "backend_invoked": False,
            "fusion_claim": False,
        },
        "compile_outcomes": {},
        "summary": {
            "expected_cell_ids": [cell.id for cell in suite.expected_cells],
            "terminal_cell_ids": [cell.id for cell in suite.expected_cells],
            "passed": 0,
            "failed": len(suite.expected_cells),
            "blocked": 0,
            "all_cells_terminal": True,
            "outcome": "failed",
            "fusion_claim": False,
            "candidate_reference_arithmetic": "candidate_reference_identical",
            "candidate_distinction": "fullgraph_inductor_compilation_only",
        },
        "outcome": "failed",
    }


def _unavailable_capability_records() -> dict[str, dict[str, Any]]:
    empty: dict[str, Any] = {
        "available": False,
        "reasons": [],
        "torch_version": None,
        "cuda_version": None,
        "rocm_version": None,
        "device_index": None,
        "device_name": None,
        "compute_capability": None,
        "native_bf16": None,
        "inductor_available": None,
        "allocation_succeeded": False,
        "detail": None,
    }

    def record(reason: str, **evidence: object) -> dict[str, Any]:
        value = dict(empty)
        value["reasons"] = [reason]
        value.update(evidence)
        return value

    return {
        "torch_missing": record("torch_missing", detail="ModuleNotFoundError: no torch"),
        "torch_version_mismatch": record("torch_version_mismatch", torch_version="2.7.1"),
        "cuda_unavailable": record(
            "cuda_unavailable",
            torch_version="2.8.0+cu128",
            cuda_version="12.8",
        ),
        "rocm_unsupported": record(
            "rocm_unsupported",
            torch_version="2.8.0+rocm6.3",
            rocm_version="6.3",
        ),
        "compute_capability_too_low": record(
            "compute_capability_too_low",
            torch_version="2.8.0+cu128",
            cuda_version="12.8",
            device_index=0,
            device_name="NVIDIA T4",
            compute_capability=[7, 5],
        ),
        "bf16_unsupported": record(
            "bf16_unsupported",
            torch_version="2.8.0+cu128",
            cuda_version="12.8",
            device_index=0,
            device_name="NVIDIA RTX 3090",
            compute_capability=[8, 6],
            native_bf16=False,
        ),
        "inductor_unavailable": record(
            "inductor_unavailable",
            torch_version="2.8.0+cu128",
            cuda_version="12.8",
            device_index=0,
            device_name="NVIDIA H100",
            compute_capability=[9, 0],
            native_bf16=True,
            inductor_available=False,
        ),
        "allocation_failed": record(
            "allocation_failed",
            torch_version="2.8.0+cu128",
            cuda_version="12.8",
            device_index=0,
            device_name="NVIDIA H100",
            compute_capability=[9, 0],
            native_bf16=True,
            inductor_available=True,
            detail="RuntimeError: allocation exploded",
        ),
        "device_probe_failed": record(
            "device_probe_failed",
            torch_version="2.8.0+cu128",
            cuda_version="12.8",
            device_index=0,
            device_name="NVIDIA H100",
            compute_capability=[9, 0],
            detail="RuntimeError: BF16 probe exploded",
        ),
    }


def _unavailable_payload(capability: dict[str, Any]) -> dict[str, Any]:
    value = _available_failed_payload()
    value["capability"] = capability
    value["materialization"] = []
    value["observations"] = []
    value["attempts"] = []
    value["compile_outcomes"] = {}
    environment = cast(dict[str, Any], value["environment"])
    for field in (
        "torch_version",
        "cuda_version",
        "rocm_version",
        "device_index",
        "device_name",
        "compute_capability",
    ):
        environment[field] = capability[field]
    environment["backend_invoked"] = None
    expected_ids = [cell.id for cell in load_suite(MLP).expected_cells]
    value["summary"] = {
        "expected_cell_ids": expected_ids,
        "terminal_cell_ids": [],
        "passed": 0,
        "failed": 0,
        "blocked": 0,
        "all_cells_terminal": False,
        "outcome": "aborted",
        "fusion_claim": False,
        "candidate_reference_arithmetic": "candidate_reference_identical",
        "candidate_distinction": "fullgraph_inductor_compilation_only",
        "capability_reasons": list(capability["reasons"]),
    }
    value["outcome"] = "aborted"
    return value


_IMPOSSIBLE_CAPABILITY_EVIDENCE: dict[str, tuple[tuple[str, object], ...]] = {
    "torch_missing": (
        ("torch_version", "2.8.0"),
        ("cuda_version", "12.8"),
        ("rocm_version", "6.3"),
        ("device_index", 0),
        ("device_name", "NVIDIA H100"),
        ("compute_capability", [9, 0]),
        ("native_bf16", True),
        ("inductor_available", True),
        ("allocation_succeeded", True),
        ("detail", None),
        ("detail", ""),
    ),
    "torch_version_mismatch": (
        ("torch_version", "2.8.0+cu128"),
        ("cuda_version", "12.8"),
        ("rocm_version", "6.3"),
        ("device_index", 0),
        ("device_name", "NVIDIA H100"),
        ("compute_capability", [9, 0]),
        ("native_bf16", False),
        ("inductor_available", False),
        ("allocation_succeeded", True),
        ("detail", "unexpected"),
    ),
    "cuda_unavailable": (
        ("torch_version", None),
        ("rocm_version", "6.3"),
        ("device_index", 0),
        ("device_name", "NVIDIA H100"),
        ("compute_capability", [9, 0]),
        ("native_bf16", False),
        ("inductor_available", False),
        ("allocation_succeeded", True),
        ("detail", "unexpected"),
    ),
    "rocm_unsupported": (
        ("torch_version", None),
        ("rocm_version", None),
        ("device_index", 0),
        ("device_name", "AMD Instinct"),
        ("compute_capability", [9, 0]),
        ("native_bf16", True),
        ("inductor_available", False),
        ("allocation_succeeded", True),
        ("detail", "unexpected"),
    ),
    "compute_capability_too_low": (
        ("torch_version", None),
        ("cuda_version", None),
        ("rocm_version", "6.3"),
        ("device_index", None),
        ("device_name", None),
        ("compute_capability", None),
        ("compute_capability", [8, 0]),
        ("native_bf16", False),
        ("inductor_available", False),
        ("allocation_succeeded", True),
        ("detail", "unexpected"),
    ),
    "bf16_unsupported": (
        ("torch_version", None),
        ("cuda_version", None),
        ("rocm_version", "6.3"),
        ("device_index", None),
        ("device_name", None),
        ("compute_capability", None),
        ("compute_capability", [7, 5]),
        ("native_bf16", None),
        ("native_bf16", True),
        ("inductor_available", False),
        ("allocation_succeeded", True),
        ("detail", "unexpected"),
    ),
    "inductor_unavailable": (
        ("torch_version", None),
        ("cuda_version", None),
        ("rocm_version", "6.3"),
        ("device_index", None),
        ("device_name", None),
        ("compute_capability", None),
        ("compute_capability", [7, 5]),
        ("native_bf16", None),
        ("native_bf16", False),
        ("inductor_available", None),
        ("inductor_available", True),
        ("allocation_succeeded", True),
        ("detail", "unexpected"),
    ),
    "allocation_failed": (
        ("torch_version", None),
        ("cuda_version", None),
        ("rocm_version", "6.3"),
        ("device_index", None),
        ("device_name", None),
        ("compute_capability", None),
        ("compute_capability", [7, 5]),
        ("native_bf16", None),
        ("native_bf16", False),
        ("inductor_available", None),
        ("inductor_available", False),
        ("allocation_succeeded", True),
        ("detail", None),
        ("detail", ""),
    ),
    "device_probe_failed": (
        ("torch_version", None),
        ("rocm_version", "6.3"),
        ("device_index", None),
        ("device_name", None),
        ("compute_capability", None),
        ("compute_capability", [7, 5]),
        ("native_bf16", False),
        ("inductor_available", False),
        ("allocation_succeeded", True),
        ("detail", None),
        ("detail", ""),
    ),
}


@pytest.mark.parametrize("reason", tuple(_IMPOSSIBLE_CAPABILITY_EVIDENCE))
def test_unavailable_reason_accepts_only_producer_stage_evidence(reason: str) -> None:
    capability = _unavailable_capability_records()[reason]
    parsed = local.CapabilityProbe.from_dict(capability)
    assert parsed.reason == reason

    payload = _unavailable_payload(capability)
    assert _parse_result(payload).capability == parsed

    for field, bad in _IMPOSSIBLE_CAPABILITY_EVIDENCE[reason]:
        mutated = dict(capability)
        mutated[field] = bad
        with pytest.raises(SchemaError):
            local.CapabilityProbe.from_dict(mutated)
        rejected_payload = _unavailable_payload(mutated)
        with pytest.raises(SchemaError):
            _parse_result(rejected_payload)

    second_reason = "torch_missing" if reason != "torch_missing" else "cuda_unavailable"
    multiple = dict(capability)
    multiple["reasons"] = [reason, second_reason]
    with pytest.raises(SchemaError, match="one rejection reason"):
        local.CapabilityProbe.from_dict(multiple)


def test_version_and_device_probe_failures_accept_each_producer_variant() -> None:
    records = _unavailable_capability_records()
    missing_version = dict(records["torch_version_mismatch"])
    missing_version["torch_version"] = None
    no_device = dict(records["device_probe_failed"])
    no_device.update(
        cuda_version=None,
        device_index=None,
        device_name=None,
        compute_capability=None,
        detail="RuntimeError: CUDA availability probe exploded",
    )
    assert local.CapabilityProbe.from_dict(missing_version).reason == "torch_version_mismatch"
    assert local.CapabilityProbe.from_dict(no_device).reason == "device_probe_failed"


@pytest.mark.parametrize(
    ("field", "bad"),
    (
        ("reasons", ["allocation_failed"]),
        ("torch_version", "2.7.1"),
        ("cuda_version", None),
        ("rocm_version", "6.3"),
        ("device_index", None),
        ("device_name", None),
        ("compute_capability", None),
        ("compute_capability", [7, 5]),
        ("native_bf16", False),
        ("inductor_available", False),
        ("allocation_succeeded", False),
        ("detail", "unexpected"),
    ),
)
def test_available_capability_requires_exact_passing_evidence(field: str, bad: object) -> None:
    capability = cast(dict[str, Any], _available_failed_payload()["capability"])
    mutated = dict(capability)
    mutated[field] = bad
    with pytest.raises(SchemaError, match="exact passing evidence"):
        local.CapabilityProbe.from_dict(mutated)


def test_available_failed_result_from_dict_enforces_full_execution_links() -> None:
    value = _available_failed_payload()
    parsed = _parse_result(value)
    assert parsed.to_dict() == {
        **value,
        "verified_suite_path": "logical/suite.json",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("attempt_transition", "from_state"),
        ("observation_order", "expected-cell order"),
        ("timing_key", "correctness key"),
        ("summary_count", "summary"),
        ("summary_fusion", "summary"),
        ("outcome", "outcome"),
        ("environment_backend", "backend evidence"),
        ("capability_reason", "available local capability"),
    ],
)
def test_result_parser_rejects_transition_order_key_summary_and_evidence_mismatches(
    mutation: str, message: str
) -> None:
    value = _available_failed_payload()
    if mutation == "attempt_transition":
        value["attempts"][1]["from_state"] = "pending"
    elif mutation == "observation_order":
        value["observations"][0], value["observations"][1] = (
            value["observations"][1],
            value["observations"][0],
        )
        value["summary"]["terminal_cell_ids"][0], value["summary"]["terminal_cell_ids"][1] = (
            value["summary"]["terminal_cell_ids"][1],
            value["summary"]["terminal_cell_ids"][0],
        )
    elif mutation == "timing_key":
        timing = next(
            item["timing"] for item in value["observations"] if item["timing"] is not None
        )
        timing["correctness_key"] = "b" * 64
    elif mutation == "summary_count":
        value["summary"]["failed"] -= 1
    elif mutation == "summary_fusion":
        value["summary"]["fusion_claim"] = True
    elif mutation == "outcome":
        value["outcome"] = "completed"
        value["summary"]["outcome"] = "completed"
    elif mutation == "environment_backend":
        value["environment"]["backend_invoked"] = True
    else:
        value["capability"]["reasons"] = ["torch_missing"]
    with pytest.raises(SchemaError, match=message):
        _parse_result(value)


def test_capability_unavailable_result_rejects_execution_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> Any:
        raise ModuleNotFoundError("no torch")

    monkeypatch.setattr(importlib, "import_module", missing)
    value = local.run_local_suite(MLP).to_dict()
    value["attempts"] = [
        {
            "attempt_id": 1,
            "cell_id": load_suite(MLP).expected_cells[0].id,
            "stage": "correctness",
            "status": "running",
            "from_state": "pending",
            "to_state": "running",
            "reason": None,
        }
    ]
    with pytest.raises(SchemaError, match="execution evidence"):
        _parse_result(value)


def _materialization_for_first_correctness() -> dict[str, Any]:
    suite = load_suite(MLP)
    case = suite.cases[0]
    cell = next(item for item in suite.expected_cells if item.stage == "correctness")
    specs = {item.id: item for item in suite.tensors}
    schedule = local._resolve_draw_schedule(suite, case)
    return {
        "suite_sha256": hashlib.sha256(MLP.read_bytes()).hexdigest(),
        "case_id": cell.case_id,
        "arm_id": cell.arm_id,
        "input_seed": case.input_seed,
        "tensor_order": [item.tensor_id for item in schedule],
        "tensors": [
            {
                "tensor_id": item.tensor_id,
                "role": item.role,
                "shape": list(item.shape),
                "draw": "normal_0_1_fp32_cpu",
                "normal_scale": item.normal_scale,
                "normal_offset": item.normal_offset,
                "cpu_dtype": "float32",
                "storage_dtype": "bfloat16",
                "device": "cuda:0",
                "contiguous": True,
                "alignment_bytes": specs[item.tensor_id].alignment,
                "alignment_satisfied": True,
                "storage_sha256": "a" * 64,
            }
            for item in schedule
        ],
    }


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("suite_sha256", "b" * 64, "verified suite"),
        ("input_seed", 2, "input seed"),
        ("tensor_order", ["wrong"], "tensor order"),
    ],
)
def test_result_parser_enforces_materialization_linkage(
    field: str, bad: object, message: str
) -> None:
    value = _available_failed_payload()
    materialization = _materialization_for_first_correctness()
    materialization[field] = bad
    value["materialization"] = [materialization]
    with pytest.raises(SchemaError, match=message):
        _parse_result(value)


def test_result_parser_requires_compile_and_materialization_for_passing_candidate() -> None:
    value = _available_failed_payload()
    suite = load_suite(MLP)
    index = next(
        index
        for index, cell in enumerate(suite.expected_cells)
        if cell.stage == "correctness"
        and next(arm for arm in suite.arms if arm.id == cell.arm_id).role == "candidate"
    )
    observation = value["observations"][index]
    observation["status"] = "passed"
    correctness = observation["correctness"]
    correctness.update(
        status="passed",
        failure_kind=None,
        message=None,
        output={
            "shape": [
                suite.cases[0].shape_dict[name]
                for name in next(item for item in suite.tensors if item.role == "output").shape
            ],
            "device": "cuda:0",
            "dtype": "torch.bfloat16",
            "layout": "torch.strided",
            "contiguous": True,
        },
        input_storage_unchanged=True,
        output_disjoint=True,
        finite=True,
        close=True,
        max_abs_error=0.0,
    )
    final_attempt = value["attempts"][2 * index + 1]
    final_attempt.update(status="success", to_state="passed", reason=None)
    timing = value["observations"][index + 1]["timing"]
    timing["failure_kind"] = "execution"
    value["attempts"][2 * (index + 1) + 1]["reason"] = "execution"
    value["summary"].update(passed=1, failed=len(suite.expected_cells) - 1)
    with pytest.raises(SchemaError, match="materialization"):
        _parse_result(value)
