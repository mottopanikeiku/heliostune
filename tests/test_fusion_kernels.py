from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple

import pytest

from heliostune.fusion_kernels import (
    RESIDUAL_RMSNORM_CONFIG_BY_ENTRYPOINT,
    RESIDUAL_RMSNORM_CONFIGS,
    RMSNormTritonConfig,
    compile_residual_rmsnorm,
    compiled_kernel_evidence,
    load_residual_rmsnorm,
)

ROOT = Path(__file__).resolve().parents[1]
CPU_SOURCE = ROOT / "src" / "heliostune" / "fusion_kernels.py"
GPU_SOURCE = ROOT / "src" / "heliostune" / "_fusion_gpu.py"
GPU_TEXT = GPU_SOURCE.read_text(encoding="utf-8")
GPU_TREE = ast.parse(GPU_TEXT)


def _function(name: str) -> ast.FunctionDef:
    matches = [
        node for node in GPU_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _call_name(call: ast.Call) -> str | None:
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _wrapped_launches(node: ast.AST) -> list[ast.Call]:
    launches: list[ast.Call] = []
    for call in (candidate for candidate in ast.walk(node) if isinstance(candidate, ast.Call)):
        if not isinstance(call.func, ast.Subscript):
            continue
        wrapped = call.func.value
        if isinstance(wrapped, ast.Call) and _call_name(wrapped) == "wrap_triton":
            launches.append(call)
    return launches


def test_cpu_config_is_frozen_and_slotted() -> None:
    config = RESIDUAL_RMSNORM_CONFIGS[0]
    assert isinstance(config, RMSNormTritonConfig)
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.num_warps = 8  # type: ignore[misc]


def test_cpu_configs_are_the_exact_four_native_candidates() -> None:
    assert [config.config_id for config in RESIDUAL_RMSNORM_CONFIGS] == [
        "rmsnorm-triton-w4",
        "rmsnorm-triton-w8",
        "rmsnorm-triton-w16",
        "rmsnorm-triton-w32",
    ]
    assert [config.num_warps for config in RESIDUAL_RMSNORM_CONFIGS] == [4, 8, 16, 32]
    assert {(config.block_size, config.num_stages) for config in RESIDUAL_RMSNORM_CONFIGS} == {
        (4096, 1)
    }


def test_cpu_entrypoint_registry_is_exact_and_immutable() -> None:
    expected = {f"heliostune_fusion_v2::residual_rmsnorm_w{warps}" for warps in (4, 8, 16, 32)}
    assert set(RESIDUAL_RMSNORM_CONFIG_BY_ENTRYPOINT) == expected
    assert tuple(RESIDUAL_RMSNORM_CONFIG_BY_ENTRYPOINT.values()) == RESIDUAL_RMSNORM_CONFIGS
    with pytest.raises(TypeError):
        RESIDUAL_RMSNORM_CONFIG_BY_ENTRYPOINT["other::kernel"] = RESIDUAL_RMSNORM_CONFIGS[0]  # type: ignore[index]


def test_cpu_module_import_does_not_load_gpu_dependencies() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, heliostune.fusion_kernels; "
                "assert 'heliostune._fusion_gpu' not in sys.modules; "
                "assert 'torch' not in sys.modules; "
                "assert 'triton' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == ""


def test_cpu_module_has_no_gpu_imports_or_mlp_symbol() -> None:
    tree = ast.parse(CPU_SOURCE.read_text(encoding="utf-8"))
    top_level_imports = {
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    }
    top_level_from_imports = {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "torch" not in top_level_imports | top_level_from_imports
    assert "triton" not in top_level_imports | top_level_from_imports
    assert "mlp" not in CPU_SOURCE.read_text(encoding="utf-8").lower()
    assert "mlp" not in GPU_TEXT.lower()


def test_gpu_source_defines_one_jit_row_kernel_and_one_registered_op() -> None:
    jit_functions = [
        node
        for node in GPU_TREE.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "triton"
            and decorator.attr == "jit"
            for decorator in node.decorator_list
        )
    ]
    registered = [
        node
        for node in GPU_TREE.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Call) and _call_name(decorator) == "triton_op"
            for decorator in node.decorator_list
        )
    ]
    assert [function.name for function in jit_functions] == ["_residual_rmsnorm_kernel"]
    assert [function.name for function in registered] == ["_residual_rmsnorm"]


def test_registered_op_has_exact_namespace_and_empty_mutation_schema() -> None:
    operation = _function("_residual_rmsnorm")
    decorators = [
        decorator
        for decorator in operation.decorator_list
        if isinstance(decorator, ast.Call) and _call_name(decorator) == "triton_op"
    ]
    assert len(decorators) == 1
    decorator = decorators[0]
    assert len(decorator.args) == 1
    assert ast.literal_eval(decorator.args[0]) == "heliostune_fusion_v2::residual_rmsnorm"
    keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
    assert ast.literal_eval(keywords["mutates_args"]) == ()


def test_registered_op_contains_one_wrapped_launch() -> None:
    operation = _function("_residual_rmsnorm")
    launches = _wrapped_launches(operation)
    assert len(launches) == 1
    wrapped = launches[0].func
    assert isinstance(wrapped, ast.Subscript)
    assert isinstance(wrapped.value, ast.Call)
    assert len(wrapped.value.args) == 1
    assert isinstance(wrapped.value.args[0], ast.Name)
    assert wrapped.value.args[0].id == "_residual_rmsnorm_kernel"


def test_registered_op_has_no_extra_operations_around_launch() -> None:
    operation = _function("_residual_rmsnorm")
    assert len(operation.body) == 3
    allocation, launch, returned = operation.body
    assert isinstance(allocation, ast.Assign)
    assert isinstance(allocation.value, ast.Call)
    assert _call_name(allocation.value) == "torch.empty_like"
    assert isinstance(launch, ast.Expr) and launch.value in _wrapped_launches(operation)
    assert isinstance(returned, ast.Return)
    assert isinstance(returned.value, ast.Name) and returned.value.id == "output"


def test_wrapped_launch_has_static_grid_shape_and_compile_constants() -> None:
    launch = _wrapped_launches(_function("_residual_rmsnorm"))[0]
    subscript = launch.func
    assert isinstance(subscript, ast.Subscript)
    assert ast.literal_eval(subscript.slice) == (128,)
    keywords = {keyword.arg: keyword.value for keyword in launch.keywords}
    assert ast.literal_eval(keywords["N_COLS"]) == 4096
    assert ast.literal_eval(keywords["BLOCK_SIZE"]) == 4096
    assert ast.literal_eval(keywords["num_stages"]) == 1
    assert isinstance(keywords["num_warps"], ast.Name)
    assert keywords["num_warps"].id == "num_warps"
    wrapper_warps = {
        ast.literal_eval(call.args[-1])
        for name in (
            "residual_rmsnorm_w4",
            "residual_rmsnorm_w8",
            "residual_rmsnorm_w16",
            "residual_rmsnorm_w32",
        )
        for call in ast.walk(_function(name))
        if isinstance(call, ast.Call) and _call_name(call) == "_residual_rmsnorm"
    }
    assert wrapper_warps == {4, 8, 16, 32}
    wrapper_names = (
        "residual_rmsnorm_w4",
        "residual_rmsnorm_w8",
        "residual_rmsnorm_w16",
        "residual_rmsnorm_w32",
    )
    for name in wrapper_names:
        first_statement = _function(name).body[0]
        assert isinstance(first_statement, ast.Expr)
        assert isinstance(first_statement.value, ast.Call)
        assert _call_name(first_statement.value) == "_validate_residual_rmsnorm_inputs"
    validation_source = ast.unparse(_function("_validate_residual_rmsnorm_inputs"))
    assert "(128, 4096)" in validation_source
    assert "(4096,)" in validation_source
    assert validation_source.count("torch.bfloat16") >= 3
    assert validation_source.count(".is_cuda") == 3
    assert validation_source.count(".is_contiguous()") == 3


def test_row_kernel_enforces_fp32_math_bf16_output_and_no_input_mutation() -> None:
    kernel = _function("_residual_rmsnorm_kernel")
    calls = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)]
    names = [_call_name(call) for call in calls]
    assert names.count("tl.load") == 3
    assert names.count("tl.store") == 1
    assert names.count("tl.sum") == 1
    assert names.count("tl.rsqrt") == 1
    fp32_casts = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "to"
        and call.args
        and ast.unparse(call.args[0]) == "tl.float32"
    ]
    bf16_casts = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "to"
        and call.args
        and ast.unparse(call.args[0]) == "tl.bfloat16"
    ]
    assert len(fp32_casts) == 3
    assert len(bf16_casts) == 1
    memory_calls = [call for call in calls if _call_name(call) in {"tl.load", "tl.store"}]
    assert all(any(keyword.arg == "mask" for keyword in call.keywords) for call in memory_calls)
    store = next(call for call in calls if _call_name(call) == "tl.store")
    assert isinstance(store.args[0], ast.BinOp)
    assert isinstance(store.args[0].left, ast.Name)
    assert store.args[0].left.id == "output_ptr"
    assert not any(name is not None and "atomic" in name for name in names)


def test_invalid_entrypoint_fails_before_lazy_gpu_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "heliostune._fusion_gpu", raising=False)
    with pytest.raises(ValueError, match="unknown residual RMSNorm entrypoint"):
        load_residual_rmsnorm("os.system")
    assert "heliostune._fusion_gpu" not in sys.modules


def test_invalid_compile_entrypoint_fails_before_lazy_gpu_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "heliostune._fusion_gpu", raising=False)
    with pytest.raises(ValueError, match="unknown residual RMSNorm entrypoint"):
        compile_residual_rmsnorm("os.system", object(), object(), object())
    assert "heliostune._fusion_gpu" not in sys.modules


def test_gpu_compile_path_warms_without_launch_and_touches_run_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[tuple[object, ...], dict[str, object]]] = []
    warmups: list[tuple[tuple[object, ...], dict[str, object]]] = []
    run_accesses: list[None] = []
    allocations: list[object] = []

    class FakeTensor:
        shape: tuple[int, ...]
        dtype: object
        is_cuda = True
        device = "cuda:0"

        def __init__(self, shape: tuple[int, ...], dtype: object) -> None:
            self.shape = shape
            self.dtype = dtype

        def is_contiguous(self) -> bool:
            return True

    class FakeCompiled:
        @property
        def run(self) -> object:
            run_accesses.append(None)
            return object()

    class FakeJITKernel:
        def __init__(self, function: object) -> None:
            self.function = function

        def __call__(self, *args: object, **kwargs: object) -> None:
            launches.append((args, kwargs))

        def warmup(self, *args: object, **kwargs: object) -> FakeCompiled:
            warmups.append((args, kwargs))
            return FakeCompiled()

    bfloat16 = object()
    torch_module = ModuleType("torch")
    torch_module.Tensor = FakeTensor  # type: ignore[attr-defined]
    torch_module.bfloat16 = bfloat16  # type: ignore[attr-defined]

    def empty_like(value: object) -> object:
        output = object()
        allocations.append(value)
        return output

    torch_module.empty_like = empty_like  # type: ignore[attr-defined]
    torch_library = ModuleType("torch.library")

    def triton_op(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        return lambda function: function

    torch_library.triton_op = triton_op  # type: ignore[attr-defined]
    torch_library.wrap_triton = lambda kernel: kernel  # type: ignore[attr-defined]
    triton_module = ModuleType("triton")
    triton_module.jit = FakeJITKernel  # type: ignore[attr-defined]
    language_module = ModuleType("triton.language")
    language_module.constexpr = object()  # type: ignore[attr-defined]
    triton_module.language = language_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.library", torch_library)
    monkeypatch.setitem(sys.modules, "triton", triton_module)
    monkeypatch.setitem(sys.modules, "triton.language", language_module)

    spec = importlib.util.spec_from_file_location("_fusion_gpu_fake", GPU_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    x = FakeTensor((128, 4096), bfloat16)
    residual = FakeTensor((128, 4096), bfloat16)
    gamma = FakeTensor((4096,), bfloat16)
    config = RESIDUAL_RMSNORM_CONFIGS[1]
    compiled = module.compile_residual_rmsnorm(config, x, residual, gamma)
    assert isinstance(compiled, FakeCompiled)
    assert allocations == [x]
    assert launches == []
    assert run_accesses == [None]
    assert len(warmups) == 1
    args, kwargs = warmups[0]
    assert args[:3] == (x, residual, gamma)
    assert len(args) == 4
    assert kwargs == {
        "grid": (128,),
        "N_COLS": 4096,
        "BLOCK_SIZE": 4096,
        "num_warps": 8,
        "num_stages": 1,
    }

    invalid = replace(config, num_warps=7)
    with pytest.raises(ValueError, match="unsupported residual RMSNorm compile config"):
        module.compile_residual_rmsnorm(invalid, x, residual, gamma)
    assert len(warmups) == 1


def test_compiled_kernel_evidence_is_exact_hashed_and_sorted() -> None:
    @dataclass(frozen=True)
    class Target:
        backend: str
        arch: str
        warp_size: int

    class Metadata(NamedTuple):
        shared: int
        num_warps: int
        num_ctas: int
        num_stages: int
        cluster_dims: tuple[int, int, int]
        target: Target

    text = "α ptx\n"
    binary = b"\x00\xffcubin"
    compiled = SimpleNamespace(
        hash="kernel-hash",
        name="residual_rmsnorm",
        metadata=Metadata(2048, 8, 1, 1, (1, 1, 1), Target("cuda", "sm90", 32)),
        n_regs=64,
        n_spills=0,
        n_max_threads=256,
        asm={"ptx": text, "cubin": binary},
    )

    evidence = compiled_kernel_evidence(compiled, RESIDUAL_RMSNORM_CONFIGS[1])
    assert set(evidence) == {
        "status",
        "error",
        "kernel_name",
        "kernel_hash",
        "target",
        "metadata",
        "n_regs",
        "n_spills",
        "n_max_threads",
        "asm_stages",
        "resource_gate_passed",
    }
    assert evidence["status"] == "compiled"
    assert evidence["error"] is None
    assert evidence["kernel_name"] == "residual_rmsnorm"
    assert evidence["kernel_hash"] == "kernel-hash"
    assert evidence["target"] == {"backend": "cuda", "arch": "sm90", "warp_size": 32}
    assert evidence["metadata"] == {
        "shared": 2048,
        "num_warps": 8,
        "num_ctas": 1,
        "num_stages": 1,
        "cluster_dims": [1, 1, 1],
        "target": {"backend": "cuda", "arch": "sm90", "warp_size": 32},
    }
    assert evidence["resource_gate_passed"] is True
    assert evidence["asm_stages"] == [
        {
            "stage": "cubin",
            "bytes": len(binary),
            "sha256": sha256(binary).hexdigest(),
        },
        {
            "stage": "ptx",
            "bytes": len(text.encode()),
            "sha256": sha256(text.encode()).hexdigest(),
        },
    ]


def test_compiled_kernel_evidence_rejects_missing_or_invalid_metadata() -> None:
    valid = SimpleNamespace(
        hash="hash",
        name="name",
        metadata={
            "shared": 0,
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 1,
            "target": {"backend": "cuda", "arch": "sm90", "warp_size": 32},
        },
        n_regs=1,
        n_spills=0,
        n_max_threads=1,
        asm={"ptx": ""},
    )
    with pytest.raises(ValueError, match="metadata"):
        compiled_kernel_evidence(
            SimpleNamespace(**{**vars(valid), "metadata": object()}),
            RESIDUAL_RMSNORM_CONFIGS[0],
        )
    with pytest.raises(ValueError, match="n_regs"):
        compiled_kernel_evidence(
            SimpleNamespace(**{**vars(valid), "n_regs": True}),
            RESIDUAL_RMSNORM_CONFIGS[0],
        )
    with pytest.raises(ValueError, match="four registered"):
        compiled_kernel_evidence(
            valid,
            replace(RESIDUAL_RMSNORM_CONFIGS[0], block_size=2048),
        )


def test_gpu_smoke_matches_fp32_reference_when_pinned_cuda_stack_is_available() -> None:
    torch: Any = pytest.importorskip("torch")
    triton: Any = pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not torch.__version__.startswith("2.8.0") or triton.__version__ != "3.4.0":
        pytest.skip("the pinned PyTorch 2.8/Triton 3.4 stack is unavailable")

    from heliostune._fusion_gpu import load_residual_rmsnorm as load_gpu

    torch.manual_seed(17)
    x = torch.randn((128, 4096), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    gamma = (1.0 + 0.02 * torch.randn((4096,), device="cuda")).to(torch.bfloat16)
    z = x.float() + residual.float()
    mean_square = torch.mean(z * z, dim=-1, keepdim=True, dtype=torch.float32)
    expected = (z * torch.rsqrt(mean_square + 1e-5) * gamma.float()).to(torch.bfloat16)
    for warps in (4, 8, 16, 32):
        entrypoint = f"heliostune_fusion_v2::residual_rmsnorm_w{warps}"
        actual = load_gpu(entrypoint)(x, residual, gamma)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.0078125)
