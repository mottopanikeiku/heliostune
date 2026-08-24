"""Manual Triton FP16 matmul and GPU benchmark collection."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Literal

import torch
import triton
import triton.language as tl

from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, KernelConfig, Workload
from heliostune.errors import ProtocolError
from heliostune.schema import HardwareProfile, Measurement
from heliostune.validation import exact_int, finite_float, nonblank_string


@triton.jit
def _matmul_kernel(  # type: ignore[no-untyped-def]
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Compute one grouped tile of ``C = A @ B``."""
    program_id = tl.program_id(axis=0)
    programs_m = tl.cdiv(m, BLOCK_M)
    programs_n = tl.cdiv(n, BLOCK_N)
    programs_per_group = GROUP_M * programs_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_m = min(programs_m - first_program_m, GROUP_M)
    program_m = first_program_m + (program_id % programs_per_group) % group_m
    program_n = (program_id % programs_per_group) // group_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_offset in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] + k_offset * BLOCK_K < k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offsets_k[:, None] + k_offset * BLOCK_K < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offsets_cm = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_cn = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator.to(tl.float16),
        mask=(offsets_cm[:, None] < m) & (offsets_cn[None, :] < n),
    )


def matmul(a: torch.Tensor, b: torch.Tensor, config: KernelConfig) -> torch.Tensor:
    """Multiply two CUDA FP16 matrices with a specific manual launch configuration."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul inputs must be two-dimensional")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}")
    if a.device.type != "cuda" or b.device.type != "cuda" or a.device != b.device:
        raise ValueError("matmul inputs must be on the same CUDA device")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul inputs must have dtype torch.float16")

    m, k = a.shape
    _, n = b.shape
    output = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),)
    _matmul_kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        GROUP_M=config.group_m,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output


def get_hardware_profile(
    gpu: str | None = None, device: torch.device | None = None
) -> HardwareProfile:
    """Read the active CUDA device properties into the portable benchmark schema."""
    selected_device = (
        device if device is not None else torch.device("cuda", torch.cuda.current_device())
    )
    properties = torch.cuda.get_device_properties(selected_device)
    return HardwareProfile(
        gpu=gpu or properties.name,
        device_name=properties.name,
        compute_capability=(properties.major, properties.minor),
        multiprocessor_count=properties.multi_processor_count,
        total_memory_gb=properties.total_memory / (1024**3),
        cuda_version=(None if torch.version.cuda is None else str(torch.version.cuda)),
        torch_version=str(torch.__version__),
        triton_version=str(triton.__version__),
    )


def _timed_do_bench(
    function: Callable[[], torch.Tensor],
    *,
    warmup_ms: float,
    rep_ms: float,
) -> tuple[tuple[float, float, float], float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    values = triton.testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.2, 0.5, 0.8],
    )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1_000
    p20, median, p80 = (float(value) for value in values)
    return (p20, median, p80), wall_ms


def _within_tolerance(
    difference: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> bool:
    return bool(torch.all(difference <= (atol + rtol * torch.abs(expected))).item())


def _benchmark_config(
    a: torch.Tensor,
    b: torch.Tensor,
    expected: torch.Tensor,
    difference: torch.Tensor,
    hardware: HardwareProfile,
    workload: Workload,
    config: KernelConfig,
    torch_quantiles_ms: tuple[float, float, float],
    torch_benchmark_wall_ms: float,
    bank: int,
    warmup_ms: float,
    rep_ms: float,
    atol: float,
    rtol: float,
) -> Measurement:
    output: torch.Tensor | None = None
    max_abs_error: float | None = None
    compile_ms: float | None = None
    failure_stage: Literal["compile", "correctness", "benchmark"] = "compile"
    torch_p20_ms, torch_latency_ms, torch_p80_ms = torch_quantiles_ms
    try:
        torch.cuda.synchronize(a.device)
        compile_started = time.perf_counter()
        output = matmul(a, b, config)
        torch.cuda.synchronize(a.device)
        compile_ms = (time.perf_counter() - compile_started) * 1_000

        failure_stage = "correctness"
        if not bool(torch.isfinite(output).all().item()):
            return Measurement(
                hardware=hardware,
                workload=workload,
                config=config,
                latency_ms=None,
                torch_latency_ms=torch_latency_ms,
                bank=bank,
                correct=False,
                torch_latency_p20_ms=torch_p20_ms,
                torch_latency_p80_ms=torch_p80_ms,
                compile_ms=compile_ms,
                torch_benchmark_wall_ms=torch_benchmark_wall_ms,
                failure_stage="correctness",
                error="correctness check failed: non-finite output",
            )
        torch.sub(output, expected, out=difference)
        difference.abs_()
        max_abs_error = float(difference.max().item())
        within_tolerance = _within_tolerance(
            difference,
            expected,
            atol=atol,
            rtol=rtol,
        )
        if not within_tolerance:
            return Measurement(
                hardware=hardware,
                workload=workload,
                config=config,
                latency_ms=None,
                torch_latency_ms=torch_latency_ms,
                bank=bank,
                correct=False,
                max_abs_error=max_abs_error,
                torch_latency_p20_ms=torch_p20_ms,
                torch_latency_p80_ms=torch_p80_ms,
                compile_ms=compile_ms,
                torch_benchmark_wall_ms=torch_benchmark_wall_ms,
                failure_stage="correctness",
                error=f"correctness check failed (atol={atol}, rtol={rtol})",
            )

        failure_stage = "benchmark"
        triton_quantiles_ms, benchmark_wall_ms = _timed_do_bench(
            lambda: matmul(a, b, config),
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
        latency_p20_ms, latency_ms, latency_p80_ms = triton_quantiles_ms
        return Measurement(
            hardware=hardware,
            workload=workload,
            config=config,
            latency_ms=latency_ms,
            torch_latency_ms=torch_latency_ms,
            bank=bank,
            correct=True,
            max_abs_error=max_abs_error,
            latency_p20_ms=latency_p20_ms,
            latency_p80_ms=latency_p80_ms,
            torch_latency_p20_ms=torch_p20_ms,
            torch_latency_p80_ms=torch_p80_ms,
            compile_ms=compile_ms,
            benchmark_wall_ms=benchmark_wall_ms,
            torch_benchmark_wall_ms=torch_benchmark_wall_ms,
        )
    except Exception as exc:
        return Measurement(
            hardware=hardware,
            workload=workload,
            config=config,
            latency_ms=None,
            torch_latency_ms=torch_latency_ms,
            bank=bank,
            correct=False,
            max_abs_error=max_abs_error,
            torch_latency_p20_ms=torch_p20_ms,
            torch_latency_p80_ms=torch_p80_ms,
            compile_ms=compile_ms,
            torch_benchmark_wall_ms=torch_benchmark_wall_ms,
            failure_stage=failure_stage,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        del output


def _validated_manifest(
    *,
    gpu: str | None,
    bank: int,
    configs: Sequence[KernelConfig],
    workloads: Sequence[Workload],
    warmup_ms: int | float,
    rep_ms: int | float,
    atol: int | float,
    rtol: int | float,
) -> tuple[
    int,
    tuple[KernelConfig, ...],
    tuple[Workload, ...],
    float,
    float,
    float,
    float,
]:
    if gpu is not None:
        nonblank_string(gpu, context="gpu")
    validated_bank = exact_int(bank, context="bank", minimum=0)
    validated_warmup = finite_float(warmup_ms, context="warmup_ms", minimum=0)
    validated_rep = finite_float(rep_ms, context="rep_ms", strictly_positive=True)
    validated_atol = finite_float(atol, context="atol", minimum=0)
    validated_rtol = finite_float(rtol, context="rtol", minimum=0)

    config_manifest = tuple(configs)
    if not config_manifest:
        raise ProtocolError("config manifest must not be empty")
    if any(type(config) is not KernelConfig for config in config_manifest):
        raise ProtocolError("config manifest must contain only KernelConfig values")
    config_keys = tuple(config.key for config in config_manifest)
    if len(set(config_keys)) != len(config_keys):
        raise ProtocolError("config manifest contains duplicate configurations")
    for config in config_manifest:
        for name in ("block_m", "block_n", "block_k"):
            value = getattr(config, name)
            if value <= 0 or value & (value - 1):
                raise ProtocolError(f"config {config.key} has invalid {name}")

    workload_manifest = tuple(workloads)
    if not workload_manifest:
        raise ProtocolError("workload manifest must not be empty")
    if any(type(workload) is not Workload for workload in workload_manifest):
        raise ProtocolError("workload manifest must contain only Workload values")
    workload_keys = tuple(workload.key for workload in workload_manifest)
    if len(set(workload_keys)) != len(workload_keys):
        raise ProtocolError("workload manifest contains duplicate workloads")
    return (
        validated_bank,
        config_manifest,
        workload_manifest,
        validated_warmup,
        validated_rep,
        validated_atol,
        validated_rtol,
    )


@torch.inference_mode()
def benchmark_measurements(
    gpu: str | None = None,
    *,
    bank: int = 0,
    configs: Sequence[KernelConfig] = DEFAULT_CONFIGS,
    workloads: Sequence[Workload] = DEFAULT_WORKLOADS,
    warmup_ms: int | float = 25,
    rep_ms: int | float = 100,
    atol: int | float = 1e-2,
    rtol: int | float = 1e-2,
    hardware_profile: HardwareProfile | None = None,
    workload_order_seed: int | None = None,
    config_order_seeds: Mapping[str, int] | None = None,
    tensor_seeds: Mapping[str, int] | None = None,
) -> list[Measurement]:
    """Benchmark all workload/config cells while containing per-config failures."""
    (
        bank,
        config_manifest,
        workload_manifest,
        warmup,
        repetition,
        absolute_tolerance,
        relative_tolerance,
    ) = _validated_manifest(
        gpu=gpu,
        bank=bank,
        configs=configs,
        workloads=workloads,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        atol=atol,
        rtol=rtol,
    )
    if workload_order_seed is not None:
        workload_order_seed = exact_int(
            workload_order_seed,
            context="workload_order_seed",
            minimum=0,
        )
    workload_keys = {workload.key for workload in workload_manifest}
    for name, values in (
        ("config_order_seeds", config_order_seeds),
        ("tensor_seeds", tensor_seeds),
    ):
        if values is not None:
            if set(values) != workload_keys:
                raise ProtocolError(f"{name} keys must exactly match the workload manifest")
            for workload_key, seed in values.items():
                exact_int(seed, context=f"{name}[{workload_key!r}]", minimum=0)
    device = torch.device("cuda", torch.cuda.current_device())
    hardware = get_hardware_profile(gpu, device) if hardware_profile is None else hardware_profile
    if gpu is not None and hardware.gpu != gpu:
        raise ProtocolError(f"provided hardware profile is for {hardware.gpu!r}, expected {gpu!r}")
    measurements: list[Measurement] = []
    randomizer = random.Random(bank if workload_order_seed is None else workload_order_seed)
    ordered_workloads = list(workload_manifest)
    randomizer.shuffle(ordered_workloads)

    for workload_index, workload in enumerate(ordered_workloads):
        try:
            torch.manual_seed(
                bank * 10_000 + workload_index
                if tensor_seeds is None
                else tensor_seeds[workload.key]
            )
            a = torch.empty((workload.m, workload.k), device=device, dtype=torch.float16)
            b = torch.empty((workload.k, workload.n), device=device, dtype=torch.float16)
            a.uniform_(-1.0, 1.0)
            b.uniform_(-1.0, 1.0)
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                expected = torch.mm(a, b, out_dtype=torch.float32)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            difference = torch.empty_like(expected)
            benchmark_torch = partial(torch.matmul, a, b)
            torch_quantiles_ms, torch_benchmark_wall_ms = _timed_do_bench(
                benchmark_torch,
                warmup_ms=warmup,
                rep_ms=repetition,
            )
        except Exception as exc:
            raise ProtocolError(
                f"workload setup failed for {workload.key}: {type(exc).__name__}: {exc}"
            ) from exc

        ordered_configs = list(config_manifest)
        if config_order_seeds is None:
            randomizer.shuffle(ordered_configs)
        else:
            random.Random(config_order_seeds[workload.key]).shuffle(ordered_configs)
        for config in ordered_configs:
            measurements.append(
                _benchmark_config(
                    a,
                    b,
                    expected,
                    difference,
                    hardware,
                    workload,
                    config,
                    torch_quantiles_ms,
                    torch_benchmark_wall_ms,
                    bank,
                    warmup,
                    repetition,
                    absolute_tolerance,
                    relative_tolerance,
                )
            )
        del difference, expected, b, a

    return measurements


def collect_benchmarks(
    gpu: str | None = None,
    *,
    bank: int = 0,
    configs: Sequence[KernelConfig] = DEFAULT_CONFIGS,
    workloads: Sequence[Workload] = DEFAULT_WORKLOADS,
    warmup_ms: int | float = 25,
    rep_ms: int | float = 100,
    atol: int | float = 1e-2,
    rtol: int | float = 1e-2,
    hardware_profile: HardwareProfile | None = None,
    workload_order_seed: int | None = None,
    config_order_seeds: Mapping[str, int] | None = None,
    tensor_seeds: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Return benchmark measurements as strict schema-v2 records."""
    return [
        measurement.to_dict()
        for measurement in benchmark_measurements(
            gpu,
            bank=bank,
            configs=configs,
            workloads=workloads,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
            atol=atol,
            rtol=rtol,
            hardware_profile=hardware_profile,
            workload_order_seed=workload_order_seed,
            config_order_seeds=config_order_seeds,
            tensor_seeds=tensor_seeds,
        )
    ]
