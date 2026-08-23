"""Manual Triton FP16 matmul and GPU benchmark collection."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from typing import Any

import torch
import triton
import triton.language as tl

from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, KernelConfig, Workload
from heliostune.schema import HardwareProfile, Measurement


@triton.jit
def _matmul_kernel(
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


def _benchmark_config(
    a: torch.Tensor,
    b: torch.Tensor,
    expected: torch.Tensor,
    hardware: HardwareProfile,
    workload: Workload,
    config: KernelConfig,
    torch_latency_ms: float,
    replicate: int,
    warmup_ms: int,
    rep_ms: int,
    atol: float,
    rtol: float,
) -> Measurement:
    output: torch.Tensor | None = None
    max_abs_error: float | None = None
    compile_ms: float | None = None
    try:
        compile_started = time.perf_counter()
        output = matmul(a, b, config)
        torch.cuda.synchronize(a.device)
        compile_ms = (time.perf_counter() - compile_started) * 1_000
        if not bool(torch.isfinite(output).all().item()):
            return Measurement(
                hardware=hardware,
                workload=workload,
                config=config,
                latency_ms=None,
                torch_latency_ms=torch_latency_ms,
                replicate=replicate,
                correct=False,
                compile_ms=compile_ms,
                error="correctness check failed: non-finite output",
            )
        max_abs_error = float((output.float() - expected.float()).abs().max().item())
        if not torch.allclose(output, expected, atol=atol, rtol=rtol):
            return Measurement(
                hardware=hardware,
                workload=workload,
                config=config,
                latency_ms=None,
                torch_latency_ms=torch_latency_ms,
                replicate=replicate,
                correct=False,
                max_abs_error=max_abs_error,
                compile_ms=compile_ms,
                error=f"correctness check failed (atol={atol}, rtol={rtol})",
            )

        latency_p20_ms, latency_ms, latency_p80_ms = (
            float(value)
            for value in triton.testing.do_bench(
                lambda: matmul(a, b, config),
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=[0.2, 0.5, 0.8],
            )
        )
        return Measurement(
            hardware=hardware,
            workload=workload,
            config=config,
            latency_ms=latency_ms,
            torch_latency_ms=torch_latency_ms,
            replicate=replicate,
            correct=True,
            max_abs_error=max_abs_error,
            latency_p20_ms=latency_p20_ms,
            latency_p80_ms=latency_p80_ms,
            compile_ms=compile_ms,
        )
    except Exception as exc:
        return Measurement(
            hardware=hardware,
            workload=workload,
            config=config,
            latency_ms=None,
            torch_latency_ms=torch_latency_ms,
            replicate=replicate,
            correct=False,
            max_abs_error=max_abs_error,
            compile_ms=compile_ms,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        del output


def benchmark_measurements(
    gpu: str | None = None,
    *,
    replicate: int = 0,
    configs: Sequence[KernelConfig] = DEFAULT_CONFIGS,
    workloads: Sequence[Workload] = DEFAULT_WORKLOADS,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> list[Measurement]:
    """Benchmark all workload/config cells while containing per-config failures."""
    if replicate < 0:
        raise ValueError("replicate must be non-negative")
    if warmup_ms < 0 or rep_ms <= 0:
        raise ValueError("warmup_ms must be non-negative and rep_ms must be positive")
    device = torch.device("cuda", torch.cuda.current_device())
    hardware = get_hardware_profile(gpu, device)
    measurements: list[Measurement] = []
    randomizer = random.Random(replicate)
    ordered_workloads = list(workloads)
    randomizer.shuffle(ordered_workloads)

    for workload_index, workload in enumerate(ordered_workloads):
        torch.manual_seed(replicate * 10_000 + workload_index)
        a = torch.empty((workload.m, workload.k), device=device, dtype=torch.float16)
        b = torch.empty((workload.k, workload.n), device=device, dtype=torch.float16)
        a.uniform_(-1.0, 1.0)
        b.uniform_(-1.0, 1.0)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            expected = torch.matmul(a.float(), b.float()).to(torch.float16)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch_latency_ms = float(
            triton.testing.do_bench(
                lambda left=a, right=b: torch.matmul(left, right),
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=[0.5],
            )
        )
        ordered_configs = list(configs)
        randomizer.shuffle(ordered_configs)
        for config in ordered_configs:
            measurements.append(
                _benchmark_config(
                    a,
                    b,
                    expected,
                    hardware,
                    workload,
                    config,
                    torch_latency_ms,
                    replicate,
                    warmup_ms,
                    rep_ms,
                    atol,
                    rtol,
                )
            )
        del expected, b, a
        torch.cuda.empty_cache()

    return measurements


def collect_benchmarks(
    gpu: str | None = None,
    *,
    replicate: int = 0,
    configs: Sequence[KernelConfig] = DEFAULT_CONFIGS,
    workloads: Sequence[Workload] = DEFAULT_WORKLOADS,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> list[dict[str, Any]]:
    """Return benchmark measurements as JSON-serializable records."""
    return [
        measurement.to_dict()
        for measurement in benchmark_measurements(
            gpu,
            replicate=replicate,
            configs=configs,
            workloads=workloads,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
            atol=atol,
            rtol=rtol,
        )
    ]
