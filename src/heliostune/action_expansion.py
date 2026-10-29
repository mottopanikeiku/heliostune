"""GPU-only launches and measurement for the frozen action expansion.

Split-K uses FP16 tensor-core dot products and FP32 partial tiles, then a second
Triton kernel sums the workspace in FP32 and narrows once to FP16. Allocation,
partial computation, and reduction are all included in each launch. No atomics,
Torch reduction, alternate implementations, or runtime fallback are used.

``measure_unit`` times one bank/workload unit with the published protocol, and
``collect_on_device`` runs a whole collection on the current CUDA device. Run
identity, checkpoints and scheduling live in ``heliostune.action_collection``.
"""

from __future__ import annotations

import importlib.metadata
import math
import time
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from heliostune.action_collection import (
    ATOL,
    REP_MS,
    RTOL,
    WARMUP_MS,
    RunSpec,
    begin_session,
    collection_workloads,
    is_complete,
    new_result,
    run_collection,
    save_checkpoint,
    start_or_resume,
    tensor_seed,
    timing_order,
)
from heliostune.action_configs import (
    SPLIT_K_M_LIMIT,
    ActionCandidate,
    SplitKConfig,
    candidates_for,
)
from heliostune.configs import DEFAULT_CONFIGS, HopperGemmConfig, KernelConfig, Workload
from heliostune.hardware import expectation_for_gpu, validate_hardware
from heliostune.hopper_kernel import hopper_matmul
from heliostune.kernel import _timed_do_bench, _within_tolerance, get_hardware_profile, matmul


@triton.jit
def _split_k_partials(  # type: ignore[no-untyped-def]
    a_ptr,
    b_ptr,
    partials_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    program_m = tl.program_id(axis=0)
    program_n = tl.program_id(axis=1)
    split_id = tl.program_id(axis=2)
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    tiles_per_split = tl.cdiv(tl.cdiv(k, BLOCK_K), SPLIT_K)
    split_start = split_id * tiles_per_split * BLOCK_K

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tile in range(tiles_per_split):
        current_k = split_start + tile * BLOCK_K + offsets_k
        a = tl.load(
            a_ptr + offsets_m[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=(offsets_m[:, None] < m) & (current_k[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptr + current_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
            mask=(current_k[:, None] < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)

    partial_ptrs = partials_ptr + split_id * m * n + offsets_m[:, None] * n + offsets_n[None, :]
    tl.store(
        partial_ptrs,
        accumulator,
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


@triton.jit
def _split_k_reduce(  # type: ignore[no-untyped-def]
    partials_ptr,
    output_ptr,
    elements,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    partials = tl.load(
        partials_ptr + splits[:, None] * elements + offsets[None, :],
        mask=offsets[None, :] < elements,
        other=0.0,
    )
    result = tl.sum(partials, axis=0)
    tl.store(output_ptr + offsets, result.to(tl.float16), mask=offsets < elements)


def _split_k_matmul(a: torch.Tensor, b: torch.Tensor, config: SplitKConfig) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul inputs must be two-dimensional")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}")
    if a.device.type != "cuda" or b.device.type != "cuda" or a.device != b.device:
        raise ValueError("matmul inputs must be on the same CUDA device")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul inputs must have dtype torch.float16")

    m, k = a.shape
    n = b.shape[1]
    if not 1 <= m <= SPLIT_K_M_LIMIT or n < 1 or k < 1:
        raise ValueError("split-K requires 1 <= M <= 31 and positive N and K")

    output = torch.empty((m, n), device=a.device, dtype=torch.float16)
    partials = torch.empty((config.split_k, m, n), device=a.device, dtype=torch.float32)
    grid = (
        triton.cdiv(m, config.block_m),
        triton.cdiv(n, config.block_n),
        config.split_k,
    )
    _split_k_partials[grid](
        a,
        b,
        partials,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        SPLIT_K=config.split_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    _split_k_reduce[(triton.cdiv(m * n, 256),)](
        partials,
        output,
        m * n,
        SPLIT_K=config.split_k,
        BLOCK=256,
        num_warps=4,
        num_stages=1,
    )
    return output


def launch(a: torch.Tensor, b: torch.Tensor, candidate: ActionCandidate) -> torch.Tensor:
    """Allocate and launch exactly the requested action, propagating every failure."""
    config = candidate.config
    if isinstance(config, KernelConfig):
        return matmul(a, b, config)
    if isinstance(config, HopperGemmConfig):
        return hopper_matmul(a, b, config)
    if isinstance(config, SplitKConfig):
        return _split_k_matmul(a, b, config)
    raise TypeError("unknown action configuration")


def measure_unit(
    bank: int,
    index: int,
    workload: Workload,
    finalists: frozenset[str] | None,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Check and time one bank/workload unit; every failure stays a row with its stage.

    This is the published H100 collector's unit body: seeded FP16 operands, an FP32
    reference with TF32 disabled, torch first, then the seeded candidate shuffle.
    """
    torch.manual_seed(tensor_seed(bank, index))
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
    actions: dict[str, tuple[str, Any, Callable[[], torch.Tensor]]] = {
        c.key: ("old", c.to_dict(), partial(matmul, a, b, c)) for c in DEFAULT_CONFIGS
    }
    for candidate in candidates_for(workload):
        actions[candidate.key] = (
            candidate.family,
            candidate.to_dict(),
            partial(launch, a, b, candidate),
        )
    actions["torch"] = ("torch", None, partial(torch.matmul, a, b))
    rows: list[dict[str, Any]] = []
    for key in timing_order(bank, index, workload, finalists):
        family, config, function = actions[key]
        row: dict[str, Any] = {
            "workload_key": workload.key,
            "workload": workload.to_dict(),
            "bank": bank,
            "family": family,
            "config_key": key,
            "config": config,
            "correct": False,
            "error": None,
            "p20_ms": None,
            "median_ms": None,
            "p80_ms": None,
            "max_abs_error": None,
            "compile_seconds": None,
            "timing_seconds": None,
        }
        stage = "compile"
        try:
            torch.cuda.synchronize()
            compile_started = time.monotonic()
            output = function()
            torch.cuda.synchronize()
            row["compile_seconds"] = time.monotonic() - compile_started
            stage = "correctness"
            if not bool(torch.isfinite(output).all().item()):
                raise ValueError("non-finite output")
            torch.sub(output, expected, out=difference)
            difference.abs_()
            row["max_abs_error"] = float(difference.max().item())
            if not _within_tolerance(difference, expected, atol=ATOL, rtol=RTOL):
                raise ValueError("outside unchanged atol=rtol=1e-2 tolerance")
            row["correct"] = True
            del output
            stage = "timing"
            quantiles, wall_ms = _timed_do_bench(function, warmup_ms=WARMUP_MS, rep_ms=REP_MS)
            if not all(math.isfinite(v) and v > 0 for v in quantiles):
                raise ValueError("non-positive or non-finite timing")
            row.update(
                p20_ms=quantiles[0],
                median_ms=quantiles[1],
                p80_ms=quantiles[2],
                timing_seconds=wall_ms / 1000,
            )
        except Exception as exc:
            row["error"] = f"{stage}: {type(exc).__name__}: {exc}"
        rows.append(row)
    del difference, expected, b, a, actions
    return rows


def collect_on_device(
    spec: RunSpec,
    checkpoint: Path,
    *,
    after_save: Callable[[], None] = lambda: None,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """Run or resume one collection on the current CUDA device.

    A GPU label with a registry expectation is validated first. Either way, the
    first session's exact hardware profile becomes part of the run's identity.
    """
    started = time.monotonic()
    device = torch.device("cuda", torch.cuda.current_device())
    hardware = get_hardware_profile(spec.gpu, device)
    if spec.validates_hardware:
        validate_hardware(hardware, expectation_for_gpu(spec.gpu))
    versions = {
        package: importlib.metadata.version(package) for package in ("torch", "triton", "numpy")
    }
    precision = {
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    workloads = collection_workloads(spec.pilot)
    fresh = new_result(spec, hardware.to_dict(), versions, precision)
    result, completed = start_or_resume(checkpoint, fresh, workloads)
    if is_complete(completed, workloads):
        return result
    session = begin_session(result, spec, hardware.to_dict(), str(uuid.uuid4()))

    def save(data: dict[str, Any], units: set[tuple[int, str]]) -> None:
        save_checkpoint(checkpoint, data, units)
        after_save()

    return run_collection(
        result,
        completed,
        workloads,
        partial(measure_unit, device=device),
        save,
        session=session,
        started=started,
        clock=time.monotonic,
        log=log,
    )
