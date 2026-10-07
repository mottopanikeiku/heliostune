"""GPU-only launches for the frozen action expansion.

Split-K uses FP16 tensor-core dot products and FP32 partial tiles, then a second
Triton kernel sums the workspace in FP32 and narrows once to FP16. Allocation,
partial computation, and reduction are all included in each launch. No atomics,
Torch reduction, alternate implementations, or runtime fallback are used.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from heliostune.action_configs import (
    SPLIT_K_M_LIMIT,
    ActionCandidate,
    SplitKConfig,
)
from heliostune.configs import HopperGemmConfig, KernelConfig
from heliostune.hopper_kernel import hopper_matmul
from heliostune.kernel import matmul


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
