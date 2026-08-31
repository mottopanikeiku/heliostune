"""GPU-only native Triton residual RMSNorm implementation.

This module targets the pinned PyTorch 2.8 and Triton 3.4 optional dependencies.
It must be reached through :mod:`heliostune.fusion_kernels`, never imported by
CPU-only discovery code.
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


class _RMSNormTritonConfig(Protocol):
    @property
    def config_id(self) -> str: ...

    @property
    def entrypoint(self) -> str: ...

    @property
    def block_size(self) -> int: ...

    @property
    def num_warps(self) -> int: ...

    @property
    def num_stages(self) -> int: ...


_COMPILE_CONFIGS = frozenset(
    {
        (
            f"rmsnorm-triton-w{num_warps}",
            f"heliostune_fusion_v2::residual_rmsnorm_w{num_warps}",
            4096,
            num_warps,
            1,
        )
        for num_warps in (4, 8, 16, 32)
    }
)


@triton.jit
def _residual_rmsnorm_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    residual_ptr,
    gamma_ptr,
    output_ptr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute one fixed-width residual RMSNorm row per Triton program."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < N_COLS
    row_offsets = row * N_COLS + columns

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + residual
    sum_squares = tl.sum(z * z, axis=0, dtype=tl.float32)  # type: ignore[attr-defined]
    inverse_rms = tl.rsqrt(  # type: ignore[attr-defined]
        sum_squares * (1.0 / N_COLS) + 1e-5
    )
    gamma = tl.load(gamma_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    output = (z * inverse_rms * gamma).to(tl.bfloat16)  # type: ignore[attr-defined]
    tl.store(output_ptr + row_offsets, output, mask=mask)


@triton_op("heliostune_fusion_v2::residual_rmsnorm", mutates_args=())
def _residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    num_warps: int,
) -> torch.Tensor:
    output = torch.empty_like(x)
    wrap_triton(_residual_rmsnorm_kernel)[(128,)](
        x,
        residual,
        gamma,
        output,
        N_COLS=4096,
        BLOCK_SIZE=4096,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _validate_residual_rmsnorm_inputs(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> None:
    if x.shape != (128, 4096) or residual.shape != (128, 4096):
        raise ValueError("x and residual must have shape (128, 4096)")
    if gamma.shape != (4096,):
        raise ValueError("gamma must have shape (4096,)")
    if (
        x.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or gamma.dtype != torch.bfloat16
    ):
        raise ValueError("x, residual, and gamma must have dtype torch.bfloat16")
    if not x.is_cuda or not residual.is_cuda or not gamma.is_cuda:
        raise ValueError("x, residual, and gamma must be CUDA tensors")
    if residual.device != x.device or gamma.device != x.device:
        raise ValueError("x, residual, and gamma must be on the same CUDA device")
    if not x.is_contiguous() or not residual.is_contiguous() or not gamma.is_contiguous():
        raise ValueError("x, residual, and gamma must be contiguous")


def _validate_compile_config(config: _RMSNormTritonConfig) -> None:
    try:
        values = (
            config.config_id,
            config.entrypoint,
            config.block_size,
            config.num_warps,
            config.num_stages,
        )
    except AttributeError as exc:
        raise ValueError("invalid residual RMSNorm compile config") from exc
    if (
        type(values[0]) is not str
        or type(values[1]) is not str
        or any(type(value) is not int for value in values[2:])
        or values not in _COMPILE_CONFIGS
    ):
        raise ValueError("unsupported residual RMSNorm compile config")


def compile_residual_rmsnorm(
    config: _RMSNormTritonConfig,
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
) -> object:
    """Compile one exact candidate and initialize handles without launching it."""
    _validate_compile_config(config)
    _validate_residual_rmsnorm_inputs(x, residual, gamma)
    output = torch.empty_like(x)
    compiled = _residual_rmsnorm_kernel.warmup(
        x,
        residual,
        gamma,
        output,
        grid=(128,),
        N_COLS=4096,
        BLOCK_SIZE=config.block_size,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    _ = compiled.run
    return compiled


def residual_rmsnorm_w4(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    _validate_residual_rmsnorm_inputs(x, residual, gamma)
    return _residual_rmsnorm(x, residual, gamma, 4)  # type: ignore[no-any-return]


def residual_rmsnorm_w8(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    _validate_residual_rmsnorm_inputs(x, residual, gamma)
    return _residual_rmsnorm(x, residual, gamma, 8)  # type: ignore[no-any-return]


def residual_rmsnorm_w16(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    _validate_residual_rmsnorm_inputs(x, residual, gamma)
    return _residual_rmsnorm(x, residual, gamma, 16)  # type: ignore[no-any-return]


def residual_rmsnorm_w32(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    _validate_residual_rmsnorm_inputs(x, residual, gamma)
    return _residual_rmsnorm(x, residual, gamma, 32)  # type: ignore[no-any-return]


ResidualRMSNorm = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_GPU_ENTRYPOINTS: MappingProxyType[str, ResidualRMSNorm] = MappingProxyType(
    {
        "heliostune_fusion_v2::residual_rmsnorm_w4": residual_rmsnorm_w4,
        "heliostune_fusion_v2::residual_rmsnorm_w8": residual_rmsnorm_w8,
        "heliostune_fusion_v2::residual_rmsnorm_w16": residual_rmsnorm_w16,
        "heliostune_fusion_v2::residual_rmsnorm_w32": residual_rmsnorm_w32,
    }
)


def load_residual_rmsnorm(entrypoint: str) -> ResidualRMSNorm:
    """Return the callable for one exact, closed native entrypoint."""
    try:
        return _GPU_ENTRYPOINTS[entrypoint]
    except KeyError as exc:
        raise ValueError(f"unknown residual RMSNorm GPU entrypoint: {entrypoint!r}") from exc
