"""Hopper-explicit Triton matmul candidates and their pre-timing correctness gate.

The frozen kernel in :mod:`heliostune.kernel` is a plain tiled ``tl.load`` /
``tl.dot`` loop whose only scheduling knobs are ``GROUP_M`` swizzling and
``num_stages``. It uses no Hopper hardware feature -- no tensor memory
accelerator, no warp specialisation, no persistent scheduling -- and it has no
skinny-``M`` variant, so the 32 published workloads with ``M <= 8`` are measured
with a tile shape that masks away at least fifteen of every sixteen rows and
launches only ``ceil(N / block_n)`` programs onto 132 streaming multiprocessors.

This module adds the two candidate kernels that close those gaps:

``hopper_matmul``
    A persistent TMA matmul modelled on the Triton ``v3.4.0`` tutorial
    ``python/tutorials/09-persistent-matmul.py``
    (``matmul_kernel_descriptor_persistent``, lines 481-591). Device-side tensor
    descriptors, a grid capped at the streaming-multiprocessor count, optional
    automatic warp specialisation on the persistent loop, and an optional
    subtiled epilogue.

``skinny_gemv``
    A memory-bound skinny-``M`` kernel with a split-``K`` reduction. It does not
    use the tensor cores at all: a decode-shaped projection has arithmetic
    intensity below one, so the cost is streaming ``B`` and the only thing that
    matters is having enough programs in flight to saturate memory bandwidth.

Every Triton entry point used here was read at the ``v3.4.0`` tag:

* ``tl.make_tensor_descriptor`` -- ``python/triton/language/core.py:2250``;
  block-shape and stride constraints enforced at
  ``python/triton/language/semantic.py:1847-1886``.
* ``tensor_descriptor.load`` / ``.store`` -- ``python/triton/language/core.py``
  lines 1406 and 1416. Reads outside the tensor yield zeros, writes are dropped.
* ``tl.range(..., num_stages=, flatten=, warp_specialize=)`` --
  ``python/triton/language/core.py:3198``.
* ``triton.set_allocator`` -- ``python/triton/runtime/_allocation.py:26``,
  re-exported at ``python/triton/__init__.py:23``. Device-side descriptors
  require it.
* ``tl.atomic_add`` -- ``python/triton/language/core.py:2368``; ``sem`` and
  ``scope`` spellings at ``python/triton/language/semantic.py:952-978``.
* ``tl.reshape`` / ``tl.permute`` / ``tl.split`` for the subtiled epilogue --
  ``tl.split`` splits the last dimension, which must have size two
  (``python/triton/language/core.py:1854``).

The bundled ``typings/triton`` stubs only describe the subset of the language
that the frozen kernel uses, so calls into the descriptor, persistent-loop and
atomic surfaces carry a narrowly coded ``attr-defined`` suppression. Widening
those stubs belongs with the stub file, not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from heliostune.configs import (
    HOPPER_GEMM_CONFIGS,
    SKINNY_GEMV_CONFIGS,
    TENSOR_DESCRIPTOR_ALIGNMENT,
    HopperGemmConfig,
    SkinnyGemvConfig,
    Workload,
)
from heliostune.errors import ProtocolError
from heliostune.hardware import expectation_for_gpu, validate_hardware
from heliostune.hopper_spec import EDGE_WORKLOADS as EDGE_WORKLOADS
from heliostune.hopper_spec import SKINNY_M_LIMIT as SKINNY_M_LIMIT
from heliostune.hopper_spec import validation_workloads as validation_workloads
from heliostune.kernel import _within_tolerance, get_hardware_profile
from heliostune.validation import exact_int, finite_float, nonblank_string

HOPPER_COMPUTE_CAPABILITY = (9, 0)


# --------------------------------------------------------------------------
# Device facts and the TMA workspace allocator
# --------------------------------------------------------------------------


_DEVICE_FACTS: dict[int, tuple[int, tuple[int, int]]] = {}
_ALLOCATOR_INSTALLED = False


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _device_facts(device: torch.device) -> tuple[int, tuple[int, int]]:
    """Return ``(multiprocessor_count, compute_capability)``, cached per device.

    The launchers need both on every call to size the persistent grid and to
    decide the ``flatten`` policy. Querying the driver inside a ``do_bench``
    loop would charge that lookup to the kernel, so it is memoised.
    """
    index = _device_index(device)
    cached = _DEVICE_FACTS.get(index)
    if cached is None:
        properties = torch.cuda.get_device_properties(index)
        cached = (properties.multi_processor_count, (properties.major, properties.minor))
        _DEVICE_FACTS[index] = cached
    return cached


def _tma_workspace(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    """Back Triton's device-side descriptor workspace with a CUDA byte buffer."""
    del alignment, stream  # CUDA allocations are already 512-byte aligned.
    return torch.empty(size, device="cuda", dtype=torch.int8)


def install_tma_allocator() -> None:
    """Register the descriptor workspace allocator exactly once per process.

    ``tl.make_tensor_descriptor`` allocates global scratch at launch time and
    raises if no allocator is registered
    (``python/triton/runtime/_allocation.py:17-21`` at ``v3.4.0``).
    """
    global _ALLOCATOR_INSTALLED
    if not _ALLOCATOR_INSTALLED:
        triton.set_allocator(_tma_workspace)  # type: ignore[attr-defined]
        _ALLOCATOR_INSTALLED = True


# --------------------------------------------------------------------------
# Hopper persistent TMA matmul
# --------------------------------------------------------------------------


@triton.jit
def _persistent_tile_ids(  # type: ignore[no-untyped-def]
    tile_id,
    programs_per_group,
    programs_m,
    GROUP_M: tl.constexpr,
):
    """Map a flat persistent tile index onto grouped ``(program_m, program_n)``."""
    group_id = tile_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_m = min(programs_m - first_program_m, GROUP_M)
    program_m = first_program_m + (tile_id % group_m)
    program_n = (tile_id % programs_per_group) // group_m
    return program_m, program_n


@triton.jit
def _hopper_matmul_kernel(  # type: ignore[no-untyped-def]
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    """Compute ``C = A @ B`` with one persistent program per multiprocessor."""
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    programs_m = tl.cdiv(m, BLOCK_M)
    programs_n = tl.cdiv(n, BLOCK_N)
    k_tiles = tl.cdiv(k, BLOCK_K)
    num_tiles = programs_m * programs_n

    a_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
        a_ptr,
        shape=[m, k],
        strides=[k, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
        b_ptr,
        shape=[k, n],
        strides=[n, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
        c_ptr,
        shape=[m, n],
        strides=[n, 1],
        block_shape=[BLOCK_M, BLOCK_N // 2 if EPILOGUE_SUBTILE else BLOCK_N],
    )

    # Carry a second tile counter so the epilogue does not depend on a value the
    # prologue also consumes; the tutorial does the same to keep the pipeliner
    # from serialising the two.
    epilogue_tile_id = start_pid - NUM_SMS
    programs_per_group = GROUP_M * programs_n

    for tile_id in tl.range(  # type: ignore[attr-defined]
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE,
    ):
        program_m, program_n = _persistent_tile_ids(
            tile_id, programs_per_group, programs_m, GROUP_M
        )
        offsets_m = program_m * BLOCK_M
        offsets_n = program_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_index in range(k_tiles):
            offsets_k = k_index * BLOCK_K
            a = a_desc.load([offsets_m, offsets_k])
            b = b_desc.load([offsets_k, offsets_n])
            accumulator = tl.dot(a, b, accumulator)

        epilogue_tile_id += NUM_SMS
        program_m, program_n = _persistent_tile_ids(
            epilogue_tile_id, programs_per_group, programs_m, GROUP_M
        )
        offsets_cm = program_m * BLOCK_M
        offsets_cn = program_n * BLOCK_N
        if EPILOGUE_SUBTILE:
            split = tl.permute(  # type: ignore[attr-defined]
                tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2)),  # type: ignore[attr-defined]
                (0, 2, 1),
            )
            low, high = tl.split(split)  # type: ignore[attr-defined]
            c_desc.store([offsets_cm, offsets_cn], low.to(dtype))
            c_desc.store([offsets_cm, offsets_cn + BLOCK_N // 2], high.to(dtype))
        else:
            c_desc.store([offsets_cm, offsets_cn], accumulator.to(dtype))


def _validated_operands(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul inputs must be two-dimensional")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}")
    if a.device.type != "cuda" or b.device.type != "cuda" or a.device != b.device:
        raise ValueError("matmul inputs must be on the same CUDA device")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul inputs must have dtype torch.float16")
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b.shape[1])
    return m, n, k


def hopper_matmul(a: torch.Tensor, b: torch.Tensor, config: HopperGemmConfig) -> torch.Tensor:
    """Multiply two CUDA FP16 matrices with the persistent TMA candidate kernel."""
    m, n, k = _validated_operands(a, b)
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("tensor descriptors require row-major contiguous inputs")
    if k % TENSOR_DESCRIPTOR_ALIGNMENT or n % TENSOR_DESCRIPTOR_ALIGNMENT:
        raise ValueError(
            "tensor descriptors require 16-byte aligned leading strides: "
            f"K={k} and N={n} must both be multiples of {TENSOR_DESCRIPTOR_ALIGNMENT}"
        )

    multiprocessors, capability = _device_facts(a.device)
    # Hopper cannot flatten a warp-specialised persistent loop; the tagged
    # tutorial hard-codes the same rule at 09-persistent-matmul.py:578.
    flatten = not (config.warp_specialize and capability[0] == HOPPER_COMPUTE_CAPABILITY[0])
    if config.epilogue_subtile and not flatten:
        raise ValueError(f"config {config.key} subtiles an epilogue that cannot be flattened")

    install_tma_allocator()
    output = torch.empty((m, n), device=a.device, dtype=torch.float16)
    tiles = triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n)
    grid = (min(multiprocessors, tiles),)
    _hopper_matmul_kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        GROUP_M=config.group_m,
        NUM_SMS=multiprocessors,
        EPILOGUE_SUBTILE=config.epilogue_subtile,
        WARP_SPECIALIZE=config.warp_specialize,
        FLATTEN=flatten,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output


# --------------------------------------------------------------------------
# Split-K skinny-M GEMV
# --------------------------------------------------------------------------


@triton.jit
def _skinny_gemv_kernel(  # type: ignore[no-untyped-def]
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
    SPLIT_K: tl.constexpr,
):
    """Accumulate one ``K`` slice of a skinny ``C = A @ B`` without tensor cores."""
    program_n = tl.program_id(axis=0)
    program_k = tl.program_id(axis=1)
    program_m = tl.program_id(axis=2)

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    mask_m = offsets_m < m
    mask_n = offsets_n < n

    # Contiguous K chunks rather than a strided sweep: each program then walks a
    # single dense span of B, which is what the DRAM page and L2 want.
    k_tiles = tl.cdiv(k, BLOCK_K)
    tiles_per_split = tl.cdiv(k_tiles, SPLIT_K)
    first_tile = program_k * tiles_per_split
    last_tile = min(first_tile + tiles_per_split, k_tiles)

    base_k = first_tile * BLOCK_K
    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + (base_k + offsets_k)[None, :] * stride_ak
    b_ptrs = b_ptr + (base_k + offsets_k)[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tile in range(first_tile, last_tile):
        mask_k = tile * BLOCK_K + offsets_k < k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        # Arithmetic intensity here is below one, so the FP32 broadcast-reduce
        # costs nothing that the B stream is not already paying for.
        products = a.to(tl.float32)[:, :, None] * b.to(tl.float32)[None, :, :]
        accumulator += tl.sum(products, axis=1)  # type: ignore[attr-defined]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator.to(tl.float16), mask=mask_c)
    else:
        tl.atomic_add(  # type: ignore[attr-defined]
            c_ptrs,
            accumulator,
            mask=mask_c,
            sem="relaxed",
            scope="gpu",
        )


def skinny_gemv(a: torch.Tensor, b: torch.Tensor, config: SkinnyGemvConfig) -> torch.Tensor:
    """Multiply two CUDA FP16 matrices with the split-K skinny-``M`` candidate.

    With ``split_k == 1`` the kernel writes FP16 directly. Above it the partial
    sums land in an FP32 buffer through relaxed atomics and are narrowed once,
    which keeps the reduction in FP32 while the cross-program order stays free.
    """
    m, n, k = _validated_operands(a, b)
    device = a.device
    grid = (
        triton.cdiv(n, config.block_n),
        config.split_k,
        triton.cdiv(m, config.block_m),
    )
    accumulate = config.split_k > 1
    target = (
        torch.zeros((m, n), device=device, dtype=torch.float32)
        if accumulate
        else torch.empty((m, n), device=device, dtype=torch.float16)
    )
    _skinny_gemv_kernel[grid](
        a,
        b,
        target,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        target.stride(0),
        target.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        SPLIT_K=config.split_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return target.to(torch.float16) if accumulate else target


# --------------------------------------------------------------------------
# Correctness gate
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KernelValidation:
    """One kernel/config/workload correctness outcome."""

    kernel: str
    config_key: str
    workload_key: str
    correct: bool
    max_abs_error: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel": self.kernel,
            "config_key": self.config_key,
            "workload_key": self.workload_key,
            "correct": self.correct,
            "max_abs_error": self.max_abs_error,
            "error": self.error,
        }


def _reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the FP32-output reference the frozen gate compares against."""
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _check(
    kernel: str,
    config_key: str,
    workload: Workload,
    output: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> KernelValidation:
    if not bool(torch.isfinite(output).all().item()):
        return KernelValidation(
            kernel,
            config_key,
            workload.key,
            False,
            error="correctness check failed: non-finite output",
        )
    difference = torch.abs(output.to(torch.float32) - expected)
    max_abs_error = float(difference.max().item())
    if not _within_tolerance(difference, expected, atol=atol, rtol=rtol):
        return KernelValidation(
            kernel,
            config_key,
            workload.key,
            False,
            max_abs_error=max_abs_error,
            error=f"correctness check failed (atol={atol}, rtol={rtol})",
        )
    return KernelValidation(kernel, config_key, workload.key, True, max_abs_error=max_abs_error)


def _validated_candidates(
    hopper_configs: Sequence[HopperGemmConfig],
    gemv_configs: Sequence[SkinnyGemvConfig],
) -> tuple[tuple[HopperGemmConfig, ...], tuple[SkinnyGemvConfig, ...]]:
    hopper = tuple(hopper_configs)
    gemv = tuple(gemv_configs)
    if not hopper and not gemv:
        raise ProtocolError("candidate manifest must not be empty")
    if any(type(config) is not HopperGemmConfig for config in hopper):
        raise ProtocolError("hopper manifest must contain only HopperGemmConfig values")
    if any(type(config) is not SkinnyGemvConfig for config in gemv):
        raise ProtocolError("gemv manifest must contain only SkinnyGemvConfig values")
    for label, keys in (
        ("hopper", tuple(config.key for config in hopper)),
        ("gemv", tuple(config.key for config in gemv)),
    ):
        if len(set(keys)) != len(keys):
            raise ProtocolError(f"{label} manifest contains duplicate configurations")
    return hopper, gemv


@torch.inference_mode()
def validate_candidate_kernels(
    gpu: str | None = None,
    *,
    hopper_configs: Sequence[HopperGemmConfig] = HOPPER_GEMM_CONFIGS,
    gemv_configs: Sequence[SkinnyGemvConfig] = SKINNY_GEMV_CONFIGS,
    workloads: Sequence[Workload] | None = None,
    atol: int | float = 1e-2,
    rtol: int | float = 1e-2,
    tensor_seed: int = 0,
) -> list[KernelValidation]:
    """Check every candidate against an FP32 reference before anything is timed.

    Hardware identity is verified before a single tensor is allocated, matching
    the paid-GPU rule in ``CONTRIBUTING.md``. A configuration that raises is
    recorded as a failure and never retried; the caller decides what to do with
    the journal.
    """
    hopper, gemv = _validated_candidates(hopper_configs, gemv_configs)
    absolute_tolerance = finite_float(atol, context="atol", minimum=0)
    relative_tolerance = finite_float(rtol, context="rtol", minimum=0)
    seed = exact_int(tensor_seed, context="tensor_seed", minimum=0)
    if gpu is not None:
        nonblank_string(gpu, context="gpu")

    device = torch.device("cuda", torch.cuda.current_device())
    profile = get_hardware_profile(gpu, device)
    if gpu is not None:
        validate_hardware(profile, expectation_for_gpu(gpu))
    if hopper and profile.compute_capability < HOPPER_COMPUTE_CAPABILITY:
        raise ProtocolError(
            "the persistent tensor-descriptor kernel requires compute capability "
            f"{HOPPER_COMPUTE_CAPABILITY[0]}.{HOPPER_COMPUTE_CAPABILITY[1]} or newer, "
            f"observed {profile.compute_capability[0]}.{profile.compute_capability[1]}"
        )

    sample = validation_workloads() if workloads is None else tuple(workloads)
    if not sample:
        raise ProtocolError("validation workload manifest must not be empty")

    results: list[KernelValidation] = []
    for index, workload in enumerate(sample):
        torch.manual_seed(seed + index)
        a = torch.empty((workload.m, workload.k), device=device, dtype=torch.float16)
        b = torch.empty((workload.k, workload.n), device=device, dtype=torch.float16)
        a.uniform_(-1.0, 1.0)
        b.uniform_(-1.0, 1.0)
        expected = _reference(a, b)

        candidates: list[tuple[str, str, HopperGemmConfig | SkinnyGemvConfig]] = [
            ("hopper_matmul", config.key, config) for config in hopper
        ]
        if workload.m <= SKINNY_M_LIMIT:
            candidates.extend(("skinny_gemv", config.key, config) for config in gemv)

        for kernel, config_key, config in candidates:
            try:
                output = (
                    hopper_matmul(a, b, config)
                    if isinstance(config, HopperGemmConfig)
                    else skinny_gemv(a, b, config)
                )
                torch.cuda.synchronize(device)
            # A configuration that raises is journaled, never retried.
            except Exception as exc:
                results.append(
                    KernelValidation(
                        kernel,
                        config_key,
                        workload.key,
                        False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            results.append(
                _check(
                    kernel,
                    config_key,
                    workload,
                    output,
                    expected,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )
            )
            del output
        del expected, b, a

    return results


def assert_candidate_kernels_correct(
    gpu: str | None = None,
    *,
    hopper_configs: Sequence[HopperGemmConfig] = HOPPER_GEMM_CONFIGS,
    gemv_configs: Sequence[SkinnyGemvConfig] = SKINNY_GEMV_CONFIGS,
    workloads: Sequence[Workload] | None = None,
    atol: int | float = 1e-2,
    rtol: int | float = 1e-2,
    tensor_seed: int = 0,
) -> list[KernelValidation]:
    """Run :func:`validate_candidate_kernels` and raise on the first failing set.

    This is the call a paid benchmark entry point makes before it starts timing:
    a wrong kernel then costs one short validation pass instead of a full sweep.
    """
    results = validate_candidate_kernels(
        gpu,
        hopper_configs=hopper_configs,
        gemv_configs=gemv_configs,
        workloads=workloads,
        atol=atol,
        rtol=rtol,
        tensor_seed=tensor_seed,
    )
    failures = [result for result in results if not result.correct]
    if failures:
        detail = "; ".join(
            f"{failure.kernel}[{failure.config_key}]@{failure.workload_key}: {failure.error}"
            for failure in failures[:8]
        )
        raise ProtocolError(
            f"{len(failures)} of {len(results)} candidate kernel checks failed: {detail}"
        )
    return results
