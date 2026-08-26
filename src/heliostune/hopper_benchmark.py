"""Regime-aware one-bank H100 timing for the exploratory Hopper study."""

from __future__ import annotations

import random
from collections.abc import Sequence
from functools import partial
from typing import Literal, TypeAlias, cast

import torch

from heliostune.configs import (
    DEFAULT_WORKLOADS,
    HOPPER_GEMM_CONFIGS,
    SKINNY_GEMV_CONFIGS,
    HopperGemmConfig,
    SkinnyGemvConfig,
    Workload,
)
from heliostune.errors import ProtocolError
from heliostune.hardware import expectation_for_gpu, validate_hardware
from heliostune.hopper_kernel import SKINNY_M_LIMIT, hopper_matmul, skinny_gemv
from heliostune.kernel import _timed_do_bench, _within_tolerance, get_hardware_profile
from heliostune.validation import exact_int, nonblank_string

HOPPER_BENCHMARK_GPU = "H100"
HOPPER_BENCHMARK_BANK = 0
CORRECTNESS_ATOL = 1e-2
CORRECTNESS_RTOL = 1e-2
QUANTILES: tuple[float, float, float] = (0.2, 0.5, 0.8)

CandidateConfig: TypeAlias = HopperGemmConfig | SkinnyGemvConfig
Regime: TypeAlias = Literal["skinny_gemv", "hopper_gemm"]


def validate_hopper_benchmark_request(
    *,
    gpu: object,
    bank: object,
    warmup_ms: object,
    rep_ms: object,
    workload_keys: object,
) -> tuple[int, int, tuple[Workload, ...]]:
    """Strictly validate a remote timing request without touching CUDA."""
    selected_gpu = nonblank_string(gpu, context="gpu")
    if selected_gpu != HOPPER_BENCHMARK_GPU:
        raise ProtocolError(
            f"Hopper benchmark requires gpu {HOPPER_BENCHMARK_GPU!r}, got {selected_gpu!r}"
        )
    selected_bank = exact_int(bank, context="bank", minimum=0)
    if selected_bank != HOPPER_BENCHMARK_BANK:
        raise ProtocolError(
            f"Hopper benchmark requires bank {HOPPER_BENCHMARK_BANK}, got {selected_bank}"
        )
    warmup = exact_int(warmup_ms, context="warmup_ms", minimum=1)
    repetition = exact_int(rep_ms, context="rep_ms", minimum=1)
    if type(workload_keys) is not tuple:
        raise ProtocolError("workload_keys must be a tuple")
    raw_keys = cast(tuple[object, ...], workload_keys)
    if not raw_keys:
        raise ProtocolError("workload_keys must not be empty")
    if any(type(key) is not str or not key or key != key.strip() for key in raw_keys):
        raise ProtocolError("workload_keys must contain only nonblank strings without whitespace")
    if len(set(raw_keys)) != len(raw_keys):
        raise ProtocolError("workload_keys must be unique")

    requested = set(cast(tuple[str, ...], raw_keys))
    known = {workload.key for workload in DEFAULT_WORKLOADS}
    unknown = sorted(requested - known)
    if unknown:
        raise ProtocolError(f"workload_keys contain unknown workloads: {unknown}")
    workloads = tuple(workload for workload in DEFAULT_WORKLOADS if workload.key in requested)
    return warmup, repetition, workloads


def tensor_seed_schedule(
    workload_keys: tuple[str, ...], *, bank: int = HOPPER_BENCHMARK_BANK
) -> dict[str, int]:
    """Return benchmark_measurements-compatible seeds in canonical key order.

    The full frozen workload manifest is shuffled before the requested subset is
    filtered. Consequently a workload's tensors are identical whether it is
    benchmarked alone or as part of the complete study.
    """
    _, _, workloads = validate_hopper_benchmark_request(
        gpu=HOPPER_BENCHMARK_GPU,
        bank=bank,
        warmup_ms=1,
        rep_ms=1,
        workload_keys=workload_keys,
    )
    shuffled = list(DEFAULT_WORKLOADS)
    random.Random(bank).shuffle(shuffled)
    shuffled_indices = {workload.key: index for index, workload in enumerate(shuffled)}
    return {workload.key: bank * 10_000 + shuffled_indices[workload.key] for workload in workloads}


def regime_for_workload(workload: Workload) -> Regime:
    """Select the sole candidate regime allowed for a workload."""
    if type(workload) is not Workload:
        raise ProtocolError("candidate workload must be a Workload")
    return "skinny_gemv" if workload.m <= SKINNY_M_LIMIT else "hopper_gemm"


def configs_for_workload(workload: Workload) -> tuple[CandidateConfig, ...]:
    """Return every regime candidate in deterministic config-key order."""
    configs: Sequence[CandidateConfig] = (
        SKINNY_GEMV_CONFIGS
        if regime_for_workload(workload) == "skinny_gemv"
        else HOPPER_GEMM_CONFIGS
    )
    return tuple(sorted(configs, key=lambda config: config.key))


def expected_candidate_row_count(workloads: Sequence[Workload] = DEFAULT_WORKLOADS) -> int:
    """Return the exact regime-aware candidate cross-product size."""
    manifest = tuple(workloads)
    if any(type(workload) is not Workload for workload in manifest):
        raise ProtocolError("expected-count manifest must contain only Workload values")
    if len({workload.key for workload in manifest}) != len(manifest):
        raise ProtocolError("expected-count manifest contains duplicate workloads")
    return sum(len(configs_for_workload(workload)) for workload in manifest)


def _timing_payload(quantiles_ms: tuple[float, float, float], wall_ms: float) -> dict[str, float]:
    p20_ms, median_ms, p80_ms = quantiles_ms
    return {
        "p20_ms": p20_ms,
        "median_ms": median_ms,
        "p80_ms": p80_ms,
        "wall_ms": wall_ms,
    }


def _reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _candidate_output(
    a: torch.Tensor,
    b: torch.Tensor,
    config: CandidateConfig,
) -> torch.Tensor:
    if isinstance(config, SkinnyGemvConfig):
        return skinny_gemv(a, b, config)
    return hopper_matmul(a, b, config)


def _config_manifest() -> dict[str, list[dict[str, object]]]:
    return {
        "hopper_gemm": [dict(config.to_dict()) for config in HOPPER_GEMM_CONFIGS],
        "skinny_gemv": [dict(config.to_dict()) for config in SKINNY_GEMV_CONFIGS],
    }


def _protocol_payload(
    *, warmup_ms: int, rep_ms: int, workloads: tuple[Workload, ...]
) -> dict[str, object]:
    skinny_count = sum(workload.m <= SKINNY_M_LIMIT for workload in workloads)
    hopper_count = len(workloads) - skinny_count
    skinny_rows = skinny_count * len(SKINNY_GEMV_CONFIGS)
    hopper_rows = hopper_count * len(HOPPER_GEMM_CONFIGS)
    return {
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
        "quantiles": list(QUANTILES),
        "candidate_policy": {
            "skinny_gemv": {
                "condition": f"m <= {SKINNY_M_LIMIT}",
                "config_set": "SKINNY_GEMV_CONFIGS",
                "config_count": len(SKINNY_GEMV_CONFIGS),
            },
            "hopper_gemm": {
                "condition": f"m > {SKINNY_M_LIMIT}",
                "config_set": "HOPPER_GEMM_CONFIGS",
                "config_count": len(HOPPER_GEMM_CONFIGS),
            },
        },
        "expected_workloads": len(workloads),
        "expected_skinny_workloads": skinny_count,
        "expected_hopper_workloads": hopper_count,
        "expected_skinny_rows": skinny_rows,
        "expected_hopper_rows": hopper_rows,
        "expected_candidate_rows": skinny_rows + hopper_rows,
        "torch_measurements": len(workloads),
    }


@torch.inference_mode()
def benchmark_hopper_candidates(
    *,
    gpu: str,
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
) -> dict[str, object]:
    """Time every frozen candidate for the requested bank-0 workloads on H100.

    This remote engine deliberately returns no local provenance or verification
    claims. The durable caller validates the completed payload before binding it
    to a wheel, manifest, correctness gate, and local artifact.
    """
    warmup, repetition, workloads = validate_hopper_benchmark_request(
        gpu=gpu,
        bank=bank,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        workload_keys=workload_keys,
    )
    seeds = tensor_seed_schedule(tuple(workload.key for workload in workloads), bank=bank)

    # Paid-hardware identity is checked before the first tensor allocation.
    device = torch.device("cuda", torch.cuda.current_device())
    hardware = get_hardware_profile(gpu, device)
    validate_hardware(hardware, expectation_for_gpu(HOPPER_BENCHMARK_GPU))

    rows: list[dict[str, object]] = []
    for workload in workloads:
        seed = seeds[workload.key]
        try:
            torch.manual_seed(seed)
            a = torch.empty((workload.m, workload.k), device=device, dtype=torch.float16)
            b = torch.empty((workload.k, workload.n), device=device, dtype=torch.float16)
            a.uniform_(-1.0, 1.0)
            b.uniform_(-1.0, 1.0)
            expected = _reference(a, b)
            difference = torch.empty_like(expected)
            torch_quantiles, torch_wall_ms = _timed_do_bench(
                partial(torch.matmul, a, b),
                warmup_ms=float(warmup),
                rep_ms=float(repetition),
            )
        except Exception as exc:
            raise ProtocolError(
                f"workload setup failed for {workload.key}: {type(exc).__name__}: {exc}"
            ) from exc

        torch_payload = _timing_payload(torch_quantiles, torch_wall_ms)
        regime = regime_for_workload(workload)
        for config in configs_for_workload(workload):
            try:
                output = _candidate_output(a, b, config)
                torch.cuda.synchronize(device)
                finite = bool(torch.isfinite(output).all().item())
                torch.sub(output, expected, out=difference)
                difference.abs_()
                max_abs_error = float(difference.max().item())
                correct = finite and _within_tolerance(
                    difference,
                    expected,
                    atol=CORRECTNESS_ATOL,
                    rtol=CORRECTNESS_RTOL,
                )
            except Exception as exc:
                raise ProtocolError(
                    f"candidate execution failed for {workload.key}/{config.key}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not correct:
                reason = (
                    "non-finite output"
                    if not finite
                    else (f"outside atol={CORRECTNESS_ATOL}, rtol={CORRECTNESS_RTOL}")
                )
                raise ProtocolError(
                    f"candidate correctness failed for {workload.key}/{config.key}: "
                    f"{reason}; max_abs_error={max_abs_error}"
                )

            try:
                candidate_quantiles, candidate_wall_ms = _timed_do_bench(
                    partial(_candidate_output, a, b, config),
                    warmup_ms=float(warmup),
                    rep_ms=float(repetition),
                )
            except Exception as exc:
                raise ProtocolError(
                    f"candidate timing failed for {workload.key}/{config.key}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            rows.append(
                {
                    "workload_key": workload.key,
                    "workload": workload.to_dict(),
                    "regime": regime,
                    "config_kind": regime,
                    "config_key": config.key,
                    "config": config.to_dict(),
                    "bank": bank,
                    "seed": seed,
                    "latency": _timing_payload(candidate_quantiles, candidate_wall_ms),
                    "torch": dict(torch_payload),
                    "correct": True,
                    "max_abs_error": max_abs_error,
                }
            )
            del output
        del difference, expected, b, a

    expected_rows = expected_candidate_row_count(workloads)
    if len(rows) != expected_rows:
        raise ProtocolError(f"benchmark produced {len(rows)} rows, expected {expected_rows}")
    return {
        "hardware": hardware.to_dict(),
        "protocol": _protocol_payload(
            warmup_ms=warmup,
            rep_ms=repetition,
            workloads=workloads,
        ),
        "configs": _config_manifest(),
        "workloads": [workload.to_dict() for workload in workloads],
        "rows": rows,
    }
