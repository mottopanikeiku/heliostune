"""Deterministic synthetic latency matrices for local smoke tests and examples."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, KernelConfig, Workload
from heliostune.errors import SchemaError
from heliostune.schema import HardwareProfile, Measurement
from heliostune.validation import exact_int

SYNTHETIC_HARDWARE: tuple[HardwareProfile, ...] = (
    HardwareProfile("sim-source", "Synthetic source GPU", (8, 9), 60, 24.0),
    HardwareProfile("sim-target", "Synthetic target GPU", (8, 6), 72, 24.0),
)


def _configuration_efficiency(
    workload: Workload,
    config: KernelConfig,
    hardware: HardwareProfile,
) -> float:
    ideal_m = min(128, max(16, 2 ** round(math.log2(workload.m))))
    ideal_n = 64 if hardware.gpu == "sim-source" else 128
    if workload.projection == "ffn-up":
        ideal_n = 128
    ideal_warps = 4 if workload.m <= 32 else 8
    distance = (
        0.16 * abs(math.log2(config.block_m / ideal_m))
        + 0.10 * abs(math.log2(config.block_n / ideal_n))
        + 0.04 * abs(math.log2(config.block_k / 32))
        + 0.035 * abs(config.num_warps - ideal_warps)
        + 0.025 * abs(config.num_stages - 3)
    )
    return max(0.42, 0.96 - distance)


def synthetic_measurements(
    seed: int = 7,
    banks: Sequence[int] = (0, 1, 2),
) -> list[Measurement]:
    """Generate a complete two-GPU matrix with transferable but shifted optima."""
    validated_seed = exact_int(seed, context="synthetic seed", minimum=0)
    bank_order = tuple(banks)
    if not bank_order:
        raise SchemaError("synthetic banks must not be empty")
    validated_banks = tuple(
        exact_int(bank, context=f"synthetic banks[{index}]", minimum=0)
        for index, bank in enumerate(bank_order)
    )
    if len(set(validated_banks)) != len(validated_banks):
        raise SchemaError("synthetic banks must be unique")

    measurements: list[Measurement] = []
    for hardware_index, hardware in enumerate(SYNTHETIC_HARDWARE):
        throughput_tflops = 22.0 if hardware.gpu == "sim-source" else 28.0
        for bank in validated_banks:
            rng = np.random.default_rng(
                np.random.SeedSequence([validated_seed, hardware_index, bank])
            )
            for workload in DEFAULT_WORKLOADS:
                compute_floor_ms = workload.flops / (throughput_tflops * 1e12) * 1e3
                weight_traffic_ms = workload.k * workload.n * 2 / (420e9) * 1e3
                base_latency_ms = max(0.012, compute_floor_ms + weight_traffic_ms)
                torch_latency_ms = base_latency_ms / 0.78
                for config in DEFAULT_CONFIGS:
                    efficiency = _configuration_efficiency(workload, config, hardware)
                    stable_jitter = float(rng.normal(0.0, 0.006))
                    latency_ms = base_latency_ms / max(0.35, efficiency + stable_jitter)
                    measurements.append(
                        Measurement(
                            hardware=hardware,
                            workload=workload,
                            config=config,
                            bank=bank,
                            latency_ms=latency_ms,
                            torch_latency_ms=torch_latency_ms,
                            correct=True,
                            max_abs_error=0.001,
                            latency_p20_ms=latency_ms * 0.99,
                            latency_p80_ms=latency_ms * 1.01,
                            torch_latency_p20_ms=torch_latency_ms * 0.99,
                            torch_latency_p80_ms=torch_latency_ms * 1.01,
                            compile_ms=0.5 + latency_ms,
                            benchmark_wall_ms=100.0 + latency_ms,
                            torch_benchmark_wall_ms=100.0 + torch_latency_ms,
                        )
                    )
    return measurements
