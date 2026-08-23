"""Deterministic synthetic latency matrices for local smoke tests and examples."""

from __future__ import annotations

import math

import numpy as np

from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, KernelConfig, Workload
from heliostune.schema import HardwareProfile, Measurement


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


def synthetic_measurements(seed: int = 7, replicates: int = 3) -> list[Measurement]:
    """Generate a complete two-GPU matrix with transferable but shifted optima."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    measurements: list[Measurement] = []
    for hardware in SYNTHETIC_HARDWARE:
        throughput_tflops = 22.0 if hardware.gpu == "sim-source" else 28.0
        for replicate in range(replicates):
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
                            replicate=replicate,
                            latency_ms=latency_ms,
                            torch_latency_ms=torch_latency_ms,
                            correct=True,
                            max_abs_error=0.001,
                        )
                    )
    return measurements
