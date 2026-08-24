"""Frozen v2 and profile-aware v3 workload/hardware/launch features."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from heliostune.configs import KernelConfig, Workload
from heliostune.schema import HardwareProfile

V2_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "log_m",
    "log_n",
    "log_k",
    "log_m_over_n",
    "log_k_over_n",
    "log_flops",
    "block_m",
    "block_n",
    "block_k",
    "warps",
    "stages",
    "group_m",
    "m_tile_coverage",
    "m_divisible",
    "n_divisible",
    "k_divisible",
    "multiprocessors",
    "memory_gb",
    "compute_capability",
    "m_x_block_m",
    "n_x_block_n",
    "k_x_block_k",
    "sm_x_warps",
    "memory_x_tile_area",
)

V3_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "log_m",
    "log_n",
    "log_k",
    "block_m",
    "block_n",
    "block_k",
    "warps",
    "stages",
    "group_m",
    "m_tile_coverage",
    "m_divisible",
    "multiprocessors",
    "memory_gb",
    "compute_capability",
    "m_x_block_m",
    "n_x_block_n",
    "k_x_block_k",
    "sm_x_warps",
    "memory_x_tile_area",
)
_V3_INDICES = (0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24)


def _v2_values(
    workload: Workload,
    config: KernelConfig,
    hardware: HardwareProfile,
) -> tuple[float, ...]:
    log_m = math.log2(workload.m) / 14.0
    log_n = math.log2(workload.n) / 14.0
    log_k = math.log2(workload.k) / 14.0
    block_m = math.log2(config.block_m) / 7.0
    block_n = math.log2(config.block_n) / 7.0
    block_k = math.log2(config.block_k) / 7.0
    sm_count = hardware.multiprocessor_count / 128.0
    memory_gb = hardware.total_memory_gb / 48.0
    compute_capability = (hardware.compute_capability[0] + hardware.compute_capability[1] / 10) / 10
    tile_area = (config.block_m * config.block_n) / (128.0 * 128.0)
    return (
        1.0,
        log_m,
        log_n,
        log_k,
        (math.log2(workload.m) - math.log2(workload.n)) / 14.0,
        (math.log2(workload.k) - math.log2(workload.n)) / 14.0,
        math.log2(workload.flops) / 42.0,
        block_m,
        block_n,
        block_k,
        math.log2(config.num_warps) / 3.0,
        config.num_stages / 5.0,
        config.group_m / 8.0,
        min(workload.m, config.block_m) / config.block_m,
        float(workload.m % config.block_m == 0),
        float(workload.n % config.block_n == 0),
        float(workload.k % config.block_k == 0),
        sm_count,
        memory_gb,
        compute_capability,
        log_m * block_m,
        log_n * block_n,
        log_k * block_k,
        sm_count * math.log2(config.num_warps) / 3.0,
        memory_gb * tile_area,
    )


def v2_joint_features(
    workload: Workload,
    config: KernelConfig,
    hardware: HardwareProfile,
) -> NDArray[np.float64]:
    """Return the immutable 25-column feature basis used by Parhelion v2."""
    return np.asarray(_v2_values(workload, config, hardware), dtype=np.float64)


def v3_joint_features(
    workload: Workload,
    config: KernelConfig,
    hardware: HardwareProfile,
) -> NDArray[np.float64]:
    """Return the 20-column v3 basis with exact affine/constant columns removed."""
    values = _v2_values(workload, config, hardware)
    return np.asarray(tuple(values[index] for index in _V3_INDICES), dtype=np.float64)


def v3_feature_rank(
    workloads: tuple[Workload, ...],
    configs: tuple[KernelConfig, ...],
    hardware_profiles: tuple[HardwareProfile, ...],
) -> int:
    if not workloads or not configs or not hardware_profiles:
        raise ValueError("v3 feature rank requires workloads, configs, and hardware profiles")
    matrix = np.stack(
        [
            v3_joint_features(workload, config, hardware)
            for hardware in hardware_profiles
            for workload in workloads
            for config in configs
        ]
    )
    return int(np.linalg.matrix_rank(matrix))
