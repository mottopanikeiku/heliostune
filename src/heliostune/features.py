"""Numerically bounded joint workload, hardware, and launch features."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from heliostune.configs import KernelConfig, Workload
from heliostune.schema import HardwareProfile

FEATURE_NAMES: tuple[str, ...] = (
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


def joint_features(
    workload: Workload,
    config: KernelConfig,
    hardware: HardwareProfile,
) -> NDArray[np.float64]:
    """Return a stable feature vector with every continuous value near ``[-1, 1]``."""

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

    values = (
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
    return np.asarray(values, dtype=np.float64)
