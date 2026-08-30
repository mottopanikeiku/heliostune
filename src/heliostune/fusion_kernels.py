"""CPU-safe registry for the frozen native Triton fusion candidates.

The GPU implementation is imported only after an exact, closed entrypoint lookup.
Importing this module therefore does not require the optional ``torch`` or
``triton`` dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class RMSNormTritonConfig:
    """Compile-time configuration for one residual RMSNorm entrypoint."""

    config_id: str
    entrypoint: str
    block_size: int
    num_warps: int
    num_stages: int


RESIDUAL_RMSNORM_CONFIGS: tuple[RMSNormTritonConfig, ...] = (
    RMSNormTritonConfig(
        config_id="rmsnorm-triton-w4",
        entrypoint="heliostune_fusion_v2::residual_rmsnorm_w4",
        block_size=4096,
        num_warps=4,
        num_stages=1,
    ),
    RMSNormTritonConfig(
        config_id="rmsnorm-triton-w8",
        entrypoint="heliostune_fusion_v2::residual_rmsnorm_w8",
        block_size=4096,
        num_warps=8,
        num_stages=1,
    ),
    RMSNormTritonConfig(
        config_id="rmsnorm-triton-w16",
        entrypoint="heliostune_fusion_v2::residual_rmsnorm_w16",
        block_size=4096,
        num_warps=16,
        num_stages=1,
    ),
    RMSNormTritonConfig(
        config_id="rmsnorm-triton-w32",
        entrypoint="heliostune_fusion_v2::residual_rmsnorm_w32",
        block_size=4096,
        num_warps=32,
        num_stages=1,
    ),
)

RESIDUAL_RMSNORM_CONFIG_BY_ENTRYPOINT: Mapping[str, RMSNormTritonConfig] = MappingProxyType(
    {config.entrypoint: config for config in RESIDUAL_RMSNORM_CONFIGS}
)


def load_residual_rmsnorm(entrypoint: str) -> Callable[..., object]:
    """Load the GPU callable for one known residual RMSNorm entrypoint.

    Lookup deliberately precedes the optional-dependency import so untrusted suite
    text cannot select an import target and invalid names fail on CPU-only hosts.
    """
    try:
        config = RESIDUAL_RMSNORM_CONFIG_BY_ENTRYPOINT[entrypoint]
    except KeyError as exc:
        raise ValueError(f"unknown residual RMSNorm entrypoint: {entrypoint!r}") from exc

    from heliostune._fusion_gpu import load_residual_rmsnorm as load_gpu_residual_rmsnorm

    return load_gpu_residual_rmsnorm(config.entrypoint)
