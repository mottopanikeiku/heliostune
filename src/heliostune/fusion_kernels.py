"""CPU-safe registry for the frozen native Triton fusion candidates.

The GPU implementation is imported only after an exact, closed entrypoint lookup.
Importing this module therefore does not require the optional ``torch`` or
``triton`` dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Any


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

def _residual_rmsnorm_config(entrypoint: str) -> RMSNormTritonConfig:
    try:
        return RESIDUAL_RMSNORM_CONFIG_BY_ENTRYPOINT[entrypoint]
    except KeyError as exc:
        raise ValueError(f"unknown residual RMSNorm entrypoint: {entrypoint!r}") from exc



def load_residual_rmsnorm(entrypoint: str) -> Callable[..., object]:
    """Load the GPU callable for one known residual RMSNorm entrypoint.

    Lookup deliberately precedes the optional-dependency import so untrusted suite
    text cannot select an import target and invalid names fail on CPU-only hosts.
    """
    config = _residual_rmsnorm_config(entrypoint)

    from heliostune._fusion_gpu import load_residual_rmsnorm as load_gpu_residual_rmsnorm

    return load_gpu_residual_rmsnorm(config.entrypoint)


def compile_residual_rmsnorm(
    entrypoint: str, x: Any, residual: Any, gamma: Any
) -> object:
    """Compile one known native candidate without launching it."""
    config = _residual_rmsnorm_config(entrypoint)

    from heliostune._fusion_gpu import (
        compile_residual_rmsnorm as compile_gpu_residual_rmsnorm,
    )

    return compile_gpu_residual_rmsnorm(config, x, residual, gamma)


def _evidence_value(value: object, *, path: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) and key for key in value):
            raise ValueError(f"{path} must have non-empty string keys")
        return {
            key: _evidence_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items())
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _evidence_value(getattr(value, field.name), path=f"{path}.{field.name}")
            for field in fields(value)
        }
    as_dict = getattr(value, "_asdict", None)
    if callable(as_dict):
        return _evidence_value(as_dict(), path=path)
    if isinstance(value, (tuple, list)):
        return [_evidence_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{path} contains unsupported value {type(value).__name__}")


def _required_attribute(compiled: object, name: str) -> object:
    try:
        return getattr(compiled, name)
    except AttributeError as exc:
        raise ValueError(f"compiled kernel is missing {name!r}") from exc


def _non_empty_string_attribute(compiled: object, name: str) -> str:
    value = _required_attribute(compiled, name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"compiled kernel {name!r} must be a non-empty string")
    return value


def _non_negative_int_attribute(compiled: object, name: str, *, positive: bool = False) -> int:
    value = _required_attribute(compiled, name)
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"compiled kernel {name!r} must be a {qualifier} integer")
    return value


def compiled_kernel_evidence(
    compiled: object, config: RMSNormTritonConfig
) -> dict[str, Any]:
    """Return strict, deterministic evidence from a compiled Triton kernel."""
    registered = RESIDUAL_RMSNORM_CONFIG_BY_ENTRYPOINT.get(config.entrypoint)
    if registered != config:
        raise ValueError("config is not one of the four registered residual RMSNorm configs")
    metadata = _evidence_value(_required_attribute(compiled, "metadata"), path="metadata")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("compiled kernel metadata must be a non-empty record")
    target = metadata.get("target")
    if not isinstance(target, dict) or not target:
        raise ValueError("compiled kernel metadata target must be a non-empty record")
    if not isinstance(target.get("backend"), str) or not target["backend"]:
        raise ValueError("compiled kernel target backend must be a non-empty string")
    arch = target.get("arch")
    if (type(arch) is not int or arch <= 0) and (not isinstance(arch, str) or not arch):
        raise ValueError("compiled kernel target arch must be a positive integer or non-empty string")
    if type(target.get("warp_size")) is not int or target["warp_size"] <= 0:
        raise ValueError("compiled kernel target warp_size must be a positive integer")
    for name, minimum in (
        ("shared", 0),
        ("num_warps", 1),
        ("num_ctas", 1),
        ("num_stages", 1),
    ):
        value = metadata.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"compiled kernel metadata {name!r} must be an integer >= {minimum}"
            )
    if (
        metadata["num_warps"] != config.num_warps
        or metadata["num_stages"] != config.num_stages
    ):
        raise ValueError("compiled kernel metadata does not match the requested config")

    asm = _required_attribute(compiled, "asm")
    if not isinstance(asm, Mapping) or not asm:
        raise ValueError("compiled kernel asm must be a non-empty mapping")
    if not all(isinstance(stage, str) and stage for stage in asm):
        raise ValueError("compiled kernel asm stage names must be non-empty strings")
    asm_stages: list[dict[str, Any]] = []
    for stage, payload in sorted(asm.items()):
        if isinstance(payload, str):
            encoded = payload.encode("utf-8")
        elif isinstance(payload, bytes):
            encoded = payload
        else:
            raise ValueError(f"compiled kernel asm stage {stage!r} must be text or bytes")
        asm_stages.append(
            {
                "stage": stage,
                "bytes": len(encoded),
                "sha256": sha256(encoded).hexdigest(),
            }
        )

    n_regs = _non_negative_int_attribute(compiled, "n_regs")
    n_spills = _non_negative_int_attribute(compiled, "n_spills")
    n_max_threads = _non_negative_int_attribute(compiled, "n_max_threads", positive=True)

    return {
        "status": "compiled",
        "error": None,
        "kernel_name": _non_empty_string_attribute(compiled, "name"),
        "kernel_hash": _non_empty_string_attribute(compiled, "hash"),
        "target": target,
        "metadata": metadata,
        "n_regs": n_regs,
        "n_spills": n_spills,
        "n_max_threads": n_max_threads,
        "asm_stages": asm_stages,
        "resource_gate_passed": n_spills == 0,
    }
