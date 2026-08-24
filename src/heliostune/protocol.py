"""Frozen Parhelion v3 runtime, method, and deterministic seed contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

from heliostune.artifacts import read_json
from heliostune.errors import ProtocolError
from heliostune.validation import exact_fields, exact_int, exact_object, nonblank_string

V3_PRIOR_PRECISION = 1.0
V3_NOISE_VARIANCE = 0.05
V3_BANKS = (0, 1, 2, 3, 4)
V3_BUDGETS = tuple(range(1, 17))
V3_PRIMARY_BUDGETS = tuple(range(1, 9))
V3_VALIDATION_SEEDS = tuple(range(30))
V3_FINAL_SEEDS = tuple(range(50))
V3_K_GRID = (1, 3, 8, 16)
V3_TEMPERATURE_GRID = (0.2, 0.7, 2.0)
V3_TRANSFER_STRENGTH_GRID = (0.0, 0.02, 0.08, 0.2)
V3_SEED_PURPOSES = frozenset(
    {
        "tensor",
        "collector-workload-order",
        "collector-config-order",
        "replay-workload-order",
        "random-policy",
        "cold-thompson",
        "anchored-cold-thompson",
        "parhelion-thompson",
        "parhelion-no-anchor",
    }
)
V3_METHOD_ROLES: Mapping[str, str] = MappingProxyType(
    {
        "static_multisource": "zero_query",
        "random": "sequential",
        "single_source_nearest": "sequential",
        "multisource_retrieval": "sequential",
        "cold_thompson": "sequential",
        "anchored_cold_thompson": "sequential",
        "pooled_source_thompson": "sequential",
        "parhelion_thompson": "sequential",
        "parhelion_no_forced_anchor": "sequential",
        "parhelion_no_transfer": "sequential",
        "official_triton_config_exhaustive": "exhaustive",
        "torch": "external",
        "heldout_reference": "reference",
        "exhaustive": "exhaustive",
    }
)
V3_PILOT_WORKLOAD_KEYS = (
    "mistral-7b-attention-qkv-decode-1-m1-n6144-k4096",
    "mistral-7b-attention-qkv-decode-7-m7-n6144-k4096",
)
V3_PILOT_CONFIG_KEYS = (
    "m16n32k32-w4s3g8",
    "m16n64k32-w4s3g8",
    "m16n128k32-w4s3g8",
)


def v3_seed(
    *,
    purpose: str,
    gpu: str | None = None,
    bank: int | None = None,
    heldout_model: str | None = None,
    workload_key: str | None = None,
    policy_seed: int | None = None,
    round_index: int | None = None,
) -> int:
    """Return the first eight SHA-256 bytes of the exact v3 seed preimage."""
    if purpose not in V3_SEED_PURPOSES:
        raise ProtocolError(f"unknown Parhelion v3 seed purpose {purpose!r}")
    for name, text_value in (("gpu", gpu), ("heldout_model", heldout_model)):
        if text_value is not None:
            nonblank_string(text_value, context=f"v3 seed {name}")
    if workload_key is not None:
        nonblank_string(workload_key, context="v3 seed workload_key")
    for name, numeric_value in (
        ("bank", bank),
        ("policy_seed", policy_seed),
        ("round_index", round_index),
    ):
        if numeric_value is not None:
            exact_int(numeric_value, context=f"v3 seed {name}", minimum=0)
    fields = (
        "parhelion-v3",
        gpu or "na",
        "na" if bank is None else str(bank),
        heldout_model or "na",
        workload_key or "all",
        "na" if policy_seed is None else str(policy_seed),
        "na" if round_index is None else str(round_index),
        purpose,
    )
    preimage = "\0".join(fields).encode()
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big")


def load_v3_protocol(path: str | Path) -> dict[str, object]:
    value = exact_object(read_json(path), context="Parhelion v3 protocol")
    schema_version = exact_int(value.get("schema_version"), context="v3 schema_version")
    if schema_version != 1:
        raise ProtocolError(f"unsupported Parhelion v3 protocol schema {schema_version}")
    if value.get("study_id") not in {
        "parhelion-v3-development",
        "parhelion-v3-h200-freeze",
    }:
        raise ProtocolError(f"unexpected Parhelion v3 study_id {value.get('study_id')!r}")
    return value


def require_v3_runtime(protocol: Mapping[str, object]) -> None:
    """Reject a non-frozen analysis runtime before any campaign data access."""
    runtime = exact_fields(
        protocol.get("analysis_runtime"),
        required=("implementation", "python_major_minor", "numpy"),
        context="v3 analysis_runtime",
    )
    implementation = nonblank_string(runtime["implementation"], context="v3 runtime implementation")
    version = runtime["python_major_minor"]
    if type(version) is not list or len(version) != 2:
        raise ProtocolError("v3 python_major_minor must be a two-element array")
    expected_python = tuple(
        exact_int(component, context="v3 python_major_minor", minimum=0) for component in version
    )
    expected_numpy = nonblank_string(runtime["numpy"], context="v3 runtime numpy")
    actual_python = sys.version_info[:2]
    actual_implementation = platform.python_implementation()
    if actual_implementation != implementation or actual_python != expected_python:
        raise ProtocolError(
            f"Parhelion v3 requires {implementation} {expected_python[0]}.{expected_python[1]}.x; "
            f"running {actual_implementation} {actual_python[0]}.{actual_python[1]}"
        )
    if np.__version__ != expected_numpy:
        raise ProtocolError(
            f"Parhelion v3 requires numpy {expected_numpy}; running {np.__version__}"
        )


def runtime_manifest() -> dict[str, object]:
    """Return the full local analysis/package identity for campaign manifests."""
    distributions: dict[str, str | None] = {}
    for name in (
        "heliostune",
        "numpy",
        "rich",
        "zstandard",
        "torch",
        "triton",
        "modal",
    ):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "distributions": distributions,
    }


__all__ = [
    "V3_BANKS",
    "V3_BUDGETS",
    "V3_FINAL_SEEDS",
    "V3_K_GRID",
    "V3_METHOD_ROLES",
    "V3_NOISE_VARIANCE",
    "V3_PILOT_CONFIG_KEYS",
    "V3_PILOT_WORKLOAD_KEYS",
    "V3_PRIMARY_BUDGETS",
    "V3_PRIOR_PRECISION",
    "V3_SEED_PURPOSES",
    "V3_TEMPERATURE_GRID",
    "V3_TRANSFER_STRENGTH_GRID",
    "V3_VALIDATION_SEEDS",
    "load_v3_protocol",
    "require_v3_runtime",
    "runtime_manifest",
    "v3_seed",
]
