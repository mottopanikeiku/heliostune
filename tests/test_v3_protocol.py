from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from heliostune.artifacts import read_json
from heliostune.errors import ProtocolError
from heliostune.protocol import (
    V3_METHOD_ROLES,
    V3_PILOT_CONFIG_KEYS,
    V3_PILOT_WORKLOAD_KEYS,
    load_v3_protocol,
    require_v3_runtime,
    v3_seed,
)

_REPO = Path(__file__).resolve().parents[1]
_PROTOCOL = _REPO / "benchmarks/parhelion-v3-development-protocol.json"
_SCRIPT = _REPO / "scripts/build_parhelion_v3_protocol.py"
_ASSEMBLER = _REPO / "scripts/assemble_parhelion_v3.py"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_parhelion_v3_protocol", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_assembler() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assemble_parhelion_v3", _ASSEMBLER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_seed_uses_exact_preimage_and_big_endian_prefix() -> None:
    preimage = "\0".join(
        (
            "parhelion-v3",
            "H200",
            "2",
            "mistral-7b",
            "workload-key",
            "17",
            "3",
            "parhelion-thompson",
        )
    ).encode()
    expected = int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big")

    assert (
        v3_seed(
            purpose="parhelion-thompson",
            gpu="H200",
            bank=2,
            heldout_model="mistral-7b",
            workload_key="workload-key",
            policy_seed=17,
            round_index=3,
        )
        == expected
    )
    default_preimage = b"parhelion-v3\0na\0na\0na\0all\0na\0na\0tensor"
    assert v3_seed(purpose="tensor") == int.from_bytes(
        hashlib.sha256(default_preimage).digest()[:8], "big"
    )
    with pytest.raises(ProtocolError, match="unknown.*purpose"):
        v3_seed(purpose="not-declared")


def test_v3_protocol_is_deterministic_and_serializes_exact_contract() -> None:
    builder = _load_builder()
    protocol = read_json(_PROTOCOL)

    rebuilt = builder.build_protocol()
    rebuilt["implementation_sha256"] = protocol["implementation_sha256"]
    assert rebuilt == protocol
    assert len(protocol["candidate_configs"]) == 52
    assert sum(row["official_source"] for row in protocol["candidate_configs"]) == 16
    assert len(protocol["workloads"]) == 96
    assert protocol["pilot"] == {
        "workload_keys": list(V3_PILOT_WORKLOAD_KEYS),
        "config_keys": list(V3_PILOT_CONFIG_KEYS),
        "cells": 6,
        "calls": 1,
    }
    assert {row["key"]: row["role"] for row in protocol["methods"]} == dict(V3_METHOD_ROLES)
    assert protocol["pruning"]["rank_gate_l4_a10"] == 18
    assert protocol["pruning"]["rank_gate_with_a100"] == 19
    assert protocol["pruning"]["rank_gate_with_h200"] == 20


def test_v3_assembly_uses_phase_specific_default_protocol(tmp_path: Path) -> None:
    assembler = _load_assembler()
    explicit = tmp_path / "explicit.json"

    assert assembler._phase_protocol("validation", None).name == (
        "parhelion-v3-development-protocol.json"
    )
    assert assembler._phase_protocol("final", None).name == "parhelion-v3-h200-freeze.json"
    assert assembler._phase_protocol("final", explicit) == explicit


def test_v3_runtime_gate_runs_before_campaign_data_access() -> None:
    protocol = load_v3_protocol(_PROTOCOL)
    if sys.version_info[:2] == (3, 11) and np.__version__ == "2.4.6":
        require_v3_runtime(protocol)
    else:
        with pytest.raises(ProtocolError, match="Parhelion v3 requires"):
            require_v3_runtime(protocol)
