from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from heliostune.action_configs import (
    ALL_CANDIDATES,
    NON_SPLIT_CANDIDATES,
    SPLIT_K_CANDIDATES,
    ActionCandidate,
    SplitKConfig,
    candidates_for,
)
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, HopperGemmConfig, KernelConfig
from heliostune.errors import SchemaError


def test_action_enumeration_does_not_import_gpu_dependencies() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from heliostune.action_configs import candidates_for; "
                "from heliostune.configs import DEFAULT_WORKLOADS; "
                "assert all(candidates_for(w) for w in DEFAULT_WORKLOADS); "
                "assert 'torch' not in sys.modules; "
                "assert 'triton' not in sys.modules; "
                "assert 'heliostune.action_expansion' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0


def test_exact_non_split_candidates_are_frozen() -> None:
    tiles = tuple(c.config for c in NON_SPLIT_CANDIDATES if c.family == "tile")
    persistent = tuple(c.config for c in NON_SPLIT_CANDIDATES if c.family == "persistent")
    assert set(tiles) == {
        KernelConfig(128, 128, 64, 4, 5),
        KernelConfig(128, 256, 64, 8, 4),
        KernelConfig(256, 128, 64, 8, 4),
        KernelConfig(128, 128, 128, 8, 3),
    }
    assert set(persistent) == {
        HopperGemmConfig(128, 128, 64, 4, 4),
        HopperGemmConfig(128, 256, 64, 8, 3),
    }
    assert len(NON_SPLIT_CANDIDATES) == 6
    assert all(
        not c.epilogue_subtile and not c.warp_specialize
        for c in persistent
        if isinstance(c, HopperGemmConfig)
    )


def test_exact_tensor_core_split_candidates_are_frozen() -> None:
    assert {c.config for c in SPLIT_K_CANDIDATES} == {
        SplitKConfig(16, 64, 64, 4, 3, 4),
        SplitKConfig(16, 64, 64, 4, 3, 16),
        SplitKConfig(16, 128, 64, 4, 3, 4),
        SplitKConfig(16, 128, 64, 4, 3, 16),
    }
    assert len(SPLIT_K_CANDIDATES) == 4
    assert all(c.family == "split_k" for c in SPLIT_K_CANDIDATES)


def test_every_original_workload_gets_only_its_applicable_frozen_candidates() -> None:
    assert len(DEFAULT_WORKLOADS) == 96
    for workload in DEFAULT_WORKLOADS:
        candidates = candidates_for(workload)
        assert candidates is (ALL_CANDIDATES if workload.m <= 31 else NON_SPLIT_CANDIDATES)
        assert len(candidates) == (10 if workload.m <= 31 else 6)
        assert all(c in candidates for c in NON_SPLIT_CANDIDATES)
    edge = replace(DEFAULT_WORKLOADS[0], m=32)
    assert candidates_for(edge) is NON_SPLIT_CANDIDATES
    assert candidates_for(replace(edge, m=31)) is ALL_CANDIDATES


def test_candidate_ids_are_sorted_unique_and_disjoint_from_the_old_actions() -> None:
    keys = tuple(c.key for c in ALL_CANDIDATES)
    assert keys == tuple(sorted(keys))
    assert len(keys) == len(set(keys)) == 10
    assert set(keys).isdisjoint(c.key for c in DEFAULT_CONFIGS)
    for candidate in ALL_CANDIDATES:
        assert candidate.key == f"{candidate.family}:{candidate.config.key}"
        assert candidate.to_dict() == candidate.config.to_dict()


def test_candidate_and_config_are_immutable_and_serialization_is_a_copy() -> None:
    candidate = SPLIT_K_CANDIDATES[0]
    with pytest.raises(FrozenInstanceError):
        candidate.config = candidate.config  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        candidate.config.block_m = 32  # type: ignore[misc]
    serialized = candidate.to_dict()
    serialized["block_m"] = 32
    assert candidate.to_dict()["block_m"] == 16


@pytest.mark.parametrize(
    "field", ("block_m", "block_n", "block_k", "num_warps", "num_stages", "split_k")
)
@pytest.mark.parametrize("invalid", (True, 16.0, "16"))
def test_split_config_requires_exact_integers(field: str, invalid: object) -> None:
    with pytest.raises(SchemaError, match=f"split-K config {field} must be an integer"):
        replace(SplitKConfig(16, 64, 64, 4, 3, 4), **{field: invalid})


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("block_m", 8, "block_m must be at least 16"),
        ("block_n", 24, "block_n must be a power of two"),
        ("block_k", 8, "block_k must be at least 16"),
        ("num_warps", 3, "num_warps must be one of"),
        ("num_stages", 0, "num_stages must be at least 1"),
        ("split_k", 1, "split_k must be at least 2"),
        ("split_k", 3, "split_k must be a power of two"),
    ),
)
def test_split_config_rejects_unsupported_dimensions(
    field: str, invalid: int, message: str
) -> None:
    with pytest.raises(SchemaError, match=message):
        replace(SplitKConfig(16, 64, 64, 4, 3, 4), **{field: invalid})


def test_candidate_rejects_unknown_config() -> None:
    with pytest.raises(TypeError, match="candidate config must be"):
        ActionCandidate(object())  # type: ignore[arg-type]
