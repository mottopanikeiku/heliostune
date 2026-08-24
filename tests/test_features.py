from __future__ import annotations

import numpy as np
import pytest

from heliostune.configs import (
    DEFAULT_CONFIGS,
    DEFAULT_WORKLOADS,
    PARHELION_V3_CANDIDATE_CONFIGS,
    PARHELION_V3_OFFICIAL_CONFIG_KEYS,
)
from heliostune.features import (
    V2_FEATURE_NAMES,
    V3_FEATURE_NAMES,
    v2_joint_features,
    v3_joint_features,
)
from heliostune.schema import HardwareProfile

_PROFILES = (
    HardwareProfile("L4", "NVIDIA L4", (8, 9), 58, 22.0343017578125),
    HardwareProfile("A10", "NVIDIA A10", (8, 6), 72, 22.0582275390625),
    HardwareProfile("A100-80GB", "NVIDIA A100-SXM4-80GB", (8, 0), 108, 79.0),
    HardwareProfile("H200", "NVIDIA H200", (9, 0), 132, 140.0),
)
_REMOVED = {
    "log_m_over_n",
    "log_k_over_n",
    "log_flops",
    "n_divisible",
    "k_divisible",
}


def test_v3_candidate_manifest_is_exact_sorted_union() -> None:
    assert len(DEFAULT_CONFIGS) == 36
    assert len(PARHELION_V3_OFFICIAL_CONFIG_KEYS) == 16
    assert len(PARHELION_V3_CANDIDATE_CONFIGS) == 52
    assert tuple(config.key for config in PARHELION_V3_CANDIDATE_CONFIGS) == tuple(
        sorted(config.key for config in PARHELION_V3_CANDIDATE_CONFIGS)
    )
    assert {config.key for config in DEFAULT_CONFIGS}.issubset(
        {config.key for config in PARHELION_V3_CANDIDATE_CONFIGS}
    )
    assert (
        sum(
            config.key in PARHELION_V3_OFFICIAL_CONFIG_KEYS
            for config in PARHELION_V3_CANDIDATE_CONFIGS
        )
        == 16
    )


def test_v3_names_align_with_v2_values_after_declared_removals() -> None:
    workload = DEFAULT_WORKLOADS[17]
    config = PARHELION_V3_CANDIDATE_CONFIGS[23]
    profile = _PROFILES[-1]

    v2 = v2_joint_features(workload, config, profile)
    v3 = v3_joint_features(workload, config, profile)
    retained = [
        value for name, value in zip(V2_FEATURE_NAMES, v2, strict=True) if name not in _REMOVED
    ]

    assert len(V2_FEATURE_NAMES) == len(v2) == 25
    assert len(V3_FEATURE_NAMES) == len(v3) == 20
    assert tuple(name for name in V2_FEATURE_NAMES if name not in _REMOVED) == V3_FEATURE_NAMES
    np.testing.assert_array_equal(v3, retained)
    values = dict(zip(V2_FEATURE_NAMES, v2, strict=True))
    assert values["log_m_over_n"] == pytest.approx(values["log_m"] - values["log_n"])
    assert values["log_k_over_n"] == pytest.approx(values["log_k"] - values["log_n"])
    assert values["log_flops"] == pytest.approx(
        (values["log_m"] + values["log_n"] + values["log_k"]) / 3 + 1 / 42
    )


def test_removed_divisibility_columns_are_constant_for_declared_design() -> None:
    for workload in DEFAULT_WORKLOADS:
        for config in PARHELION_V3_CANDIDATE_CONFIGS:
            assert workload.n % config.block_n == 0
            assert workload.k % config.block_k == 0


def test_profile_aware_fold_ranks_are_exactly_18_19_20() -> None:
    for heldout_model in sorted({workload.model for workload in DEFAULT_WORKLOADS}):
        heldout_shapes = {
            (workload.m, workload.n, workload.k)
            for workload in DEFAULT_WORKLOADS
            if workload.model == heldout_model
        }
        eligible = tuple(
            workload
            for workload in DEFAULT_WORKLOADS
            if workload.model != heldout_model
            and (workload.m, workload.n, workload.k) not in heldout_shapes
        )
        ranks = []
        for profile_count in (2, 3, 4):
            matrix = np.stack(
                [
                    v3_joint_features(workload, config, profile)
                    for profile in _PROFILES[:profile_count]
                    for workload in eligible
                    for config in PARHELION_V3_CANDIDATE_CONFIGS
                ]
            )
            ranks.append(int(np.linalg.matrix_rank(matrix)))
        assert ranks == [18, 19, 20]
