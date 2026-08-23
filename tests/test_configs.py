from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, MODEL_SPECS


def test_frozen_manifests_are_unique_and_grouped() -> None:
    assert len(DEFAULT_CONFIGS) == 36
    assert len({config.key for config in DEFAULT_CONFIGS}) == len(DEFAULT_CONFIGS)
    assert len(DEFAULT_WORKLOADS) == 96
    assert len({workload.key for workload in DEFAULT_WORKLOADS}) == len(DEFAULT_WORKLOADS)
    assert {workload.model for workload in DEFAULT_WORKLOADS} == {
        model.name for model in MODEL_SPECS
    }
    assert all(
        sum(workload.model == model.name for workload in DEFAULT_WORKLOADS) == 24
        for model in MODEL_SPECS
    )


def test_workloads_cover_irregular_token_counts() -> None:
    irregular = [workload for workload in DEFAULT_WORKLOADS if workload.m & (workload.m - 1)]
    assert len(irregular) >= len(DEFAULT_WORKLOADS) // 2
    assert {workload.regime for workload in DEFAULT_WORKLOADS} == {
        "decode-1",
        "decode-7",
        "mixed-31",
        "mixed-96",
        "prefill-257",
        "prefill-1024",
    }
