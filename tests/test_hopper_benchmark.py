from __future__ import annotations

import random

import pytest

from heliostune.configs import (
    DEFAULT_WORKLOADS,
    HOPPER_GEMM_CONFIGS,
    SKINNY_GEMV_CONFIGS,
    HopperGemmConfig,
    SkinnyGemvConfig,
)
from heliostune.errors import HeliostuneError, ProtocolError
from heliostune.schema import HardwareProfile

pytest.importorskip("torch")
pytest.importorskip("triton")
from heliostune import hopper_benchmark


def _cuda_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> int:
        raise AssertionError("CUDA was touched before request validation")

    monkeypatch.setattr("torch.cuda.current_device", fail)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("gpu", "H200", "requires gpu 'H100'"),
        ("gpu", " H100", "gpu must be nonblank with no surrounding whitespace"),
        ("bank", 1, "requires bank 0"),
        ("bank", True, "bank must be an integer"),
        ("warmup_ms", 0, "warmup_ms must be at least 1"),
        ("warmup_ms", 1.0, "warmup_ms must be an integer"),
        ("rep_ms", -1, "rep_ms must be at least 1"),
        ("rep_ms", True, "rep_ms must be an integer"),
        ("workload_keys", [], "workload_keys must be a tuple"),
        ("workload_keys", (), "workload_keys must not be empty"),
        (
            "workload_keys",
            (DEFAULT_WORKLOADS[0].key, DEFAULT_WORKLOADS[0].key),
            "workload_keys must be unique",
        ),
        ("workload_keys", ("not-a-workload",), "unknown workloads"),
    ],
)
def test_invalid_requests_are_rejected_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: object,
    message: str,
) -> None:
    _cuda_must_not_run(monkeypatch)
    arguments: dict[str, object] = {
        "gpu": "H100",
        "bank": 0,
        "warmup_ms": 25,
        "rep_ms": 100,
        "workload_keys": (DEFAULT_WORKLOADS[0].key,),
    }
    arguments[field] = invalid

    with pytest.raises(HeliostuneError, match=message):
        hopper_benchmark.benchmark_hopper_candidates(**arguments)  # type: ignore[arg-type]


def test_request_selection_preserves_canonical_workload_order() -> None:
    selected = (DEFAULT_WORKLOADS[70], DEFAULT_WORKLOADS[2], DEFAULT_WORKLOADS[41])
    warmup, repetition, workloads = hopper_benchmark.validate_hopper_benchmark_request(
        gpu="H100",
        bank=0,
        warmup_ms=25,
        rep_ms=100,
        workload_keys=tuple(workload.key for workload in selected),
    )

    assert (warmup, repetition) == (25, 100)
    assert workloads == (DEFAULT_WORKLOADS[2], DEFAULT_WORKLOADS[41], DEFAULT_WORKLOADS[70])


def test_tensor_seed_schedule_matches_full_legacy_shuffle_and_is_subset_stable() -> None:
    shuffled = list(DEFAULT_WORKLOADS)
    random.Random(0).shuffle(shuffled)
    expected = {workload.key: index for index, workload in enumerate(shuffled)}
    all_keys = tuple(workload.key for workload in DEFAULT_WORKLOADS)
    full_schedule = hopper_benchmark.tensor_seed_schedule(all_keys)

    assert full_schedule == {workload.key: expected[workload.key] for workload in DEFAULT_WORKLOADS}
    assert set(full_schedule.values()) == set(range(96))
    for workload in (DEFAULT_WORKLOADS[0], DEFAULT_WORKLOADS[37], DEFAULT_WORKLOADS[-1]):
        assert hopper_benchmark.tensor_seed_schedule((workload.key,)) == {
            workload.key: full_schedule[workload.key]
        }


def test_regime_config_selection_is_complete_sorted_and_disjoint() -> None:
    skinny = next(workload for workload in DEFAULT_WORKLOADS if workload.m <= 8)
    hopper = next(workload for workload in DEFAULT_WORKLOADS if workload.m > 8)
    skinny_configs = hopper_benchmark.configs_for_workload(skinny)
    hopper_configs = hopper_benchmark.configs_for_workload(hopper)

    assert hopper_benchmark.regime_for_workload(skinny) == "skinny_gemv"
    assert hopper_benchmark.regime_for_workload(hopper) == "hopper_gemm"
    assert len(skinny_configs) == 48 == len(SKINNY_GEMV_CONFIGS)
    assert len(hopper_configs) == 23 == len(HOPPER_GEMM_CONFIGS)
    assert all(type(config) is SkinnyGemvConfig for config in skinny_configs)
    assert all(type(config) is HopperGemmConfig for config in hopper_configs)
    assert [config.key for config in skinny_configs] == sorted(
        config.key for config in skinny_configs
    )
    assert [config.key for config in hopper_configs] == sorted(
        config.key for config in hopper_configs
    )


def test_expected_candidate_rows_are_exactly_3008() -> None:
    skinny_workloads = tuple(workload for workload in DEFAULT_WORKLOADS if workload.m <= 8)
    hopper_workloads = tuple(workload for workload in DEFAULT_WORKLOADS if workload.m > 8)

    assert len(DEFAULT_WORKLOADS) == 96
    assert len(skinny_workloads) == 32
    assert len(hopper_workloads) == 64
    assert len(skinny_workloads) * len(SKINNY_GEMV_CONFIGS) == 32 * 48 == 1536
    assert len(hopper_workloads) * len(HOPPER_GEMM_CONFIGS) == 64 * 23 == 1472
    assert 1536 + 1472 == 3008
    assert hopper_benchmark.expected_candidate_row_count(DEFAULT_WORKLOADS) == 3008


def test_protocol_payload_declares_the_full_cross_product() -> None:
    protocol = hopper_benchmark._protocol_payload(
        warmup_ms=25,
        rep_ms=100,
        workloads=DEFAULT_WORKLOADS,
    )

    assert protocol == {
        "warmup_ms": 25,
        "rep_ms": 100,
        "quantiles": [0.2, 0.5, 0.8],
        "candidate_policy": {
            "skinny_gemv": {
                "condition": "m <= 8",
                "config_set": "SKINNY_GEMV_CONFIGS",
                "config_count": 48,
            },
            "hopper_gemm": {
                "condition": "m > 8",
                "config_set": "HOPPER_GEMM_CONFIGS",
                "config_count": 23,
            },
        },
        "expected_workloads": 96,
        "expected_skinny_workloads": 32,
        "expected_hopper_workloads": 64,
        "expected_skinny_rows": 1536,
        "expected_hopper_rows": 1472,
        "expected_candidate_rows": 3008,
        "torch_measurements": 96,
    }


def test_hardware_identity_is_rejected_before_tensor_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_hardware = HardwareProfile(
        gpu="H100",
        device_name="NVIDIA H200",
        compute_capability=(9, 0),
        multiprocessor_count=132,
        total_memory_gb=80.0,
    )
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr(
        hopper_benchmark, "get_hardware_profile", lambda gpu, device: wrong_hardware
    )

    def allocation_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("tensor allocation occurred before H100 identity validation")

    monkeypatch.setattr("torch.empty", allocation_must_not_run)
    key = DEFAULT_WORKLOADS[0].key
    with pytest.raises(ProtocolError, match="hardware name mismatch"):
        hopper_benchmark.benchmark_hopper_candidates(
            gpu="H100",
            bank=0,
            warmup_ms=25,
            rep_ms=100,
            workload_keys=(key,),
        )
