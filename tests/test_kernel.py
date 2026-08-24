from __future__ import annotations

import pytest

from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, KernelConfig
from heliostune.errors import HeliostuneError

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
kernel = pytest.importorskip("heliostune.kernel")
_within_tolerance = kernel._within_tolerance
benchmark_measurements = kernel.benchmark_measurements


def _cuda_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> int:
        raise AssertionError("CUDA was touched before manifest validation")

    monkeypatch.setattr("torch.cuda.current_device", fail)


@pytest.mark.parametrize(
    "arguments",
    [
        {"bank": True},
        {"warmup_ms": float("nan")},
        {"rep_ms": float("inf")},
        {"atol": True},
        {"rtol": -1.0},
        {"configs": ()},
        {"configs": (DEFAULT_CONFIGS[0], DEFAULT_CONFIGS[0])},
        {"workloads": ()},
        {"workloads": (DEFAULT_WORKLOADS[0], DEFAULT_WORKLOADS[0])},
        {"workload_order_seed": True},
        {"config_order_seeds": {}},
        {"tensor_seeds": {DEFAULT_WORKLOADS[0].key: 1}},
    ],
)
def test_bad_benchmark_manifest_is_rejected_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
) -> None:
    _cuda_must_not_run(monkeypatch)
    with pytest.raises(HeliostuneError):
        benchmark_measurements(**arguments)


def test_forged_non_power_of_two_config_is_rejected_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_must_not_run(monkeypatch)
    config = object.__new__(KernelConfig)
    for name, value in {
        "block_m": 24,
        "block_n": 32,
        "block_k": 32,
        "num_warps": 4,
        "num_stages": 3,
        "group_m": 8,
    }.items():
        object.__setattr__(config, name, value)

    with pytest.raises(HeliostuneError, match="invalid block_m"):
        benchmark_measurements(configs=(config,))


def test_fp16_output_uses_fp32_difference_workspace_for_correctness() -> None:
    output = torch.tensor([1.0, 1.125], dtype=torch.float16)
    expected = torch.tensor([1.0, 1.0], dtype=torch.float32)
    difference = torch.empty_like(expected)
    torch.sub(output, expected, out=difference)
    difference.abs_()

    assert _within_tolerance(difference, expected, atol=0.0, rtol=0.125)
    assert not _within_tolerance(difference, expected, atol=0.0, rtol=0.124)
