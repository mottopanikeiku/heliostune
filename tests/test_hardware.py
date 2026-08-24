from __future__ import annotations

from dataclasses import replace

import pytest

from heliostune.errors import ProtocolError
from heliostune.hardware import HARDWARE_EXPECTATIONS, expectation_for_gpu, validate_hardware
from heliostune.schema import HardwareProfile

_VALID_PROFILES = {
    "L4": HardwareProfile("L4", "NVIDIA L4", (8, 9), 58, 22.0),
    "A10": HardwareProfile("A10", "NVIDIA A10", (8, 6), 72, 22.0),
    "T4": HardwareProfile("T4", "Tesla T4", (7, 5), 40, 15.0),
    "H100": HardwareProfile("H100", "NVIDIA H100 80GB HBM3", (9, 0), 120, 80.0),
    "A100-80GB": HardwareProfile("A100-80GB", "NVIDIA A100-SXM4-80GB", (8, 0), 108, 80.0),
    "H200": HardwareProfile("H200", "NVIDIA H200", (9, 0), 132, 140.0),
}


def test_every_declared_fleet_accepts_its_exact_profile() -> None:
    assert set(HARDWARE_EXPECTATIONS) == set(_VALID_PROFILES)
    for gpu, profile in _VALID_PROFILES.items():
        validate_hardware(profile, expectation_for_gpu(gpu))


@pytest.mark.parametrize(
    ("gpu", "change", "message"),
    [
        ("L4", {"compute_capability": (8, 6)}, "capability mismatch"),
        ("A10", {"device_name": "NVIDIA A100"}, "name mismatch"),
        ("T4", {"total_memory_gb": 18.0}, "memory mismatch"),
        ("H100", {"device_name": "NVIDIA H200"}, "name mismatch"),
        ("A100-80GB", {"total_memory_gb": 40.0}, "memory mismatch"),
        ("H200", {"device_name": "NVIDIA B200"}, "name mismatch"),
    ],
)
def test_hardware_gate_rejects_wrong_runtime_identity(
    gpu: str,
    change: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ProtocolError, match=message):
        validate_hardware(
            replace(_VALID_PROFILES[gpu], **change),
            expectation_for_gpu(gpu),
        )


def test_hardware_gate_rejects_selector_substitution() -> None:
    with pytest.raises(ProtocolError, match="selector mismatch"):
        validate_hardware(_VALID_PROFILES["H100"], expectation_for_gpu("H200"))


def test_unknown_hardware_expectation_is_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="unknown hardware expectation"):
        expectation_for_gpu("RTX4090")
