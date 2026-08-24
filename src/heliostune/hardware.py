"""Pure hardware identity expectations for paid benchmark fleets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from heliostune.errors import ProtocolError
from heliostune.schema import HardwareProfile


@dataclass(frozen=True, slots=True)
class HardwareExpectation:
    """One allowed fleet identity and memory envelope."""

    gpu: str
    compute_capability: tuple[int, int]
    minimum_memory_gb: float
    maximum_memory_gb: float
    required_name_tokens: tuple[str, ...] = ()
    excluded_name_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.gpu or self.gpu != self.gpu.strip():
            raise ValueError("hardware expectation gpu must be nonblank without whitespace")
        if (
            type(self.compute_capability) is not tuple
            or len(self.compute_capability) != 2
            or any(
                type(component) is not int or component < 0 for component in self.compute_capability
            )
        ):
            raise ValueError("hardware expectation capability must be a non-negative integer pair")
        if (
            not math.isfinite(self.minimum_memory_gb)
            or not math.isfinite(self.maximum_memory_gb)
            or self.minimum_memory_gb <= 0
            or self.maximum_memory_gb < self.minimum_memory_gb
        ):
            raise ValueError("hardware expectation memory range must be finite and positive")
        for collection_name, tokens in (
            ("required_name_tokens", self.required_name_tokens),
            ("excluded_name_tokens", self.excluded_name_tokens),
        ):
            if any(
                type(token) is not str or not token or token != token.strip() for token in tokens
            ):
                raise ValueError(
                    f"hardware expectation {collection_name} contains an invalid token"
                )


HARDWARE_EXPECTATIONS: Mapping[str, HardwareExpectation] = MappingProxyType(
    {
        "L4": HardwareExpectation("L4", (8, 9), 20.0, 26.0, ("L4",)),
        "A10": HardwareExpectation("A10", (8, 6), 20.0, 26.0, ("A10",), ("A100",)),
        "T4": HardwareExpectation("T4", (7, 5), 13.0, 17.0, ("T4",)),
        "H100": HardwareExpectation("H100", (9, 0), 75.0, 85.0, ("H100",), ("H200",)),
        "A100-80GB": HardwareExpectation(
            "A100-80GB",
            (8, 0),
            75.0,
            85.0,
            ("A100",),
        ),
        "H200": HardwareExpectation(
            "H200",
            (9, 0),
            135.0,
            145.0,
            ("H200",),
            ("H100", "B200"),
        ),
    }
)


def expectation_for_gpu(gpu: str) -> HardwareExpectation:
    try:
        return HARDWARE_EXPECTATIONS[gpu]
    except KeyError as exc:
        choices = ", ".join(HARDWARE_EXPECTATIONS)
        raise ProtocolError(f"unknown hardware expectation {gpu!r}; choose from {choices}") from exc


def validate_hardware(
    profile: HardwareProfile,
    expectation: HardwareExpectation,
) -> None:
    """Reject a runtime profile that does not match its requested paid fleet."""
    context = f"requested {expectation.gpu}, observed {profile.device_name!r}"
    if profile.gpu != expectation.gpu:
        raise ProtocolError(
            f"hardware selector mismatch: expected {expectation.gpu!r}, got {profile.gpu!r}"
        )
    if profile.compute_capability != expectation.compute_capability:
        raise ProtocolError(
            f"hardware capability mismatch for {context}: expected "
            f"{expectation.compute_capability}, got {profile.compute_capability}"
        )
    if (
        not expectation.minimum_memory_gb
        <= profile.total_memory_gb
        <= expectation.maximum_memory_gb
    ):
        raise ProtocolError(
            f"hardware memory mismatch for {context}: expected "
            f"{expectation.minimum_memory_gb:g}-{expectation.maximum_memory_gb:g} GiB, "
            f"got {profile.total_memory_gb:.3f} GiB"
        )
    normalized_name = profile.device_name.upper()
    missing = [
        token for token in expectation.required_name_tokens if token.upper() not in normalized_name
    ]
    forbidden = [
        token for token in expectation.excluded_name_tokens if token.upper() in normalized_name
    ]
    if missing or forbidden:
        details: list[str] = []
        if missing:
            details.append(f"missing name token(s) {missing!r}")
        if forbidden:
            details.append(f"contains excluded token(s) {forbidden!r}")
        raise ProtocolError(f"hardware name mismatch for {context}: {'; '.join(details)}")
