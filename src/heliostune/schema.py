"""Portable JSONL schema for GPU benchmark observations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, TextIO

from heliostune.configs import KernelConfig, Workload


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Runtime properties used to condition the tuning model."""

    gpu: str
    device_name: str
    compute_capability: tuple[int, int]
    multiprocessor_count: int
    total_memory_gb: float

    def __post_init__(self) -> None:
        if not self.gpu or not self.device_name:
            raise ValueError("gpu and device_name must not be empty")
        if self.multiprocessor_count <= 0 or self.total_memory_gb <= 0:
            raise ValueError("hardware capacity values must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["compute_capability"] = list(self.compute_capability)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HardwareProfile:
        capability = value["compute_capability"]
        return cls(
            gpu=str(value["gpu"]),
            device_name=str(value["device_name"]),
            compute_capability=(int(capability[0]), int(capability[1])),
            multiprocessor_count=int(value["multiprocessor_count"]),
            total_memory_gb=float(value["total_memory_gb"]),
        )


@dataclass(frozen=True, slots=True)
class Measurement:
    """One configuration/workload measurement, including explicit failures."""

    hardware: HardwareProfile
    workload: Workload
    config: KernelConfig
    latency_ms: float | None
    torch_latency_ms: float
    correct: bool
    max_abs_error: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.latency_ms is not None and self.latency_ms <= 0:
            raise ValueError("latency_ms must be positive when present")
        if self.torch_latency_ms <= 0:
            raise ValueError("torch_latency_ms must be positive")
        if self.correct and self.latency_ms is None:
            raise ValueError("a correct measurement requires a latency")
        if not self.correct and not self.error:
            raise ValueError("a failed measurement requires an error")

    @property
    def usable(self) -> bool:
        return self.correct and self.latency_ms is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "hardware": self.hardware.to_dict(),
            "workload": self.workload.to_dict(),
            "config": self.config.to_dict(),
            "latency_ms": self.latency_ms,
            "torch_latency_ms": self.torch_latency_ms,
            "correct": self.correct,
            "max_abs_error": self.max_abs_error,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Measurement:
        if value.get("schema_version") != 1:
            raise ValueError(f"unsupported schema version: {value.get('schema_version')!r}")
        return cls(
            hardware=HardwareProfile.from_dict(value["hardware"]),
            workload=Workload.from_dict(value["workload"]),
            config=KernelConfig.from_dict(value["config"]),
            latency_ms=(None if value["latency_ms"] is None else float(value["latency_ms"])),
            torch_latency_ms=float(value["torch_latency_ms"]),
            correct=bool(value["correct"]),
            max_abs_error=(
                None if value.get("max_abs_error") is None else float(value["max_abs_error"])
            ),
            error=(None if value.get("error") is None else str(value["error"])),
        )


def write_jsonl(measurements: Iterable[Measurement], destination: TextIO) -> None:
    for measurement in measurements:
        destination.write(json.dumps(measurement.to_dict(), separators=(",", ":")))
        destination.write("\n")


def read_jsonl(source: TextIO) -> list[Measurement]:
    measurements: list[Measurement] = []
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            measurements.append(Measurement.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid measurement on line {line_number}: {exc}") from exc
    return measurements
