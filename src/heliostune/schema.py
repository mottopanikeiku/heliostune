"""Portable strict JSONL schema for GPU benchmark observations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO, cast

from heliostune.artifacts import strict_json_dumps, strict_json_loads
from heliostune.configs import KernelConfig, Workload
from heliostune.errors import SchemaError
from heliostune.validation import (
    exact_bool,
    exact_fields,
    exact_int,
    exact_object,
    finite_float,
    integer_pair,
    nonblank_string,
    optional_finite_float,
    optional_nonblank_string,
)

FailureStage = Literal["compile", "correctness", "benchmark", "legacy_unknown"] | None

_HARDWARE_FIELDS = (
    "gpu",
    "device_name",
    "compute_capability",
    "multiprocessor_count",
    "total_memory_gb",
    "cuda_version",
    "torch_version",
    "triton_version",
)
_V1_FIELDS = (
    "schema_version",
    "hardware",
    "workload",
    "config",
    "replicate",
    "latency_ms",
    "torch_latency_ms",
    "correct",
    "max_abs_error",
    "latency_p20_ms",
    "latency_p80_ms",
    "compile_ms",
    "error",
)
_V2_FIELDS = (
    "schema_version",
    "hardware",
    "workload",
    "config",
    "bank",
    "latency_ms",
    "torch_latency_ms",
    "correct",
    "max_abs_error",
    "latency_p20_ms",
    "latency_p80_ms",
    "torch_latency_p20_ms",
    "torch_latency_p80_ms",
    "compile_ms",
    "benchmark_wall_ms",
    "torch_benchmark_wall_ms",
    "failure_stage",
    "error",
)
_FAILURE_STAGES = {"compile", "correctness", "benchmark", "legacy_unknown"}


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Runtime properties used to condition the tuning model."""

    gpu: str
    device_name: str
    compute_capability: tuple[int, int]
    multiprocessor_count: int
    total_memory_gb: float
    cuda_version: str | None = None
    torch_version: str | None = None
    triton_version: str | None = None

    def __post_init__(self) -> None:
        nonblank_string(self.gpu, context="hardware gpu")
        nonblank_string(self.device_name, context="hardware device_name")
        if type(self.compute_capability) is not tuple or len(self.compute_capability) != 2:
            raise SchemaError("hardware compute_capability must be a two-element integer tuple")
        for index, component in enumerate(self.compute_capability):
            exact_int(component, context=f"hardware compute_capability[{index}]", minimum=0)
        exact_int(
            self.multiprocessor_count,
            context="hardware multiprocessor_count",
            minimum=1,
        )
        finite_float(
            self.total_memory_gb,
            context="hardware total_memory_gb",
            strictly_positive=True,
        )
        optional_nonblank_string(self.cuda_version, context="hardware cuda_version")
        optional_nonblank_string(self.torch_version, context="hardware torch_version")
        optional_nonblank_string(self.triton_version, context="hardware triton_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu": self.gpu,
            "device_name": self.device_name,
            "compute_capability": list(self.compute_capability),
            "multiprocessor_count": self.multiprocessor_count,
            "total_memory_gb": self.total_memory_gb,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> HardwareProfile:
        data = exact_fields(value, required=_HARDWARE_FIELDS, context="hardware profile")
        return cls(
            gpu=nonblank_string(data["gpu"], context="hardware gpu"),
            device_name=nonblank_string(data["device_name"], context="hardware device_name"),
            compute_capability=integer_pair(
                data["compute_capability"], context="hardware compute_capability"
            ),
            multiprocessor_count=exact_int(
                data["multiprocessor_count"],
                context="hardware multiprocessor_count",
                minimum=1,
            ),
            total_memory_gb=finite_float(
                data["total_memory_gb"],
                context="hardware total_memory_gb",
                strictly_positive=True,
            ),
            cuda_version=optional_nonblank_string(
                data["cuda_version"], context="hardware cuda_version"
            ),
            torch_version=optional_nonblank_string(
                data["torch_version"], context="hardware torch_version"
            ),
            triton_version=optional_nonblank_string(
                data["triton_version"], context="hardware triton_version"
            ),
        )


def _validate_quantiles(
    *,
    label: str,
    low: float | None,
    median: float | None,
    high: float | None,
) -> None:
    if (low is None) != (high is None):
        raise SchemaError(f"{label} p20 and p80 must be present together")
    if low is None:
        return
    assert high is not None
    if median is None:
        raise SchemaError(f"{label} quantiles require a median")
    if not low <= median <= high:
        raise SchemaError(f"{label} quantiles must satisfy p20 <= median <= p80")


@dataclass(frozen=True, slots=True)
class Measurement:
    """One configuration/workload measurement, including explicit failures."""

    hardware: HardwareProfile
    workload: Workload
    config: KernelConfig
    latency_ms: float | None
    torch_latency_ms: float
    correct: bool
    bank: int = 0
    max_abs_error: float | None = None
    error: str | None = None
    latency_p20_ms: float | None = None
    latency_p80_ms: float | None = None
    torch_latency_p20_ms: float | None = None
    torch_latency_p80_ms: float | None = None
    compile_ms: float | None = None
    benchmark_wall_ms: float | None = None
    torch_benchmark_wall_ms: float | None = None
    failure_stage: FailureStage = None

    def __post_init__(self) -> None:
        exact_int(self.bank, context="measurement bank", minimum=0)
        exact_bool(self.correct, context="measurement correct")
        latency = optional_finite_float(
            self.latency_ms,
            context="measurement latency_ms",
            strictly_positive=True,
        )
        torch_latency = finite_float(
            self.torch_latency_ms,
            context="measurement torch_latency_ms",
            strictly_positive=True,
        )
        max_abs_error = optional_finite_float(
            self.max_abs_error,
            context="measurement max_abs_error",
            minimum=0,
        )
        latency_p20 = optional_finite_float(
            self.latency_p20_ms,
            context="measurement latency_p20_ms",
            strictly_positive=True,
        )
        latency_p80 = optional_finite_float(
            self.latency_p80_ms,
            context="measurement latency_p80_ms",
            strictly_positive=True,
        )
        torch_latency_p20 = optional_finite_float(
            self.torch_latency_p20_ms,
            context="measurement torch_latency_p20_ms",
            strictly_positive=True,
        )
        torch_latency_p80 = optional_finite_float(
            self.torch_latency_p80_ms,
            context="measurement torch_latency_p80_ms",
            strictly_positive=True,
        )
        optional_finite_float(
            self.compile_ms,
            context="measurement compile_ms",
            minimum=0,
        )
        optional_finite_float(
            self.benchmark_wall_ms,
            context="measurement benchmark_wall_ms",
            strictly_positive=True,
        )
        optional_finite_float(
            self.torch_benchmark_wall_ms,
            context="measurement torch_benchmark_wall_ms",
            strictly_positive=True,
        )
        error = optional_nonblank_string(self.error, context="measurement error")
        if self.failure_stage is not None:
            nonblank_string(self.failure_stage, context="measurement failure_stage")
            if self.failure_stage not in _FAILURE_STAGES:
                raise SchemaError(f"unknown measurement failure_stage {self.failure_stage!r}")

        _validate_quantiles(
            label="Triton latency",
            low=latency_p20,
            median=latency,
            high=latency_p80,
        )
        _validate_quantiles(
            label="torch latency",
            low=torch_latency_p20,
            median=torch_latency,
            high=torch_latency_p80,
        )

        if self.correct:
            if latency is None:
                raise SchemaError("a successful measurement requires a Triton latency")
            if error is not None:
                raise SchemaError("a successful measurement must not contain an error")
            if self.failure_stage is not None:
                raise SchemaError("a successful measurement must not contain a failure stage")
        else:
            if latency is not None:
                raise SchemaError("a failed measurement must not contain a Triton latency")
            if error is None:
                raise SchemaError("a failed measurement requires a nonblank error")
            if self.failure_stage not in _FAILURE_STAGES:
                raise SchemaError("a failed measurement requires a classified failure stage")

        # Keep the validated values live so static analyzers do not mistake the
        # validation calls above for unused conversions.
        _ = max_abs_error

    @property
    def usable(self) -> bool:
        return self.correct and self.latency_ms is not None

    @property
    def correctness_classified(self) -> bool:
        return self.failure_stage in {None, "compile", "correctness", "benchmark"}

    def to_dict(self) -> dict[str, object]:
        if self.failure_stage == "legacy_unknown":
            raise SchemaError(
                "a schema-v1 failure must be explicitly classified before serialization"
            )
        return {
            "schema_version": 2,
            "hardware": self.hardware.to_dict(),
            "workload": self.workload.to_dict(),
            "config": self.config.to_dict(),
            "bank": self.bank,
            "latency_ms": self.latency_ms,
            "torch_latency_ms": self.torch_latency_ms,
            "correct": self.correct,
            "max_abs_error": self.max_abs_error,
            "latency_p20_ms": self.latency_p20_ms,
            "latency_p80_ms": self.latency_p80_ms,
            "torch_latency_p20_ms": self.torch_latency_p20_ms,
            "torch_latency_p80_ms": self.torch_latency_p80_ms,
            "compile_ms": self.compile_ms,
            "benchmark_wall_ms": self.benchmark_wall_ms,
            "torch_benchmark_wall_ms": self.torch_benchmark_wall_ms,
            "failure_stage": self.failure_stage,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: object) -> Measurement:
        outer = exact_object(value, context="measurement")
        if "schema_version" not in outer:
            raise SchemaError("measurement has missing field 'schema_version'")
        schema_version = exact_int(outer["schema_version"], context="measurement schema_version")
        if schema_version == 1:
            data = exact_fields(outer, required=_V1_FIELDS, context="schema-v1 measurement")
        elif schema_version == 2:
            data = exact_fields(outer, required=_V2_FIELDS, context="schema-v2 measurement")
        else:
            raise SchemaError(f"unsupported measurement schema version {schema_version!r}")

        correct = exact_bool(data["correct"], context="measurement correct")
        if schema_version == 1:
            bank = exact_int(data["replicate"], context="measurement replicate", minimum=0)
            torch_latency_p20_ms = None
            torch_latency_p80_ms = None
            benchmark_wall_ms = None
            torch_benchmark_wall_ms = None
            failure_stage: FailureStage = None if correct else "legacy_unknown"
        else:
            bank = exact_int(data["bank"], context="measurement bank", minimum=0)
            torch_latency_p20_ms = optional_finite_float(
                data["torch_latency_p20_ms"],
                context="measurement torch_latency_p20_ms",
                strictly_positive=True,
            )
            torch_latency_p80_ms = optional_finite_float(
                data["torch_latency_p80_ms"],
                context="measurement torch_latency_p80_ms",
                strictly_positive=True,
            )
            benchmark_wall_ms = optional_finite_float(
                data["benchmark_wall_ms"],
                context="measurement benchmark_wall_ms",
                strictly_positive=True,
            )
            torch_benchmark_wall_ms = optional_finite_float(
                data["torch_benchmark_wall_ms"],
                context="measurement torch_benchmark_wall_ms",
                strictly_positive=True,
            )
            raw_stage = data["failure_stage"]
            if raw_stage is None:
                failure_stage = None
            else:
                parsed_stage = nonblank_string(raw_stage, context="measurement failure_stage")
                if parsed_stage not in {"compile", "correctness", "benchmark"}:
                    raise SchemaError(
                        "schema-v2 failure_stage must be compile, correctness, or benchmark"
                    )
                failure_stage = cast(FailureStage, parsed_stage)

        return cls(
            hardware=HardwareProfile.from_dict(data["hardware"]),
            workload=Workload.from_dict(data["workload"]),
            config=KernelConfig.from_dict(data["config"]),
            bank=bank,
            latency_ms=optional_finite_float(
                data["latency_ms"],
                context="measurement latency_ms",
                strictly_positive=True,
            ),
            torch_latency_ms=finite_float(
                data["torch_latency_ms"],
                context="measurement torch_latency_ms",
                strictly_positive=True,
            ),
            correct=correct,
            max_abs_error=optional_finite_float(
                data["max_abs_error"],
                context="measurement max_abs_error",
                minimum=0,
            ),
            latency_p20_ms=optional_finite_float(
                data["latency_p20_ms"],
                context="measurement latency_p20_ms",
                strictly_positive=True,
            ),
            latency_p80_ms=optional_finite_float(
                data["latency_p80_ms"],
                context="measurement latency_p80_ms",
                strictly_positive=True,
            ),
            torch_latency_p20_ms=torch_latency_p20_ms,
            torch_latency_p80_ms=torch_latency_p80_ms,
            compile_ms=optional_finite_float(
                data["compile_ms"],
                context="measurement compile_ms",
                minimum=0,
            ),
            benchmark_wall_ms=benchmark_wall_ms,
            torch_benchmark_wall_ms=torch_benchmark_wall_ms,
            failure_stage=failure_stage,
            error=optional_nonblank_string(data["error"], context="measurement error"),
        )


def write_jsonl(measurements: Iterable[Measurement], destination: TextIO) -> None:
    for measurement in measurements:
        destination.write(strict_json_dumps(measurement.to_dict(), compact=True))
        destination.write("\n")


def read_jsonl(
    source: TextIO,
    *,
    source_name: str | Path | None = None,
) -> list[Measurement]:
    measurements: list[Measurement] = []
    name = source_name if source_name is not None else getattr(source, "name", "<jsonl>")
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            raise SchemaError(f"{name}:{line_number}: blank JSONL records are not permitted")
        decoded = strict_json_loads(line, source=name, line_number=line_number)
        try:
            measurements.append(Measurement.from_dict(decoded))
        except SchemaError as exc:
            raise SchemaError(f"{name}:{line_number}: invalid measurement: {exc}") from exc
    return measurements
