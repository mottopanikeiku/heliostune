from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard

from heliostune.artifacts import (
    read_json,
    read_measurements,
    write_json_atomic,
    write_measurements_atomic,
    write_text_atomic,
)
from heliostune.configs import KernelConfig, Workload
from heliostune.errors import SchemaError
from heliostune.schema import HardwareProfile, Measurement


def _row() -> Measurement:
    return Measurement(
        hardware=HardwareProfile("L4", "NVIDIA L4", (8, 9), 58, 22.0),
        workload=Workload(1, 32, 32, "model", "projection", "decode-1"),
        config=KernelConfig(16, 32, 32, 4, 3),
        bank=0,
        latency_ms=1.0,
        torch_latency_ms=2.0,
        correct=True,
        latency_p20_ms=0.9,
        latency_p80_ms=1.1,
        torch_latency_p20_ms=1.9,
        torch_latency_p80_ms=2.1,
        compile_ms=3.0,
        benchmark_wall_ms=100.0,
        torch_benchmark_wall_ms=101.0,
    )


def test_measurements_round_trip_plain_and_zstd(tmp_path: Path) -> None:
    row = _row()
    plain = tmp_path / "rows.jsonl"
    compressed = tmp_path / "rows.jsonl.zst"

    write_measurements_atomic(plain, [row])
    write_measurements_atomic(compressed, [row])

    assert read_measurements(plain) == [row]
    assert read_measurements(compressed) == [row]
    frame = zstandard.get_frame_parameters(compressed.read_bytes())
    assert frame.has_checksum
    assert frame.content_size == zstandard.CONTENTSIZE_UNKNOWN


def test_zstd_output_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl.zst"
    second = tmp_path / "second.jsonl.zst"

    write_measurements_atomic(first, [_row()])
    write_measurements_atomic(second, [_row()])

    assert first.read_bytes() == second.read_bytes()


def test_failed_atomic_serialization_preserves_existing_output(tmp_path: Path) -> None:
    destination = tmp_path / "rows.jsonl"
    write_text_atomic(destination, "existing\n")
    legacy_failure = Measurement(
        hardware=_row().hardware,
        workload=_row().workload,
        config=_row().config,
        bank=0,
        latency_ms=None,
        torch_latency_ms=2.0,
        correct=False,
        error="legacy failure",
        failure_stage="legacy_unknown",
    )

    with pytest.raises(SchemaError, match="explicitly classified"):
        write_measurements_atomic(destination, [legacy_failure])

    assert destination.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_strict_json_rejects_duplicates_and_nan_with_path(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match=r"duplicate.json.*duplicate.*a"):
        read_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match=r"nonfinite.json.*non-finite"):
        read_json(nonfinite)


def test_atomic_json_is_sorted_finite_and_newline_terminated(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    write_json_atomic(destination, {"z": 1, "a": 2})

    payload = destination.read_text(encoding="utf-8")
    assert payload == json.dumps({"a": 2, "z": 1}, indent=2, sort_keys=True) + "\n"

    with pytest.raises(SchemaError, match="strict JSON"):
        write_json_atomic(destination, {"bad": float("nan")})
    assert destination.read_text(encoding="utf-8") == payload
