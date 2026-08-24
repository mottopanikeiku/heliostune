"""Assemble the exact four-GPU Parhelion replay archive after H100 collection."""

from __future__ import annotations

import hashlib
import io
import math
import subprocess
from collections import Counter
from pathlib import Path

from heliostune.schema import Measurement, read_jsonl

_REPO = Path(__file__).resolve().parents[1]
_L4_A10 = _REPO / "benchmarks/data/measurements.jsonl.zst"
_T4 = _REPO / "artifacts/t4-measurements.jsonl"
_H100 = _REPO / "artifacts/h100-measurements.jsonl"
_OUTPUT = _REPO / "artifacts/parhelion-final.jsonl"
_EXPECTED_SOURCE_SHA256 = "afe086002f25291d7ace321c61771a1fbb2eba6056422e7dda07455898c4bf72"
_EXPECTED_T4_SHA256 = "f3910e73aec68d48015b0fb9d0469ff2be8c3210d5777dab61e4a9cdd660d217"
_EXPECTED_RECORDS_PER_GPU = 96 * 36 * 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decompress_source() -> bytes:
    if _sha256(_L4_A10) != _EXPECTED_SOURCE_SHA256:
        raise ValueError("L4+A10 compressed source checksum does not match the freeze")
    completed = subprocess.run(
        ["zstd", "-dc", str(_L4_A10)],
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def _read_payload(payload: bytes, label: str) -> list[Measurement]:
    if not payload.endswith(b"\n"):
        raise ValueError(f"{label} JSONL must end with a newline")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} JSONL is not UTF-8") from exc
    return read_jsonl(io.StringIO(text))


def _validate_chunk(
    measurements: list[Measurement],
    expected_gpus: tuple[str, ...],
    label: str,
) -> None:
    counts = Counter(measurement.hardware.gpu for measurement in measurements)
    expected_counts = {gpu: _EXPECTED_RECORDS_PER_GPU for gpu in expected_gpus}
    if counts != expected_counts:
        raise ValueError(
            f"{label} GPU record counts are {dict(counts)}, expected {expected_counts}"
        )

    keys: set[tuple[str, str, str, int]] = set()
    for measurement in measurements:
        key = (
            measurement.hardware.gpu,
            measurement.workload.key,
            measurement.config.key,
            measurement.replicate,
        )
        if key in keys:
            raise ValueError(f"duplicate {label} measurement {key}")
        keys.add(key)
        if measurement.replicate not in {0, 1, 2}:
            raise ValueError(f"unexpected {label} measurement bank {measurement.replicate}")
        if (
            not measurement.usable
            or measurement.latency_ms is None
            or not math.isfinite(measurement.latency_ms)
            or not math.isfinite(measurement.torch_latency_ms)
        ):
            raise ValueError(f"unusable or non-finite {label} measurement {key}")


def main() -> None:
    if _sha256(_T4) != _EXPECTED_T4_SHA256:
        raise ValueError("T4 checksum does not match the pre-H100 freeze")

    source_payload = _decompress_source()
    t4_payload = _T4.read_bytes()
    h100_payload = _H100.read_bytes()
    source_rows = _read_payload(source_payload, "L4+A10")
    t4_rows = _read_payload(t4_payload, "T4")
    h100_rows = _read_payload(h100_payload, "H100")
    _validate_chunk(source_rows, ("L4", "A10"), "L4+A10")
    _validate_chunk(t4_rows, ("T4",), "T4")
    _validate_chunk(h100_rows, ("H100",), "H100")

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_bytes(source_payload + t4_payload + h100_payload)
    print(f"h100_sha256={_sha256(_H100)}")
    print(f"final_sha256={_sha256(_OUTPUT)}")
    print(f"records={len(source_rows) + len(t4_rows) + len(h100_rows)}")


if __name__ == "__main__":
    main()
