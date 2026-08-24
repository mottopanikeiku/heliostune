"""Verify and atomically assemble the exact historical four-GPU v2 JSONL bytes."""

from __future__ import annotations

import argparse
import hashlib
import io
from collections import Counter
from pathlib import Path

import zstandard

from heliostune.artifacts import write_bytes_atomic
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS
from heliostune.hardware import expectation_for_gpu, validate_hardware
from heliostune.replay import BenchmarkTable
from heliostune.schema import Measurement, read_jsonl

_REPO = Path(__file__).resolve().parents[1]
_L4_A10 = _REPO / "benchmarks/data/measurements.jsonl.zst"
_T4 = _REPO / "benchmarks/data/t4-measurements.jsonl.zst"
_DEFAULT_H100 = _REPO / "benchmarks/data/h100-measurements.jsonl.zst"
_PUBLISHED_FINAL = _REPO / "benchmarks/data/parhelion-v2-measurements.jsonl.zst"
_EXPECTED_COMPRESSED = {
    _L4_A10: "afe086002f25291d7ace321c61771a1fbb2eba6056422e7dda07455898c4bf72",
    _T4: "f00c2a1498937707c872f667cf210de5e7319218c7df56e05a15174441a36c00",
    _DEFAULT_H100: "91b94ec42a71b1832203a6c5cdedf0102cf8d04ae9992d739fffa5351b7ac206",
    _PUBLISHED_FINAL: "ed6ec6ee8c3b61b451ea1276fc6f3925e82f70b5e208e9195c924ef6acc7343f",
}
_EXPECTED_H100_RAW = "747f30a97711e549c886aedf5a93d4386d53def7c65f93f3ef5b8dd112bc1dd8"
_EXPECTED_FINAL_RAW = "f417bd7e8167d277e39678266c84e405bbf7606485b916e363c0feb7d418be5d"
_EXPECTED_PER_GPU = 96 * 36 * 3


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_payload(path: Path, *, expected_compressed: str | None = None) -> bytes:
    if expected_compressed is not None and _sha256(path) != expected_compressed:
        raise ValueError(f"compressed checksum mismatch: {path}")
    try:
        with (
            path.open("rb") as source,
            zstandard.ZstdDecompressor().stream_reader(source) as decoded,
        ):
            payload = decoded.read()
    except (OSError, zstandard.ZstdError) as exc:
        raise ValueError(f"cannot decompress {path}: {exc}") from exc
    if not payload.endswith(b"\n"):
        raise ValueError(f"JSONL must end with a newline: {path}")
    return payload


def _strict_rows(payload: bytes, label: str) -> list[Measurement]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{label} JSONL is not UTF-8") from exc
    return read_jsonl(io.StringIO(text), source_name=label)


def _validate_grid(rows: list[Measurement], expected_gpus: tuple[str, ...], label: str) -> None:
    counts = Counter(row.hardware.gpu for row in rows)
    expected_counts = {gpu: _EXPECTED_PER_GPU for gpu in expected_gpus}
    if counts != expected_counts:
        raise ValueError(f"{label} GPU counts are {dict(counts)}, expected {expected_counts}")
    table = BenchmarkTable(rows)
    expected_workloads = {workload.key for workload in DEFAULT_WORKLOADS}
    expected_configs = {config.key for config in DEFAULT_CONFIGS}
    for gpu in expected_gpus:
        if {workload.key for workload in table.workloads(gpu)} != expected_workloads:
            raise ValueError(f"{label} does not contain the frozen 96-workload grid on {gpu}")
        if {config.key for config in table.configs(gpu)} != expected_configs:
            raise ValueError(f"{label} does not contain the frozen 36-config grid on {gpu}")
        table.validate_matrix(gpu, (0, 1, 2))
        validate_hardware(table.hardware(gpu), expectation_for_gpu(gpu))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100", type=Path, default=_DEFAULT_H100)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO / "artifacts/parhelion-final.jsonl",
    )
    args = parser.parse_args(argv)

    l4_a10_payload = _raw_payload(_L4_A10, expected_compressed=_EXPECTED_COMPRESSED[_L4_A10])
    t4_payload = _raw_payload(_T4, expected_compressed=_EXPECTED_COMPRESSED[_T4])
    expected_h100_compressed = _EXPECTED_COMPRESSED.get(args.h100.resolve())
    h100_payload = _raw_payload(args.h100, expected_compressed=expected_h100_compressed)
    if _sha256_bytes(h100_payload) != _EXPECTED_H100_RAW:
        raise ValueError("H100 uncompressed checksum does not match the frozen sole invocation")

    l4_a10_rows = _strict_rows(l4_a10_payload, "L4+A10")
    t4_rows = _strict_rows(t4_payload, "T4")
    h100_rows = _strict_rows(h100_payload, "H100")
    _validate_grid(l4_a10_rows, ("L4", "A10"), "L4+A10")
    _validate_grid(t4_rows, ("T4",), "T4")
    _validate_grid(h100_rows, ("H100",), "H100")

    final_payload = l4_a10_payload + t4_payload + h100_payload
    if _sha256_bytes(final_payload) != _EXPECTED_FINAL_RAW:
        raise ValueError("assembled archive checksum does not match the frozen four-GPU input")
    if _sha256(_PUBLISHED_FINAL) != _EXPECTED_COMPRESSED[_PUBLISHED_FINAL]:
        raise ValueError("published compressed final archive checksum changed")
    if _raw_payload(_PUBLISHED_FINAL) != final_payload:
        raise ValueError("published final archive bytes differ from exact assembly")

    write_bytes_atomic(args.output, final_payload)
    print(f"h100_sha256={_sha256_bytes(h100_payload)}")
    print(f"final_sha256={_sha256_bytes(final_payload)}")
    print(f"records={len(l4_a10_rows) + len(t4_rows) + len(h100_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
