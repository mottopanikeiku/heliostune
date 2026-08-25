"""Preserve raw mixed A100 evidence and derive one explicit engineering domain."""

from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from heliostune.artifacts import (
    read_json,
    read_measurements,
    write_bytes_atomic,
    write_json_atomic,
    write_measurements_atomic,
)
from heliostune.protocol import load_v3_protocol, require_v3_runtime
from heliostune.schema import Measurement
from heliostune.v3_artifacts import sha256_file
from heliostune.validation import exact_object

_REPO = Path(__file__).resolve().parents[1]
_PROTOCOL = _REPO / "benchmarks/parhelion-v3-development-protocol.json"
_INPUT = _REPO / "benchmarks/data/parhelion-v3-validation.jsonl.zst"
_MANIFEST = Path(f"{_INPUT}.manifest.json")
_JOURNAL = Path(f"{_INPUT}.attempts.jsonl")
_RAW = _REPO / "benchmarks/data/parhelion-v3-validation-raw-mixed-a100.jsonl.zst"
_RAW_MANIFEST = Path(f"{_RAW}.source-manifest.json")
_RAW_JOURNAL = Path(f"{_RAW}.attempts.jsonl")
_SOURCE_DEVICE_NAMES = frozenset(
    {
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
    }
)
_CANONICAL_DEVICE_NAME = "NVIDIA A100 80GB (mixed PCIe/SXM engineering domain)"


def canonicalize_rows(
    rows: Sequence[Measurement],
) -> tuple[tuple[Measurement, ...], dict[str, int]]:
    """Canonicalize only A100 device names while preserving every measured value."""
    counts = Counter(row.hardware.device_name for row in rows if row.hardware.gpu == "A100-80GB")
    if set(counts) != _SOURCE_DEVICE_NAMES:
        raise ValueError(
            "engineering normalization requires both exact A100 PCIe and SXM source names; "
            f"observed {dict(sorted(counts.items()))}"
        )
    normalized: list[Measurement] = []
    for row in rows:
        if row.hardware.gpu != "A100-80GB":
            normalized.append(row)
            continue
        hardware = replace(row.hardware, device_name=_CANONICAL_DEVICE_NAME)
        changed = replace(row, hardware=hardware)
        if replace(changed, hardware=row.hardware) != row:
            raise AssertionError("A100 normalization changed a measurement field")
        normalized.append(changed)
    return tuple(normalized), dict(sorted(counts.items()))


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    protocol = load_v3_protocol(_PROTOCOL)
    require_v3_runtime(protocol)
    if any(path.exists() for path in (_RAW, _RAW_MANIFEST, _RAW_JOURNAL)):
        raise ValueError("raw mixed-A100 preservation paths already exist")
    for path in (_INPUT, _MANIFEST, _JOURNAL):
        if not path.is_file():
            raise ValueError(f"required validation artifact is missing: {path}")

    raw_sha256 = sha256_file(_INPUT)
    raw_manifest_sha256 = sha256_file(_MANIFEST)
    raw_journal_sha256 = sha256_file(_JOURNAL)
    rows = read_measurements(_INPUT)
    normalized, source_counts = canonicalize_rows(rows)

    write_bytes_atomic(_RAW, _INPUT.read_bytes())
    write_bytes_atomic(_RAW_MANIFEST, _MANIFEST.read_bytes())
    write_bytes_atomic(_RAW_JOURNAL, _JOURNAL.read_bytes())
    write_measurements_atomic(_INPUT, normalized)

    manifest = exact_object(read_json(_MANIFEST), context="v3 validation manifest")
    data = exact_object(manifest.get("data"), context="v3 validation data binding")
    data["sha256"] = sha256_file(_INPUT)
    data["rows"] = len(normalized)
    manifest["head_commit"] = _head()
    manifest["engineering_hardware_normalization"] = {
        "analysis_status": "operator_authorized_protocol_deviation",
        "logical_gpu": "A100-80GB",
        "source_device_name_counts": source_counts,
        "canonical_device_name": _CANONICAL_DEVICE_NAME,
        "transformation": "device_name replacement only; all measured values are unchanged",
        "reason": (
            "Modal returned PCIe for bank 0 and SXM for banks 1-4; the engineering engine "
            "requires one profile for a logical GPU"
        ),
        "validity_limit": (
            "A100 parameter selection is confounded by PCIe/SXM subvariant transfer and is "
            "not a confirmatory protocol result"
        ),
        "raw_archive": {
            "path": str(_RAW.relative_to(_REPO)),
            "sha256": raw_sha256,
            "rows": len(rows),
            "source_manifest_path": str(_RAW_MANIFEST.relative_to(_REPO)),
            "source_manifest_sha256": raw_manifest_sha256,
            "attempt_journal_path": str(_RAW_JOURNAL.relative_to(_REPO)),
            "attempt_journal_sha256": raw_journal_sha256,
        },
    }
    write_json_atomic(_MANIFEST, manifest)
    print(f"rows={len(normalized)} canonical_sha256={sha256_file(_INPUT)} raw_sha256={raw_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
