"""Capture immutable bytes present at the audited HeliosTune commit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

from heliostune.artifacts import write_text_atomic

_REPO: Final = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT: Final = _REPO / "benchmarks/historical-artifact-baseline.json"
_AUDITED_COMMIT: Final = "08d73bc6327fabe917f56e1c9f25fbd0e8e936cf"
_PRESENT_ARTIFACTS: Final = (
    "benchmarks/manifest.json",
    "benchmarks/parhelion-v2-development-protocol.json",
    "benchmarks/parhelion-v2-h100-freeze.json",
    "benchmarks/parhelion-v2-h100-freeze.sha256",
    "benchmarks/parhelion-v2-post-run-manifest.json",
    "benchmarks/parhelion-v2-post-run-manifest.sha256",
    "benchmarks/data/h100-measurements.jsonl.zst",
    "benchmarks/data/measurements.jsonl.zst",
    "benchmarks/data/parhelion-v2-measurements.jsonl.zst",
    "benchmarks/data/t4-measurements.jsonl.zst",
    "benchmarks/results/a10-to-l4.json",
    "benchmarks/results/l4-to-a10.json",
    "benchmarks/results/parhelion-h100-final.json",
    "benchmarks/results/parhelion-t4-selection.json",
    "benchmarks/results/parhelion-t4-validation.json",
    "site/a10-to-l4.html",
    "site/a10-to-l4.json",
    "site/index.html",
    "site/l4-to-a10.json",
    "site/parhelion-h100-final.json",
    "site/parhelion-t4-selection.json",
    "site/v1.html",
)
_ABSENT_ALIASES: Final = {
    "artifacts/h100-final-summary.json": (
        "benchmarks/results/parhelion-h100-final.json",
        "765b347a2675b0647f9f58bd6ba36904dfcf2761be31b7e3b930b63a2ad28abd",
        "765b347a2675b0647f9f58bd6ba36904dfcf2761be31b7e3b930b63a2ad28abd",
        "identity",
    ),
    "artifacts/h100-measurements.jsonl": (
        "benchmarks/data/h100-measurements.jsonl.zst",
        "91b94ec42a71b1832203a6c5cdedf0102cf8d04ae9992d739fffa5351b7ac206",
        "747f30a97711e549c886aedf5a93d4386d53def7c65f93f3ef5b8dd112bc1dd8",
        "zstd_of_recorded_content",
    ),
    "artifacts/h100-report.html": (
        "site/index.html",
        "149dc7c1a705a5d7189e575274440a72ddbcba59e183554d7a0bd6fd739beb61",
        "149dc7c1a705a5d7189e575274440a72ddbcba59e183554d7a0bd6fd739beb61",
        "identity",
    ),
    "artifacts/parhelion-final.jsonl": (
        "benchmarks/data/parhelion-v2-measurements.jsonl.zst",
        "ed6ec6ee8c3b61b451ea1276fc6f3925e82f70b5e208e9195c924ef6acc7343f",
        "f417bd7e8167d277e39678266c84e405bbf7606485b916e363c0feb7d418be5d",
        "zstd_of_recorded_content",
    ),
    "artifacts/parhelion-selection.json": (
        "benchmarks/results/parhelion-t4-selection.json",
        "7968170990d8ae969d9b1e02baf7afa3dff0c6446497330c599aa4163ab516ca",
        "7968170990d8ae969d9b1e02baf7afa3dff0c6446497330c599aa4163ab516ca",
        "identity",
    ),
    "artifacts/t4-final-summary.json": (
        "benchmarks/results/parhelion-t4-validation.json",
        "794d8f38a1b6fecaece0abdb38cdaa7ab5cdbf1bdfaefbde518c32daf84c28f4",
        "794d8f38a1b6fecaece0abdb38cdaa7ab5cdbf1bdfaefbde518c32daf84c28f4",
        "identity",
    ),
    "artifacts/t4-measurements.jsonl": (
        "benchmarks/data/t4-measurements.jsonl.zst",
        "f00c2a1498937707c872f667cf210de5e7319218c7df56e05a15174441a36c00",
        "f3910e73aec68d48015b0fb9d0469ff2be8c3210d5777dab61e4a9cdd660d217",
        "zstd_of_recorded_content",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(repo: Path = _REPO) -> dict[str, object]:
    present: dict[str, object] = {}
    for relative in _PRESENT_ARTIFACTS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"historical artifact is missing: {relative}")
        present[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "status": "present_at_audited_commit",
        }
    aliases: dict[str, object] = {}
    for alias, (
        published_path,
        published_sha256,
        recorded_sha256,
        representation,
    ) in _ABSENT_ALIASES.items():
        if (repo / alias).exists():
            raise ValueError(f"freeze-only alias unexpectedly exists: {alias}")
        published = repo / published_path
        if not published.is_file():
            raise FileNotFoundError(f"published replacement is missing: {published_path}")
        actual_sha256 = _sha256(published)
        if actual_sha256 != published_sha256:
            raise ValueError(
                f"published replacement digest mismatch for {published_path}: "
                f"expected {published_sha256}, got {actual_sha256}"
            )
        aliases[alias] = {
            "published_replacement": published_path,
            "published_replacement_sha256": actual_sha256,
            "recorded_sha256": recorded_sha256,
            "replacement_representation": representation,
            "status": "not_present_at_audited_commit",
        }
    return {
        "schema_version": 1,
        "audited_commit": _AUDITED_COMMIT,
        "absent_freeze_aliases": aliases,
        "present_artifacts": present,
    }


def _serialized(payload: object) -> str:
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rendered = _serialized(capture())
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"historical baseline is stale: {args.output}")
        return 0
    write_text_atomic(args.output, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
