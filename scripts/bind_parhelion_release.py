"""Bind immutable run and artifact provenance into the final Parhelion summary."""

from __future__ import annotations

from pathlib import Path

from heliostune.artifacts import read_json, write_json_atomic
from heliostune.validation import exact_object

_REPO = Path(__file__).resolve().parents[1]
_SUMMARY = _REPO / "artifacts/h100-final-summary.json"
_RELEASE_PROVENANCE = {
    "sole_h100_run": "https://modal.com/apps/mottopanikeiku/main/ap-y68ldw4RUmTotSEIxGdqPz",
    "algorithm_commit": "811b05bb65bc978e44ca8fa32ceeeab315acf391",
    "freeze_commit": "c395630b9bcb4ef6a501d9a34696783620381c3c",
    "freeze_sha256": "c9c7138ef812166756746687463f81b88b63f905fb6998b9f468f1b0dadb0b4a",
    "raw_h100_sha256": "747f30a97711e549c886aedf5a93d4386d53def7c65f93f3ef5b8dd112bc1dd8",
    "final_archive_sha256": "f417bd7e8167d277e39678266c84e405bbf7606485b916e363c0feb7d418be5d",
    "post_run_manifest_path": "benchmarks/parhelion-v2-post-run-manifest.json",
}


def main() -> None:
    summary = exact_object(read_json(_SUMMARY), context="H100 final summary")
    existing = summary.get("release_provenance")
    if existing is not None and existing != _RELEASE_PROVENANCE:
        raise ValueError("summary contains conflicting release provenance")
    summary["release_provenance"] = _RELEASE_PROVENANCE
    write_json_atomic(_SUMMARY, summary)


if __name__ == "__main__":
    main()
