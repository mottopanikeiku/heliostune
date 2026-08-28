"""Bind A100 selection and validation bytes before the sole H200 invocation."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from heliostune.artifacts import read_json, write_json_atomic, write_text_atomic
from heliostune.protocol import (
    load_v3_protocol,
    require_v3_runtime,
    runtime_manifest,
)
from heliostune.v3_artifacts import sha256_file

_REPO = Path(__file__).resolve().parents[1]
_PROTOCOL = _REPO / "benchmarks/parhelion-v3-development-protocol.json"
_CONFIG = _REPO / "benchmarks/parhelion-v3-config-manifest.json"
_VALIDATION = _REPO / "benchmarks/data/parhelion-v3-validation.jsonl.zst"
_SELECTION = _REPO / "benchmarks/results/parhelion-v3-a100-selection.json"
_OUTPUT = _REPO / "benchmarks/parhelion-v3-h200-freeze.json"
_HASH_OUTPUT = _REPO / "benchmarks/parhelion-v3-h200-freeze.sha256"
_IMPLEMENTATION_PATHS = (
    "modal_bench.py",
    "scripts/assemble_parhelion_v3.py",
    "scripts/canonicalize_parhelion_v3_a100.py",
    "scripts/freeze_parhelion_v3.py",
    "scripts/prune_parhelion_v3_configs.py",
    "src/heliostune/bandit.py",
    "src/heliostune/collection.py",
    "src/heliostune/configs.py",
    "src/heliostune/features.py",
    "src/heliostune/kernel.py",
    "src/heliostune/protocol.py",
    "src/heliostune/retrieval.py",
    "src/heliostune/schema.py",
    "src/heliostune/v3_artifacts.py",
    "src/heliostune/v3_engine.py",
)


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_source_clean() -> None:
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "src",
            "scripts",
            "modal_bench.py",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=_REPO,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("freeze requires committed implementation sources")


def main() -> int:
    protocol = load_v3_protocol(_PROTOCOL)
    require_v3_runtime(protocol)
    _require_source_clean()
    head = _head()
    config = cast(Mapping[str, object], read_json(_CONFIG))
    validation_manifest_path = Path(f"{_VALIDATION}.manifest.json")
    validation_journal_path = Path(f"{_VALIDATION}.attempts.jsonl")
    validation_manifest = cast(Mapping[str, object], read_json(validation_manifest_path))
    if validation_manifest.get("head_commit") != head:
        raise ValueError("validation manifest HEAD does not match candidate merge SHA")
    selection = cast(Mapping[str, object], read_json(_SELECTION))
    if selection.get("study_id") != "parhelion-v3-a100-selection":
        raise ValueError("unexpected A100 selection study ID")
    selected = cast(Mapping[str, object], selection.get("selected"))
    retained = config.get("retained_config_keys")
    official = config.get("retained_official_config_keys")
    if not isinstance(retained, list) or not isinstance(official, list) or not official:
        raise ValueError("freeze config manifest is incomplete")

    payload = {
        "schema_version": 1,
        "study_id": "parhelion-v3-h200-freeze",
        "freeze_status": "predates_h200_and_permits_one_target_invocation",
        "candidate_merge_sha": head,
        "analysis_runtime": protocol["analysis_runtime"],
        "runtime_at_freeze": runtime_manifest(),
        "development_protocol": {
            "path": str(_PROTOCOL.relative_to(_REPO)),
            "sha256": sha256_file(_PROTOCOL),
        },
        "config_manifest": {
            "path": str(_CONFIG.relative_to(_REPO)),
            "sha256": sha256_file(_CONFIG),
            "retained_config_keys": retained,
            "official_config_keys": official,
        },
        "a100_selection": {
            "path": str(_SELECTION.relative_to(_REPO)),
            "sha256": sha256_file(_SELECTION),
            "selected": selected,
        },
        "validation_archive": {
            "path": str(_VALIDATION.relative_to(_REPO)),
            "sha256": sha256_file(_VALIDATION),
            "manifest_path": str(validation_manifest_path.relative_to(_REPO)),
            "manifest_sha256": sha256_file(validation_manifest_path),
            "attempt_journal_path": str(validation_journal_path.relative_to(_REPO)),
            "attempt_journal_sha256": sha256_file(validation_journal_path),
            "head_commit": validation_manifest["head_commit"],
            "call_ids": validation_manifest["call_ids"],
        },
        "implementation_sha256": {
            relative: sha256_file(_REPO / relative) for relative in _IMPLEMENTATION_PATHS
        },
        "workloads": protocol["workloads"],
        "candidate_configs": protocol["candidate_configs"],
        "banks": cast(Mapping[str, object], protocol["benchmark"])["banks"],
        "budgets": protocol["budgets"],
        "primary_auc_budgets": protocol["primary_auc_budgets"],
        "final_seeds": protocol["final_seeds"],
        "seed_contract": protocol["seed_contract"],
        "policy_contract": protocol["policy_contract"],
        "evaluation_contract": protocol["evaluation_contract"],
        "h200_identity_gate": {
            "selector": "H200",
            "compute_capability": [9, 0],
            "memory_gb": [135, 145],
            "required_name": "H200",
            "excluded_names": ["H100", "B200"],
        },
        "commands": {
            "wheel": ".venv-v3/bin/python scripts/build_modal_wheel.py",
            "collection": (
                ".venv-v3/bin/modal run modal_bench.py --protocol "
                "benchmarks/parhelion-v3-h200-freeze.json --config-manifest "
                "benchmarks/parhelion-v3-config-manifest.json --gpus H200 "
                "--banks 0,1,2,3,4 --output artifacts/parhelion-v3-h200.jsonl.zst"
            ),
            "assembly": ".venv-v3/bin/python scripts/assemble_parhelion_v3.py --phase final",
            "comparison": (
                ".venv-v3/bin/heliostune compare-v3 "
                "benchmarks/data/parhelion-v3-final.jsonl.zst --freeze "
                "benchmarks/parhelion-v3-h200-freeze.json"
            ),
        },
        "failure_rules": {
            "identity_or_completeness_failure": (
                "publish benchmarks/parhelion-v3-h200-failure-manifest.json; "
                "no performance report, substitution, or rerun"
            ),
            "negative_or_null_result": "publish as complete",
            "invocation_limit": 1,
        },
    }
    write_json_atomic(_OUTPUT, payload)
    digest = sha256_file(_OUTPUT)
    write_text_atomic(_HASH_OUTPUT, f"{digest}  {_OUTPUT.name}\n")
    print(f"freeze_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
