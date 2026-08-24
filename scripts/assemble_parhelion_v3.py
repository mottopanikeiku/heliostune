"""Assemble validated Parhelion v3 source or final archives with bound journals."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from heliostune.artifacts import (
    read_json,
    read_measurements,
    write_bytes_atomic,
    write_json_atomic,
    write_measurements_atomic,
)
from heliostune.configs import DEFAULT_WORKLOADS, PARHELION_V3_CANDIDATE_CONFIGS
from heliostune.protocol import (
    load_v3_protocol,
    require_v3_runtime,
    runtime_manifest,
)
from heliostune.v3_artifacts import (
    sha256_file,
    validate_collection,
    validate_fold_ranks,
)

_REPO = Path(__file__).resolve().parents[1]
_DEVELOPMENT_PROTOCOL = _REPO / "benchmarks/parhelion-v3-development-protocol.json"
_CONFIG_MANIFEST = _REPO / "artifacts/parhelion-v3-config-manifest.json"
_CANDIDATE_INPUT = _REPO / "artifacts/parhelion-v3-candidate-bank0.jsonl.zst"
_VALIDATION_BANKS_INPUT = _REPO / "artifacts/parhelion-v3-validation-banks1-4.jsonl.zst"
_VALIDATION_OUTPUT = _REPO / "benchmarks/data/parhelion-v3-validation.jsonl.zst"
_H200_INPUT = _REPO / "artifacts/parhelion-v3-h200.jsonl.zst"
_H200_OUTPUT = _REPO / "benchmarks/data/parhelion-v3-h200.jsonl.zst"
_FINAL_OUTPUT = _REPO / "benchmarks/data/parhelion-v3-final.jsonl.zst"


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _config_keys(manifest_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    manifest = cast(Mapping[str, object], read_json(manifest_path))
    retained = manifest.get("retained_config_keys")
    official = manifest.get("retained_official_config_keys")
    if not isinstance(retained, list) or not isinstance(official, list):
        raise ValueError("v3 config manifest lacks retained/official keys")
    retained_keys = tuple(str(key) for key in retained)
    official_keys = tuple(str(key) for key in official)
    if retained_keys != tuple(sorted(retained_keys)) or not official_keys:
        raise ValueError("v3 retained config order/official subset is invalid")
    return retained_keys, official_keys


def _workload_keys(protocol: Mapping[str, object]) -> tuple[str, ...]:
    rows = protocol.get("workloads")
    if not isinstance(rows, list):
        raise ValueError("v3 protocol workloads are missing")
    return tuple(str(cast(Mapping[str, object], row)["key"]) for row in rows)


def _completed_calls(sidecar: Mapping[str, object]) -> list[Mapping[str, object]]:
    calls = sidecar.get("calls")
    if not isinstance(calls, list):
        raise ValueError("collection sidecar calls are missing")
    return [
        cast(Mapping[str, object], call)
        for call in calls
        if isinstance(call, Mapping) and call.get("status") == "completed"
    ]


def _concat_journals(paths: tuple[Path, ...], destination: Path) -> None:
    payload = b"".join(path.read_bytes() for path in paths)
    if payload and not payload.endswith(b"\n"):
        raise ValueError("attempt journal must end with a newline")
    write_bytes_atomic(destination, payload)


def _copy_candidate_publication(
    candidate,
    protocol_path: Path,
    config_manifest_path: Path,
) -> None:
    destination = _REPO / "benchmarks/data/parhelion-v3-candidate-bank0.jsonl.zst"
    journal = Path(f"{destination}.attempts.jsonl")
    sidecar = Path(f"{destination}.manifest.json")
    write_bytes_atomic(destination, candidate.data_path.read_bytes())
    write_bytes_atomic(journal, candidate.journal_path.read_bytes())
    write_json_atomic(
        sidecar,
        {
            "schema_version": 1,
            "study_id": "parhelion-v3-candidate-bank0-publication",
            "data": {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "rows": len(candidate.rows),
            },
            "attempt_journal": {
                "path": str(journal),
                "sha256": sha256_file(journal),
            },
            "source": {
                "data_path": str(candidate.data_path),
                "data_sha256": sha256_file(candidate.data_path),
                "sidecar_path": str(candidate.sidecar_path),
                "sidecar_sha256": sha256_file(candidate.sidecar_path),
            },
            "protocol_sha256": sha256_file(protocol_path),
            "config_manifest_sha256": sha256_file(config_manifest_path),
            "head_commit": candidate.head_commit,
            "call_ids": list(candidate.call_ids),
            "runtime": runtime_manifest(),
        },
    )


def _assemble_validation(
    protocol_path: Path,
    config_manifest_path: Path,
    candidate_path: Path,
    banks_path: Path,
) -> None:
    protocol = load_v3_protocol(protocol_path)
    require_v3_runtime(protocol)
    retained_keys, official_keys = _config_keys(config_manifest_path)
    workload_keys = _workload_keys(protocol)
    candidate_keys = tuple(config.key for config in PARHELION_V3_CANDIDATE_CONFIGS)
    gpus = ("L4", "A10", "A100-80GB")
    candidate = validate_collection(
        candidate_path,
        protocol_path=protocol_path,
        expected_gpus=gpus,
        expected_banks=(0,),
        expected_workload_keys=workload_keys,
        expected_config_keys=candidate_keys,
        allow_prunable_failures=True,
    )
    banks = validate_collection(
        banks_path,
        protocol_path=protocol_path,
        expected_gpus=gpus,
        expected_banks=(1, 2, 3, 4),
        expected_workload_keys=workload_keys,
        expected_config_keys=retained_keys,
    )
    if candidate.head_commit != banks.head_commit or candidate.head_commit != _head():
        raise ValueError("v3 validation source HEADs differ from assembly checkout")
    retained = set(retained_keys)
    candidate_rows = tuple(row for row in candidate.rows if row.config.key in retained)
    if any(not row.usable for row in candidate_rows):
        raise ValueError("retained v3 candidate bank-0 rows contain failures")
    rows = tuple(
        sorted(
            (*candidate_rows, *banks.rows),
            key=lambda row: (
                row.hardware.gpu,
                row.bank,
                row.workload.key,
                row.config.key,
            ),
        )
    )
    configs_by_key = {config.key: config for config in PARHELION_V3_CANDIDATE_CONFIGS}
    configs = tuple(configs_by_key[key] for key in retained_keys)
    ranks = validate_fold_ranks(
        rows,
        configs=configs,
        workloads=DEFAULT_WORKLOADS,
        profile_order=gpus,
        minimum_rank=19,
    )
    expected_rows = len(gpus) * len(workload_keys) * len(retained_keys) * 5
    if len(rows) != expected_rows:
        raise ValueError(f"v3 validation rows={len(rows)}, expected={expected_rows}")

    _copy_candidate_publication(candidate, protocol_path, config_manifest_path)
    write_bytes_atomic(
        _REPO / "benchmarks/parhelion-v3-config-manifest.json",
        config_manifest_path.read_bytes(),
    )
    write_measurements_atomic(_VALIDATION_OUTPUT, rows)
    aggregate_journal = Path(f"{_VALIDATION_OUTPUT}.attempts.jsonl")
    _concat_journals((candidate.journal_path, banks.journal_path), aggregate_journal)
    calls = [*_completed_calls(candidate.sidecar), *_completed_calls(banks.sidecar)]
    if len(calls) != 15:
        raise ValueError(f"v3 validation requires 15 calls, got {len(calls)}")
    write_json_atomic(
        Path(f"{_VALIDATION_OUTPUT}.manifest.json"),
        {
            "schema_version": 1,
            "study_id": "parhelion-v3-validation-archive",
            "data": {
                "path": str(_VALIDATION_OUTPUT),
                "sha256": sha256_file(_VALIDATION_OUTPUT),
                "rows": len(rows),
                "gpus": list(gpus),
                "banks": [0, 1, 2, 3, 4],
            },
            "attempt_journal": {
                "path": str(aggregate_journal),
                "sha256": sha256_file(aggregate_journal),
            },
            "sources": [
                {
                    "data": str(source.data_path),
                    "data_sha256": sha256_file(source.data_path),
                    "sidecar": str(source.sidecar_path),
                    "sidecar_sha256": sha256_file(source.sidecar_path),
                    "journal": str(source.journal_path),
                    "journal_sha256": sha256_file(source.journal_path),
                }
                for source in (candidate, banks)
            ],
            "calls": calls,
            "call_ids": [str(call["call_id"]) for call in calls],
            "protocol": {
                "path": str(protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "config_manifest": {
                "path": str(config_manifest_path),
                "sha256": sha256_file(config_manifest_path),
                "retained_config_keys": list(retained_keys),
                "official_config_keys": list(official_keys),
            },
            "rank_gate": {"minimum": 19, "folds": ranks},
            "head_commit": candidate.head_commit,
            "runtime": runtime_manifest(),
        },
    )
    print(f"validation_rows={len(rows)} retained={len(retained_keys)}")


def _validation_publication() -> tuple[tuple, Mapping[str, object]]:
    manifest_path = Path(f"{_VALIDATION_OUTPUT}.manifest.json")
    manifest = cast(Mapping[str, object], read_json(manifest_path))
    data = cast(Mapping[str, object], manifest["data"])
    if data.get("sha256") != sha256_file(_VALIDATION_OUTPUT):
        raise ValueError("published validation data digest mismatch")
    return tuple(read_measurements(_VALIDATION_OUTPUT)), manifest


def _assemble_final(
    freeze_path: Path,
    config_manifest_path: Path,
    h200_path: Path,
) -> None:
    freeze = load_v3_protocol(freeze_path)
    require_v3_runtime(freeze)
    retained_keys, official_keys = _config_keys(config_manifest_path)
    validation_rows, validation_manifest = _validation_publication()
    workload_keys = tuple(workload.key for workload in DEFAULT_WORKLOADS)
    h200 = validate_collection(
        h200_path,
        protocol_path=freeze_path,
        expected_gpus=("H200",),
        expected_banks=(0, 1, 2, 3, 4),
        expected_workload_keys=workload_keys,
        expected_config_keys=retained_keys,
    )
    if h200.head_commit != _head():
        raise ValueError("H200 sidecar HEAD does not equal the freeze-merge checkout")
    rows = tuple(
        sorted(
            (*validation_rows, *h200.rows),
            key=lambda row: (
                row.hardware.gpu,
                row.bank,
                row.workload.key,
                row.config.key,
            ),
        )
    )
    configs_by_key = {config.key: config for config in PARHELION_V3_CANDIDATE_CONFIGS}
    configs = tuple(configs_by_key[key] for key in retained_keys)
    ranks = validate_fold_ranks(
        rows,
        configs=configs,
        workloads=DEFAULT_WORKLOADS,
        profile_order=("L4", "A10", "A100-80GB", "H200"),
        minimum_rank=20,
    )

    write_bytes_atomic(_H200_OUTPUT, h200.data_path.read_bytes())
    h200_journal = Path(f"{_H200_OUTPUT}.attempts.jsonl")
    write_bytes_atomic(h200_journal, h200.journal_path.read_bytes())
    h200_manifest_path = Path(f"{_H200_OUTPUT}.manifest.json")
    write_json_atomic(
        h200_manifest_path,
        {
            "schema_version": 1,
            "study_id": "parhelion-v3-h200-target",
            "data": {
                "path": str(_H200_OUTPUT),
                "sha256": sha256_file(_H200_OUTPUT),
                "rows": len(h200.rows),
            },
            "attempt_journal": {
                "path": str(h200_journal),
                "sha256": sha256_file(h200_journal),
            },
            "source_sidecar": {
                "path": str(h200.sidecar_path),
                "sha256": sha256_file(h200.sidecar_path),
            },
            "call_ids": list(h200.call_ids),
            "head_commit": h200.head_commit,
            "freeze_sha256": sha256_file(freeze_path),
            "runtime": runtime_manifest(),
        },
    )
    write_measurements_atomic(_FINAL_OUTPUT, rows)
    final_manifest_path = Path(f"{_FINAL_OUTPUT}.manifest.json")
    write_json_atomic(
        final_manifest_path,
        {
            "schema_version": 1,
            "study_id": "parhelion-v3-final-archive",
            "data": {
                "path": str(_FINAL_OUTPUT),
                "sha256": sha256_file(_FINAL_OUTPUT),
                "rows": len(rows),
                "gpus": ["L4", "A10", "A100-80GB", "H200"],
                "banks": [0, 1, 2, 3, 4],
            },
            "sources": {
                "validation_manifest_sha256": sha256_file(
                    Path(f"{_VALIDATION_OUTPUT}.manifest.json")
                ),
                "h200_manifest_sha256": sha256_file(h200_manifest_path),
            },
            "config_manifest_sha256": sha256_file(config_manifest_path),
            "official_config_keys": list(official_keys),
            "rank_gate": {"minimum": 20, "folds": ranks},
            "freeze_sha256": sha256_file(freeze_path),
            "head_commit": h200.head_commit,
            "validation_head_commit": validation_manifest["head_commit"],
            "runtime": runtime_manifest(),
        },
    )
    print(f"h200_rows={len(h200.rows)} final_rows={len(rows)}")


def _write_failure(path: Path, phase: str, error: BaseException, inputs: list[Path]) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "study_id": f"parhelion-v3-{phase}-failure",
            "phase": phase,
            "error": f"{type(error).__name__}: {error}",
            "head_commit": _head(),
            "runtime": runtime_manifest(),
            "available_inputs": [
                {
                    "path": str(input_path),
                    "sha256": sha256_file(input_path) if input_path.is_file() else None,
                }
                for input_path in inputs
            ],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "final"), required=True)
    parser.add_argument("--protocol", type=Path, default=_DEVELOPMENT_PROTOCOL)
    parser.add_argument("--config-manifest", type=Path, default=_CONFIG_MANIFEST)
    parser.add_argument("--candidate-input", type=Path, default=_CANDIDATE_INPUT)
    parser.add_argument("--validation-banks-input", type=Path, default=_VALIDATION_BANKS_INPUT)
    parser.add_argument("--h200-input", type=Path, default=_H200_INPUT)
    args = parser.parse_args(argv)
    try:
        if args.phase == "validation":
            _assemble_validation(
                args.protocol,
                args.config_manifest,
                args.candidate_input,
                args.validation_banks_input,
            )
        else:
            _assemble_final(args.protocol, args.config_manifest, args.h200_input)
    except BaseException as exc:
        failure_path = (
            _REPO / "benchmarks/parhelion-v3-validation-failure.json"
            if args.phase == "validation"
            else _REPO / "benchmarks/parhelion-v3-h200-failure-manifest.json"
        )
        _write_failure(
            failure_path,
            args.phase,
            exc,
            [
                args.protocol,
                args.config_manifest,
                args.candidate_input,
                Path(f"{args.candidate_input}.attempts.jsonl"),
                Path(f"{args.candidate_input}.manifest.json"),
                args.validation_banks_input,
                Path(f"{args.validation_banks_input}.attempts.jsonl"),
                Path(f"{args.validation_banks_input}.manifest.json"),
                args.h200_input,
                Path(f"{args.h200_input}.attempts.jsonl"),
                Path(f"{args.h200_input}.manifest.json"),
            ],
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
