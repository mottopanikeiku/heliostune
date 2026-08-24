"""Strict sidecar, matrix, wall-time, hardware, and rank gates for v3 artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from heliostune.artifacts import read_json, read_measurements
from heliostune.configs import KernelConfig, Workload
from heliostune.features import v3_feature_rank
from heliostune.hardware import expectation_for_gpu, validate_hardware
from heliostune.schema import HardwareProfile, Measurement
from heliostune.validation import exact_object


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ValidatedCollection:
    data_path: Path
    sidecar_path: Path
    journal_path: Path
    rows: tuple[Measurement, ...]
    sidecar: Mapping[str, object]
    head_commit: str
    call_ids: tuple[str, ...]


def validate_collection(
    data_path: str | Path,
    *,
    protocol_path: str | Path,
    expected_gpus: Sequence[str],
    expected_banks: Sequence[int],
    expected_workload_keys: Sequence[str],
    expected_config_keys: Sequence[str],
    allow_prunable_failures: bool = False,
) -> ValidatedCollection:
    data = Path(data_path)
    sidecar_path = Path(f"{data}.manifest.json")
    sidecar = exact_object(read_json(sidecar_path), context="collection sidecar")
    data_binding = exact_object(sidecar.get("data"), context="collection sidecar data")
    if data_binding.get("sha256") != sha256_file(data):
        raise ValueError(f"collection data digest mismatch: {data}")
    request = exact_object(sidecar.get("request"), context="collection sidecar request")
    if request.get("gpus") != list(expected_gpus):
        raise ValueError(f"collection GPU request mismatch: {data}")
    if request.get("banks") != list(expected_banks):
        raise ValueError(f"collection bank request mismatch: {data}")
    if request.get("workload_keys") != list(expected_workload_keys):
        raise ValueError(f"collection workload request mismatch: {data}")
    if request.get("config_keys") != list(expected_config_keys):
        raise ValueError(f"collection config request mismatch: {data}")
    if request.get("seed_protocol") != "parhelion-v3":
        raise ValueError(f"collection does not use the v3 seed protocol: {data}")
    binding = exact_object(sidecar.get("binding"), context="collection sidecar binding")
    if binding.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError(f"collection protocol digest mismatch: {data}")
    journal_binding = exact_object(
        sidecar.get("attempt_journal"), context="collection attempt journal"
    )
    journal_path = Path(str(journal_binding.get("path")))
    if not journal_path.is_file() or journal_binding.get("sha256") != sha256_file(journal_path):
        raise ValueError(f"collection attempt journal digest mismatch: {data}")
    calls = sidecar.get("calls")
    if not isinstance(calls, list):
        raise ValueError(f"collection sidecar calls are missing: {data}")
    completed = [
        exact_object(call, context="collection call")
        for call in calls
        if isinstance(call, Mapping) and call.get("status") == "completed"
    ]
    expected_call_count = len(expected_gpus) * len(expected_banks)
    if len(completed) != expected_call_count:
        raise ValueError(
            f"collection completed calls={len(completed)}, expected={expected_call_count}: {data}"
        )
    call_ids = tuple(str(call["call_id"]) for call in completed)
    if len(set(call_ids)) != len(call_ids):
        raise ValueError(f"collection call IDs are not unique: {data}")
    facts = exact_object(sidecar.get("facts"), context="collection sidecar facts")
    head_commit = str(facts.get("head_commit"))
    if len(head_commit) != 40:
        raise ValueError(f"collection HEAD is invalid: {data}")

    rows = tuple(read_measurements(data))
    expected_cells = {
        (gpu, bank, workload, config)
        for gpu in expected_gpus
        for bank in expected_banks
        for workload in expected_workload_keys
        for config in expected_config_keys
    }
    actual_cells = {(row.hardware.gpu, row.bank, row.workload.key, row.config.key) for row in rows}
    if len(rows) != len(expected_cells) or actual_cells != expected_cells:
        raise ValueError(f"collection does not match exact requested cross-product: {data}")
    profiles: dict[str, HardwareProfile] = {}
    torch_timings: dict[tuple[str, int, str], tuple[float, float, float, float]] = {}
    for row in rows:
        profile = profiles.setdefault(row.hardware.gpu, row.hardware)
        if profile != row.hardware:
            raise ValueError(f"collection hardware profile changes on {row.hardware.gpu}")
        validate_hardware(row.hardware, expectation_for_gpu(row.hardware.gpu))
        if (
            row.torch_latency_p20_ms is None
            or row.torch_latency_p80_ms is None
            or row.torch_benchmark_wall_ms is None
        ):
            raise ValueError("v3 row lacks torch quantiles/wall")
        torch_timing = (
            row.torch_latency_ms,
            row.torch_latency_p20_ms,
            row.torch_latency_p80_ms,
            row.torch_benchmark_wall_ms,
        )
        known = torch_timings.setdefault(
            (row.hardware.gpu, row.bank, row.workload.key), torch_timing
        )
        if known != torch_timing:
            raise ValueError("v3 repeated torch timing differs across configs")
        if row.usable:
            if row.failure_stage is not None or row.benchmark_wall_ms is None:
                raise ValueError("v3 usable row lacks benchmark wall or has failure stage")
        elif not allow_prunable_failures:
            raise ValueError(
                f"v3 retained collection contains failure: {row.hardware.gpu}/"
                f"{row.workload.key}/{row.config.key}/bank-{row.bank}"
            )
        elif row.failure_stage not in {"compile", "correctness"}:
            raise ValueError("v3 candidate failure is not compile/correctness prunable")
    return ValidatedCollection(
        data_path=data,
        sidecar_path=sidecar_path,
        journal_path=journal_path,
        rows=rows,
        sidecar=sidecar,
        head_commit=head_commit,
        call_ids=call_ids,
    )


def validate_fold_ranks(
    rows: Sequence[Measurement],
    *,
    configs: tuple[KernelConfig, ...],
    workloads: tuple[Workload, ...],
    profile_order: Sequence[str],
    minimum_rank: int,
) -> dict[str, int]:
    profiles = {
        row.hardware.gpu: row.hardware
        for row in rows
        if row.bank == 0 and row.hardware.gpu in profile_order
    }
    if set(profiles) != set(profile_order):
        raise ValueError("rank gate is missing a declared hardware profile")
    ranks: dict[str, int] = {}
    for heldout_model in sorted({workload.model for workload in workloads}):
        shapes = {
            (workload.m, workload.n, workload.k)
            for workload in workloads
            if workload.model == heldout_model
        }
        eligible = tuple(
            workload
            for workload in workloads
            if workload.model != heldout_model
            and (workload.m, workload.n, workload.k) not in shapes
        )
        rank = v3_feature_rank(
            eligible,
            configs,
            tuple(profiles[gpu] for gpu in profile_order),
        )
        ranks[heldout_model] = rank
        if rank < minimum_rank:
            raise ValueError(f"v3 feature rank {rank} is below {minimum_rank} for {heldout_model}")
    return ranks


__all__ = ["ValidatedCollection", "sha256_file", "validate_collection", "validate_fold_ranks"]
