"""Build or byte-check the pre-collection Parhelion v3 development protocol."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

from heliostune.artifacts import write_json_atomic
from heliostune.configs import (
    DEFAULT_WORKLOADS,
    PARHELION_V3_CANDIDATE_CONFIGS,
    PARHELION_V3_OFFICIAL_CONFIG_KEYS,
    TRITON_TUTORIAL_COMMIT,
    TRITON_TUTORIAL_CONFIG_PATH,
)
from heliostune.features import V2_FEATURE_NAMES, V3_FEATURE_NAMES
from heliostune.protocol import (
    V3_BANKS,
    V3_BUDGETS,
    V3_FINAL_SEEDS,
    V3_K_GRID,
    V3_METHOD_ROLES,
    V3_NOISE_VARIANCE,
    V3_PILOT_CONFIG_KEYS,
    V3_PILOT_WORKLOAD_KEYS,
    V3_PRIMARY_BUDGETS,
    V3_PRIOR_PRECISION,
    V3_SEED_PURPOSES,
    V3_TEMPERATURE_GRID,
    V3_TRANSFER_STRENGTH_GRID,
    V3_VALIDATION_SEEDS,
)

_REPO = Path(__file__).resolve().parents[1]
_OUTPUT = _REPO / "benchmarks/parhelion-v3-development-protocol.json"
_SOURCE_PATHS = (
    "modal_bench.py",
    "scripts/build_parhelion_v3_protocol.py",
    "scripts/assemble_parhelion_v3.py",
    "scripts/freeze_parhelion_v3.py",
    "scripts/prune_parhelion_v3_configs.py",
    "src/heliostune/cli.py",
    "src/heliostune/bandit.py",
    "src/heliostune/collection.py",
    "src/heliostune/configs.py",
    "src/heliostune/features.py",
    "src/heliostune/hardware.py",
    "src/heliostune/kernel.py",
    "src/heliostune/protocol.py",
    "src/heliostune/retrieval.py",
    "src/heliostune/schema.py",
    "src/heliostune/v3_artifacts.py",
    "src/heliostune/v3_engine.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_protocol() -> dict[str, object]:
    return {
        "schema_version": 1,
        "study_id": "parhelion-v3-development",
        "protocol_status": "predeclared_before_paid_performance_collection",
        "analysis_runtime": {
            "implementation": "CPython",
            "python_major_minor": [3, 11],
            "numpy": "2.4.6",
        },
        "benchmark": {
            "dtype": "float16",
            "atol": 0.01,
            "rtol": 0.01,
            "warmup_ms": 25,
            "repetition_ms": 100,
            "workloads": 96,
            "banks": list(V3_BANKS),
            "bank_roles": {
                "0": "policy observations and query cost",
                "1": "reference configuration selection",
                "2": "primary recommendation evaluation",
                "3": "first separate sensitivity evaluation",
                "4": "second separate sensitivity evaluation",
            },
        },
        "hardware": {
            "source_gpus": ["L4", "A10"],
            "validation_gpu": "A100-80GB",
            "final_archive_gpus": ["L4", "A10", "A100-80GB"],
            "target_gpu": "H200",
        },
        "budgets": list(V3_BUDGETS),
        "primary_auc_budgets": list(V3_PRIMARY_BUDGETS),
        "validation_seeds": list(V3_VALIDATION_SEEDS),
        "final_seeds": list(V3_FINAL_SEEDS),
        "selection_grids": {
            "k": list(V3_K_GRID),
            "temperature": list(V3_TEMPERATURE_GRID),
            "transfer_strength": list(V3_TRANSFER_STRENGTH_GRID),
            "selection_rule": (
                "maximize A100-80GB equal-fold/equal-seed/equal-budget AUC1-8 with "
                "ascending numeric-tuple ties; select baselines independently then Parhelion"
            ),
        },
        "candidate_source": {
            "default_config_count": 36,
            "official_config_count": 16,
            "candidate_count": 52,
            "triton_commit": TRITON_TUTORIAL_COMMIT,
            "triton_path": TRITON_TUTORIAL_CONFIG_PATH,
            "triton_symbol": "get_cuda_autotune_config",
        },
        "candidate_configs": [
            config.to_dict()
            | {
                "key": config.key,
                "official_source": config.key in PARHELION_V3_OFFICIAL_CONFIG_KEYS,
            }
            for config in PARHELION_V3_CANDIDATE_CONFIGS
        ],
        "pruning": {
            "inspection_gpus": ["L4", "A10", "A100-80GB"],
            "inspection_bank": 0,
            "prunable_failure_stages": ["compile", "correctness"],
            "benchmark_failure_rule": "abort",
            "minimum_retained_configs": 36,
            "minimum_retained_official_configs": 1,
            "rank_gate_l4_a10": 18,
            "rank_gate_with_a100": 19,
            "rank_gate_with_h200": 20,
            "later_banks": [1, 2, 3, 4],
        },
        "feature_schemas": {
            "v2_names": list(V2_FEATURE_NAMES),
            "v3_names": list(V3_FEATURE_NAMES),
            "v3_removed_v2_columns": [
                "log_m_over_n",
                "log_k_over_n",
                "log_flops",
                "n_divisible",
                "k_divisible",
            ],
        },
        "workloads": [workload.to_dict() | {"key": workload.key} for workload in DEFAULT_WORKLOADS],
        "pilot": {
            "workload_keys": list(V3_PILOT_WORKLOAD_KEYS),
            "config_keys": list(V3_PILOT_CONFIG_KEYS),
            "cells": 6,
            "calls": 1,
        },
        "seed_contract": {
            "preimage": (
                "parhelion-v3\\0{gpu-or-na}\\0{bank-or-na}\\0{heldout-model-or-na}"
                "\\0{workload-key-or-all}\\0{policy-seed-decimal-or-na}"
                "\\0{round-decimal-or-na}\\0{purpose}"
            ),
            "derivation": "first eight SHA-256 bytes interpreted as a big-endian integer",
            "purposes": sorted(V3_SEED_PURPOSES),
            "stream_reuse": {
                "pooled_source_thompson": "cold-thompson",
                "parhelion_no_transfer": "parhelion-thompson",
            },
        },
        "methods": [{"key": key, "role": role} for key, role in V3_METHOD_ROLES.items()],
        "policy_contract": {
            "prior_precision": V3_PRIOR_PRECISION,
            "noise_variance": V3_NOISE_VARIANCE,
            "reward": "log_tflops_reward on bank-0 latency",
            "source_update_order": (
                "declared GPU order, ascending workload key, ascending retained config key "
                "after family/exact-shape exclusion"
            ),
            "choice": (
                "one Thompson draw over all currently unqueried actions; without replacement; "
                "update selected bank-0 reward only"
            ),
            "recommendation": (
                "lowest-latency paid bank-0 incumbent with ascending config-key ties"
            ),
            "primary": (
                "50 same-seed equal-four-fold Parhelion-minus-anchored-cold AUC1-8 "
                "effects; superiority only when the two-sided 95% Student-t lower bound > 0"
            ),
            "independent_sensitivity": (
                "separate target posteriors from the same fold source likelihood/prior"
            ),
        },
        "evaluation_contract": {
            "primary_bank": 2,
            "sensitivity_banks": [3, 4],
            "sensitivity_rule": (
                "score unchanged bank-0 incumbent and bank-1 reference separately on each "
                "bank; never average sensitivity banks or use them for selection"
            ),
            "queries_to_95": (
                "first sequential budget 1-16 at fraction >= 0.95; null when unattained; "
                "statistics.median_low over successful seeds only"
            ),
            "deterministic_uncertainty": (
                "four equal-weight fold values, minimum, maximum, sample SD; no interval"
            ),
            "stochastic_uncertainty": (
                "policy-seed Student-t Monte Carlo interval conditional on fixed data/campaign"
            ),
        },
        "cost_contract": {
            "query_wall": (
                "sum bank-0 benchmark_wall_ms only; total and divided-by-96 workload mean"
            ),
            "torch": "sum 96 bank-2 torch_benchmark_wall_ms values",
            "static": "zero target-query cost",
            "official_exhaustive": "sum every surviving official bank-0 probe",
            "compile_analysis": (
                "collector-order descriptive count/p10/median/p90/max for all compile_ms and "
                "first observed row per config; numpy.quantile(method='linear'); no interval"
            ),
        },
        "software": {
            "python": "3.11.x",
            "numpy": "2.4.6",
            "rich": "14.3.4",
            "zstandard": "0.25.0",
            "torch": "2.8.0",
            "triton": "3.4.0",
            "modal": "1.5.4",
        },
        "implementation_sha256": {
            relative: _sha256(_REPO / relative) for relative in _SOURCE_PATHS
        },
        "failure_outcomes": {
            "pre_h200": "publish bound validation failure manifest and software/protocol",
            "post_freeze_h200": (
                "publish bound H200 failure manifest; no substitution, performance report, or rerun"
            ),
            "negative_or_null_result": "publish as complete",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        with tempfile.TemporaryDirectory(prefix="parhelion-v3-protocol-") as temporary:
            generated = Path(temporary) / _OUTPUT.name
            write_json_atomic(generated, build_protocol())
            if not _OUTPUT.is_file() or _OUTPUT.read_bytes() != generated.read_bytes():
                raise SystemExit(f"v3 development protocol is stale: {_OUTPUT}")
    else:
        write_json_atomic(_OUTPUT, build_protocol())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
