"""CPU-only launch/traffic predictions, not measurements or a latency model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from heliostune.action_configs import ActionCandidate, SplitKConfig, candidates_for
from heliostune.configs import (
    DEFAULT_CONFIGS,
    DEFAULT_WORKLOADS,
    HopperGemmConfig,
    KernelConfig,
    Workload,
)
from heliostune.validation import exact_int

REPRESENTATIVE_MODEL = "qwen2.5-7b"
REPRESENTATIVE_PROJECTION = "ffn-down"
SPLIT_K_REDUCE_BLOCK = 256


def _ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def predict(
    workload: Workload, config: KernelConfig | HopperGemmConfig | SplitKConfig, *, sms: int
) -> dict[str, Any]:
    """Count source-level work for allocating FP16 GEMM with FP32 accumulation.

    Traffic is logical tensor traffic: one ideal read of each input, one output
    write, plus a write/read of each split-K partial. It is NOT measured HBM
    traffic. The no-cross-tile-reuse model reloads A per N tile and B per M tile.
    Neither includes cache lines, descriptors, spills, allocator traffic or
    compiler scheduling. FLOPs count useful GEMM work, excluding reduction.
    """
    exact_int(sms, context="SM count", minimum=1)
    m, n, k = workload.m, workload.n, workload.k
    if isinstance(config, SplitKConfig) and ActionCandidate(config) not in candidates_for(workload):
        raise ValueError("split-K candidate is outside the frozen workload/config manifest")
    tm, tn = _ceil(m, config.block_m), _ceil(n, config.block_n)
    split = config.split_k if isinstance(config, SplitKConfig) else 1
    kt = _ceil(_ceil(k, config.block_k), split)
    padded_m, padded_n, padded_k = (
        tm * config.block_m,
        tn * config.block_n,
        kt * split * config.block_k,
    )
    tiles = tm * tn
    persistent = isinstance(config, HopperGemmConfig)
    programs = min(sms, tiles) if persistent else tiles * split
    tiles_per_program = _ceil(tiles, programs) if persistent else None
    workspace = 4 * split * m * n if split > 1 else 0
    ideal = 2 * (m * k + k * n + m * n)
    no_reuse = 2 * (tn * m * k + tm * k * n + m * n)
    padded_flops = 2 * padded_m * padded_n * padded_k
    return {
        "label": "INFERENCE: analytical prediction, not measured traffic or latency",
        "workload_key": workload.key,
        "shape": [m, n, k],
        "sm_count": sms,
        "kernel_launches": 2 if split > 1 else 1,
        "output_tiles": tiles,
        "matmul_programs": programs,
        "programs_per_sm": programs / sms,
        "persistent_tiles_per_program_max": tiles_per_program,
        "persistent_tile_balance": tiles / (programs * tiles_per_program)
        if tiles_per_program is not None
        else None,
        "reduction_programs": _ceil(m * n, SPLIT_K_REDUCE_BLOCK) if split > 1 else 0,
        "k_tiles_per_split": kt,
        "masked_row_fraction": 1 - m / padded_m,
        "padded_dot_flops": padded_flops,
        "useful_gemm_flops": workload.flops,
        "useful_dot_fraction": workload.flops / padded_flops,
        "workspace_bytes": workspace,
        "workspace_write_read_bytes": 2 * workspace,
        "ideal_tensor_bytes": ideal,
        "ideal_plus_workspace_bytes": ideal + 2 * workspace,
        "no_cross_tile_reuse_bytes": no_reuse + 2 * workspace,
        "ideal_intensity_flops_per_byte": workload.flops / ideal,
        "workspace_adjusted_intensity_flops_per_byte": workload.flops / (ideal + 2 * workspace),
        "no_cross_tile_reuse_intensity_flops_per_byte": workload.flops / (no_reuse + 2 * workspace),
    }


def _measured(workload: Mapping[str, Any], arm: str, config_key: str) -> dict[str, Any] | None:
    if workload["selected"][arm]["config_key"] != config_key:
        return None
    scored = workload["bank2"][arm]
    torch = workload["bank2"]["torch"]
    if scored["status"] != "scored" or torch["status"] != "scored":
        return {
            "status": scored["status"],
            "median_us": None,
            "torch_us": None,
            "ratio_to_torch": None,
        }
    return {
        "status": "scored",
        "median_us": scored["median_ms"] * 1000,
        "torch_us": torch["median_ms"] * 1000,
        "ratio_to_torch": workload[f"{arm}_to_torch"],
    }


def build_dossier(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Predict every applicable new action plus the bank-1 old winner per representative.

    The representative workloads are one fixed projection at every corpus M,
    never chosen by timing. Measured fields come only from the bank-2 finalists;
    a prediction without a bank-2 measurement is not evidence about latency.
    """
    sms = exact_int(summary["hardware"]["multiprocessor_count"], context="SM count", minimum=1)
    measured = {w["workload_key"]: w for w in summary["workloads"]}
    old_configs = {c.key: c for c in DEFAULT_CONFIGS}
    predictions = []
    for w in DEFAULT_WORKLOADS:
        if w.model != REPRESENTATIVE_MODEL or w.projection != REPRESENTATIVE_PROJECTION:
            continue
        record = measured[w.key]
        old_key = record["selected"]["old"]["config_key"]
        arms: list[tuple[str, str, str, KernelConfig | HopperGemmConfig | SplitKConfig]] = [
            ("old", "old", old_key, old_configs[old_key])
        ]
        arms += [("new", c.family, c.key, c.config) for c in candidates_for(w)]
        for arm, family, key, config in arms:
            predictions.append(
                {
                    "arm": arm,
                    "family": family,
                    "config_key": key,
                    "config": config.to_dict(),
                    "bank1_selected": record["selected"][arm]["config_key"] == key,
                    "bank2_measurement": _measured(record, arm, key),
                    **predict(w, config, sms=sms),
                }
            )
    return {
        "schema_version": 1,
        "evidence_label": "INFERENCE",
        "representatives": (
            "Qwen2.5-7B FFN-down at every corpus M: all applicable new actions plus the "
            "bank-1-selected old action. Workloads are fixed by projection, not by timing."
        ),
        "sm_count": sms,
        "formulas": {
            "tiles": "Tm=ceil(M/BM), Tn=ceil(N/BN), Tk=ceil(ceil(K/BK)/S); S=1 except split-K",
            "launches": "ordinary/persistent: 1; tensor-core split-K: 2 (partials and reduction)",
            "programs": (
                "ordinary: Tm*Tn; persistent: min(SMs,Tm*Tn); split-K: S*Tm*Tn "
                f"plus ceil(M*N/{SPLIT_K_REDUCE_BLOCK}) reduction programs"
            ),
            "persistent_balance": (
                "each persistent program strides tiles by SMs: at most ceil(Tm*Tn/P) tiles "
                "per program; balance = Tm*Tn/(P*ceil(Tm*Tn/P)), 1 when every program "
                "gets the same tile count"
            ),
            "masked_rows": "1-M/(Tm*BM); masked dot lanes, NOT a fraction of latency or HBM bytes",
            "workspace": "4*S*M*N FP32 bytes; write plus read = 8*S*M*N; unpadded; zero without split-K",
            "ideal_bytes": "2*(M*K+K*N+M*N) FP16 bytes, one read per operand and one output write",
            "intensity": (
                "useful GEMM FLOPs 2*M*N*K divided by logical bytes; split-K adds "
                "workspace traffic to the denominator"
            ),
            "no_cross_tile_reuse": "2*(Tn*M*K+Tm*K*N+M*N) + workspace write/read; no inter-tile input reuse",
            "padded_dot": "2*(Tm*BM)*(Tn*BN)*(S*Tk*BK); source-level padded dot work, not instruction counts",
        },
        "provenance": [
            {
                "family": "old / tile",
                "source": "src/heliostune/kernel.py",
                "origin": (
                    "Repository-authored conventional grouped tl.load/tl.dot kernel in the "
                    "style of Triton's matrix-multiplication tutorial. The four larger new "
                    "tile actions change launch parameters, not the algorithm. No novel GEMM "
                    "algorithm claim."
                ),
            },
            {
                "family": "persistent",
                "source": "src/heliostune/hopper_kernel.py",
                "origin": (
                    "Tutorial-derived: Triton v3.4.0 09-persistent-matmul.py "
                    "descriptor-persistent kernel, adapted and validated here; not an original "
                    "persistent algorithm. Both expanded-bank configs disable warp "
                    "specialisation and epilogue subtiling."
                ),
            },
            {
                "family": "split_k",
                "source": "src/heliostune/action_expansion.py",
                "origin": (
                    "Implemented in this repository: tensor-core partial tiles, unpadded FP32 "
                    "workspace, separate Triton reduction, one FP16 narrowing. Split-K itself "
                    "is an established technique, not a new algorithm. Distinct from the older "
                    "non-tensor-core skinny_gemv."
                ),
            },
        ],
        "limits": [
            "No profiler counters, emitted instruction counts, occupancy model or predicted microseconds.",
            "Persistent program count is not achieved occupancy; masked rows are not measured wasted time.",
            "Logical traffic is not HBM traffic: caching/reloads, spills, descriptor scratch and allocation are unmodelled.",
            "Pipeline stages and warp counts change scheduling/resources but not these source-level tensor byte counts.",
            "A prediction beside a measurement is a juxtaposition, not a causal explanation of that measurement.",
        ],
        "predictions": predictions,
    }
