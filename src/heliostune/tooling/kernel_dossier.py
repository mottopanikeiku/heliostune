"""CPU-only launch/traffic predictions, not measurements or a latency model."""

from __future__ import annotations

from typing import Any

from heliostune.action_configs import ActionCandidate, SplitKConfig, candidates_for
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, HopperGemmConfig, Workload
from heliostune.validation import exact_int


def _ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def predict(workload: Workload, candidate: ActionCandidate, *, sms: int) -> dict[str, Any]:
    """Count source-level work for allocating FP16 GEMM with FP32 accumulation.

    Traffic is logical tensor traffic: one ideal read of each input, one output
    write, plus a write/read of each split-K partial. It is NOT measured HBM
    traffic. The no-cross-tile-reuse model reloads A per N tile and B per M tile.
    Neither includes cache lines, descriptors, spills, allocator traffic or
    compiler scheduling. FLOPs count useful GEMM work, excluding reduction.
    """
    exact_int(sms, context="SM count", minimum=1)
    c = candidate.config
    m, n, k = workload.m, workload.n, workload.k
    if isinstance(c, SplitKConfig) and candidate not in candidates_for(workload):
        raise ValueError("split-K candidate is outside the frozen workload/config manifest")
    tm, tn = _ceil(m, c.block_m), _ceil(n, c.block_n)
    split = c.split_k if isinstance(c, SplitKConfig) else 1
    kt = _ceil(_ceil(k, c.block_k), split)
    padded_m, padded_n, padded_k = tm * c.block_m, tn * c.block_n, kt * split * c.block_k
    tiles = tm * tn
    workspace = 4 * split * m * n if split > 1 else 0
    ideal = 2 * (m * k + k * n + m * n)
    no_reuse = 2 * (tn * m * k + tm * k * n + m * n)
    return {
        "label": "INFERENCE: analytical prediction, not measured traffic or latency",
        "workload_key": workload.key,
        "shape": [m, n, k],
        "family": candidate.family,
        "config_key": candidate.key,
        "config": candidate.to_dict(),
        "sm_count": sms,
        "kernel_launches": 2 if split > 1 else 1,
        "output_tiles": tiles,
        "matmul_programs": min(sms, tiles) if isinstance(c, HopperGemmConfig) else tiles * split,
        "reduction_programs": _ceil(m * n, 256) if split > 1 else 0,
        "k_tiles_per_split": kt,
        "masked_row_fraction": 1 - m / padded_m,
        "padded_dot_flops": 2 * padded_m * padded_n * padded_k,
        "useful_gemm_flops": workload.flops,
        "useful_dot_fraction": workload.flops / (2 * padded_m * padded_n * padded_k),
        "workspace_bytes": workspace,
        "workspace_write_read_bytes": 2 * workspace,
        "ideal_tensor_bytes": ideal,
        "ideal_plus_workspace_bytes": ideal + 2 * workspace,
        "no_cross_tile_reuse_bytes": no_reuse + 2 * workspace,
        "ideal_intensity_flops_per_byte": workload.flops / ideal,
        "workspace_adjusted_intensity_flops_per_byte": workload.flops / (ideal + 2 * workspace),
        "no_cross_tile_reuse_intensity_flops_per_byte": workload.flops / (no_reuse + 2 * workspace),
    }


def build_dossier(*, sms: int) -> dict[str, Any]:
    # One fixed projection across every corpus M regime; not selected for speed.
    workloads = [w for w in DEFAULT_WORKLOADS if w.model == "qwen2.5-7b" and w.projection == "ffn-down"]
    old = ActionCandidate(DEFAULT_CONFIGS[0])
    return {
        "schema_version": 1,
        "evidence_label": "INFERENCE",
        "representatives": "Qwen FFN-down at every corpus M, all applicable new actions plus the first original manifest action; no timing-based selection",
        "formulas": {
            "tiles": "Tm=ceil(M/BM), Tn=ceil(N/BN), Tk=ceil(ceil(K/BK)/S); S=1 except split-K",
            "launches": "ordinary/persistent: 1; tensor-core split-K: 2 (partials and reduction)",
            "programs": "ordinary: Tm*Tn; persistent: min(SMs,Tm*Tn); split-K: S*Tm*Tn plus ceil(M*N/256) reduction programs",
            "masked_rows": "1-M/(Tm*BM); masked dot lanes, NOT a fraction of latency or HBM bytes",
            "workspace": "4*S*M*N FP32 bytes; write plus read = 8*S*M*N; unpadded; zero without split-K",
            "ideal_bytes": "2*(M*K+K*N+M*N) FP16 bytes, one read per operand and one output write",
            "intensity": "useful GEMM FLOPs 2*M*N*K divided by logical bytes; split-K adds workspace traffic to denominator",
            "no_cross_tile_reuse": "2*(Tn*M*K+Tm*K*N+M*N) + workspace write/read; no inter-tile input reuse",
            "padded_dot": "2*(Tm*BM)*(Tn*BN)*(S*Tk*BK); source-level padded dot work, not instruction counts",
        },
        "provenance": [
            {"family": "tile", "source": "src/heliostune/kernel.py", "origin": "Repository-authored conventional grouped tl.load/tl.dot implementation in the style of Triton's matrix-multiplication tutorial; the larger actions change parameters, not the algorithm. No novel GEMM algorithm claim."},
            {"family": "persistent", "source": "src/heliostune/hopper_kernel.py", "origin": "Tutorial-derived: Triton v3.4.0 09-persistent-matmul.py descriptor-persistent kernel. Repository adaptation/validation; not an original persistent algorithm. Expanded-bank configs disable warp specialization and epilogue subtiling."},
            {"family": "split_k", "source": "src/heliostune/action_expansion.py", "origin": "Original implementation in this repository: tensor-core partial tiles, unpadded FP32 workspace, separate Triton reduction, one FP16 narrowing. Split-K itself is an established technique, not a new algorithm. Distinct from the older non-tensor-core skinny_gemv."},
        ],
        "limits": [
            "No profiler counters, emitted instruction counts, occupancy model or predicted microseconds.",
            "Persistent program count is not achieved occupancy; masked rows are not measured wasted time.",
            "Logical traffic is not HBM traffic: caching/reloads, spills, descriptor scratch and allocation are unmodelled.",
            "Pipeline stages and warp counts change scheduling/resources but not these source-level tensor byte counts.",
        ],
        "predictions": [predict(w, c, sms=sms) for w in workloads for c in (old, *candidates_for(w))],
    }
