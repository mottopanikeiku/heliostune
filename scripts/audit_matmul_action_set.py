"""Audit a committed four-GPU timing matrix on CPU, with detailed H100 JSON and SVGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from heliostune.replay import BenchmarkTable
from heliostune.tooling.artifacts import read_measurements
from heliostune.tooling.matmul_audit import build_audit, gpu_overview, render_by_m, render_shapes

_REPO = Path(__file__).resolve().parents[1]
_INPUT = "benchmarks/data/parhelion-v2-measurements.jsonl.zst"
_PUBLISHED = "benchmarks/results/parhelion-h100-final.json"
_FREEZE = "benchmarks/parhelion-v2-h100-freeze.json"
_SOURCE_MANIFEST = "benchmarks/manifest.json"
_POST_MANIFEST = "benchmarks/parhelion-v2-post-run-manifest.json"


def analyze(repo: Path = _REPO) -> dict[str, Any]:
    published = json.loads((repo / _PUBLISHED).read_text(encoding="utf-8"))
    target = published["target_gpu"]
    table = BenchmarkTable(read_measurements(repo / _INPUT))
    # Bind this audit to the final study's exact corpus and action set, not a new subset.
    if [w.key for w in table.workloads(target)] != published["experiment"]["workload_keys"]:
        raise ValueError("measurement workload roster differs from the published final study")
    if [c.key for c in table.configs(target)] != published["experiment"]["config_keys"]:
        raise ValueError("measurement config roster differs from the published final study")
    for gpu in table.gpus:
        table.validate_protocol(gpu, target)
    audit = build_audit(table, target)
    equal_fold = audit["overall"]["reference_over_torch_equal_fold_mean"]
    if not math.isclose(equal_fold, published["auc"]["torch"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("audit aggregation differs from the published torch curve")
    audit["inputs"] = [
        {"path": path, "sha256": hashlib.sha256((repo / path).read_bytes()).hexdigest()}
        for path in (_INPUT, _PUBLISHED, _FREEZE, _SOURCE_MANIFEST, _POST_MANIFEST)
    ]
    frozen = json.loads((repo / _FREEZE).read_text(encoding="utf-8"))
    freeze = frozen["final_evaluation"]
    original = json.loads((repo / _SOURCE_MANIFEST).read_text(encoding="utf-8"))
    post = json.loads((repo / _POST_MANIFEST).read_text(encoding="utf-8"))
    audit["by_gpu"] = gpu_overview(table)
    audit["overview_note"] = (
        "Separate within-GPU summaries, not a pooled four-GPU estimate or policy replay. "
        "L4/A10 are earlier acquisition; T4 is validation acquisition; H100 is final acquisition. "
        "All use the same workload/config rosters, bank-1 selection and bank-2 scoring."
    )
    for overview in audit["by_gpu"]:
        earlier = overview["gpu"] in {"L4", "A10"}
        overview["collection"] = {
            "stage": (
                "earlier L4/A10"
                if earlier
                else "T4 validation"
                if overview["gpu"] == "T4"
                else "H100 final"
            ),
            "metadata_source": _SOURCE_MANIFEST if earlier else _POST_MANIFEST,
            "source_snapshot_commit": (
                original["protocol_commit"] if earlier else post["commits"]["collector"]
            ),
            "warmup_ms": original["scope"]["warmup_ms"] if earlier else freeze["warmup_ms"],
            "repetition_ms": (
                original["scope"]["repetition_ms"] if earlier else freeze["repetition_ms"]
            ),
            "timing_settings_provenance": (
                _SOURCE_MANIFEST
                if earlier
                else _FREEZE
                if overview["gpu"] == "H100"
                else (
                    "collector defaults (25/100 ms), not confirmed T4 invocation arguments; "
                    "the post-run manifest does not record T4 timing overrides"
                )
            ),
            "contract": (
                "Allocating FP16 timed callables; separate FP32-input correctness reference "
                "with TF32 disabled, converted to FP16. Torch median only. "
                "Same nominal timing/numerical settings, different acquisition stages; "
                "runtime accumulation/device-state uncertainty remains."
            ),
        }
    audit["timing_contract"] = {
        "source": {
            "path": "src/heliostune/kernel.py",
            "commit": frozen["collector_commit"],
            "functions": ["benchmark_measurements", "_benchmark_config", "_matmul_kernel"],
        },
        "warmup_ms": freeze["warmup_ms"],
        "repetition_ms": freeze["repetition_ms"],
        "inputs_output": "row-major CUDA FP16 A[M,K], B[K,N], FP16 output; Triton FP32 accumulator",
        "correctness": "Triton checked against FP32-input torch.matmul with TF32 disabled, converted to FP16, atol=rtol=0.01; distinct from the timed FP16 torch comparator",
        "timing": "triton.testing.do_bench warmed allocating callables; Triton p20/median/p80, torch median only; not CUDA-graph or end-to-end serving latency",
        "excluded": "compilation, policy compute, physical source acquisition, amortization and production interference",
        "torch_precision": "historical torch backend accumulation settings are not fully recorded in the timing rows; no assertion of matched stricter precision",
    }
    audit["published_policy_context"] = {
        "source": _PUBLISHED,
        "auc_1_to_8": published["auc"],
        "parhelion_transfer_strength": published["hyperparameters"]["parhelion"][
            "transfer_strength"
        ],
        "pooled_transfer_strength": published["hyperparameters"]["pooled_source_thompson"][
            "transfer_strength"
        ],
        "note": "budget curves average fold geometric means over seeds; torch is evaluation-only, outside the action set; AUC is mean of eight budgets",
    }
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_REPO / "results")
    args = parser.parse_args(argv)
    audit = analyze()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Compact JSON, deterministic field/row order, no clock timestamps or machine-local paths.
    (args.output_dir / "action-set-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    for name, figure in (
        ("action-set-by-m.svg", render_by_m(audit)),
        ("action-set-shapes.svg", render_shapes(audit)),
    ):
        (args.output_dir / name).write_text(figure, encoding="utf-8")
    print(
        json.dumps(
            {"overall": audit["overall"], "by_m": audit["by_m"], "by_gpu": audit["by_gpu"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
