"""Summarize committed tuner result JSON without replaying kernels or importing torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_INPUTS = (
    "parhelion-h100-final.json",
    "parhelion-t4-selection.json",
    "l4-to-a10.json",
    "a10-to-l4.json",
)
_H100_METHODS = (
    "cold_thompson",
    "single_source_nearest",
    "multisource_retrieval",
    "parhelion_thompson",
    "torch",
)
_V1_METHODS = ("cold_thompson", "nearest_shape", "transfer_thompson")
_BUDGETS = tuple(range(1, 9))
_LABEL_OVERRIDES = {
    "parhelion_thompson": "Retrieval-anchored Thompson sampling",
    "transfer_thompson": "Transfer Thompson sampling",
}


def load_results(repo: Path) -> dict[str, Any]:
    return {
        name: json.loads((repo / "benchmarks/results" / name).read_text(encoding="utf-8"))
        for name in _INPUTS
    }


def _points(result: dict[str, Any], method: str) -> dict[int, dict[str, Any]]:
    return {point["budget"]: point for point in result["methods"][method]}


def _number(value: float) -> str:
    return f"{value:.6f}"


def _link(name: str) -> str:
    return f"[{name}](../benchmarks/results/{name})"


def _label(result: dict[str, Any], method: str) -> str:
    return _LABEL_OVERRIDES.get(method, str(result["method_labels"][method]))


def build_summary(repo: Path = _REPO) -> str:
    results = load_results(repo)
    h100 = results[_INPUTS[0]]
    selection = results[_INPUTS[1]]
    configs = h100["configs"]
    lines = [
        "# Tuner comparison",
        "",
        "Cold Thompson sampling, nearest-shape reuse and cross-GPU transfer, "
        "compared using saved timing measurements.",
        "",
        "## What the ratio measures",
        "",
        f"The reference is the fastest of the {configs} Triton configurations, chosen on "
        "the selection pass and timed on the scoring pass. For each workload the ratio "
        "is reference latency / recommendation latency, both timed on the scoring pass. "
        "Higher is better. Policies see a separate timing pass. A ratio above 1 can "
        "come from a faster implementation outside the 36 configurations or variation "
        "between timing passes.",
        "",
        "`torch.matmul` is a baseline; the tuner cannot choose it. "
        "Its constant curve is repeated at each budget for comparison.",
        "",
        "AUC 1–8 is the arithmetic mean of the eight reported budget ratios, not a "
        "trapezoidal integral. The stored curves first aggregate workload ratios "
        "geometrically, then average equally across model-family folds and policy seeds.",
        "",
        "## H100: shape-disjoint multi-source replay",
        "",
        f"Source: {_link(_INPUTS[0])}. Target: {h100['target_gpu']}; "
        f"source GPUs: {', '.join(h100['source_gpus'])}; {h100['workloads']} workloads, "
        f"{h100['model_families']} model families, {h100['seeds']} policy seeds. "
        "Source archives exclude both the held-out model family and exact target shapes.",
        "",
        "Each probe times one configuration on a held-out workload. "
        "For each seed, Thompson-sampling methods share a posterior across workloads "
        "and update it in paired batches.",
        "",
        "| Method (JSON key) | AUC 1–8 | Budget-8 ratio | Budget-8 95% interval |",
        "| --- | ---: | ---: | --- |",
    ]
    for method in _H100_METHODS:
        point = _points(h100, method)[8]
        lines.append(
            f"| {_label(h100, method)} (`{method}`) | "
            f"{_number(h100['auc'][method])} | {_number(point['mean_fraction_oracle'])} | "
            f"[{_number(point['ci95_low'])}, {_number(point['ci95_high'])}] |"
        )
    lines += [
        "",
        "### Ratios at each target-probe budget",
        "",
        "| Method | " + " | ".join(str(budget) for budget in _BUDGETS) + " |",
        "| --- | " + " | ".join("---:" for _ in _BUDGETS) + " |",
    ]
    for method in _H100_METHODS:
        points = _points(h100, method)
        values = " | ".join(_number(points[b]["mean_fraction_oracle"]) for b in _BUDGETS)
        lines.append(f"| {_label(h100, method)} | {values} |")
    paired = h100["headline"]["paired_auc_delta_vs_primary"]
    lines += [
        "",
        f"Retrieval-anchored Thompson sampling minus {_label(h100, paired['comparator'])}: "
        f"AUC difference {_number(paired['mean_auc_delta'])}, 95% Student-t interval "
        f"[{_number(paired['ci95_low'])}, {_number(paired['ci95_high'])}], "
        f"{paired['paired_seeds']} paired seeds.",
        "",
        "### Transfer parameters",
        "",
        f"Parameters were chosen on {selection['validation_gpu']} "
        f"({_link(_INPUTS[1])}). The final H100 run used these settings:",
        "",
        "| Method | Final settings |",
        "| --- | --- |",
    ]
    for method in ("single_source_nearest", "multisource_retrieval", "parhelion"):
        settings = json.dumps(h100["hyperparameters"][method], sort_keys=True)
        lines.append(f"| `{method}` | `{settings}` |")
    lines += [
        "",
        "T4 selected a transfer strength of "
        f"`{h100['hyperparameters']['parhelion']['transfer_strength']}`. "
        "The retrieval-anchored method still used its source-derived first probe, "
        "then adapted without a transferred posterior.",
        "",
        "## Earlier single-source transfer replay (separate protocol)",
        "",
        "These results exclude the held-out model family, but do not use the final H100 "
        "protocol's exact-target-shape exclusion. `nearest_shape` here is not the "
        "shape-disjoint `single_source_nearest` baseline above. Do not pool the rows "
        "across protocols or GPUs.",
        "",
        "| Source → target | Method (JSON key) | AUC 1–8 | Budget-8 ratio |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in _INPUTS[2:]:
        result = results[name]
        for method in _V1_METHODS:
            point = _points(result, method)[8]
            lines.append(
                f"| {result['source_gpu']} → {result['target_gpu']} | "
                f"{_label(result, method)} (`{method}`) | "
                f"{_number(result['primary_metrics']['fraction_reference_auc'][method])} | "
                f"{_number(point['mean_fraction_oracle'])} |"
            )
    lines += [
        "",
        "## Input keys and limits",
        "",
        f"- H100 AUC: {_link(_INPUTS[0])}, `auc.<method>`; budget ratios and intervals: "
        "`methods.<method>[budget == b].mean_fraction_oracle`, `ci95_low`, `ci95_high`. "
        "The paired difference is `headline.paired_auc_delta_vs_primary`.",
        f"- Earlier AUC: {_link(_INPUTS[2])} and {_link(_INPUTS[3])}, "
        "`primary_metrics.fraction_reference_auc.<method>`; budget ratios: "
        "`methods.<method>[budget == 8].mean_fraction_oracle`.",
        "- Method keys match the input JSON. Retrieval-only is `multisource_retrieval`; "
        "it probes source-ranked actions and returns the best measured incumbent "
        "without Thompson posterior adaptation.",
        "- H100 budget intervals are mean ± 1.96 standard errors, a normal approximation: "
        "across seeds after equal-fold averaging for stochastic methods, and across folds "
        "for deterministic ones. The paired AUC interval uses Student-t.",
        "- The 36 configurations cover only part of the legal Triton kernel space. "
        "Timings omit compilation, end-to-end serving overhead and production interference.",
        "- Probe budgets are simulated from saved exhaustive timing matrices. "
        "Source measurements and the two selection/scoring passes are outside that budget.",
        "",
        "Regenerate on CPU from the repository root:",
        "",
        "```sh",
        "nice -n 10 python scripts/summarize_tuner_results.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_REPO / "results/tuner-comparison.md")
    args = parser.parse_args(argv)
    summary = build_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
