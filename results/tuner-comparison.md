# Tuner comparison

Cold Thompson sampling, nearest-shape reuse and cross-GPU transfer, compared using saved timing measurements.

## What the ratio measures

The reference is the fastest of the 36 Triton configurations, chosen on the selection pass and timed on the scoring pass. For each workload the ratio is reference latency / recommendation latency, both timed on the scoring pass. Higher is better. Policies see a separate timing pass. A ratio above 1 can come from a faster implementation outside the 36 configurations or variation between timing passes.

`torch.matmul` is a baseline; the tuner cannot choose it. Its constant curve is repeated at each budget for comparison.

AUC 1–8 is the arithmetic mean of the eight reported budget ratios, not a trapezoidal integral. The stored curves first aggregate workload ratios geometrically, then average equally across model-family folds and policy seeds.

## H100: shape-disjoint multi-source replay

Source: [parhelion-h100-final.json](../benchmarks/results/parhelion-h100-final.json). Target: H100; source GPUs: L4, A10, T4; 96 workloads, 4 model families, 30 policy seeds. Source archives exclude both the held-out model family and exact target shapes.

Each probe times one configuration on a held-out workload. For each seed, Thompson-sampling methods share a posterior across workloads and update it in paired batches.

| Method (JSON key) | AUC 1–8 | Budget-8 ratio | Budget-8 95% interval |
| --- | ---: | ---: | --- |
| Cold Thompson sampling (`cold_thompson`) | 0.958434 | 0.996774 | [0.996305, 0.997243] |
| Single-source nearest-shape reuse (`single_source_nearest`) | 0.903481 | 0.967012 | [0.965321, 0.968703] |
| Multi-source retrieval (`multisource_retrieval`) | 0.903191 | 0.940747 | [0.932537, 0.948957] |
| Retrieval-anchored Thompson sampling (`parhelion_thompson`) | 0.950259 | 0.996523 | [0.996193, 0.996852] |
| torch.matmul (`torch`) | 1.610259 | 1.610259 | [1.578385, 1.642134] |

### Ratios at each target-probe budget

| Method | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold Thompson sampling | 0.795474 | 0.937754 | 0.973266 | 0.984639 | 0.990430 | 0.993617 | 0.995515 | 0.996774 |
| Single-source nearest-shape reuse | 0.780684 | 0.826188 | 0.869107 | 0.909309 | 0.954383 | 0.958733 | 0.962429 | 0.967012 |
| Multi-source retrieval | 0.822779 | 0.859410 | 0.909368 | 0.913814 | 0.920401 | 0.925783 | 0.933229 | 0.940747 |
| Retrieval-anchored Thompson sampling | 0.822779 | 0.888394 | 0.948305 | 0.973530 | 0.985897 | 0.991791 | 0.994855 | 0.996523 |
| torch.matmul | 1.610259 | 1.610259 | 1.610259 | 1.610259 | 1.610259 | 1.610259 | 1.610259 | 1.610259 |

Retrieval-anchored Thompson sampling minus torch.matmul: AUC difference -0.660000, 95% Student-t interval [-0.661424, -0.658576], 30 paired seeds.

### Transfer parameters

Parameters were chosen on T4 ([parhelion-t4-selection.json](../benchmarks/results/parhelion-t4-selection.json)). The final H100 run used these settings:

| Method | Final settings |
| --- | --- |
| `single_source_nearest` | `{"contract": "parameter-free one-nearest-workload retrieval", "k": 1, "source_gpu": "L4", "temperature": 1.0}` |
| `multisource_retrieval` | `{"k": 8, "temperature": 0.2}` |
| `parhelion` | `{"k": 16, "temperature": 2.0, "transfer_strength": 0.0}` |

T4 selected a transfer strength of `0.0`. The retrieval-anchored method still used its source-derived first probe, then adapted without a transferred posterior.

## Earlier single-source transfer replay (separate protocol)

These results exclude the held-out model family, but do not use the final H100 protocol's exact-target-shape exclusion. `nearest_shape` here is not the shape-disjoint `single_source_nearest` baseline above. Do not pool the rows across protocols or GPUs.

| Source → target | Method (JSON key) | AUC 1–8 | Budget-8 ratio |
| --- | --- | ---: | ---: |
| L4 → A10 | Cold Thompson sampling (`cold_thompson`) | 0.901771 | 0.925325 |
| L4 → A10 | Nearest-shape reuse (`nearest_shape`) | 0.966882 | 0.995416 |
| L4 → A10 | Transfer Thompson sampling (`transfer_thompson`) | 0.923658 | 0.933853 |
| A10 → L4 | Cold Thompson sampling (`cold_thompson`) | 0.947153 | 0.956246 |
| A10 → L4 | Nearest-shape reuse (`nearest_shape`) | 0.989635 | 0.996003 |
| A10 → L4 | Transfer Thompson sampling (`transfer_thompson`) | 0.948280 | 0.956900 |

## Input keys and limits

- H100 AUC: [parhelion-h100-final.json](../benchmarks/results/parhelion-h100-final.json), `auc.<method>`; budget ratios and intervals: `methods.<method>[budget == b].mean_fraction_oracle`, `ci95_low`, `ci95_high`. The paired difference is `headline.paired_auc_delta_vs_primary`.
- Earlier AUC: [l4-to-a10.json](../benchmarks/results/l4-to-a10.json) and [a10-to-l4.json](../benchmarks/results/a10-to-l4.json), `primary_metrics.fraction_reference_auc.<method>`; budget ratios: `methods.<method>[budget == 8].mean_fraction_oracle`.
- Method keys match the input JSON. Retrieval-only is `multisource_retrieval`; it probes source-ranked actions and returns the best measured incumbent without Thompson posterior adaptation.
- H100 budget intervals are mean ± 1.96 standard errors, a normal approximation: across seeds after equal-fold averaging for stochastic methods, and across folds for deterministic ones. The paired AUC interval uses Student-t.
- The 36 configurations cover only part of the legal Triton kernel space. Timings omit compilation, end-to-end serving overhead and production interference.
- Probe budgets are simulated from saved exhaustive timing matrices. Source measurements and the two selection/scoring passes are outside that budget.

Regenerate on CPU from the repository root:

```sh
nice -n 10 python scripts/summarize_tuner_results.py
```
