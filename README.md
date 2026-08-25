# HeliosTune

A measured study of retrieval, Bayesian adaptation, and source-to-target launch selection for FP16 Triton matrix multiplication.

**Live evidence:** [Parhelion L4+A10+T4 → H100 report](https://mottopanikeiku.github.io/heliostune/) · [post-hoc v2 causal addendum](https://mottopanikeiku.github.io/heliostune/parhelion-v2-addendum.html) · [v3 pilot failure evidence](benchmarks/parhelion-v3-validation-failure.json) · [artifact catalog](benchmarks/research-artifact-manifest.json) · [v1 L4 ↔ A10 report](https://mottopanikeiku.github.io/heliostune/v1.html) · [post-run chain of custody](benchmarks/parhelion-v2-post-run-manifest.json) · [H100 freeze](benchmarks/parhelion-v2-h100-freeze.json)

## Result

Parhelion is a retrieval-anchored Bayesian linear Thompson tuner built in response to the v1 result: nearest-shape reuse beat the original transferred posterior in both L4↔A10 directions. Parhelion converts a family- and shape-disjoint multi-GPU archive into four action-conditioned retrieval statistics, pays for the consensus retrieval action as query one, then adapts on target bank-0 observations.

The staged result is negative under its frozen primary comparison. On the untouched Modal H100 domain, Parhelion reached **99.65%** of the held-out curated Triton reference at eight probes per workload and **0.9503 AUC** over budgets 1–8. It did **not** outperform the T4-frozen `torch.matmul` comparator: paired AUC delta **−0.6600**, two-sided 95% Student-t Monte Carlo interval **[−0.6614, −0.6586]** over 30 paired policy seeds, conditional on the fixed matrix, corpus, archive, and campaign. Superiority was not demonstrated.

| H100 method | AUC, budgets 1–8 | Fraction of reference at budget 8 |
|---|---:|---:|
| Static multi-source best | 63.40% | 63.40% |
| Random search | 84.19% | 92.82% |
| Multi-source retrieval | 90.32% | 94.07% |
| Single-source nearest shape | 90.35% | 96.70% |
| **Parhelion** | **95.03%** | **99.65%** |
| Cold Thompson | 95.84% | 99.68% |
| Pooled-source Thompson | 95.84% | 99.68% |
| `torch.matmul` | 161.03% | 161.03% |

Parhelion improves on its retrieval-only anchor by **4.71 percentage points AUC**, but trails cold Thompson by **0.82 points**. It reaches 95% of the reference after four probes; cold Thompson needs three and nearest-shape reuse needs five. The selected pooled transfer strength and Parhelion source-likelihood strength are both zero, so this corpus does not support a positive transferred-posterior claim. Parhelion still uses the frozen source archive to construct its retrieval anchor and retrieval covariates.

The [post-hoc causal addendum](site/parhelion-v2-addendum.html) does not alter that confirmatory endpoint. With the same paid retrieval action at budget one, Parhelion's AUC was **0.00368 lower** than anchored cold Thompson; the exploratory paired policy-seed interval was **[−0.00566, −0.00169]**. Every new contrast is labeled `post_hoc_exploratory`, carries no superiority claim, and reuses the immutable H100 matrix without selection or recollection.

## Parhelion v3 campaign outcome

The predeclared H200 campaign terminated at its single L4 pilot FunctionCall, before any timing row was returned. The remote container failed while importing `modal_bench.py` because it searched a container-relative local wheel directory. Under the protocol's pre-H200 failure rule, that FunctionCall was not retried; the candidate matrix, A100 validation, freeze, H200 invocation, and v3 performance report were not produced. This is the complete v3 campaign outcome, not a hardware-performance result.

The [validation-failure manifest](benchmarks/parhelion-v3-validation-failure.json) binds the frozen development protocol, failed commit and wheel, Modal app and FunctionCall IDs, two-record append-only journal, observed import error, zero-spawn recovery, and every absent downstream artifact. Release 0.4.0 publishes the software, existing causal addendum, protocol, and failure evidence. It also corrects remote wheel discovery for future independent campaigns without changing the failed protocol bytes or assigning a result retroactively.

Values above are fractions of a bank-1-selected, bank-2-scored best configuration from the curated 36-action Triton manifest. They are not fractions of a hardware ceiling. `torch.matmul` can exceed 1.0 because it is outside that manifest and is evaluation-only.

## What changed from v1

The [v1 bidirectional study](https://mottopanikeiku.github.io/heliostune/v1.html) measured 20,736 L4/A10 rows. The original transferred linear posterior improved over cold start but lost to nearest-shape reuse:


| Direction | Cold Thompson | Helios transfer | Nearest shape |
|---|---:|---:|---:|
| L4 → A10 | 90.18% | 92.37% | **96.69%** |
| A10 → L4 | 94.72% | 94.83% | **98.96%** |

The historical v1 replay held out model families only. Parhelion v2 uses the stricter family-plus-exact-shape split: before any source-derived rank, normalization, feature, or posterior is built, it also excludes source rows sharing a held-out target `(M,N,K)` shape. The v1 bytes and claims remain unchanged.

Parhelion makes retrieval explicit rather than hiding it inside a learned prior:

1. Compute frozen Euclidean neighbor distance over `log2(M,N,K)/14`.
2. Convert each source workload/action row to centered log-TFLOP/s advantage.
3. Append weighted advantage mean, weighted variance, neighbor distance, and source-GPU sign agreement to the joint workload/launch/hardware features.
4. Query the independently T4-selected consensus retrieval action first.
5. Use a target-updated Bayesian linear Thompson posterior for later probes and recommend only the best measured bank-0 incumbent.

The closest ideas are established: nearest-task warm starts, transferred cost models, residual transfer, and contextual linear Thompson sampling. Relevant prior work includes [AutoTVM](https://arxiv.org/abs/1805.08166), [nearest-dataset SMBO initialization](https://ojs.aaai.org/index.php/AAAI/article/view/9354), [linear Thompson sampling](https://proceedings.mlr.press/v28/agrawal13.html), [RGPE/TST-R transfer Bayesian optimization](https://arxiv.org/abs/1802.02219), and [Transfer-Tuning](https://arxiv.org/abs/2201.05587). The bounded contribution here is the exact forced-anchor, centered multi-GPU retrieval representation, target-only update path, and staged hardware evaluation—not a claim to be the first transfer autotuner or a new regret theorem.

## Frozen evaluation

- **Corpus:** 96 workloads = four model families × four projection types × six token regimes.
- **Action set:** 36 launch configurations varying tiles, warps, stages, and grouping.
- **Timing banks:** bank 0 is policy-visible; bank 1 selects the manifest reference; bank 2 scores recommendations.
- **Leakage control:** each fold excludes the complete target model family and every source row sharing an exact target `(M,N,K)` shape before retrieval, centering, source modeling, or selection.
- **Shared adaptation:** budget `b` means `b` target probes per workload. A fold uses `24b` probes; all four folds use `96b`. One posterior is shared across the 24 workloads in a fold, so this is batched adaptation, not 96 independent tuners.
- **T4 validation:** L4+A10 source archive; 12 seeds; method-local retrieval (12 points), pooled-source (4 strengths), and Parhelion (48 points) grids. Selected Parhelion `(k=16, T=2.0, α=0)`, retrieval `(k=8, T=0.2)`, pooled `α=0`, and primary comparator `torch`.
- **H100 final:** H100 is an untouched hardware/timing matrix on the already fixed 96-workload corpus, not an unseen-workload study. L4+A10+T4 are sources; 30 seeds, selected parameters, and the comparator were frozen before the sole H100 invocation.
- **Physical cost:** 31,104 source measurements plus 10,368 H100 measurements. The simulated budget-8 online cost is 768 target queries per live method across all folds; physical target collection remains exhaustive.
- **Numerical gate:** all 41,472 four-GPU cells passed the FP32-reference correctness check.

Parhelion and retrieval-only make the same paid query at budget one: both score 0.82277885 of the held-out reference on H100. The report exposes all four held-out-family tables, every method/budget point, exact-shape exclusions, source rows, and the target-selected strongest method as descriptive-only context. Stochastic intervals are policy-seed Monte Carlo intervals conditional on the fixed data and campaign; deterministic fold ranges are descriptive, not confidence intervals.

## Chain of custody

- Algorithm and development protocol: [`811b05b`](https://github.com/mottopanikeiku/heliostune/commit/811b05bb65bc978e44ca8fa32ceeeab315acf391)
- Executable H100 freeze: [`c395630`](https://github.com/mottopanikeiku/heliostune/commit/c395630b9bcb4ef6a501d9a34696783620381c3c), SHA-256 `c9c7138e…`
- Persisted T4 validation run: [Modal `ap-qxI4D2…`](https://modal.com/apps/mottopanikeiku/main/ap-qxI4D2xUvtPfuhtgiSfVgm)
- Sole H100 run: [Modal `ap-y68ldw…`](https://modal.com/apps/mottopanikeiku/main/ap-y68ldw4RUmTotSEIxGdqPz), exact selector `H100!`
- Raw H100 SHA-256: `747f30a9…`
- Four-GPU replay archive SHA-256: `f417bd7e…`
- Final summary SHA-256: `765b347a…`

The hashed [post-run manifest](benchmarks/parhelion-v2-post-run-manifest.json) binds the exact historical runs, commits, commands, compressed and uncompressed data, selection, summary, and report. The [pre-H100 freeze](benchmarks/parhelion-v2-h100-freeze.json) records the no-pilot/no-rerun rule, hardware identity gate, selected parameters, source order, seeds, budgets, collector settings, failure rule, and implementation/data digests. The [research artifact catalog](benchmarks/research-artifact-manifest.json) verifies every historical and published alias digest; the separate [addendum manifest](benchmarks/parhelion-v2-addendum-manifest.json) binds the immutable input, implementation, exploratory result, and new report without touching historical bytes.

The v3 [development protocol](benchmarks/parhelion-v3-development-protocol.json), [terminal failure manifest](benchmarks/parhelion-v3-validation-failure.json), and [attempt journal](benchmarks/data/parhelion-v3-pilot-failure.attempts.jsonl) preserve the stopped H200 campaign. The catalog binds all three; no candidate, A100, or H200 performance artifact exists.

## Run locally

Python 3.11–3.13 and uv 0.12.5 are supported. From a clean clone:

```bash
git clone https://github.com/mottopanikeiku/heliostune.git
cd heliostune
uv sync --locked --extra dev
uv run heliostune --help
uv run heliostune --version
uv run heliostune demo --output-dir /tmp/heliostune-demo --max-budget 2 --seeds 2
uv run heliostune inspect /tmp/heliostune-demo/measurements.jsonl
```

Native zstandard inspection needs no external `zstd` executable:

```bash
uv run heliostune inspect benchmarks/data/parhelion-v2-measurements.jsonl.zst
```

The stable root API exposes strict artifact I/O without importing replay or reporting:

```python
from heliostune import __version__, read_measurements, write_measurements_atomic

rows = read_measurements("benchmarks/data/t4-measurements.jsonl.zst")
write_measurements_atomic("/tmp/t4-copy.jsonl.zst", rows)
print(__version__, len(rows))
```

Run the complete CPU quality gates with:

```bash
uv lock --check
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run coverage run --branch -m pytest
uv run coverage report
```

Verify the catalog and byte-regenerate the exploratory addendum:

```bash
uv run heliostune verify-catalog benchmarks/research-artifact-manifest.json
uv run python scripts/build_parhelion_v2_addendum.py --check
```

Reproduce the frozen H100 replay directly from the compressed archive:

```bash
uv run heliostune compare-multisource \
  benchmarks/data/parhelion-v2-measurements.jsonl.zst \
  --sources L4,A10,T4 --target H100 --max-budget 8 --seeds 30 \
  --k 16 --temperature 2.0 --transfer-strength 0.0 \
  --retrieval-k 8 --retrieval-temperature 0.2 \
  --pooled-transfer-strength 0.0 --primary-comparator torch \
  --protocol-role final --output artifacts/h100-final-summary.json
uv run heliostune report artifacts/h100-final-summary.json \
  --output artifacts/h100-report.html
```

The local `heliostune demo` is synthetic and supports no hardware claim.

## Collect on Modal

Build the exact committed wheel before any paid call, then invoke the durable bank protocol:

```bash
uv run python scripts/build_modal_wheel.py
uv run --extra modal modal run modal_bench.py \
  --gpus T4 --banks 0,1,2 --warmup-ms 25 --rep-ms 100 \
  --output artifacts/t4-measurements.jsonl.zst
```

Output parents are created automatically. Each call ID is fsynced to
`${output}.attempts.jsonl` before any result retrieval; `${output}.manifest.json`
binds the request, journal, wheel, source, hardware, and final data digests. Resume
an interrupted retrieval with `--resume-attempts PATH`; it reconstructs recorded
`FunctionCall` IDs and spawns no replacements.

`modal_bench.py` gates `L4`, `A10`, `T4`, `H100`, `A100-80GB`, and `H200` identities before tensor allocation. H100 uses Modal's exact `H100!` selector. Do not rerun the published H100 protocol; this command is for an independent study.

## Repository map

- `src/heliostune/artifacts.py` — strict JSON/JSONL decoding and atomic zstandard persistence
- `src/heliostune/collection.py` — paid-call planning, fsynced attempt journals, resume, and commit
- `src/heliostune/hardware.py` — pure fleet identity and memory gates
- `src/heliostune/retrieval.py` — shape index and four action-conditioned archive statistics
- `src/heliostune/multisource.py` — public multi-source replay facade
- `src/heliostune/multisource_engine.py` — prepared folds and method-local evaluators
- `src/heliostune/v2_addendum.py` — frozen v2 causal ablations and workload endpoints
- `src/heliostune/uncertainty.py` — policy-seed intervals and deterministic fold summaries
- `src/heliostune/selection.py` — strict staged T4 selector
- `src/heliostune/bandit.py` — atomic Gaussian information-form posterior updates
- `src/heliostune/kernel.py` — manual Triton matmul and measured collector
- `src/heliostune/report_model.py` — immutable renderer input contract
- `src/heliostune/report.py` — self-contained offline evidence report
- `scripts/build_modal_wheel.py` — reproducible committed-wheel builder
- `scripts/build_parhelion_v2_addendum.py` — byte-exact exploratory result/report builder
- `scripts/verify_research_artifacts.py` — full catalog, alias, count, and frozen-point verifier
- `scripts/assemble_parhelion_final.py` — historical v2 archive verifier
- `benchmarks/` — frozen protocols, chain manifests, compressed matrices, selections, and results
- `site/` — offline final report, downloadable JSON, and archived v1 report

## Scope

This is a steady-state FP16 microkernel configuration-selection study over one fixed 96-workload corpus, one curated 36-arm space, and four Modal GPU fleets. It does not establish generalization to arbitrary GPUs or model families, global Triton optimality, Bayesian calibration, compilation-time savings, end-to-end serving gains, or production interference robustness. T4 is a validation domain, not independent final evidence. H100 is one untouched hardware domain, not proof of universal cross-architecture transfer.

## License

MIT
