# HeliosTune

Transfer a Bayesian autotuning prior between GPU targets, spend a fixed number of target-side probes, and test whether it actually beats simpler launch-selection methods.

**Measured report:** [L4 → A10](https://mottopanikeiku.github.io/heliostune/) · [A10 → L4](https://mottopanikeiku.github.io/heliostune/a10-to-l4.html) · [raw release manifest](benchmarks/manifest.json)

HeliosTune implements a manual FP16 Triton matmul and a Gaussian linear Thompson sampler over joint workload, launch, and hardware descriptors. The source posterior becomes a discounted power prior on the target; each target observation is incorporated with a rank-one precision update. The study compares it with static source-best selection, random search, nearest-shape reuse, cold-start Thompson sampling, PyTorch, and exhaustive tuning over the frozen manifest.

## Result

The transfer prior helped over cold start in both directions. It did **not** beat nearest-shape reuse. That negative result is the important one: on this action space, a simple retrieval heuristic carried source information more effectively than the linear Bayesian model.

| Direction | Static | Random | Cold Thompson | **Helios transfer** | **Nearest shape** |
|---|---:|---:|---:|---:|---:|
| L4 → A10 | 83.05% | 87.94% | 90.18% | **92.37%** | **96.69%** |
| A10 → L4 | 90.78% | 91.14% | 94.72% | **94.83%** | **98.96%** |

Values are mean fraction-of-held-out-reference AUC across target budgets 1–8; higher is better. Helios improved on cold Thompson by 2.19 percentage points for L4 → A10 and 0.11 points for A10 → L4, but trailed nearest-shape reuse by 4.32 and 4.14 points respectively. No subset or favorable direction is hidden.

## Experimental controls

- **96 model-derived workloads:** four projection types × six token regimes × four public model families.
- **36 frozen launch configurations:** `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, warps, stages, and grouping vary without target-latency pruning.
- **Three timing banks:** bank 0 is policy-visible, bank 1 selects the best-of-manifest reference, and bank 2 evaluates every recommendation.
- **Grouped transfer folds:** one complete model family is held out at a time. Its shapes never enter the source posterior or source baselines.
- **Paired replay:** 30 policy seeds, identical per-round workload permutations, and budgets of 1–8 distinct configuration probes per held-out workload.
- **Independent evaluation:** recommendations and the bank-1 reference winner are both scored only on bank 2.
- **Numerical gate:** every one of the 20,736 measured GPU/configuration/workload/bank records passed the FP32-reference correctness check.

The frozen protocol is commit [`5919cbb`](https://github.com/mottopanikeiku/heliostune/commit/5919cbb4a9d7684ac835ab7bfd89879ac8c82344), which predates the full target collection. The [Modal run](https://modal.com/apps/mottopanikeiku/main/ap-ccgVXq137C9p6vVUxPlvXA) allocated actual **NVIDIA L4** and **NVIDIA A10** devices. The compressed JSONL matrix, checksum, exact software versions, timing durations, and device properties are published under [`benchmarks/`](benchmarks/).

## Method

For workload $s$, launch action $a$, and GPU descriptor $h$, HeliosTune constructs a bounded feature vector $\phi(s,a,h)$. The linear reward model is

$$
y = \phi(s,a,h)^\top \theta + \epsilon, \qquad \epsilon \sim \mathcal{N}(0,\sigma^2).
$$

A source observation contributes the sufficient statistics

$$
\Lambda \leftarrow \Lambda + \sigma^{-2}\phi\phi^\top,
\qquad
\eta \leftarrow \eta + \sigma^{-2}\phi y.
$$

Target initialization uses a power prior with frozen strength $\alpha=0.08$:

$$
\Lambda_{t,0}=\lambda I+\alpha(\Lambda_s-\lambda I),
\qquad
\eta_{t,0}=\alpha\eta_s.
$$

This discounts source likelihood information without double-counting the source ridge prior. Thompson samples and posterior means use Cholesky solves; the implementation never forms a covariance inverse.

The online reward is `-log(latency_ms)`. PyTorch timing is evaluation-only and is not an uncharged calibration input to the policy.

## Workload corpus

Dimensions come directly from public model configurations:

- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json)
- [Qwen2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json)
- [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json)
- [Granite 3.1 8B](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct/blob/main/config.json)

Each family contributes fused attention QKV, attention output, FFN up, and FFN down projections at token counts 1, 7, 31, 96, 257, and 1024. Irregular counts prevent a tile-aligned-only benchmark.

## Run locally

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are supported.

```bash
uv sync --extra dev
uv run pytest
uv run heliostune demo --output-dir artifacts/demo
```

The demo produces deterministic synthetic measurements, a replay summary, and a standalone offline report. Synthetic reports are visibly labeled and support no hardware-performance claims.

Analyze a published matrix:

```bash
zstd -d benchmarks/data/measurements.jsonl.zst -o measurements.jsonl
uv run heliostune inspect measurements.jsonl
uv run heliostune compare measurements.jsonl \
  --source L4 --target A10 --output summary.json
uv run heliostune report summary.json --output index.html
```

## Collect on Modal

```bash
uv sync --extra gpu
modal setup
uv run modal run modal_bench.py \
  --replicates 3 \
  --warmup-ms 25 \
  --rep-ms 100 \
  --output artifacts/measurements.jsonl
```

The runner launches L4 and A10 jobs concurrently for every bank, reads the actual device properties at runtime, randomizes workload/configuration order per bank, compiles and checks before timing, and records p20/p50/p80 steady-state latency. A small two-workload pilot is available with `--pilot`.

## Repository map

- `src/heliostune/kernel.py` — manual grouped-program Triton matmul and collector
- `src/heliostune/bandit.py` — Gaussian posterior and discounted transfer prior
- `src/heliostune/replay.py` — leak-resistant folds, policies, baselines, and metrics
- `src/heliostune/report.py` — self-contained high-signal HTML report
- `modal_bench.py` — concurrent Modal collection on L4 and A10
- `benchmarks/` — manifest, compressed measurements, and directional summaries
- `tests/` — manifest, posterior, replay-isolation, and report contracts

## Scope

This is a steady-state FP16 microkernel configuration-selection study on two GPU models and a curated 36-arm space. It does not establish generalization to arbitrary unseen GPUs, global Triton optimality, compilation-time savings, end-to-end serving gains, or production interference robustness. Source acquisition costs 2,592 visible observations per fold and is disclosed separately from the target query budget.

## License

MIT
