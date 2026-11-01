# Evaluation methods

The original transfer experiments selected among 36 Triton matmul
configurations on a fixed corpus of 96 workloads: four model families, four
projection types and six token regimes. The later
[action-expansion comparison](../action-expansion.md) uses a separate collection
and a different selection procedure.

## Retrieval-anchored Thompson sampling

The retrieval index represents a workload by `log2(M, N, K) / 14` and uses
Euclidean distance with weights proportional to `exp(-distance / temperature)`.
Each source workload's action scores are centered log-TFLOP/s values.

For each target workload and candidate configuration, the index supplies four
features: weighted mean advantage, weighted variance, neighbor distance and
agreement across source GPUs. These augment the workload, launch and hardware
features used by Bayesian linear Thompson sampling. The first target query
measures the retrieval-selected action; later queries update the posterior
with target measurements. The recommendation is the best action actually
measured so far. See [retrieval.py](../../src/heliostune/retrieval.py) and
[multisource_engine.py](../../src/heliostune/multisource_engine.py).

The T4 comparison selected `k=16`, temperature `2.0` and transfer strength `0`
for the retrieval-anchored tuner. Retrieval-only selected `k=8` and temperature
`0.2`; pooled-source Thompson also selected transfer strength `0`. These
settings are in the [H100 protocol](../../benchmarks/parhelion-v2-h100-freeze.json).
The historical JSON keys, including `parhelion_thompson`, remain unchanged.

## Selection and scoring

The H100 replay uses three independently collected measurement sets, called
banks in the stored data:

| Stored key | Use |
|---|---|
| `bank=0` | Source observations, target queries and policy updates |
| `bank=1` | Select the fastest valid configuration in the 36-action reference set |
| `bank=2` | Score fixed recommendations and the selected reference |

Each fold holds out a model family. Before building source ranks,
normalization, retrieval features or priors, it excludes both that family and
every source row with an exact target `(M,N,K)` shape. The earlier L4/A10 study
held out families only, so its results are a separate comparison.

A budget of `b` means `b` distinct target probes per workload: `24b` per fold
and `96b` across all four folds. Each adaptive method shares one posterior
across the 24 workloads in a fold. At budget eight that is 768 simulated target
queries per method. The underlying collection was exhaustive: 31,104 source
measurements and 10,368 H100 measurements. Replay budgets exclude source
acquisition and the reference and scoring measurements.

L4 and A10 supplied the source archive for T4 parameter selection. The H100
comparison then used L4, A10 and T4 as sources, with 30 policy seeds and the
T4-selected settings. H100 was a held-out device domain on the same workload
corpus. `torch.matmul` was the selected performance comparator and remained
outside the tuner's action set.

## Ratios and uncertainty

For a workload, the historical transfer score is:

```text
reference latency on bank 2 / recommended-action latency on bank 2
```

Higher is better. The reference is the bank-1-selected Triton configuration.
A score of one means parity with that reference; it is not a hardware ceiling.
A library implementation outside the 36-action set can score above one.

Curves aggregate workload ratios geometrically, then average equally across
model-family folds and policy seeds. The reported AUC for budgets 1–8 is the
arithmetic mean of those eight points. It is not a trapezoidal integral.

The paired primary AUC interval uses Student-t uncertainty over policy seeds.
The stored budget intervals instead use mean ± 1.96 standard errors: across
seeds after fold averaging for stochastic methods, and across folds for
deterministic methods. The [comparison tables](../../results/tuner-comparison.md)
identify the exact result keys and formulas.

## Timing and numerical contracts

Compare candidate and baseline calls with the same inputs, output contract,
precision settings, allocation boundary and timing procedure. Include workspace
allocation and every kernel needed to produce the output. Keep compilation,
startup and tuning costs separate from warmed call medians.

The original matmul collections used FP16 allocating calls, 25 ms warmup,
100 ms timing and p20/median/p80 summaries. The later expansion uses a
correctness reference computed from the FP16 operands with FP32 output and
TF32 disabled. The historical collector instead created separate FP32-input
reference tensors and converted the result to FP16. Compare old and new
configurations within the expansion collection, rather than mixing their
latencies with the earlier runs.

For a new experiment, choose the workload weights, candidates, comparator,
correctness tolerance, practical performance margin and stopping rule before
collecting evaluation measurements. Keep configuration selection separate from
scoring, and keep failed compilation, correctness and timing attempts in the
output. A minimum selected on the scoring set is an optimistic diagnostic.

## Replaying the original H100 result

The following command belongs to the
[`v0.5.0` source snapshot](https://github.com/mottopanikeiku/quartet/tree/v0.5.0).
From that checkout, install the locked development environment and replay the
stored archive without a GPU:

```bash
nice -n 10 uv sync --locked --extra dev
nice -n 10 uv run heliostune compare-multisource \
  benchmarks/data/parhelion-v2-measurements.jsonl.zst \
  --sources L4,A10,T4 --target H100 --max-budget 8 --seeds 30 \
  --k 16 --temperature 2.0 --transfer-strength 0.0 \
  --retrieval-k 8 --retrieval-temperature 0.2 \
  --pooled-transfer-strength 0.0 --primary-comparator torch \
  --protocol-role final \
  --release-provenance benchmarks/parhelion-h100-release-provenance.json \
  --output artifacts/replay/h100-final-summary.json
nice -n 10 cmp artifacts/replay/h100-final-summary.json benchmarks/results/parhelion-h100-final.json
nice -n 10 uv run heliostune report artifacts/replay/h100-final-summary.json \
  --output artifacts/replay/h100-report.html
```

Use `artifacts/replay/`. The catalog reserves `artifacts/h100-final-summary.json`
and `artifacts/h100-report.html` as historical aliases that must remain absent.
For current analysis commands, see [Contributing](../../CONTRIBUTING.md).

## Collecting the original action set

`modal_bench.py` collects the original 36 configurations. A new collection
requires a new protocol, output path and explicit approval of its cost bound;
do not reuse a published study's output. From a clean committed checkout:

```bash
nice -n 10 uv run --locked python scripts/build_modal_wheel.py
nice -n 10 uv run --locked --extra modal modal run modal_bench.py \
  --gpus T4 --banks 0,1,2 --warmup-ms 25 --rep-ms 100 \
  --output artifacts/transfer-repeat/NAME.jsonl.zst
```

The collector validates the GPU before tensor allocation. Each call ID is
saved to the output's `.attempts.jsonl` journal before result retrieval; the
adjacent `.manifest.json` binds the request, wheel, sources, hardware and data.
Use `--resume-attempts PATH` to retrieve recorded calls after an interruption.
That path restores existing call IDs without spawning replacements.

## Interpretation

- Policy-seed intervals measure policy randomness conditional on the fixed
  matrices. They do not measure variation across workloads, GPU instances or
  collection sessions.
- Timing quantiles describe within-run spread. Population uncertainty needs
  paired timing samples and independently sampled sessions or devices.
- These are matmul configuration experiments on a fixed corpus. They do not
  measure end-to-end serving latency or compilation-time savings.
- The H200 engineering comparison followed a failed pilot and a changed retry
  rule. Its A100 selection data also mix PCIe and SXM devices. Keep it separate
  from the original H100 comparison.

The [study index](INDEX.md) links the original reports, results and protocols.
The [format reference](EXPERIMENT_SCOPE.md) documents the implemented
validation and replay interfaces.
