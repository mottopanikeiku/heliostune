# quartet

I built a Triton matmul autotuner and compared it with `torch.matmul` on 96 LLM-shaped workloads across T4, A10, L4 and H100. The original kernels land within 3% of torch on A10 and L4; on H100, adding ten kernel configurations cut geometric-mean latency by 27%.

[![H100 latency relative to torch, before and after expanding the kernel set](site/h100-by-m.svg)](https://mottopanikeiku.github.io/quartet/)

Picking the best of all 36 kernels on one timing pass and re-timing on another lands within 0.7% of the best kernel on the second pass, on every GPU. The large T4 and H100 gaps come from the kernels, not the search.

[Full report](https://mottopanikeiku.github.io/quartet/) · [Measurement method](docs/action-expansion.md) · [Contributing](CONTRIBUTING.md)

## Four GPUs, the same 96 workloads

The suite covers attention and FFN projections from Granite, Mistral, Phi and Qwen at M = 1, 7, 31, 96, 257 and 1024. There are 84 distinct (M, N, K) shapes; some models share a shape.

These are the original 36 configurations, exhaustively selected on timing pass 1 and scored on pass 2. Ratios are geometric means over the 96 workloads; lower is faster. The last column compares that selection with the fastest configuration on the scoring pass.

| GPU | Triton / torch latency | Selection overhead |
|---|---:|---:|
| T4 | 3.840× | 0.65% |
| A10 | 1.021× | 0.32% |
| L4 | 0.986× | 0.37% |
| H100 | 1.610× | 0.08% |

On L4, Triton wins on average for M ≤ 257 and loses at M = 1024. On A10 it wins for M ≤ 96 and loses on the larger matrices. H100 and T4 need better kernels across all six M groups. [Per-GPU and per-shape results](results/kernel-bank-diagnosis.json).

## Closing the H100 gap

I added four larger tiles, two persistent TMA configurations and four tensor-core split-K configurations. Remeasuring the old and new sets in the same H100 session brought latency from 1.61× to 1.17× torch. The new configurations improve 95 of 96 workloads.

| H100 kernel set | Latency / torch | Wins / 96 |
|---|---:|---:|
| Original 36 | 1.608× | 0 |
| Ten new configurations | 1.174× | 9 |
| Combined set | 1.173× | 9 |

Six wins use split-K and three use larger ordinary tiles. The largest saving is 5.55% on Qwen's FFN-down projection at `(M, N, K) = (1, 3584, 18944)`: 63.456 µs versus torch's 67.184 µs. Persistent TMA wins none. [All nine wins and selected configurations](results/action-expansion-summary.json) · [Every workload plotted](results/action-expansion.svg).

## How the tuner works

- The original Triton kernel varies tile sizes, warp count, pipeline depth and grouped program ordering across 36 configurations.
- Linear Thompson sampling probes configurations and updates a latency model. Retrieval can seed the search with measurements from other GPUs, with held-out model families and exact target shapes excluded in the H100 comparison.
- Search uses one timing pass; a separate pass selects the exhaustive reference, and a third scores both recommendations and the reference. The H100 kernel expansion instead uses two passes: select old and new winners, then time only those fixed winners alongside torch.
- Split-K divides the reduction across programs, writes FP32 partial tiles and reduces them in a second Triton kernel. Timing includes allocations, workspace and both launches. The persistent implementation builds on [Triton's persistent-matmul tutorial](https://github.com/triton-lang/triton/blob/v3.4.0/python/tutorials/09-persistent-matmul.py).

In the original H100 replay, cold Thompson sampling reaches 99.7% of the exhaustive reference's speed after eight probes out of 36. Cross-GPU posterior transfer did not improve it; the selected transfer strength was zero. That result led me to work on the kernel set. [Policy comparison](results/tuner-comparison.md).

The implementation is in [`bandit.py`](src/heliostune/bandit.py), [`retrieval.py`](src/heliostune/retrieval.py), [`kernel.py`](src/heliostune/kernel.py) and [`action_expansion.py`](src/heliostune/action_expansion.py).

## Run it

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are required. The installed package and CLI are `heliostune`.

Replay the original A10 → L4 comparison on CPU, using one policy seed for a quick run:

```sh
uv sync --locked --extra dev
uv run heliostune compare benchmarks/data/measurements.jsonl.zst --source A10 --target L4 --max-budget 8 --seeds 1 --output artifacts/a10-to-l4.json
uv run python scripts/build_site.py --check
```

Use `--seeds 30` for the original comparison. The report builder uses the stored summaries; omit `--check` to rebuild the page and figure. [Contributor instructions](CONTRIBUTING.md) cover the full analyses and tests.

To inspect the local H100 collection without starting a GPU run:

```sh
uv run python scripts/collect_action_expansion.py --run-id h100-repeat-1 --gpu H100 --dry-run
```

On a CUDA machine, install the `gpu` extra and remove `--dry-run` to collect timings. Start from a clean commit. Each run ID has its own checkpoint; resuming requires the same code, measurement settings and hardware profile. The measurements here use Torch 2.8.0, Triton 3.4.0, FP16 operands, 25 ms warmup and 100 ms timing windows.

## What these measurements cover

- Warmed, allocating matmul calls. Compilation, tuning cost and end-to-end serving latency are separate.
- The H100 expansion is one device session. Five of its nine torch wins are below 1%; individual timing samples were not retained.
- The H100 expansion checks FP16 inputs against FP32-output matmul at `atol=rtol=0.01`. Torch's reduced-FP16 reduction flag was enabled.
- The four-GPU table measures exhaustive selection. Thompson sampling and transfer have not been rerun on the expanded kernel set.
- The baseline is `torch.matmul`, without separate cuBLASLt/CUTLASS tuning, preallocated outputs or CUDA graphs.

Earlier transfer, H200 and fusion experiments are linked from the [report](https://mottopanikeiku.github.io/quartet/#archive). Their result files remain unchanged.

MIT [license](LICENSE).
