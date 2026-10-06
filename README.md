# HeliosTune

HeliosTune is a small GPU matrix-multiplication autotuning experiment using nearby-shape retrieval and Bayesian linear Thompson sampling.

The separate Hopper candidate implementation follows [Triton's persistent-matmul tutorial](https://github.com/triton-lang/triton/blob/v3.4.0/python/tutorials/09-persistent-matmul.py), as attributed in [`hopper_kernel.py`](src/heliostune/hopper_kernel.py).

**Question:** can measurements from other GPU models reduce the number of target-GPU probes needed to choose a fast configuration?

The tuner builds retrieval features from nearby matrix shapes in [`retrieval.py`](src/heliostune/retrieval.py), then updates a target posterior in [`bandit.py`](src/heliostune/bandit.py). [`multisource_engine.py`](src/heliostune/multisource_engine.py) runs the multi-GPU comparison; its retrieval-anchored policy is called Parhelion.

**Result: transferred-posterior gains were unsupported.** On H100, cold Thompson beat Parhelion, and `torch.matmul` beat both by a large margin. The action set, not just the search policy, needs work.

## Results

From the committed [H100 result](benchmarks/results/parhelion-h100-final.json), summarized by the new [CPU-only comparison table](results/tuner-comparison.md):

| H100 method | Mean score over probe budgets 1–8 | Score at budget 8 |
|---|---:|---:|
| Cold Thompson | 0.9584 | 0.9968 |
| Single-source nearest-shape reuse | 0.9035 | 0.9670 |
| Retrieval-only | 0.9032 | 0.9407 |
| Parhelion | 0.9503 | 0.9965 |
| `torch.matmul` | **1.6103** | **1.6103** |

**The denominator is a held-out reference from a curated 36-configuration Triton action set, not the best possible kernel or a hardware ceiling.** Bank 1 selects that reference; bank 2 scores recommendations. `torch.matmul` is an evaluation-only comparator outside the action set, so scores above 1 are possible. “99.65% of reference” does not mean Parhelion approaches vendor-library performance.

Both selected transfer strengths were zero ([T4 selection](benchmarks/results/parhelion-t4-selection.json)). Parhelion still uses source measurements for its first action and retrieval features; the result does not show that retrieval is useless. In the earlier, less strict L4/A10 study, nearest-shape reuse beat the transferred posterior in both directions ([L4 → A10](benchmarks/results/l4-to-a10.json), [A10 → L4](benchmarks/results/a10-to-l4.json)).

The [H100 exploratory addendum](benchmarks/results/parhelion-v2-addendum.json) and [later H200 engineering result](benchmarks/results/parhelion-v3-h200-engineering.json) are separate analyses, not replacements for this negative primary comparison.

## Reproduce the comparison

From a checkout, with Python installed:

```bash
nice -n 19 python3 scripts/summarize_tuner_results.py
uv sync --locked --extra dev
nice -n 19 uv run --locked pytest -q -x tests/test_tuner_results.py tests/test_bandit.py tests/test_retrieval.py
```

The first command regenerates the table from committed results using only the standard library: local CPU, no GPU, no paid service. It summarizes existing measurements; it does not recollect timings or rerun the policies. The other commands install the locked CPU development environment and test the summary and tuner behavior. Original collection required the GPUs described in the retained study documents.

## Code and next experiment

The tuner remains in `src/heliostune/`. Report generation, run-record checking and offline bundle analysis are separated into [`src/heliostune/tooling/`](src/heliostune/tooling/). Existing benchmark data and published reports are retained. [Historical study documents](docs/history/INDEX.md) record the original protocols, commands and later engineering attempts.

The next useful experiment is a **wider action set with `torch.matmul` treated as a real baseline**, not another transfer variant. [The plan](docs/NEXT.md) specifies the comparison and what can be done before paying for GPU time.

## Limitations

- One fixed workload corpus and collected GPU instances; these are not population estimates.
- H100 holds out hardware measurements, not a new workload corpus; source filtering excludes target families and exact shapes.
- A posterior is shared across workloads in each fold, rather than tuning each workload independently.
- Policy-seed intervals condition on the fixed measurements; they do not capture new-device variability.
- The earlier L4/A10 study and later H200 engineering run use different protocols and should not be pooled.

## Prior work

The experiment builds on [AutoTVM](https://arxiv.org/abs/1805.08166), [nearest-dataset initialization](https://ojs.aaai.org/index.php/AAAI/article/view/9354), [linear Thompson sampling](https://proceedings.mlr.press/v28/agrawal13.html), [RGPE transfer Bayesian optimization](https://arxiv.org/abs/1802.02219), and [Transfer-Tuning](https://arxiv.org/abs/2201.05587). It is not a claim to invent transfer autotuning.

MIT license; see [LICENSE](LICENSE).
