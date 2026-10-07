# H100 action-set comparison

I will test whether a small Hopper-aware expansion closes the gap between the original 36 Triton configurations and `torch.matmul`. The earlier saved-data audit found almost no gap between the selected old configuration and the best old configuration, but that did not show whether a different kernel family would help.

## Fixed action set

I keep the original 36 configurations unchanged. I add four ordinary tensor-core tiles `(BM, BN, BK, warps, stages)`: `(128,128,64,4,5)`, `(128,256,64,8,4)`, `(256,128,64,8,4)` and `(128,128,128,8,3)`. I also use the existing persistent TMA implementation with configurations `(128,128,64,4,4)` and `(128,256,64,8,3)`, both without warp specialization or a subtiled epilogue. These six candidates apply to every workload.

For M=1,7,31, I add four tensor-core split-K configurations with BM=16, BK=64, four warps, three stages, BN in {64,128}, and split factor in {4,16}. Each uses an FP32 partial workspace and a separate Triton FP32 reduction that converts once to FP16. Allocating both buffers and executing both kernels are part of the timed call; I do not time an isolated partial kernel.

The CPU manifest in `src/heliostune/action_configs.py` defines the ten new actions. `src/heliostune/action_expansion.py` launches them without fallback. I use Torch 2.8.0 and Triton 3.4.0, whose persistent tensor-descriptor implementation is already present in this project and based on [Triton's pinned persistent-matmul tutorial](https://github.com/triton-lang/triton/blob/v3.4.0/python/tutorials/09-persistent-matmul.py).

## Selection and measurement

I use all 96 original named workloads, representing 84 distinct (M,N,K) shapes, on one H100. For bank 1, I measure all 36 old configurations and all applicable new configurations. I select the lowest correct median separately within the old and new sets, breaking ties by configuration key. Bank 2 measures only those two fixed winners and `torch.matmul`; no bank-2 outcome changes the selected action. The expanded union additionally chooses old versus new using bank-1 medians, not bank 2.

Both banks use uniform FP16 operands. The workload order is shuffled with the bank number; tensor seed is bank*10000 plus the index in that shuffled order, matching the collector's original default seed convention. Candidate order uses bank*10000+index+70000 so the larger cross-product has a recorded deterministic shuffle. The reference is `torch.mm` with FP32 output and TF32 disabled only for that reference call, then restored. Every candidate must satisfy the unchanged elementwise atol=rtol=0.01 comparison and finite-output check before timing. Failed compilation, correctness and timing attempts remain in the raw data; they are not replaced by another configuration after bank 2.

Timing follows the original protocol: allocating CUDA FP16 calls, Triton `do_bench`, 25 ms warmup, 100 ms repetition interval, and p20/median/p80. I measure `torch.matmul` in both banks in the same session and record the precision flags, GPU identity and software versions. I do not compare current times directly with earlier measurements from another session.

## Primary analysis

For each workload, the primary ratio is the bank-2 median of the bank-1-selected new action divided by its bank-2 torch median; below one is a Triton win. I report old/new/expanded-union win counts, all per-workload ratios and selected configurations, geometric means over comparable scored workloads, M-group summaries and unscored failures. I distinguish 96 named workloads from 84 unique shapes and make no statistical-significance or hardware-cause claim from a single device and descriptive quantiles.

The pilot measures four large-K workloads at M=1,7,31,1024 and is saved separately. It sizes compilation and timing cost but does not change the fixed action set, timing windows, numerical tolerance or estimator. The full run repeats all workloads independently and may reuse compiled binaries. I book one H100 container at a time, one CPU core and 4 GiB host memory, with a four-minute pilot and at most a 28-minute full call inside a $2.50 budget. Actual elapsed cloud minutes and a conservative cost estimate are added to the result metadata after each run. Compilation, tuning and cold-start cost are reported separately from warmed call medians.

Raw measurements are `results/action-expansion-raw.json`; `scripts/analyze_action_expansion.py` writes `results/action-expansion-summary.json` and `results/action-expansion.svg`. The experiment records the commit containing this plan and the fixed implementation. Pilot and full data remain separate; all scientific conclusions use the full run.
