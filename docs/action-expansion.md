# H100 action-set comparison

The H100 comparison measures how ten additional Triton configurations change the gap to `torch.matmul`. The original 36 configurations remain unchanged.

## Fixed action set

I keep the original 36 configurations unchanged. I add four ordinary tensor-core tiles `(BM, BN, BK, warps, stages)`: `(128,128,64,4,5)`, `(128,256,64,8,4)`, `(256,128,64,8,4)` and `(128,128,128,8,3)`. I also use the existing persistent TMA implementation with configurations `(128,128,64,4,4)` and `(128,256,64,8,3)`, both without warp specialization or a subtiled epilogue. These six candidates apply to every workload.

For M=1,7,31, I add four tensor-core split-K configurations with BM=16, BK=64, four warps, three stages, BN in {64,128}, and split factor in {4,16}. Each uses an FP32 partial workspace and a separate Triton FP32 reduction that converts once to FP16. Allocating both buffers and executing both kernels are part of the timed call; I do not time an isolated partial kernel.

The CPU manifest in `src/heliostune/action_configs.py` defines the ten new actions. `src/heliostune/action_expansion.py` launches them without fallback. I use Torch 2.8.0 and Triton 3.4.0, whose persistent tensor-descriptor implementation is already present in this project and based on [Triton's pinned persistent-matmul tutorial](https://github.com/triton-lang/triton/blob/v3.4.0/python/tutorials/09-persistent-matmul.py).

## Selection and measurement

The suite has 96 named workloads and 84 distinct (M,N,K) shapes. Pass 1 measures all 36 original configurations and every applicable new configuration. For each workload, it selects the lowest correct median separately within the old and new sets, breaking ties by configuration key. Pass 2 measures those two fixed winners and `torch.matmul`. The combined set chooses between old and new using pass-1 medians. In the JSON files, these passes are called banks 1 and 2.

Both banks use uniform FP16 operands. The workload order is shuffled with the bank number; tensor seed is bank*10000 plus the index in that shuffled order, matching the collector's original default seed convention. Candidate order uses bank*10000+index+70000 so the larger cross-product has a recorded deterministic shuffle. The reference is `torch.mm` with FP32 output and TF32 disabled only for that reference call, then restored. Every candidate must satisfy the unchanged elementwise atol=rtol=0.01 comparison and finite-output check before timing. Failed compilation, correctness and timing attempts remain in the raw data; they are not replaced by another configuration after bank 2.

Timing follows the original protocol: allocating CUDA FP16 calls, Triton `do_bench`, 25 ms warmup, 100 ms repetition interval, and p20/median/p80. I measure `torch.matmul` in both banks in the same session and record the precision flags, GPU identity and software versions. I do not compare current times directly with earlier measurements from another session.

## Primary analysis

For each workload, the ratio is the pass-2 median of the selected new action divided by the same pass's torch median. Below one is a Triton win. The summary includes old, new and combined-set win counts, per-workload ratios and selected configurations, geometric means, M-group summaries and failures.

The pilot measures four large-K workloads at M=1,7,31,1024 and is saved separately. It estimates compilation and timing cost without changing the action set, timing windows or numerical tolerance. The full run repeats every workload and may reuse compiled binaries.

Raw measurements are in [`results/action-expansion-raw.json`](../results/action-expansion-raw.json). [`scripts/analyze_action_expansion.py`](../scripts/analyze_action_expansion.py) builds the summary and per-workload figure. Each collection records the source commit and the latest commit to this method document. Pilot timings are excluded from the reported result.
