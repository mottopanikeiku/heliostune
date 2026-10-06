# Next experiment: fix the action set before testing transfer again

## Why

The committed [H100 result](../benchmarks/results/parhelion-h100-final.json) gives `torch.matmul` an average score of 1.6103 against the curated Triton reference, versus 0.9503 for Parhelion and 0.9584 for cold Thompson. A better search policy over the same configurations cannot close that gap if the useful kernels are absent. The selected source-posterior strengths were zero; another transfer grid is not the first priority.

This repository already tried a separate skinny-GEMV and Hopper-GEMM engineering expansion. Both stopped below their performance thresholds ([result](../benchmarks/results/hopper-h100-engineering-summary-v2.json), `global_decision` and `regimes`). Do not repeat that screen and call it a new experiment. First inspect its failed candidates and timing/precision contracts; a wider set needs a concrete reason to offer capabilities those candidates lacked.

The new [comparison table](../results/tuner-comparison.md) makes this gap visible without collecting more data. Its scores are relative to the existing bank-1-selected, bank-2-scored reference. It is a summary of the original analyses, not a new measurement or policy replay.

## The experiment

Keep the existing H100 study unchanged. Run a separate action-space comparison only after an owner-approved hardware and spending decision:

1. Preserve the old configurations as a labeled subset. Add a broader tile/warp/stage search and suitable candidate implementations for the skinny and large GEMM shapes. Inspect the existing kernel constraints before declaring which candidates are valid.
2. Treat `torch.matmul` as a first-class baseline and possible deployment fallback. Use the same input tensors, dtype, accumulation contract, tolerance, synchronization and timing regime for every arm. Do not mix a numerically weaker arm with a stronger reference without stating and testing the difference.
3. Select new actions and policy settings on development hardware/workloads only. Keep source/target family and exact-shape exclusions. Collect independent timing banks for policy observations, reference selection and scoring; never select and score the best candidate on the same bank.
4. Compare cold Thompson, nearest-shape reuse, retrieval-only and Parhelion on both the old subset and the expanded set. Report absolute latency, candidate/reference ratios, compile cost, invalid configurations, and the fraction of workloads where the best independently scored candidate beats `torch.matmul`.
5. Stop searching for transfer gains if the expanded set still has no practical advantage over `torch.matmul`. If it does, test whether retrieval reduces probes to reach that advantage; avoid hiding an action-set improvement inside a policy comparison.

A useful success criterion is lower held-out latency than `torch.matmul` for a stated workload subset after accounting for tuning and compilation, not “near 100%” of a limited candidate reference. Choose the practical margin before timing begins.

## Cost and time

**Work done now:** standard-library extraction of committed results, on local CPU, $0. No new GPU timing or paid service was used.

**Next no-cost work:** audit candidate constraints and draft the separate comparison inputs, approximately one working day of engineering (estimate, not measured). Dry-run parsing and candidate enumeration on CPU; this cannot establish correctness or speed of GPU kernels.

**Measurement work:** requires an accessible NVIDIA GPU compatible with the chosen implementations. If the owner provides existing local access, incremental paid-service spend can remain $0; no such access is assumed here. Time and any rental cost remain unknown until a small owner-approved compile/timing pilot measures them. Do not dispatch cloud jobs to obtain that estimate without approval. The old measurements cannot answer how newly added actions perform.

## Owner decisions

- Is there already available, unpaid GPU access for a separate comparison?
- Which numerical contract and real workload subset matter for deployment?
- Is a `torch.matmul` fallback an acceptable outcome, including when compilation/tuning is not amortized?

The existing data, published reports, negative H100 conclusion and separately labeled H200 engineering run remain unchanged.
