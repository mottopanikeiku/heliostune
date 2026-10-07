# Next comparison: widen the actions before testing transfer again

## What the existing measurements settle

I audited the committed H100 timing matrix. Torch's stored median is lower for all 96 workloads, even than the fastest curated configuration chosen on bank 2 itself. The independently selected reference-to-torch ratios range from 1.001563× to 2.329776×. [Audit numbers](../results/action-set-audit.json), [ratios by M](../results/action-set-by-m.svg), [individual shapes](../results/action-set-shapes.svg).

I also report the earlier L4/A10 and T4 validation collections separately in the audit's `by_gpu` overview. L4/A10 have mixed torch/Triton wins; the H100 action-set conclusion is not a universal claim about every GPU. The same nominal timing settings do not make different acquisition stages a pooled policy experiment.

The bank-2 minimum is an optimistic in-sample diagnostic. It is not a recommendation selected independently of evaluation. It does show that no search policy can discover a faster-than-torch action within this fixed measured matrix. The near-tie at `(M,N,K)=(1,18944,3584)` saves only 0.096 µs; missing torch quantiles and paired raw samples prevent a significance claim.

The gap spans every measured M, not just decode. Larger relative gaps and larger absolute savings are different: the M=1024 group has the lowest geometric-mean ratio, 1.404365×, but the largest median saving, 40.856 µs. Repeated shapes across model families are not extra devices.

The [original H100 comparison](../benchmarks/results/parhelion-h100-final.json) selected zero posterior-transfer strengths. I cannot attribute the measured library gap to bandwidth, occupancy, launch overhead, or a particular instruction without profiling and a matched intervention. Tile geometry alone is not a causal explanation.

I also retain the separate skinny-GEMV and Hopper-GEMM engineering expansion. Both stopped below their performance thresholds ([result](../benchmarks/results/hopper-h100-engineering-summary-v2.json)). A new candidate needs to offer something those failed candidates did not; a wider parameter grid is not evidence of a faster kernel.

## A separate comparison

I would keep the existing study unchanged and compare:

1. The old configuration subset, broader candidate implementations, and `torch.matmul` as both baseline and possible fallback.
2. Identical input tensors, layouts, output dtype, explicit accumulation settings, correctness tolerances and timing boundaries. The [historical collector](https://github.com/mottopanikeiku/heliostune/blob/fe5beda065f6afb5b2c9ddd9a58e1d2b573b6abd/src/heliostune/kernel.py) used a separate FP32-input correctness reference with TF32 disabled, converted to FP16; the timed comparator used FP16 inputs. I would not assume equal accumulation from passing that tolerance.
3. Development-only candidate and policy selection, preserving family and exact-shape exclusions. Independent banks for observations, reference selection and scoring; raw paired timing blocks and device state for uncertainty estimates.
4. Absolute latency, independently scored candidate-to-torch ratios, compilation and tuning cost, invalid configurations, and workload coverage. Cold Thompson, nearest-shape reuse, retrieval-only and Parhelion would run on both action sets.

Success would mean lower held-out latency than torch for a stated deployment subset, after tuning and compilation are amortized. A score near the limited Triton reference is not success against the library. I would choose the practical margin and workload weighting before collecting timings.

This audit needs only CPU analysis of existing data. The separate comparison needs compatible NVIDIA hardware and an explicit deployment/numerical contract. Its collection time and cost are unknown; the old measurements cannot establish the performance of new actions. I have not collected new GPU timings or dispatched a cloud job.
