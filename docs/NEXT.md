# Next comparison: test policies on the expanded actions

## What I measured

I collected a separate H100 comparison after committing the [action-expansion plan](action-expansion-plan.md). Ten new configurations cover larger tiles, deeper pipelines, persistent TMA and split-K; I did not change the original 36 configurations. All 96 named workloads have complete selection and scoring banks, including same-session torch. The [raw data](../results/action-expansion-raw.json), [summary](../results/action-expansion-summary.json) and [figure](../results/action-expansion.svg) report the result.

The new bank's fixed bank-1 winners beat torch on 9 workloads, tie once and lose on 86. The old bank loses on all 96. New/torch geometric-mean latency is 1.174062, versus old/torch 1.608227; new/old is 0.730035. New beats old on 95 workloads. This is a substantial reduction in the measured action-set gap, not a general victory over the library.

Six nominal torch wins use split-K and three use larger ordinary tiles. The largest latency reduction is 5.55% at `(M,N,K)=(1,3584,18944)`. Five of the nine margins are below 1%; no result establishes statistical significance. Persistent TMA is selected for 16 workloads, but none beats torch. No selected new action wins at M=31,257,1024. These facts do not establish which hardware resource caused a gap.

## What remains unchanged

The [historical audit](../results/action-set-audit.json) still reports torch below every stored H100 Triton candidate, including the optimistic same-bank minimum. L4/A10 have mixed wins, so neither the historical H100 gap nor this expansion is a universal GPU claim. I do not pool acquisition stages or device types.

The [original transfer comparison](../results/tuner-comparison.md) favored cold Thompson over Parhelion and selected zero posterior-transfer strengths. I have not rerun those policies on the expanded bank. A better kernel bank cannot retroactively establish a transfer gain.

The new correctness reference has FP16 inputs and FP32 output with TF32 disabled. The [historical collector](https://github.com/mottopanikeiku/heliostune/blob/fe5beda065f6afb5b2c9ddd9a58e1d2b573b6abd/src/heliostune/kernel.py) instead used separate FP32-input reference tensors, converted to FP16. The tolerance and nominal timing windows are unchanged, but that does not make the two collections the same numerical experiment. I compare old and new actions only within the new collection.

## The next question

I would compare cold Thompson, nearest-shape reuse, retrieval-only and Parhelion on both the old bank and the expanded union, including torch as a possible deployment fallback. I would retain development-only policy selection, exact-shape and family exclusions, and independent scoring. Bank-1 latency chooses the old/new union winner; bank-2 latency does not select a policy or candidate.

Before collecting more data, I would choose a deployment workload weighting and practical performance margin. The next acquisition should retain paired timing blocks and device state, and repeat on independent sessions or devices. Quantiles in this run describe timing spread, not uncertainty in a population mean.

Success would mean lower held-out deployment latency after amortizing compilation and tuning, under a stated numerical contract. Warmed allocating matmul medians include split-K workspace and reduction but not startup, compilation or serving costs. The [cloud estimate](../results/action-expansion-cost.json) includes the failed startup and pilot separately; it is not a provider invoice. The full collection completed in one session without checkpoint recovery.
