# HeliosTune

An actively maintained retrieval-first GPU autotuning and evidence-control/replay research program with reproducible negative, failed, non-confirmatory, and stopped results.

> [!IMPORTANT]
> **Maintained project.** The transferred-posterior superiority hypothesis is
> concluded and unsupported: the frozen primary H100 comparison was negative,
> selected transfer strengths were zero, the predeclared H200 campaign stopped
> before any timing row, its separate operator-authorized engineering run was
> non-confirmatory, and the engineering expansion gates stopped below their
> thresholds. These outcomes remain negative, stopped, or non-confirmatory as
> originally reported.
>
> HeliosTune continues as a retrieval-first GPU autotuning and
> evidence-control/replay project. `v0.5.0` is the current published immutable
> release; `v0.6.0` is the active development milestone. Contributions may add
> work at new versioned paths, but must never rewrite frozen evidence or
> strengthen its claims. In v0.6, issue #31 implements additive transitive
> plugin → suite custody and canonical attempt chaining for new local/native
> bundles. CPU-only issue #32 (durable canonical VerificationRecords) and issue
> #33 (deterministic network-disabled analyzer replay) remain the next gates,
> followed by dependency separation and a no-cost one-domain design gate. Only
> after those gates may a new, separately approved, predeclared paid protocol be
> proposed. The maximum authorized spend remains **$0**; this roadmap authorizes
> no GPU execution and never rewrites frozen evidence.

**Published evidence:** [Parhelion v3 H200 engineering report](site/parhelion-v3-engineering.html) · [Parhelion L4+A10+T4 → H100 report](https://mottopanikeiku.github.io/heliostune/) · [post-hoc v2 causal addendum](https://mottopanikeiku.github.io/heliostune/parhelion-v2-addendum.html) · [v3 pilot failure evidence](benchmarks/parhelion-v3-validation-failure.json) · [artifact catalog](benchmarks/research-artifact-manifest.json) · [v1 L4 ↔ A10 report](https://mottopanikeiku.github.io/heliostune/v1.html) · [post-run chain of custody](benchmarks/parhelion-v2-post-run-manifest.json)

**Engineering evidence:** Hopper H100 [report](site/hopper-h100-engineering.html), hardened [v2 summary](benchmarks/results/hopper-h100-engineering-summary-v2.json), and [v2 publication manifest](benchmarks/hopper-h100-engineering-manifest-v2.json) · H100 precision [report](site/h100-precision-probe.html), [summary](benchmarks/results/h100-precision-probe-summary.json), and [publication manifest](benchmarks/h100-precision-probe-manifest.json)

**Native RMSNorm H100 stage gate:** the retained [report](site/native-rmsnorm-h100.html), strict [summary](benchmarks/results/native-rmsnorm-h100-summary.json), compressed [raw evidence](benchmarks/data/native-rmsnorm-h100.json.zst), and [publication manifest](benchmarks/native-rmsnorm-h100-manifest.json) preserve the exact exploratory outcome and both remote attempts without upgrading them to methodology-v1 authenticity or claim eligibility.

**Fusion remote exploratory receipts:** the deterministic [report](site/fusion-remote-exploratory.html), strict [summary](benchmarks/results/fusion-remote-exploratory-summary.json), compressed [raw evidence](benchmarks/data/fusion-remote-exploratory.json.zst), and [manifest](benchmarks/fusion-remote-exploratory-manifest.json) preserve four client-authorized H100 attempts: two gated-MLP calls unresolved after 401 errors and cancellation requests, plus one completed gated-MLP and one completed residual-RMSNorm call. App IDs are operator-recorded with no artifact binding; FunctionCall IDs are bound by retained remote journals. Completed correctness, compile, and timing values are measured facts only; `candidate / reference` is candidate median divided by reference median, values below 1 indicate the lower returned candidate median, and the reciprocal direction is also reported. The evidence makes no fusion or superiority claim, every completed receipt records `publication_eligible=false`, and provider physical starts or restarts, provider attempt count, total GPU time, and actual cost are unknown; no attestation is present. The attempts are bound to different historical HEAD commits and wheel digests and must not be treated as one interchangeable build.

**Methodology:** [HeliosTune methodology v1](METHODOLOGY.md) is the active, partially implemented evidence-control design target; it is non-retroactive and does not claim that legacy evidence conforms. [Experiment scope](EXPERIMENT_SCOPE.md) records the implemented declaration and additive bundle-custody state plus the bounded continuation roadmap while separating vocabulary, schema/template identity, structural source availability, backend capability, correctness observations, and performance observations.

**v0.6 custody status:** `heliostune verify-bundle` can check an additive,
offline-only inventory of all suites referenced by the bound plugin, an exact
selected-suite descriptor, and a canonical predecessor-linked attempt journal.
The reserved flat roles are `plugin_suite_<index>`, `selected_suite`, and
`attempt_chain`; partial opt-in fails closed. Older valid v1 bundles without the
descriptors remain parseable with custody/chain `not_checked`, never promoted.
New local/native producers require custody, chain, and no-retry reconciliation
to be `checked` before atomic no-replace publication. These internal digest and
filesystem-containment checks are not signatures, provider authentication,
retry/billing reconciliation, analyzer replay, complete offline reproduction,
or claim eligibility.

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

The separate one-bank H100 engineering expansion screen is also negative. Its ratios are `torch_ms / same-bank best_candidate_ms`, so a ratio of at least **1.05** means the selected candidate is at least 5% faster and a ratio below 1 means it is slower. Skinny GEMV stopped at geomean **0.395875** with **0/32** workloads meeting the 1.05 threshold, and Hopper GEMM stopped at **0.773365** with **0/64**. Candidate selection and scoring reused bank 0, so these same-bank ratios are optimistic; the result is one-instance engineering-gate evidence (preserved as a legacy artifact, not retroactively upgraded to `heliostune.protocol/1`). The global decision was **STOP**, no three-bank collection followed, and no superiority claim is made. The operator-recorded paid app was [Modal `ap-ryV3BXdW1g2TGp5LIg6MDH`](https://modal.com/apps/mottopanikeiku/main/ap-ryV3BXdW1g2TGp5LIg6MDH), with artifact-bound FunctionCall `fc-01M0XX8KZ2WZQWPA4V2SYVGNWX`; its hardened methodology-compatible derivation is published as the [v2 summary](benchmarks/results/hopper-h100-engineering-summary-v2.json), [v2 manifest](benchmarks/hopper-h100-engineering-manifest-v2.json), and [report](site/hopper-h100-engineering.html), with the [compressed raw artifact](benchmarks/data/hopper-h100-engineering.json.zst) retained as cataloged evidence. The original [v1 summary](benchmarks/results/hopper-h100-engineering-summary.json) and [v1 manifest](benchmarks/hopper-h100-engineering-manifest.json) are retained immutably.

## Parhelion v3 campaign outcome

The predeclared H200 campaign, as closed and released in v0.4.0, terminated at its single L4 pilot FunctionCall before any timing row was returned. The remote container failed while importing `modal_bench.py` because it searched a container-relative local wheel directory. Under that campaign's pre-H200 failure rule, the FunctionCall was not retried and no downstream artifact belonged to the frozen study. This remains the complete original campaign outcome, not a hardware-performance result.

The [validation-failure manifest](benchmarks/parhelion-v3-validation-failure.json) binds the frozen development protocol, failed commit and wheel, Modal app and FunctionCall IDs, two-record append-only journal, observed import error, zero-spawn recovery, and every absent downstream artifact. Release 0.4.0 published the software, existing causal addendum, protocol, and failure evidence. Its remote wheel-discovery correction did not change the failed protocol bytes or authorize a retry; the later operator-authorized engineering work remained a separate campaign.

### Operator-authorized H200 engineering benchmark

After v0.4.0, the operator explicitly overrode the no-retry rule and authorized a separate engineering run. The fixed pilot, three-GPU candidate collection, A100 validation, and five-bank H200 collection completed with bound call journals. This run is labeled `operator_authorized_engineering_protocol_deviation`; it does not reopen the original campaign or carry a confirmatory superiority claim.

On H200, Parhelion reached **0.94772 AUC** over budgets 1–8 and **99.89%** of the retained-config reference at budget eight. Anchored cold Thompson reached **0.94497 AUC** and **99.84%** at budget eight. The paired 50-seed Parhelion-minus-anchored-cold AUC difference was **+0.00275**, with a conditional two-sided 95% Student-t interval **[+0.00062, +0.00487]**. Banks 3 and 4 produced similarly positive engineering contrasts, but the protocol deviation precludes a confirmatory claim.

| H200 engineering method | AUC, budgets 1–8 | Fraction at budget 8 | Median queries to 95% |
|---|---:|---:|---:|
| Random | 74.26% | 83.95% | — |
| Single-source nearest | 74.76% | 81.94% | — |
| Cold Thompson | 88.29% | 98.15% | 6 |
| Pooled-source Thompson | 88.29% | 98.15% | 6 |
| Parhelion without forced anchor | 89.04% | 99.05% | 6 |
| Multi-source retrieval | 94.40% | 99.84% | 4 |
| Anchored cold Thompson | 94.50% | 99.84% | 4 |
| **Parhelion** | **94.77%** | **99.89%** | **4** |

The selected transfer strength was zero, and Parhelion exactly matched its no-transfer ablation. The observed gain therefore comes from the retrieval representation and forced first query, not a transferred source posterior. A100 bank 0 landed on PCIe while banks 1–4 landed on SXM; the raw bytes are preserved and the derived archive explicitly canonicalizes only `device_name`. That mixed-subvariant selection domain is a material validity limit. See the [engineering result](benchmarks/results/parhelion-v3-h200-engineering.json) and [self-contained report](site/parhelion-v3-engineering.html).

The H100 values in the first table are fractions of a bank-1-selected, bank-2-scored best configuration from the curated 36-action Triton manifest. They are not fractions of a hardware ceiling. `torch.matmul` can exceed 1.0 because it is outside that manifest and is evaluation-only.

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

The hashed [post-run manifest](benchmarks/parhelion-v2-post-run-manifest.json) binds the exact historical runs, commits, commands, compressed and uncompressed data, selection, summary, and report. The [pre-H100 freeze](benchmarks/parhelion-v2-h100-freeze.json) records the no-pilot/no-rerun rule, hardware identity gate, selected parameters, source order, seeds, budgets, collector settings, failure rule, and implementation/data digests. The [research artifact catalog](benchmarks/research-artifact-manifest.json) verifies every historical and published alias digest; the separate [addendum manifest](benchmarks/parhelion-v2-addendum-manifest.json) binds the immutable input, implementation, exploratory result, and new report without touching historical bytes. For the engineering publications, each linked publication manifest binds raw data, the attempt journal, and its summary; the canonical research catalog separately binds each generated report digest.

The v3 [development protocol](benchmarks/parhelion-v3-development-protocol.json), [terminal failure manifest](benchmarks/parhelion-v3-validation-failure.json), and [failed attempt journal](benchmarks/data/parhelion-v3-pilot-failure.attempts.jsonl) preserve the original stopped campaign. The separate engineering chain binds the [successful pilot](benchmarks/data/parhelion-v3-pilot-operator-retry.jsonl.zst), [mixed-A100 validation archive](benchmarks/data/parhelion-v3-validation.jsonl.zst), [H200 freeze](benchmarks/parhelion-v3-h200-freeze.json), [final archive](benchmarks/data/parhelion-v3-final.jsonl.zst), and [engineering result](benchmarks/results/parhelion-v3-h200-engineering.json) without rewriting the v0.4.0 failure evidence.

The additive [H200 derivation manifest](benchmarks/parhelion-v3-h200-engineering-derivation-manifest.json) binds the final post-run reproducer, every consumed input, and the committed result/report. Its check recomputes every scientific/result field and the report while reusing only the committed environment-specific `runtime` provenance; it does not claim environment-independent full-result regeneration or predeclared code custody.

### Hardened releases and preservation

`v0.5.0` is the current published release. Its no-input manual action was
dispatched from protected `main`, gated by the `release` environment, and
checked out, built, and smoke-checked the dispatch event's exact `GITHUB_SHA`
before tagging that tested snapshot. Later `main` development does not change
the source selected for that immutable release.

For every version, the release publishes a wheel, sdist, `SHA256SUMS`, and a
workflow-verified Git bundle containing the semantic tag history; every release
asset is attested. Preserve the Git bundle together with `SHA256SUMS`: the
checksum file verifies the downloaded assets, while the bundle preserves the
released source and tag history independently of mutable GitHub references.
Published releases are immutable; v0.6 work uses new code and paths.

## Run locally

The current release, `v0.5.0`, records Python 3.11–3.13 and uv 0.12.5. To
inspect or replay its published snapshot from a clean clone:

```bash
git clone https://github.com/mottopanikeiku/heliostune.git
cd heliostune
git checkout --detach v0.5.0
uv sync --locked --extra dev
uv run heliostune --help
uv run heliostune --version
uv run heliostune demo --output-dir /tmp/heliostune-demo --max-budget 2 --seeds 2
uv run heliostune inspect /tmp/heliostune-demo/measurements.jsonl
```

After returning to an active v0.6 source checkout, inspect its strict CPU-only
structural surface:

```bash
uv run heliostune verify-plugin path/to/plugin.json
uv run heliostune verify-suite path/to/suite.json
uv run heliostune verify-bundle path/to/bundle.json
uv run heliostune list-scope
```

These v0.6 commands do not execute a backend or establish correctness,
performance, authenticity, provider retry/billing truth, analyzer replay,
complete offline reproduction, or claim eligibility. `verify-bundle` reports
custody, attempt-chain, and reconciliation limitation explicitly; it does not
yet emit the VerificationRecord planned by issue #32.
The generic local and Modal branches for the two frozen
reference templates, and their existing
[exploratory receipts](benchmarks/results/fusion-remote-exploratory-summary.json),
are unchanged. Both legacy suites completed correctness and timing, but no
fusion, superiority, provider attempt-count, cost, attestation, or
publication-eligibility claim is made.

The exact
[`residual_rmsnorm_triton.v1` suite](benchmarks/suites/residual-rmsnorm-triton-v1.json)
and
[`fusion-triton-rmsnorm-plugin` plugin](benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json)
have a digest-dispatched native executor and a retained H100 SM90
observation. The observation does not change the declarations: all six arms'
local and remote capability states remain `unprobed` with null evidence digests,
and nothing here says that the current host can execute the suite. On an exact
H100 SM90 environment with the pinned GPU dependencies, the local invocation is:

```bash
uv run --extra gpu heliostune run-local-suite \
  benchmarks/suites/residual-rmsnorm-triton-v1.json \
  --plugin benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json \
  --output /tmp/heliostune-native-rmsnorm
```

The local result schema is `heliostune.local_executor/2`; its native protocol
binds `heliostune.native_fusion_executor/2` and is published as a strict
`heliostune.bundle/1` exploratory bundle. Each of the four
`block_size=4096`, `num_warps=4|8|16|32`, `num_stages=1` candidates must pass,
in order: compile plus complete zero-spill resource evidence; canonical
correctness plus deterministic `zeros`, `cancellation`, and structured
`overflow` probes; exactly one matching CUDA profiler event for one invocation;
and the frozen 10-warmup/50-repetition timing policy. A failed gate is retained,
blocks that arm's later gates, and is never replaced by eager fallback or retry.
The deterministic stage-gate analyzer compares the fastest fully eligible
candidate with the faster passing eager/Inductor baseline and authorizes only
exploratory expansion at a speedup of at least `1.10x`; it emits no claim.

In the retained H100 observation, all four native candidates passed the
correctness, complete zero-spill resource, and profile gates and were eligible;
each profiled one-invocation check observed exactly one matching CUDA kernel
event. This observation still has `fusion_claim=false`. The fastest native arm,
`rmsnorm-triton-w8`, had median **0.0505920015 ms**, versus **0.085072 ms** for
eager and **0.045952 ms** for Inductor. The fair
`best_baseline_median / candidate_median` ratio was **0.908286**, below the
predeclared **1.10** threshold, so the retained decision is
**`STOP_BELOW_THRESHOLD`**: no expansion, correctness claim, performance claim,
fusion claim, or publication-eligibility claim follows.

The retained guarded paid Modal path documents executor API
`heliostune.modal_fusion_executor/2`:

```bash
uv run python scripts/build_modal_wheel.py
uv run --extra modal modal run modal_fusion_executor.py::main \
  --suite benchmarks/suites/residual-rmsnorm-triton-v1.json \
  --plugin benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json \
  --output "artifacts/fusion-remote/residual-rmsnorm-triton-v1-$(date -u +%Y%m%dT%H%M%S%N)"
```

Do not make that paid call without separate explicit approval of a predeclared
protocol and bound. An approved invocation must use a clean committed `HEAD`,
the freshly built and verified wheel plus adjacent manifest, and a fresh
versioned output path. The client permits one spawn and no
automatic retry, but Modal's physical starts and restarts are unobservable, so
provider physical attempt count, total GPU time and its upper bound, total time
upper bound, and actual cost remain unknown. A remote run publishes
`heliostune.remote-receipt/1`, not a methodology bundle. Local and remote
artifacts bind the exact plugin/suite bytes, package-wide source identity, the
four critical executor-source path/size/digests, and, remotely, the verified
wheel, manifest, commit, request, journal, and returned result. Failures and
blocked cells remain evidence; lost acknowledgement, timeout, interruption, or
unproven cancellation remains `unresolved`.

The publication retains the first attempt as unresolved after its result
exceeded the 6144-byte inline transport limit; that receipt establishes no
execution result. A compact-result attempt then completed. The static
publication preserves both receipts and the completed observations, but it is
not a methodology bundle and `publication_eligible=false`.

Native gated MLP remains absent after an unfavorable feasibility audit.
Attention, KV-cache, MoE, quantized-linear, and FP8 are retained schema
vocabulary and staged design inventory rather than implemented backends.

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

Verify the catalog and deterministic post-run derivations:

```bash
uv run heliostune verify-catalog benchmarks/research-artifact-manifest.json
uv run python scripts/build_parhelion_v2_addendum.py --check
uv run --python 3.11 python scripts/build_parhelion_v3_engineering_result.py --check
```

From the clean clone checked out at `v0.5.0` above, reproduce the frozen H100 replay directly from the compressed archive:

```bash
uv run heliostune compare-multisource \
  benchmarks/data/parhelion-v2-measurements.jsonl.zst \
  --sources L4,A10,T4 --target H100 --max-budget 8 --seeds 30 \
  --k 16 --temperature 2.0 --transfer-strength 0.0 \
  --retrieval-k 8 --retrieval-temperature 0.2 \
  --pooled-transfer-strength 0.0 --primary-comparator torch \
  --protocol-role final \
  --release-provenance benchmarks/parhelion-h100-release-provenance.json \
  --output artifacts/replay/h100-final-summary.json
cmp artifacts/replay/h100-final-summary.json benchmarks/results/parhelion-h100-final.json
uv run heliostune report artifacts/replay/h100-final-summary.json \
  --output artifacts/replay/h100-report.html
```

The replay writes under `artifacts/replay/` because the research artifact catalog binds `artifacts/h100-final-summary.json` and `artifacts/h100-report.html` as historical freeze-only aliases that must stay absent, so writing to those exact paths would make `heliostune verify-catalog` fail.

The local `heliostune demo` is synthetic and supports no hardware claim.

## Retained Modal collection procedure

The commands below document the preserved collection procedure; they are not
authorization for a paid call. Any new campaign in HeliosTune requires a new
versioned protocol and artifact paths, separate explicit approval, and must not
rewrite frozen evidence:

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

`modal_bench.py` gates `L4`, `A10`, `T4`, `H100`, `A100-80GB`, and `H200` identities before tensor allocation. H100 uses Modal's exact `H100!` selector. Do not rerun the published H100 protocol; any independent study requires a new frozen protocol, new versioned paths, and separate approval.

## Repository map

- `src/heliostune/artifacts.py` — strict JSON/JSONL decoding and atomic zstandard persistence
- `src/heliostune/methodology.py` — strict methodology roots, descriptor-contained bundle verification, transitive plugin/suite custody, and canonical attempt chaining
- `src/heliostune/scope.py` — strict plugin/suite v1 declarations and shared transitive inventory verification
- `src/heliostune/collection.py` — paid-call planning, fsynced attempt journals, resume, and commit
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
- `scripts/build_parhelion_v3_engineering_result.py` — H200 scientific-result replay with explicit frozen runtime provenance
- `scripts/verify_research_artifacts.py` — full catalog, alias, count, and frozen-point verifier
- `scripts/assemble_parhelion_final.py` — historical v2 archive verifier
- `benchmarks/` — frozen protocols, chain manifests, compressed matrices, selections, and results
- `benchmarks/plugins/` and `benchmarks/suites/` — frozen reference declarations and suite hashes
- `EXPERIMENT_SCOPE.md` — declaration state matrix, numeric/fusion scope, baselines, and promotion rules
- `site/` — offline published report, downloadable JSON, and retained v1 report

## Scope

The published Parhelion/Hopper results are a steady-state FP16 microkernel
configuration-selection study over one fixed 96-workload corpus, one curated
36-arm space, and four Modal GPU fleets. They do not establish generalization
to arbitrary GPUs or model families, global Triton optimality, Bayesian
calibration, compilation-time savings, end-to-end serving gains, or production
interference robustness. T4 is a validation domain, not independent final
evidence. H100 is one untouched hardware domain, not proof of universal
cross-architecture transfer.

The `heliostune.plugin/1` and `heliostune.suite/1` declarations broaden the
*representable* scope without broadening those historical claims. Only
`gated_mlp_epilogue.v1` and `residual_rmsnorm.v1` are frozen initial fusion
templates, constrained to FP16/BF16 input/storage, FP32 accumulation,
FP16/BF16/FP32 output, null quantization, and disabled TF32. Attention and KV
cache, quantized linear, MoE, and FP8 remain catalog/design inventory governed
by the ordered VerificationRecord, analyzer-replay, dependency-split, and
one-domain no-cost design gates above. Inventory is not capability,
implementation, correctness, or performance evidence, and the roadmap
authorizes no paid campaign. See
[Experiment scope](EXPERIMENT_SCOPE.md) for the active state matrix, exact
contracts, hashes, continuation gates, and evidence-preservation requirements.

## License

MIT
