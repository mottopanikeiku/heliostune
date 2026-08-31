# HeliosTune experiment scope

## Status

This document describes the currently implemented `heliostune.plugin/1` and
`heliostune.suite/1` declaration surface and the intended order for expanding
HeliosTune beyond its legacy FP16 GEMM studies. It is narrower than the
normative evidence lifecycle in [METHODOLOGY.md](METHODOLOGY.md): validating a
declaration is not executing a study, validating correctness, measuring
performance, or making a claim.

The new schemas are additive and non-retroactive. The published Parhelion and
Hopper studies remain immutable legacy plugins and evidence. They have not been
migrated to plugin/suite v1, and the new declarations do not upgrade their
eligibility, custody, numerical checks, or conclusions.

## Keep the state axes separate

The following states answer different questions and must not be collapsed into
one "supported" label.

| Axis | What it says | Where it is represented now | What it does not say |
|---|---|---|---|
| Vocabulary | A domain, dtype, shape operator, or case option is a recognized closed enum value. | Strict plugin/suite parser and `heliostune list-scope`. | That any frozen suite uses it or any backend can execute it. |
| Schema | A `heliostune.plugin/1` or `heliostune.suite/1` document has exact JSON types, no unknown or duplicate fields, and satisfies its cross-field rules. | `verify-plugin` and `verify-suite`. | That referenced code imports, a GPU is present, or a case is correct or fast. |
| Template | A suite ID has frozen cases, arms, numeric contracts, fusion semantics, expected cells, and exact artifact bytes. | The two generic reference templates and the separately integrated native Triton template, with their SHA-256 values below. | That a local or remote capability probe passed, or that the template is correct or fast. |
| Backend capability | A particular arm was not probed, or a retained probe found it available or unavailable. Local and remote are separate. | Suite arm `local_capability` and `remote_capability`: `unprobed`, `available`, or `unavailable`. | Correctness, performance, portability to another target, or claim eligibility. |
| Correctness observation | A retained execution observation passed the frozen numerical and semantic checks for one case, arm, input seed, and environment. | A narrow local executor/bundle observation, not a plugin or suite declaration. | That timing passed, that another case passed, or that an arm is faster. |
| Performance observation | Retained timing samples were collected under a frozen timing policy after a passing correctness observation. | A narrow local executor/bundle observation, not a plugin or suite declaration. | A win, generalization, or statistical claim without the rest of the protocol and evidence checks. |

A capability state is exactly `unprobed | available | unavailable`. `unprobed`
requires `evidence_sha256: null`; `available` and `unavailable` require the
lowercase SHA-256 of retained capability evidence. Local and remote capability
records are independent: neither is inferred from the other. An availability
probe is still only an execution precondition, not a numerical or performance
result.

## Declaration roots and custody

A plugin root uses the exact literal `heliostune.plugin/1`. Its closed fields
are `schema`, `plugin_id`, `version`, `template_status`, ordered `domains`,
ordered `arm_ids`, and ordered `suite_refs`. Every suite reference contains a
normalized, non-escaping relative `path`, the SHA-256 of the exact referenced
bytes, and the expected `suite_id` and `revision`.

A suite root uses the exact literal `heliostune.suite/1`. Its closed root fields
are `schema`, `suite_id`, `revision`, `plugin_id`, `plugin_version`,
`template_id`, `template_status`, `domain`, `numeric_contracts`, `tensors`,
`arms`, `cases`, `correctness_policies`, `timing_policies`, ordered
`expected_cells`, and `executor_rule`. Arm records
carry inline shape constraints and separate local/remote capability evidence;
case records carry the closed template-specific semantics. Objects are closed,
JSON scalar types are exact, numbers are finite, and digest strings are
lowercase 64-character SHA-256 values. Vocabulary membership is closed rather
than an extension point. Broader baseline hierarchy resolution remains a
promotion/evidence requirement below; the narrow reference templates do not
claim that a complete performance-baseline manifest has been implemented.

`heliostune verify-plugin PATH` resolves each suite path relative to the plugin
file and verifies the referenced bytes and suite structure. This makes
standalone plugin validation transitive through plugin → suite. It does **not**
extend the current generic `heliostune.bundle/1` verifier: generic
EvidenceBundle custody does not yet resolve a bound plugin's suite references,
and no transitive plugin → suite custody claim is made for bundles.

## Domain and dtype scope

The domain vocabulary currently includes:

- `dense_gemm`
- `fused_mlp`
- `rmsnorm_residual`
- `attention`
- `kv_cache`
- `moe`
- `quantized_linear`

The dtype vocabulary is:

- `fp32`
- `tf32`
- `fp16`
- `bf16`
- `fp8_e4m3fn`
- `fp8_e5m2`
- `int8`
- `int4`
- `uint4`

These lists define what declarations can name, not what this repository can
execute. In particular, TF32 is an arithmetic mode, advanced formats require
explicit numerical semantics, and a dtype token alone never implies a storage
layout, scale format, accumulator, instruction path, output format, or error
envelope.

### Initial executable-template numeric contract

Both initial fusion templates are constrained to all of the following:

| Contract component | Allowed value |
|---|---|
| Input and weight storage | `fp16` or `bf16` |
| Accumulation | `fp32` |
| Stored output | `fp16`, `bf16`, or `fp32` |
| Quantization | `null` |
| TF32 | `false` |

Those constraints apply to each case/arm numeric contract; vocabulary entries
outside them cannot appear in either initial template. Structural validation
rejects an initial template that requests FP8, integer or sub-byte storage,
quantization, a non-FP32 accumulator, or TF32.

The schema can structurally represent advanced dtype specifications, but not in
either initial template. FP8 must name `fp8_e4m3fn` or `fp8_e5m2`. `int4` and
`uint4` packing is exactly four bits, names its axis, and selects
`low_nibble_first | high_nibble_first`; sub-byte tensor storage requires a
`packed` layout. Any FP8 or integer numeric contract requires non-null
quantization metadata with `per_tensor | per_channel | per_group`, a scale
dtype from `fp32 | fp16 | bf16`, matching `scalar | channel | group` scale
layout, `static | dynamic` calibration, and a positive group size exactly for
`per_group`. TF32 is true exactly when the accumulation dtype is `tf32`.

That representational surface is deliberately not an advanced-suite freeze. It
does not yet close every zero-point, dequantization, rounding, saturation,
instruction-readback, reference, and error-policy choice needed for a fair FP8
or integer campaign. Those choices require a separate reviewed suite/schema
revision, capable arm and baseline set, retained capability evidence, and
passing correctness observations. Advanced dtypes cannot be added to either
initial suite by editing its bytes.

## Frozen reference-template fusion suites

The generic local and remote executor branch remains integrated with exactly two
reference template IDs:

1. `gated_mlp_epilogue.v1`
2. `residual_rmsnorm.v1`

Their closed case semantics and execution plans remain unchanged. The separately
digest-dispatched `residual_rmsnorm_triton.v1` native executor is documented
below; it does not change these reference branches or their legacy evidence.

The committed reference declarations are:

| Artifact | ID | Path | SHA-256 |
|---|---|---|---|
| Plugin | `fusion-reference-plugin` | [`benchmarks/plugins/fusion-reference-plugin-v1.json`](benchmarks/plugins/fusion-reference-plugin-v1.json) | `9d696f135a5e62ef622a88d85a7bb03e8fa76bddd0bf57ebf20b2eb4c1d1edc1` |
| Suite | `gated_mlp_epilogue.v1` | [`benchmarks/suites/gated-mlp-epilogue-v1.json`](benchmarks/suites/gated-mlp-epilogue-v1.json) | `407487a6aa7dc157dcd4aa7bcab698168813bf0a79916d70d91163dc384fe8a8` |
| Suite | `residual_rmsnorm.v1` | [`benchmarks/suites/residual-rmsnorm-v1.json`](benchmarks/suites/residual-rmsnorm-v1.json) | `a318a59bca434b97d073e0ae76f827814213c0a68b0c4263b19c81f98be8f9ee` |

Their `template_status` is `reference_template_not_execution_freeze`: the
bytes and hashes are frozen reference declarations, not a capability probe,
execution freeze, or permission to dispatch work. Changing any byte produces a
different artifact identity and requires an explicit new revision.

## Native Triton residual RMSNorm: integrated, GPU unobserved

The first native Triton candidate is committed at new immutable paths without
changing either reference plugin, either reference suite, or any existing
runtime evidence byte:

| Artifact | ID | Path | SHA-256 |
|---|---|---|---|
| Plugin | `fusion-triton-rmsnorm-plugin` | [`benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json`](benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json) | `ce4a497113adf1ee82ed995fb4ba671a8a1664d756321499d91187056ca0d815` |
| Suite | `residual-rmsnorm-triton` | [`benchmarks/suites/residual-rmsnorm-triton-v1.json`](benchmarks/suites/residual-rmsnorm-triton-v1.json) | `23f7397f2adee93cd9f7919aaf075c0f8b5e92cd6d4257ce4c54197d3c98035f` |

`residual_rmsnorm_triton.v1` retains
`reference_template_not_execution_freeze`: that status freezes declaration
identity and does not itself establish capability or authorize paid dispatch.
The exact declaration digest is now integrated with a dedicated local executor
and Modal executor API, but no SM90 run has been retained. All six arms still
declare both capability states `unprobed` with null evidence digests.

The suite freezes one case: contiguous BF16 `x` and `residual` tensors of shape
`[128, 4096]`, a BF16 `gamma` tensor of shape `[4096]`, pre-normalization
residual addition, epsilon `1e-5`, FP32 arithmetic and reduction, and one BF16
output. Its arms are one eager reference, one Inductor comparator, and these
four native candidates:

| Arm | Structural entrypoint key | `block_size` | `num_warps` | `num_stages` |
|---|---|---:|---:|---:|
| `rmsnorm-triton-w4` | `heliostune_fusion_v2::residual_rmsnorm_w4` | 4096 | 4 | 1 |
| `rmsnorm-triton-w8` | `heliostune_fusion_v2::residual_rmsnorm_w8` | 4096 | 8 | 1 |
| `rmsnorm-triton-w16` | `heliostune_fusion_v2::residual_rmsnorm_w16` | 4096 | 16 | 1 |
| `rmsnorm-triton-w32` | `heliostune_fusion_v2::residual_rmsnorm_w32` | 4096 | 32 | 1 |

The static plan contains exactly twelve cells: one correctness cell followed
by one timing cell for each of the six arms. These are planned cells, not
observations.

The CPU-safe registry in `heliostune.fusion_kernels` exports
`RMSNormTritonConfig`, `RESIDUAL_RMSNORM_CONFIGS`,
`RESIDUAL_RMSNORM_CONFIG_BY_ENTRYPOINT`, and
`load_residual_rmsnorm(entrypoint)`. The integrated executor reaches the GPU-only
`heliostune._fusion_gpu` module lazily; ordinary development and declaration
imports still do not require PyTorch or Triton. The registered custom-op
namespace is `heliostune_fusion_v2::residual_rmsnorm`.

The native result's `heliostune.executor-sources/1` inventory binds the complete
installed `heliostune` package source set by aggregate SHA-256 and file count,
then records path, byte count, and SHA-256 for each execution-critical source:

| Source | Path | SHA-256 |
|---|---|---|
| CPU-safe registry | [`src/heliostune/fusion_kernels.py`](src/heliostune/fusion_kernels.py) | `d1d26114f67b54147337ecf483b75972d3cedbb42f68e2957ceab44ebb4eb7db` |
| GPU-only implementation | [`src/heliostune/_fusion_gpu.py`](src/heliostune/_fusion_gpu.py) | `5f39f6c76a2c542c984bc1be44ca4cd1ccb11c620843a4802f37054f8b0298ef` |
| Native executor | [`src/heliostune/native_fusion_executor.py`](src/heliostune/native_fusion_executor.py) | `53464d63a07f2446fbc403dafd9eb31bd8bc085cae7b5d81ee617edcf51b0da5` |
| Digest dispatcher | [`src/heliostune/local_executor.py`](src/heliostune/local_executor.py) | `258fe67af81a3f079be9c58451736c1f78ff1c145fe668f5a78e075d4027ac44` |

The local bundle writer rechecks that inventory after execution and binds the
exact plugin and suite bytes. The remote path additionally byte-compares the
complete wheel package against the clean source tree and binds its package
source digest through intent, request, result, journal, receipt, wheel manifest,
and committed `HEAD`. These custody records identify code; they are not
capability or execution evidence.

For each native candidate the implemented gates are, in order:

1. compile and complete resource inspection, including `n_spills == 0` and exact
   config/target/kernel identity;
2. canonical frozen-case correctness followed by deterministic `zeros`,
   `cancellation`, and structured `overflow` correctness probes;
3. one warmed invocation with exactly one CUDA profiler event whose name and
   hash match the compiled kernel, plus input/output revalidation; and
4. timing with exactly 10 warmups and 50 retained repetitions.

A failure is retained without eager fallback or retry and blocks that arm's
later gates. Capability rejection invokes no backend and terminalizes every cell
as blocked. The deterministic `heliostune.native-fusion-stage-gate/1` analyzer
requires complete passing eager and Inductor baselines, selects the fastest
fully eligible native candidate, and returns `expand_exploratory` only when
`best_baseline_median / candidate_median >= 1.10`. It always emits
`confirmatory: false`, `fusion_claim: false`, `publication_eligible: false`, and
an empty claims list. With no retained SM90 observation, there is no native
runtime-capability, correctness, one-kernel/fusion, or performance claim.

## Executor integration and reference semantics

### Local CUDA execution

`heliostune run-local-suite SUITE --output DIR` authenticates the selected suite
digest before dispatch. The generic branch for the two frozen reference
templates remains unchanged; use `--plugin PLUGIN` when a suite is not at its
committed template path. It requires the `gpu` extra (including exactly PyTorch
2.8.0), native BF16 support, compute capability 8.0 or newer, and Inductor. For
gated MLP, candidate and reference arithmetic is identical: each projection is
`torch.mm(x.float(), weight.float().T)`, followed by SiLU, multiplication, and
BF16 conversion. The candidate differs only by full-graph Inductor compilation.
Those arms implement the frozen PyTorch formulas, not arbitrary plugin
entrypoints, and neither compilation nor backend invocation proves fusion.

The separately integrated native branch accepts only the exact plugin/suite
pair above and requires an H100 with compute capability 9.0, PyTorch 2.8.0, and
Triton 3.4.0:

```text
uv run --extra gpu heliostune run-local-suite benchmarks/suites/residual-rmsnorm-triton-v1.json --plugin benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json --output /tmp/heliostune-native-rmsnorm
```

This is an invocation contract, not a statement that the current host qualifies.
The native serialized result is `heliostune.local_executor/2`; the written
exploratory output is a strict `heliostune.bundle/1` whose protocol executor API
is `heliostune.native_fusion_executor/2`. The bundle is structurally verified
and preserves passing, failed, blocked, and aborted observations, but is not a
correctness/performance conclusion or publication-eligible claim.

### Remote Modal H100 execution

Build and verify a fresh wheel and its adjacent supplemental manifest from a
clean final Git `HEAD` immediately before each frozen-suite invocation. Run the
gated MLP suite with:

```text
uv run python scripts/build_modal_wheel.py
uv run --extra modal modal run modal_fusion_executor.py::main --suite benchmarks/suites/gated-mlp-epilogue-v1.json --plugin benchmarks/plugins/fusion-reference-plugin-v1.json --output "artifacts/fusion-remote/gated-mlp-epilogue-v1-$(date -u +%Y%m%dT%H%M%S%N)"
```

Run the residual RMSNorm suite with:

```text
uv run python scripts/build_modal_wheel.py
uv run --extra modal modal run modal_fusion_executor.py::main --suite benchmarks/suites/residual-rmsnorm-v1.json --plugin benchmarks/plugins/fusion-reference-plugin-v1.json --output "artifacts/fusion-remote/residual-rmsnorm-v1-$(date -u +%Y%m%dT%H%M%S%N)"
```

The native Modal executor is also implemented, but this guarded command is not
authorization to spend. Run it only for an explicitly approved paid attempt from
a clean committed final `HEAD`:

```text
uv run python scripts/build_modal_wheel.py
uv run --extra modal modal run modal_fusion_executor.py::main --suite benchmarks/suites/residual-rmsnorm-triton-v1.json --plugin benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json --output "artifacts/fusion-remote/residual-rmsnorm-triton-v1-$(date -u +%Y%m%dT%H%M%S%N)"
```

The native suite digest selects `heliostune.modal_fusion_executor/2`; the
reference suite digests continue to select `/1`. No paid native attempt or SM90
observation is reported here.

Each invocation must use its freshly built wheel and a fresh, unique output
directory; never reuse an output directory from an earlier run.

Preflight opens the wheel, verifies ZIP and `RECORD` integrity, and
byte-compares every packaged `heliostune` source/resource file with the clean
`src/heliostune` tree. A manifest that agrees with a tampered wheel cannot
replace that check. The exact verified suite, plugin, and manifest bytes are
retained before dispatch and are the bytes written into the receipt; mutable
input paths are not reread after spawning.

If a previously dispatched call has an exact canonical journal whose terminal
state is already `completed` but receipt publication was interrupted, recover
only that retained result **before rebuilding or replacing the bound wheel or
its adjacent manifest**:

```text
uv run --extra modal python scripts/reconcile_remote_receipt.py --output artifacts/fusion-remote/EXISTING-COMPLETED-OUTPUT
```

This command performs completed-result retrieval only:
`modal.FunctionCall.from_id(call_id).get()` for the single call ID already
bound by the journal. It never spawns, restores, retries, loads, installs, or
executes the historical wheel. It requires the output directory to remain
absent, verifies the retained suite, plugin, old wheel-manifest, intent,
journal, transport, and result bindings, and publishes only the missing
receipt. Intermediate, failed, aborted, cancellation, and unresolved journal
states are refused without changing the journal. A retrieval or validation
failure likewise leaves the journal unchanged and creates no receipt. Current
`HEAD` need not equal the historical intent: the exact retained intent and old
wheel manifest/source bindings are the authority for this recovery path.

The client creates descriptor-pinned, exclusive, fsynced intent and journal
tombstones before its only authorized spawn. It uses `retries=0`, the strict
`H100!` selector, one single-use container, blocked network and Modal-resource
access, and a 3600-second **per-execution** timeout. A returned result is
accepted only after strict request, suite, plugin, wheel, manifest, source,
commit, selector, H100 hardware, environment, and digest-selected local-result
type bindings (`LocalExecutionResult` for `/1`,
`NativeFusionExecutionResult` for `/2`) are checked.

Modal 1.5.4 defines `MAX_ASYNC_OBJECT_SIZE_BYTES = 8 * 1024` in
`modal/_utils/blob_utils.py`; `.spawn()` results above that inline threshold
require blob transport, which is unavailable with restricted Modal-resource
access. The remote function therefore returns a canonical
`heliostune.remote-transport/1` wrapper no larger than 6 KiB. It uses the pinned
`zstandard==0.25.0` codec with fixed deterministic options and standard base64
to carry the canonical result envelope. This compressed wrapper is only a
transport implementation detail: the client strictly bounds and verifies it,
then the receipt retains the decoded canonical result envelope as strict JSON,
exactly as it did without transport compression.

The output is a `heliostune.remote-receipt/1` receipt, **not** a
`heliostune.bundle/1` methodology bundle. Its root is published last by
descriptor-relative staging and atomic no-replace rename. The root and strict
verifier inventory the intent, journal, optional result envelope, retained
suite/plugin/manifest bytes, wheel/source/commit bindings, and lifecycle state.
Completed, failed, and capability-aborted returned results preserve their exact
terminal outcome. A lost spawn acknowledgement, retrieval exception, timeout,
interrupt, malformed result, or unproven cancellation is `unresolved`.

Modal may physically start or restart the same input despite `retries=0`.
Provider physical attempts are unobservable: the receipt proves only zero or
one client-authorized spawn. Therefore 3600 seconds is not a total GPU-time
bound, total GPU time has no stated upper bound, and actual cost is unknown.
Attestation is `none`, `publication_eligible` is false, and the receipt makes no
fusion or performance claim beyond the returned local observations.

### `gated_mlp_epilogue.v1`

Each case declares:

- `activation`: `silu | gelu`;
- `gate_up_layout`: `separate | packed`;
- `bias`: boolean;
- `residual`: boolean;
- `output_arity`: exactly `1`; and
- an ordered fusion boundary listing the operations included in the timed arm.

The ordered boundary prevents a fused arm from being compared with a baseline
that performs different work, hides an extra launch, or excludes a declared
bias/residual operation. `packed` describes the gate/up organization; it does
not mean quantized sub-byte packing.

### `residual_rmsnorm.v1`

Each case declares:

- `epsilon`: finite and strictly positive;
- `gamma`: boolean;
- `residual_position`: `pre | post`;
- `output_arity`: `1 | 2`; and
- an ordered fusion boundary listing the operations and outputs included in the
  timed arm.

Pre- and post-residual RMSNorm are different semantics, as are one- and
two-output variants. They must remain separate cases and cannot be pooled or
silently adapted by a backend.

### Shapes and expected cells

Arm shape constraints are inline and inspectable. Each constraint is exactly a
dimension name, an operation from `divisible_by | min | max | equal`, and an
integer value. Applicability must be derivable from these frozen constraints;
an executor may not silently pad, truncate, reshape, or substitute an arm.

Expected cells are an ordered static plan. For every timing cell, the suite must
contain an earlier correctness-stage cell with the same case, arm, and input
seed. That is a static plan invariant, not a claim that correctness has passed.
At runtime, timing dispatch additionally requires a retained **passing**
correctness observation for that exact case/arm/input-seed key under the frozen
contract. A planned correctness cell, capability state, prior run, or pass on a
different seed is insufficient.

## Baselines

A fusion suite must resolve the applicable baseline hierarchy before evaluation
outcomes are observed. At minimum, it must declare the production incumbent
when one exists, direct vendor or domain-library implementations when
applicable, compiled-framework default and tuned modes when available, eager
framework, a selection reference, and any evaluation oracle. Every required
slot is either bound to an arm or has a retained pre-evaluation reason and
capability-evidence digest for inapplicability/unavailability.

Candidate and baseline arms compared for performance must perform the same
ordered fusion boundary and outputs, consume identical input bytes, use the
same numeric contract, and share timing and environment strata. Tuning claims
also account for candidate count, compile/search observations, workspace and
amortization. An evaluation oracle remains evaluation-only. Missing a strong
applicable baseline blocks promotion; it is not evidence that the baseline is
slow.

## Staged catalog candidates

The following areas are scope candidates, not frozen executable suite IDs:

| Candidate | Required design work before a suite revision |
|---|---|
| Attention and KV cache | Freeze dense/paged semantics, causal and masking behavior, head/group mapping, sequence and ragged layouts, cache update/read boundaries, decode/prefill regimes, reference behavior, and applicable framework/vendor baselines. |
| Quantized linear | Freeze exact integer/sub-byte packing, signedness, group/axis/block scales, zero points, dequantization and accumulation, rounding/saturation, calibration provenance, output contract, and vendor/domain baselines. |
| MoE | Freeze routing and tie behavior, top-k/capacity/overflow policy, dispatch/combine boundaries, token/expert imbalance and ragged shapes, determinism, reference outputs, and distributed/communication scope if present. |
| FP8 | Freeze `fp8_e4m3fn` versus `fp8_e5m2`, scale provenance/granularity, casting and saturation/nonfinite behavior, accumulator/output formats, hardware instruction/readback requirements, error policy, and matched FP16/BF16/vendor baselines. |

Catalog inclusion may record vocabulary and design status. It is not template
status, capability evidence, correctness, performance, or authorization for a
paid run.

A native gated-MLP candidate is explicitly deferred after an unfavorable
feasibility audit. No native gated-MLP plugin, suite, kernel, or execution plan
is included in this revision.

## Promotion rules and implementation order

A catalog-only candidate becomes an executable suite only through a separate
reviewed revision that:

1. freezes semantic cases, fusion boundaries and tensor/output contracts;
2. freezes the exact numeric and error contract, representative and adversarial
   inputs, and seeds;
3. resolves the baseline hierarchy and tuning parity;
4. freezes arms, inline shape constraints, regimes and correctness-before-timing
   expected cells;
5. publishes exact plugin/suite bytes and SHA-256 custody;
6. adds focused structural and behavioral acceptance coverage; and
7. continues to report backend, correctness and performance states separately.

Execution of any promoted candidate still needs a matching backend
implementation and retained probe evidence.
Performance work needs retained passing correctness observations, a frozen
timing protocol and complete evidence lifecycle. A paid campaign additionally
needs an independently approved, frozen paid plan; nothing in this roadmap
promises or authorizes one.

The implemented local and remote executors stop at the frozen gated MLP and
residual RMSNorm reference templates. The native Triton RMSNorm declaration and
source registry remain outside those executors. Attention/KV-cache,
quantized-linear, MoE and FP8 work require their own suite revisions, backend
implementations, and promotion reviews rather than being folded into an
existing template.

## Focused acceptance boundary

The declaration implementation is defended by exactly these twelve named
acceptance tests in `tests/test_scope.py`:

1. `test_legacy_byte_preservation_snapshots`
2. `test_exact_key_and_type_roundtrip`
3. `test_dtype_cross_rules`
4. `test_quantization_cross_rules`
5. `test_capability_evidence_states`
6. `test_gated_mlp_semantics`
7. `test_rmsnorm_semantics`
8. `test_inline_shape_applicability`
9. `test_correctness_before_timing_static_plan`
10. `test_executor_observation_limitation_exposed`
11. `test_plugin_suite_digest_and_path_closure`
12. `test_vocabulary_vs_execution_separation`

Together they cover legacy-byte non-regression, strict closed roots and exact
types, dtype/quantization cross-rules, capability evidence, all frozen
case-semantic unions, inline shape applicability, static and runtime
correctness gates, standalone plugin → suite custody, the integrated native
template's exact structural six-arm/twelve-cell closure, and the separation
between vocabulary, structural source availability, and execution. Those
declaration tests do not substitute for the native executor, bundle, analyzer,
and remote-receipt CPU tests, and none of those CPU tests substitutes for GPU
compilation/resource, correctness, profile, timing, or publication acceptance
described in
[METHODOLOGY.md](METHODOLOGY.md#10-acceptance-tests).

## Inspect and verify

The CPU-only declaration commands are:

```bash
uv run heliostune verify-plugin path/to/plugin.json
uv run heliostune verify-suite path/to/suite.json
uv run heliostune list-scope
```

`verify-plugin` checks the strict plugin root and resolves every relative suite
path and digest. `verify-suite` checks one strict standalone suite. Their
success output reports structure and counts and explicitly disclaims execution,
correctness, and performance observation. `list-scope` prints the closed domain
and dtype vocabularies, the two generic reference template IDs, and the
separately reported native implementation/GPU-unobserved status. The native
Triton RMSNorm suite remains outside `EXECUTABLE_TEMPLATE_IDS` because that
declaration list is not a capability registry.

The declaration commands do not execute kernels or establish correctness or
performance. The generic local and Modal executor branches continue to support
the two frozen reference templates, whose existing
[post-hoc exploratory evidence](benchmarks/results/fusion-remote-exploratory-summary.json)
is unchanged: both suites completed correctness and timing, but plugin
capability declarations remain unprobed, the candidate/reference arithmetic is
identical apart from full-graph compilation, no fusion claim is made, and
receipts are not publication-eligible methodology bundles. The separately
integrated native Triton local `/2` executor and Modal `/2` receipt path have no
retained GPU observation, and all native capability declarations remain
unprobed. No generic executor exists for the staged attention, KV-cache, MoE,
quantized-linear, or FP8 domains.

For the wider protocol, evidence, claim and legacy rules, see
[METHODOLOGY.md](METHODOLOGY.md). For contributor requirements and promotion
review, see [CONTRIBUTING.md](CONTRIBUTING.md).
