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
| Template | A suite ID has frozen cases, arms, numeric contracts, fusion semantics, expected cells, and exact artifact bytes. | The two initial suite template JSON files and their SHA-256 values. | That a local or remote backend is implemented. |
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

## Frozen initial fusion suites

Only two suite template IDs are currently executable-suite declarations:

1. `gated_mlp_epilogue.v1`
2. `residual_rmsnorm.v1`

“Executable-suite declaration” means the case semantics and execution plan are
closed enough for the narrow local executor described below. It does not mean a
generic executor is implemented.

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

### Local CUDA execution

`heliostune run-local-suite SUITE --output DIR` executes only these two frozen
templates on a qualifying NVIDIA CUDA device; use `--plugin PLUGIN` when the
suite is not the committed template path. It requires the `gpu` extra (including
exactly PyTorch 2.8.0), native BF16 support, compute capability 8.0 or newer, and
the Inductor backend. For gated MLP, candidate and reference arithmetic is
identical: each projection is `torch.mm(x.float(), weight.float().T)`, followed
by SiLU, multiplication, and BF16 conversion. The candidate differs only by
full-graph Inductor compilation. The arms implement the frozen PyTorch
reference-template formulas, not arbitrary plugin entrypoints, and neither
compilation nor backend invocation is evidence that operations fused. The
written exploratory bundle is structurally verified only: it does not establish
a performance conclusion, claim eligibility, or publication eligibility.

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

Each invocation must use its freshly built wheel and a fresh, unique output
directory; never reuse an output directory from an earlier run.

Preflight opens the wheel, verifies ZIP and `RECORD` integrity, and
byte-compares every packaged `heliostune` source/resource file with the clean
`src/heliostune` tree. A manifest that agrees with a tampered wheel cannot
replace that check. The exact verified suite, plugin, and manifest bytes are
retained before dispatch and are the bytes written into the receipt; mutable
input paths are not reread after spawning.

The client creates descriptor-pinned, exclusive, fsynced intent and journal
tombstones before its only authorized spawn. It uses `retries=0`, the strict
`H100!` selector, one single-use container, blocked network and Modal-resource
access, and a 3600-second **per-execution** timeout. A returned result is
accepted only after strict request, suite, plugin, wheel, manifest, source,
commit, selector, H100 hardware, environment, and `LocalExecutionResult`
bindings are checked.

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

The implemented local executor stops at the frozen gated MLP and residual
RMSNorm reference templates. Attention/KV-cache, quantized-linear, MoE and FP8
work require their own suite revisions, backend implementations, and promotion
reviews rather than being folded into either initial template.

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
types, dtype/quantization cross-rules, capability evidence, both case-semantic
unions, inline shape applicability, static and runtime correctness gates,
standalone plugin → suite custody, and the separation between vocabulary and
execution. Those declaration tests do not stand in for the separate executor,
observation, bundle-custody, analysis, or publication acceptance tests described
in [METHODOLOGY.md](METHODOLOGY.md#10-acceptance-tests).

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
and dtype vocabularies, the two initial template IDs, and the current backend
status.

These commands do not import plugin implementation code, run a capability
probe, dispatch a kernel, validate numerical outputs, collect timing, verify a
generic EvidenceBundle, or support a performance claim. Current generic local
and remote backends for plugin/suite v1 are unimplemented.

For the wider protocol, evidence, claim and legacy rules, see
[METHODOLOGY.md](METHODOLOGY.md). For contributor requirements and promotion
review, see [CONTRIBUTING.md](CONTRIBUTING.md).
