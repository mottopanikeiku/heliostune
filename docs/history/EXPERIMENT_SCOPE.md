# Suite and bundle formats

The `heliostune` CLI can inspect plugin and suite declarations, verify stored
bundles and replay registered CPU analyzers. The format names and historical
JSON keys are unchanged. Implementations live in
[`src/heliostune/tooling/`](../../src/heliostune/tooling/).

## Inspecting declarations

These commands require no GPU:

```bash
nice -n 10 uv run --locked heliostune list-scope
nice -n 10 uv run --locked heliostune verify-plugin benchmarks/plugins/fusion-reference-plugin-v1.json
nice -n 10 uv run --locked heliostune verify-suite benchmarks/suites/residual-rmsnorm-v1.json
nice -n 10 uv run --locked heliostune verify-protocol benchmarks/methodology-protocol-v1-template.json
nice -n 10 uv run --locked heliostune verify-catalog benchmarks/research-artifact-manifest.json
```

`verify-plugin` checks the plugin and each referenced suite's identity and
SHA-256. `verify-suite` checks a standalone declaration. Neither command
executes a kernel. `list-scope` reports recognized vocabulary and template IDs;
local and remote capability remain separate `unprobed`, `available` or
`unavailable` observations.

## Plugins and suites

[`scope.py`](../../src/heliostune/tooling/scope.py) defines the exact fields for
`heliostune.plugin/1` and `heliostune.suite/1`. Objects reject unknown or duplicate
keys, wrong scalar types, nonfinite numbers and malformed digests. Plugin suite
references contain a normalized relative path, exact SHA-256, suite ID and
revision. A suite names its plugin, numerical contracts, tensors, arms, cases,
correctness and timing policies, and ordered execution plan.

The registered templates are:

| Template | Declaration | Plugin |
|---|---|---|
| `gated_mlp_epilogue.v1` | [Gated MLP](../../benchmarks/suites/gated-mlp-epilogue-v1.json) | [Reference plugin](../../benchmarks/plugins/fusion-reference-plugin-v1.json) |
| `residual_rmsnorm.v1` | [Residual RMSNorm](../../benchmarks/suites/residual-rmsnorm-v1.json) | [Reference plugin](../../benchmarks/plugins/fusion-reference-plugin-v1.json) |
| `residual_rmsnorm_triton.v1` | [Native RMSNorm](../../benchmarks/suites/residual-rmsnorm-triton-v1.json) | [Native plugin](../../benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json) |

Dispatch uses the exact suite and plugin digests in
[`fusion_execution_registry.py`](../../src/heliostune/fusion_execution_registry.py).
Changing an operation, shape, numerical contract, arm or execution plan requires
a new declaration revision.

The domain vocabulary includes `dense_gemm`, `fused_mlp`, `rmsnorm_residual`,
`attention`, `kv_cache`, `moe` and `quantized_linear`. The dtype vocabulary
includes `fp32`, `tf32`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp8_e5m2`, `int8`,
`int4` and `uint4`. Attention, KV-cache, MoE, quantized-linear and FP8 execution
are not implemented by these suite executors.

The two reference templates allow FP16/BF16 input and weight storage, FP32
accumulation, FP16/BF16/FP32 output, null quantization and disabled TF32.
Advanced-format declarations have additional packing and scale fields, but
cannot be substituted into these templates. Four-bit storage requires a packed
layout, packing axis and nibble order. FP8 and integer contracts require
quantization metadata; that alone does not supply an executable kernel.

Gated MLP cases specify activation, separate or packed gate/up layout, bias,
residual, one output and an ordered list of operations. RMSNorm cases specify
positive epsilon, gamma, pre- or post-residual position, one or two outputs and
an ordered list of operations. Candidate and baseline must perform the same
operations and produce the same outputs. The shape constraints are inline
`divisible_by`, `min`, `max` or `equal` predicates; executors cannot silently
pad, truncate or substitute cases.

Every timing entry must have an earlier correctness entry for the same case,
arm and input seed. Execution also requires a passing observation for that
exact combination before timing it.

## GPU execution

The reference executor needs the `gpu` extra, PyTorch 2.8.0, native BF16 support,
compute capability 8.0 or newer, and Inductor. Its candidate and reference use
the same PyTorch formulas; the candidate uses full-graph compilation.

The native RMSNorm executor requires an H100 SM90, PyTorch 2.8.0 and Triton 3.4.0:

```bash
nice -n 10 uv run --locked --extra gpu heliostune run-local-suite \
  benchmarks/suites/residual-rmsnorm-triton-v1.json \
  --plugin benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json \
  --output artifacts/native-rmsnorm/NAME
```

Choose a new output path. The four native configurations use block size 4096,
4/8/16/32 warps and one stage. Each must pass compilation with zero spills,
correctness including zero/cancellation/overflow inputs, and a profiler check
for exactly one matching CUDA kernel event. Timing then uses 10 warmups and
50 repetitions. A failed stage blocks later stages for that arm, without an
eager fallback or retry.

The local native output uses `heliostune.local_executor/2`, packaged as a
`heliostune.bundle/1`. The [stored H100 result](../../benchmarks/results/native-rmsnorm-h100-summary.json)
selected the eight-warp candidate at 0.050592 ms versus Inductor at 0.045952 ms.
The baseline/candidate ratio was 0.908286, below the 1.10 expansion threshold,
so the decision was `STOP_BELOW_THRESHOLD`. The declaration's capability fields
remain `unprobed`; the stored run does not establish availability on another
host.

### Remote collection and interrupted retrieval

Paid runs require the approval and hardware checks in
[Contributing](../../CONTRIBUTING.md#collecting-action-expansion-measurements).
Build the wheel from a clean committed checkout and use a new output directory:

```bash
nice -n 10 uv run --locked python scripts/build_modal_wheel.py
nice -n 10 uv run --locked --extra modal modal run modal_fusion_executor.py::main \
  --suite benchmarks/suites/residual-rmsnorm-triton-v1.json \
  --plugin benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json \
  --output artifacts/fusion-remote/NAME
```

The native suite selects `heliostune.modal_fusion_executor/2`; the reference
suites select `/1`. Remote output is `heliostune.remote-receipt/1`. It binds the
request, attempt journal, exact declarations, wheel, source and returned
result. A lost acknowledgement, timeout or unproven cancellation stays
`unresolved`. The client allows one spawn; provider restarts and actual cost
are not observable from that journal.

If the journal already says `completed` but receipt publication was interrupted,
retrieve the existing result before rebuilding or replacing the bound wheel
or its adjacent manifest:

```bash
nice -n 10 uv run --locked --extra modal python scripts/reconcile_remote_receipt.py --output artifacts/fusion-remote/NAME
```

This retrieves the journal's existing call ID and writes the missing receipt.
The destination directory must still be absent. It never starts a replacement
call, and refuses failed, aborted, intermediate or unresolved journal states.

## Protocols and bundles

[`methodology.py`](../../src/heliostune/tooling/methodology.py) defines
`heliostune.protocol/1` and `heliostune.bundle/1`; the
[protocol template](../../benchmarks/methodology-protocol-v1-template.json)
shows the root structure. A protocol binds workload, candidate, comparator,
split, numerical, timing, analysis and execution objects by digest. Its
`evidence_class` is `exploratory`, `engineering_gate` or `confirmatory`.

A bundle inventories the protocol, attempt journal and artifacts by relative
path, byte count and SHA-256. Verification checks unique paths and roles,
referenced bytes, lifecycle/outcome compatibility, expected and terminal work
IDs, coverage counts and attempt transitions. The supported lifecycle labels
run from `DRAFT` through `RESOLVED`, `FROZEN`, `DISPATCHED`, `SEALED`, `VERIFIED`,
`ANALYZED` and `PUBLISHED`. They do not imply that every verification check ran.
Older study manifests retain their own schemas and interpretation.

Plugin-suite verification uses contiguous `plugin_suite_<index>` roles and
matching flat `plugin_suite_<index>.json` paths in plugin-reference order.
`selected_suite.json` contains only `schema: "heliostune.selected-suite/1"` and
`plugin_suite_index`. Its canonical encoding is two-space-indented, sorted-key
JSON with one trailing LF. Every suite digest, identity, revision and plugin
back-reference must match. A reserved role opts into the complete format;
partial or malformed inventories are rejected. Older bundles without it report
`plugin_suite_custody: not_checked`.

The optional `attempt_chain` role at `attempt_chain.json` contains
`{"schema":"heliostune.attempt-chain/1"}` using the same canonical encoding.
Chained journal rows have exactly `cell_id`, `predecessor_sha256` and `status`,
in that order, as compact JSON followed by LF. The initial head is SHA-256 of
empty bytes. Each row's digest becomes the next head; the bundle stores the
last head. Reordering, truncation, invalid transitions or noncanonical encoding
fails verification. Legacy unchained journals report the chain as
`not_checked`. The chain checks internal consistency, not authorship.

Bundle reads use one open directory descriptor, reject symlinks and repeated
file identities, and hash and parse each file through the same descriptor.
Limits are 2 MiB for the root, 32 MiB per component and 64 MiB per verification
or capture. New local producers verify their staged inventory before and
after atomic publication without overwriting an existing destination.

## Verification records

For a bundle at `path/to/bundle/bundle.json`:

```bash
nice -n 10 uv run --locked heliostune verify-bundle path/to/bundle/bundle.json
nice -n 10 uv run --locked heliostune verify-bundle path/to/bundle/bundle.json --format json
nice -n 10 uv run --locked heliostune verify-bundle path/to/bundle/bundle.json --output path/to/bundle.verification.json
```

The record schema is `heliostune.verification-record/1`. It contains verifier
identity, bundle identities, lifecycle, evidence class, controls and two
eligibility booleans. The twelve control names are:

```text
protocol_ancestry                 evidence_nonpromotion
semantic_content_beyond_digests   plugin_suite_custody
attempt_journal_hash_chain       attempt_reconciliation
claim_eligibility                analyzer_replay
provenance_tier_derivation       signature_cryptography
catalog_membership              offline_reproduction
```

Each status is `checked`, `not_checked`, `not_applicable` or `failed`.
`claim_eligible` and `publication_eligible` are true only when all twelve are
`checked`. Structural verification leaves unimplemented checks unset; a zero
exit status alone does not make a result eligible.

[`verification.py`](../../src/heliostune/tooling/verification.py) defines the
current source roster and accepts the two historical rosters when loading old
records. Source identity hashes the ordered paths, sizes and digests. It
identifies installed bytes without authenticating their origin.

Records use sorted-key, two-space-indented JSON plus LF, with no absolute paths
or runtime identifiers. `--format json` writes to stdout. `--output` implies
JSON and accepts only a new sibling of the bundle directory. It uses
`O_TMPFILE` and atomic no-replace `linkat`, and fails if either is unavailable.
Explicit `--format text --output` is invalid. Failures exit 2; a post-link
failure reports committed/ambiguous state and leaves the linked file intact.

## Offline analyzer replay

```bash
nice -n 10 uv run --locked heliostune replay-bundle path/to/bundle/bundle.json
nice -n 10 uv run --locked heliostune replay-bundle path/to/bundle/bundle.json --format json
nice -n 10 uv run --locked heliostune replay-bundle path/to/bundle/bundle.json --output path/to/bundle.replay-verification.json
```

Replay requires checked plugin-suite, attempt-chain and attempt-reconciliation
controls. The bundle's `analyzer` artifact is `application/json`, bound by
`protocol.analysis.analyzer_sha256`. Its `heliostune.analyzer-manifest/1` fields
are `schema`, `analyzer_id`, `runner_api`, `implementation`, `inputs`, `outputs`
and `representation`. The runner API is `heliostune.offline-replay/1` and the
representation is `byte_exact`.

Implementation sources, inputs and outputs are nonempty ordered lists of
`{role, media_type, bytes, sha256}` bindings; roles are unique across all three.
The implementation also has a `source_sha256`. A manifest cannot supply a
command, import or executable path. The installed registry selects the code.

The sole analyzer ID is `heliostune.reference.integer-summary/1`. Its
`analyzer_source` binding maps to the installed
[`_reference_analyzer.py`](../../src/heliostune/tooling/_reference_analyzer.py).
Captured source bytes are compared, never executed. Input role `analysis_input`
contains canonical JSON with a `values` array of 1–4096 signed integers. Output
role `analysis_summary` contains the input digest, count, minimum, maximum and
sum. Both use `application/json`.

The parent captures inputs and declared outputs through the bounded bundle
reader, runs the registered analyzer twice in separate isolated workspaces,
and requires both outputs to match the captured bytes. It rejects timeouts,
nonzero exits, stderr, malformed or oversized output, role/order mismatches
and differences between runs. It reverifies the same bundle descriptor after
capture and replay. See the [security policy](../../SECURITY.md) for the Linux
isolation requirements.

Successful replay changes only `analyzer_replay` and `offline_reproduction` to
`checked`, then recomputes eligibility. It performs no GPU collection or
network download. Signature verification, provider accounting, full experiment
reproduction and statistical validation remain separate, unimplemented checks.
