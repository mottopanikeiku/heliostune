# HeliosTune experiment scope

## Status

This active scope record describes the implemented `heliostune.plugin/1` and
`heliostune.suite/1` declaration surface, additive structural custody controls
for new `heliostune.bundle/1` outputs, canonical CPU-only
`heliostune.verification-record/1`, audited registry-only analyzer replay, and
the bounded roadmap for surfaces not yet implemented. It is narrower than the
normative evidence lifecycle in [METHODOLOGY.md](METHODOLOGY.md): checked
internal custody, a durable record, and a successful same-host analyzer drill
are not execution validation, authenticity, semantic or statistical
correctness, provider truth, cross-host/GPU reproduction, full dependency or
campaign reproduction, or a claim.

The schemas are additive and non-retroactive. The published Parhelion and
Hopper studies remain immutable legacy plugins and evidence. They have not been
migrated to plugin/suite v1, and new declarations do not upgrade their
eligibility, custody, numerical checks, or conclusions. New work may land in
this repository only at new versioned paths with its own custody.

## Keep the state axes separate

The following states answer different questions and must not be collapsed into
one "supported" label.

| Axis | What it says | Where it is represented now | What it does not say |
|---|---|---|---|
| Vocabulary | A domain, dtype, shape operator, or case option is a recognized closed enum value. | Strict plugin/suite parser and `heliostune list-scope`. | That any frozen suite uses it or any backend can execute it. |
| Schema | A `heliostune.plugin/1` or `heliostune.suite/1` document has exact JSON types, no unknown or duplicate fields, and satisfies its cross-field rules. | `verify-plugin` and `verify-suite`. | That referenced code imports, a GPU is present, or a case is correct or fast. |
| Template | A suite ID has frozen cases, arms, numeric contracts, fusion semantics, expected cells, and exact artifact bytes. | The two generic reference templates and the separately integrated native Triton template, with their SHA-256 values below. | That a local or remote capability probe passed, or that the template is correct or fast. |
| Bundle inventory custody | A new bundle's opted-in plugin and all transitive suites are internally closed by exact inventoried bytes, identities, order, and digests. | Reserved flat artifacts plus `heliostune verify-bundle`. | Who authored or executed the bytes, provider attempts/billing, semantic correctness, analyzer replay, full reproduction, or claim eligibility. |
| Verification record | A canonical location-free record binds the exact structurally verified bundle, twelve explicit control statuses, and a descriptive nine-file verifier identity. | `heliostune verify-bundle PATH --format json` or sibling-only `--output SIBLING_PATH`; successful replay emits an exact upgraded record through the parallel `replay-bundle` modes. | A signature, authentication, provider truth, semantic or statistical correctness, full dependency/campaign reproduction, or eligibility when any control is not `checked`. |
| Offline analyzer replay | A strict manifest selects one fixed audited CPU analyzer and binds its implementation sources plus ordered input/output artifact identities; two isolated runs reproduce the pre-captured committed outputs byte-for-byte. | `heliostune replay-bundle PATH` after custody, attempt-chain, and reconciliation controls are checked. | Artifact-supplied code execution, authenticity, cross-host bit reproducibility, semantic/statistical truth, GPU recollection, claim promotion, or complete software/campaign reproduction. |
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
file and verifies the referenced bytes and suite structure. The same in-memory
inventory validator now supports an additive strict bundle mode that never
follows the plugin's filesystem suite paths: it traverses only bytes already
listed in the bundle artifact inventory, so verification is offline and cannot
silently substitute a suite from the checkout.

In plugin `suite_refs` order, strict bundles reserve contiguous roles
`plugin_suite_<index>` at flat paths `plugin_suite_<index>.json`. Role
`selected_suite` at `selected_suite.json` is an exact closed descriptor with
only schema literal `heliostune.selected-suite/1` and nonnegative integer
`plugin_suite_index: index`. Its
payload uses two-space-indented, lexicographically sorted-key strict JSON plus
one trailing LF. The integer binds the inventoried suite the bundle declares
selected; it does not prove backend execution. The verifier checks
each reference's digest, suite ID and revision; each suite's plugin ID and
canonical decimal plugin-version back-reference; the protocol plugin identity
and digest; and the plugin's first-seen aggregate domain and arm order.

This inventory is a strict opt-in. With none of its reserved roles, an older
valid v1 bundle remains parseable and reports
`plugin_suite_custody: not_checked`; that status is not evidence of custody.
Any role beginning
`plugin_suite` or `selected_suite` activates complete mode, and a malformed
prefix, missing descriptor, missing or gapped index, extra suite, identity
mismatch, path mismatch, or digest mismatch fails closed.

Attempt chaining is independently selected by reserved role `attempt_chain` at
`attempt_chain.json`, containing the exact closed value
`{"schema":"heliostune.attempt-chain/1"}`. Its payload uses the same indented,
sorted-key strict JSON plus one trailing LF. Let $H_0$ be the SHA-256 of empty
bytes, `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Each row has exactly the ordered fields `cell_id`,
`predecessor_sha256`, and `status`, encoded as compact strict JSON plus one LF;
its predecessor is the prior head, and the new head is the SHA-256 of those
exact row bytes. The bundle root's unchanged `hash_chain_head` stores the final
head, including $H_0$ for an empty journal. Noncanonical encoding, CRLF,
missing LF, blank lines, reorder, truncation, duplicate or unknown transitions,
or a broken predecessor fails. Any `attempt_chain`-prefixed role activates
strict handling, and every form except the exact role, path, media type, and
payload above fails. With no such prefix, only exact legacy two-field rows
remain accepted and chain status is `not_checked`; mixed or partial opt-in never
falls back. This digest chain is an internal consistency
control, not authentication or a provider signature.

Arbitrary bundle verification opens the resolved bundle directory once, opens
normalized relative components descriptor-relatively without following
symlinks, and hashes and parses each regular file through the same descriptor.
Duplicate `(st_dev, st_ino)` identities are rejected, so hard links cannot
alias two declared roles. Staging producers pass their already-open directory
to `verify_bundle_v1_from_directory_fd`, which duplicates the descriptor.
Pre-rename and post-rename verification operate solely through that pinned fd;
any supplied path is a diagnostic label, not path-resolution authority. New
local and native producers keep all dynamic files flat, emit the complete
inventory and both descriptors, and require
plugin custody, attempt chaining, and no-retry reconciliation to be `checked`
before atomic no-replace rename. Reconciliation is checked only when journal
rows evidence all final logical states, retry policy is `none`,
`max_physical_attempts` is one, physical equals logical, and orphaned is zero.
Provider retry/billing reconciliation remains unchecked.

## Canonical verification record

Issue #32, implemented only after issue #31's custody and attempt-chain
controls, adds a durable CPU-only `VerificationRecordV1` without changing the
`heliostune.bundle/1` root. The exact closed record has top-level fields
`schema`, `verifier`, `bundle`, `lifecycle`, `evidence_class`, `controls`,
`claim_eligible`, and `publication_eligible`, with schema literal
`heliostune.verification-record/1`.

`verifier` contains `package`, `version`, `source_sha256`, and ordered `sources`
whose entries contain `path`, `bytes`, and `sha256`. `bundle` contains the
unchanged bundle `schema` and `bundle_id`; location-free root `bytes` and
`sha256`; protocol `path`, `bytes`, `sha256`, `study_id`, and `revision`;
attempts `path`, `bytes`, `sha256`, and `hash_chain_head`; and artifact entries
with `role`, `path`, `media_type`, `bytes`, and `sha256`, sorted by `(role,
path)`. `lifecycle` is the exact bundle `state` and `outcome`; `evidence_class`
is copied from the bound protocol.

The twelve exact control names are `protocol_ancestry`,
`evidence_nonpromotion`, `semantic_content_beyond_digests`,
`plugin_suite_custody`, `attempt_journal_hash_chain`,
`attempt_reconciliation`, `claim_eligibility`, `analyzer_replay`,
`provenance_tier_derivation`, `signature_cryptography`,
`catalog_membership`, and `offline_reproduction`. Each status is exactly
`checked | not_checked | not_applicable | failed`. Both eligibility booleans
equal `all_checked`, meaning every one of the twelve statuses is `checked`; any
`not_checked`, `not_applicable`, or `failed` status forces both false.
Lifecycle state, evidence class, and provenance never alter that formula, so
even `VERIFIED`, `ANALYZED`, or `PUBLISHED` labels do not confer verification
or eligibility. Base `verify-bundle` records retain replay controls as
`not_checked`; successful `replay-bundle` records change exactly
`analyzer_replay` and `offline_reproduction` to `checked`, while other deferred
controls keep the current records ineligible.

Verifier source identity is a descriptive self-identity over this fixed
lexicographically ordered roster: `heliostune/_offline_worker.py`,
`heliostune/_reference_analyzer.py`, `heliostune/artifacts.py`,
`heliostune/errors.py`, `heliostune/methodology.py`,
`heliostune/offline_replay.py`, `heliostune/scope.py`,
`heliostune/validation.py`, and `heliostune/verification.py`. The aggregate
SHA-256 uses domain `b"heliostune.verification-sources/1\0"` followed, for each
entry, by its UTF-8 path length as eight-byte big-endian, path bytes, byte count
as eight-byte big-endian, and raw 32-byte digest. Import-time and build-time
captures must match. This identifies installed source bytes; it does not
authenticate them.

Canonical output is two-space-indented, sorted-key strict JSON with one
trailing LF and excludes absolute bundle, runtime, and output paths, timestamps,
hostnames, PIDs, executables, and random IDs. Canonical loading rejects any byte
representation that does not re-encode identically. Safe file output first
builds and encodes a record that exactly matches its `VerifiedBundle`, then
requires a new sibling file whose existing parent is the verified bundle
directory's immediate parent with the exact device/inode captured through the
pinned descriptor. That runtime identity is non-wire; arbitrary destinations
use JSON stdout redirection. The relationship is rechecked immediately before
and after the irreversible no-replace link. This does not promise topology
immutability against hostile renames: the requested pathname may become stale
or unrecoverable. Pre-link failure creates no destination; post-link failure
reports committed/ambiguous state: the complete linked destination is not
rolled back, but directory durability may be ambiguous. The writer uses unnamed
`O_TMPFILE` storage and an unprivileged procfd source for atomic no-replace
`linkat`; it fails closed if either capability is unavailable. Closing the
unnamed fd performs cleanup. Building the base record is not analyzer replay.
Neither a base nor upgraded record is a signature, authentication, provider
truth, semantic or statistical correctness, claim promotion, or full
dependency/campaign reproduction.

## Audited CPU offline analyzer replay

Issue #33 adds a strict replay opt-in without changing
`heliostune.bundle/1`. The bundle artifact role remains `analyzer` with media
type `application/json`, and `protocol.analysis.analyzer_sha256` binds its exact
bytes. Those bytes must canonically encode exact-field `AnalyzerManifestV1`:
schema `heliostune.analyzer-manifest/1`, a registry `analyzer_id`, runner API
`heliostune.offline-replay/1`, `AnalyzerImplementationV1{source_sha256,sources}`,
nonempty ordered `inputs` and `outputs`, and representation `byte_exact`.
Source, input, and output entries are
`AnalyzerArtifactBindingV1{role,media_type,bytes,sha256}`. All three lists are
nonempty and ordered; roles are nonempty, unique, and disjoint across the
lists. A manifest contains no artifact path, command, module, import, or
entrypoint.

The fixed registry—not bundle bytes—maps analyzer IDs to already imported
callables and static source/role/media-type specifications. The manifest must
exactly match that specification and every bound bundle artifact byte count and
digest. The initial and only registered ID is
`heliostune.reference.integer-summary/1`. Its sole implementation binding is
role `analyzer_source` with media type `text/x-python`; the registry privately
maps it to installed `heliostune/_reference_analyzer.py`, and captured bundle
source bytes are compared but never executed. Its input is `analysis_input` /
`application/json`, and its output is `analysis_summary` /
`application/json`. It accepts exact canonical JSON whose sole field `values`
is an array of 1–4096 signed, exact JSON integers, and emits canonical JSON
that binds the input SHA-256, count, minimum, maximum, and sum. During invocation
it uses only already-loaded pure functions: no imports or external actions.

Replay first rebuilds the base record and requires
`plugin_suite_custody`, `attempt_journal_hash_chain`, and
`attempt_reconciliation` to be `checked`; caller-supplied statuses cannot
substitute. The bundle directory is opened no-follow and must match its retained
device/inode. Manifest, implementation source, input, and committed output bytes
are captured through the same bounded descriptor-relative methodology reader
with byte/digest checks. The reader rejects roots above 2 MiB, individual
components above 32 MiB, and more than 64 MiB in one verification/capture
before bulk reads. The same directory fd is reverified after capture and after
replay.

The parent runs the fixed worker twice, each time in a distinct empty
workspace. Its fixed absolute launch prefix is `/usr/bin/setpriv
--no-new-privs /usr/bin/unshare --user --map-root-user --net --mount --pid
--fork --kill-child=SIGKILL --mount-proc`; it then starts the absolute current
Python executable with `-B -P -s -m heliostune._offline_worker`. The exact
child environment is `HOME=/`, `LC_ALL=C`, `LANG=C`, `TZ=UTC`,
`PYTHONHASHSEED=0`, and `PYTHONDONTWRITEBYTECODE=1`. The bounded request is
canonical base64 on stdin; bounded stdout/stderr use regular files; timeout or
communication failure kills and reaps the process group. The request binds the
manifest implementation and complete parent verifier identity; the worker
independently recaptures its installed version and nine-file source closure and
requires an exact match before invoking the registry.

After preloading the fixed registry and reading its request, the worker requires
inner PID 1, effective UID/GID 0, one-ID user/group maps, and
`NoNewPrivs: 1`. It mounts a fresh empty mode-0555 tmpfs with
`nosuid,nodev,noexec`, bind-remounts it read-only, re-enters the mounted path,
and verifies `ST_RDONLY` plus an `EROFS` write probe before chrooting and
changing to `/`. It closes non-stdio descriptors, applies CPU/address-space/
file-size/fd/process/output bounds, and installs a deny-and-latch audit hook.
Attempts to open/import, use sockets or DNS, invoke subprocess/fork/exec/
`os.system`/`ctypes`, or add an audit hook fail the run even when the analyzer
catches the immediate exception. The namespaces, read-only empty tmpfs chroot,
and resource/process bounds are the sandbox. The Python audit hook is a
tripwire, not the primary sandbox. If any required executable, namespace,
mount, chroot, or other isolation setup is unavailable, replay fails closed
with no weaker fallback.
Tests may skip only positive real-sandbox drills after a fixed capability probe
establishes a host user-namespace policy denial. Runtime replay still fails and
cannot produce checked controls.

The parent rejects timeout, nonzero status, any stderr, malformed, trailing, or
oversized output, output role/count/order mismatch, run-one/run-two byte
difference, or any difference from the pre-captured committed output bytes.
`OfflineReplayResult` retains the original `VerifiedBundle`, manifest, both
runs' output identities, and the upgraded record.
`build_replay_verification_record_v1` accepts only that success-only result,
never a caller-supplied record/status map; it reconstructs the base record,
changes exactly `analyzer_replay` and `offline_reproduction` to `checked`,
leaves all other controls and bound facts identical, and recomputes eligibility.
`write_offline_replay_record_v1(path, result)` validates the same runner-minted
result before using the existing safe record publisher. Neither replaces the
`VerifiedBundle` limitations.

This proves only same-host registered-analyzer reproduction of declared
committed derived bytes. It does not establish authenticity, cross-host bit
reproducibility, provider truth, semantic or statistical correctness, GPU
recollection, claim eligibility/promotion, or full dependency/campaign
reproduction. It downloads nothing, invokes no GPU/backend, makes no paid call,
and keeps the authorized spend at **$0**.

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

## Native Triton residual RMSNorm: retained H100 stage-gate outcome

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
The exact declaration digest is integrated with a dedicated local executor and
Modal executor API and now has one retained H100 SM90 observation. That
observation does not promote declaration state: all six arms still declare
both capability states `unprobed` with null evidence digests.

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

The static declaration plan contains exactly twelve cells: one correctness cell
followed by one timing cell for each of the six arms. The cells remain declaration
plan identity; the retained H100 observations are published separately below.

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
| CPU-safe registry | [`src/heliostune/fusion_kernels.py`](src/heliostune/fusion_kernels.py) | `4577047cc30310bd3be4fa165be7d256d8007dbfded1830203eaa4c8968ef40e` |
| GPU-only implementation | [`src/heliostune/_fusion_gpu.py`](src/heliostune/_fusion_gpu.py) | `5f39f6c76a2c542c984bc1be44ca4cd1ccb11c620843a4802f37054f8b0298ef` |
| Native executor | [`src/heliostune/native_fusion_executor.py`](src/heliostune/native_fusion_executor.py) | `84994f00b004d4f277386624866afd759f138625015d81b5c2733dee999f6b9c` |
| Digest dispatcher | [`src/heliostune/local_executor.py`](src/heliostune/local_executor.py) | `0b11075af36909d5799467d4c31d1594d4dc25e4918ca91ca65600cc708db1ce` |

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

A failure is retained without eager fallback or automatic retry and blocks that
arm's later gates. Capability rejection invokes no backend and terminalizes
every cell as blocked. The deterministic
`heliostune.native-fusion-stage-gate/1` analyzer requires complete passing eager
and Inductor baselines, selects the fastest fully eligible native candidate, and
returns `expand_exploratory` only when
`best_baseline_median / candidate_median >= 1.10`. It always emits
`confirmatory: false`, `fusion_claim: false`, `publication_eligible: false`, and
an empty claims list.

The retained H100 [report](site/native-rmsnorm-h100.html), strict
[summary](benchmarks/results/native-rmsnorm-h100-summary.json), compressed
[raw evidence](benchmarks/data/native-rmsnorm-h100.json.zst), and
[publication manifest](benchmarks/native-rmsnorm-h100-manifest.json) retain the
completed stage gate. All four native candidates passed the correctness,
complete zero-spill resource, and profile gates and were eligible. Each profiled
one-invocation check observed exactly one matching CUDA kernel event, while the
analysis remains `fusion_claim=false`. The fastest native candidate was
`rmsnorm-triton-w8` at median **0.0505920015 ms**; eager measured **0.085072
ms** and Inductor measured **0.045952 ms**. The fair
`best_baseline_median / candidate_median` ratio was **0.908286**, below the
predeclared **1.10** threshold. The retained decision is
**`STOP_BELOW_THRESHOLD`**, with no expansion and no correctness, performance,
fusion, or publication-eligibility claim.

The publication also retains the first attempt as unresolved after its result
exceeded the 6144-byte inline transport limit; it establishes no execution
result. A compact-result attempt completed afterward. Modal provider physical
starts and restarts, total GPU time and its upper bound, total time upper bound,
and actual cost are unknown. These are remote receipts and a static evidence
publication, not a methodology bundle, and `publication_eligible=false`.

## Executor integration and reference semantics

### Local CUDA execution

`heliostune run-local-suite SUITE --output DIR` verifies the selected suite
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
is `heliostune.native_fusion_executor/2`. New outputs include the full
plugin-suite inventory, selected-suite and attempt-chain descriptors, and
canonical chained rows. Publication requires successful structural custody,
chain verification, and no-retry reconciliation before atomic rename. These
checks preserve passing, failed, blocked, and aborted observations, but do not
establish a correctness/performance conclusion, provider authenticity,
analyzer replay, full offline reproduction, or publication eligibility.

### Historical remote Modal H100 execution

The commands below preserve the historical invocation contract; they do not
authorize a paid call. The procedure built and verified a fresh wheel and its
adjacent supplemental manifest from a clean committed Git `HEAD`. Retained gated
MLP reference command:

```text
uv run python scripts/build_modal_wheel.py
uv run --extra modal modal run modal_fusion_executor.py::main --suite benchmarks/suites/gated-mlp-epilogue-v1.json --plugin benchmarks/plugins/fusion-reference-plugin-v1.json --output "artifacts/fusion-remote/gated-mlp-epilogue-v1-$(date -u +%Y%m%dT%H%M%S%N)"
```

Retained residual RMSNorm reference command:

```text
uv run python scripts/build_modal_wheel.py
uv run --extra modal modal run modal_fusion_executor.py::main --suite benchmarks/suites/residual-rmsnorm-v1.json --plugin benchmarks/plugins/fusion-reference-plugin-v1.json --output "artifacts/fusion-remote/residual-rmsnorm-v1-$(date -u +%Y%m%dT%H%M%S%N)"
```

The native Modal executor is also implemented. Its guarded command is retained
for provenance and may run only under a separately approved, predeclared paid
protocol and cost bound at a fresh versioned path:

```text
uv run python scripts/build_modal_wheel.py
uv run --extra modal modal run modal_fusion_executor.py::main --suite benchmarks/suites/residual-rmsnorm-triton-v1.json --plugin benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json --output "artifacts/fusion-remote/residual-rmsnorm-triton-v1-$(date -u +%Y%m%dT%H%M%S%N)"
```

The native suite digest selects `heliostune.modal_fusion_executor/2`; the
reference suite digests continue to select `/1`. The native H100 publication
above retains one unresolved transport-overflow attempt and one completed
compact-result attempt without changing these dispatch identities.

The preserved invocation contract requires a freshly built wheel and a fresh,
unique output directory; an earlier output directory must never be reused.

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

## Bounded continuation roadmap

The active v0.6 roadmap is ordered and fail-closed:

| Stage | Current state and required work | Stop boundary |
|---|---|---|
| 1a. Issue #31: custody and attempt chain | Implemented first for new local/native bundles: unchanged v1 roots gain additive full plugin → suite inventory, selected-suite identity, canonical predecessor chaining, descriptor-contained reads, and truthful no-retry reconciliation. CPU fixtures cover completed, failed, aborted, legacy-unchecked, and tampered bundles. | This closes internal custody and chaining only. **STOP** on any missing/mismatched inventoried byte, reserved descriptor, predecessor, lifecycle/accounting fact, or unsafe path. No signature, provider-truth, analyzer, or full-reproduction claim follows. |
| 1b. Issue #32: VerificationRecord | Implemented after #31: durable canonical CPU-only records bind exact inputs, descriptive nine-file verifier identity, and every control without upgrading deferred controls. | **STOP** on nondeterministic, noncanonical, incomplete, mismatched, failed-control, overwrite, or non-sibling file output. A base record is not eligibility or analyzer replay. |
| 1c. Issue #33: analyzer replay | Implemented: strict digest-bound manifests select only the fixed audited CPU registry; descriptor-pinned capture, fixed user/network/mount/PID namespaces, an empty read-only `nosuid,nodev,noexec` tmpfs chroot with namespace/no-new-privileges self-checks, audit tripwire, resource bounds, two distinct workspaces, committed-byte comparison, and exact record upgrade fail closed. | **STOP** on undeclared access, registry/source/role mismatch, network or filesystem action, unavailable isolation, stderr/nonzero/timeout, malformed output, nondeterminism, committed-byte mismatch, or record drift. Success is only same-host reproduction of registered-analyzer derived bytes. |
| 2. Issue #34: dependency split | Next: separate actively maintained execution dependencies from frozen reproduction pins, and verify that each historical environment remains reproducible without constraining the active environment. | **STOP** if an active dependency update changes frozen reproduction identity or if a reproduction pin silently governs new execution. |
| 3. One-domain no-cost design gate | After #34, select at most one inventory domain below and freeze a reviewed feasibility/capability design: semantics, numerics, applicability, baseline hierarchy, backend requirements, custody, expected cells, and explicit infeasibility criteria. The gate performs no paid execution and makes no correctness or performance claim. | **STOP** on missing semantics, baseline, feasible backend design, custody plan, or bounded execution design. Other inventory domains remain inventory. |
| 4. Optional paid-protocol proposal | Only after the preceding stages pass may a new, predeclared protocol with new versioned paths and its own frozen paid plan be proposed for separate approval. | A passed design gate permits review of a proposal, not dispatch. Without explicit approval and a cost bound, **STOP** before any paid call. |

This roadmap authorizes no GPU spending; its maximum authorized spend is
**$0**. Stage evidence and decisions must be retained even when they stop
continuation. Issues #31–#33 do not check signatures or authenticity, provider
physical retries or billing, semantic/statistical truth, GPU recollection,
cross-host reproduction, or full dependency/campaign reproduction. Base records
retain replay controls as `not_checked`; successful issue-#33 drills check only
the two replay controls, while every other deferred status remains truthful.

The stage-3 inventory contains no frozen executable suite ID:

| Candidate | Design requirements for a new suite revision |
|---|---|
| Attention and KV cache | Freeze dense/paged semantics, causal and masking behavior, head/group mapping, sequence and ragged layouts, cache update/read boundaries, decode/prefill regimes, reference behavior, and applicable framework/vendor baselines. |
| Quantized linear | Freeze exact integer/sub-byte packing, signedness, group/axis/block scales, zero points, dequantization and accumulation, rounding/saturation, calibration provenance, output contract, and vendor/domain baselines. |
| MoE | Freeze routing and tie behavior, top-k/capacity/overflow policy, dispatch/combine boundaries, token/expert imbalance and ragged shapes, determinism, reference outputs, and distributed/communication scope if present. |
| FP8 | Freeze `fp8_e4m3fn` versus `fp8_e5m2`, scale provenance/granularity, casting and saturation/nonfinite behavior, accumulator/output formats, hardware instruction/readback requirements, error policy, and matched FP16/BF16/vendor baselines. |

Catalog inclusion records vocabulary and design status only. Retaining a row is
not template status, capability evidence, correctness, performance, an
implementation commitment, or authorization for a paid run.

The native gated-MLP candidate was deferred after an unfavorable feasibility
audit. No native gated-MLP plugin, suite, kernel, or execution plan currently
exists; that retained outcome is unchanged and the candidate does not bypass
the ordered roadmap.

### Promotion requirements for the selected domain

HeliosTune may turn the single stage-3 selection into an executable suite only
through a separate reviewed revision that:

1. freezes semantic cases, fusion boundaries and tensor/output contracts;
2. freezes the exact numeric and error contract, representative and adversarial
   inputs, and seeds;
3. resolves the baseline hierarchy and tuning parity;
4. freezes arms, inline shape constraints, regimes and correctness-before-timing
   expected cells;
5. publishes exact plugin/suite bytes and SHA-256 custody;
6. adds focused structural and behavioral acceptance coverage; and
7. continues to report backend, correctness and performance states separately.

Execution still needs a matching backend implementation and retained probe
evidence. Performance work needs retained passing correctness observations, a
frozen timing protocol, and a complete evidence lifecycle. A paid proposal
additionally needs its own independently approved, frozen paid plan; neither
this document nor a merged suite revision authorizes dispatch.

The current local and remote reference branches retain the frozen gated MLP and
residual RMSNorm reference templates. The native Triton RMSNorm suite is
structurally executable: its frozen digest dispatches to dedicated local and
Modal native executors, and the retained H100 observation establishes only the
narrow stage-gate facts published above. It does not promote the suite's
capability declarations or authorize a claim. Attention/KV-cache,
quantized-linear, MoE, and FP8 remain inventory until the ordered gate selects
one through a new suite revision, backend, custody record, and review.

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

Separate focused tests in `tests/test_methodology.py`,
`tests/test_local_bundle.py`, and `tests/test_native_fusion_bundle.py` defend the
issue-#31 bundle surface: strict/legacy opt-in selection, complete in-memory
plugin-suite inventory, selected-suite identity, canonical predecessor bytes
and empty head, sealed-state reconciliation, descriptor/inode containment, and
producer pre-rename postconditions. Issue-#32 verification and CLI tests defend
deterministic source and bundle identity, all four statuses, eligibility
nonpromotion, canonical round trips, and descriptor-identified sibling-only
atomic no-replace output. They cover bundle-parent rechecks, pre-link absence,
and retained complete records with committed/ambiguous state after post-link
failure or hostile rebinding.

`tests/test_offline_replay.py` and the replay CLI cases defend issue #33's
manifest closure, exact registry/source/role matching, reference analyzer input
contract, prerequisite controls, descriptor-pinned capture/reverification,
fixed namespace/chroot/audit boundaries, resource/output bounds, two-workspace
determinism, committed-byte comparison, exact record upgrade, and fail-closed
file output. CI runs that focused CPU replay file explicitly. Passing it
establishes only the bounded same-host replay behavior; it is no substitute for
GPU, authenticity, semantics/statistics, claim, dependency, or campaign tests.

## Inspect and verify

The CPU-only structural, record, and replay commands are:

```bash
uv run heliostune verify-plugin path/to/plugin.json
uv run heliostune verify-suite path/to/suite.json
uv run heliostune verify-bundle path/to/bundle/bundle.json
uv run heliostune verify-bundle path/to/bundle/bundle.json --format json
uv run heliostune verify-bundle path/to/bundle/bundle.json --output path/to/bundle.verification.json
uv run heliostune replay-bundle path/to/bundle/bundle.json
uv run heliostune replay-bundle path/to/bundle/bundle.json --format json
uv run heliostune replay-bundle path/to/bundle/bundle.json --output path/to/bundle.replay-verification.json
uv run heliostune list-scope
```

`verify-plugin` checks the strict plugin root and resolves every relative suite
path and digest. `verify-suite` checks one strict standalone suite. With no
output flags, both bundle commands retain human-readable text. `verify-bundle`
stops after structural verification; `replay-bundle` emits only after the
isolated two-run drill and record upgrade succeed. `--format json` writes exact
canonical record bytes to stdout without Rich. `--output PATH` implies JSON and
silently creates a new sibling file only when its existing parent is the
descriptor-identified immediate parent of the verified bundle directory;
replay uses its exact-result validator before the same safe record publisher.
Arbitrary paths use external JSON stdout redirection. Explicit `--format text
--output PATH` fails before verification or replay.

Structurally verified base records exit zero with deferred controls but remain
ineligible. Replay exits zero only after both runs and committed outputs match.
A failed control or verification/manifest/isolation/audit/replay/build/encoding/
pre-link write error exits 2 without success bytes or a destination. The
relationship is rechecked immediately before and after the irreversible
no-replace link. A hostile rename or rebinding can make the requested pathname
stale or unrecoverable. A post-link error exits 2 and reports
committed/ambiguous state: the complete linked destination is not rolled back,
but directory durability may be ambiguous.

Replay is CPU-only, network-disabled, and $0; it makes no backend/GPU execution,
semantic or statistical correctness, performance, provider, billing,
signature/authentication, cross-host reproducibility, claim-promotion, or full
dependency/campaign-reproduction assertion. `list-scope` prints the closed
domain and dtype vocabularies and all three structurally executable template
IDs in `EXECUTABLE_TEMPLATE_IDS`, then reports native implementation status
separately. Membership in that tuple records structural executability, not
retained runtime capability; the separately published H100 observation does
not promote any declaration's `unprobed` capability state.

The generic local and Modal executor branches remain implemented for the two
frozen reference templates, whose existing
[post-hoc exploratory evidence](benchmarks/results/fusion-remote-exploratory-summary.json)
is unchanged: both suites completed correctness and timing, but plugin
capability declarations remain unprobed, the candidate/reference arithmetic is
identical apart from full-graph compilation, no fusion claim is made, and
receipts are not publication-eligible methodology bundles. The separately
integrated native Triton local `/2` executor and Modal `/2` receipt path have the
retained H100 stage-gate observation described above; its declarations remain
unprobed and its publication is not methodology-bundle or claim eligible. No
generic executor exists for the staged attention, KV-cache, MoE,
quantized-linear, or FP8 domains.

For the protocol, evidence, claim, and legacy rules, see
[METHODOLOGY.md](METHODOLOGY.md). For contribution and evidence-preservation
requirements in this repository, see [CONTRIBUTING.md](CONTRIBUTING.md).
