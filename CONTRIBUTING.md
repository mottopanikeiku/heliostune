# Contributing

> [!IMPORTANT]
> HeliosTune is active and maintained. Contributions to retrieval-first GPU
> autotuning and evidence-control/replay tooling are welcome. The
> transferred-posterior superiority hypothesis is concluded and unsupported;
> do not reopen it by relabeling or strengthening the frozen results. New work
> belongs at new versioned paths with separate custody, while existing evidence
> bytes and claims remain immutable.

## Continuation boundaries

Work proceeds in order. The active v0.6 milestone implemented issue #31's
additive transitive plugin → suite custody and canonical attempt chain for new
local/native bundles, issue #32's durable canonical CPU-only
VerificationRecords, and issue #33's audited deterministic offline CPU analyzer
replay. The next gates are:

1. issue #34: separate active execution dependencies from frozen reproduction
   pins; then
2. run a no-cost feasibility/capability design gate for one new domain.

Each stage must stop until its own implementation and CPU evidence are complete.
Issue #33 changes exactly the replay controls only after an actual isolated
two-run drill; it does not confer claim eligibility. The final design gate
selects at most one domain and authorizes no GPU execution. Only after all gates
pass may contributors propose a new predeclared paid protocol at new versioned
paths; dispatch still requires separate explicit approval and a frozen cost
bound. The maximum authorized spend for this roadmap remains **$0**.

## CPU changes

Use the locked environment and run the focused behavioral test first, then the
complete quality gates:

```bash
uv sync --locked --extra dev
uv run pytest tests/test_offline_replay.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Changes to public types or numerical code must also pass mypy with the real GPU extra. Tests must defend observable behavior, boundary conditions, or evidence invariants rather than implementation text.

## GPU changes

Never start a paid Modal call from an uncommitted tree or a source-mounted image, or without separate explicit approval of a predeclared protocol and cost bound. Build the committed wheel with `scripts/build_modal_wheel.py`, preserve the attempt journal, validate the requested hardware before tensor allocation, and use the exact protocol, banks, dependency versions, and failure rules declared for the campaign. Do not automatically retry a failed or unresolved call.

## Hardened releases

`v0.5.0` is the current published immutable release. `release.yml` is a no-input
manual action that accepts only protected `main` and requires approval through
the `release` environment. It checked out, built, tested, and smoke-checked the
dispatch event's exact `GITHUB_SHA` before tagging that tested snapshot;
operators did not create or push the tag in advance. Later `main` development
for v0.6 does not change the immutable released snapshot.

Every version publishes a wheel, sdist, `SHA256SUMS`, and a verified Git bundle
containing semantic-tag history. Every release asset is attested. The workflow
retains `id-token: write` and `attestations: write` because the provenance action
consumes both permissions and writes the signed statements to the repository
attestation store. Distribution remains GitHub Releases only, with no PyPI
publication. Published versions are immutable; subsequent work receives a new
version.

## Evidence preservation

For a new study in this repository, [HeliosTune methodology v1](METHODOLOGY.md) defines the intended evidence contract: use the exact `heliostune.protocol/1` and `heliostune.bundle/1` schemas or catalog the work explicitly as legacy. Wrapping older bytes never upgrades their eligibility.

Files already present under `site/`, the historical benchmark manifest, frozen Parhelion v2 protocol/manifests, existing compressed data/result artifacts, and the two reference fusion plugin/suite declarations are immutable. New analysis or structural candidates use new versioned paths, record input/source/output SHA-256 digests as applicable, distinguish confirmatory from post-hoc work, and report negative, null, or failed outcomes without substitution. A pull request that adds evidence must state the study ID, analysis status, sampling unit, conditioning set, and chain of custody.

## Experiment-scope preservation

Before changing scope, read [Experiment scope](EXPERIMENT_SCOPE.md) and preserve the distinctions around
`heliostune.plugin/1`, `heliostune.suite/1`, their templates, or the closed
domain/dtype vocabularies. Keep vocabulary membership, structural schema
support, frozen-template inclusion, source registry availability, local/remote
backend capability, correctness observations, and performance observations as
separate states. Schema or template validation and a lazy native source registry
alone must not be described as execution support.

A plugin suite reference must remain a normalized relative path with the exact
suite SHA-256. Changing template semantics, cases, arms, numeric contracts,
fusion boundaries, shape constraints, baselines, regimes, seeds, or expected
cells creates a new suite revision and hash; do not edit a frozen template in
place.

New local/native methodology bundles opt into full custody with flat,
contiguous `plugin_suite_<index>` roles and matching
`plugin_suite_<index>.json` paths in plugin `suite_refs` order. They also include
`selected_suite` at `selected_suite.json`, whose exact closed descriptor has
only schema literal `heliostune.selected-suite/1` and nonnegative integer
`plugin_suite_index: index`. Encode
the descriptor as two-space-indented, lexicographically sorted-key strict JSON
plus one trailing LF. Bundle verification must traverse these inventoried bytes
only—not the plugin's
filesystem paths—and verify every digest, ID, revision, canonical decimal
plugin-version back-reference, and aggregate domain/arm order.

The attempt-chain opt-in is role `attempt_chain` at `attempt_chain.json` with
the exact closed value `{"schema":"heliostune.attempt-chain/1"}`. Encode its
descriptor with the same indented, sorted-key strict JSON plus one trailing LF.
$H_0$ is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
the lowercase SHA-256 of empty bytes. Every journal transition then has exactly
the ordered fields `cell_id`, `predecessor_sha256`, and `status`, encoded as
compact strict JSON plus one LF. Its predecessor is the prior head and its new
head is the SHA-256 of those exact row bytes. Do not pretty-print, reorder keys,
write CRLF, omit the LF, or derive the root head from the whole journal.

The descriptors are additive opt-ins, not permission for partial adoption.
Any role beginning `plugin_suite` or `selected_suite` requires the complete
contiguous inventory. Any `attempt_chain`-prefixed role requires the exact role,
path, media type, payload, and canonical chained rows above. Partial, mixed,
malformed, aliased, or digest-mismatched forms fail closed. Older valid
v1 bundles with no descriptors remain parseable only with the applicable
custody/chain status `not_checked`; never describe that as checked evidence.
The chain provides internal consistency, not a signature or authentication.

Verification records are additive and must never mutate the unchanged
`heliostune.bundle/1` schema or frozen evidence. The exact closed
`heliostune.verification-record/1` root fields are `schema`, `verifier`,
`bundle`, `lifecycle`, `evidence_class`, `controls`, `claim_eligible`, and
`publication_eligible`. Preserve the location-free identities: root
`{bytes,sha256}`; protocol `{path,bytes,sha256,study_id,revision}`; attempts
`{path,bytes,sha256,hash_chain_head}`; and artifact
`{role,path,media_type,bytes,sha256}` entries sorted by `(role,path)`.
`lifecycle` is exactly `{state,outcome}`. Do not add absolute bundle, runtime,
or output paths, timestamps, hostnames, PIDs, executables, or random IDs.

The descriptive verifier identity is package/version plus the fixed
lexicographically ordered source roster `heliostune/_offline_worker.py`,
`heliostune/_reference_analyzer.py`, `heliostune/artifacts.py`,
`heliostune/errors.py`, `heliostune/methodology.py`,
`heliostune/offline_replay.py`, `heliostune/scope.py`,
`heliostune/validation.py`, and `heliostune/verification.py`. Aggregate those
bytes with SHA-256 domain `b"heliostune.verification-sources/1\0"` and, for each
entry, eight-byte big-endian UTF-8 path length, path bytes, eight-byte
big-endian byte count, and raw 32-byte digest. Import-time and build-time
captures must match; unreadable or changed installed sources fail. This is
self-identification, not authentication.

Preserve all twelve control names: `protocol_ancestry`,
`evidence_nonpromotion`, `semantic_content_beyond_digests`,
`plugin_suite_custody`, `attempt_journal_hash_chain`,
`attempt_reconciliation`, `claim_eligibility`, `analyzer_replay`,
`provenance_tier_derivation`, `signature_cryptography`,
`catalog_membership`, and `offline_reproduction`. Their only statuses are
`checked`, `not_checked`, `not_applicable`, and `failed`. Both eligibility
booleans must equal whether every one of the twelve statuses is `checked`; any
other status forces both false. Lifecycle, evidence, and provenance labels do
not enter that formula. Never treat `VERIFIED`, `ANALYZED`, or `PUBLISHED` as
conferring verification or eligibility.

Offline analyzer replay is registry-only. A bundle supplies role `analyzer`
with media type `application/json`; `protocol.analysis.analyzer_sha256` binds
those bytes. Its canonical `AnalyzerManifestV1` has exactly `schema`,
`analyzer_id`, `runner_api`, `implementation`, `inputs`, `outputs`, and
`representation`. Preserve literals `heliostune.analyzer-manifest/1`,
`heliostune.offline-replay/1`, and `byte_exact`.
`AnalyzerImplementationV1` contains only `source_sha256` and ordered `sources`;
each source, input, and output `AnalyzerArtifactBindingV1` contains only `role`,
`media_type`, `bytes`, and `sha256`. All three lists are nonempty and ordered;
roles are nonempty, unique, and disjoint across the lists. Never add a path,
command, import/module name, entrypoint, dependency-fetch hook, or
artifact-controlled executable choice.

Any registry addition must be audited source committed in the package and must
declare an exact static implementation source roster plus ordered
input/output role/media-type contracts. Manifest values and captured bundle
bytes/digests must equal that registry specification. The initial reference ID
is exactly `heliostune.reference.integer-summary/1`; its implementation binding
is `analyzer_source` / `text/x-python`, privately mapped by the registry to
installed `heliostune/_reference_analyzer.py` while captured source bytes are
compared but never executed. Its input is `analysis_input` /
`application/json`, and its output is `analysis_summary` /
`application/json`. Preserve its exact canonical 1–4096 signed-integer input
domain and canonical input-SHA-256/count/min/max/sum output. Registry callables
accept and return ordered `(role, bytes)` tuples and must use only already
loaded pure functions during invocation.

Do not weaken the replay boundary. The parent must require the base record's
custody, attempt-chain, and reconciliation controls to be checked; open and
retain the bundle directory no-follow; capture declared manifest/source/input/
committed-output bytes with the shared bounded descriptor reader; and reverify
the same fd after capture and replay. Preserve the 2 MiB root, 32 MiB component,
and 64 MiB aggregate read limits; declared/descriptor sizes must fail before
bulk reads.

Run the fixed worker twice in distinct empty workspaces through
`/usr/bin/setpriv --no-new-privs /usr/bin/unshare --user --map-root-user --net
--mount --pid --fork --kill-child=SIGKILL --mount-proc`, then the absolute
current Python executable with
`-B -P -s -m heliostune._offline_worker`. Preserve the exact environment
`HOME=/`, `LC_ALL=C`, `LANG=C`, `TZ=UTC`, `PYTHONHASHSEED=0`,
`PYTHONDONTWRITEBYTECODE=1`; bounded canonical base64 request; regular-file
stdout/stderr capture; whole-process-group timeout and communication-failure
handling; worker resource/process bounds; and non-stdio-fd closure. The request
must bind both the exact manifest implementation and the parent's complete
`VerifierIdentityV1`; the child must independently recapture and match its
installed package version and source closure before replay.

Before chroot, require PID 1, effective UID/GID 0, one-ID user/group mappings,
and `NoNewPrivs: 1`; mount a fresh empty mode-0555 tmpfs with
`nosuid,nodev,noexec`; bind-remount it read-only; re-enter the mounted path; and
verify both `ST_RDONLY` and an `EROFS` write probe. The deny-and-latch audit hook
must make attempted open/import/socket/DNS/subprocess/fork/exec/`os.system`/
`ctypes`/audit-hook actions fail even when an analyzer catches the immediate
exception. It is a tripwire, not the primary sandbox; user/network namespaces,
the read-only empty tmpfs chroot, and resource/process limits provide the
primary isolation. Never add a fallback when any required executable,
namespace, mount, chroot, or other isolation setup is unavailable.
Reject nonzero status, timeout, communication failure, any stderr, malformed,
trailing, or oversized output, role/count/order mismatch, nondeterministic
runs, or output differing from pre-captured committed bytes.

`build_replay_verification_record_v1` must accept only a success-only
`OfflineReplayResult`, never a caller-supplied record or status map. It
reconstructs the base record and changes exactly `analyzer_replay` and
`offline_reproduction` to `checked`, preserving every other control and
bundle/lifecycle/evidence fact and recomputing eligibility.
`write_offline_replay_record_v1(path, result)` must validate that same
runner-minted result before using the safe record publisher. Neither may
replace the original `VerifiedBundle` limitations. Success means only same-host
reproduction of declared committed
bytes by the registered CPU analyzer—not authenticity, cross-host bit
reproducibility, provider truth, semantic/statistical truth, GPU recollection,
claim eligibility/promotion, or full dependency/campaign reproduction. Replay
must download nothing, invoke no backend/GPU, and make no paid call; its spend
boundary is **$0**.

Canonical record bytes are two-space-indented, sorted-key strict JSON with one
trailing LF; loaders require byte-identical re-encoding. File output requires
an exact record/`VerifiedBundle` match and must finish building and encoding
before touching output. The destination must be a new sibling of the bundle
directory: its existing parent must be that directory's immediate parent and
match the device/inode captured through the pinned bundle descriptor. This
runtime parent identity is non-wire; arbitrary destinations use JSON stdout
redirection. The writer pins the exact parent without following symlinks, opens
unnamed `O_TMPFILE` storage there, fsyncs the payload, and uses an unprivileged
procfd source for atomic no-replace `linkat`. Lack of either capability fails
closed. Closing the unnamed fd performs cleanup. Recheck the
bundle-parent relationship immediately before and after linking. A successful
link is irreversible, but no topology immutability is promised against hostile
renames: the requested pathname may become stale or unrecoverable. Pre-link
failure creates no destination. Post-link failure reports committed/ambiguous
state: the complete linked destination is not rolled back, but directory
durability may be ambiguous.
Preserve the agreed CLI:

```bash
uv run heliostune verify-bundle path/to/bundle/bundle.json
uv run heliostune verify-bundle path/to/bundle/bundle.json --format json
uv run heliostune verify-bundle path/to/bundle/bundle.json --output path/to/bundle.verification.json
uv run heliostune replay-bundle path/to/bundle/bundle.json
uv run heliostune replay-bundle path/to/bundle/bundle.json --format json
uv run heliostune replay-bundle path/to/bundle/bundle.json --output path/to/bundle.replay-verification.json
```

No flags retain human text; successful replay text is headed `Bundle offline
replay verified` and names the path, analyzer ID, runner API, sandbox, two
runs, each output role/byte count/SHA-256, every original limitation, every
upgraded control, and both eligibility booleans. `--format json` emits exact
record bytes to stdout without Rich; `--output PATH` implies JSON and is
silent. Explicit `--format text --output PATH` must fail before verification
or replay. Structurally verified base records with deferred controls exit zero
and remain ineligible. Replay emits only after the complete drill succeeds.
Any failed control or verification/manifest/isolation/audit/replay/build/
encoding/pre-link write error exits 2 with no success bytes or destination. A
post-link failure also exits 2 and reports committed/ambiguous state; hostile
rebinding can make the requested pathname stale or unrecoverable. The complete
linked destination is not rolled back, but directory durability may be
ambiguous. Neither record is a signature, authentication, provider truth,
semantic/statistical correctness, claim promotion, or full
dependency/campaign reproduction.

The two runtime-integrated reference templates permit only FP16/BF16
input/storage, FP32 accumulation, FP16/BF16/FP32 output, null quantization, and
disabled TF32. Advanced dtype work requires a separate revision with explicit
format, storage/packing, scale/zero-point, accumulator/output,
rounding/saturation, precision-readback, reference, and error semantics as
applicable. A timing plan must have an earlier correctness cell for the same
case, arm, and input seed; an executor must also retain a passing observation
for that exact key before timing dispatch.

For a new or promoted suite, the pull request must:

1. freeze complete case semantics, ordered fusion boundaries, tensor/output
   contracts, inline shape constraints, numeric/reference policies, and seeds;
2. resolve the applicable production, vendor/domain, compiled-framework,
   eager-framework, selection-reference, and evaluation-oracle baseline slots,
   with retained pre-evaluation evidence for unavailable/inapplicable slots;
3. freeze arm applicability, tuning allowances, regimes, and the ordered
   correctness-before-timing plan;
4. add focused strict-schema, cross-field, digest/reference, and behavioral
   tests, then run `verify-plugin`, `verify-suite`, and `list-scope`; and
5. update this guide, [METHODOLOGY.md](METHODOLOGY.md), and the README state
   summary without converting declaration support into a correctness or
   performance claim.

Native residual RMSNorm is executor-integrated only for the exact immutable
[`benchmarks/suites/residual-rmsnorm-triton-v1.json`](benchmarks/suites/residual-rmsnorm-triton-v1.json)
and
[`benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json`](benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json)
pair. The local result is `heliostune.local_executor/2`, published through
`heliostune.native_fusion_executor/2` as a strict `heliostune.bundle/1`; the
digest-selected Modal API is `heliostune.modal_fusion_executor/2`, published as
`heliostune.remote-receipt/1`. A retained H100 stage-gate observation exists,
but it does not promote capability or establish methodology-v1 authenticity:
all six arms' local and remote declarations remain `unprobed`, and contributors
must not describe the suite as available or executable on their current host.

The four native configurations remain fixed at `block_size=4096`,
`num_warps=4|8|16|32`, and `num_stages=1`. Preserve the gate order: compile and
complete zero-spill resource evidence; canonical correctness and deterministic
`zeros`/`cancellation`/structured-`overflow` probes; exactly one matching CUDA
profiler event for one invocation with input/output revalidation; then exactly
10 warmups and 50 timing repetitions. A failure must be retained without eager
fallback or retry and must block that arm's later gates. Capability rejection
must invoke no backend and retain all cells as blocked.

The retained H100 [report](site/native-rmsnorm-h100.html), strict
[summary](benchmarks/results/native-rmsnorm-h100-summary.json), compressed
[raw evidence](benchmarks/data/native-rmsnorm-h100.json.zst), and
[publication manifest](benchmarks/native-rmsnorm-h100-manifest.json) record all
four native candidates passing correctness, complete zero-spill resource, and
profile gates and becoming eligible. Each profiled one-invocation check observed
exactly one matching CUDA kernel event, but `fusion_claim=false`. The native
winner was `rmsnorm-triton-w8` at median **0.0505920015 ms**, versus eager at
**0.085072 ms** and Inductor at **0.045952 ms**. The fair
`best_baseline_median / candidate_median` ratio was **0.908286**, below the
predeclared **1.10** threshold, so the exact decision is
**`STOP_BELOW_THRESHOLD`**, with no expansion or correctness, performance,
fusion, or publication-eligibility claim.

The first attempt remains an unresolved receipt after its result exceeded the
6144-byte inline transport limit and establishes no execution result; a compact
transport retry completed afterward. Provider physical starts and restarts,
total GPU time and its upper bound, total time upper bound, and actual cost
remain unknown. This publication is retained static evidence, not a methodology
bundle, and remains `publication_eligible=false`.

Do not substitute copied or renamed declarations in either documented command.
Local invocation is restricted to the exact H100 SM90, PyTorch 2.8.0, and Triton
3.4.0 predicate. No documented command authorizes a paid Modal invocation. An
approved run must follow the clean committed-wheel rule, use the separately
approved frozen protocol and cost bound, write to a fresh versioned output path,
and preserve the one-spawn journal. `retries=0` does not make Modal physical
starts observable. An unresolved receipt must remain unresolved; a separately
authorized follow-up needs a fresh output and attempt journal and
must not overwrite or reinterpret the earlier receipt.

Preserve complete source custody: exact plugin/suite bytes, the package-wide
source digest and count, and the path/size/digest inventory for
`fusion_kernels.py`, `_fusion_gpu.py`, `native_fusion_executor.py`, and
`local_executor.py`. Remote custody additionally binds the byte-verified wheel,
adjacent manifest, clean commit, request, journal, and returned result.

Do not weaken descriptor-pinned publication. Arbitrary bundle verification must
open the resolved bundle directory once, open normalized components
descriptor-relatively without following symlinks, hash and parse each regular
file from the same descriptor, and reject duplicate `(st_dev, st_ino)`
identities. Staging producers must call
`verify_bundle_v1_from_directory_fd` with their already-open staging directory;
the verifier duplicates that descriptor, and both pre-rename and post-rename
verification operate solely through the pinned fd. Any supplied path is a
diagnostic label only, never path-resolution authority. New local/native
producers must emit both descriptors and the full
inventory, then require `plugin_suite_custody`,
`attempt_journal_hash_chain`, and no-retry `attempt_reconciliation` to be
`checked` before atomic no-replace rename. Provider physical retries/billing,
signatures/authenticity, analyzer replay, and complete offline reproduction
remain `not_checked`, as does every applicable control for which no check ran.
Preserve failure and unresolved-state retention.

The deterministic stage gate may authorize exploratory expansion only at
`1.10x` or better versus the faster complete eager/Inductor baseline. The
retained H100 ratio was `0.908286`, so it authorized no expansion. The analysis
remains non-confirmatory, makes no correctness, fusion, or performance claim,
and is not publication eligible. Native gated MLP remains absent after an
unfavorable feasibility audit. Attention/KV cache, quantized linear, MoE, and
FP8 remain catalog/design inventory for the ordered no-cost domain gate, not
executable suites or authorization for promotion. At most one may advance only
after the analyzer-replay and dependency-split stages are complete, through its
own reviewed revision and no-cost feasibility/capability
evidence. Any later paid proposal requires a new frozen protocol, approved
bound, committed bytes, new versioned paths, and the full evidence controls
above.

The published Parhelion and Hopper studies remain immutable legacy plugins.
Do not relabel or migrate them to plugin/suite v1 merely because a declaration
can represent some of their vocabulary.
