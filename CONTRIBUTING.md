# Contributing

## CPU changes

Use the locked environment and run the focused behavioral test first, then the complete quality gates:

```bash
uv sync --locked --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Changes to public types or numerical code must also pass mypy with the real GPU extra. Tests must defend observable behavior, boundary conditions, or evidence invariants rather than implementation text.

## GPU changes

Never start a paid Modal call from an uncommitted tree or a source-mounted image. Build the committed wheel with `scripts/build_modal_wheel.py`, preserve the attempt journal, validate the requested hardware before tensor allocation, and use the exact protocol, banks, dependency versions, and failure rules declared for the campaign. Do not automatically retry a failed or unresolved call.

## Releases

Distribution is GitHub Releases only. There is deliberately no PyPI publish step, and none should be added without a separate decision: the wheel and sdist attached to a tag are the only supported artifacts, and `pip install` from an index is not a published installation path even though `pyproject.toml` carries complete metadata for one.

`release.yml` keeps `id-token: write` and `attestations: write` because both are consumed: `actions/attest-build-provenance` mints its signed provenance statement through the OIDC token (`id-token: write`) and writes the resulting attestation to the repository's attestation store (`attestations: write`). Neither permission is vestigial and neither may be dropped while the attestation step exists.

A tag matching `v*` runs the shared `checks.yml` workflow first; the release job depends on it via `needs`, so ruff, mypy, coverage, and the 3.11/3.12/3.13 matrix all gate the release. The release concurrency group is tag-scoped and does not cancel in progress, because a half-created release is worse than a duplicated run.

## Evidence changes

The normative contract for new evidence is [HeliosTune methodology v1](METHODOLOGY.md). A new study must use the exact `heliostune.protocol/1` and `heliostune.bundle/1` schemas or be explicitly cataloged as legacy; wrapping older bytes never upgrades their eligibility.

Files already present under `site/`, the historical benchmark manifest, frozen Parhelion v2 protocol/manifests, existing compressed data/result artifacts, and the two reference fusion plugin/suite declarations are immutable. New analysis or structural candidates use new paths, record input/source/output SHA-256 digests as applicable, distinguish confirmatory from post-hoc work, and report negative, null, or failed outcomes without substitution. Pull requests that alter evidence must state the study ID, analysis status, sampling unit, conditioning set, and chain of custody.

## Experiment-scope changes

Read [Experiment scope](EXPERIMENT_SCOPE.md) before changing
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
place. Generic EvidenceBundle transitive plugin → suite custody is not
implemented and must not be claimed until its verifier and tests exist.

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
`heliostune.remote-receipt/1`. Implementation is not capability: no retained
SM90 run exists, all six arms' local and remote declarations remain `unprobed`,
and contributors must not describe the suite as available or executable on
their current host.

The four native configurations remain fixed at `block_size=4096`,
`num_warps=4|8|16|32`, and `num_stages=1`. Preserve the gate order: compile and
complete zero-spill resource evidence; canonical correctness and deterministic
`zeros`/`cancellation`/structured-`overflow` probes; exactly one matching CUDA
profiler event for one invocation with input/output revalidation; then exactly
10 warmups and 50 timing repetitions. A failure must be retained without eager
fallback or retry and must block that arm's later gates. Capability rejection
must invoke no backend and retain all cells as blocked.

Do not substitute copied or renamed declarations in either documented command.
Local invocation is restricted to the exact H100 SM90, PyTorch 2.8.0, and Triton
3.4.0 predicate. Before any paid Modal invocation, follow the clean committed
wheel rule above, obtain explicit authorization and approved bounds, use a fresh
unique output path, and preserve the one-spawn journal. `retries=0` does not make
Modal physical starts observable; never state a total GPU-time or cost bound and
never retry an unresolved receipt.

Preserve complete source custody: exact plugin/suite bytes, the package-wide
source digest and count, and the path/size/digest inventory for
`fusion_kernels.py`, `_fusion_gpu.py`, `native_fusion_executor.py`, and
`local_executor.py`. Remote custody additionally binds the byte-verified wheel,
adjacent manifest, clean commit, request, journal, and returned result. Do not
weaken descriptor-pinned publication, atomic no-replace output, or failure and
unresolved-state retention.

The deterministic stage gate may authorize exploratory expansion only at
`1.10x` or better versus the faster complete eager/Inductor baseline. It remains
non-confirmatory, makes no correctness, fusion, or performance claim, and is not
publication eligible. Native gated MLP is deferred after an unfavorable
feasibility audit and must remain absent from this increment. Attention/KV
cache, quantized linear, MoE, and FP8 are catalog/design candidates until
separate promotion revisions meet the above requirements. A scope or template
pull request does not authorize a paid campaign; paid execution needs its own
frozen protocol, approved bounds, committed bytes, and evidence controls.

The published Parhelion and Hopper studies remain immutable legacy plugins.
Do not relabel or migrate them to plugin/suite v1 merely because a declaration
can represent some of their vocabulary.
