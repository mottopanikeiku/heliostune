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

Files already present under `site/`, the historical benchmark manifest, frozen Parhelion v2 protocol/manifests, and existing compressed data/result artifacts are immutable. New analysis uses a new path, records input/source/output SHA-256 digests, distinguishes confirmatory from post-hoc work, and reports negative, null, or failed outcomes without substitution. Pull requests that alter evidence must state the study ID, analysis status, sampling unit, conditioning set, and chain of custody.

## Experiment-scope changes

Read [Experiment scope](EXPERIMENT_SCOPE.md) before changing
`heliostune.plugin/1`, `heliostune.suite/1`, their templates, or the closed
domain/dtype vocabularies. Keep vocabulary membership, structural schema
support, frozen-template inclusion, local/remote backend capability,
correctness observations, and performance observations as separate states.
Schema or template validation must not import plugin code or be described as
execution support.

A plugin suite reference must remain a normalized relative path with the exact
suite SHA-256. Changing template semantics, cases, arms, numeric contracts,
fusion boundaries, shape constraints, baselines, regimes, seeds, or expected
cells creates a new suite revision and hash; do not edit a frozen template in
place. Generic EvidenceBundle transitive plugin → suite custody is not
implemented and must not be claimed until its verifier and tests exist.

The initial templates permit only FP16/BF16 input/storage, FP32 accumulation,
FP16/BF16/FP32 output, null quantization, and disabled TF32. Advanced dtype
work requires a separate revision with explicit format, storage/packing,
scale/zero-point, accumulator/output, rounding/saturation, precision-readback,
reference, and error semantics as applicable. A timing plan must have an
earlier correctness cell for the same case, arm, and input seed; an executor
must also retain a passing observation for that exact key before timing
dispatch.

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

Implementation priority is fused gated MLP, then residual RMSNorm. Attention/KV
cache, quantized linear, MoE, and FP8 are catalog/design candidates until
separate promotion revisions meet the above requirements. A scope or template
pull request does not authorize a paid campaign; paid execution needs its own
frozen protocol, approved bounds, committed bytes, and evidence controls.

The published Parhelion and Hopper studies remain immutable legacy plugins.
Do not relabel or migrate them to plugin/suite v1 merely because a declaration
can represent some of their vocabulary.
