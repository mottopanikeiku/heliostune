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
