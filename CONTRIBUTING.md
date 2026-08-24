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

## Evidence changes

Files already present under `site/`, the historical benchmark manifest, frozen Parhelion v2 protocol/manifests, and existing compressed data/result artifacts are immutable. New analysis uses a new path, records input/source/output SHA-256 digests, distinguishes confirmatory from post-hoc work, and reports negative, null, or failed outcomes without substitution. Pull requests that alter evidence must state the study ID, analysis status, sampling unit, conditioning set, and chain of custody.
