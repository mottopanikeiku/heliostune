# Contributing

The tuner lives in `src/heliostune/bandit.py`, `retrieval.py` and
`multisource_engine.py`. It can be tested without a GPU. Offline analysis,
artifact verification and report code live in `src/heliostune/tooling/`.

## Local development

```bash
nice -n 10 uv sync --locked --extra dev
nice -n 10 uv run --locked pytest -q -x tests/test_bandit.py tests/test_retrieval.py tests/test_multisource.py tests/test_tuner_results.py tests/test_matmul_audit.py
nice -n 10 uv run --locked pytest -q -x
```

Run the relevant behavioral tests before the full suite. The `dev` extra does
not install Torch or Triton. GPU tests need the `gpu` extra and compatible
hardware; isolated replay tests also need Linux user namespaces. A skipped test
leaves that execution path untested. Keep dependency changes in `uv.lock` and
pin workflow actions to commit SHAs.

## Regenerating analyses

These commands read stored measurements and need no GPU:

```bash
nice -n 10 uv run --locked python scripts/audit_matmul_action_set.py
nice -n 10 uv run --locked python scripts/summarize_tuner_results.py
nice -n 10 uv run --locked python scripts/analyze_action_expansion.py
nice -n 10 uv run --locked python scripts/diagnose_kernel_bank.py
nice -n 10 uv run --locked python scripts/build_site.py
```

The first two rebuild the historical action-set audit and tuner comparison.
The expansion analyzer writes the summary and SVG from
`results/action-expansion-raw.json`. The diagnosis reads the stored action
matrices; the site builder uses it and the expansion summary to render
`site/index.html` and the README's `site/h100-by-m.svg` figure. The last two
commands accept `--check` to compare generated output without writing files.

Historical reports are bound by the benchmark manifests. Leave those reports
and their input measurements unchanged; use new paths for new experiments.
The old landing page is `site/transfer.html`, which
`historical_artifact_path` resolves for historical `site/index.html` bindings.
The [study index](docs/history/INDEX.md) links reports and their inputs.

## Collecting action-expansion measurements

Read the [measurement protocol](docs/action-expansion.md). Both collectors use
`measure_unit` and `collect_on_device` in `src/heliostune/action_expansion.py`;
`src/heliostune/action_collection.py` defines run identity and scheduling.
Commit all tracked changes before collecting. Validate the requested hardware
before allocating tensors. Paid calls need explicit approval of the protocol
and cost bound; do not automatically retry a failed or unresolved call.

Choose a lowercase run ID, such as `h100-repeat-1`. This CPU-only command prints
the collection plan without importing Torch:

```bash
nice -n 10 uv run --locked python scripts/collect_action_expansion.py --run-id h100-repeat-1 --gpu H100 --dry-run
```

On a CUDA machine:

```bash
nice -n 10 uv sync --locked --extra dev --extra gpu
nice -n 10 uv run --locked python scripts/collect_action_expansion.py --run-id h100-repeat-1 --gpu H100
```

Use the actual hardware label, such as `A100-80GB`. Labels without an expectation
in `src/heliostune/hardware.py` require `--allow-unlisted-gpu`; the run then binds
to the first observed hardware profile. Tensor-descriptor actions may fail on
older GPUs. Failed actions stay in the output and cannot be selected.

For an approved Modal H100 run, use Python 3.13 and build the wheel from the
clean checkout:

```bash
nice -n 10 uv sync --locked --python 3.13 --extra dev --extra modal
nice -n 10 uv run --locked python scripts/build_modal_wheel.py
nice -n 10 uv run --locked modal run modal_action_expansion.py --run-id h100-repeat-1
```

`--pilot` collects four workloads separately. It does not supply observations
to the full summary. The full Modal call has a 27-minute remote timeout.

Checkpoints use `h100-repeat-1/full/checkpoint.json` or `h100-repeat-1/pilot/checkpoint.json`, under
`~/.cache/heliostune/action-runs` locally or `/data/runs` on the
`heliostune-day-actions` Modal volume. The collector saves after each completed
workload and measurement-set unit. Resuming reuses completed units only when
the run ID, protocol commit, hardware profile, source hashes, precision settings,
protocol and actions match. Changes require a new run ID. Each collection
session is recorded separately.

Raw output goes to `results/action-expansion-runs/h100-repeat-1.json` or
`h100-repeat-1-pilot.json` and cannot replace an existing file. To analyze a full run:

```bash
nice -n 10 uv run --locked python scripts/analyze_action_expansion.py --input results/action-expansion-runs/h100-repeat-1.json --output results/action-expansion-runs/h100-repeat-1-summary.json --figure results/action-expansion-runs/h100-repeat-1.svg
```

The original full run predates run IDs. Its checkpoint was
`/data/full-checkpoint.json`; current collectors do not read that file.

## Reporting results

Keep new measurements separate from analyses of existing data. Name the inputs,
hardware, numerical contract and numerator/denominator for each comparison.
Keep failures in the results. Select configurations on one measurement set and
score the fixed choices on another; a minimum chosen on the scoring set is an
optimistic diagnostic. See [evaluation methods](docs/history/METHODOLOGY.md) and
[bundle and suite formats](docs/history/EXPERIMENT_SCOPE.md).
