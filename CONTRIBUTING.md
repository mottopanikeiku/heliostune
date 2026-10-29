# Contributing

I keep the tuner in `src/heliostune/bandit.py`, `retrieval.py` and `multisource_engine.py`, independent of GPU execution for CPU tests. Offline analysis and report utilities live in `src/heliostune/tooling/`.

## Local changes

```bash
uv sync --locked --extra dev
nice -n 19 uv run --locked pytest -q -x tests/test_bandit.py tests/test_retrieval.py tests/test_multisource.py tests/test_tuner_results.py tests/test_matmul_audit.py
nice -n 19 uv run --locked pytest -q -x
```

I run targeted behavioral tests before the full suite. The CPU development environment does not install GPU dependencies. Tests that need unavailable Linux isolation or GPU facilities may skip; a skip is not a successful execution of that path.

I keep resolved Python versions in `uv.lock` and workflow actions pinned to commit SHAs. I update CodeQL's initialization and analysis steps together because different action versions cannot share their analysis configuration.

For the four-GPU overview and detailed H100 action-set audit, I run `nice -n 19 uv run --locked python scripts/audit_matmul_action_set.py`. It reads committed measurements and writes `results/action-set-audit.json` and two SVGs. `nice -n 19 python3 scripts/summarize_tuner_results.py` regenerates the policy comparison table. Neither command collects timings. The audit tests check computation on small hand-calculated matrices, separate-device aggregation and reproduction of committed outputs.

For the new H100 action expansion, `uv run python scripts/analyze_action_expansion.py` regenerates the committed summary and SVG from `results/action-expansion-raw.json` without a GPU. I cover the fixed CPU manifest and strict analysis with `tests/test_action_configs.py` and `tests/test_action_expansion_analysis.py`; `tests/test_readme_claims.py` checks the numbers quoted in the README and `docs/NEXT.md` against the committed summary. `tests/test_action_expansion.py` additionally needs compatible Torch, Triton and NVIDIA hardware.

`uv run python scripts/diagnose_kernel_bank.py` re-reads every stored action matrix and writes `results/kernel-bank-diagnosis.json`. `uv run python scripts/build_site.py` then renders the Pages landing page, `site/index.html`, from that file and the expansion summary. Both accept `--check`, which compares the committed file byte for byte and which CI runs. Neither collects timings. I keep each archive file and hardware profile as a separate view because several archives repeat the same measurements. `tests/test_bank_diagnosis.py`, `tests/test_kernel_dossier.py` and `tests/test_site.py` check hand-calculated aggregation, regret, coverage and formulas, byte-exact reproduction of both outputs, rounding of stated bounds, and every link on the page.

I do not edit the historical `site/*.html` reports; historical manifests bind their bytes. The former landing page is preserved unchanged at `site/transfer.html`, and `historical_artifact_path` resolves its frozen `site/index.html` bindings there.

Every collection has a run ID. On a Modal H100, I use Python 3.13 with the `dev` and `modal` extras, build the source-bound wheel with `uv run python scripts/build_modal_wheel.py`, then run `uv run modal run modal_action_expansion.py --run-id NAME` from a clean tracked commit. On any other CUDA machine, I install the `gpu` extra and run `uv run python scripts/collect_action_expansion.py --run-id NAME --gpu LABEL`, for example `--gpu A100-80GB`. Both use the same measurement code, `measure_unit` and `collect_on_device` in `src/heliostune/action_expansion.py`. The scheduling, run identity and checkpoint rules are in `src/heliostune/action_collection.py`. The full Modal collection has a 27-minute remote timeout. `--pilot` is a separate four-workload run, not an input to the full summary.

Each run ID and collection kind gets its own checkpoint, `NAME/full/checkpoint.json` or `NAME/pilot/checkpoint.json`, under `/data/runs` on the `heliostune-day-actions` Modal volume or under `~/.cache/heliostune/action-runs` locally. The collector saves after every completed workload/bank unit and reuses completed units on resume. Before it continues, it checks that the run ID, plan commit, exact hardware profile, package and collector hashes, precision settings, protocol and action set all match the checkpoint. Any difference stops the run and needs a new run ID, so two experiments cannot end up in one artifact. It also records every collection session, so a resumed run is not described as one session.

For a fresh repeat I choose a new run ID; no code change is needed. `scripts/collect_action_expansion.py --dry-run` writes the run plan (identity, unit order, seeds and checkpoint path) without importing Torch, so I can check a run on CPU first. A GPU label that has no expectation in `src/heliostune/hardware.py` needs `--allow-unlisted-gpu`, and the run is then bound to the exact profile it first observes. Raw artifacts go to `results/action-expansion-runs/NAME.json` (or `NAME-pilot.json`) and never overwrite an existing file; `scripts/analyze_action_expansion.py --input ... --output ... --figure ...` summarizes a full run. I have not run the expanded bank on a GPU older than Hopper. If the persistent tensor-descriptor actions fail there, they are recorded as failed rows and cannot be selected. The source commit and software hashes in each raw artifact identify the implementation that actually ran. My published full run used commit `318187e3b71ad77fc98293292291b8473d000357`, before run IDs existed; that collector wrote its checkpoint to `/data/full-checkpoint.json`, which new runs never read.

## Results and claims

I leave measurements and reports backing published numbers unchanged. New analyses use new paths and identify their inputs. I distinguish new measurements from summaries or replays and report negative outcomes plainly. A score relative to an enumerated Triton reference is not a fraction of the hardware's best achievable performance. Choosing a minimum on the scoring bank is a descriptive diagnostic, not an independently evaluated policy.

The [next comparison](docs/NEXT.md) describes the scientific question. The [historical documents](docs/history/INDEX.md) retain earlier protocols and commands; they describe the checkout used for those studies.
