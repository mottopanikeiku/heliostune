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

For the new H100 action expansion, `uv run python scripts/analyze_action_expansion.py` regenerates the committed summary and SVG from `results/action-expansion-raw.json` without a GPU. I cover the fixed CPU manifest and strict analysis with `tests/test_action_configs.py` and `tests/test_action_expansion_analysis.py`; `tests/test_action_expansion.py` additionally needs compatible Torch, Triton and NVIDIA hardware.

To collect on H100, I use Python 3.13 with the `dev` and `modal` extras, build the source-bound wheel with `uv run python scripts/build_modal_wheel.py`, then run `uv run modal run modal_action_expansion.py` from a clean tracked commit. The default full collection has a 27-minute remote timeout; `--pilot` is a separate four-workload run, not an input to the full summary. The collector writes each completed workload/bank to the `heliostune-day-actions` Modal volume and reuses completed units on resume. It checks the package and collector hashes, action set, plan, precision and protocol before combining units. It also records every collection session, so a resumed run is not silently described as one session.

For a fresh repeat in the same Modal account, choose an unused volume name in `modal_action_expansion.py`, commit that change, rebuild the wheel and collect into a new output path. Keep the existing measurements and checkpoint separate. The source commit and software hashes in each raw artifact identify the implementation that actually ran; my published full run used commit `318187e3b71ad77fc98293292291b8473d000357`.

## Results and claims

I leave measurements and reports backing published numbers unchanged. New analyses use new paths and identify their inputs. I distinguish new measurements from summaries or replays and report negative outcomes plainly. A score relative to an enumerated Triton reference is not a fraction of the hardware's best achievable performance. Choosing a minimum on the scoring bank is a descriptive diagnostic, not an independently evaluated policy.

The [next comparison](docs/NEXT.md) describes the scientific question. The [historical documents](docs/history/INDEX.md) retain earlier protocols and commands; they describe the checkout used for those studies.
