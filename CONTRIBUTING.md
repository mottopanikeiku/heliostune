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

## Results and claims

I leave measurements and reports backing published numbers unchanged. New analyses use new paths and identify their inputs. I distinguish new measurements from summaries or replays and report negative outcomes plainly. A score relative to an enumerated Triton reference is not a fraction of the hardware's best achievable performance. Choosing a minimum on the scoring bank is a descriptive diagnostic, not an independently evaluated policy.

The [next comparison](docs/NEXT.md) describes the scientific question. The [historical documents](docs/history/INDEX.md) retain earlier protocols and commands; they describe the checkout used for those studies.
