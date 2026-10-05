# Contributing

Start with the tuner in `src/heliostune/bandit.py`, `retrieval.py` and `multisource_engine.py`. Report generation and run-checking utilities live in `src/heliostune/tooling/`. Keep the core independent of GPU execution when testing it on CPU.

## Local changes

```bash
uv sync --locked --extra dev
nice -n 19 uv run --locked pytest -q -x tests/test_bandit.py tests/test_retrieval.py tests/test_multisource.py tests/test_tuner_results.py
nice -n 19 uv run --locked pytest -q -x
```

Run the targeted behavioral tests first. GPU collection needs suitable hardware and explicit spending approval; installing the CPU development environment does not grant either. Tests that use unavailable isolation or GPU facilities may skip, and those skips should be reported rather than presented as successful execution.

For the current comparison table, run `nice -n 19 python3 scripts/summarize_tuner_results.py`. It reads committed results and does not collect timings. Keep generated summaries deterministic and traceable to their input files.

## Results and claims

Do not alter data or reports backing published numbers. New analyses belong at new paths, identify their inputs and distinguish new measurements from summaries or replays. Report null, negative and failed outcomes plainly. A score relative to an enumerated Triton reference is not a fraction of the hardware's best achievable performance.

The next proposed experiment is in [docs/NEXT.md](docs/NEXT.md). Detailed older study, release and run-record procedures are retained in the [historical documents](docs/history/INDEX.md); their paths and commands describe the previous checkout. New code uses `heliostune.tooling`, not the old module locations.
