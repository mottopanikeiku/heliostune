# Earlier studies

The [evaluation methods](METHODOLOGY.md) explain retrieval, measurement splits,
query budgets and uncertainty. The [format reference](EXPERIMENT_SCOPE.md)
documents the implemented declaration, verification and replay interfaces.
Current setup and collection commands are in [Contributing](../../CONTRIBUTING.md).

## Reports and results

| Study | Report | Result |
|---|---|---|
| L4/A10 single-source transfer | [Report](../../site/v1.html) | [L4 → A10](../../benchmarks/results/l4-to-a10.json), [A10 → L4](../../benchmarks/results/a10-to-l4.json) |
| H100 multi-source transfer | [Report](../../site/transfer.html) | [Summary](../../benchmarks/results/parhelion-h100-final.json) |
| H100 retrieval and posterior ablations | [Report](../../site/parhelion-v2-addendum.html) | [Summary](../../benchmarks/results/parhelion-v2-addendum.json) |
| H200 engineering comparison | [Report](../../site/parhelion-v3-engineering.html) | [Summary](../../benchmarks/results/parhelion-v3-h200-engineering.json) |
| Hopper kernel comparison | [Report](../../site/hopper-h100-engineering.html) | [Summary](../../benchmarks/results/hopper-h100-engineering-summary-v2.json) |
| H100 precision probe | [Report](../../site/h100-precision-probe.html) | [Summary](../../benchmarks/results/h100-precision-probe-summary.json) |
| Remote fusion templates | [Report](../../site/fusion-remote-exploratory.html) | [Summary](../../benchmarks/results/fusion-remote-exploratory-summary.json) |
| Native residual RMSNorm | [Report](../../site/native-rmsnorm-h100.html) | [Summary](../../benchmarks/results/native-rmsnorm-h100-summary.json) |

Cold Thompson outperformed the retrieval-anchored tuner on the original H100
comparison. The selected posterior-transfer strengths were zero. The
[tuner comparison](../../results/tuner-comparison.md) contains the exact curves
and input keys.

The H200 study's
[failed pilot](../../benchmarks/parhelion-v3-validation-failure.json) returned no
timings. A separate engineering run followed a change to the retry rule and
used mixed PCIe/SXM A100 selection data. Its result remains separate from that
failed study and from the original H100 comparison.

## Preserving and checking reports

The [artifact catalog](../../benchmarks/research-artifact-manifest.json) and
study manifests bind the historical measurements and reports by digest. Keep
those files unchanged and write new analyses to new paths. Historical filenames
and JSON keys retain their original spelling.

The old landing page is preserved at `site/transfer.html`; the catalog resolver
maps historical `site/index.html` references to it. The current landing page is
generated separately.

The native RMSNorm, remote fusion, and Hopper/precision checkers reconstruct
their results from stored inputs. Their historical publisher-source digests
come from source snapshot
[`9f29906220f7c5e4d7662f60738fd483fcbe5fdc`](https://github.com/mottopanikeiku/quartet/tree/9f29906220f7c5e4d7662f60738fd483fcbe5fdc),
before the report modules moved into `heliostune.tooling`. Keep that commit
available in the clone:

```bash
nice -n 10 uv run --locked python scripts/publish_native_rmsnorm_h100.py --check
nice -n 10 uv run --locked python scripts/publish_fusion_remote_results.py --check
nice -n 10 uv run --locked python scripts/publish_hopper_engineering_result.py --check
```

These checks do not collect GPU measurements. New generation uses current
sources. The latest action-expansion protocol is
[documented separately](../action-expansion.md).

## Release archives

Release assets include a wheel, source distribution, Git bundle and
`SHA256SUMS`. Keep the bundle with the checksums: it preserves released source
and semantic tag history independently of the current branch. Released assets
remain unchanged; fixes belong in a new version.
