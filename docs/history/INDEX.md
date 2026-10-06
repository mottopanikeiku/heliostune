# Historical study documents

These documents are retained from commit [`9f29906`](https://github.com/mottopanikeiku/heliostune/tree/9f29906220f7c5e4d7662f60738fd483fcbe5fdc). They describe earlier studies and the previous package layout, not the current entry point or a new experiment. Their contents are preserved; relative paths and commands refer to that checkout. Use the original linked GitHub versions below when following repository-root links.

- [Previous README](README.md) · [original rendered page](https://github.com/mottopanikeiku/heliostune/blob/9f29906220f7c5e4d7662f60738fd483fcbe5fdc/README.md): full result history, original reproduction commands and run details.
- [Methodology](METHODOLOGY.md) · [original](https://github.com/mottopanikeiku/heliostune/blob/9f29906220f7c5e4d7662f60738fd483fcbe5fdc/METHODOLOGY.md): run-record and analysis requirements.
- [Experiment scope](EXPERIMENT_SCOPE.md) · [original](https://github.com/mottopanikeiku/heliostune/blob/9f29906220f7c5e4d7662f60738fd483fcbe5fdc/EXPERIMENT_SCOPE.md): declaration formats, supported cases and implementation state at that commit.
- [Previous contributor guide](CONTRIBUTING.md) · [original](https://github.com/mottopanikeiku/heliostune/blob/9f29906220f7c5e4d7662f60738fd483fcbe5fdc/CONTRIBUTING.md): release and campaign procedures.
- [Previous security policy](SECURITY.md) · [original](https://github.com/mottopanikeiku/heliostune/blob/9f29906220f7c5e4d7662f60738fd483fcbe5fdc/SECURITY.md): detailed isolated-worker boundaries.

The current package places report and run-checking modules under `heliostune.tooling`; historical records retain their original module identities. Published data and reports remain under [`benchmarks/`](../../benchmarks/) and [`site/`](../../site/), unchanged. For current contributor commands, see [CONTRIBUTING.md](../../CONTRIBUTING.md). For the next scientific question, see [NEXT.md](../NEXT.md).

The native-RMSNorm, fusion, and Hopper/precision publication checks still regenerate their result fields from the retained inputs. For historical publisher/source digests they read this pinned Git snapshot, rather than pretending the relocated current sources produced the original files. Those checks require a clone containing the linked commit; new generation uses current sources.
