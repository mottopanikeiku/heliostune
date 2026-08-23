"""Command-line workflows for replay analysis and report generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from heliostune.multisource import compare_multisource
from heliostune.replay import BenchmarkTable, compare_methods
from heliostune.schema import read_jsonl, write_jsonl
from heliostune.selection import select_parhelion

_CONSOLE = Console()


def _read_measurements(path: Path):  # type: ignore[no-untyped-def]
    with path.open(encoding="utf-8") as source:
        return read_jsonl(source)


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_sources(value: str) -> tuple[str, ...]:
    sources = tuple(source.strip() for source in value.split(","))
    if not sources or any(not source for source in sources):
        raise ValueError("sources must be a non-empty comma-separated list")
    return sources


def _compare(args: argparse.Namespace) -> int:
    summary = compare_methods(
        _read_measurements(args.input),
        source_gpu=args.source,
        target_gpu=args.target,
        max_budget=args.max_budget,
        seeds=args.seeds,
        transfer_strength=args.transfer_strength,
    )
    _write_json(summary, args.output)
    _CONSOLE.print(f"Wrote replay summary to [bold]{args.output}[/bold]")
    return 0


def _compare_multisource(args: argparse.Namespace) -> int:
    summary = compare_multisource(
        _read_measurements(args.input),
        source_gpus=_parse_sources(args.sources),
        target_gpu=args.target,
        max_budget=args.max_budget,
        seeds=args.seeds,
        k=args.k,
        temperature=args.temperature,
        transfer_strength=args.transfer_strength,
        retrieval_k=args.retrieval_k,
        retrieval_temperature=args.retrieval_temperature,
        pooled_transfer_strength=args.pooled_transfer_strength,
        primary_comparator=args.primary_comparator,
        protocol_role=args.protocol_role,
    )
    _write_json(summary, args.output)
    _CONSOLE.print(f"Wrote multi-source replay summary to [bold]{args.output}[/bold]")
    return 0


def _select_parhelion(args: argparse.Namespace) -> int:
    selection, summary = select_parhelion(
        _read_measurements(args.input),
        jobs=args.jobs,
    )
    _write_json(selection, args.output)
    _write_json(summary, args.summary_output)
    _CONSOLE.print(f"Wrote frozen Parhelion selection to [bold]{args.output}[/bold]")
    _CONSOLE.print(f"Wrote selected T4 replay to [bold]{args.summary_output}[/bold]")
    return 0


def _report(args: argparse.Namespace) -> int:
    from heliostune.report import render_report

    summary = json.loads(args.input.read_text(encoding="utf-8"))
    render_report(summary, args.output)
    _CONSOLE.print(f"Wrote standalone report to [bold]{args.output}[/bold]")
    return 0


def _demo(args: argparse.Namespace) -> int:
    from heliostune.report import render_report
    from heliostune.synthetic import synthetic_measurements

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "measurements.jsonl"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "index.html"
    measurements = synthetic_measurements(seed=args.seed)
    with data_path.open("w", encoding="utf-8") as destination:
        write_jsonl(measurements, destination)
    summary = compare_methods(
        measurements,
        source_gpu="sim-source",
        target_gpu="sim-target",
        max_budget=args.max_budget,
        seeds=args.seeds,
        transfer_strength=args.transfer_strength,
    )
    summary["data_kind"] = "synthetic"
    summary["limitations"].insert(
        0,
        "This local demo is synthetic; only published Modal artifacts support hardware claims.",
    )
    _write_json(summary, summary_path)
    render_report(summary, report_path)
    _CONSOLE.print(f"Synthetic data: [bold]{data_path}[/bold]")
    _CONSOLE.print(f"Replay summary: [bold]{summary_path}[/bold]")
    _CONSOLE.print(f"Offline report: [bold]{report_path}[/bold]")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    table = BenchmarkTable(tuple(_read_measurements(args.input)))
    view = Table(title="HeliosTune benchmark matrix")
    view.add_column("GPU")
    view.add_column("Device")
    view.add_column("Workloads", justify="right")
    view.add_column("Configs", justify="right")
    view.add_column("Records", justify="right")
    view.add_column("Failures", justify="right")
    for gpu in table.gpus:
        records = [item for item in table.measurements if item.hardware.gpu == gpu]
        view.add_row(
            gpu,
            table.hardware(gpu).device_name,
            str(len(table.workloads(gpu))),
            str(len(table.configs(gpu))),
            str(len(records)),
            str(sum(not item.usable for item in records)),
        )
    _CONSOLE.print(view)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heliostune",
        description="Transferable Bayesian autotuning for Triton LLM matmuls",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="replay tuning methods over a latency matrix")
    compare.add_argument("input", type=Path)
    compare.add_argument("--source", required=True, help="source GPU label in the dataset")
    compare.add_argument("--target", required=True, help="target GPU label in the dataset")
    compare.add_argument("--max-budget", type=int, default=8)
    compare.add_argument("--seeds", type=int, default=30)
    compare.add_argument("--transfer-strength", type=float, default=0.08)
    compare.add_argument("--output", type=Path, default=Path("summary.json"))
    compare.set_defaults(handler=_compare)

    multisource = subparsers.add_parser(
        "compare-multisource", help="replay Parhelion and baselines from a multi-GPU archive"
    )
    multisource.add_argument("input", type=Path)
    multisource.add_argument("--sources", required=True, help="comma-separated source GPU labels")
    multisource.add_argument("--target", required=True, help="target GPU label in the dataset")
    multisource.add_argument("--max-budget", type=int, default=8)
    multisource.add_argument("--seeds", type=int, default=30)
    multisource.add_argument("--k", type=int)
    multisource.add_argument("--temperature", type=float)
    multisource.add_argument("--transfer-strength", type=float)
    multisource.add_argument("--retrieval-k", type=int)
    multisource.add_argument("--retrieval-temperature", type=float)
    multisource.add_argument("--pooled-transfer-strength", type=float)
    multisource.add_argument("--primary-comparator")
    multisource.add_argument(
        "--protocol-role",
        choices=("development", "validation", "final"),
        default="development",
    )
    multisource.add_argument("--output", type=Path, default=Path("multisource-summary.json"))
    multisource.set_defaults(handler=_compare_multisource)

    selection = subparsers.add_parser(
        "select-parhelion", help="run the frozen 48-point Parhelion grid on T4"
    )
    selection.add_argument("input", type=Path)
    selection.add_argument("--jobs", type=int, default=1)
    selection.add_argument("--output", type=Path, default=Path("parhelion-selection.json"))
    selection.add_argument("--summary-output", type=Path, default=Path("t4-summary.json"))
    selection.set_defaults(handler=_select_parhelion)

    report = subparsers.add_parser("report", help="render a replay summary as standalone HTML")
    report.add_argument("input", type=Path)
    report.add_argument("--output", type=Path, default=Path("index.html"))
    report.set_defaults(handler=_report)

    demo = subparsers.add_parser("demo", help="run a deterministic local synthetic experiment")
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--max-budget", type=int, default=8)
    demo.add_argument("--seeds", type=int, default=30)
    demo.add_argument("--transfer-strength", type=float, default=0.08)
    demo.set_defaults(handler=_demo)

    inspect = subparsers.add_parser("inspect", help="show coverage and failures in a JSONL matrix")
    inspect.add_argument("input", type=Path)
    inspect.set_defaults(handler=_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
