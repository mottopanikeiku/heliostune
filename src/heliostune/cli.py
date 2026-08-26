"""Command-line workflows for replay analysis and report generation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TypeVar

from rich.console import Console
from rich.table import Table

from heliostune.artifacts import (
    read_json,
    read_measurements,
    write_bytes_atomic,
    write_json_atomic,
    write_measurements_atomic,
)
from heliostune.errors import HeliostuneError, ProtocolError
from heliostune.multisource import compare_multisource
from heliostune.replay import compare_methods
from heliostune.selection import select_parhelion
from heliostune.validation import exact_object

_CONSOLE = Console()
_ResultT = TypeVar("_ResultT")


def _strict_identifier(value: str) -> str:
    if not value or value != value.strip():
        raise argparse.ArgumentTypeError("must be nonblank with no surrounding whitespace")
    return value


def _strict_csv(value: str) -> tuple[str, ...]:
    items = tuple(value.split(","))
    if not items or any(not item or item != item.strip() for item in items):
        raise argparse.ArgumentTypeError(
            "must be a non-empty comma-separated list without whitespace"
        )
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("must not contain duplicates")
    return items


def _positive_int(value: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("must be a positive decimal integer")
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive decimal integer")
    return result


def _nonnegative_int(value: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("must be a non-negative decimal integer")
    return int(value)


def _finite_float(value: str) -> float:
    if not value or value != value.strip():
        raise argparse.ArgumentTypeError("must be a finite number without whitespace")
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be a finite number")
    return result


def _positive_float(value: str) -> float:
    result = _finite_float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _unit_float(value: str) -> float:
    result = _finite_float(value)
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("must be finite and between zero and one")
    return result


def _reject_output_collisions(*paths: Path) -> None:
    normalized = tuple(path.resolve() for path in paths)
    if len(set(normalized)) != len(normalized):
        raise ProtocolError("output paths must be distinct")
    collisions = [path for path in paths if path.exists()]
    if collisions:
        raise ProtocolError(
            "refusing to replace existing output(s): " + ", ".join(str(path) for path in collisions)
        )


def _protocol_call(label: str, function: Callable[[], _ResultT]) -> _ResultT:
    try:
        return function()
    except HeliostuneError:
        raise
    except ValueError as exc:
        raise ProtocolError(f"{label}: {exc}") from exc


def _commit_staged_files(staged: Mapping[Path, Path]) -> None:
    payloads = {destination: source.read_bytes() for destination, source in staged.items()}
    for destination, payload in payloads.items():
        write_bytes_atomic(destination, payload)


def _compare(args: argparse.Namespace) -> int:
    _reject_output_collisions(args.output)
    measurements = read_measurements(args.input)
    summary = _protocol_call(
        "replay protocol violation",
        lambda: compare_methods(
            measurements,
            source_gpu=args.source,
            target_gpu=args.target,
            max_budget=args.max_budget,
            seeds=args.seeds,
            transfer_strength=args.transfer_strength,
        ),
    )
    write_json_atomic(args.output, summary)
    _CONSOLE.print(f"Wrote replay summary to [bold]{args.output}[/bold]")
    return 0


def _compare_multisource(args: argparse.Namespace) -> int:
    _reject_output_collisions(args.output)
    measurements = read_measurements(args.input)
    release_provenance = (
        None
        if args.release_provenance is None
        else exact_object(read_json(args.release_provenance), context="release provenance")
    )
    summary = _protocol_call(
        "multi-source replay protocol violation",
        lambda: compare_multisource(
            measurements,
            source_gpus=args.sources,
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
            release_provenance=release_provenance,
        ),
    )
    write_json_atomic(args.output, summary)
    _CONSOLE.print(f"Wrote multi-source replay summary to [bold]{args.output}[/bold]")
    return 0


def _select_parhelion(args: argparse.Namespace) -> int:
    _reject_output_collisions(args.output, args.summary_output)
    measurements = read_measurements(args.input)
    selection, summary = _protocol_call(
        "selection protocol violation",
        lambda: select_parhelion(measurements, jobs=args.jobs),
    )
    with tempfile.TemporaryDirectory(prefix="heliostune-select-") as temporary:
        root = Path(temporary)
        staged_selection = root / "selection.json"
        staged_summary = root / "summary.json"
        write_json_atomic(staged_selection, selection)
        write_json_atomic(staged_summary, summary)
        _commit_staged_files(
            {
                args.output: staged_selection,
                args.summary_output: staged_summary,
            }
        )
    _CONSOLE.print(f"Wrote frozen Parhelion selection to [bold]{args.output}[/bold]")
    _CONSOLE.print(f"Wrote selected T4 replay to [bold]{args.summary_output}[/bold]")
    return 0


def _select_v3(args: argparse.Namespace) -> int:
    from heliostune.collection import sha256_file
    from heliostune.protocol import (
        load_v3_protocol,
        require_v3_runtime,
        runtime_manifest,
    )
    from heliostune.v3_engine import prepare_v3, select_v3_parameters

    _reject_output_collisions(args.output, args.summary_output)
    protocol = load_v3_protocol(args.protocol)
    require_v3_runtime(protocol)
    config_manifest = exact_object(
        read_json(args.config_manifest),
        context="v3 retained config manifest",
    )
    retained = config_manifest.get("retained_config_keys")
    official = config_manifest.get("retained_official_config_keys")
    if not isinstance(retained, list) or not isinstance(official, list):
        raise ProtocolError("v3 config manifest lacks retained/official key lists")
    measurements = read_measurements(args.input)
    prepared = _protocol_call(
        "v3 preparation violation",
        lambda: prepare_v3(
            protocol,
            measurements,
            source_gpus=("L4", "A10"),
            target_gpu="A100-80GB",
            retained_config_keys=tuple(str(key) for key in retained),
            official_config_keys=tuple(str(key) for key in official),
            seeds=tuple(range(30)),
        ),
    )
    selection = _protocol_call(
        "v3 selection violation",
        lambda: select_v3_parameters(prepared),
    )
    summary = {
        "schema_version": 1,
        "study_id": "parhelion-v3-a100-selection-summary",
        "selected": selection["selected"],
        "jobs": args.jobs,
        "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "protocol": {
            "path": str(args.protocol),
            "sha256": sha256_file(args.protocol),
        },
        "config_manifest": {
            "path": str(args.config_manifest),
            "sha256": sha256_file(args.config_manifest),
        },
        "runtime": runtime_manifest(),
    }
    selection["jobs"] = args.jobs
    selection["runtime"] = runtime_manifest()
    with tempfile.TemporaryDirectory(prefix="heliostune-select-v3-") as temporary:
        root = Path(temporary)
        staged_selection = root / "selection.json"
        staged_summary = root / "summary.json"
        write_json_atomic(staged_selection, selection)
        write_json_atomic(staged_summary, summary)
        _commit_staged_files(
            {
                args.output: staged_selection,
                args.summary_output: staged_summary,
            }
        )
    _CONSOLE.print(f"Wrote frozen A100 v3 selection to [bold]{args.output}[/bold]")
    _CONSOLE.print(f"Wrote A100 v3 summary to [bold]{args.summary_output}[/bold]")
    return 0


def _report(args: argparse.Namespace) -> int:
    from heliostune.report import render_report

    _reject_output_collisions(args.output)
    summary = exact_object(read_json(args.input), context="report summary")
    render_report(summary, args.output)
    _CONSOLE.print(f"Wrote standalone report to [bold]{args.output}[/bold]")
    return 0


def _demo(args: argparse.Namespace) -> int:
    from heliostune.report import render_report
    from heliostune.synthetic import synthetic_measurements

    data_path = args.output_dir / "measurements.jsonl"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "index.html"
    _reject_output_collisions(data_path, summary_path, report_path)
    measurements = synthetic_measurements(seed=args.seed)
    summary = _protocol_call(
        "synthetic replay protocol violation",
        lambda: compare_methods(
            measurements,
            source_gpu="sim-source",
            target_gpu="sim-target",
            max_budget=args.max_budget,
            seeds=args.seeds,
            transfer_strength=args.transfer_strength,
        ),
    )
    summary["data_kind"] = "synthetic"
    summary["limitations"].insert(
        0,
        "This local demo is synthetic; only published Modal artifacts support hardware claims.",
    )
    with tempfile.TemporaryDirectory(prefix="heliostune-demo-") as temporary:
        root = Path(temporary)
        staged_data = root / "measurements.jsonl"
        staged_summary = root / "summary.json"
        staged_report = root / "index.html"
        write_measurements_atomic(staged_data, measurements)
        write_json_atomic(staged_summary, summary)
        render_report(summary, staged_report)
        _commit_staged_files(
            {
                data_path: staged_data,
                summary_path: staged_summary,
                report_path: staged_report,
            }
        )
    _CONSOLE.print(f"Synthetic data: [bold]{data_path}[/bold]")
    _CONSOLE.print(f"Replay summary: [bold]{summary_path}[/bold]")
    _CONSOLE.print(f"Offline report: [bold]{report_path}[/bold]")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    measurements = read_measurements(args.input)
    view = Table(title="HeliosTune benchmark matrix")
    view.add_column("GPU")
    view.add_column("Device")
    view.add_column("Workloads", justify="right")
    view.add_column("Configs", justify="right")
    view.add_column("Records", justify="right")
    view.add_column("Failures", justify="right")
    gpus = sorted({measurement.hardware.gpu for measurement in measurements})
    for gpu in gpus:
        records = [item for item in measurements if item.hardware.gpu == gpu]
        profiles = {item.hardware for item in records}
        if len(profiles) != 1:
            raise ProtocolError(f"inconsistent hardware profiles for {gpu!r}")
        profile = next(iter(profiles))
        view.add_row(
            gpu,
            profile.device_name,
            str(len({item.workload.key for item in records})),
            str(len({item.config.key for item in records})),
            str(len(records)),
            str(sum(not item.usable for item in records)),
        )
    _CONSOLE.print(view)
    return 0


def _verify_catalog(args: argparse.Namespace) -> int:
    from heliostune.catalog import verify_research_catalog

    facts = verify_research_catalog(args.catalog)
    _CONSOLE.print(
        "Verified research catalog: "
        f"[bold]{facts['measurement_rows']}[/bold] measurement rows, "
        f"[bold]{facts['json_artifacts']}[/bold] JSON artifacts, "
        f"[bold]{facts['html_reports']}[/bold] HTML reports, "
        f"[bold]{facts['file_artifacts']}[/bold] other files, "
        f"[bold]{facts['aliases']}[/bold] historical aliases"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heliostune",
        description="Transferable Bayesian autotuning for Triton LLM matmuls",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('heliostune')}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="replay tuning methods over a latency matrix")
    compare.add_argument("input", type=Path)
    compare.add_argument("--source", required=True, type=_strict_identifier)
    compare.add_argument("--target", required=True, type=_strict_identifier)
    compare.add_argument("--max-budget", type=_positive_int, default=8)
    compare.add_argument("--seeds", type=_positive_int, default=30)
    compare.add_argument("--transfer-strength", type=_unit_float, default=0.08)
    compare.add_argument("--output", type=Path, default=Path("summary.json"))
    compare.set_defaults(handler=_compare)

    multisource = subparsers.add_parser(
        "compare-multisource",
        help="replay Parhelion and baselines from a multi-GPU archive",
    )
    multisource.add_argument("input", type=Path)
    multisource.add_argument("--sources", required=True, type=_strict_csv)
    multisource.add_argument("--target", required=True, type=_strict_identifier)
    multisource.add_argument("--max-budget", type=_positive_int, default=8)
    multisource.add_argument("--seeds", type=_positive_int, default=30)
    multisource.add_argument("--k", type=_positive_int)
    multisource.add_argument("--temperature", type=_positive_float)
    multisource.add_argument("--transfer-strength", type=_unit_float)
    multisource.add_argument("--retrieval-k", type=_positive_int)
    multisource.add_argument("--retrieval-temperature", type=_positive_float)
    multisource.add_argument("--pooled-transfer-strength", type=_unit_float)
    multisource.add_argument("--primary-comparator", type=_strict_identifier)
    multisource.add_argument(
        "--release-provenance",
        type=str,
        default=None,
        help="JSON file whose object is embedded as release_provenance",
    )
    multisource.add_argument(
        "--protocol-role",
        choices=("development", "validation", "final"),
        default="development",
    )
    multisource.add_argument("--output", type=Path, default=Path("multisource-summary.json"))
    multisource.set_defaults(handler=_compare_multisource)

    selection = subparsers.add_parser(
        "select-parhelion",
        help="run the frozen method-local Parhelion grids on T4",
    )
    selection.add_argument("input", type=Path)
    selection.add_argument("--jobs", type=_positive_int, default=1)
    selection.add_argument("--output", type=Path, default=Path("parhelion-selection.json"))
    selection.add_argument("--summary-output", type=Path, default=Path("t4-summary.json"))
    selection.set_defaults(handler=_select_parhelion)

    selection_v3 = subparsers.add_parser(
        "select-v3",
        help="run frozen method-local Parhelion v3 grids on A100-80GB",
    )
    selection_v3.add_argument("input", type=Path)
    selection_v3.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmarks/parhelion-v3-development-protocol.json"),
    )
    selection_v3.add_argument("--config-manifest", type=Path, required=True)
    selection_v3.add_argument("--jobs", type=_positive_int, default=1)
    selection_v3.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/parhelion-v3-a100-selection.json"),
    )
    selection_v3.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/parhelion-v3-a100-summary.json"),
    )
    selection_v3.set_defaults(handler=_select_v3)

    report = subparsers.add_parser("report", help="render a replay summary as standalone HTML")
    report.add_argument("input", type=Path)
    report.add_argument("--output", type=Path, default=Path("index.html"))
    report.set_defaults(handler=_report)

    demo = subparsers.add_parser("demo", help="run a deterministic local synthetic experiment")
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed", type=_nonnegative_int, default=7)
    demo.add_argument("--max-budget", type=_positive_int, default=8)
    demo.add_argument("--seeds", type=_positive_int, default=30)
    demo.add_argument("--transfer-strength", type=_unit_float, default=0.08)
    demo.set_defaults(handler=_demo)

    inspect = subparsers.add_parser("inspect", help="show coverage and failures in a JSONL matrix")
    inspect.add_argument("input", type=Path)
    inspect.set_defaults(handler=_inspect)

    verify_catalog = subparsers.add_parser(
        "verify-catalog",
        help="verify every research artifact digest, count, alias, and frozen v2 estimate",
    )
    verify_catalog.add_argument(
        "catalog",
        type=Path,
        nargs="?",
        default=Path("benchmarks/research-artifact-manifest.json"),
    )
    verify_catalog.set_defaults(handler=_verify_catalog)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (HeliostuneError, OSError, json.JSONDecodeError) as exc:
        print(f"heliostune: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
