"""Command-line workflows for replay analysis and report generation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import TypeVar

import zstandard
from rich.console import Console
from rich.table import Table

from heliostune.artifacts import (
    read_json,
    read_measurements,
    write_bytes_atomic,
    write_json_atomic,
    write_measurements_atomic,
)
from heliostune.errors import ArtifactError, HeliostuneError, ProtocolError, SchemaError
from heliostune.multisource import compare_multisource
from heliostune.multisource_engine import (
    ReleaseProvenance,
    validate_release_provenance,
)
from heliostune.replay import compare_methods
from heliostune.schema import Measurement, read_jsonl
from heliostune.selection import select_parhelion
from heliostune.validation import exact_object, nonblank_string

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


_MAX_RELEASE_ARCHIVE_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_ARCHIVE_READ_CHUNK_BYTES = 1024 * 1024
_ZSTD_MAX_BLOCK_BYTES = 128 * 1024
_ZSTD_SKIPPABLE_MAGIC_MIN = 0x184D2A50
_ZSTD_SKIPPABLE_MAGIC_MAX = 0x184D2A5F


@dataclass(frozen=True, slots=True)
class _ArchiveSnapshot:
    compressed: bytes
    canonical: bytes
    transport_sha256: str
    canonical_sha256: str


def _incomplete_zstd_frame(source: Path) -> ArtifactError:
    return ArtifactError(
        f"cannot authenticate release archive {source}: truncated or incomplete zstd frame"
    )


def _validate_zstd_frame_boundaries(compressed: bytes, *, source: Path) -> None:
    """Prove frame completeness without the second inflation that decompressobj would cause."""
    if not compressed:
        raise ArtifactError(f"cannot authenticate release archive {source}: empty zstd input")

    view = memoryview(compressed)
    offset = 0
    while offset < len(view):
        remaining = len(view) - offset
        if remaining < len(zstandard.FRAME_HEADER):
            tail = bytes(view[offset:])
            skippable_prefix = b"\x50\x2a\x4d".startswith(tail)
            if zstandard.FRAME_HEADER.startswith(tail) or skippable_prefix:
                raise _incomplete_zstd_frame(source)
            raise ArtifactError(
                f"cannot authenticate release archive {source}: "
                f"trailing non-zstd data at byte offset {offset}"
            )

        magic = int.from_bytes(view[offset : offset + 4], "little")
        if _ZSTD_SKIPPABLE_MAGIC_MIN <= magic <= _ZSTD_SKIPPABLE_MAGIC_MAX:
            if remaining < 8:
                raise _incomplete_zstd_frame(source)
            payload_size = int.from_bytes(view[offset + 4 : offset + 8], "little")
            frame_end = offset + 8 + payload_size
            if frame_end > len(view):
                raise _incomplete_zstd_frame(source)
            offset = frame_end
            continue

        if view[offset : offset + 4] != zstandard.FRAME_HEADER:
            raise ArtifactError(
                f"cannot authenticate release archive {source}: "
                f"trailing non-zstd data at byte offset {offset}"
            )

        try:
            header_size = zstandard.frame_header_size(view[offset:])
        except zstandard.ZstdError as exc:
            raise ArtifactError(
                f"cannot authenticate release archive {source}: "
                f"malformed or incomplete zstd frame: {exc}"
            ) from exc

        descriptor = view[offset + 4]
        cursor = offset + header_size
        while True:
            if len(view) - cursor < 3:
                raise _incomplete_zstd_frame(source)
            block_header = int.from_bytes(view[cursor : cursor + 3], "little")
            cursor += 3
            last_block = bool(block_header & 1)
            block_type = (block_header >> 1) & 0b11
            block_size = block_header >> 3
            if block_type == 0b11:
                raise ArtifactError(
                    f"cannot authenticate release archive {source}: "
                    f"reserved zstd block type at byte offset {cursor - 3}"
                )
            if block_size > _ZSTD_MAX_BLOCK_BYTES:
                raise ArtifactError(
                    f"cannot authenticate release archive {source}: "
                    f"oversized zstd block at byte offset {cursor - 3}"
                )
            payload_size = 1 if block_type == 0b01 else block_size
            if len(view) - cursor < payload_size:
                raise _incomplete_zstd_frame(source)
            cursor += payload_size
            if last_block:
                break

        if descriptor & 0b100:
            if len(view) - cursor < 4:
                raise _incomplete_zstd_frame(source)
            cursor += 4
        offset = cursor


def _decompress_archive(
    compressed: bytes,
    *,
    source_path: Path,
    maximum_uncompressed_bytes: int,
) -> bytes:
    _validate_zstd_frame_boundaries(compressed, source=source_path)
    canonical = bytearray()
    try:
        # python-zstandard 0.25.0 forwards this byte count directly to
        # ZSTD_DCtx_setMaxWindowSize, bounding the decoder's history allocation too.
        with zstandard.ZstdDecompressor(max_window_size=maximum_uncompressed_bytes).stream_reader(
            compressed,
            read_across_frames=True,
        ) as decoded:
            while True:
                allowance = maximum_uncompressed_bytes - len(canonical)
                block = decoded.read(min(_ARCHIVE_READ_CHUNK_BYTES, allowance + 1))
                if not block:
                    break
                canonical.extend(block)
                if len(canonical) > maximum_uncompressed_bytes:
                    raise ArtifactError(
                        f"release archive {source_path} exceeds maximum uncompressed size "
                        f"of {maximum_uncompressed_bytes} bytes"
                    )
    except ArtifactError:
        raise
    except (OSError, zstandard.ZstdError) as exc:
        raise ArtifactError(f"cannot authenticate release archive {source_path}: {exc}") from exc
    return bytes(canonical)


def _read_archive_snapshot(path: Path) -> _ArchiveSnapshot:
    """Read and fully decode the immutable replay snapshot exactly once."""
    if not path.name.endswith(".zst"):
        raise SchemaError("release replay input must be a zstd-compressed archive")
    try:
        compressed = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read release archive {path}: {exc}") from exc
    canonical = _decompress_archive(
        compressed,
        source_path=path,
        maximum_uncompressed_bytes=_MAX_RELEASE_ARCHIVE_UNCOMPRESSED_BYTES,
    )
    return _ArchiveSnapshot(
        compressed=compressed,
        canonical=canonical,
        transport_sha256=hashlib.sha256(compressed).hexdigest(),
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _measurements_from_snapshot(
    snapshot: _ArchiveSnapshot,
    *,
    source_path: Path,
) -> list[Measurement]:
    try:
        canonical_text = snapshot.canonical.decode("utf-8")
    except UnicodeError as exc:
        raise ArtifactError(f"cannot decode measurement artifact {source_path}: {exc}") from exc
    return read_jsonl(io.StringIO(canonical_text, newline=""), source_name=source_path)


def _repository_manifest(
    provenance_path: Path,
    relative_manifest: str,
) -> tuple[Path, Path]:
    provenance = provenance_path.resolve()
    roots = [ancestor.parent for ancestor in provenance.parents if ancestor.name == "benchmarks"]
    roots.append(Path.cwd().resolve())
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        benchmarks = (root / "benchmarks").resolve()
        manifest = (root / relative_manifest).resolve()
        try:
            manifest.relative_to(benchmarks)
        except ValueError:
            continue
        if manifest.is_file():
            return root, manifest
    raise ArtifactError(
        "release provenance manifest does not exist under repository benchmarks: "
        f"{relative_manifest}"
    )


def _manifest_string(
    manifest: Mapping[str, object],
    path: tuple[str, ...],
) -> str:
    current = manifest
    for position, key in enumerate(path):
        context = f"post-run manifest {'.'.join(path[: position + 1])}"
        if key not in current:
            raise SchemaError(f"{context} is required for release authentication")
        value = current[key]
        if position == len(path) - 1:
            return nonblank_string(value, context=context)
        current = exact_object(value, context=context)
    raise AssertionError("manifest path must not be empty")


def _manifest_archive_path(
    value: str,
    *,
    repository: Path,
    manifest_path: Path,
) -> Path:
    relative = PurePosixPath(value)
    if (
        not relative.parts
        or "\\" in value
        or relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise SchemaError("post-run manifest archive path must be normalized and non-escaping")
    base = (
        repository if relative.parts and relative.parts[0] == "benchmarks" else manifest_path.parent
    )
    resolved = (base / value).resolve()
    try:
        resolved.relative_to((repository / "benchmarks").resolve())
    except ValueError as exc:
        raise SchemaError(
            "post-run manifest archive path must remain under repository benchmarks"
        ) from exc
    return resolved


def _verify_release_provenance(
    input_path: Path,
    provenance_path: Path,
    provenance: ReleaseProvenance,
    snapshot: _ArchiveSnapshot,
) -> ReleaseProvenance:
    repository, manifest_path = _repository_manifest(
        provenance_path,
        provenance.post_run_manifest_path,
    )
    manifest = exact_object(read_json(manifest_path), context="post-run manifest")

    fact_bindings = (
        (
            ("commits", "algorithm_and_development_protocol"),
            provenance.algorithm_commit,
            "algorithm commit",
        ),
        (
            ("commits", "executable_h100_freeze"),
            provenance.freeze_commit,
            "freeze commit",
        ),
        (("freeze", "sha256"), provenance.freeze_sha256, "freeze SHA-256"),
        (
            ("runs", "h100_final_sole_invocation"),
            provenance.sole_h100_run,
            "sole H100 run",
        ),
        (
            ("data", "h100_raw", "uncompressed_sha256"),
            provenance.raw_h100_sha256,
            "raw H100 SHA-256",
        ),
    )
    for path, expected, label in fact_bindings:
        if _manifest_string(manifest, path) != expected:
            raise ArtifactError(f"release {label} does not match the post-run manifest")

    archive_path_value = _manifest_string(
        manifest,
        ("data", "four_gpu_replay_archive", "published_path"),
    )
    bound_archive = _manifest_archive_path(
        archive_path_value,
        repository=repository,
        manifest_path=manifest_path,
    )
    if bound_archive != input_path.resolve():
        raise ArtifactError(
            "post-run manifest archive path does not bind the supplied replay input"
        )

    manifest_transport_sha256 = _manifest_string(
        manifest,
        ("data", "four_gpu_replay_archive", "compressed_sha256"),
    )
    manifest_canonical_sha256 = _manifest_string(
        manifest,
        ("data", "four_gpu_replay_archive", "uncompressed_sha256"),
    )
    if manifest_transport_sha256 != snapshot.transport_sha256:
        raise ArtifactError(
            "supplied replay input compressed SHA-256 does not match the post-run manifest"
        )
    if snapshot.canonical_sha256 != provenance.final_archive_sha256:
        raise ArtifactError(
            "supplied replay input canonical SHA-256 does not match release provenance"
        )
    if manifest_canonical_sha256 != provenance.final_archive_sha256:
        raise ArtifactError("release final archive SHA-256 does not match the post-run manifest")
    return provenance


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
    release_provenance = None
    archive_snapshot = None
    if args.release_provenance is not None:
        provenance_path = Path(args.release_provenance)
        release_provenance = validate_release_provenance(
            exact_object(read_json(provenance_path), context="release provenance")
        )
        archive_snapshot = _read_archive_snapshot(args.input)
        release_provenance = _verify_release_provenance(
            args.input,
            provenance_path,
            release_provenance,
            archive_snapshot,
        )
    measurements = (
        _measurements_from_snapshot(archive_snapshot, source_path=args.input)
        if archive_snapshot is not None
        else read_measurements(args.input)
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
    from heliostune.engineering_report import (
        ENGINEERING_STUDY_IDS,
        render_engineering_report,
    )
    from heliostune.report import render_report

    _reject_output_collisions(args.output)
    summary = exact_object(read_json(args.input), context="report summary")
    study_id = summary.get("study_id")
    if type(study_id) is str and study_id in ENGINEERING_STUDY_IDS:
        render_engineering_report(summary, args.output)
    else:
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
        f"[bold]{facts['compressed_raw_artifacts']}[/bold] compressed raw artifacts, "
        f"[bold]{facts['aliases']}[/bold] historical aliases"
    )
    return 0


def _display_value(value: str | int | Path) -> str:
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)[1:-1]


def _print_facts(heading: str, facts: Sequence[tuple[str, str | int | Path]]) -> None:
    lines = [heading]
    lines.extend(f"{name}: {_display_value(value)}" for name, value in facts)
    _CONSOLE.print(
        "\n".join(lines),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def _print_structural_verification(
    kind: str, facts: Sequence[tuple[str, str | int | Path]]
) -> None:
    _print_facts(f"{kind} structurally verified", facts)


def _list_scope(_args: argparse.Namespace) -> int:
    from heliostune.scope import (
        DOMAIN_VOCABULARY,
        DTYPE_VOCABULARY,
        EXECUTABLE_TEMPLATE_IDS,
    )

    _print_facts(
        "Scope vocabulary and execution status",
        (
            ("dtype_schema_vocabulary", ",".join(DTYPE_VOCABULARY)),
            ("domain_schema_vocabulary", ",".join(DOMAIN_VOCABULARY)),
            ("frozen_executable_suite_templates", ",".join(EXECUTABLE_TEMPLATE_IDS)),
            ("template_input_storage_dtypes", "fp16,bf16"),
            ("template_domains", "fused_mlp,rmsnorm_residual"),
            (
                "suite_template_status",
                "available only for fp16/bf16 input/storage in fused_mlp,rmsnorm_residual",
            ),
            (
                "generic_local_runtime_backend",
                "implemented for the two frozen reference templates",
            ),
            (
                "generic_local_runtime_requirements",
                "torch==2.8.0,cuda,compute_capability>=8.0,native_bf16,inductor",
            ),
            (
                "generic_local_runtime_gpu_validation",
                "validated remotely on H100 for both frozen templates",
            ),
            (
                "native_local_runtime_backend",
                "implemented for residual_rmsnorm_triton.v1",
            ),
            (
                "native_local_runtime_gpu_evidence",
                "unobserved",
            ),
            (
                "generic_remote_runtime_backend",
                "implemented for the two frozen reference templates via Modal receipt",
            ),
            (
                "generic_remote_runtime_gpu_validation",
                "two completed exploratory H100 receipts",
            ),
            (
                "generic_remote_runtime_evidence",
                "benchmarks/results/fusion-remote-exploratory-summary.json",
            ),
            ("generic_remote_receipt_schema", "heliostune.remote-receipt/1"),
            ("generic_remote_provider_physical_attempts", "not_observable"),
            (
                "limitation",
                "Schema verification alone does not claim execution, correctness, or performance. "
                "The two generic frozen templates have exploratory H100 receipts only; native "
                "local runtime GPU evidence is unobserved; plugin capability declarations remain "
                "unprobed and Modal provider restarts remain unobservable.",
            ),
        ),
    )
    return 0


def _verify_plugin(args: argparse.Namespace) -> int:
    from heliostune.scope import verify_plugin

    verified = verify_plugin(args.path)
    plugin = verified.plugin
    arms = [arm for suite in verified.suites for arm in suite.suite.arms]
    local_states = [arm.local_capability.state for arm in arms]
    remote_states = [arm.remote_capability.state for arm in arms]
    _print_structural_verification(
        "Plugin",
        (
            ("path", verified.path),
            ("plugin", plugin.plugin_id),
            ("version", plugin.version),
            ("domains", len(plugin.domains)),
            ("arms", len(plugin.arm_ids)),
            ("suites", len(verified.suites)),
            ("local.unprobed", local_states.count("unprobed")),
            ("local.available", local_states.count("available")),
            ("local.unavailable", local_states.count("unavailable")),
            ("remote.unprobed", remote_states.count("unprobed")),
            ("remote.available", remote_states.count("available")),
            ("remote.unavailable", remote_states.count("unavailable")),
            (
                "limitation",
                "Structural verification does not validate executability, "
                "correctness, or performance.",
            ),
        ),
    )
    return 0


def _verify_suite(args: argparse.Namespace) -> int:
    from heliostune.scope import verify_suite

    verified = verify_suite(args.path)
    suite = verified.suite
    _print_structural_verification(
        "Suite",
        (
            ("path", verified.path),
            ("suite", suite.suite_id),
            ("template", suite.template_id),
            ("revision", suite.revision),
            ("domain", suite.domain),
            ("cases", len(suite.cases)),
            ("arms", len(suite.arms)),
            ("cells", len(suite.expected_cells)),
            ("numeric_contracts", len(suite.numeric_contracts)),
            (
                "limitation",
                "Correctness passage and execution are not observed; no "
                "performance validation is claimed.",
            ),
        ),
    )
    return 0


def _verify_protocol(args: argparse.Namespace) -> int:
    from heliostune.methodology import verify_protocol_v1

    verified = verify_protocol_v1(args.path)
    protocol = verified.protocol
    _print_structural_verification(
        "Protocol",
        (
            ("path", verified.path),
            ("schema", protocol.schema),
            ("study", protocol.study_id),
            ("revision", protocol.revision),
            ("evidence_class", protocol.evidence_class),
            ("bytes", verified.bytes),
            ("sha256", verified.sha256),
        ),
    )
    return 0


def _verify_bundle(args: argparse.Namespace) -> int:
    from heliostune.methodology import verify_bundle_v1

    verified = verify_bundle_v1(args.path)
    bundle = verified.bundle
    facts: list[tuple[str, str | int | Path]] = [
        ("path", verified.root_path),
        ("lifecycle", bundle.lifecycle.state),
        ("outcome", bundle.lifecycle.outcome),
        ("evidence_class", verified.protocol.protocol.evidence_class),
        ("root_bytes", verified.root_bytes),
        ("root_sha256", verified.root_sha256),
        ("protocol_bytes", verified.protocol.bytes),
        ("protocol_sha256", verified.protocol.sha256),
        ("referenced_file_count", len(verified.referenced_paths)),
        ("artifact_count", len(bundle.artifacts)),
        ("signature_count", len(bundle.signatures)),
        ("logical_attempts", bundle.attempts.logical),
        ("physical_attempts", bundle.attempts.physical),
        ("terminal_attempts", bundle.attempts.terminal),
        ("orphaned_attempts", bundle.attempts.orphaned),
        ("attempts_sha256", bundle.attempts.sha256),
        ("expected_cells", bundle.coverage.expected_cells),
        ("terminal_cells", bundle.coverage.terminal_cells),
        ("successes", bundle.coverage.successes),
        ("failures", bundle.coverage.failures),
    ]
    for index, artifact in enumerate(bundle.artifacts):
        facts.extend(
            (
                (f"artifact[{index}].role", artifact.role),
                (f"artifact[{index}].bytes", artifact.bytes),
                (f"artifact[{index}].sha256", artifact.sha256),
            )
        )
    for limitation in dataclass_fields(verified.limitations):
        facts.append(
            (
                f"limitation.{limitation.name}",
                getattr(verified.limitations, limitation.name),
            )
        )
    _print_structural_verification("Bundle", facts)
    return 0


def _is_local_repository(repository: Path) -> bool:
    return (repository / "pyproject.toml").is_file() and (repository / "src/heliostune").is_dir()


def _local_suite_repository(suite: Path) -> Path | None:
    resolved = suite.resolve()
    suites_directory = resolved.parent
    benchmarks_directory = suites_directory.parent
    if suites_directory.name != "suites" or benchmarks_directory.name != "benchmarks":
        return None

    repository = benchmarks_directory.parent
    return repository if _is_local_repository(repository) else None


def _local_output_directory(output: Path) -> Path:
    resolved = output.resolve()
    for ancestor in (resolved, *resolved.parents):
        if ancestor.name not in {"benchmarks", "site"}:
            continue
        if _is_local_repository(ancestor.parent):
            raise ArtifactError(
                f"refusing local suite output inside protected repository directory "
                f"{ancestor.name}: {output}"
            )

    if output.is_symlink():
        raise ArtifactError(f"refusing existing local suite output symlink: {output}")
    if output.exists():
        if not output.is_dir():
            raise ArtifactError(
                f"local suite output destination exists and is not a directory: {output}"
            )
        try:
            next(output.iterdir())
        except StopIteration:
            pass
        else:
            raise ArtifactError(
                f"refusing existing nonempty local suite output directory: {output}"
            )
    return output


def _local_plugin_path(suite: Path, explicit_plugin: Path | None) -> Path:
    if explicit_plugin is not None:
        return explicit_plugin

    repository = _local_suite_repository(suite)
    if repository is None:
        raise ArtifactError(
            "--plugin is required unless SUITE resolves to a committed reference template"
        )
    templates = {
        (repository / "benchmarks/suites/gated-mlp-epilogue-v1.json").resolve(): (
            repository / "benchmarks/plugins/fusion-reference-plugin-v1.json"
        ),
        (repository / "benchmarks/suites/residual-rmsnorm-v1.json").resolve(): (
            repository / "benchmarks/plugins/fusion-reference-plugin-v1.json"
        ),
        (repository / "benchmarks/suites/residual-rmsnorm-triton-v1.json").resolve(): (
            repository / "benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json"
        ),
    }
    try:
        return templates[suite.resolve()]
    except KeyError:
        raise ArtifactError(
            "--plugin is required unless SUITE resolves to a committed reference template"
        ) from None


def _run_local_suite(args: argparse.Namespace) -> int:
    output_dir = _local_output_directory(args.output)
    plugin_path = _local_plugin_path(args.suite, args.plugin)

    from heliostune.local_executor import (
        NATIVE_RMSNORM_SUITE_SHA256,
        LocalExecutionResult,
        execute_local_suite,
    )
    from heliostune.scope import verify_suite

    try:
        selected_suite = verify_suite(args.suite)
    except HeliostuneError:
        selected_suite = None
    native_selected = (
        selected_suite is not None
        and selected_suite.sha256 == NATIVE_RMSNORM_SUITE_SHA256
    )
    if native_selected:
        if output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError as exc:
                raise ArtifactError(
                    f"local suite output directory is no longer empty: {output_dir}"
                ) from exc
        from heliostune.native_fusion_bundle import preflight_native_fusion_bundle

        preflight_native_fusion_bundle(
            args.suite,
            plugin_path=plugin_path,
            output_dir=output_dir,
        )


    result = execute_local_suite(args.suite)
    if not native_selected and output_dir.exists():
        try:
            output_dir.rmdir()
        except OSError as exc:
            raise ArtifactError(
                f"local suite output directory is no longer empty: {output_dir}"
            ) from exc
    if result.verified_suite_sha256 == NATIVE_RMSNORM_SUITE_SHA256:
        from heliostune.native_fusion_bundle import write_native_fusion_bundle
        from heliostune.native_fusion_executor import NativeFusionExecutionResult

        if not isinstance(result, NativeFusionExecutionResult):
            raise SchemaError("native suite digest returned a non-native execution result")
        verified = write_native_fusion_bundle(
            result,
            plugin_path=plugin_path,
            output_dir=output_dir,
        )
    else:
        from heliostune.local_bundle import write_local_bundle

        if not isinstance(result, LocalExecutionResult):
            raise SchemaError("legacy suite digest returned a non-legacy execution result")
        verified = write_local_bundle(
            result,
            plugin_path=plugin_path,
            output_dir=output_dir,
        )
    coverage = verified.bundle.coverage
    facts: list[tuple[str, str | int | Path]] = [
        ("suite", result.suite_id),
        (
            "local_cuda_capability",
            "available" if result.capability.available else "unavailable",
        ),
        ("outcome", result.outcome),
        ("cells.expected", coverage.expected_cells),
        ("cells.terminal", coverage.terminal_cells),
        ("cells.successes", coverage.successes),
        ("cells.failures", coverage.failures),
        ("bundle_root", verified.root_path),
        ("bundle_root_sha256", verified.root_sha256),
    ]
    for limitation in dataclass_fields(verified.limitations):
        facts.append(
            (
                f"structural_limitation.{limitation.name}",
                getattr(verified.limitations, limitation.name),
            )
        )
    facts.append(
        (
            "limitation",
            "Bundle verification is structural only; it does not establish "
            "execution semantics, comparative performance, or claim eligibility.",
        )
    )
    _print_facts("Local CUDA suite result recorded", facts)
    return (
        0
        if result.outcome == "completed"
        and coverage.failures == 0
        and coverage.terminal_cells == coverage.expected_cells
        else 2
    )


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

    verify_protocol = subparsers.add_parser(
        "verify-protocol",
        help=(
            "strictly verify a non-retroactive methodology v1 protocol "
            "(legacy manifests are rejected)"
        ),
        description=(
            "Legacy manifests are rejected. Non-retroactive v1 structural "
            "verification for a methodology protocol."
        ),
    )
    verify_protocol.add_argument("path", type=Path, metavar="PATH")
    verify_protocol.set_defaults(handler=_verify_protocol)

    verify_bundle = subparsers.add_parser(
        "verify-bundle",
        help=(
            "strictly verify a non-retroactive methodology v1 evidence bundle "
            "(legacy manifests are rejected)"
        ),
        description=(
            "Legacy manifests are rejected. Non-retroactive v1 structural "
            "verification for a methodology evidence bundle."
        ),
    )
    verify_bundle.add_argument("path", type=Path, metavar="PATH")
    verify_bundle.set_defaults(handler=_verify_bundle)

    verify_plugin = subparsers.add_parser(
        "verify-plugin",
        help="structurally verify a plugin and its transitively referenced suites",
        description=(
            "Legacy plugin artifacts are rejected. Strict structural verification "
            "of a heliostune.plugin/1 artifact and the relative suite paths and "
            "SHA-256 digests it references. This does not validate execution, "
            "correctness, or performance."
        ),
    )
    verify_plugin.add_argument("path", type=Path, metavar="PATH")
    verify_plugin.set_defaults(handler=_verify_plugin)

    verify_suite = subparsers.add_parser(
        "verify-suite",
        help="structurally verify a frozen suite declaration",
        description=(
            "Legacy suite artifacts are rejected. Strict structural verification "
            "of a heliostune.suite/1 artifact. Correctness passage and execution "
            "are not observed."
        ),
    )
    verify_suite.add_argument("path", type=Path, metavar="PATH")
    verify_suite.set_defaults(handler=_verify_suite)

    run_local_suite = subparsers.add_parser(
        "run-local-suite",
        help="execute one frozen suite on local CUDA and write an evidence bundle",
        description=(
            "Execute one frozen reference suite on a qualifying local CUDA device, "
            "retain correctness and timing observations, and write a structurally "
            "verified exploratory evidence bundle."
        ),
    )
    run_local_suite.add_argument("suite", type=Path, metavar="SUITE")
    run_local_suite.add_argument(
        "--plugin",
        type=Path,
        metavar="PLUGIN",
        help=(
            "plugin artifact bound into the bundle; defaults to the committed "
            "reference plugin only for a committed suite template"
        ),
    )
    run_local_suite.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="new or empty destination directory outside benchmarks/ and site/",
    )
    run_local_suite.set_defaults(handler=_run_local_suite)

    list_scope = subparsers.add_parser(
        "list-scope",
        help="list schema vocabulary, frozen templates, and runtime implementation status",
        description=(
            "List schema vocabulary, the narrow frozen suite templates, and explicitly scoped "
            "local/remote runtime implementation and validation status."
        ),
    )
    list_scope.set_defaults(handler=_list_scope)
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
