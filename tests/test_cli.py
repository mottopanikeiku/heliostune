from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import shutil
from collections.abc import Callable
from importlib.metadata import version
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import zstandard

import heliostune.cli as cli
from heliostune.artifacts import write_json_atomic
from heliostune.errors import ArtifactError, SchemaError
from heliostune.methodology import VerificationLimitations
from heliostune.multisource_engine import ReleaseProvenance, validate_release_provenance


@pytest.mark.parametrize(
    "arguments",
    [
        ["compare", "rows.jsonl", "--source", "L4", "--target", "A10", "--seeds", "0"],
        [
            "compare",
            "rows.jsonl",
            "--source",
            "L4",
            "--target",
            "A10",
            "--transfer-strength",
            "NaN",
        ],
        [
            "compare-multisource",
            "rows.jsonl",
            "--sources",
            "L4, A10",
            "--target",
            "T4",
        ],
        [
            "compare-multisource",
            "rows.jsonl",
            "--sources",
            "L4,L4",
            "--target",
            "T4",
        ],
        ["demo", "--seed", "-1"],
    ],
)
def test_argparse_rejects_bad_numeric_and_list_values(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(arguments)
    assert raised.value.code == 2
    assert ": error:" in capsys.readouterr().err


class _FakeParser:
    def __init__(self, handler: Callable[[argparse.Namespace], int]) -> None:
        self._handler = handler

    def parse_args(self, _argv: object) -> argparse.Namespace:
        return argparse.Namespace(handler=self._handler)


def _raising_handler(error: BaseException) -> Callable[[argparse.Namespace], int]:
    def handler(_args: argparse.Namespace) -> int:
        raise error

    return handler


@pytest.mark.parametrize(
    "error",
    [
        SchemaError("bad schema"),
        OSError("disk failed"),
        json.JSONDecodeError("bad json", "{", 1),
    ],
)
def test_main_formats_only_expected_user_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
) -> None:
    monkeypatch.setattr(cli, "build_parser", lambda: _FakeParser(_raising_handler(error)))

    assert cli.main([]) == 2
    assert capsys.readouterr().err.startswith("heliostune: error: ")


@pytest.mark.parametrize("error", [ValueError("bug"), TypeError("bug"), KeyboardInterrupt()])
def test_main_does_not_catch_programmer_faults_or_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    monkeypatch.setattr(cli, "build_parser", lambda: _FakeParser(_raising_handler(error)))
    with pytest.raises(
        type(error), match="bug" if not isinstance(error, KeyboardInterrupt) else None
    ):
        cli.main([])


def test_output_collision_is_rejected_before_input_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "summary.json"
    output.write_text("existing", encoding="utf-8")

    def unexpected_read(_path: Path) -> object:
        raise AssertionError("input was accessed before output collision rejection")

    monkeypatch.setattr(cli, "read_measurements", unexpected_read)
    result = cli.main(
        [
            "compare",
            str(tmp_path / "missing.jsonl"),
            "--source",
            "L4",
            "--target",
            "A10",
            "--output",
            str(output),
        ]
    )

    assert result == 2
    assert "refusing to replace existing output" in capsys.readouterr().err


def test_invalid_report_is_normalized_before_output_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "bad-summary.json"
    output = tmp_path / "report.html"
    write_json_atomic(
        source,
        {
            "source_gpu": "L4",
            "target_gpu": "A10",
            "workloads": 1,
            "configs": 1,
            "methods": {"new_method": []},
        },
    )

    assert cli.main(["report", str(source), "--output", str(output)]) == 2
    assert not output.exists()
    assert "heliostune: error:" in capsys.readouterr().err


def test_demo_stages_complete_strict_payloads(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"

    assert (
        cli.main(
            [
                "demo",
                "--output-dir",
                str(output_dir),
                "--max-budget",
                "1",
                "--seeds",
                "1",
            ]
        )
        == 0
    )

    assert (output_dir / "measurements.jsonl").is_file()
    assert (output_dir / "summary.json").is_file()
    report = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Content-Security-Policy" in report
    assert "control-panel" in report


def _release_fixture(
    root: Path,
    *,
    manifest_freeze_sha256: str | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    benchmarks = root / "benchmarks"
    data = benchmarks / "data"
    data.mkdir(parents=True)
    canonical = b'{"canonical":"archive"}\n'
    compressed = zstandard.ZstdCompressor(level=3).compress(canonical)
    archive = data / "replay.jsonl.zst"
    archive.write_bytes(compressed)
    transport_sha256 = hashlib.sha256(compressed).hexdigest()
    canonical_sha256 = hashlib.sha256(canonical).hexdigest()
    freeze_sha256 = "c" * 64
    provenance: dict[str, object] = {
        "algorithm_commit": "a" * 40,
        "freeze_commit": "b" * 40,
        "freeze_sha256": freeze_sha256,
        "sole_h100_run": "https://modal.com/apps/example/main/ap-release",
        "raw_h100_sha256": "d" * 64,
        "final_archive_sha256": canonical_sha256,
        "post_run_manifest_path": "benchmarks/post-run.json",
    }
    manifest = {
        "commits": {
            "algorithm_and_development_protocol": provenance["algorithm_commit"],
            "executable_h100_freeze": provenance["freeze_commit"],
        },
        "runs": {"h100_final_sole_invocation": provenance["sole_h100_run"]},
        "freeze": {
            "sha256": (freeze_sha256 if manifest_freeze_sha256 is None else manifest_freeze_sha256)
        },
        "data": {
            "h100_raw": {"uncompressed_sha256": provenance["raw_h100_sha256"]},
            "four_gpu_replay_archive": {
                "published_path": "data/replay.jsonl.zst",
                "compressed_sha256": transport_sha256,
                "uncompressed_sha256": canonical_sha256,
            },
        },
    }
    provenance_path = benchmarks / "release-provenance.json"
    write_json_atomic(provenance_path, provenance)
    write_json_atomic(benchmarks / "post-run.json", manifest)
    return archive, provenance_path, provenance


def test_release_authentication_distinguishes_transport_and_canonical_digests(
    tmp_path: Path,
) -> None:
    archive, provenance_path, provenance = _release_fixture(tmp_path)
    validated = validate_release_provenance(provenance)
    snapshot = cli._read_archive_snapshot(archive)

    authenticated = cli._verify_release_provenance(
        archive,
        provenance_path,
        validated,
        snapshot,
    )

    assert snapshot.transport_sha256 != snapshot.canonical_sha256
    assert snapshot.canonical_sha256 == provenance["final_archive_sha256"]
    assert authenticated is validated
    assert dict(authenticated) == provenance


@pytest.mark.parametrize(
    "path",
    [
        ("commits", "algorithm_and_development_protocol"),
        ("commits", "executable_h100_freeze"),
        ("freeze", "sha256"),
        ("runs", "h100_final_sole_invocation"),
        ("data", "h100_raw", "uncompressed_sha256"),
        ("data", "four_gpu_replay_archive", "published_path"),
        ("data", "four_gpu_replay_archive", "compressed_sha256"),
        ("data", "four_gpu_replay_archive", "uncompressed_sha256"),
    ],
)
def test_release_authentication_requires_every_published_manifest_fact(
    tmp_path: Path,
    path: tuple[str, ...],
) -> None:
    archive, provenance_path, provenance = _release_fixture(tmp_path)
    manifest_path = tmp_path / "benchmarks/post-run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = manifest
    for key in path[:-1]:
        current = current[key]
    del current[path[-1]]
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(SchemaError, match="is required for release authentication"):
        cli._verify_release_provenance(
            archive,
            provenance_path,
            validate_release_provenance(provenance),
            cli._read_archive_snapshot(archive),
        )


def _compressed_frame(payload: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=3, write_checksum=True).compress(payload)


def _highly_compressible_frame(payload_size: int) -> bytes:
    compressed = BytesIO()
    chunk = b"x" * (1024 * 1024)
    full_chunks, remainder = divmod(payload_size, len(chunk))
    with zstandard.ZstdCompressor(level=3, write_checksum=True).stream_writer(
        compressed,
        size=payload_size,
        closefd=False,
    ) as writer:
        for _ in range(full_chunks):
            writer.write(chunk)
        writer.write(chunk[:remainder])
    return compressed.getvalue()


def test_release_archive_accepts_exact_uncompressed_limit(tmp_path: Path) -> None:
    canonical = b"x" * 1024
    assert (
        cli._decompress_archive(
            _compressed_frame(canonical),
            source_path=tmp_path / "exact.jsonl.zst",
            maximum_uncompressed_bytes=len(canonical),
        )
        == canonical
    )


def test_release_archive_rejects_decompression_bomb_over_limit(tmp_path: Path) -> None:
    canonical = b"x" * 1025
    with pytest.raises(ArtifactError, match="exceeds maximum uncompressed size of 1024 bytes"):
        cli._decompress_archive(
            _compressed_frame(canonical),
            source_path=tmp_path / "bomb.jsonl.zst",
            maximum_uncompressed_bytes=1024,
        )


def test_release_archive_rejects_highly_compressible_bomb_over_64_mib(
    tmp_path: Path,
) -> None:
    limit = cli._MAX_RELEASE_ARCHIVE_UNCOMPRESSED_BYTES
    compressed = _highly_compressible_frame(limit + 1)
    assert len(compressed) < 4096

    with pytest.raises(
        ArtifactError,
        match=f"exceeds maximum uncompressed size of {limit} bytes",
    ):
        cli._decompress_archive(
            compressed,
            source_path=tmp_path / "bomb-64-mib.jsonl.zst",
            maximum_uncompressed_bytes=limit,
        )


def test_release_archive_rejects_truncated_final_frame(tmp_path: Path) -> None:
    compressed = _compressed_frame(b'{"complete":true}\n')
    with pytest.raises(ArtifactError, match="truncated or incomplete zstd frame"):
        cli._decompress_archive(
            compressed[:-1],
            source_path=tmp_path / "truncated.jsonl.zst",
            maximum_uncompressed_bytes=1024,
        )


def test_release_archive_rejects_empty_zstd_stream(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="empty zstd input"):
        cli._decompress_archive(
            b"",
            source_path=tmp_path / "empty.jsonl.zst",
            maximum_uncompressed_bytes=1024,
        )


def test_release_archive_rejects_malformed_frame(tmp_path: Path) -> None:
    compressed = bytearray(_compressed_frame(b'{"complete":true}\n'))
    first_block = zstandard.frame_header_size(compressed)
    compressed[first_block] = (compressed[first_block] & ~0b110) | 0b110

    with pytest.raises(ArtifactError, match="reserved zstd block type"):
        cli._decompress_archive(
            bytes(compressed),
            source_path=tmp_path / "malformed.jsonl.zst",
            maximum_uncompressed_bytes=1024,
        )


def test_release_archive_consumes_all_zstd_frames(tmp_path: Path) -> None:
    compressed = _compressed_frame(b'{"frame":1}\n') + _compressed_frame(b'{"frame":2}\n')
    assert (
        cli._decompress_archive(
            compressed,
            source_path=tmp_path / "multiframe.jsonl.zst",
            maximum_uncompressed_bytes=1024,
        )
        == b'{"frame":1}\n{"frame":2}\n'
    )


def test_release_archive_applies_limit_cumulatively_across_frames(tmp_path: Path) -> None:
    compressed = _compressed_frame(b"x" * 600) + _compressed_frame(b"y" * 600)

    with pytest.raises(ArtifactError, match="exceeds maximum uncompressed size of 1024 bytes"):
        cli._decompress_archive(
            compressed,
            source_path=tmp_path / "multiframe-bomb.jsonl.zst",
            maximum_uncompressed_bytes=1024,
        )


def test_release_archive_rejects_trailing_garbage(tmp_path: Path) -> None:
    compressed = _compressed_frame(b'{"complete":true}\n') + b"not-a-zstd-frame"
    with pytest.raises(ArtifactError, match="cannot authenticate release archive"):
        cli._decompress_archive(
            compressed,
            source_path=tmp_path / "trailing.jsonl.zst",
            maximum_uncompressed_bytes=1024,
        )


def test_release_replay_uses_authenticated_snapshot_after_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, provenance_path, provenance = _release_fixture(tmp_path)
    supplied = archive.with_name("supplied.jsonl.zst")
    supplied.symlink_to(archive)
    replacement = archive.with_name("replacement.jsonl.zst")
    replacement.write_bytes(_compressed_frame(b'{"replacement":true}\n'))
    original_verify = cli._verify_release_provenance
    parsed_marker = object()

    def verify_then_swap(
        input_path: Path,
        supplied_provenance_path: Path,
        validated: ReleaseProvenance,
        snapshot: cli._ArchiveSnapshot,
    ) -> ReleaseProvenance:
        result = original_verify(
            input_path,
            supplied_provenance_path,
            validated,
            snapshot,
        )
        supplied.unlink()
        supplied.symlink_to(replacement)
        return result

    def parse_snapshot(source: StringIO, *, source_name: Path) -> list[object]:
        assert source.read() == '{"canonical":"archive"}\n'
        assert source_name == supplied
        return [parsed_marker]

    def compare_snapshot(measurements: object, **kwargs: object) -> dict[str, object]:
        assert measurements == [parsed_marker]
        release = kwargs["release_provenance"]
        assert isinstance(release, ReleaseProvenance)
        return {"release_provenance": dict(release)}

    monkeypatch.setattr(cli, "_verify_release_provenance", verify_then_swap)
    monkeypatch.setattr(cli, "read_jsonl", parse_snapshot)
    monkeypatch.setattr(cli, "compare_multisource", compare_snapshot)
    monkeypatch.setattr(
        cli,
        "read_measurements",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("authenticated input path was reopened")
        ),
    )
    output = tmp_path / "summary.json"

    assert (
        cli.main(
            [
                "compare-multisource",
                str(supplied),
                "--sources",
                "L4,A10,T4",
                "--target",
                "H100",
                "--release-provenance",
                str(provenance_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {"release_provenance": provenance}


@pytest.mark.parametrize("mismatch", ["archive", "manifest"])
def test_release_mismatch_fails_before_measurement_or_replay_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mismatch: str,
) -> None:
    archive, provenance_path, provenance = _release_fixture(
        tmp_path,
        manifest_freeze_sha256="f" * 64 if mismatch == "manifest" else None,
    )
    if mismatch == "archive":
        provenance["final_archive_sha256"] = "e" * 64
        write_json_atomic(provenance_path, provenance)

    def unexpected_access(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("measurement data or replay was accessed before authentication")

    monkeypatch.setattr(cli, "read_measurements", unexpected_access)
    monkeypatch.setattr(cli, "compare_multisource", unexpected_access)
    result = cli.main(
        [
            "compare-multisource",
            str(archive),
            "--sources",
            "L4,A10,T4",
            "--target",
            "H100",
            "--release-provenance",
            str(provenance_path),
            "--output",
            str(tmp_path / "summary.json"),
        ]
    )

    assert result == 2
    assert "does not match" in capsys.readouterr().err


def test_committed_h100_release_replay_remains_byte_identical(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "h100-final.json"

    result = cli.main(
        [
            "compare-multisource",
            str(repository / "benchmarks/data/parhelion-v2-measurements.jsonl.zst"),
            "--sources",
            "L4,A10,T4",
            "--target",
            "H100",
            "--max-budget",
            "8",
            "--seeds",
            "30",
            "--k",
            "16",
            "--temperature",
            "2.0",
            "--transfer-strength",
            "0.0",
            "--retrieval-k",
            "8",
            "--retrieval-temperature",
            "0.2",
            "--pooled-transfer-strength",
            "0.0",
            "--primary-comparator",
            "torch",
            "--protocol-role",
            "final",
            "--release-provenance",
            str(repository / "benchmarks/parhelion-h100-release-provenance.json"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        "765b347a2675b0647f9f58bd6ba36904dfcf2761be31b7e3b930b63a2ad28abd"
    )


def _methodology_bundle_fixture(
    tmp_path: Path,
    *,
    study_id: str = "[bold]literal[/bold]\nsecond line",
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    repository = Path(__file__).resolve().parents[1]
    template = repository / "benchmarks/methodology-protocol-v1-template.json"
    protocol_data: dict[str, object] = json.loads(template.read_text(encoding="utf-8"))
    protocol_data["study_id"] = study_id

    directory = tmp_path / 'bundle [v1] "quoted"'
    directory.mkdir()
    protocol = directory / 'protocol [v1] "quoted".json'
    attempts = directory / "attempts [v1].jsonl"
    artifact = directory / "result [v1].json"
    root = directory / 'bundle [v1] "quoted".json'
    artifact.write_bytes(b'{"result":"structural"}\n')

    role_payloads = {
        "plugin": b"plugin\n",
        "workloads": b"workloads\n",
        "candidates": b"candidates\n",
        "comparators": b"comparators\n",
        "splits": b"splits\n",
        "numerics": b"numerics\n",
        "timing": b"timing\n",
        "analyzer": b"analyzer\n",
        "expected_cells": b"[]\n",
        "terminal_cells": b"[]\n",
        "environment_predicate": b"environment predicate\n",
        "failure_policy": b"failure policy\n",
    }
    role_paths: dict[str, Path] = {}
    role_digests: dict[str, str] = {}
    for role, payload in role_payloads.items():
        path = directory / f"{role}.artifact"
        path.write_bytes(payload)
        role_paths[role] = path
        role_digests[role] = hashlib.sha256(payload).hexdigest()

    plugin = cast(dict[str, object], protocol_data["plugin"])
    semantic = cast(dict[str, object], protocol_data["semantic"])
    analysis = cast(dict[str, object], protocol_data["analysis"])
    execution = cast(dict[str, object], protocol_data["execution"])
    plugin["artifact_sha256"] = role_digests["plugin"]
    for role in ("workloads", "candidates", "comparators", "splits", "numerics", "timing"):
        semantic[f"{role}_sha256"] = role_digests[role]
    analysis["analyzer_sha256"] = role_digests["analyzer"]
    execution["expected_cells_sha256"] = role_digests["expected_cells"]
    execution["environment_predicate_sha256"] = role_digests["environment_predicate"]
    execution["failure_policy_sha256"] = role_digests["failure_policy"]

    write_json_atomic(protocol, protocol_data)
    attempts.write_bytes(b"")
    protocol_payload = protocol.read_bytes()
    attempts_payload = attempts.read_bytes()
    artifact_payload = artifact.read_bytes()
    bundle_artifacts: list[dict[str, object]] = [
        {
            "role": "[bold]result[/bold]\ncontinued",
            "path": artifact.name,
            "media_type": "application/json",
            "bytes": len(artifact_payload),
            "sha256": hashlib.sha256(artifact_payload).hexdigest(),
        }
    ]
    bundle_artifacts.extend(
        {
            "role": role,
            "path": role_paths[role].name,
            "media_type": (
                "application/json"
                if role in {"expected_cells", "terminal_cells"}
                else "application/octet-stream"
            ),
            "bytes": len(payload),
            "sha256": role_digests[role],
        }
        for role, payload in role_payloads.items()
    )
    bundle: dict[str, object] = {
        "schema": "heliostune.bundle/1",
        "bundle_id": "cli-structural-verification",
        "created_at": "2026-08-26T12:00:00Z",
        "protocol": {
            "path": protocol.name,
            "sha256": hashlib.sha256(protocol_payload).hexdigest(),
            "bytes": len(protocol_payload),
        },
        "lifecycle": {"state": "SEALED", "outcome": "completed"},
        "attempts": {
            "path": attempts.name,
            "sha256": hashlib.sha256(attempts_payload).hexdigest(),
            "hash_chain_head": "0" * 64,
            "logical": 0,
            "physical": 0,
            "terminal": 0,
            "orphaned": 0,
        },
        "coverage": {
            "expected_cells": 0,
            "terminal_cells": 0,
            "successes": 0,
            "failures": 0,
        },
        "artifacts": bundle_artifacts,
        "provenance": {
            "attestation": "none",
            "offline_reproduction": "not_checked",
        },
        "signatures": [],
    }
    write_json_atomic(root, bundle)
    return root, protocol, attempts, artifact, bundle


def test_verify_protocol_accepts_committed_v1_template(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol = repository / "benchmarks/methodology-protocol-v1-template.json"

    assert cli.main(["verify-protocol", str(protocol)]) == 0

    output = capsys.readouterr().out
    assert "Protocol structurally verified" in output
    assert "schema: heliostune.protocol/1" in output
    assert "study: methodology-protocol-v1-template-reference-not-a-study-freeze" in output
    assert "revision: 1" in output
    assert "evidence_class: exploratory" in output
    assert f"bytes: {len(protocol.read_bytes())}" in output
    assert f"sha256: {hashlib.sha256(protocol.read_bytes()).hexdigest()}" in output


def test_verify_bundle_prints_verified_inventory_and_all_limitations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, protocol, attempts, artifact, _bundle = _methodology_bundle_fixture(tmp_path)

    assert cli.main(["verify-bundle", str(root)]) == 0

    output = capsys.readouterr().out
    assert "Bundle structurally verified" in output
    assert "lifecycle: SEALED" in output
    assert "outcome: completed" in output
    assert "evidence_class: exploratory" in output
    assert f"root_sha256: {hashlib.sha256(root.read_bytes()).hexdigest()}" in output
    assert f"protocol_sha256: {hashlib.sha256(protocol.read_bytes()).hexdigest()}" in output
    assert f"attempts_sha256: {hashlib.sha256(attempts.read_bytes()).hexdigest()}" in output
    assert f"artifact[0].sha256: {hashlib.sha256(artifact.read_bytes()).hexdigest()}" in output
    assert "referenced_file_count: 15" in output
    assert "artifact_count: 13" in output
    assert "signature_count: 0" in output
    assert "expected_cells: 0" in output
    assert "terminal_cells: 0" in output
    for limitation in (
        "protocol_ancestry",
        "evidence_nonpromotion",
        "semantic_content_beyond_digests",
        "plugin_suite_custody",
        "attempt_journal_hash_chain",
        "claim_eligibility",
        "analyzer_replay",
        "provenance_tier_derivation",
        "signature_cryptography",
        "catalog_membership",
        "offline_reproduction",
    ):
        assert f"limitation.{limitation}: not_checked" in output
    assert "limitation.attempt_reconciliation: checked" in output
    assert output.count(": not_checked") == 11
    assert "verification_record_schema: heliostune.verification-record/1" in output
    assert "verifier_source_sha256: " in output
    assert "claim_eligible: false" in output
    assert "publication_eligible: false" in output
    assert "publication eligible" not in output.lower()
    assert "authenticated" not in output.lower()


@pytest.mark.parametrize(
    ("arguments", "output_format", "output"),
    [
        (["verify-bundle", "bundle.json"], None, None),
        (["verify-bundle", "bundle.json", "--format", "text"], "text", None),
        (["verify-bundle", "bundle.json", "--format", "json"], "json", None),
        (
            ["verify-bundle", "bundle.json", "--output", "record.json"],
            None,
            Path("record.json"),
        ),
        (
            [
                "verify-bundle",
                "bundle.json",
                "--format",
                "json",
                "--output",
                "record.json",
            ],
            "json",
            Path("record.json"),
        ),
        (
            [
                "verify-bundle",
                "bundle.json",
                "--format",
                "text",
                "--output",
                "record.json",
            ],
            "text",
            Path("record.json"),
        ),
    ],
)
def test_verify_bundle_parser_preserves_omitted_format(
    arguments: list[str],
    output_format: str | None,
    output: Path | None,
) -> None:
    parsed = cli.build_parser().parse_args(arguments)
    assert parsed.output_format == output_format
    assert parsed.output == output


def test_verify_bundle_rejects_explicit_text_file_before_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import heliostune.methodology as methodology

    def forbidden_verify(_path: Path) -> object:
        raise AssertionError("verification must not run")

    monkeypatch.setattr(methodology, "verify_bundle_v1", forbidden_verify)
    output = tmp_path / "record.json"

    assert (
        cli.main(
            [
                "verify-bundle",
                str(tmp_path / "missing.json"),
                "--format",
                "text",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "--format text cannot be combined with --output" in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_verify_bundle_json_stdout_is_exact_canonical_bytes_and_bypasses_rich(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    from heliostune.methodology import verify_bundle_v1
    from heliostune.verification import (
        build_verification_record_v1,
        encode_verification_record_v1,
    )

    root, _protocol, _attempts, _artifact, _bundle = _methodology_bundle_fixture(tmp_path)
    expected = encode_verification_record_v1(build_verification_record_v1(verify_bundle_v1(root)))

    def forbidden_rich(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("JSON output must bypass Rich")

    monkeypatch.setattr(cli._CONSOLE, "print", forbidden_rich)
    assert cli.main(["verify-bundle", str(root), "--format", "json"]) == 0
    captured = capfdbinary.readouterr()
    assert captured.out == expected
    assert captured.err == b""
    assert captured.out.endswith(b"\n")
    assert not captured.out.endswith(b"\n\n")


def test_verify_bundle_record_is_repeatable_and_location_independent(
    tmp_path: Path,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    first_parent = tmp_path / "first"
    first_parent.mkdir()
    first, _protocol, _attempts, _artifact, _bundle = _methodology_bundle_fixture(first_parent)
    second_directory = tmp_path / "second" / first.parent.name
    second_directory.parent.mkdir()
    shutil.copytree(first.parent, second_directory)
    second = second_directory / first.name

    assert cli.main(["verify-bundle", str(first), "--format", "json"]) == 0
    first_bytes = capfdbinary.readouterr().out
    assert cli.main(["verify-bundle", str(first), "--format", "json"]) == 0
    repeated_bytes = capfdbinary.readouterr().out
    assert cli.main(["verify-bundle", str(second), "--format", "json"]) == 0
    relocated_bytes = capfdbinary.readouterr().out

    assert first_bytes == repeated_bytes == relocated_bytes


def test_verify_bundle_output_file_matches_stdout_and_is_silent(
    tmp_path: Path,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    root, _protocol, _attempts, _artifact, _bundle = _methodology_bundle_fixture(tmp_path)
    output = tmp_path / "verification-record.json"

    assert cli.main(["verify-bundle", str(root), "--format", "json"]) == 0
    stdout_bytes = capfdbinary.readouterr().out
    assert cli.main(["verify-bundle", str(root), "--output", str(output)]) == 0
    captured = capfdbinary.readouterr()

    assert captured.out == b""
    assert captured.err == b""
    assert output.read_bytes() == stdout_bytes


def test_verify_bundle_deferred_controls_exit_zero_without_lifecycle_promotion(
    tmp_path: Path,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    root, _protocol, _attempts, _artifact, bundle = _methodology_bundle_fixture(tmp_path)
    lifecycle = cast(dict[str, object], bundle["lifecycle"])
    lifecycle["state"] = "PUBLISHED"
    write_json_atomic(root, bundle)

    assert cli.main(["verify-bundle", str(root), "--format", "json"]) == 0
    captured = capfdbinary.readouterr()
    record = json.loads(captured.out)

    assert captured.err == b""
    assert record["lifecycle"] == {"state": "PUBLISHED", "outcome": "completed"}
    assert "not_checked" in record["controls"].values()
    assert record["claim_eligible"] is False
    assert record["publication_eligible"] is False


def test_verify_bundle_failed_control_suppresses_stdout_and_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    import heliostune.verification as verification

    root, _protocol, _attempts, _artifact, _bundle = _methodology_bundle_fixture(tmp_path)
    output = tmp_path / "verification-record.json"
    monkeypatch.setattr(
        verification,
        "build_verification_record_v1",
        lambda _verified: SimpleNamespace(has_failed_controls=True),
    )

    assert cli.main(["verify-bundle", str(root), "--format", "json"]) == 2
    stdout_failure = capfdbinary.readouterr()
    assert stdout_failure.out == b""
    assert b"failed controls" in stdout_failure.err

    assert cli.main(["verify-bundle", str(root), "--output", str(output)]) == 2
    file_failure = capfdbinary.readouterr()
    assert file_failure.out == b""
    assert b"failed controls" in file_failure.err
    assert not output.exists()


def test_verify_bundle_malformed_bundle_creates_no_record(
    tmp_path: Path,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    root, _protocol, _attempts, _artifact, _bundle = _methodology_bundle_fixture(tmp_path)
    root.write_bytes(b"{\n")
    output = tmp_path / "verification-record.json"

    assert cli.main(["verify-bundle", str(root), "--output", str(output)]) == 2
    captured = capfdbinary.readouterr()
    assert captured.out == b""
    assert captured.err
    assert not output.exists()


@pytest.mark.parametrize("stage", ["build", "encode", "write"])
def test_verify_bundle_record_failure_emits_no_success_output(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    import heliostune.verification as verification

    root, _protocol, _attempts, _artifact, _bundle = _methodology_bundle_fixture(tmp_path)
    output = tmp_path / "verification-record.json"

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ArtifactError(f"{stage} failed")

    if stage == "build":
        monkeypatch.setattr(verification, "build_verification_record_v1", fail)
        arguments = ["verify-bundle", str(root), "--format", "json"]
    elif stage == "encode":
        monkeypatch.setattr(verification, "encode_verification_record_v1", fail)
        arguments = ["verify-bundle", str(root), "--format", "json"]
    else:
        monkeypatch.setattr(verification, "write_verification_record_v1", fail)
        arguments = ["verify-bundle", str(root), "--output", str(output)]

    assert cli.main(arguments) == 2
    captured = capfdbinary.readouterr()
    assert captured.out == b""
    assert f"{stage} failed".encode() in captured.err
    assert not output.exists()


@pytest.mark.parametrize(
    "collision",
    [
        "existing",
        "bundle",
        "protocol",
        "attempts",
        "artifact",
        "symlink",
        "hardlink",
        "dangling_symlink",
    ],
)
def test_verify_bundle_output_collision_preserves_existing_objects(
    collision: str,
    tmp_path: Path,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    root, protocol, attempts, artifact, _bundle = _methodology_bundle_fixture(tmp_path)
    if collision == "existing":
        output = tmp_path / "existing.json"
        output.write_bytes(b"keep existing\n")
    elif collision == "bundle":
        output = root
    elif collision == "protocol":
        output = protocol
    elif collision == "attempts":
        output = attempts
    elif collision == "artifact":
        output = artifact
    elif collision == "symlink":
        output = tmp_path / "record-link.json"
        output.symlink_to(root)
    elif collision == "hardlink":
        output = tmp_path / "record-hardlink.json"
        output.hardlink_to(root)
    else:
        output = tmp_path / "dangling-record-link.json"
        output.symlink_to(tmp_path / "missing-target.json")

    before = output.lstat()
    before_payload = None if collision == "dangling_symlink" else output.read_bytes()
    before_link = output.readlink() if output.is_symlink() else None

    assert cli.main(["verify-bundle", str(root), "--output", str(output)]) == 2
    captured = capfdbinary.readouterr()
    assert captured.out == b""
    assert captured.err
    assert output.lstat().st_ino == before.st_ino
    if output.is_symlink():
        assert output.readlink() == before_link
    if before_payload is not None:
        assert output.read_bytes() == before_payload


def test_verification_output_escapes_content_and_handles_unusual_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, protocol, _attempts, _artifact, _bundle = _methodology_bundle_fixture(tmp_path)

    assert cli.main(["verify-protocol", str(protocol)]) == 0
    protocol_output = capsys.readouterr().out
    assert "study: [bold]literal[/bold]\\nsecond line" in protocol_output
    assert "\nsecond line" not in protocol_output
    assert f"path: {json.dumps(str(protocol))[1:-1]}" in protocol_output

    assert cli.main(["verify-bundle", str(root)]) == 0
    bundle_output = capsys.readouterr().out
    assert "artifact[0].role: [bold]result[/bold]\\ncontinued" in bundle_output
    assert "\ncontinued" not in bundle_output
    assert f"path: {json.dumps(str(root))[1:-1]}" in bundle_output


def test_verify_bundle_rejects_tampered_referenced_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _protocol, _attempts, artifact, _bundle = _methodology_bundle_fixture(tmp_path)
    artifact.write_bytes(b"x" * len(artifact.read_bytes()))

    assert cli.main(["verify-bundle", str(root)]) == 2
    captured = capsys.readouterr()
    assert "SHA-256 mismatch" in captured.err
    assert "structurally verified" not in captured.out


def test_verify_bundle_rejects_traversal_reference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _protocol, _attempts, _artifact, bundle = _methodology_bundle_fixture(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    artifacts = bundle["artifacts"]
    assert isinstance(artifacts, list)
    first_artifact = artifacts[0]
    assert isinstance(first_artifact, dict)
    first_artifact["path"] = "../outside.json"
    write_json_atomic(root, bundle)

    assert cli.main(["verify-bundle", str(root)]) == 2
    captured = capsys.readouterr()
    assert "non-escaping" in captured.err
    assert "structurally verified" not in captured.out


def test_verify_bundle_rejects_symlink_escape(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _protocol, _attempts, artifact, _bundle = _methodology_bundle_fixture(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(outside)

    assert cli.main(["verify-bundle", str(root)]) == 2
    captured = capsys.readouterr()
    assert "escapes the bundle directory" in captured.err
    assert "structurally verified" not in captured.out


def test_verification_commands_reject_legacy_and_missing_manifests(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, protocol, _attempts, _artifact, bundle = _methodology_bundle_fixture(tmp_path)
    protocol_data: dict[str, object] = json.loads(protocol.read_text(encoding="utf-8"))
    protocol_data["schema"] = "heliostune.protocol/0"
    write_json_atomic(protocol, protocol_data)

    assert cli.main(["verify-protocol", str(protocol)]) == 2
    assert "protocol schema must be 'heliostune.protocol/1'" in capsys.readouterr().err

    bundle["schema"] = "heliostune.bundle/0"
    write_json_atomic(root, bundle)
    assert cli.main(["verify-bundle", str(root)]) == 2
    assert "bundle schema must be 'heliostune.bundle/1'" in capsys.readouterr().err

    assert cli.main(["verify-protocol", str(tmp_path / "missing.json")]) == 2
    assert "cannot resolve protocol artifact" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["verify-protocol", "verify-bundle"])
def test_verification_help_states_non_retroactive_strict_scope(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args([command, "--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "non-retroactive" in output.lower()
    assert "legacy manifests are rejected" in output.lower()


@pytest.mark.parametrize(
    ("arguments", "handler"),
    [
        (
            ["compare", "rows.jsonl", "--source", "L4", "--target", "A10"],
            cli._compare,
        ),
        (
            ["compare-multisource", "rows.jsonl", "--sources", "L4", "--target", "A10"],
            cli._compare_multisource,
        ),
        (["select-parhelion", "rows.jsonl"], cli._select_parhelion),
        (
            ["select-v3", "rows.jsonl", "--config-manifest", "configs.json"],
            cli._select_v3,
        ),
        (["report", "summary.json"], cli._report),
        (["demo"], cli._demo),
        (["inspect", "rows.jsonl"], cli._inspect),
        (["verify-catalog"], cli._verify_catalog),
    ],
)
def test_existing_command_dispatch_is_unchanged(
    arguments: list[str],
    handler: Callable[[argparse.Namespace], int],
) -> None:
    assert cli.build_parser().parse_args(arguments).handler is handler


def test_no_init_command_is_exposed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(["init"])
    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_version_is_installed_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == f"heliostune {version('heliostune')}"


def _copy_scope_templates(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = Path(__file__).resolve().parents[1]
    relative_paths = (
        Path("plugins/fusion-reference-plugin-v1.json"),
        Path("suites/gated-mlp-epilogue-v1.json"),
        Path("suites/residual-rmsnorm-v1.json"),
    )
    copied: list[Path] = []
    for relative in relative_paths:
        source = repository / "benchmarks" / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        copied.append(destination)
    return copied[0], copied[1], copied[2]


def test_verify_plugin_accepts_committed_template_and_reports_structural_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    plugin = repository / "benchmarks/plugins/fusion-reference-plugin-v1.json"

    assert cli.main(["verify-plugin", str(plugin)]) == 0

    output = capsys.readouterr().out
    assert "Plugin structurally verified" in output
    assert "plugin: fusion-reference-plugin" in output
    assert "version: 1" in output
    assert "domains: 2" in output
    assert "arms: 4" in output
    assert "suites: 2" in output
    assert "local.unprobed: 4" in output
    assert "local.available: 0" in output
    assert "remote.unprobed: 4" in output
    assert "remote.available: 0" in output
    assert "does not validate executability, correctness, or performance" in output


def test_verify_plugin_transitively_rejects_tampered_suite_digest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugin, gated_suite, _rmsnorm_suite = _copy_scope_templates(tmp_path)
    gated_suite.write_bytes(gated_suite.read_bytes() + b"\n")

    assert cli.main(["verify-plugin", str(plugin)]) == 2

    captured = capsys.readouterr()
    assert "suite digest mismatch" in captured.err
    assert "structurally verified" not in captured.out


def test_verify_plugin_rejects_escaping_suite_path_without_verified_claim(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugin, _gated_suite, _rmsnorm_suite = _copy_scope_templates(tmp_path)
    data: dict[str, object] = json.loads(plugin.read_text(encoding="utf-8"))
    suite_refs = cast(list[dict[str, object]], data["suite_refs"])
    suite_refs[0]["path"] = "../../outside.json"
    write_json_atomic(plugin, data)

    assert cli.main(["verify-plugin", str(plugin)]) == 2

    captured = capsys.readouterr()
    assert "allowed only once as the leading component" in captured.err
    assert "structurally verified" not in captured.out


def test_verify_plugin_rejects_legacy_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugin, _gated_suite, _rmsnorm_suite = _copy_scope_templates(tmp_path)
    data: dict[str, object] = json.loads(plugin.read_text(encoding="utf-8"))
    data["schema"] = "heliostune.plugin/0"
    write_json_atomic(plugin, data)

    assert cli.main(["verify-plugin", str(plugin)]) == 2
    assert "plugin schema must be 'heliostune.plugin/1'" in capsys.readouterr().err


def test_verify_plugin_output_escapes_untrusted_identifiers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugin, gated_suite, rmsnorm_suite = _copy_scope_templates(tmp_path)
    escaped_id = "[bold]literal[/bold]\nsecond line"
    data: dict[str, object] = json.loads(plugin.read_text(encoding="utf-8"))
    data["plugin_id"] = escaped_id
    for suite in (gated_suite, rmsnorm_suite):
        suite_data: dict[str, object] = json.loads(suite.read_text(encoding="utf-8"))
        suite_data["plugin_id"] = escaped_id
        write_json_atomic(suite, suite_data)
    suite_refs = cast(list[dict[str, object]], data["suite_refs"])
    for suite_ref in suite_refs:
        suite_path = (plugin.parent / cast(str, suite_ref["path"])).resolve()
        suite_ref["sha256"] = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    write_json_atomic(plugin, data)

    assert cli.main(["verify-plugin", str(plugin)]) == 0

    output = capsys.readouterr().out
    assert "plugin: [bold]literal[/bold]\\nsecond line" in output
    assert "\nsecond line" not in output
    assert f"path: {json.dumps(str(plugin))[1:-1]}" in output


def test_verify_suite_accepts_committed_gated_mlp_template_and_reports_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    suite = repository / "benchmarks/suites/gated-mlp-epilogue-v1.json"

    assert cli.main(["verify-suite", str(suite)]) == 0

    output = capsys.readouterr().out
    assert "Suite structurally verified" in output
    assert "suite: gated-mlp-epilogue-reference" in output
    assert "template: gated_mlp_epilogue.v1" in output
    assert "revision: 1" in output
    assert "domain: fused_mlp" in output
    assert "cases: 1" in output
    assert "arms: 2" in output
    assert "cells: 4" in output
    assert "numeric_contracts: 1" in output
    assert "Correctness passage and execution are not observed" in output


def test_verify_suite_accepts_committed_residual_rmsnorm_template(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    suite = repository / "benchmarks/suites/residual-rmsnorm-v1.json"

    assert cli.main(["verify-suite", str(suite)]) == 0

    output = capsys.readouterr().out
    assert "suite: residual-rmsnorm-reference" in output
    assert "template: residual_rmsnorm.v1" in output
    assert "domain: rmsnorm_residual" in output
    assert "cases: 1" in output
    assert "arms: 2" in output
    assert "cells: 4" in output


def test_verify_suite_output_escapes_untrusted_identifiers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _plugin, suite, _rmsnorm_suite = _copy_scope_templates(tmp_path)
    data: dict[str, object] = json.loads(suite.read_text(encoding="utf-8"))
    data["suite_id"] = "[bold]literal[/bold]\nsecond line"
    write_json_atomic(suite, data)

    assert cli.main(["verify-suite", str(suite)]) == 0

    output = capsys.readouterr().out
    assert "suite: [bold]literal[/bold]\\nsecond line" in output
    assert "\nsecond line" not in output
    assert f"path: {json.dumps(str(suite))[1:-1]}" in output


@pytest.mark.parametrize(
    ("command", "relative_path", "schema"),
    [
        ("verify-plugin", "plugins/fusion-reference-plugin-v1.json", "heliostune.plugin/0"),
        ("verify-suite", "suites/gated-mlp-epilogue-v1.json", "heliostune.suite/0"),
    ],
)
def test_scope_verification_rejects_legacy_and_unknown_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    relative_path: str,
    schema: str,
) -> None:
    _copy_scope_templates(tmp_path)
    artifact = tmp_path / relative_path
    data: dict[str, object] = json.loads(artifact.read_text(encoding="utf-8"))
    data["schema"] = schema
    write_json_atomic(artifact, data)

    assert cli.main([command, str(artifact)]) == 2
    legacy = capsys.readouterr()
    kind = command.removeprefix("verify-")
    assert f"{kind} schema must be 'heliostune.{kind}/1'" in legacy.err
    assert "structurally verified" not in legacy.out

    data["schema"] = f"heliostune.{kind}/1"
    data["unknown"] = True
    write_json_atomic(artifact, data)
    assert cli.main([command, str(artifact)]) == 2
    unknown = capsys.readouterr()
    assert "has unknown fields ['unknown']" in unknown.err
    assert "structurally verified" not in unknown.out


def test_list_scope_reports_complete_schema_vocabularies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["list-scope"]) == 0

    output = capsys.readouterr().out
    assert (
        "dtype_schema_vocabulary: fp32,tf32,fp16,bf16,fp8_e4m3fn,fp8_e5m2,int8,int4,uint4"
    ) in output
    assert (
        "domain_schema_vocabulary: "
        "dense_gemm,fused_mlp,rmsnorm_residual,attention,kv_cache,moe,"
        "quantized_linear"
    ) in output


def test_list_scope_reports_narrow_templates_and_scoped_runtime_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["list-scope"]) == 0

    assert capsys.readouterr().out == (
        "Scope vocabulary and execution status\n"
        "dtype_schema_vocabulary: "
        "fp32,tf32,fp16,bf16,fp8_e4m3fn,fp8_e5m2,int8,int4,uint4\n"
        "domain_schema_vocabulary: "
        "dense_gemm,fused_mlp,rmsnorm_residual,attention,kv_cache,moe,quantized_linear\n"
        "frozen_executable_suite_templates: "
        "gated_mlp_epilogue.v1,residual_rmsnorm.v1,residual_rmsnorm_triton.v1\n"
        "template_input_storage_dtypes: fp16,bf16\n"
        "template_domains: fused_mlp,rmsnorm_residual\n"
        "suite_template_status: available only for fp16/bf16 input/storage in "
        "fused_mlp,rmsnorm_residual\n"
        "generic_local_runtime_backend: implemented for the two frozen reference templates\n"
        "generic_local_runtime_requirements: "
        "torch==2.8.0,cuda,compute_capability>=8.0,native_bf16,inductor\n"
        "generic_local_runtime_gpu_validation: "
        "validated remotely on H100 for both frozen templates\n"
        "native_local_runtime_backend: implemented for residual_rmsnorm_triton.v1\n"
        "native_local_runtime_gpu_evidence: one retained remote H100 stage-gate observation\n"
        "native_local_runtime_stage_gate_status: STOP_BELOW_THRESHOLD; execution gates passed; "
        "1.1x expansion threshold not met; one completed receipt and one unresolved "
        "transport-overflow receipt\n"
        "native_local_runtime_evidence: "
        "benchmarks/results/native-rmsnorm-h100-summary.json\n"
        "generic_remote_runtime_backend: implemented for the two frozen reference templates "
        "via Modal receipt\n"
        "generic_remote_runtime_gpu_validation: two completed exploratory H100 receipts\n"
        "generic_remote_runtime_evidence: "
        "benchmarks/results/fusion-remote-exploratory-summary.json\n"
        "generic_remote_receipt_schema: heliostune.remote-receipt/1\n"
        "generic_remote_provider_physical_attempts: not_observable\n"
        "limitation: Schema verification alone does not claim execution, correctness, or "
        "performance. The two generic frozen templates have exploratory H100 receipts only. "
        "Native evidence is one exploratory remote H100 stage-gate observation; capability "
        "declarations remain unprobed; correctness gates passed only for the exact frozen "
        "observation; no performance, fusion, or publication claim is made; Modal provider "
        "restarts remain unknown.\n"
    )


@pytest.mark.parametrize(
    ("arguments", "handler"),
    [
        (["verify-plugin", "plugin.json"], cli._verify_plugin),
        (["verify-suite", "suite.json"], cli._verify_suite),
        (["list-scope"], cli._list_scope),
    ],
)
def test_scope_commands_are_registered_and_describe_structural_limitations(
    arguments: list[str],
    handler: Callable[[argparse.Namespace], int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.build_parser().parse_args(arguments).handler is handler
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args([arguments[0], "--help"])
    assert raised.value.code == 0
    help_output = " ".join(capsys.readouterr().out.lower().split())
    if arguments[0] == "verify-plugin":
        assert "structural" in help_output
        assert "legacy plugin artifacts are rejected" in help_output
        assert "does not validate execution, correctness, or performance" in help_output
    elif arguments[0] == "verify-suite":
        assert "structural" in help_output
        assert "legacy suite artifacts are rejected" in help_output
        assert "correctness passage and execution are not observed" in help_output
    else:
        assert "local/remote runtime implementation and validation status" in help_output


def _fake_local_result(
    *,
    outcome: str,
    capability: str,
    suite_id: str = "gated_mlp_epilogue.v1",
) -> object:
    from heliostune.local_executor import (
        RMSNORM_SUITE_SHA256,
        CapabilityProbe,
        LocalExecutionResult,
    )

    available = capability == "available"
    return LocalExecutionResult(
        "suite.json",
        RMSNORM_SUITE_SHA256,
        b"",
        suite_id,
        CapabilityProbe(
            available,
            () if available else ("cuda_unavailable",),
            "2.8.0" if available else None,
            "12.8" if available else None,
            None,
            0 if available else None,
            "fake CUDA device" if available else None,
            (9, 0) if available else None,
            available,
            available,
            available,
            None if available else "unavailable",
        ),
        (),
        (),
        (),
        {},
        {},
        {},
        cast(Any, outcome),
    )


def _fake_verified_local_bundle(
    root_path: Path,
    *,
    expected: int = 4,
    terminal: int = 4,
    successes: int = 4,
    failures: int = 0,
) -> SimpleNamespace:
    coverage = SimpleNamespace(
        expected_cells=expected,
        terminal_cells=terminal,
        successes=successes,
        failures=failures,
    )
    return SimpleNamespace(
        root_path=root_path,
        root_sha256="a" * 64,
        bundle=SimpleNamespace(coverage=coverage),
        limitations=VerificationLimitations(
            plugin_suite_custody="checked",
            attempt_journal_hash_chain="checked",
            attempt_reconciliation="checked",
        ),
    )


def test_run_local_suite_completed_writes_and_reports_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from heliostune import local_bundle, local_executor

    suite = tmp_path / "suite.json"
    plugin = tmp_path / "plugin.json"
    output = tmp_path / "local-output"
    result = _fake_local_result(outcome="completed", capability="available")
    calls: list[tuple[object, Path, Path]] = []

    monkeypatch.setattr(local_executor, "execute_local_suite", lambda path: result)

    def write_bundle(
        observed: object,
        *,
        plugin_path: Path,
        output_dir: Path,
    ) -> SimpleNamespace:
        calls.append((observed, plugin_path, output_dir))
        return _fake_verified_local_bundle(output_dir / "bundle.json")

    monkeypatch.setattr(local_bundle, "write_local_bundle", write_bundle)

    assert (
        cli.main(
            [
                "run-local-suite",
                str(suite),
                "--plugin",
                str(plugin),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [(result, plugin, output)]
    printed = capsys.readouterr().out
    assert "local_cuda_capability: available" in printed
    assert "Local CUDA suite result recorded" in printed
    assert "suite: gated_mlp_epilogue.v1" in printed
    assert "outcome: completed" in printed
    assert "cells.expected: 4" in printed
    assert "cells.terminal: 4" in printed
    assert "cells.successes: 4" in printed
    assert "cells.failures: 0" in printed
    assert f"bundle_root: {output / 'bundle.json'}" in printed
    assert "structural_limitation.protocol_ancestry: not_checked" in printed
    assert "structural_limitation.plugin_suite_custody: checked" in printed
    assert "structural_limitation.attempt_journal_hash_chain: checked" in printed
    assert "structural_limitation.attempt_reconciliation: checked" in printed
    assert "Bundle verification is structural only" in printed
    assert "speedup" not in printed.lower()
    assert "publication eligible" not in printed.lower()


def test_run_local_suite_dispatches_native_digest_and_type_to_native_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heliostune import (
        local_bundle,
        local_executor,
        native_fusion_bundle,
        native_fusion_executor,
    )
    from heliostune.local_executor import CapabilityProbe

    repository = Path(cli.__file__).resolve().parents[2]
    suite = repository / "benchmarks/suites/residual-rmsnorm-triton-v1.json"
    plugin = repository / "benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json"
    capability = CapabilityProbe(
        False,
        ("torch_missing",),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        "strict CPU CLI fake",
    )
    monkeypatch.setattr(
        native_fusion_executor,
        "_probe_capability",
        lambda: (capability, None, None),
    )
    result = native_fusion_executor.run_native_fusion_suite(suite)
    monkeypatch.setattr(local_executor, "execute_local_suite", lambda _path: result)
    monkeypatch.setattr(
        local_bundle,
        "write_local_bundle",
        lambda *_args, **_kwargs: pytest.fail("legacy writer received native result"),
    )
    calls: list[tuple[object, Path, Path]] = []

    def write_bundle(
        observed: object,
        *,
        plugin_path: Path,
        output_dir: Path,
    ) -> SimpleNamespace:
        calls.append((observed, plugin_path, output_dir))
        return _fake_verified_local_bundle(
            output_dir / "bundle.json",
            expected=12,
            terminal=12,
            successes=0,
            failures=12,
        )

    monkeypatch.setattr(native_fusion_bundle, "write_native_fusion_bundle", write_bundle)
    output = tmp_path / "native"
    assert (
        cli.main(
            [
                "run-local-suite",
                str(suite),
                "--plugin",
                str(plugin),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert calls == [(result, plugin, output)]


@pytest.mark.parametrize(
    "hazard",
    ("bad-plugin", "missing-plugin", "symlink-parent", "changed-source"),
)
def test_native_preflight_hazards_prevent_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hazard: str,
) -> None:
    from heliostune import local_executor, native_fusion_bundle

    repository = Path(cli.__file__).resolve().parents[2]
    suite = repository / "benchmarks/suites/residual-rmsnorm-triton-v1.json"
    plugin = repository / "benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json"
    output = tmp_path / "native"
    if hazard == "bad-plugin":
        plugin = repository / "benchmarks/plugins/fusion-reference-plugin-v1.json"
    elif hazard == "missing-plugin":
        plugin = tmp_path / "missing-plugin.json"
    elif hazard == "symlink-parent":
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        output = linked_parent / "native"
    else:
        monkeypatch.setattr(
            native_fusion_bundle,
            "_bound_executor_sources",
            lambda: (_ for _ in ()).throw(
                ArtifactError("native executor sources changed after module import")
            ),
        )

    executed = False

    def forbidden_execute(_path: Path) -> object:
        nonlocal executed
        executed = True
        raise AssertionError("native execution must not start after failed preflight")

    monkeypatch.setattr(local_executor, "execute_local_suite", forbidden_execute)
    assert (
        cli.main(
            [
                "run-local-suite",
                str(suite),
                "--plugin",
                str(plugin),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert executed is False


def test_native_source_race_after_execution_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heliostune import local_executor, native_fusion_bundle, native_fusion_executor
    from heliostune.local_executor import CapabilityProbe

    repository = Path(cli.__file__).resolve().parents[2]
    suite = repository / "benchmarks/suites/residual-rmsnorm-triton-v1.json"
    plugin = repository / "benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json"
    capability = CapabilityProbe(
        False,
        ("torch_missing",),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        "strict CPU source-race fake",
    )
    monkeypatch.setattr(
        native_fusion_executor,
        "_probe_capability",
        lambda: (capability, None, None),
    )
    result = native_fusion_executor.run_native_fusion_suite(suite)
    changed_sources = {
        "schema": "heliostune.executor-sources/1",
        "sources": [
            dict(cast(dict[str, object], item))
            for item in cast(list[object], result.executor_sources["sources"])
        ],
    }
    cast(list[dict[str, object]], changed_sources["sources"])[0]["sha256"] = "0" * 64

    def execute_then_mutate(_path: Path) -> object:
        monkeypatch.setattr(
            native_fusion_bundle,
            "_bound_executor_sources",
            lambda: changed_sources,
        )
        return result

    monkeypatch.setattr(local_executor, "execute_local_suite", execute_then_mutate)
    output = tmp_path / "native"
    assert (
        cli.main(
            [
                "run-local-suite",
                str(suite),
                "--plugin",
                str(plugin),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


@pytest.mark.parametrize(
    ("outcome", "capability", "successes", "failures"),
    [
        ("aborted", "unavailable", 0, 0),
        ("failed", "available", 3, 1),
    ],
)
def test_run_local_suite_noncompleted_still_writes_bundle_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
    capability: str,
    successes: int,
    failures: int,
) -> None:
    from heliostune import local_bundle, local_executor

    output = tmp_path / outcome
    result = _fake_local_result(outcome=outcome, capability=capability)
    written: list[object] = []
    monkeypatch.setattr(local_executor, "execute_local_suite", lambda _path: result)

    def write_bundle(
        observed: object,
        *,
        plugin_path: Path,
        output_dir: Path,
    ) -> SimpleNamespace:
        written.append(observed)
        return _fake_verified_local_bundle(
            output_dir / "bundle.json",
            expected=4,
            terminal=successes + failures,
            successes=successes,
            failures=failures,
        )

    monkeypatch.setattr(local_bundle, "write_local_bundle", write_bundle)

    assert (
        cli.main(
            [
                "run-local-suite",
                str(tmp_path / "suite.json"),
                "--plugin",
                str(tmp_path / "plugin.json"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert written == [result]
    printed = capsys.readouterr().out
    assert f"local_cuda_capability: {capability}" in printed
    assert f"outcome: {outcome}" in printed
    assert f"cells.failures: {failures}" in printed


@pytest.mark.parametrize("protected_name", ["benchmarks", "site"])
def test_run_local_suite_rejects_protected_repository_output_before_execution(
    protected_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from heliostune import local_executor

    def unexpected_run(_path: Path) -> object:
        raise AssertionError("executor ran before output protection")

    monkeypatch.setattr(local_executor, "execute_local_suite", unexpected_run)
    repository = Path(cli.__file__).resolve().parents[2]
    suite = tmp_path / "external-suite-copy.json"
    output = repository / protected_name / "local-suite-test-output"
    monkeypatch.setattr(cli, "__file__", "/wheel/site-packages/heliostune/cli.py")

    assert (
        cli.main(
            [
                "run-local-suite",
                str(suite),
                "--plugin",
                "plugin.json",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert f"inside protected repository directory {protected_name}" in captured.err
    assert not output.exists()


def test_run_local_suite_rejects_existing_nonempty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from heliostune import local_executor

    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("user data", encoding="utf-8")

    def unexpected_run(_path: Path) -> object:
        raise AssertionError("executor ran before destination protection")

    monkeypatch.setattr(local_executor, "execute_local_suite", unexpected_run)

    assert (
        cli.main(
            [
                "run-local-suite",
                "suite.json",
                "--plugin",
                "plugin.json",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "existing nonempty local suite output directory" in capsys.readouterr().err
    assert (output / "keep").read_text(encoding="utf-8") == "user data"


def test_run_local_suite_accepts_existing_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heliostune import local_bundle, local_executor

    output = tmp_path / "empty"
    output.mkdir()
    result = _fake_local_result(outcome="completed", capability="available")
    monkeypatch.setattr(local_executor, "execute_local_suite", lambda _path: result)

    def write_bundle(
        observed: object,
        *,
        plugin_path: Path,
        output_dir: Path,
    ) -> SimpleNamespace:
        assert observed is result
        assert not output_dir.exists()
        return _fake_verified_local_bundle(output_dir / "bundle.json")

    monkeypatch.setattr(local_bundle, "write_local_bundle", write_bundle)

    assert (
        cli.main(
            [
                "run-local-suite",
                "suite.json",
                "--plugin",
                "plugin.json",
                "--output",
                str(output),
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("suite_name", "plugin_name"),
    [
        ("gated-mlp-epilogue-v1.json", "fusion-reference-plugin-v1.json"),
        ("residual-rmsnorm-v1.json", "fusion-reference-plugin-v1.json"),
        ("residual-rmsnorm-triton-v1.json", "fusion-triton-rmsnorm-plugin-v1.json"),
    ],
)
def test_run_local_suite_uses_digest_family_default_plugin_only_for_committed_templates(
    suite_name: str,
    plugin_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heliostune import local_bundle, local_executor

    repository = Path(cli.__file__).resolve().parents[2]
    suite = repository / "benchmarks/suites" / suite_name
    expected_plugin = repository / "benchmarks/plugins" / plugin_name
    result = _fake_local_result(outcome="completed", capability="available")
    monkeypatch.setattr(local_executor, "execute_local_suite", lambda _path: result)
    monkeypatch.setattr(cli, "__file__", "/wheel/site-packages/heliostune/cli.py")

    def write_bundle(
        _observed: object,
        *,
        plugin_path: Path,
        output_dir: Path,
    ) -> SimpleNamespace:
        assert plugin_path == expected_plugin
        return _fake_verified_local_bundle(output_dir / "bundle.json")

    monkeypatch.setattr(local_bundle, "write_local_bundle", write_bundle)

    assert (
        cli.main(
            [
                "run-local-suite",
                str(suite),
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    copied_suite = tmp_path / suite_name
    copied_suite.write_bytes(suite.read_bytes())
    assert (
        cli.main(
            [
                "run-local-suite",
                str(copied_suite),
                "--output",
                str(tmp_path / "copied-output"),
            ]
        )
        == 2
    )


def test_run_local_suite_escapes_unusual_identifiers_and_bundle_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from heliostune import local_bundle, local_executor

    output = tmp_path / "output"
    unusual_id = "[bold]suite[/bold]\nsecond"
    result = _fake_local_result(
        outcome="completed",
        capability="available",
        suite_id=unusual_id,
    )
    monkeypatch.setattr(local_executor, "execute_local_suite", lambda _path: result)
    unusual_root = output / "[bold]literal[/bold]\nsecond"
    monkeypatch.setattr(
        local_bundle,
        "write_local_bundle",
        lambda *_args, **_kwargs: _fake_verified_local_bundle(unusual_root),
    )

    assert (
        cli.main(
            [
                "run-local-suite",
                "suite.json",
                "--plugin",
                "plugin.json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "[bold]suite[/bold]\\nsecond" in printed
    assert unusual_id not in printed
    assert "[bold]literal[/bold]\\nsecond" in printed
    assert "[bold]literal[/bold]\nsecond" not in printed


def test_run_local_suite_parser_and_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parsed = cli.build_parser().parse_args(
        [
            "run-local-suite",
            "suite.json",
            "--plugin",
            "plugin.json",
            "--output",
            "bundle-dir",
        ]
    )
    assert parsed.handler is cli._run_local_suite
    assert parsed.suite == Path("suite.json")
    assert parsed.plugin == Path("plugin.json")
    assert parsed.output == Path("bundle-dir")

    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(["run-local-suite", "--help"])
    assert raised.value.code == 0
    help_output = " ".join(capsys.readouterr().out.lower().split())
    assert "local cuda" in help_output
    assert "structurally verified exploratory evidence bundle" in help_output
    assert "--plugin plugin" in help_output
    assert "--output dir" in help_output


def test_other_commands_do_not_import_torch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("a non-execution command imported torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert cli.main(["list-scope"]) == 0
    assert "Scope vocabulary and execution status" in capsys.readouterr().out
