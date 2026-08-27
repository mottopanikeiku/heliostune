from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from importlib.metadata import version
from io import BytesIO, StringIO
from pathlib import Path
from typing import cast

import pytest
import zstandard

import heliostune.cli as cli
from heliostune.artifacts import write_json_atomic
from heliostune.errors import ArtifactError, SchemaError
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
    assert output.count(": not_checked") == 10
    assert "publication eligible" not in output.lower()
    assert "authenticated" not in output.lower()


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
    assert "escapes plugin containment root" in captured.err
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


def test_list_scope_reports_only_narrow_templates_and_unimplemented_backends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["list-scope"]) == 0

    output = capsys.readouterr().out
    assert (
        "frozen_executable_suite_templates: gated_mlp_epilogue.v1,residual_rmsnorm.v1"
    ) in output
    assert "template_input_storage_dtypes: fp16,bf16" in output
    assert "template_domains: fused_mlp,rmsnorm_residual" in output
    assert (
        "suite_template_status: available only for fp16/bf16 input/storage in "
        "fused_mlp,rmsnorm_residual"
    ) in output
    assert "generic_local_runtime_backend: unimplemented" in output
    assert "generic_remote_runtime_backend: unimplemented" in output
    assert "do not claim runtime availability, correctness, or performance" in output
    assert "runtime_backend: available" not in output


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
        assert "unimplemented generic runtime backend status" in help_output
