from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path

import pytest
import zstandard

import heliostune.artifacts as artifacts_module
from heliostune.artifacts import (
    read_json,
    read_measurements,
    write_bytes_atomic_noreplace,
    write_bytes_atomic_noreplace_at,
    write_json_atomic,
    write_measurements_atomic,
    write_text_atomic,
)
from heliostune.configs import KernelConfig, Workload
from heliostune.errors import ArtifactError, SchemaError
from heliostune.schema import HardwareProfile, Measurement


def _row() -> Measurement:
    return Measurement(
        hardware=HardwareProfile("L4", "NVIDIA L4", (8, 9), 58, 22.0),
        workload=Workload(1, 32, 32, "model", "projection", "decode-1"),
        config=KernelConfig(16, 32, 32, 4, 3),
        bank=0,
        latency_ms=1.0,
        torch_latency_ms=2.0,
        correct=True,
        latency_p20_ms=0.9,
        latency_p80_ms=1.1,
        torch_latency_p20_ms=1.9,
        torch_latency_p80_ms=2.1,
        compile_ms=3.0,
        benchmark_wall_ms=100.0,
        torch_benchmark_wall_ms=101.0,
    )


def test_measurements_round_trip_plain_and_zstd(tmp_path: Path) -> None:
    row = _row()
    plain = tmp_path / "rows.jsonl"
    compressed = tmp_path / "rows.jsonl.zst"

    write_measurements_atomic(plain, [row])
    write_measurements_atomic(compressed, [row])

    assert read_measurements(plain) == [row]
    assert read_measurements(compressed) == [row]
    frame = zstandard.get_frame_parameters(compressed.read_bytes())
    assert frame.has_checksum
    assert frame.content_size == zstandard.CONTENTSIZE_UNKNOWN


def test_zstd_output_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl.zst"
    second = tmp_path / "second.jsonl.zst"

    write_measurements_atomic(first, [_row()])
    write_measurements_atomic(second, [_row()])

    assert first.read_bytes() == second.read_bytes()


def test_failed_atomic_serialization_preserves_existing_output(tmp_path: Path) -> None:
    destination = tmp_path / "rows.jsonl"
    write_text_atomic(destination, "existing\n")
    legacy_failure = Measurement(
        hardware=_row().hardware,
        workload=_row().workload,
        config=_row().config,
        bank=0,
        latency_ms=None,
        torch_latency_ms=2.0,
        correct=False,
        error="legacy failure",
        failure_stage="legacy_unknown",
    )

    with pytest.raises(SchemaError, match="explicitly classified"):
        write_measurements_atomic(destination, [legacy_failure])

    assert destination.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_strict_json_rejects_duplicates_and_nan_with_path(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match=r"duplicate.json.*duplicate.*a"):
        read_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match=r"nonfinite.json.*non-finite"):
        read_json(nonfinite)


def test_atomic_json_is_sorted_finite_and_newline_terminated(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    write_json_atomic(destination, {"z": 1, "a": 2})

    payload = destination.read_text(encoding="utf-8")
    assert payload == json.dumps({"a": 2, "z": 1}, indent=2, sort_keys=True) + "\n"

    with pytest.raises(SchemaError, match="strict JSON"):
        write_json_atomic(destination, {"bad": float("nan")})
    assert destination.read_text(encoding="utf-8") == payload


def test_atomic_noreplace_publishes_exact_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "record.json"
    payload = b"\x00exact bytes\n\xff"

    write_bytes_atomic_noreplace(destination, payload)

    assert destination.read_bytes() == payload
    assert not list(tmp_path.glob(".heliostune-*.tmp"))


@pytest.mark.parametrize("kind", ["regular", "symlink", "dangling", "hard-link"])
def test_atomic_noreplace_preserves_existing_entry(
    tmp_path: Path,
    kind: str,
) -> None:
    destination = tmp_path / "record.json"
    source = tmp_path / "source"
    source.write_bytes(b"existing")
    if kind == "regular":
        destination.write_bytes(b"existing")
    elif kind == "symlink":
        destination.symlink_to(source)
    elif kind == "dangling":
        destination.symlink_to(tmp_path / "missing")
    else:
        destination.hardlink_to(source)
    original_lstat = destination.lstat()

    with pytest.raises(ArtifactError, match="cannot commit"):
        write_bytes_atomic_noreplace(destination, b"replacement")

    current_lstat = destination.lstat()
    assert (current_lstat.st_dev, current_lstat.st_ino) == (
        original_lstat.st_dev,
        original_lstat.st_ino,
    )
    if kind == "dangling":
        assert destination.readlink() == tmp_path / "missing"
    else:
        assert destination.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".heliostune-*.tmp"))


def test_atomic_noreplace_concurrent_destination_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record.json"
    real_link = artifacts_module._link_fd_noreplace

    def create_destination_then_link(
        source_fd: int,
        directory_fd: int,
        name: str,
    ) -> None:
        destination.write_bytes(b"race winner")
        real_link(source_fd, directory_fd, name)

    monkeypatch.setattr(
        artifacts_module,
        "_link_fd_noreplace",
        create_destination_then_link,
    )

    with pytest.raises(ArtifactError, match="cannot commit"):
        write_bytes_atomic_noreplace(destination, b"loser")

    assert destination.read_bytes() == b"race winner"
    assert not list(tmp_path.glob(".heliostune-*.tmp"))


@pytest.mark.parametrize("failure", ["write", "file-fsync", "link"])
def test_atomic_noreplace_prelink_failure_creates_no_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    destination = tmp_path / "record.json"
    if failure == "write":
        monkeypatch.setattr(
            os,
            "write",
            lambda descriptor, payload: (_ for _ in ()).throw(OSError("write failed")),
        )
    elif failure == "file-fsync":
        monkeypatch.setattr(
            os,
            "fsync",
            lambda descriptor: (_ for _ in ()).throw(OSError("fsync failed")),
        )
    else:
        monkeypatch.setattr(
            artifacts_module,
            "_link_fd_noreplace",
            lambda source_fd, directory_fd, name: (_ for _ in ()).throw(OSError("link failed")),
        )

    with pytest.raises(ArtifactError, match="cannot commit"):
        write_bytes_atomic_noreplace(destination, b"payload")

    assert not destination.exists()
    assert not list(tmp_path.glob(".heliostune-*.tmp"))


def test_atomic_noreplace_link_return_exception_is_classified_as_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record.json"
    real_link = artifacts_module._link_fd_noreplace

    def link_then_raise(
        source_fd: int,
        directory_fd: int,
        name: str,
    ) -> None:
        real_link(source_fd, directory_fd, name)
        raise OSError("interrupted after link")

    monkeypatch.setattr(artifacts_module, "_link_fd_noreplace", link_then_raise)

    with pytest.raises(
        ArtifactError,
        match="committed through the pinned parent.*requested pathname may be stale",
    ):
        write_bytes_atomic_noreplace(destination, b"payload")

    assert destination.read_bytes() == b"payload"


def test_atomic_noreplace_falls_back_to_verified_procfd_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record.json"
    real_call_linkat = artifacts_module._call_linkat
    observed_flags: list[int] = []

    def deny_empty_path_then_link_procfd(
        source_fd: int,
        source: bytes,
        directory_fd: int,
        name: str,
        flags: int,
    ) -> None:
        observed_flags.append(flags)
        if flags == artifacts_module._AT_EMPTY_PATH:
            raise OSError(errno.EPERM, "AT_EMPTY_PATH unavailable")
        real_call_linkat(source_fd, source, directory_fd, name, flags)

    monkeypatch.setattr(
        artifacts_module,
        "_call_linkat",
        deny_empty_path_then_link_procfd,
    )

    write_bytes_atomic_noreplace(destination, b"payload")

    assert observed_flags == [
        artifacts_module._AT_EMPTY_PATH,
        artifacts_module._AT_SYMLINK_FOLLOW,
    ]
    assert destination.read_bytes() == b"payload"


def test_atomic_noreplace_rejects_mismatched_procfd_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record.json"
    attacker = tmp_path / "attacker"
    attacker.write_bytes(b"attacker")
    real_stat = os.stat
    procfd_prefix = f"{artifacts_module._PROC_SELF_FD}{os.sep}"

    def substitute_procfd_identity(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        if os.fsdecode(path).startswith(procfd_prefix):
            return real_stat(attacker)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", substitute_procfd_identity)

    with pytest.raises(ArtifactError, match="procfd source identity changed"):
        write_bytes_atomic_noreplace(destination, b"trusted")

    assert not destination.exists()
    assert attacker.read_bytes() == b"attacker"


def test_atomic_noreplace_postlink_failure_preserves_complete_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record.json"
    real_fsync = os.fsync
    calls = 0

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_directory_fsync)

    with pytest.raises(
        ArtifactError,
        match="committed through the pinned parent.*requested pathname may be stale",
    ):
        write_bytes_atomic_noreplace(destination, b"payload")

    assert destination.read_bytes() == b"payload"
    assert not list(tmp_path.glob(".heliostune-*.tmp"))


def test_atomic_noreplace_parent_swap_reports_committed_stale_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    destination = parent / "record.json"
    real_link = artifacts_module._link_fd_noreplace

    def swap_parent_then_link(
        source_fd: int,
        directory_fd: int,
        name: str,
    ) -> None:
        parent.rename(moved_parent)
        parent.mkdir()
        real_link(source_fd, directory_fd, name)

    monkeypatch.setattr(
        artifacts_module,
        "_link_fd_noreplace",
        swap_parent_then_link,
    )

    with pytest.raises(
        ArtifactError,
        match="committed through the pinned parent.*requested pathname may be stale",
    ):
        write_bytes_atomic_noreplace(destination, b"payload")

    assert not destination.exists()
    assert (moved_parent / destination.name).read_bytes() == b"payload"
    assert not list(moved_parent.glob(".heliostune-*.tmp"))


def test_atomic_noreplace_at_keeps_caller_fd_and_rejects_non_directory(
    tmp_path: Path,
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    source = tmp_path / "source"
    source.write_bytes(b"source")
    source_fd = os.open(source, os.O_RDONLY)
    try:
        write_bytes_atomic_noreplace_at(directory_fd, "record.json", b"payload")
        assert stat.S_ISDIR(os.fstat(directory_fd).st_mode)
        with pytest.raises(ArtifactError, match="not a directory"):
            write_bytes_atomic_noreplace_at(source_fd, "other.json", b"payload")
        assert os.fstat(source_fd).st_size == len(b"source")
    finally:
        os.close(source_fd)
        os.close(directory_fd)

    assert (tmp_path / "record.json").read_bytes() == b"payload"
    assert not (tmp_path / "other.json").exists()


def test_atomic_noreplace_at_dual_fd_relationship_success(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    output_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    bundle_fd = os.open(bundle, os.O_RDONLY | os.O_DIRECTORY)
    identity = os.fstat(bundle_fd)
    try:
        write_bytes_atomic_noreplace_at(
            output_fd,
            "record.json",
            b"payload",
            expected_parent_path=tmp_path,
            bundle_directory_fd=bundle_fd,
            bundle_directory_name=bundle.name,
            expected_bundle_identity=(identity.st_dev, identity.st_ino),
        )
        assert stat.S_ISDIR(os.fstat(output_fd).st_mode)
        assert stat.S_ISDIR(os.fstat(bundle_fd).st_mode)
    finally:
        os.close(bundle_fd)
        os.close(output_fd)

    assert (tmp_path / "record.json").read_bytes() == b"payload"


def test_atomic_noreplace_at_rejects_bad_bundle_relationship_prelink(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    output_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    bundle_fd = os.open(bundle, os.O_RDONLY | os.O_DIRECTORY)
    identity = os.fstat(bundle_fd)
    try:
        with pytest.raises(ArtifactError, match="bundle directory relationship changed"):
            write_bytes_atomic_noreplace_at(
                output_fd,
                "record.json",
                b"payload",
                bundle_directory_fd=bundle_fd,
                bundle_directory_name=bundle.name,
                expected_bundle_identity=(identity.st_dev, identity.st_ino + 1),
            )
    finally:
        os.close(bundle_fd)
        os.close(output_fd)

    assert not (tmp_path / "record.json").exists()
    assert not list(tmp_path.glob(".heliostune-*.tmp"))


def test_atomic_noreplace_at_bundle_rebind_after_link_preserves_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    moved_bundle = tmp_path / "moved-bundle"
    output_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    bundle_fd = os.open(bundle, os.O_RDONLY | os.O_DIRECTORY)
    identity = os.fstat(bundle_fd)
    real_link = artifacts_module._link_fd_noreplace

    def rebind_bundle_then_link(
        source_fd: int,
        directory_fd: int,
        name: str,
    ) -> None:
        bundle.rename(moved_bundle)
        bundle.mkdir()
        real_link(source_fd, directory_fd, name)

    monkeypatch.setattr(
        artifacts_module,
        "_link_fd_noreplace",
        rebind_bundle_then_link,
    )
    try:
        with pytest.raises(
            ArtifactError,
            match="committed through the pinned parent.*requested pathname may be stale",
        ):
            write_bytes_atomic_noreplace_at(
                output_fd,
                "record.json",
                b"payload",
                expected_parent_path=tmp_path,
                bundle_directory_fd=bundle_fd,
                bundle_directory_name=bundle.name,
                expected_bundle_identity=(identity.st_dev, identity.st_ino),
            )
    finally:
        os.close(bundle_fd)
        os.close(output_fd)

    assert (tmp_path / "record.json").read_bytes() == b"payload"
    assert not list(tmp_path.glob(".heliostune-*.tmp"))


def test_atomic_noreplace_uses_unnamed_inode_and_accepts_long_basename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / ("a-private-" + "x" * 180 + ".json")
    decoy = tmp_path / "attacker-controlled-temp"
    real_link = artifacts_module._link_fd_noreplace

    def substitute_visible_name_then_link(
        source_fd: int,
        directory_fd: int,
        name: str,
    ) -> None:
        assert list(tmp_path.iterdir()) == []
        decoy.write_bytes(b"attacker")
        real_link(source_fd, directory_fd, name)

    monkeypatch.setattr(
        artifacts_module,
        "_link_fd_noreplace",
        substitute_visible_name_then_link,
    )

    write_bytes_atomic_noreplace(destination, b"trusted")

    assert destination.read_bytes() == b"trusted"
    assert decoy.read_bytes() == b"attacker"


def test_atomic_noreplace_requires_existing_real_parent(tmp_path: Path) -> None:
    missing_destination = tmp_path / "missing" / "record.json"
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ArtifactError, match="cannot commit"):
        write_bytes_atomic_noreplace(missing_destination, b"payload")
    with pytest.raises(ArtifactError, match="cannot commit"):
        write_bytes_atomic_noreplace(linked_parent / "record.json", b"payload")

    assert not missing_destination.parent.exists()
    assert not (real_parent / "record.json").exists()
