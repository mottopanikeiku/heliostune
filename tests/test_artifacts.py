from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import zstandard

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
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


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

    with pytest.raises(ArtifactError):
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
    assert source.read_bytes() == b"existing"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_atomic_noreplace_completes_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record.json"
    real_write = os.write

    def short_write(descriptor: int, payload: bytes | memoryview) -> int:
        return real_write(descriptor, payload[:2])

    monkeypatch.setattr(os, "write", short_write)

    write_bytes_atomic_noreplace(destination, b"complete payload")

    assert destination.read_bytes() == b"complete payload"


def test_atomic_noreplace_at_keeps_caller_directory_open(tmp_path: Path) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        write_bytes_atomic_noreplace_at(directory_fd, "record.json", b"payload")

        assert stat.S_ISDIR(os.fstat(directory_fd).st_mode)
        assert (tmp_path / "record.json").read_bytes() == b"payload"
    finally:
        os.close(directory_fd)


def test_atomic_noreplace_at_rejects_non_directory_without_closing_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"source")
    source_fd = os.open(source, os.O_RDONLY)
    try:
        with pytest.raises(ArtifactError, match="not a directory"):
            write_bytes_atomic_noreplace_at(source_fd, "record.json", b"payload")

        assert os.fstat(source_fd).st_size == len(b"source")
        assert not (tmp_path / "record.json").exists()
    finally:
        os.close(source_fd)


@pytest.mark.parametrize("name", ["", ".", "..", "nested/record.json", "../record.json"])
def test_atomic_noreplace_at_rejects_non_plain_names(
    tmp_path: Path,
    name: str,
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ArtifactError, match="plain file name"):
            write_bytes_atomic_noreplace_at(directory_fd, name, b"payload")
    finally:
        os.close(directory_fd)

    assert list(tmp_path.iterdir()) == []


def test_atomic_noreplace_at_detects_expected_parent_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    real_link = os.link

    def swap_parent_then_link(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        parent.rename(moved_parent)
        parent.mkdir()
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", swap_parent_then_link)
    try:
        with pytest.raises(ArtifactError, match="parent directory changed"):
            write_bytes_atomic_noreplace_at(
                directory_fd,
                "record.json",
                b"payload",
                expected_parent_path=parent,
            )

        assert stat.S_ISDIR(os.fstat(directory_fd).st_mode)
    finally:
        os.close(directory_fd)

    assert not (parent / "record.json").exists()
    assert not (moved_parent / "record.json").exists()
    assert not list(moved_parent.glob(".record.json.*.tmp"))


def test_atomic_noreplace_at_concurrent_destination_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    destination = tmp_path / "record.json"
    real_link = os.link

    def create_destination_then_link(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        destination.write_bytes(b"race winner")
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", create_destination_then_link)
    try:
        with pytest.raises(ArtifactError):
            write_bytes_atomic_noreplace_at(directory_fd, destination.name, b"loser")

        assert stat.S_ISDIR(os.fstat(directory_fd).st_mode)
    finally:
        os.close(directory_fd)

    assert destination.read_bytes() == b"race winner"
    assert not list(tmp_path.glob(".record.json.*.tmp"))


def test_atomic_noreplace_concurrent_destination_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record.json"
    real_link = os.link

    def create_destination_then_link(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        destination.write_bytes(b"race winner")
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", create_destination_then_link)

    with pytest.raises(ArtifactError):
        write_bytes_atomic_noreplace(destination, b"loser")

    assert destination.read_bytes() == b"race winner"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


@pytest.mark.parametrize("failure", ["write", "file-fsync", "link", "directory-fsync"])
def test_atomic_noreplace_failure_leaves_no_output_or_temp(
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
    elif failure == "link":
        monkeypatch.setattr(
            os,
            "link",
            lambda source, target, **kwargs: (_ for _ in ()).throw(OSError("link failed")),
        )
    else:
        real_fsync = os.fsync
        calls = 0

        def fail_second_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("directory fsync failed")
            real_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", fail_second_fsync)

    with pytest.raises(ArtifactError):
        write_bytes_atomic_noreplace(destination, b"payload")

    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_atomic_noreplace_parent_swap_is_detected_and_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    destination = parent / "record.json"
    real_link = os.link

    def swap_parent_then_link(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        parent.rename(moved_parent)
        parent.mkdir()
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", swap_parent_then_link)

    with pytest.raises(ArtifactError, match="parent directory changed"):
        write_bytes_atomic_noreplace(destination, b"payload")

    assert not destination.exists()
    assert not (moved_parent / destination.name).exists()
    assert not list(moved_parent.glob(f".{destination.name}.*.tmp"))


def test_atomic_noreplace_requires_existing_non_symlink_parent(
    tmp_path: Path,
) -> None:
    missing_destination = tmp_path / "missing" / "record.json"
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    symlink_parent = tmp_path / "linked"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ArtifactError):
        write_bytes_atomic_noreplace(missing_destination, b"payload")
    with pytest.raises(ArtifactError):
        write_bytes_atomic_noreplace(symlink_parent / "record.json", b"payload")

    assert not missing_destination.parent.exists()
    assert not (real_parent / "record.json").exists()


def test_atomic_noreplace_requires_exact_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "record.json"

    with pytest.raises(TypeError, match="payload must be bytes"):
        write_bytes_atomic_noreplace(destination, bytearray(b"payload"))  # type: ignore[arg-type]

    assert not destination.exists()
