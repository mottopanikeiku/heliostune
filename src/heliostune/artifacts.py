"""Strict decoding and durable artifact persistence."""

from __future__ import annotations

import errno
import io
import json
import os
import stat
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, TextIO

import zstandard

from heliostune.errors import ArtifactError, SchemaError

_NOREPLACE_SUPPORTED = (
    os.name == "posix"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and all(function in os.supports_dir_fd for function in (os.open, os.link, os.stat, os.unlink))
    and os.link in os.supports_follow_symlinks
    and os.stat in os.supports_follow_symlinks
)

if TYPE_CHECKING:
    from heliostune.schema import Measurement


def _location(source: str | Path, line_number: int | None = None) -> str:
    location = str(source)
    return location if line_number is None else f"{location}:{line_number}"


def _reject_constant(value: str) -> object:
    raise SchemaError(f"non-finite JSON constant {value!r} is not permitted")


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def strict_json_loads(
    payload: str,
    *,
    source: str | Path = "<json>",
    line_number: int | None = None,
) -> object:
    location = _location(source, line_number)
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{location}: invalid JSON at column {exc.colno}: {exc.msg}") from exc
    except SchemaError as exc:
        raise SchemaError(f"{location}: {exc}") from exc


def strict_json_dumps(value: object, *, compact: bool = False) -> str:
    try:
        if compact:
            return json.dumps(value, allow_nan=False, separators=(",", ":"))
        return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"value is not strict JSON: {exc}") from exc


def read_json(path: str | Path) -> object:
    source = Path(path)
    try:
        payload = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {source}: {exc}") from exc
    return strict_json_loads(payload, source=source)


def _is_zstd(path: Path) -> bool:
    return path.name.endswith(".zst")


def read_measurements(path: str | Path) -> list[Measurement]:
    from heliostune.schema import read_jsonl

    source = Path(path)
    try:
        with source.open("rb") as raw:
            if _is_zstd(source):
                with (
                    zstandard.ZstdDecompressor().stream_reader(raw) as decoded,
                    io.TextIOWrapper(decoded, encoding="utf-8", newline="") as text,
                ):
                    return read_jsonl(text, source_name=source)
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                return read_jsonl(text, source_name=source)
    except (SchemaError, ArtifactError):
        raise
    except (OSError, UnicodeError, zstandard.ZstdError) as exc:
        raise ArtifactError(f"cannot read measurement artifact {source}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _atomic_binary(destination: Path) -> Iterator[BinaryIO]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    stream = os.fdopen(descriptor, "wb")
    try:
        with stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        stream.close()
        temporary.unlink(missing_ok=True)
        raise


def write_bytes_atomic(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    try:
        with _atomic_binary(destination) as stream:
            stream.write(payload)
    except OSError as exc:
        raise ArtifactError(f"cannot commit artifact {destination}: {exc}") from exc


def _require_noreplace_support() -> None:
    if os.name != "posix" or not _NOREPLACE_SUPPORTED:
        raise ArtifactError(
            "atomic no-replace artifact publication is unsupported on this platform"
        )


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _verify_open_directory(path: Path, opened: os.stat_result) -> None:
    current = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or not _same_file_identity(opened, current):
        raise OSError(errno.ESTALE, "artifact parent directory changed", path)


def _unlink_owned_entry(
    name: str,
    *,
    directory_descriptor: int,
    identity: os.stat_result,
) -> None:
    try:
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if _same_file_identity(identity, current):
        os.unlink(name, dir_fd=directory_descriptor)


def _validate_noreplace_inputs(name: str, payload: bytes) -> None:
    if type(name) is not str:
        raise TypeError("name must be a str")
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    if (
        not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or "\0" in name
    ):
        raise ArtifactError(f"artifact name must be a plain file name: {name!r}")


def write_bytes_atomic_noreplace_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    expected_parent_path: str | Path | None = None,
) -> None:
    """Durably publish bytes relative to a caller-owned directory descriptor."""
    _require_noreplace_support()
    if type(directory_fd) is not int:
        raise TypeError("directory_fd must be an int")
    _validate_noreplace_inputs(name, payload)
    expected_parent = None if expected_parent_path is None else Path(expected_parent_path)
    destination = Path(name) if expected_parent is None else expected_parent / name

    owned_directory_fd: int | None = None
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    temporary_identity: os.stat_result | None = None
    linked = False
    try:
        owned_directory_fd = os.dup(directory_fd)
        opened_parent = os.fstat(owned_directory_fd)
        if not stat.S_ISDIR(opened_parent.st_mode):
            raise OSError(errno.ENOTDIR, "artifact descriptor is not a directory")
        if expected_parent is not None:
            _verify_open_directory(expected_parent, opened_parent)

        temporary_flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(128):
            candidate = f".{name}.{os.urandom(16).hex()}.tmp"
            try:
                temporary_descriptor = os.open(
                    candidate,
                    temporary_flags,
                    0o600,
                    dir_fd=owned_directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        else:
            raise OSError(errno.EEXIST, "cannot allocate temporary artifact")
        assert temporary_descriptor is not None
        assert temporary_name is not None

        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(temporary_descriptor, view[offset:])
            if written <= 0:
                raise OSError(errno.EIO, "artifact write made no progress")
            offset += written
        os.fsync(temporary_descriptor)
        temporary_identity = os.fstat(temporary_descriptor)

        if expected_parent is not None:
            _verify_open_directory(expected_parent, opened_parent)
        os.link(
            temporary_name,
            name,
            src_dir_fd=owned_directory_fd,
            dst_dir_fd=owned_directory_fd,
            follow_symlinks=False,
        )
        linked = True
        if expected_parent is not None:
            _verify_open_directory(expected_parent, opened_parent)
        os.unlink(temporary_name, dir_fd=owned_directory_fd)
        temporary_name = None
        os.fsync(owned_directory_fd)
    except BaseException as exc:
        if linked and owned_directory_fd is not None and temporary_identity is not None:
            with suppress(OSError):
                _unlink_owned_entry(
                    name,
                    directory_descriptor=owned_directory_fd,
                    identity=temporary_identity,
                )
        if temporary_name is not None and owned_directory_fd is not None:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=owned_directory_fd)
        if owned_directory_fd is not None:
            with suppress(OSError):
                os.fsync(owned_directory_fd)
        if isinstance(exc, OSError):
            raise ArtifactError(f"cannot commit artifact {destination}: {exc}") from exc
        raise
    finally:
        if temporary_descriptor is not None:
            with suppress(OSError):
                os.close(temporary_descriptor)
        if owned_directory_fd is not None:
            with suppress(OSError):
                os.close(owned_directory_fd)


def write_bytes_atomic_noreplace(path: str | Path, payload: bytes) -> None:
    """Durably publish bytes without replacing an existing directory entry."""
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    _require_noreplace_support()

    destination = Path(path)
    parent = destination.parent
    name = destination.name
    _validate_noreplace_inputs(name, payload)

    directory_fd: int | None = None
    try:
        directory_fd = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        write_bytes_atomic_noreplace_at(
            directory_fd,
            name,
            payload,
            expected_parent_path=parent,
        )
    except OSError as exc:
        raise ArtifactError(f"cannot commit artifact {destination}: {exc}") from exc
    finally:
        if directory_fd is not None:
            with suppress(OSError):
                os.close(directory_fd)


def write_text_atomic(path: str | Path, payload: str) -> None:
    try:
        encoded = payload.encode("utf-8")
    except UnicodeError as exc:
        raise ArtifactError(f"cannot encode text artifact {path}: {exc}") from exc
    write_bytes_atomic(path, encoded)


def write_json_atomic(path: str | Path, value: object) -> None:
    write_text_atomic(path, strict_json_dumps(value))


def _write_jsonl_text(measurements: Iterable[Measurement], destination: TextIO) -> None:
    from heliostune.schema import write_jsonl

    write_jsonl(measurements, destination)


def write_measurements_atomic(
    path: str | Path,
    measurements: Iterable[Measurement],
) -> None:
    destination = Path(path)
    try:
        with _atomic_binary(destination) as raw:
            if _is_zstd(destination):
                compressor = zstandard.ZstdCompressor(
                    level=19,
                    threads=1,
                    write_checksum=True,
                    write_content_size=False,
                )
                with compressor.stream_writer(raw, closefd=False) as encoded:
                    compressed_text = io.TextIOWrapper(encoded, encoding="utf-8", newline="\n")
                    try:
                        _write_jsonl_text(measurements, compressed_text)
                        compressed_text.flush()
                    finally:
                        compressed_text.detach()
            else:
                plain_text = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
                try:
                    _write_jsonl_text(measurements, plain_text)
                    plain_text.flush()
                finally:
                    plain_text.detach()
    except (SchemaError, ArtifactError):
        raise
    except (OSError, UnicodeError, zstandard.ZstdError) as exc:
        raise ArtifactError(f"cannot commit measurement artifact {destination}: {exc}") from exc
