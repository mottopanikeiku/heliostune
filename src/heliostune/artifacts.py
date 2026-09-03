"""Strict decoding and durable artifact persistence."""

from __future__ import annotations

import ctypes
import errno
import io
import json
import os
import stat
import sys
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, TextIO

import zstandard

from heliostune.errors import ArtifactError, SchemaError

_AT_EMPTY_PATH = 0x1000
_AT_FDCWD = -100
_AT_SYMLINK_FOLLOW = 0x400
_PROC_SELF_FD = Path("/proc/self/fd")
_LIBC_LINKAT = getattr(ctypes.CDLL(None, use_errno=True), "linkat", None)
if _LIBC_LINKAT is not None:
    _LIBC_LINKAT.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    _LIBC_LINKAT.restype = ctypes.c_int


_NOREPLACE_SUPPORTED = (
    sys.platform.startswith("linux")
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_TMPFILE")
    and all(function in os.supports_dir_fd for function in (os.open, os.stat))
    and os.stat in os.supports_follow_symlinks
    and _LIBC_LINKAT is not None
    and _PROC_SELF_FD.is_dir()
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
    if os.name != "posix" or not _NOREPLACE_SUPPORTED or not _PROC_SELF_FD.is_dir():
        raise ArtifactError(
            "atomic no-replace artifact publication is unsupported on this platform"
        )


def _call_linkat(
    source_fd: int,
    source: bytes,
    directory_fd: int,
    name: str,
    flags: int,
) -> None:
    if _LIBC_LINKAT is None:
        raise OSError(errno.ENOSYS, "linkat is unavailable")
    ctypes.set_errno(0)
    result = _LIBC_LINKAT(
        source_fd,
        source,
        directory_fd,
        os.fsencode(name),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), name)


def _link_fd_noreplace(source_fd: int, directory_fd: int, name: str) -> None:
    proc_source = _PROC_SELF_FD / str(source_fd)
    proc_identity = os.stat(proc_source)
    descriptor_identity = os.fstat(source_fd)
    if _stat_identity(proc_identity) != _stat_identity(descriptor_identity):
        raise OSError(errno.ESTALE, "procfd source identity changed", proc_source)
    try:
        _call_linkat(source_fd, b"", directory_fd, name, _AT_EMPTY_PATH)
    except OSError as exc:
        if exc.errno not in {errno.ENOENT, errno.EPERM}:
            raise
        _call_linkat(
            _AT_FDCWD,
            os.fsencode(proc_source),
            directory_fd,
            name,
            _AT_SYMLINK_FOLLOW,
        )


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _verify_open_directory(path: Path, opened: os.stat_result) -> None:
    current = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or _stat_identity(current) != _stat_identity(opened):
        raise OSError(errno.ESTALE, "artifact parent directory changed", path)


def _validate_plain_name(name: str, *, label: str) -> None:
    if type(name) is not str:
        raise TypeError(f"{label} must be a str")
    if (
        not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or "\0" in name
    ):
        raise ArtifactError(f"{label} must be a plain file name: {name!r}")


def _verify_bundle_relationship(
    output_directory_fd: int,
    output_directory_identity: os.stat_result,
    bundle_directory_fd: int,
    bundle_directory_name: str,
    expected_bundle_identity: tuple[int, int],
) -> None:
    opened_bundle = os.fstat(bundle_directory_fd)
    named_bundle = os.stat(
        bundle_directory_name,
        dir_fd=output_directory_fd,
        follow_symlinks=False,
    )
    bundle_parent = os.stat("..", dir_fd=bundle_directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened_bundle.st_mode)
        or _stat_identity(opened_bundle) != expected_bundle_identity
        or not stat.S_ISDIR(named_bundle.st_mode)
        or _stat_identity(named_bundle) != expected_bundle_identity
        or _stat_identity(bundle_parent) != _stat_identity(output_directory_identity)
    ):
        raise OSError(
            errno.ESTALE,
            "verified bundle directory relationship changed",
            bundle_directory_name,
        )


def write_bytes_atomic_noreplace_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    expected_parent_path: str | Path | None = None,
    bundle_directory_fd: int | None = None,
    bundle_directory_name: str | None = None,
    expected_bundle_identity: tuple[int, int] | None = None,
) -> None:
    """Durably publish bytes relative to a caller-owned directory descriptor."""
    _require_noreplace_support()
    if type(directory_fd) is not int:
        raise TypeError("directory_fd must be an int")
    _validate_plain_name(name, label="artifact name")
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")

    bundle_arguments = (
        bundle_directory_fd,
        bundle_directory_name,
        expected_bundle_identity,
    )
    if any(value is None for value in bundle_arguments) and not all(
        value is None for value in bundle_arguments
    ):
        raise TypeError(
            "bundle_directory_fd, bundle_directory_name, and "
            "expected_bundle_identity must be supplied together"
        )
    if bundle_directory_fd is not None:
        if type(bundle_directory_fd) is not int:
            raise TypeError("bundle_directory_fd must be an int")
        assert bundle_directory_name is not None
        _validate_plain_name(bundle_directory_name, label="bundle directory name")
        assert expected_bundle_identity is not None
        if (
            type(expected_bundle_identity) is not tuple
            or len(expected_bundle_identity) != 2
            or any(type(component) is not int for component in expected_bundle_identity)
        ):
            raise TypeError("expected_bundle_identity must be a (device, inode) tuple")

    expected_parent = None if expected_parent_path is None else Path(expected_parent_path)
    destination = Path(name) if expected_parent is None else expected_parent / name
    owned_directory_fd: int | None = None
    owned_bundle_directory_fd: int | None = None
    temporary_descriptor: int | None = None
    linked = False
    commit_synced = False
    try:
        owned_directory_fd = os.dup(directory_fd)
        if bundle_directory_fd is not None:
            owned_bundle_directory_fd = os.dup(bundle_directory_fd)
        opened_parent = os.fstat(owned_directory_fd)
        if not stat.S_ISDIR(opened_parent.st_mode):
            raise OSError(errno.ENOTDIR, "artifact descriptor is not a directory")
        if expected_parent is not None:
            _verify_open_directory(expected_parent, opened_parent)

        temporary_descriptor = os.open(
            ".",
            os.O_WRONLY | os.O_TMPFILE | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=owned_directory_fd,
        )
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(temporary_descriptor, view[offset:])
            if written <= 0:
                raise OSError(errno.EIO, "artifact write made no progress")
            offset += written
        os.fsync(temporary_descriptor)

        if expected_parent is not None:
            _verify_open_directory(expected_parent, opened_parent)
        if owned_bundle_directory_fd is not None:
            assert bundle_directory_name is not None
            assert expected_bundle_identity is not None
            _verify_bundle_relationship(
                owned_directory_fd,
                opened_parent,
                owned_bundle_directory_fd,
                bundle_directory_name,
                expected_bundle_identity,
            )
        _link_fd_noreplace(temporary_descriptor, owned_directory_fd, name)
        linked = True
        if owned_bundle_directory_fd is not None:
            assert bundle_directory_name is not None
            assert expected_bundle_identity is not None
            _verify_bundle_relationship(
                owned_directory_fd,
                opened_parent,
                owned_bundle_directory_fd,
                bundle_directory_name,
                expected_bundle_identity,
            )
        if expected_parent is not None:
            _verify_open_directory(expected_parent, opened_parent)
        os.fsync(owned_directory_fd)
        commit_synced = True
    except BaseException as exc:
        if not linked and temporary_descriptor is not None and owned_directory_fd is not None:
            with suppress(OSError):
                published = os.stat(
                    name,
                    dir_fd=owned_directory_fd,
                    follow_symlinks=False,
                )
                linked = _stat_identity(published) == _stat_identity(os.fstat(temporary_descriptor))
        recovery_synced = False
        if linked and owned_directory_fd is not None:
            try:
                os.fsync(owned_directory_fd)
                recovery_synced = True
            except OSError:
                pass
        if linked:
            if commit_synced:
                durability = "the directory entry was synchronized"
            elif recovery_synced:
                durability = "the directory entry was synchronized during recovery"
            else:
                durability = "directory-entry durability is ambiguous"
            raise ArtifactError(
                f"record {destination} was committed through the pinned parent; "
                f"the requested pathname may be stale; {durability}; "
                f"publication did not finish: {exc}"
            ) from exc
        if isinstance(exc, OSError):
            raise ArtifactError(f"cannot commit artifact {destination}: {exc}") from exc
        raise
    finally:
        if temporary_descriptor is not None:
            with suppress(OSError):
                os.close(temporary_descriptor)
        if owned_bundle_directory_fd is not None:
            with suppress(OSError):
                os.close(owned_bundle_directory_fd)
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
    _validate_plain_name(name, label="artifact name")

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
