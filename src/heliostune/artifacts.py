"""Strict decoding and durable artifact persistence."""

from __future__ import annotations

import io
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, TextIO

import zstandard

from heliostune.errors import ArtifactError, SchemaError

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
