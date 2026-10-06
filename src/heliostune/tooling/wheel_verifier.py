"""Verify wheel integrity and exact custody of packaged HeliosTune sources."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import stat
import zipfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class VerifiedWheel:
    """Digests established directly from a structurally valid wheel."""

    wheel_sha256: str
    source_sha256: str
    package_members: tuple[str, ...]


def source_entries(source_directory: str | Path) -> dict[str, bytes]:
    """Read every source/resource file that must appear under ``heliostune/``."""
    package = Path(source_directory)
    if not package.is_dir():
        raise RuntimeError(f"HeliosTune source directory does not exist: {package}")
    entries: dict[str, bytes] = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            name = f"heliostune/{path.relative_to(package).as_posix()}"
            entries[name] = path.read_bytes()
    if not entries:
        raise RuntimeError(f"HeliosTune source directory is empty: {package}")
    return entries


def source_digest(entries: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(entries.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"wheel contains unsafe member path {name!r}")


def _record_digest(payload: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")


_REQUIRED_DIST_INFO_FILES = frozenset({"METADATA", "WHEEL", "RECORD"})
_ALLOWED_DIST_INFO_FILES = _REQUIRED_DIST_INFO_FILES | frozenset(
    {"entry_points.txt", "licenses/LICENSE"}
)
_NORMALIZED_WHEEL_VERSION = re.compile(
    r"(?:(?:0|[1-9]\d*)!)?"
    r"(?:0|[1-9]\d*)(?:\.(?:0|[1-9]\d*))*"
    r"(?:(?:a|b|rc)(?:0|[1-9]\d*))?"
    r"(?:\.post(?:0|[1-9]\d*))?"
    r"(?:\.dev(?:0|[1-9]\d*))?"
    r"(?:\+[a-z0-9]+(?:\.[a-z0-9]+)*)?"
)


def _dist_info_directory(payloads: dict[str, bytes]) -> str:
    directories = {
        name.partition("/")[0] for name in payloads if name.partition("/")[0].endswith(".dist-info")
    }
    if len(directories) != 1:
        raise RuntimeError("wheel must contain exactly one .dist-info directory")
    directory = directories.pop()
    metadata_name = f"{directory}/METADATA"
    if metadata_name not in payloads:
        raise RuntimeError("wheel .dist-info directory lacks required METADATA")
    metadata = BytesParser(policy=policy.default).parsebytes(payloads[metadata_name])
    names = metadata.get_all("Name", [])
    versions = metadata.get_all("Version", [])
    if names != ["heliostune"] or len(versions) != 1:
        raise RuntimeError("wheel METADATA must identify exactly one heliostune version")
    version = versions[0]
    if _NORMALIZED_WHEEL_VERSION.fullmatch(version) is None:
        raise RuntimeError("wheel METADATA Version is not normalized for a wheel")
    expected_directory = f"heliostune-{version}.dist-info"
    if directory != expected_directory:
        raise RuntimeError(
            f"wheel .dist-info directory is {directory!r}, expected {expected_directory!r}"
        )
    return directory


def verify_wheel_against_source(
    wheel: str | Path,
    source_directory: str | Path,
) -> VerifiedWheel:
    """Verify ZIP/RECORD integrity and exact packaged source/resource bytes.

    The adjacent build manifest is intentionally not an input: it cannot make a
    forged wheel trustworthy.
    """
    wheel_path = Path(wheel)
    expected = source_entries(source_directory)
    try:
        wheel_bytes = wheel_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read Modal wheel {wheel_path}: {exc}") from exc
    try:
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise RuntimeError("wheel contains duplicate ZIP member names")
            payloads: dict[str, bytes] = {}
            for info in infos:
                _safe_member_name(info.filename.rstrip("/") if info.is_dir() else info.filename)
                mode = info.external_attr >> 16
                if mode and stat.S_ISLNK(mode):
                    raise RuntimeError(f"wheel contains symbolic link member {info.filename!r}")
                if mode & 0o111:
                    raise RuntimeError(f"wheel contains executable member {info.filename!r}")
                if info.flag_bits & 0x1:
                    raise RuntimeError(f"wheel contains encrypted member {info.filename!r}")
                if info.is_dir():
                    raise RuntimeError(
                        f"wheel contains unexpected directory member {info.filename!r}"
                    )
                payloads[info.filename] = archive.read(info)
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"wheel ZIP integrity check failed at {corrupt!r}")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Modal wheel is not a valid ZIP archive: {wheel_path}: {exc}") from exc

    dist_info_directory = _dist_info_directory(payloads)
    record_name = f"{dist_info_directory}/RECORD"
    if record_name not in payloads:
        raise RuntimeError("wheel .dist-info directory lacks required RECORD")
    try:
        rows = list(csv.reader(io.StringIO(payloads[record_name].decode("utf-8", errors="strict"))))
    except (UnicodeError, csv.Error) as exc:
        raise RuntimeError("wheel RECORD is not strict UTF-8 CSV") from exc
    recorded: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in recorded:
            raise RuntimeError("wheel RECORD contains a malformed or duplicate row")
        _safe_member_name(row[0])
        recorded[row[0]] = (row[1], row[2])
    if set(recorded) != set(payloads):
        missing = sorted(set(payloads) - set(recorded))
        extra = sorted(set(recorded) - set(payloads))
        raise RuntimeError(f"wheel RECORD inventory differs: missing={missing}, extra={extra}")
    for name, payload in payloads.items():
        encoded_hash, encoded_size = recorded[name]
        if name == record_name:
            if encoded_hash or encoded_size:
                raise RuntimeError("wheel RECORD self-row must have empty hash and size")
            continue
        if encoded_hash != f"sha256={_record_digest(payload)}":
            raise RuntimeError(f"wheel RECORD hash mismatch for {name!r}")
        if encoded_size != str(len(payload)):
            raise RuntimeError(f"wheel RECORD size mismatch for {name!r}")

    dist_info_files = {
        name.removeprefix(f"{dist_info_directory}/")
        for name in payloads
        if name.startswith(f"{dist_info_directory}/")
    }
    missing_metadata = sorted(_REQUIRED_DIST_INFO_FILES - dist_info_files)
    extra_metadata = sorted(dist_info_files - _ALLOWED_DIST_INFO_FILES)
    if missing_metadata or extra_metadata:
        raise RuntimeError(
            "wheel .dist-info inventory is not allowed: "
            f"missing={missing_metadata}, extra={extra_metadata}"
        )
    allowed_payloads = set(expected) | {f"{dist_info_directory}/{name}" for name in dist_info_files}
    extra_payloads = sorted(set(payloads) - allowed_payloads)
    if extra_payloads:
        raise RuntimeError(f"wheel contains payloads outside the allowlist: {extra_payloads}")

    packaged = {
        name: payload for name, payload in payloads.items() if name.startswith("heliostune/")
    }
    if packaged != expected:
        missing = sorted(set(expected) - set(packaged))
        extra = sorted(set(packaged) - set(expected))
        changed = sorted(
            name for name in set(expected) & set(packaged) if expected[name] != packaged[name]
        )
        raise RuntimeError(
            "wheel package content differs from source: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return VerifiedWheel(
        wheel_sha256=hashlib.sha256(wheel_bytes).hexdigest(),
        source_sha256=source_digest(packaged),
        package_members=tuple(sorted(packaged)),
    )
