"""Build and verify the one reproducible wheel installed by Modal images."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

from heliostune.artifacts import write_bytes_atomic

_REPO = Path(__file__).resolve().parents[1]
_OUTPUT = _REPO / "artifacts/modal-wheel"
_PACKAGE = _REPO / "src/heliostune"


def _run(*arguments: str, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_entries() -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for path in sorted(_PACKAGE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            entries[f"heliostune/{path.relative_to(_PACKAGE).as_posix()}"] = path.read_bytes()
    return entries


def _source_digest(entries: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(entries.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _one_wheel(directory: Path) -> Path:
    wheels = tuple(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in {directory}, found {len(wheels)}")
    return wheels[0]


def _verify_wheel(wheel: Path, expected_version: str) -> str:
    expected_sources = _source_entries()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_names = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
        if len(metadata_names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA file")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if metadata["Name"] != "heliostune" or metadata["Version"] != expected_version:
            raise RuntimeError(
                f"wheel metadata is {metadata['Name']} {metadata['Version']}, "
                f"expected heliostune {expected_version}"
            )
        packaged_sources = {
            name: archive.read(name)
            for name in names
            if name.startswith("heliostune/") and not name.endswith("/")
        }
    if packaged_sources != expected_sources:
        missing = sorted(set(expected_sources) - set(packaged_sources))
        extra = sorted(set(packaged_sources) - set(expected_sources))
        changed = sorted(
            name
            for name in set(expected_sources) & set(packaged_sources)
            if expected_sources[name] != packaged_sources[name]
        )
        raise RuntimeError(
            f"wheel package content differs from source: missing={missing}, "
            f"extra={extra}, changed={changed}"
        )
    return _source_digest(packaged_sources)


def main() -> int:
    uv_version = _run("uv", "--version").split()[1]
    if uv_version != "0.12.5":
        raise SystemExit(f"Modal wheel build requires uv 0.12.5, got {uv_version}")
    if _run("git", "status", "--porcelain"):
        raise SystemExit("Modal wheel build requires a clean Git HEAD")
    head = _run("git", "rev-parse", "HEAD")
    source_date_epoch = _run("git", "show", "-s", "--format=%at", "HEAD")
    project = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    expected_version = str(project["project"]["version"])
    environment = os.environ.copy()
    environment.update(
        {
            "SOURCE_DATE_EPOCH": source_date_epoch,
            "PYTHONHASHSEED": "0",
        }
    )

    shutil.rmtree(_OUTPUT, ignore_errors=True)
    _OUTPUT.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="heliostune-wheel-") as temporary:
        root = Path(temporary)
        builds = (root / "first", root / "second")
        for output in builds:
            output.mkdir()
            _run(
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                str(output),
                env=environment,
            )
        first = _one_wheel(builds[0])
        second = _one_wheel(builds[1])
        first_payload = first.read_bytes()
        if first_payload != second.read_bytes():
            raise RuntimeError("the two clean wheel builds are not byte-identical")
        expected_name = f"heliostune-{expected_version}-py3-none-any.whl"
        if first.name != expected_name:
            raise RuntimeError(f"wheel is named {first.name!r}, expected {expected_name!r}")
        source_sha256 = _verify_wheel(first, expected_version)
        destination = _OUTPUT / first.name
        write_bytes_atomic(destination, first_payload)

    print(f"head={head}")
    print(f"wheel={destination}")
    print(f"wheel_sha256={_sha256(destination)}")
    print(f"source_sha256={source_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
