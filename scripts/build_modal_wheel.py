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

from heliostune.artifacts import strict_json_dumps, write_bytes_atomic

_REPO = Path(__file__).resolve().parents[1]
_OUTPUT = _REPO / "artifacts/modal-wheel"
_PACKAGE = _REPO / "src/heliostune"
_MANIFEST_SCHEMA_VERSION = 1
_PYTHON_VERSION = "3.11"
_UV_VERSION = "0.12.5"
_HATCHLING_VERSION = "1.32.0"
_BUILD_DEPENDENCIES = (f"hatchling=={_HATCHLING_VERSION}",)
_PIP_DEPENDENCIES = (
    "numpy==2.4.6",
    "rich==14.3.4",
    "zstandard==0.25.0",
    "torch==2.8.0",
    "triton==3.4.0",
)


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


def _sanitized_build_environment() -> dict[str, str]:
    """Remove ambient package-index controls before resolving build dependencies."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("PIP_", "UV_"))
    }
    environment["UV_NO_CONFIG"] = "1"
    return environment


def _manifest_path(wheel: Path) -> Path:
    return wheel.with_name(f"{wheel.name}.manifest.json")


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
    environment = _sanitized_build_environment()
    uv_version = _run("uv", "--version", env=environment).split()[1]
    if uv_version != _UV_VERSION:
        raise SystemExit(f"Modal wheel build requires uv {_UV_VERSION}, got {uv_version}")
    if _run("git", "status", "--porcelain", env=environment):
        raise SystemExit("Modal wheel build requires a clean Git HEAD")
    head = _run("git", "rev-parse", "HEAD", env=environment)
    source_date_epoch = _run("git", "show", "-s", "--format=%at", "HEAD", env=environment)
    project = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    expected_version = str(project["project"]["version"])
    build_constraints = tuple(project["tool"]["uv"]["build-constraint-dependencies"])
    if build_constraints != _BUILD_DEPENDENCIES:
        raise SystemExit(
            f"Modal wheel build dependencies must be {_BUILD_DEPENDENCIES}, got {build_constraints}"
        )
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
        build_constraint = root / "build-constraints.txt"
        build_constraint.write_text("\n".join(_BUILD_DEPENDENCIES) + "\n", encoding="utf-8")
        builds = (root / "first", root / "second")
        for output in builds:
            output.mkdir()
            _run(
                "uv",
                "build",
                "--wheel",
                "--build-constraint",
                str(build_constraint),
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

    wheel_sha256 = _sha256(destination)
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "head_commit": head,
        "source_sha256": source_sha256,
        "wheel_filename": destination.name,
        "wheel_sha256": wheel_sha256,
        "python_version": _PYTHON_VERSION,
        "pip_dependencies": list(_PIP_DEPENDENCIES),
        "build_dependencies": list(_BUILD_DEPENDENCIES),
        "build_tools": {"uv": _UV_VERSION, "hatchling": _HATCHLING_VERSION},
        "wheel_install_args": ["--no-deps"],
    }
    manifest_path = _manifest_path(destination)
    write_bytes_atomic(manifest_path, strict_json_dumps(manifest).encode())

    print(f"head={head}")
    print(f"wheel={destination}")
    print(f"wheel_sha256={wheel_sha256}")
    print(f"source_sha256={source_sha256}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
