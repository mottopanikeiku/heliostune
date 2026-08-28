from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
_MODULE = _REPO / "modal_bench.py"
_BUILDER_MODULE = _REPO / "scripts/build_modal_wheel.py"
_WHEEL_NAME = "heliostune-0.4.0-py3-none-any.whl"
_HEAD = "a" * 40
_PIP_DEPENDENCIES = [
    "numpy==2.4.6",
    "rich==14.3.4",
    "zstandard==0.25.0",
    "torch==2.8.0",
    "triton==3.4.0",
]


def _source_digest(repository: Path) -> str:
    package = repository / "src/heliostune"
    digest = hashlib.sha256()
    paths = sorted(
        item for item in package.rglob("*") if item.is_file() and "__pycache__" not in item.parts
    )
    for path in paths:
        name = f"heliostune/{path.relative_to(package).as_posix()}"
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _write_wheel_and_manifest(
    directory: Path,
    *,
    repository: Path,
    head: str = _HEAD,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / _WHEEL_NAME
    wheel.write_bytes(b"tiny-real-wheel-bytes")
    manifest = wheel.with_name(f"{wheel.name}.manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "head_commit": head,
                "source_sha256": _source_digest(repository),
                "wheel_filename": wheel.name,
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "python_version": "3.11",
                "pip_dependencies": _PIP_DEPENDENCIES,
                "build_dependencies": ["hatchling==1.32.0"],
                "build_tools": {"uv": "0.12.5", "hatchling": "1.32.0"},
                "wheel_install_args": ["--no-deps"],
            }
        ),
        encoding="utf-8",
    )
    return wheel, manifest


class _FakeImage:
    def __init__(self) -> None:
        self.operations: list[tuple[str, object]] = []

    @classmethod
    def debian_slim(cls, *, python_version: str) -> _FakeImage:
        image = cls()
        image.operations.append(("python", python_version))
        return image

    def pip_install(self, *dependencies: str) -> _FakeImage:
        self.operations.append(("pip", dependencies))
        return self

    def add_local_file(
        self,
        path: Path,
        *,
        remote_path: str,
        copy: bool,
    ) -> _FakeImage:
        self.operations.append(("file", (path, remote_path, copy)))
        return self

    def run_commands(self, command: str) -> _FakeImage:
        self.operations.append(("command", command))
        return self

    def env(self, values: dict[str, str]) -> _FakeImage:
        self.operations.append(("env", values))
        return self


class _FakeApp:
    def __init__(self, name: str) -> None:
        self.name = name

    def function(self, **_kwargs: object) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda function: function

    def local_entrypoint(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda function: function


def _fake_modal() -> ModuleType:
    module = ModuleType("modal")
    module.App = _FakeApp  # type: ignore[attr-defined]
    module.Image = _FakeImage  # type: ignore[attr-defined]
    module.FunctionCall = SimpleNamespace(from_id=lambda _value: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture(scope="module")
def modal_bench(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    wheel, manifest = _write_wheel_and_manifest(
        tmp_path_factory.mktemp("import-wheel"), repository=_REPO
    )
    previous_wheel = os.environ.get("HELIOSTUNE_MODAL_WHEEL")
    previous_manifest = os.environ.get("HELIOSTUNE_MODAL_WHEEL_MANIFEST")
    previous_modal = sys.modules.get("modal")
    real_run = subprocess.run

    def clean_git(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        output = "" if arguments[1:3] == ["status", "--porcelain"] else f"{_HEAD}\n"
        return SimpleNamespace(stdout=output)

    os.environ["HELIOSTUNE_MODAL_WHEEL"] = str(wheel)
    os.environ.pop("HELIOSTUNE_MODAL_WHEEL_MANIFEST", None)
    sys.modules["modal"] = _fake_modal()
    subprocess.run = clean_git  # type: ignore[assignment]
    try:
        spec = importlib.util.spec_from_file_location("_test_modal_bench", _MODULE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        subprocess.run = real_run  # type: ignore[assignment]
        if previous_modal is None:
            del sys.modules["modal"]
        else:
            sys.modules["modal"] = previous_modal
        if previous_wheel is None:
            del os.environ["HELIOSTUNE_MODAL_WHEEL"]
        else:
            os.environ["HELIOSTUNE_MODAL_WHEEL"] = previous_wheel
        if previous_manifest is None:
            os.environ.pop("HELIOSTUNE_MODAL_WHEEL_MANIFEST", None)
        else:
            os.environ["HELIOSTUNE_MODAL_WHEEL_MANIFEST"] = previous_manifest
    assert manifest.is_file()
    return module


def _tiny_repository(tmp_path: Path) -> Path:
    package = tmp_path / "src/heliostune"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    return tmp_path


def test_wheel_builder_sanitizes_index_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("_test_build_modal_wheel", _BUILDER_MODULE)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://attacker.invalid/extra")
    monkeypatch.setenv("UV_INDEX_PRIVATE_URL", "https://attacker.invalid/private")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://attacker.invalid/default")
    monkeypatch.setenv("SAFE_SENTINEL", "kept")
    environment = builder._sanitized_build_environment()
    assert environment["SAFE_SENTINEL"] == "kept"
    assert environment["UV_NO_CONFIG"] == "1"
    assert not any(key.startswith("PIP_") for key in environment)
    assert not any(key.startswith("UV_INDEX") for key in environment)
    assert "UV_DEFAULT_INDEX" not in environment


def test_import_binds_app_and_image_without_modal_install(modal_bench: ModuleType) -> None:
    assert modal_bench.app.name == "heliostune-bench"
    assert modal_bench.image is not None


def test_manifest_binds_clean_head_source_and_wheel(
    modal_bench: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = _tiny_repository(tmp_path / "repo")
    wheel, manifest = _write_wheel_and_manifest(tmp_path / "wheel", repository=repository)
    monkeypatch.setattr(modal_bench, "_git_head", lambda root: _HEAD)
    assert modal_bench.validate_wheel_manifest(wheel, repository=repository) == manifest


def test_manifest_rejects_stale_head(
    modal_bench: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = _tiny_repository(tmp_path / "repo")
    wheel, _ = _write_wheel_and_manifest(tmp_path / "wheel", repository=repository)
    monkeypatch.setattr(modal_bench, "_git_head", lambda root: "b" * 40)
    with pytest.raises(RuntimeError, match="current HEAD"):
        modal_bench.validate_wheel_manifest(wheel, repository=repository)


def test_manifest_rejects_changed_wheel_bytes(
    modal_bench: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = _tiny_repository(tmp_path / "repo")
    wheel, _ = _write_wheel_and_manifest(tmp_path / "wheel", repository=repository)
    wheel.write_bytes(b"changed")
    monkeypatch.setattr(modal_bench, "_git_head", lambda root: _HEAD)
    with pytest.raises(RuntimeError, match="wheel_sha256"):
        modal_bench.validate_wheel_manifest(wheel, repository=repository)


def test_remote_manifest_checks_bytes_without_git(
    modal_bench: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = _tiny_repository(tmp_path / "repo")
    wheel, manifest = _write_wheel_and_manifest(tmp_path / "remote", repository=repository)
    monkeypatch.setenv("HELIOSTUNE_MODAL_WHEEL_MANIFEST", str(manifest))

    def forbidden_git(_root: Path) -> str:
        raise AssertionError("remote validation must not run git")

    monkeypatch.setattr(modal_bench, "_git_head", forbidden_git)
    assert modal_bench.validate_wheel_manifest(wheel, remote=True) == manifest


def test_build_image_copies_manifest_and_installs_without_dependencies(
    modal_bench: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / _WHEEL_NAME
    wheel.write_bytes(b"wheel")
    manifest = tmp_path / f"{_WHEEL_NAME}.manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(modal_bench, "validate_wheel_manifest", lambda *_args, **_kwargs: manifest)
    image = modal_bench.build_image(wheel)
    operations = image.operations
    assert ("pip", tuple(_PIP_DEPENDENCIES)) in operations
    assert ("command", f"python -m pip install --no-deps /root/{_WHEEL_NAME}") in operations
    files = [value for operation, value in operations if operation == "file"]
    assert files == [
        (wheel, f"/root/{_WHEEL_NAME}", True),
        (manifest, f"/root/{_WHEEL_NAME}.manifest.json", True),
    ]
    assert (
        "env",
        {
            "HELIOSTUNE_MODAL_WHEEL": f"/root/{_WHEEL_NAME}",
            "HELIOSTUNE_MODAL_WHEEL_MANIFEST": f"/root/{_WHEEL_NAME}.manifest.json",
        },
    ) in operations


@pytest.mark.parametrize("value", ["/absolute.json", "../escape.json", "x/../escape.json"])
def test_output_rejects_absolute_and_parent_paths(
    modal_bench: ModuleType,
    value: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        modal_bench._resolved_output(value, repository=tmp_path)


def test_output_rejects_direct_and_symlinked_benchmark_containment(
    modal_bench: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(benchmarks, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="benchmarks"):
        modal_bench._resolved_output("benchmarks/result.json", repository=tmp_path)
    with pytest.raises(ValueError, match="benchmarks"):
        modal_bench._resolved_output("linked/result.json", repository=tmp_path)


def test_git_commands_use_repository_cwd(
    modal_bench: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    def run(arguments: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(Path(str(kwargs["cwd"])))
        output = "" if arguments[1] == "status" else f"{_HEAD}\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(modal_bench.subprocess, "run", run)
    assert modal_bench._git_head(tmp_path) == _HEAD
    assert calls == [tmp_path, tmp_path]
