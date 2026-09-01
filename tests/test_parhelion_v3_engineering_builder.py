from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_parhelion_v3_engineering_result.py"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_test_build_parhelion_v3_engineering_result",
        _SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed_fixture(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    result = tmp_path / "result.json"
    report = tmp_path / "report.html"
    manifest = tmp_path / "manifest.json"
    result.write_text('{\n  "value": "committed"\n}\n', encoding="utf-8")
    report.write_text("<p>committed</p>\n", encoding="utf-8")
    manifest.write_text('{\n  "role": "post-run reproducer"\n}\n', encoding="utf-8")

    monkeypatch.setattr(builder, "_OUTPUT", result)
    monkeypatch.setattr(builder, "_REPORT", report)
    monkeypatch.setattr(builder, "_DERIVATION_MANIFEST", manifest)
    monkeypatch.setattr(builder, "build_result", lambda: {"value": "committed"})
    monkeypatch.setattr(builder, "render_html", lambda _result: "<p>committed</p>\n")
    monkeypatch.setattr(
        builder,
        "build_derivation_manifest",
        lambda _result, _report: {"role": "post-run reproducer"},
    )

    def fail_write(_path: Path, _payload: bytes) -> None:
        raise AssertionError("--check attempted a write")

    monkeypatch.setattr(builder, "write_bytes_atomic", fail_write)
    return result, report, manifest


def test_check_byte_matches_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    paths = _committed_fixture(builder, monkeypatch, tmp_path)
    before = [path.read_bytes() for path in paths]

    assert builder.main(["--check"]) == 0
    assert [path.read_bytes() for path in paths] == before


def test_check_rejects_mismatch_without_mutating_committed_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    paths = _committed_fixture(builder, monkeypatch, tmp_path)
    before = [path.read_bytes() for path in paths]
    monkeypatch.setattr(builder, "build_result", lambda: {"value": "different"})

    with pytest.raises(RuntimeError, match="engineering derivation output is stale"):
        builder.main(["--check"])

    assert [path.read_bytes() for path in paths] == before
