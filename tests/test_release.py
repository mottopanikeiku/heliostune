from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts/check_release_tag.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_release_tag", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_tag_must_exactly_match_installed_metadata() -> None:
    module = _load_script()
    expected = importlib.metadata.version("heliostune")
    assert module.check_tag(f"v{expected}")
    assert not module.check_tag(expected)
    assert not module.check_tag("v9.9.9")


def test_bad_release_tag_exits_two_without_git_side_effects() -> None:
    before = subprocess.run(
        ["git", "tag", "--list"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "v9.9.9"],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    after = subprocess.run(
        ["git", "tag", "--list"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert f"must be exactly v{importlib.metadata.version('heliostune')}" in completed.stderr
    assert after == before
