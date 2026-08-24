from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts/capture_historical_baseline.py"
_BASELINE = _REPO / "benchmarks/historical-artifact-baseline.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("capture_historical_baseline", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_capture_matches_frozen_baseline() -> None:
    module = _load_script()
    expected = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert module.capture(_REPO) == expected


def test_absent_aliases_bind_published_replacements() -> None:
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    for alias, binding in baseline["absent_freeze_aliases"].items():
        assert not (_REPO / alias).exists()
        assert binding["status"] == "not_present_at_audited_commit"
        replacement = _REPO / binding["published_replacement"]
        assert replacement.is_file()
        assert _sha256(replacement) == binding["published_replacement_sha256"]
