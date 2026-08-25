from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from heliostune.artifacts import read_json, write_json_atomic
from heliostune.collection import AttemptJournal
from heliostune.errors import ProtocolError

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts/build_parhelion_v3_validation_failure.py"
_MANIFEST = _REPO / "benchmarks/parhelion-v3-validation-failure.json"
_JOURNAL = _REPO / "benchmarks/data/parhelion-v3-pilot-failure.attempts.jsonl"
_PROTOCOL = _REPO / "benchmarks/parhelion-v3-development-protocol.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_parhelion_v3_validation_failure", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_failure_evidence_is_deterministic_terminal_and_pre_h200() -> None:
    module = _load_script()
    committed = read_json(_MANIFEST)
    assert committed["outcome"] == "pilot_failed_before_measurement_collection"
    assert committed["policy_application"] == {
        "pilot_retried": False,
        "candidate_matrix_invoked": False,
        "a100_validation_invoked": False,
        "h200_invoked": False,
        "replacement_run_permitted": False,
        "release_scope": "0.4.0 software, causal addendum, protocol, and failure evidence",
        "performance_report": "not produced",
    }
    if sys.version_info[:2] != (3, 11) or np.__version__ != "2.4.6":
        with pytest.raises(ProtocolError, match="Parhelion v3 requires"):
            module.build_manifest()
        return
    journal = AttemptJournal.load(_JOURNAL)

    assert module.build_manifest() == committed
    assert [record.status for record in journal.records] == ["spawned", "failed"]
    assert {record.call_id for record in journal.records} == {"fc-01M0V2ZWYR8GKXNC0MB32YFPFF"}


def test_runtime_gate_precedes_attempt_journal_access(tmp_path: Path) -> None:
    module = _load_script()
    protocol = read_json(_PROTOCOL)
    protocol["analysis_runtime"]["implementation"] = platform.python_implementation()
    protocol["analysis_runtime"]["python_major_minor"] = list(sys.version_info[:2])
    protocol["analysis_runtime"]["numpy"] = "0.0.0"
    changed_protocol = tmp_path / "changed-protocol.json"
    write_json_atomic(changed_protocol, protocol)

    with pytest.raises(ProtocolError, match="requires numpy 0.0.0"):
        module.build_manifest(
            protocol_path=changed_protocol,
            journal_path=tmp_path / "must-not-be-read.attempts.jsonl",
        )
