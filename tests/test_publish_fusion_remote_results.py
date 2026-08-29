from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from heliostune.errors import ArtifactError, SchemaError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/publish_fusion_remote_results.py"


def _load_publisher() -> ModuleType:
    name = "_test_publish_fusion_remote_results"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PUBLISHER = _load_publisher()
Mutation = Callable[[dict[str, Any]], None]


def _completed(raw: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], raw["attempts"][2])


def _rebind_receipt(attempt: dict[str, Any]) -> None:
    receipt_payload = PUBLISHER._json_bytes(attempt["receipt"])
    attempt["source_files"]["receipt_root"] = PUBLISHER._binding(
        f"{attempt['pointer_value']}/receipt.json", receipt_payload
    )


def _mutate_receipt(raw: dict[str, Any], mutation: Callable[[dict[str, Any]], None]) -> None:
    attempt = _completed(raw)
    mutation(attempt["receipt"])
    _rebind_receipt(attempt)


def _artifact_path(raw: dict[str, Any]) -> None:
    _mutate_receipt(raw, lambda receipt: receipt["intent"].__setitem__("path", "other.json"))


def _artifact_digest(raw: dict[str, Any]) -> None:
    _mutate_receipt(raw, lambda receipt: receipt["result"].__setitem__("sha256", "0" * 64))


def _request_binding(raw: dict[str, Any]) -> None:
    _mutate_receipt(
        raw, lambda receipt: receipt["bindings"].__setitem__("request_digest", "0" * 64)
    )


def _suite_binding(raw: dict[str, Any]) -> None:
    _mutate_receipt(
        raw,
        lambda receipt: receipt["bindings"]["suite"].__setitem__(
            "logical_path", "benchmarks/suites/other.json"
        ),
    )


def _plugin_binding(raw: dict[str, Any]) -> None:
    _mutate_receipt(
        raw,
        lambda receipt: receipt["bindings"]["plugin"].__setitem__(
            "logical_path", "benchmarks/plugins/other.json"
        ),
    )


def _manifest_binding(raw: dict[str, Any]) -> None:
    _mutate_receipt(
        raw,
        lambda receipt: receipt["bindings"]["wheel"]["manifest"].__setitem__("sha256", "0" * 64),
    )


def _policy_binding(raw: dict[str, Any]) -> None:
    _mutate_receipt(raw, lambda receipt: receipt["bindings"].__setitem__("block_network", False))


def _result_envelope(raw: dict[str, Any]) -> None:
    attempt = _completed(raw)
    attempt["result_envelope"]["request_digest"] = "0" * 64
    result_payload = PUBLISHER._json_bytes(attempt["result_envelope"])
    attempt["receipt"]["result"] = PUBLISHER._binding("result.json", result_payload)
    attempt["source_files"]["receipt_result"] = PUBLISHER._binding(
        f"{attempt['pointer_value']}/result.json", result_payload
    )
    _rebind_receipt(attempt)


def _call_binding(raw: dict[str, Any]) -> None:
    _completed(raw)["call"]["function_call_id"] = "fc-mutated-publication-call"


@pytest.fixture
def raw_publication() -> dict[str, Any]:
    compressed = PUBLISHER._RAW_PATH.read_bytes()
    raw = PUBLISHER._strict_json_bytes(
        PUBLISHER._decompress(compressed), context="test raw publication"
    )
    assert isinstance(raw, dict)
    return raw


@pytest.mark.parametrize(
    "mutation",
    [
        _artifact_path,
        _artifact_digest,
        _request_binding,
        _suite_binding,
        _plugin_binding,
        _policy_binding,
        _manifest_binding,
        _result_envelope,
        _call_binding,
    ],
    ids=(
        "artifact-path",
        "artifact-digest",
        "request",
        "suite",
        "plugin",
        "policy",
        "manifest",
        "result",
        "call",
    ),
)
def test_check_rejects_mutated_embedded_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_publication: dict[str, Any],
    mutation: Mutation,
) -> None:
    mutated = copy.deepcopy(raw_publication)
    mutation(mutated)
    raw_path = tmp_path / "fusion-remote-exploratory.json.zst"
    raw_path.write_bytes(PUBLISHER._compress(PUBLISHER._json_bytes(mutated)))
    monkeypatch.setattr(PUBLISHER, "_RAW_PATH", raw_path)

    with pytest.raises((ValueError, ArtifactError, SchemaError)):
        PUBLISHER.check()
