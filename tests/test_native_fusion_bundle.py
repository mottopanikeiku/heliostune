from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from heliostune.artifacts import strict_json_dumps, strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.local_executor import CapabilityProbe
from heliostune.methodology import encode_attempt_journal, verify_bundle_v1
from heliostune.native_fusion_bundle import (
    _attempt_journal,
    preflight_native_fusion_bundle,
    write_native_fusion_bundle,
)
from heliostune.native_fusion_executor import NativeFusionExecutionResult, run_native_fusion_suite
from heliostune.scope import verify_plugin
from heliostune.wheel_verifier import source_digest, source_entries

_ROOT = Path(__file__).resolve().parents[1]
_SUITE = _ROOT / "benchmarks/suites/residual-rmsnorm-triton-v1.json"
_PLUGIN = _ROOT / "benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json"
_LEGACY_PLUGIN = _ROOT / "benchmarks/plugins/fusion-reference-plugin-v1.json"
_EXTRA_ROLES = (
    "terminal_cells",
    "observations",
    "capability_probe",
    "tensor_materialization",
    "execution_summary",
    "stage_outcomes",
    "compile_evidence",
    "resource_evidence",
    "validation_evidence",
    "profile_evidence",
    "executor_sources",
    "selected_suite",
    "attempt_chain",
)
_DESCRIPTOR_PUBLICATION = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and all(
        function in os.supports_dir_fd
        for function in (os.open, os.mkdir, os.rename, os.stat, os.unlink, os.rmdir)
    )
    and Path("/proc/self/fd").is_dir()
)


def _unavailable() -> CapabilityProbe:
    return CapabilityProbe(
        False,
        ("torch_missing",),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        "strict CPU bundle fake",
    )


@pytest.fixture
def aborted(monkeypatch: pytest.MonkeyPatch) -> NativeFusionExecutionResult:
    import heliostune.native_fusion_executor as executor

    monkeypatch.setattr(executor, "_probe_capability", lambda: (_unavailable(), None, None))
    return run_native_fusion_suite(_SUITE)


def _load(path: Path) -> object:
    return strict_json_loads(path.read_text(encoding="utf-8"), source=path)


def _two_suite_plugin(tmp_path: Path) -> tuple[Path, tuple[bytes, bytes]]:
    source_suite = _load(_SUITE)
    source_plugin = _load(_PLUGIN)
    assert type(source_suite) is dict
    assert type(source_plugin) is dict

    shadow_suite = dict(source_suite)
    shadow_suite["suite_id"] = "residual-rmsnorm-triton-shadow"
    shadow_payload = strict_json_dumps(shadow_suite).encode("utf-8")
    selected_payload = _SUITE.read_bytes()

    root = tmp_path / "inventory"
    suites = root / "suites"
    plugins = root / "plugins"
    suites.mkdir(parents=True)
    plugins.mkdir()
    (suites / "shadow.json").write_bytes(shadow_payload)
    (suites / "selected.json").write_bytes(selected_payload)

    plugin = dict(source_plugin)
    plugin["suite_refs"] = [
        {
            "path": "../suites/shadow.json",
            "sha256": hashlib.sha256(shadow_payload).hexdigest(),
            "suite_id": shadow_suite["suite_id"],
            "revision": shadow_suite["revision"],
        },
        {
            "path": "../suites/selected.json",
            "sha256": hashlib.sha256(selected_payload).hexdigest(),
            "suite_id": source_suite["suite_id"],
            "revision": source_suite["revision"],
        },
    ]
    plugin_path = plugins / "plugin.json"
    plugin_path.write_bytes(strict_json_dumps(plugin).encode("utf-8"))
    return plugin_path, (shadow_payload, selected_payload)


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason="descriptor publication is unavailable")
def test_cpu_unavailable_result_publishes_closed_claimless_bundle(
    tmp_path: Path, aborted: NativeFusionExecutionResult
) -> None:
    verified = write_native_fusion_bundle(
        aborted,
        plugin_path=_PLUGIN,
        output_dir=tmp_path / "bundle",
    )

    assert verified.bundle.schema == "heliostune.bundle/1"
    assert verified.bundle.lifecycle.state == "SEALED"
    assert verified.bundle.lifecycle.outcome == "aborted"
    assert verified.bundle.coverage.expected_cells == 12
    assert verified.bundle.coverage.terminal_cells == 0
    assert verified.bundle.coverage.successes == 0
    assert verified.bundle.coverage.failures == 0
    assert verified.bundle.provenance.attestation == "none"
    assert verified.protocol.protocol.evidence_class == "exploratory"
    assert (
        verified.protocol.protocol.execution.executor_api == "heliostune.native_fusion_executor/2"
    )
    assert verified.protocol.protocol.analysis.claims == ()
    assert not verified.publication_eligible
    assert verified.limitations.plugin_suite_custody == "checked"
    assert verified.limitations.attempt_journal_hash_chain == "checked"
    assert verified.limitations.attempt_reconciliation == "checked"
    assert verify_bundle_v1(verified.root_path).bundle == verified.bundle

    roles = {artifact.role for artifact in verified.bundle.artifacts}
    assert set(_EXTRA_ROLES) <= roles
    output = verified.root_path.parent
    summary = _load(output / "execution_summary.json")
    assert type(summary) is dict
    assert summary["claims"] == []
    assert summary["fusion_claim"] is False
    assert summary["publication_eligible"] is False
    assert summary["environment"]["backend_invoked"] is False
    assert summary["summary"] == dict(aborted.summary)
    observations = (output / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(observations) == 12
    assert all(
        isinstance(row := strict_json_loads(line), dict) and row["status"] == "blocked"
        for line in observations
    )
    attempts_payload = (output / "attempts.jsonl").read_bytes()
    assert attempts_payload == b""
    assert verified.bundle.attempts.hash_chain_head == hashlib.sha256(b"").hexdigest()
    assert verified.bundle.attempts.logical == 0
    assert verified.bundle.attempts.physical == 0
    assert verified.bundle.attempts.terminal == 0
    assert _load(output / "attempt_chain.json") == {"schema": "heliostune.attempt-chain/1"}

    comparators = _load(output / "comparators.json")
    assert type(comparators) is list
    assert [item["id"] for item in comparators] == [
        "rmsnorm-eager-reference",
        "rmsnorm-inductor-comparator",
    ]
    predicate = _load(output / "environment_predicate.json")
    assert type(predicate) is dict
    assert predicate["gpu"] == "H100"
    assert predicate["architecture"] == "sm90"
    assert predicate["torch_version"] == "2.8.0"
    assert predicate["triton_version"] == "3.4.0"


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason="descriptor publication is unavailable")
def test_native_publishes_every_plugin_suite_once_and_selects_by_index(
    tmp_path: Path, aborted: NativeFusionExecutionResult
) -> None:
    plugin_path, suite_payloads = _two_suite_plugin(tmp_path)
    plugin = verify_plugin(plugin_path)
    assert tuple(item.bytes for item in plugin.suites) == suite_payloads

    verified = write_native_fusion_bundle(
        aborted,
        plugin_path=plugin_path,
        output_dir=tmp_path / "bundle",
    )
    output = verified.root_path.parent
    artifacts = {artifact.role: artifact for artifact in verified.bundle.artifacts}
    assert "suite" not in artifacts
    assert [role for role in artifacts if role.startswith("plugin_suite_")] == [
        "plugin_suite_0",
        "plugin_suite_1",
    ]
    assert artifacts["plugin_suite_0"].path == "plugin_suite_0.json"
    assert artifacts["plugin_suite_1"].path == "plugin_suite_1.json"
    assert (output / "plugin.json").read_bytes() == plugin.bytes
    assert (output / "plugin_suite_0.json").read_bytes() == suite_payloads[0]
    assert (output / "plugin_suite_1.json").read_bytes() == suite_payloads[1]
    assert _load(output / "selected_suite.json") == {
        "schema": "heliostune.selected-suite/1",
        "plugin_suite_index": 1,
    }


def test_native_nonempty_attempt_journal_uses_canonical_predecessor_chain() -> None:
    result = cast(
        NativeFusionExecutionResult,
        SimpleNamespace(
            capability=SimpleNamespace(available=True),
            outcome="completed",
            observations=(SimpleNamespace(cell_id="cell-a", status="passed"),),
            attempts=(
                {"cell_id": "cell-a", "status": "running"},
                {"cell_id": "cell-a", "status": "success"},
            ),
        ),
    )
    transitions, terminal_ids, successes, failures = _attempt_journal(result)
    assert terminal_ids == ("cell-a",)
    assert (successes, failures) == (1, 0)

    payload, final_head = encode_attempt_journal(transitions)
    predecessor = hashlib.sha256(b"").hexdigest()
    rows = payload.splitlines(keepends=True)
    assert len(rows) == 3
    for row_bytes, status in zip(rows, ("pending", "running", "success"), strict=True):
        expected = {
            "cell_id": "cell-a",
            "predecessor_sha256": predecessor,
            "status": status,
        }
        canonical = strict_json_dumps(expected, compact=True).encode("utf-8") + b"\n"
        assert row_bytes == canonical
        predecessor = hashlib.sha256(row_bytes).hexdigest()
    assert final_head == predecessor


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason="descriptor publication is unavailable")
def test_native_evidence_roles_are_lossless_and_sources_are_hashed(
    tmp_path: Path, aborted: NativeFusionExecutionResult
) -> None:
    verified = write_native_fusion_bundle(
        aborted,
        plugin_path=_PLUGIN,
        output_dir=tmp_path / "bundle",
    )
    output = verified.root_path.parent
    for role, expected in (
        ("stage_outcomes", aborted.stage_outcomes),
        ("compile_evidence", aborted.compile_evidence),
        ("resource_evidence", aborted.resource_evidence),
        ("validation_evidence", aborted.validation_evidence),
        ("profile_evidence", aborted.profile_evidence),
    ):
        assert _load(output / f"{role}.json") == dict(expected)

    inventory = _load(output / "executor_sources.json")
    assert type(inventory) is dict
    sources = inventory["sources"]
    assert type(sources) is list
    assert [item["path"] for item in sources] == [
        "fusion_kernels.py",
        "_fusion_gpu.py",
        "native_fusion_executor.py",
        "local_executor.py",
    ]
    package_file = __import__("heliostune").__file__
    assert package_file is not None
    package_dir = Path(package_file).resolve().parent
    package_entries = source_entries(package_dir)
    assert inventory["package_source_sha256"] == source_digest(package_entries)
    assert inventory["package_source_count"] == len(package_entries)
    for item in sources:
        payload = (package_dir / item["path"]).read_bytes()
        assert item["bytes"] == len(payload)
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason="descriptor publication is unavailable")
def test_tampered_native_inputs_fail_before_publication(
    tmp_path: Path, aborted: NativeFusionExecutionResult
) -> None:
    summary = dict(aborted.summary)
    summary["claims"] = ["publish"]
    with pytest.raises(SchemaError):
        write_native_fusion_bundle(
            replace(aborted, summary=summary),
            plugin_path=_PLUGIN,
            output_dir=tmp_path / "claims",
        )

    with pytest.raises(ArtifactError, match="not exactly one"):
        write_native_fusion_bundle(
            aborted,
            plugin_path=_LEGACY_PLUGIN,
            output_dir=tmp_path / "plugin",
        )

    copied_suite = tmp_path / "suite.json"
    copied_suite.write_bytes(aborted.verified_suite_bytes)
    rebound = replace(aborted, verified_suite_path=str(copied_suite))
    copied_suite.write_bytes(aborted.verified_suite_bytes + b"\n")
    with pytest.raises((ArtifactError, SchemaError)):
        write_native_fusion_bundle(
            rebound,
            plugin_path=_PLUGIN,
            output_dir=tmp_path / "source",
        )
    assert not any((tmp_path / name).exists() for name in ("claims", "plugin", "source"))


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason="descriptor preflight is unavailable")
def test_native_preflight_rejects_binding_and_destination_hazards(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not exactly one"):
        preflight_native_fusion_bundle(
            _SUITE,
            plugin_path=_LEGACY_PLUGIN,
            output_dir=tmp_path / "bad-plugin",
        )

    with pytest.raises(ArtifactError):
        preflight_native_fusion_bundle(
            _SUITE,
            plugin_path=tmp_path / "missing-plugin.json",
            output_dir=tmp_path / "missing-plugin",
        )

    missing_parent = tmp_path / "missing" / "bundle"
    with pytest.raises(ArtifactError, match="output parent"):
        preflight_native_fusion_bundle(
            _SUITE,
            plugin_path=_PLUGIN,
            output_dir=missing_parent,
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "linked-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ArtifactError, match="output parent"):
        preflight_native_fusion_bundle(
            _SUITE,
            plugin_path=_PLUGIN,
            output_dir=symlink_parent / "bundle",
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ArtifactError, match="must be absent"):
        preflight_native_fusion_bundle(
            _SUITE,
            plugin_path=_PLUGIN,
            output_dir=existing,
        )


def test_writer_rejects_source_inventory_race(
    tmp_path: Path, aborted: NativeFusionExecutionResult
) -> None:
    raw_sources = aborted.executor_sources["sources"]
    assert type(raw_sources) is list
    changed_sources = [dict(item) for item in raw_sources]
    changed_sources[0]["sha256"] = "0" * 64
    package_source_count = aborted.executor_sources["package_source_count"]
    assert type(package_source_count) is int
    mutations: tuple[dict[str, object], ...] = (
        {"sources": changed_sources},
        {"package_source_sha256": "0" * 64},
        {"package_source_count": package_source_count + 1},
    )
    for index, mutation in enumerate(mutations):
        inventory: dict[str, object] = {
            "schema": aborted.executor_sources["schema"],
            "package_source_sha256": aborted.executor_sources["package_source_sha256"],
            "package_source_count": aborted.executor_sources["package_source_count"],
            "sources": [dict(item) for item in raw_sources],
        }
        inventory.update(mutation)
        raced = replace(aborted, executor_sources=inventory)
        output = tmp_path / f"race-{index}"
        with pytest.raises(ArtifactError, match="changed since execution"):
            write_native_fusion_bundle(
                raced,
                plugin_path=_PLUGIN,
                output_dir=output,
            )
        assert not output.exists()


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason="descriptor publication is unavailable")
def test_tampered_native_source_inventory_is_rejected(
    tmp_path: Path, aborted: NativeFusionExecutionResult
) -> None:
    verified = write_native_fusion_bundle(
        aborted,
        plugin_path=_PLUGIN,
        output_dir=tmp_path / "bundle",
    )
    source = verified.root_path.parent / "executor_sources.json"
    payload = source.read_bytes()
    source.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
        verify_bundle_v1(verified.root_path)


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason="descriptor publication is unavailable")
@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_missing_or_tampered_plugin_suite_inventory_is_rejected(
    tmp_path: Path,
    aborted: NativeFusionExecutionResult,
    mutation: str,
) -> None:
    verified = write_native_fusion_bundle(
        aborted,
        plugin_path=_PLUGIN,
        output_dir=tmp_path / mutation,
    )
    suite_path = verified.root_path.parent / "plugin_suite_0.json"
    if mutation == "missing":
        suite_path.unlink()
    else:
        payload = suite_path.read_bytes()
        suite_path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    with pytest.raises(ArtifactError):
        verify_bundle_v1(verified.root_path)
