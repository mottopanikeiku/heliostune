from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

from heliostune.artifacts import strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.local_executor import CapabilityProbe
from heliostune.methodology import verify_bundle_v1
from heliostune.native_fusion_bundle import (
    preflight_native_fusion_bundle,
    write_native_fusion_bundle,
)
from heliostune.native_fusion_executor import NativeFusionExecutionResult, run_native_fusion_suite
from heliostune.wheel_verifier import source_digest, source_entries

_ROOT = Path(__file__).resolve().parents[1]
_SUITE = _ROOT / "benchmarks/suites/residual-rmsnorm-triton-v1.json"
_PLUGIN = _ROOT / "benchmarks/plugins/fusion-triton-rmsnorm-plugin-v1.json"
_LEGACY_PLUGIN = _ROOT / "benchmarks/plugins/fusion-reference-plugin-v1.json"
_EXTRA_ROLES = (
    "suite",
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
    assert verified.bundle.coverage.terminal_cells == 12
    assert verified.bundle.coverage.successes == 0
    assert verified.bundle.coverage.failures == 12
    assert verified.bundle.provenance.attestation == "none"
    assert verified.protocol.protocol.evidence_class == "exploratory"
    assert verified.protocol.protocol.execution.executor_api == "heliostune.native_fusion_executor/2"
    assert verified.protocol.protocol.analysis.claims == ()
    assert not verified.publication_eligible
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
