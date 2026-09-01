from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

import heliostune.catalog as catalog_module
import heliostune.cli as cli
from heliostune.artifacts import read_json
from heliostune.catalog import build_research_catalog, verify_research_catalog
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, MODEL_SPECS
from heliostune.errors import ProtocolError

_REPO = Path(__file__).resolve().parents[1]
_CATALOG = _REPO / "benchmarks/research-artifact-manifest.json"

_EXPECTED_STUDY_IDS = {
    "heliostune-v1-l4-a10-transfer",
    "parhelion-v2-staged-transfer",
    "parhelion-v2-post-hoc-causal-addendum",
    "parhelion-v3-h200-transfer",
    "h100-fp16-reduction-probe",
    "hopper-h100-engineering-benchmark",
    "parhelion-v3-operator-authorized-engineering",
    "fusion-remote-h100-exploratory",
    "native-rmsnorm-h100-stage-gate",
}

_EXPECTED_ARTIFACT_PATHS = {
    "heliostune-v1-l4-a10-transfer": {
        "benchmarks/data/measurements.jsonl.zst",
        "benchmarks/manifest.json",
        "benchmarks/results/a10-to-l4.json",
        "benchmarks/results/l4-to-a10.json",
    },
    "parhelion-v2-staged-transfer": {
        "benchmarks/data/h100-measurements.jsonl.zst",
        "benchmarks/data/measurements.jsonl.zst",
        "benchmarks/data/parhelion-v2-measurements.jsonl.zst",
        "benchmarks/data/t4-measurements.jsonl.zst",
        "benchmarks/parhelion-v2-development-protocol.json",
        "benchmarks/parhelion-v2-h100-freeze.json",
        "benchmarks/parhelion-v2-post-run-manifest.json",
        "benchmarks/results/parhelion-h100-final.json",
        "benchmarks/results/parhelion-t4-selection.json",
        "benchmarks/results/parhelion-t4-validation.json",
    },
    "parhelion-v2-post-hoc-causal-addendum": {
        "benchmarks/data/parhelion-v2-measurements.jsonl.zst",
        "benchmarks/parhelion-v2-addendum-manifest.json",
        "benchmarks/results/parhelion-v2-addendum.json",
        "site/parhelion-v2-addendum.html",
    },
    "parhelion-v3-h200-transfer": {
        "benchmarks/data/parhelion-v3-pilot-failure.attempts.jsonl",
        "benchmarks/parhelion-v3-development-protocol.json",
        "benchmarks/parhelion-v3-validation-failure.json",
    },
    "h100-fp16-reduction-probe": {
        "benchmarks/data/h100-precision-probe.attempts.jsonl",
        "benchmarks/data/h100-precision-probe.json.zst",
        "benchmarks/h100-precision-probe-manifest.json",
        "benchmarks/results/h100-precision-probe-summary.json",
        "site/h100-precision-probe.html",
    },
    "hopper-h100-engineering-benchmark": {
        "benchmarks/data/hopper-h100-engineering.attempts.jsonl",
        "benchmarks/data/hopper-h100-engineering.json.zst",
        "benchmarks/hopper-h100-engineering-manifest-v2.json",
        "benchmarks/hopper-h100-engineering-manifest.json",
        "benchmarks/results/hopper-h100-engineering-summary-v2.json",
        "benchmarks/results/hopper-h100-engineering-summary.json",
        "site/hopper-h100-engineering.html",
    },
    "parhelion-v3-operator-authorized-engineering": {
        "benchmarks/data/parhelion-v3-candidate-bank0.jsonl.zst",
        "benchmarks/data/parhelion-v3-candidate-bank0.jsonl.zst.attempts.jsonl",
        "benchmarks/data/parhelion-v3-candidate-bank0.jsonl.zst.manifest.json",
        "benchmarks/data/parhelion-v3-final.jsonl.zst",
        "benchmarks/data/parhelion-v3-final.jsonl.zst.manifest.json",
        "benchmarks/data/parhelion-v3-h200.jsonl.zst",
        "benchmarks/data/parhelion-v3-h200.jsonl.zst.attempts.jsonl",
        "benchmarks/data/parhelion-v3-h200.jsonl.zst.manifest.json",
        "benchmarks/data/parhelion-v3-pilot-operator-retry.jsonl.zst",
        "benchmarks/data/parhelion-v3-pilot-operator-retry.jsonl.zst.attempts.jsonl",
        "benchmarks/data/parhelion-v3-pilot-operator-retry.jsonl.zst.source-manifest.json",
        "benchmarks/data/parhelion-v3-validation-raw-mixed-a100.jsonl.zst",
        "benchmarks/data/parhelion-v3-validation-raw-mixed-a100.jsonl.zst.attempts.jsonl",
        "benchmarks/data/parhelion-v3-validation-raw-mixed-a100.jsonl.zst.source-manifest.json",
        "benchmarks/data/parhelion-v3-validation.jsonl.zst",
        "benchmarks/data/parhelion-v3-validation.jsonl.zst.attempts.jsonl",
        "benchmarks/data/parhelion-v3-validation.jsonl.zst.manifest.json",
        "benchmarks/parhelion-v3-config-manifest.json",
        "benchmarks/parhelion-v3-h200-freeze.json",
        "benchmarks/parhelion-v3-h200-freeze.sha256",
        "benchmarks/results/parhelion-v3-a100-selection.json",
        "benchmarks/results/parhelion-v3-h200-engineering.json",
        "site/parhelion-v3-engineering.html",
    },
    "fusion-remote-h100-exploratory": {
        "benchmarks/data/fusion-remote-exploratory.json.zst",
        "benchmarks/fusion-remote-exploratory-manifest.json",
        "benchmarks/results/fusion-remote-exploratory-summary.json",
        "site/fusion-remote-exploratory.html",
    },
    "native-rmsnorm-h100-stage-gate": {
        "benchmarks/data/native-rmsnorm-h100.json.zst",
        "benchmarks/native-rmsnorm-h100-manifest.json",
        "benchmarks/results/native-rmsnorm-h100-summary.json",
        "site/native-rmsnorm-h100.html",
    },
}
_ARTIFACT_SECTIONS = (
    "data",
    "results",
    "protocol_chain",
    "reports",
    "files",
    "manifests",
    "raw_artifacts",
    "attempt_journals",
)


def _study(catalog: dict[str, Any], study_id: str) -> dict[str, Any]:
    return next(study for study in catalog["studies"] if study["study_id"] == study_id)


def _artifact_paths(study: dict[str, Any]) -> set[str]:
    paths = {
        entry["path"]
        for section in _ARTIFACT_SECTIONS
        for entry in cast(list[dict[str, Any]], study.get(section, ()))
    }
    protocol = cast(dict[str, Any] | None, study.get("protocol"))
    if protocol is not None:
        paths.add(protocol["path"])
    return paths


def _assert_mutation_fails(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    mutated = cast(dict[str, Any], deepcopy(read_json(_CATALOG)))
    mutation(mutated)
    original_read_json = read_json

    def read_json_with_mutation(path: str | Path) -> Any:
        if Path(path).resolve() == _CATALOG.resolve():
            return mutated
        return original_read_json(path)

    monkeypatch.setattr(catalog_module, "read_json", read_json_with_mutation)
    with pytest.raises(ProtocolError, match="catalog .* mismatch"):
        verify_research_catalog(_CATALOG)


def test_catalog_is_deterministic_and_serializes_exact_inventories() -> None:
    committed = cast(dict[str, Any], read_json(_CATALOG))

    assert build_research_catalog(_REPO) == committed
    assert len(committed["inventories"]["workloads"]) == len(DEFAULT_WORKLOADS) == 96
    assert len(committed["inventories"]["configs"]) == len(DEFAULT_CONFIGS) == 36
    assert [row["key"] for row in committed["inventories"]["workloads"]] == [
        workload.key for workload in DEFAULT_WORKLOADS
    ]
    assert [row["key"] for row in committed["inventories"]["configs"]] == [
        config.key for config in DEFAULT_CONFIGS
    ]

    assert {study["study_id"] for study in committed["studies"]} == _EXPECTED_STUDY_IDS
    assert {
        study["study_id"]: _artifact_paths(study)
        for study in cast(list[dict[str, Any]], committed["studies"])
    } == _EXPECTED_ARTIFACT_PATHS


def test_model_configs_use_audited_revisions_without_retroactive_claims() -> None:
    catalog = cast(dict[str, Any], read_json(_CATALOG))
    models = {row["name"]: row for row in catalog["inventories"]["model_configs"]}

    assert set(models) == {model.name for model in MODEL_SPECS}
    assert all(row["collection_revision_status"] == "not_recorded" for row in models.values())
    assert all(len(row["reproduction_revision"]) == 40 for row in models.values())
    assert all(len(row["config_sha256"]) == 64 for row in models.values())
    assert all("/resolve/main/" not in model.config_url for model in MODEL_SPECS)


def test_catalog_verifies_all_bytes_counts_aliases_and_frozen_v2_points() -> None:
    facts = verify_research_catalog(_CATALOG)

    assert facts == {
        "measurement_rows": 278_406,
        "json_artifacts": 33,
        "html_reports": 6,
        "file_artifacts": 10,
        "compressed_raw_artifacts": 4,
        "aliases": 7,
    }


def test_hardened_hopper_revision_and_immutable_v1_are_both_registered() -> None:
    built = build_research_catalog(_REPO)
    hopper = _study(built, "hopper-h100-engineering-benchmark")
    manifests = {entry["path"]: entry for entry in hopper["manifests"]}
    results = {entry["path"]: entry for entry in hopper["results"]}

    assert set(manifests) == {
        "benchmarks/hopper-h100-engineering-manifest.json",
        "benchmarks/hopper-h100-engineering-manifest-v2.json",
    }
    assert set(results) == {
        "benchmarks/results/hopper-h100-engineering-summary.json",
        "benchmarks/results/hopper-h100-engineering-summary-v2.json",
    }
    assert manifests["benchmarks/hopper-h100-engineering-manifest.json"]["status"].endswith(
        "_immutable"
    )
    assert results["benchmarks/results/hopper-h100-engineering-summary.json"]["status"].endswith(
        "_immutable"
    )
    assert manifests["benchmarks/hopper-h100-engineering-manifest-v2.json"]["status"].endswith(
        "_methodology_compatible_derivation"
    )
    assert hopper["reports"][0]["status"].endswith("_methodology_compatible_derivation")


def test_precision_catalog_uses_current_raw_schema_label() -> None:
    precision = _study(build_research_catalog(_REPO), "h100-fp16-reduction-probe")

    assert precision["measurement_schema"] == "h100-precision-probe-raw-v2"
    assert precision["raw_artifacts"][0]["schema"] == "h100-precision-probe-raw-v2"


def test_fusion_remote_catalog_records_exact_publication_boundary() -> None:
    fusion = _study(build_research_catalog(_REPO), "fusion-remote-h100-exploratory")

    assert fusion["analysis_status"] == "post_hoc_exploratory"
    assert fusion["outcome_status"] == "mixed_completed_unresolved"
    assert fusion["publication_eligible"] is False
    assert fusion["measurement_schema"] == "heliostune.fusion-remote-exploratory.raw/1"
    assert fusion["raw_artifacts"] == [
        {
            "kind": "compressed_json_artifact",
            "path": "benchmarks/data/fusion-remote-exploratory.json.zst",
            "schema": "heliostune.fusion-remote-exploratory.raw/1",
            "status": "published_mixed_completed_unresolved",
            "compression": "zstd",
            "compressed_bytes": 9475,
            "compressed_sha256": (
                "fe732172c2a8fa3698c47a5c7a8e97e6c895703c90d86a2061ddb7a11ddeec35"
            ),
            "uncompressed_bytes": 83952,
            "uncompressed_sha256": (
                "eb91be687ba089c999a72eb8d879fdb3e844e410881e6ecfe540e8993f28957a"
            ),
        }
    ]
    assert [(entry["path"], entry["schema"]) for entry in fusion["results"]] == [
        (
            "benchmarks/results/fusion-remote-exploratory-summary.json",
            "heliostune.fusion-remote-exploratory.summary/1",
        )
    ]
    assert [(entry["path"], entry["schema"]) for entry in fusion["manifests"]] == [
        (
            "benchmarks/fusion-remote-exploratory-manifest.json",
            "heliostune.fusion-remote-exploratory.manifest/1",
        )
    ]
    assert [(entry["path"], entry["schema"]) for entry in fusion["reports"]] == [
        ("site/fusion-remote-exploratory.html", "html5")
    ]
    assert all(
        entry["status"] == "published_mixed_completed_unresolved"
        for section in ("results", "manifests", "reports")
        for entry in fusion[section]
    )
    assert not any(path.startswith("/home/") for path in _artifact_paths(fusion))


def test_native_rmsnorm_catalog_records_exact_stage_gate_boundary() -> None:
    native = _study(build_research_catalog(_REPO), "native-rmsnorm-h100-stage-gate")

    assert native["analysis_status"] == "predeclared_exploratory_stage_gate"
    assert native["outcome_status"] == "STOP_BELOW_THRESHOLD"
    assert native["publication_eligible"] is False
    assert native["split_design"] == "single frozen case; two retained remote attempts"
    assert native["measurement_schema"] == "heliostune.native-rmsnorm-h100.raw/1"
    assert native["raw_artifacts"] == [
        {
            "kind": "compressed_json_artifact",
            "path": "benchmarks/data/native-rmsnorm-h100.json.zst",
            "schema": "heliostune.native-rmsnorm-h100.raw/1",
            "status": "published_stop_below_threshold",
            "compression": "zstd",
            "compressed_bytes": 21852,
            "compressed_sha256": (
                "55aa8d2bbbb22194824409920ceb6a44d3ba32d79599f36d3529ea66e9e4e8d0"
            ),
            "uncompressed_bytes": 145876,
            "uncompressed_sha256": (
                "eb66468f627d41740afc6551f66320c0ea4e1ddd3b76d9e4e37066fcd8daa958"
            ),
        }
    ]
    assert [(entry["path"], entry["schema"]) for entry in native["results"]] == [
        (
            "benchmarks/results/native-rmsnorm-h100-summary.json",
            "heliostune.native-rmsnorm-h100.summary/1",
        )
    ]
    assert [(entry["path"], entry["schema"]) for entry in native["manifests"]] == [
        (
            "benchmarks/native-rmsnorm-h100-manifest.json",
            "heliostune.native-rmsnorm-h100.manifest/1",
        )
    ]
    assert [(entry["path"], entry["schema"]) for entry in native["reports"]] == [
        ("site/native-rmsnorm-h100.html", "html5")
    ]
    assert all(
        entry["status"] == "published_stop_below_threshold"
        for section in ("results", "manifests", "reports")
        for entry in native[section]
    )
    assert not any(path.startswith("/home/") for path in _artifact_paths(native))


def test_catalog_command_reports_compressed_raw_artifacts_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli._verify_catalog(argparse.Namespace(catalog=_CATALOG)) == 0
    assert "4 compressed raw artifacts" in capsys.readouterr().out


@pytest.mark.parametrize("study_id", sorted(_EXPECTED_STUDY_IDS))
def test_catalog_rejects_omitted_registered_study(
    monkeypatch: pytest.MonkeyPatch,
    study_id: str,
) -> None:
    def omit(catalog: dict[str, Any]) -> None:
        catalog["studies"] = [
            study for study in catalog["studies"] if study["study_id"] != study_id
        ]

    _assert_mutation_fails(monkeypatch, omit)


def test_catalog_rejects_swapped_study_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def swap(catalog: dict[str, Any]) -> None:
        first = _study(catalog, "heliostune-v1-l4-a10-transfer")
        second = _study(catalog, "parhelion-v3-h200-transfer")
        first["analysis_status"], second["analysis_status"] = (
            second["analysis_status"],
            first["analysis_status"],
        )

    _assert_mutation_fails(monkeypatch, swap)


def test_catalog_rejects_swapped_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def swap(catalog: dict[str, Any]) -> None:
        study = _study(catalog, "h100-fp16-reduction-probe")
        manifest = study["manifests"][0]
        summary = study["results"][0]
        manifest["path"], summary["path"] = summary["path"], manifest["path"]

    _assert_mutation_fails(monkeypatch, swap)


def test_catalog_rejects_swapped_artifact_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def swap(catalog: dict[str, Any]) -> None:
        precision = _study(catalog, "h100-fp16-reduction-probe")["raw_artifacts"][0]
        hopper = _study(catalog, "hopper-h100-engineering-benchmark")["raw_artifacts"][0]
        precision["compressed_sha256"], hopper["compressed_sha256"] = (
            hopper["compressed_sha256"],
            precision["compressed_sha256"],
        )

    _assert_mutation_fails(monkeypatch, swap)


@pytest.mark.parametrize(
    "section",
    ["reports", "raw_artifacts", "attempt_journals", "manifests"],
)
def test_catalog_rejects_omitted_new_study_artifact(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
) -> None:
    def omit(catalog: dict[str, Any]) -> None:
        _study(catalog, "h100-fp16-reduction-probe")[section] = []

    _assert_mutation_fails(monkeypatch, omit)


@pytest.mark.parametrize(
    "section",
    ["reports", "raw_artifacts", "manifests", "results"],
)
def test_catalog_rejects_omitted_fusion_remote_artifact(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
) -> None:
    def omit(catalog: dict[str, Any]) -> None:
        _study(catalog, "fusion-remote-h100-exploratory")[section] = []

    _assert_mutation_fails(monkeypatch, omit)


def test_catalog_rejects_fusion_remote_status_and_path_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def alter(catalog: dict[str, Any]) -> None:
        fusion = _study(catalog, "fusion-remote-h100-exploratory")
        fusion["analysis_status"] = "historical_confirmatory"
        fusion["results"][0]["path"], fusion["manifests"][0]["path"] = (
            fusion["manifests"][0]["path"],
            fusion["results"][0]["path"],
        )

    _assert_mutation_fails(monkeypatch, alter)


@pytest.mark.parametrize(
    ("section", "digest_key"),
    [
        ("raw_artifacts", "compressed_sha256"),
        ("results", "sha256"),
        ("manifests", "sha256"),
        ("reports", "sha256"),
    ],
)
def test_catalog_rejects_fusion_remote_digest_change(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    digest_key: str,
) -> None:
    def alter(catalog: dict[str, Any]) -> None:
        entry = _study(catalog, "fusion-remote-h100-exploratory")[section][0]
        entry[digest_key] = "0" * 64

    _assert_mutation_fails(monkeypatch, alter)


def test_every_registered_artifact_file_exists() -> None:
    catalog = build_research_catalog(_REPO)
    registered_paths = {
        path
        for study in cast(list[dict[str, Any]], catalog["studies"])
        for path in _artifact_paths(study)
    }

    assert registered_paths
    assert all((_REPO / path).is_file() for path in registered_paths)
