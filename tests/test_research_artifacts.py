from __future__ import annotations

from pathlib import Path

from heliostune.artifacts import read_json
from heliostune.catalog import build_research_catalog, verify_research_catalog
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, MODEL_SPECS

_REPO = Path(__file__).resolve().parents[1]
_CATALOG = _REPO / "benchmarks/research-artifact-manifest.json"


def test_catalog_is_deterministic_and_serializes_exact_inventories() -> None:
    committed = read_json(_CATALOG)

    assert build_research_catalog(_REPO) == committed
    assert len(committed["inventories"]["workloads"]) == len(DEFAULT_WORKLOADS) == 96
    assert len(committed["inventories"]["configs"]) == len(DEFAULT_CONFIGS) == 36
    assert [row["key"] for row in committed["inventories"]["workloads"]] == [
        workload.key for workload in DEFAULT_WORKLOADS
    ]
    assert [row["key"] for row in committed["inventories"]["configs"]] == [
        config.key for config in DEFAULT_CONFIGS
    ]


def test_model_configs_use_audited_revisions_without_retroactive_claims() -> None:
    catalog = read_json(_CATALOG)
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
        "json_artifacts": 23,
        "html_reports": 2,
        "file_artifacts": 8,
        "aliases": 7,
    }
