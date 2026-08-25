"""Machine-readable research artifact catalog construction and verification."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import zstandard

from heliostune.artifacts import read_json, read_measurements
from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, MODEL_SPECS
from heliostune.errors import ArtifactError, ProtocolError, SchemaError
from heliostune.multisource import compare_multisource
from heliostune.replay import BenchmarkTable
from heliostune.validation import exact_fields, exact_int, exact_object, nonblank_string

_MODEL_CONFIG_SHA256 = {
    "mistral-7b": "cf25cdf4719f181d1d1d371973285d9afe9afde0d0c6a6fd48de857555ce1e0d",
    "qwen2.5-7b": "267ce68584c5f24c3b267d934db2de68dd21d1ca677fb78ed809eb60067f7642",
    "phi-3-mini": "072d4df63228ef806a6b2b2f02a93f1d048ebd31584d0f61d1180aef36e5bcea",
    "granite-3.1-8b": "03c19685d37a17a541641fb17fa40ba371d1d892e87334d3068d90e8f296a365",
}
_MODEL_CONFIG_BYTES = {
    "mistral-7b": 571,
    "qwen2.5-7b": 686,
    "phi-3-mini": 967,
    "granite-3.1-8b": 790,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decompressed_facts(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    try:
        with (
            path.open("rb") as source,
            zstandard.ZstdDecompressor().stream_reader(source) as decoded,
        ):
            for block in iter(lambda: decoded.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    except (OSError, zstandard.ZstdError) as exc:
        raise ArtifactError(f"cannot decompress {path}: {exc}") from exc
    return size, digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _data_entry(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    rows = read_measurements(path)
    if not rows:
        raise ProtocolError(f"measurement archive is empty: {relative}")
    uncompressed_bytes, uncompressed_sha256 = _decompressed_facts(path)
    counts = Counter(row.hardware.gpu for row in rows)
    banks = sorted({row.bank for row in rows})
    hardware = sorted(
        {row.hardware for row in rows},
        key=lambda profile: profile.gpu,
    )
    return {
        "kind": "measurement_archive",
        "path": relative,
        "schema": "heliostune-measurement-v1",
        "compression": "zstd",
        "compressed_bytes": path.stat().st_size,
        "compressed_sha256": _sha256(path),
        "uncompressed_bytes": uncompressed_bytes,
        "uncompressed_sha256": uncompressed_sha256,
        "rows": len(rows),
        "failures": sum(not row.usable for row in rows),
        "gpus": dict(sorted(counts.items())),
        "banks": banks,
        "hardware": [profile.to_dict() for profile in hardware],
    }


def _json_entry(root: Path, relative: str, schema: str) -> dict[str, object]:
    path = root / relative
    read_json(path)
    return {
        "kind": "json_artifact",
        "path": relative,
        "schema": schema,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _file_entry(root: Path, relative: str, kind: str) -> dict[str, object]:
    path = root / relative
    if not path.is_file():
        raise ArtifactError(f"catalog artifact is missing: {relative}")
    return {
        "kind": kind,
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _model_catalog() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for model in MODEL_SPECS:
        revision = model.config_url.split("/resolve/", 1)[1].split("/", 1)[0]
        values.append(
            {
                "name": model.name,
                "hidden_size": model.hidden_size,
                "intermediate_size": model.intermediate_size,
                "attention_heads": model.attention_heads,
                "key_value_heads": model.key_value_heads,
                "collection_revision_status": "not_recorded",
                "reproduction_revision": revision,
                "config_url": model.config_url,
                "config_bytes": _MODEL_CONFIG_BYTES[model.name],
                "config_sha256": _MODEL_CONFIG_SHA256[model.name],
                "provenance_note": (
                    "This immutable pin reproduces the coded dimensions; it is not retroactive "
                    "proof of the model revision used during historical collection."
                ),
            }
        )
    return values


def build_research_catalog(root: str | Path) -> dict[str, object]:
    """Build the deterministic catalog payload from immutable repository artifacts."""
    repository = Path(root)
    baseline_path = repository / "benchmarks/historical-artifact-baseline.json"
    baseline = exact_object(read_json(baseline_path), context="historical artifact baseline")
    v1_data = _data_entry(repository, "benchmarks/data/measurements.jsonl.zst")
    t4_data = _data_entry(repository, "benchmarks/data/t4-measurements.jsonl.zst")
    h100_data = _data_entry(repository, "benchmarks/data/h100-measurements.jsonl.zst")
    v2_data = _data_entry(repository, "benchmarks/data/parhelion-v2-measurements.jsonl.zst")
    v1_results = [
        _json_entry(repository, "benchmarks/results/l4-to-a10.json", "heliostune-v1-summary"),
        _json_entry(repository, "benchmarks/results/a10-to-l4.json", "heliostune-v1-summary"),
    ]
    v2_results = [
        _json_entry(
            repository,
            "benchmarks/results/parhelion-t4-selection.json",
            "parhelion-v2-selection",
        ),
        _json_entry(
            repository,
            "benchmarks/results/parhelion-t4-validation.json",
            "parhelion-v2-validation-summary",
        ),
        _json_entry(
            repository,
            "benchmarks/results/parhelion-h100-final.json",
            "parhelion-v2-final-summary",
        ),
    ]
    protocol_chain = [
        _json_entry(
            repository,
            "benchmarks/parhelion-v2-development-protocol.json",
            "parhelion-v2-development-protocol",
        ),
        _json_entry(
            repository,
            "benchmarks/parhelion-v2-h100-freeze.json",
            "parhelion-v2-h100-freeze",
        ),
        _json_entry(
            repository,
            "benchmarks/parhelion-v2-post-run-manifest.json",
            "parhelion-v2-post-run-manifest",
        ),
    ]
    addendum_results = [
        _json_entry(
            repository,
            "benchmarks/results/parhelion-v2-addendum.json",
            "parhelion-v2-post-hoc-addendum",
        ),
        _json_entry(
            repository,
            "benchmarks/parhelion-v2-addendum-manifest.json",
            "parhelion-v2-addendum-manifest",
        ),
    ]
    addendum_report = _file_entry(
        repository,
        "site/parhelion-v2-addendum.html",
        "self_contained_html_report",
    )
    v3_protocol = _json_entry(
        repository,
        "benchmarks/parhelion-v3-development-protocol.json",
        "parhelion-v3-development-protocol",
    )
    v3_failure = _json_entry(
        repository,
        "benchmarks/parhelion-v3-validation-failure.json",
        "parhelion-v3-validation-failure",
    )
    v3_journal = _file_entry(
        repository,
        "benchmarks/data/parhelion-v3-pilot-failure.attempts.jsonl",
        "append_only_function_call_journal",
    )
    return {
        "schema_version": 1,
        "catalog_id": "heliostune-research-artifacts-1",
        "historical_baseline": {
            "path": _relative(repository, baseline_path),
            "sha256": _sha256(baseline_path),
            "audited_commit": baseline["audited_commit"],
        },
        "inventories": {
            "derivation": {
                "workloads": (
                    "MODEL_SPECS in declared order × attention-qkv/attention-out/ffn-up/"
                    "ffn-down × decode-1/decode-7/mixed-31/mixed-96/prefill-257/"
                    "prefill-1024"
                ),
                "configs": (
                    "block_m in 16/32/64/128 × block_n in 32/64/128 for the three "
                    "declared block_k/stage/group families; serialized in coded order"
                ),
            },
            "model_configs": _model_catalog(),
            "workloads": [
                workload.to_dict() | {"key": workload.key} for workload in DEFAULT_WORKLOADS
            ],
            "configs": [config.to_dict() | {"key": config.key} for config in DEFAULT_CONFIGS],
        },
        "studies": [
            {
                "study_id": "heliostune-v1-l4-a10-transfer",
                "analysis_status": "historical_confirmatory",
                "measurement_schema": "heliostune-measurement-v1",
                "split_design": "held-out model family only",
                "collector_commit": "5919cbb4a9d7684ac835ab7bfd89879ac8c82344",
                "algorithm_commit": "5919cbb4a9d7684ac835ab7bfd89879ac8c82344",
                "collection_run": "https://modal.com/apps/mottopanikeiku/main/ap-ccgVXq137C9p6vVUxPlvXA",
                "collection_command": "historical command bound by benchmarks/manifest.json",
                "software": {
                    "python": "3.11",
                    "cuda": "12.8",
                    "torch": "2.8.0+cu128",
                    "triton": "3.4.0",
                    "modal": "1.5.3",
                },
                "protocol": _json_entry(
                    repository, "benchmarks/manifest.json", "heliostune-v1-manifest"
                ),
                "data": [v1_data],
                "results": v1_results,
            },
            {
                "study_id": "parhelion-v2-staged-transfer",
                "analysis_status": "historical_confirmatory_primary",
                "measurement_schema": "heliostune-measurement-v1",
                "split_design": "held-out model family plus exact target (M,N,K) shape",
                "collector_commit": "fe5beda065f6afb5b2c9ddd9a58e1d2b573b6abd",
                "algorithm_commit": "811b05bb65bc978e44ca8fa32ceeeab315acf391",
                "collection_runs": {
                    "t4_validation": "https://modal.com/apps/mottopanikeiku/main/ap-qxI4D2xUvtPfuhtgiSfVgm",
                    "h100_final": "https://modal.com/apps/mottopanikeiku/main/ap-y68ldw4RUmTotSEIxGdqPz",
                },
                "collection_commands": {
                    "t4": "historical invocation bound by the H100 freeze",
                    "h100": "modal run modal_bench.py --gpus H100 --replicates 3 --warmup-ms 25 --rep-ms 100 --output artifacts/h100-measurements.jsonl",
                },
                "software": {
                    "python": "3.11",
                    "cuda": "12.8",
                    "torch": "2.8.0+cu128",
                    "triton": "3.4.0",
                    "modal": "1.5.3",
                },
                "protocol_chain": protocol_chain,
                "data": [v1_data, t4_data, h100_data, v2_data],
                "results": v2_results,
                "result_links": {
                    "selection": "benchmarks/results/parhelion-t4-selection.json",
                    "final": "benchmarks/results/parhelion-h100-final.json",
                    "report": "site/index.html",
                },
            },
            {
                "study_id": "parhelion-v2-post-hoc-causal-addendum",
                "analysis_status": "post_hoc_exploratory",
                "measurement_schema": "heliostune-measurement-v1",
                "split_design": "held-out model family plus exact target (M,N,K) shape",
                "input_study_id": "parhelion-v2-staged-transfer",
                "collection_runs": "none; reuses immutable historical v2 timing matrix",
                "generator_command": ("uv run python scripts/build_parhelion_v2_addendum.py"),
                "data": [v2_data],
                "results": addendum_results,
                "reports": [addendum_report],
            },
            {
                "study_id": "parhelion-v3-h200-transfer",
                "analysis_status": "terminated_pre_h200_after_pilot_failure",
                "measurement_schema": "heliostune-measurement-v1",
                "split_design": "predeclared held-out model family plus exact target shape",
                "collector_commit": "c0cdf0e87713aff09ee5a66b23cd366d4bae7817",
                "collection_runs": {
                    "pilot": (
                        "https://modal.com/apps/mottopanikeiku/main/ap-nWqf5qjkL9CdGVuL5lWcl6"
                    ),
                    "candidate": "not invoked",
                    "a100_validation": "not invoked",
                    "h200": "not invoked",
                },
                "protocol": v3_protocol,
                "data": [],
                "results": [v3_failure],
                "files": [v3_journal],
                "result_links": {
                    "failure": "benchmarks/parhelion-v3-validation-failure.json",
                    "attempt_journal": (
                        "benchmarks/data/parhelion-v3-pilot-failure.attempts.jsonl"
                    ),
                },
            },
        ],
        "absent_freeze_aliases": baseline["absent_freeze_aliases"],
    }


def _require_equal(actual: object, expected: object, *, context: str) -> None:
    if actual != expected:
        raise ProtocolError(f"{context} mismatch: expected {expected!r}, got {actual!r}")


def _verify_data_entry(root: Path, entry: Mapping[str, object]) -> int:
    relative = nonblank_string(entry.get("path"), context="catalog data path")
    actual = _data_entry(root, relative)
    for key in (
        "schema",
        "compression",
        "compressed_bytes",
        "compressed_sha256",
        "uncompressed_bytes",
        "uncompressed_sha256",
        "rows",
        "failures",
        "gpus",
        "banks",
        "hardware",
    ):
        _require_equal(actual[key], entry.get(key), context=f"{relative} {key}")
    rows = read_measurements(root / relative)
    BenchmarkTable(rows)
    return len(rows)


def _verify_json_entry(root: Path, entry: Mapping[str, object]) -> None:
    relative = nonblank_string(entry.get("path"), context="catalog JSON path")
    actual = _json_entry(root, relative, nonblank_string(entry.get("schema"), context="schema"))
    for key in ("bytes", "sha256"):
        _require_equal(actual[key], entry.get(key), context=f"{relative} {key}")


def _verify_file_entry(root: Path, entry: Mapping[str, object]) -> None:
    relative = nonblank_string(entry.get("path"), context="catalog file path")
    kind = nonblank_string(entry.get("kind"), context="catalog file kind")
    actual = _file_entry(root, relative, kind)
    for key in ("bytes", "sha256"):
        _require_equal(actual[key], entry.get(key), context=f"{relative} {key}")


def _verify_aliases(root: Path, aliases: Mapping[str, object]) -> None:
    for alias, raw_binding in aliases.items():
        binding = _mapping_alias(raw_binding, alias)
        if (root / alias).exists():
            raise ProtocolError(f"historical freeze-only alias unexpectedly exists: {alias}")
        replacement = root / binding["published_replacement"]
        _require_equal(
            _sha256(replacement),
            binding["published_replacement_sha256"],
            context=f"alias replacement {alias}",
        )
        if binding["replacement_representation"] == "zstd_of_recorded_content":
            _bytes, digest = _decompressed_facts(replacement)
            _require_equal(digest, binding["recorded_sha256"], context=f"alias content {alias}")
        else:
            _require_equal(
                binding["published_replacement_sha256"],
                binding["recorded_sha256"],
                context=f"alias identity {alias}",
            )


def _mapping_alias(value: object, alias: str) -> dict[str, str]:
    data = exact_fields(
        value,
        required=(
            "published_replacement",
            "published_replacement_sha256",
            "recorded_sha256",
            "replacement_representation",
            "status",
        ),
        context=f"alias {alias}",
    )
    if data["status"] != "not_present_at_audited_commit":
        raise ProtocolError(f"alias {alias} has unexpected status")
    return {
        key: nonblank_string(item, context=f"alias {alias} {key}") for key, item in data.items()
    }


def _verify_frozen_v2_points(root: Path) -> None:
    rows = read_measurements(root / "benchmarks/data/parhelion-v2-measurements.jsonl.zst")
    recomputed = compare_multisource(
        rows,
        source_gpus=("L4", "A10", "T4"),
        target_gpu="H100",
        max_budget=8,
        seeds=30,
        k=16,
        temperature=2.0,
        transfer_strength=0.0,
        retrieval_k=8,
        retrieval_temperature=0.2,
        pooled_transfer_strength=0.0,
        primary_comparator="torch",
        protocol_role="final",
    )
    frozen = exact_object(
        read_json(root / "benchmarks/results/parhelion-h100-final.json"),
        context="frozen H100 result",
    )
    for key in (
        "methods",
        "auc",
        "queries_to_95_percent_reference",
        "headline",
        "primary_metrics",
        "fold_results",
    ):
        _require_equal(recomputed.get(key), frozen.get(key), context=f"frozen v2 {key}")


def verify_research_catalog(path: str | Path) -> dict[str, int]:
    """Strictly verify every catalog path, digest, count, alias, and v2 estimate."""
    catalog_path = Path(path)
    root = catalog_path.resolve().parent.parent
    catalog = exact_fields(
        read_json(catalog_path),
        required=(
            "schema_version",
            "catalog_id",
            "historical_baseline",
            "inventories",
            "studies",
            "absent_freeze_aliases",
        ),
        context="research artifact catalog",
    )
    if exact_int(catalog["schema_version"], context="catalog schema_version") != 1:
        raise SchemaError("unsupported research catalog schema version")
    _require_equal(
        catalog["inventories"],
        build_research_catalog(root)["inventories"],
        context="catalog inventories",
    )
    baseline = exact_object(catalog["historical_baseline"], context="historical_baseline")
    baseline_path = root / nonblank_string(baseline.get("path"), context="baseline path")
    _require_equal(_sha256(baseline_path), baseline.get("sha256"), context="baseline sha256")
    historical = exact_object(read_json(baseline_path), context="historical baseline")
    for relative, raw_facts in exact_object(
        historical["present_artifacts"], context="present historical artifacts"
    ).items():
        facts = exact_object(raw_facts, context=f"historical artifact {relative}")
        _require_equal(_sha256(root / relative), facts.get("sha256"), context=relative)

    studies = catalog["studies"]
    if not isinstance(studies, Sequence) or isinstance(studies, (str, bytes)):
        raise SchemaError("catalog studies must be a sequence")
    data_rows = 0
    json_artifacts = 0
    html_reports = 0
    file_artifacts = 0
    seen_data: set[str] = set()
    for raw_study in studies:
        study = exact_object(raw_study, context="catalog study")
        nonblank_string(study.get("study_id"), context="study_id")
        nonblank_string(study.get("analysis_status"), context="analysis_status")
        for raw_entry in cast(Sequence[object], study.get("data", ())):
            entry = exact_object(raw_entry, context="catalog data entry")
            relative = nonblank_string(entry.get("path"), context="catalog data path")
            if relative not in seen_data:
                data_rows += _verify_data_entry(root, entry)
                seen_data.add(relative)
        for raw_entry in cast(Sequence[object], study.get("results", ())):
            _verify_json_entry(root, exact_object(raw_entry, context="catalog result entry"))
            json_artifacts += 1
        protocol = study.get("protocol")
        if protocol is not None:
            _verify_json_entry(root, exact_object(protocol, context="catalog protocol"))
            json_artifacts += 1
        for raw_entry in cast(Sequence[object], study.get("protocol_chain", ())):
            _verify_json_entry(root, exact_object(raw_entry, context="protocol-chain entry"))
            json_artifacts += 1
        for raw_entry in cast(Sequence[object], study.get("reports", ())):
            _verify_file_entry(
                root,
                exact_object(raw_entry, context="catalog report entry"),
            )
            html_reports += 1
        for raw_entry in cast(Sequence[object], study.get("files", ())):
            _verify_file_entry(
                root,
                exact_object(raw_entry, context="catalog file entry"),
            )
            file_artifacts += 1

    aliases = exact_object(catalog["absent_freeze_aliases"], context="catalog aliases")
    _verify_aliases(root, aliases)
    _verify_frozen_v2_points(root)
    return {
        "measurement_rows": data_rows,
        "json_artifacts": json_artifacts,
        "html_reports": html_reports,
        "file_artifacts": file_artifacts,
        "aliases": len(aliases),
    }


__all__ = ["build_research_catalog", "verify_research_catalog"]
