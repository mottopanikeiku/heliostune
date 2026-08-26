from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

import heliostune.cli as cli
import heliostune.engineering_report as engineering_report
import heliostune.report as legacy_report
from heliostune.engineering_report import (
    HOPPER_STUDY_ID,
    PRECISION_STUDY_ID,
    HopperSummary,
    PrecisionSummary,
    parse_engineering_summary,
    render_engineering_report,
)
from heliostune.errors import SchemaError

_REPOSITORY = Path(__file__).resolve().parents[1]
_HOPPER_SUMMARY = _REPOSITORY / "benchmarks/results/hopper-h100-engineering-summary-v2.json"
_PRECISION_SUMMARY = _REPOSITORY / "benchmarks/results/h100-precision-probe-summary.json"


def _read_summary(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _mapping_at(summary: dict[str, object], *path: str) -> dict[str, object]:
    current = summary
    for key in path:
        value = current[key]
        assert type(value) is dict
        current = value
    return current


def test_committed_summaries_parse_as_exact_discriminated_types() -> None:
    hopper = parse_engineering_summary(_read_summary(_HOPPER_SUMMARY))
    precision = parse_engineering_summary(_read_summary(_PRECISION_SUMMARY))

    assert isinstance(hopper, HopperSummary)
    assert hopper.study_id == HOPPER_STUDY_ID
    assert hopper.global_decision == "STOP"
    assert len(hopper.candidate_selection) == 96
    assert {item.regime for item in hopper.candidate_selection} == {
        "hopper_gemm",
        "skinny_gemv",
    }
    assert hopper.publication.raw.rows == 3008
    assert hopper.publication.journal.records == 2
    assert hopper.claim_classification.inferential is False

    assert isinstance(precision, PrecisionSummary)
    assert precision.study_id == PRECISION_STUDY_ID
    assert precision.precision_finding.classification == "does not explain"
    assert precision.precision_finding.accuracy_regression is False
    assert precision.precision_finding.metrics.torch_reduced_over_torch_strict_median == 1.0
    assert precision.publication.raw.rows == 288
    assert precision.publication.journal.records == 6


@pytest.mark.parametrize(
    ("summary_path", "mutation"),
    [
        (_HOPPER_SUMMARY, lambda value: value.pop("claim_classification")),
        (
            _HOPPER_SUMMARY,
            lambda value: _mapping_at(value, "claim_classification").__setitem__("inferential", 0),
        ),
        (
            _HOPPER_SUMMARY,
            lambda value: _mapping_at(value, "collection_accounting", "attempts").__setitem__(
                "attempted", "1"
            ),
        ),
        (
            _HOPPER_SUMMARY,
            lambda value: value.__setitem__("candidate_selection", {}),
        ),
        (
            _PRECISION_SUMMARY,
            lambda value: value.__setitem__("row_count", 288.0),
        ),
        (
            _PRECISION_SUMMARY,
            lambda value: _mapping_at(value, "precision_finding", "metrics").__setitem__(
                "torch_reduced_over_torch_strict_median", "1.0"
            ),
        ),
        (
            _PRECISION_SUMMARY,
            lambda value: _mapping_at(value, "publication", "raw").__setitem__(
                "path", "../outside.json.zst"
            ),
        ),
        (
            _PRECISION_SUMMARY,
            lambda value: _mapping_at(value, "publication", "modal").__setitem__(
                "app_url", "javascript:alert(1)"
            ),
        ),
        (
            _PRECISION_SUMMARY,
            lambda value: _mapping_at(value, "publication").__setitem__("unexpected", True),
        ),
    ],
)
def test_parser_rejects_missing_unknown_and_type_coerced_values(
    summary_path: Path,
    mutation: Callable[[dict[str, object]], object],
) -> None:
    summary = _read_summary(summary_path)
    mutation(summary)

    with pytest.raises((SchemaError, AssertionError)):
        parse_engineering_summary(summary)


def test_parser_rejects_malformed_candidate_config() -> None:
    summary = _read_summary(_HOPPER_SUMMARY)
    candidates = summary["candidate_selection"]
    assert type(candidates) is list
    first = candidates[0]
    assert type(first) is dict
    config = first["best_config"]
    assert type(config) is dict
    config["split_k"] = True

    with pytest.raises(SchemaError, match="split_k must be an integer"):
        parse_engineering_summary(summary)


def test_parser_rejects_candidate_ratio_inconsistent_with_latencies() -> None:
    summary = _read_summary(_HOPPER_SUMMARY)
    candidates = summary["candidate_selection"]
    assert type(candidates) is list
    first = candidates[0]
    assert type(first) is dict
    ratio = first["torch_over_best_candidate"]
    assert type(ratio) is float
    first["torch_over_best_candidate"] = ratio + 0.01

    with pytest.raises(SchemaError, match="must equal torch_ms / best_candidate_ms"):
        parse_engineering_summary(summary)


def test_parser_rejects_regime_percentage_inconsistent_with_win_count() -> None:
    summary = _read_summary(_HOPPER_SUMMARY)
    regime = _mapping_at(summary, "regimes", "hopper_gemm")
    regime["percent_at_least_five_percent_faster"] = 0.01

    with pytest.raises(
        SchemaError,
        match="must equal workloads_at_least_five_percent_faster / workload_count",
    ):
        parse_engineering_summary(summary)


def test_unknown_engineering_study_fails_closed() -> None:
    with pytest.raises(SchemaError, match="unsupported engineering report study_id"):
        parse_engineering_summary({"study_id": "future-engineering-study"})


def test_rendering_escapes_every_summary_value(tmp_path: Path) -> None:
    summary = _read_summary(_PRECISION_SUMMARY)
    injection = '<script src="https://evil.invalid/payload.js">owned & exposed</script>'
    limitations = summary["limitations"]
    claims = _mapping_at(summary, "claim_classification")
    claim_limitations = claims["limitations"]
    assert type(limitations) is list
    assert type(claim_limitations) is list
    limitations[0] = injection
    claim_limitations[0] = injection
    hardware = _mapping_at(summary, "publication", "hardware")
    hardware["device_name"] = '<img src=x onerror="alert(1)">'
    output = tmp_path / "precision.html"

    render_engineering_report(summary, output)
    document = output.read_text(encoding="utf-8")

    assert injection not in document
    assert '<img src=x onerror="alert(1)">' not in document
    assert "&lt;script src=&quot;https://evil.invalid/payload.js&quot;&gt;" in document
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in document
    assert "<script" not in document
    assert "https://evil.invalid" not in document.replace(
        "&lt;script src=&quot;https://evil.invalid/payload.js&quot;&gt;", ""
    )


def test_hopper_report_contains_gate_scope_and_complete_selections(tmp_path: Path) -> None:
    summary = _read_summary(_HOPPER_SUMMARY)
    output = tmp_path / "hopper.html"

    render_engineering_report(summary, output)
    document = output.read_text(encoding="utf-8")
    candidates = summary["candidate_selection"]
    assert type(candidates) is list

    assert "Exploratory engineering gate" in document
    assert '<p class="decision-value">STOP</p>' in document
    assert "No superiority claim is made." in document
    assert "Same-bank limitation" in document
    assert "Values above 1 mean torch was slower" in document
    assert "Engineering regime gate results" in document
    assert "Complete per-workload post-hoc selections (96 workloads)" in document
    assert "Three-bank collection performed" in document
    assert "not a Methodology v1 EvidenceBundle" in document
    assert "methodology-compatible typed claims" in document
    assert "Actual H100 cost" in document
    assert "Publication provenance" in document
    assert "uncompressed" in document
    for item in candidates:
        assert type(item) is dict
        workload_key = item["workload_key"]
        assert type(workload_key) is str
        assert workload_key in document


def test_report_distinguishes_manifest_and_catalog_digest_bindings(tmp_path: Path) -> None:
    output = tmp_path / "hopper.html"
    render_engineering_report(_read_summary(_HOPPER_SUMMARY), output)
    document = output.read_text(encoding="utf-8")

    assert "Journal records" in document
    assert "<dt>Attempts</dt><dd>1</dd>" in document
    assert (
        "Binds the raw archive, attempt journal, and summary; does not bind this report."
        in document
    )
    assert (
        "The canonical research catalog separately binds this generated report digest." in document
    )
    assert "manifest records summary/report digests" not in document


@pytest.mark.parametrize(
    "relative_output",
    [Path("engineering.html"), Path("nested/reports/engineering.html")],
)
def test_repository_links_are_relative_to_actual_output_path(
    tmp_path: Path,
    relative_output: Path,
) -> None:
    output = tmp_path / relative_output
    render_engineering_report(_read_summary(_HOPPER_SUMMARY), output)
    expected = Path(
        os.path.relpath(
            _REPOSITORY / "benchmarks/hopper-h100-engineering-manifest-v2.json",
            start=output.resolve().parent,
        )
    ).as_posix()

    assert f'href="{expected}"' in output.read_text(encoding="utf-8")


def test_renderer_rejects_repository_source_resolving_outside_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    (benchmarks / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    summary = _read_summary(_HOPPER_SUMMARY)
    _mapping_at(summary, "publication", "raw")["path"] = "benchmarks/escape/raw.json.zst"

    with pytest.raises(SchemaError, match="resolves outside the repository"):
        render_engineering_report(summary, tmp_path / "report.html")


@pytest.mark.parametrize(
    "required",
    [
        "Exploratory engineering diagnostic",
        "does not explain",
        "Reduced torch / strict torch",
        "Strict torch / Triton",
        "Accuracy regression",
        "Comparator limitation",
        "Attempts and cost accounting",
        "Hardware, runtime, and protocol",
        "Published raw archive",
        "Publication manifest",
        "Collector source SHA-256",
        "Wheel SHA-256",
        "Limitations",
    ],
)
def test_precision_report_contains_required_disclosures(
    tmp_path: Path,
    required: str,
) -> None:
    output = tmp_path / "precision.html"
    render_engineering_report(_read_summary(_PRECISION_SUMMARY), output)

    assert required in output.read_text(encoding="utf-8")


def test_reports_are_deterministic_and_offline(tmp_path: Path) -> None:
    summary = _read_summary(_HOPPER_SUMMARY)
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"

    render_engineering_report(summary, first)
    render_engineering_report(copy.deepcopy(summary), second)

    assert first.read_bytes() == second.read_bytes()
    document = first.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in document
    assert "script-src 'none'" in document
    assert "no network dependencies" in document
    assert "<script" not in document
    assert "cdn" not in document.lower()


@pytest.mark.parametrize(
    ("summary_path", "committed_name"),
    [
        (_HOPPER_SUMMARY, "hopper-h100-engineering.html"),
        (_PRECISION_SUMMARY, "h100-precision-probe.html"),
    ],
)
def test_committed_pages_match_deterministic_renderer_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary_path: Path,
    committed_name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    rendered = tmp_path / "site" / committed_name
    render_engineering_report(_read_summary(summary_path), rendered)

    assert rendered.read_bytes() == (_REPOSITORY / "site" / committed_name).read_bytes()


def test_cli_legacy_dispatch_reads_once_and_calls_old_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary: dict[str, object] = {
        "source_gpu": "L4",
        "target_gpu": "A10",
        "methods": {},
        "study_id": [],
    }
    input_path = tmp_path / "legacy.json"
    output_path = tmp_path / "legacy.html"
    calls: list[tuple[dict[str, object], Path]] = []
    reads = 0

    def fake_read(path: Path) -> object:
        nonlocal reads
        reads += 1
        assert path == input_path
        return summary

    def fake_legacy_renderer(value: object, output: str | Path) -> None:
        assert value is summary
        calls.append((summary, Path(output)))

    def unexpected_engineering_renderer(_value: object, _output: str | Path) -> None:
        raise AssertionError("legacy report used the engineering renderer")

    monkeypatch.setattr(cli, "read_json", fake_read)
    monkeypatch.setattr(legacy_report, "render_report", fake_legacy_renderer)
    monkeypatch.setattr(
        engineering_report,
        "render_engineering_report",
        unexpected_engineering_renderer,
    )

    result = cli._report(argparse.Namespace(input=input_path, output=output_path))

    assert result == 0
    assert reads == 1
    assert calls == [(summary, output_path)]


def test_cli_integration_renders_each_strict_summary(tmp_path: Path) -> None:
    for source, expected in (
        (_HOPPER_SUMMARY, "Exploratory engineering gate"),
        (_PRECISION_SUMMARY, "Exploratory engineering diagnostic"),
    ):
        output = tmp_path / f"{source.stem}.html"
        assert cli.main(["report", str(source), "--output", str(output)]) == 0
        document = output.read_text(encoding="utf-8")
        assert expected in document
        assert "Legacy-shaped publication" in document
