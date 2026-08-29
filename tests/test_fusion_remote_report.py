from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import heliostune.cli as cli
from heliostune.engineering_report import (
    FUSION_REMOTE_STUDY_ID,
    FusionRemoteSummary,
    parse_engineering_summary,
    render_engineering_report,
)
from heliostune.errors import SchemaError

_REPOSITORY = Path(__file__).resolve().parents[1]
_SUMMARY = _REPOSITORY / "benchmarks/results/fusion-remote-exploratory-summary.json"
_COMMITTED_REPORT = _REPOSITORY / "site/fusion-remote-exploratory.html"
_UNRESOLVED_JOURNAL_STATES = (
    "intent",
    "spawned",
    "retrieval_started",
    "cancellation_requested",
    "unresolved",
)
_UNRESOLVED_SEQUENCE = (
    "retrieval returned 401; client then requested cancellation; "
    "terminal provider outcome/cancellation success remained unresolved"
)


def _read_summary() -> dict[str, object]:
    value = json.loads(_SUMMARY.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _mapping_at(summary: dict[str, object], *path: str) -> dict[str, object]:
    current = summary
    for key in path:
        value = current[key]
        assert type(value) is dict
        current = value
    return current


def _array_at(summary: dict[str, object], key: str) -> list[object]:
    value = summary[key]
    assert type(value) is list
    return value


def _item_mapping(items: list[object], index: int) -> dict[str, object]:
    value = items[index]
    assert type(value) is dict
    return value


def test_committed_fusion_summary_parses_as_exact_discriminated_type() -> None:
    parsed = parse_engineering_summary(_read_summary())

    assert isinstance(parsed, FusionRemoteSummary)
    assert parsed.study_id == FUSION_REMOTE_STUDY_ID
    assert parsed.counts.attempts == 4
    assert parsed.counts.completed == 2
    assert parsed.counts.failed == 0
    assert parsed.counts.unresolved == 2
    assert tuple(attempt.status for attempt in parsed.attempts) == (
        "unresolved",
        "unresolved",
        "completed",
        "completed",
    )
    unresolved_attempts = tuple(
        attempt for attempt in parsed.attempts if attempt.status == "unresolved"
    )
    assert all(
        attempt.journal_states == _UNRESOLVED_JOURNAL_STATES for attempt in unresolved_attempts
    )
    assert all(
        attempt.terminal_detail == "RemoteError: AuthError(\"Received :status = '401'\")"
        for attempt in unresolved_attempts
    )
    assert all(
        attempt.app.identity_provenance == "operator_recorded" for attempt in parsed.attempts
    )
    assert all(
        attempt.call.identity_provenance == "artifact_bound_remote_journal"
        for attempt in parsed.attempts
    )
    assert all(result.fusion_claim is False for result in parsed.completed_results)
    assert all(result.publication_eligible is False for result in parsed.completed_results)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("schema", 1),
        lambda value: value.__setitem__("attempts", {}),
        lambda value: _mapping_at(value, "counts").__setitem__("attempts", "4"),
        lambda value: _mapping_at(value, "counts").__setitem__("completed", 3),
        lambda value: _mapping_at(value, "methodology").__setitem__("fusion_claim", 0),
        lambda value: _mapping_at(value, "methodology").__setitem__("publication_eligible", True),
        lambda value: _mapping_at(value, "provider_accounting").__setitem__("actual_cost_usd", 0.0),
        lambda value: _mapping_at(value, "provider_accounting").__setitem__("unexpected", None),
        lambda value: _item_mapping(_array_at(value, "attempts"), 0).__setitem__(
            "terminal_detail", None
        ),
        lambda value: _item_mapping(_array_at(value, "attempts"), 0).__setitem__(
            "status", "completed"
        ),
        lambda value: _mapping_at(
            _item_mapping(_array_at(value, "attempts"), 0), "app"
        ).__setitem__("identity_provenance", "artifact_bound"),
        lambda value: _mapping_at(
            _item_mapping(_array_at(value, "completed_results"), 0),
            "metrics",
            "descriptive_ratios",
        ).__setitem__("candidate_to_reference_median", 0.5),
        lambda value: _item_mapping(_array_at(value, "completed_results"), 0).__setitem__(
            "attempt_id", "gated-mlp-01-unresolved"
        ),
        lambda value: _item_mapping(_array_at(value, "completed_results"), 0).__setitem__(
            "publication_eligible", True
        ),
    ],
)
def test_fusion_parser_rejects_types_coherence_ratios_counts_and_statuses(
    mutation: Callable[[dict[str, object]], object],
) -> None:
    summary = _read_summary()
    mutation(summary)

    with pytest.raises(SchemaError):
        parse_engineering_summary(summary)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("head_commit", "distinct historical HEAD"),
        ("source_sha256", "distinct historical source"),
        ("wheel_sha256", "distinct historical wheel"),
    ],
)
def test_fusion_parser_rejects_historical_build_collapse(field: str, message: str) -> None:
    summary = _read_summary()
    attempts = _array_at(summary, "attempts")
    first = _item_mapping(attempts, 0)
    second = _item_mapping(attempts, 1)
    second[field] = first[field]

    with pytest.raises(SchemaError, match=message):
        parse_engineering_summary(summary)


def test_fusion_parser_rejects_reordered_unresolved_journal_states() -> None:
    summary = _read_summary()
    attempt = _item_mapping(_array_at(summary, "attempts"), 0)
    attempt["journal_states"] = [
        "intent",
        "spawned",
        "retrieval_started",
        "unresolved",
        "cancellation_requested",
    ]

    with pytest.raises(SchemaError, match="journal_states contradict unresolved status"):
        parse_engineering_summary(summary)


def test_fusion_rendering_escapes_summary_values(tmp_path: Path) -> None:
    summary = _read_summary()
    injection = '<img src=x onerror="alert(1)"> H100 & exposed'
    for item in _array_at(summary, "completed_results"):
        result = _item_mapping([item], 0)
        hardware = _mapping_at(result, "hardware")
        hardware["device_name"] = injection
    output = tmp_path / "fusion.html"

    render_engineering_report(summary, output)
    document = output.read_text(encoding="utf-8")

    assert injection not in document
    assert '<img src=x onerror="alert(1)">' not in document
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; H100 &amp; exposed" in document
    assert "<script" not in document


@pytest.mark.parametrize(
    "required",
    [
        "Exploratory receipt status",
        _UNRESOLVED_SEQUENCE,
        "No fusion or superiority claim.",
        "fusion_claim=false",
        "publication_eligible=false",
        "operator-recorded; artifact binding: none",
        "artifact-bound remote journal",
        "Returned correctness observations",
        "Returned compile observations",
        "Returned timing observations",
        "Raw-sample stability boundary",
        "candidate / reference",
        "reference / candidate",
        "no stability threshold",
        "Provider starts or restarts",
        "Unknown / unobservable",
        "Actual cost",
        "Attestation",
        "None present",
        "Historical build boundary",
        "four different historical HEAD commits",
        "report_status=not_created",
        "fusion-remote-exploratory.json.zst",
        "fusion-remote-exploratory-manifest.json",
    ],
)
def test_fusion_report_contains_required_disclosures(tmp_path: Path, required: str) -> None:
    output = tmp_path / "fusion.html"
    render_engineering_report(_read_summary(), output)

    assert required in output.read_text(encoding="utf-8")


def test_fusion_report_states_unresolved_chronology_without_reversing_causality(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fusion.html"
    render_engineering_report(_read_summary(), output)

    document = output.read_text(encoding="utf-8")
    assert document.count(_UNRESOLVED_SEQUENCE) == 4
    assert "after cancellation" not in document.lower()
    assert "401 after" not in document.lower()
    assert "unresolved after 401" not in document.lower()


def test_fusion_report_contains_every_attempt_identity_and_build_binding(tmp_path: Path) -> None:
    summary = _read_summary()
    output = tmp_path / "fusion.html"
    render_engineering_report(summary, output)
    document = output.read_text(encoding="utf-8")

    for item in _array_at(summary, "attempts"):
        attempt = _item_mapping([item], 0)
        app = _mapping_at(attempt, "app")
        call = _mapping_at(attempt, "call")
        for key in ("attempt_id", "head_commit", "source_sha256", "wheel_sha256"):
            value = attempt[key]
            assert type(value) is str
            assert value in document
        for value in (app["app_id"], call["function_call_id"]):
            assert type(value) is str
            assert value in document


def test_fusion_report_is_deterministic_accessible_and_offline(tmp_path: Path) -> None:
    summary = _read_summary()
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"

    render_engineering_report(summary, first)
    render_engineering_report(copy.deepcopy(summary), second)

    assert first.read_bytes() == second.read_bytes()
    document = first.read_text(encoding="utf-8")
    assert '<html lang="en">' in document
    assert 'name="viewport"' in document
    assert 'href="#main">Skip to report</a>' in document
    assert '<main id="main">' in document
    assert 'role="region"' in document
    assert "Content-Security-Policy" in document
    assert "script-src 'none'" in document
    assert "no scripts · no network dependencies" in document
    assert "<script" not in document
    assert "cdn" not in document.lower()


def test_committed_fusion_page_matches_deterministic_renderer_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    rendered = tmp_path / "site/fusion-remote-exploratory.html"

    render_engineering_report(_read_summary(), rendered)

    assert rendered.read_bytes() == _COMMITTED_REPORT.read_bytes()


def test_cli_report_dispatches_fusion_summary_to_strict_renderer(tmp_path: Path) -> None:
    output = tmp_path / "fusion.html"

    assert cli.main(["report", str(_SUMMARY), "--output", str(output)]) == 0

    document = output.read_text(encoding="utf-8")
    assert "H100 fusion remote receipts" in document
    assert "Exploratory receipt status" in document
    assert "Legacy-shaped publication" not in document
