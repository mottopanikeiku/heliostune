"""Build and byte-verify the non-destructive Parhelion v2 causal addendum."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from heliostune.artifacts import (
    read_json,
    read_measurements,
    write_json_atomic,
)
from heliostune.report import render_report
from heliostune.v2_addendum import build_v2_addendum_summary

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "benchmarks/data/parhelion-v2-measurements.jsonl.zst"
_HISTORICAL_RESULT = _REPO / "benchmarks/results/parhelion-h100-final.json"
_RESULT = _REPO / "benchmarks/results/parhelion-v2-addendum.json"
_REPORT = _REPO / "site/parhelion-v2-addendum.html"
_MANIFEST = _REPO / "benchmarks/parhelion-v2-addendum-manifest.json"
_PROTOCOL = _REPO / "benchmarks/parhelion-v2-development-protocol.json"
_FREEZE = _REPO / "benchmarks/parhelion-v2-h100-freeze.json"
_BASELINE = _REPO / "benchmarks/historical-artifact-baseline.json"
_IMPLEMENTATION_PATHS = (
    "src/heliostune/multisource_engine.py",
    "src/heliostune/v2_addendum.py",
    "src/heliostune/uncertainty.py",
    "src/heliostune/report_model.py",
    "src/heliostune/report.py",
    "src/heliostune/report.css",
    "scripts/build_parhelion_v2_addendum.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _historical_digests() -> dict[str, str]:
    baseline = cast(Mapping[str, object], read_json(_BASELINE))
    present = cast(Mapping[str, Mapping[str, object]], baseline["present_artifacts"])
    return {relative: _sha256(_REPO / relative) for relative in present}


def _implementation_digests() -> dict[str, str]:
    return {relative: _sha256(_REPO / relative) for relative in _IMPLEMENTATION_PATHS}


def _assert_frozen_values(summary: Mapping[str, object]) -> None:
    auc = cast(Mapping[str, Mapping[str, object]], summary["auc"])
    methods = cast(Mapping[str, list[Mapping[str, object]]], summary["methods"])
    historical = cast(Mapping[str, object], summary["historical_confirmatory_endpoint"])
    evidence = cast(Mapping[str, object], historical["evidence"])
    invariant = cast(Mapping[str, object], summary["budget_one_invariant"])
    if auc["parhelion_thompson"]["mean"] != 0.9502592348438624:
        raise RuntimeError("addendum changed frozen Parhelion AUC1-8")
    if methods["parhelion_thompson"][-1]["mean_fraction_oracle"] != 0.9965225333288278:
        raise RuntimeError("addendum changed frozen Parhelion budget-8 fraction")
    if evidence["mean_auc_delta"] != -0.6600001975593753:
        raise RuntimeError("addendum changed the historical torch comparison")
    if invariant.get("verified") is not True:
        raise RuntimeError("addendum did not verify the shared budget-one anchor")
    pooled = methods["pooled_source_thompson"]
    cold = methods["cold_thompson"]
    if [point["mean_fraction_oracle"] for point in pooled] != [
        point["mean_fraction_oracle"] for point in cold
    ]:
        raise RuntimeError("selected pooled strength zero no longer equals cold")


def _manifest(
    result: Path,
    report: Path,
    historical_digests: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "study_id": "parhelion-v2-post-hoc-causal-addendum",
        "analysis_status": "post_hoc_exploratory",
        "generator_command": "uv run python scripts/build_parhelion_v2_addendum.py",
        "check_command": "uv run python scripts/build_parhelion_v2_addendum.py --check",
        "inputs": {
            "measurement_archive": {
                "path": "benchmarks/data/parhelion-v2-measurements.jsonl.zst",
                "sha256": _sha256(_DATA),
            },
            "development_protocol": {
                "path": "benchmarks/parhelion-v2-development-protocol.json",
                "sha256": _sha256(_PROTOCOL),
            },
            "h100_freeze": {
                "path": "benchmarks/parhelion-v2-h100-freeze.json",
                "sha256": _sha256(_FREEZE),
            },
            "historical_final_result": {
                "path": "benchmarks/results/parhelion-h100-final.json",
                "sha256": _sha256(_HISTORICAL_RESULT),
                "confirmatory_endpoint": "unchanged",
            },
            "historical_baseline": {
                "path": "benchmarks/historical-artifact-baseline.json",
                "sha256": _sha256(_BASELINE),
            },
        },
        "implementation_sha256": _implementation_digests(),
        "historical_artifact_sha256": dict(historical_digests),
        "outputs": {
            "result": {
                "path": "benchmarks/results/parhelion-v2-addendum.json",
                "bytes": result.stat().st_size,
                "sha256": _sha256(result),
            },
            "report": {
                "path": "site/parhelion-v2-addendum.html",
                "bytes": report.stat().st_size,
                "sha256": _sha256(report),
            },
        },
        "invariants": {
            "parhelion_auc1_8": 0.9502592348438624,
            "parhelion_budget8_fraction": 0.9965225333288278,
            "historical_torch_auc_delta": -0.6600001975593753,
            "pooled_zero_strength_equals_cold": True,
            "parhelion_and_anchored_cold_share_budget_one": True,
            "new_contrasts_have_superiority_claims": False,
        },
    }


def _build(result: Path, report: Path, manifest: Path) -> None:
    before = _historical_digests()
    rows = read_measurements(_DATA)
    historical = cast(Mapping[str, object], read_json(_HISTORICAL_RESULT))
    summary = build_v2_addendum_summary(rows, historical)
    provenance = cast(dict[str, object], summary["provenance"])
    provenance["addendum_manifest"] = "benchmarks/parhelion-v2-addendum-manifest.json"
    provenance["input_sha256"] = _sha256(_DATA)
    provenance["implementation_sha256"] = _implementation_digests()
    _assert_frozen_values(summary)
    write_json_atomic(result, summary)
    render_report(summary, report)
    write_json_atomic(manifest, _manifest(result, report, before))
    after = _historical_digests()
    if after != before:
        changed = sorted(path for path in before if before[path] != after.get(path))
        raise RuntimeError(f"historical artifacts changed while building addendum: {changed}")


def _compare(committed: Path, generated: Path) -> None:
    if not committed.is_file():
        raise RuntimeError(f"committed addendum output is missing: {committed}")
    if committed.read_bytes() != generated.read_bytes():
        raise RuntimeError(f"addendum output is stale: {committed}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        with tempfile.TemporaryDirectory(prefix="parhelion-v2-addendum-") as temporary:
            root = Path(temporary)
            result = root / "parhelion-v2-addendum.json"
            report = root / "parhelion-v2-addendum.html"
            manifest = root / "parhelion-v2-addendum-manifest.json"
            _build(result, report, manifest)
            _compare(_RESULT, result)
            _compare(_REPORT, report)
            _compare(_MANIFEST, manifest)
    else:
        _build(_RESULT, _REPORT, _MANIFEST)
    print("Parhelion v2 addendum outputs are byte-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
