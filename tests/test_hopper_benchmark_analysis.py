from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import random
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from heliostune.artifacts import strict_json_dumps
from heliostune.configs import DEFAULT_WORKLOADS, HOPPER_GEMM_CONFIGS, SKINNY_GEMV_CONFIGS

_REPO = Path(__file__).resolve().parents[1]
_ANALYZER_PATH = _REPO / "scripts/analyze_hopper_benchmark.py"


def _load_analyzer() -> ModuleType:
    name = "_test_analyze_hopper_benchmark"
    spec = importlib.util.spec_from_file_location(name, _ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ANALYZER = _load_analyzer()


def _seeds() -> dict[str, int]:
    workloads = list(DEFAULT_WORKLOADS)
    random.Random(0).shuffle(workloads)
    return {workload.key: index for index, workload in enumerate(workloads)}


def _timing(median_ms: float) -> dict[str, float]:
    return {
        "p20_ms": median_ms * 0.9,
        "median_ms": median_ms,
        "p80_ms": median_ms * 1.1,
        "wall_ms": median_ms * 100.0,
    }


def _artifact(
    speedup: float | Callable[[str, str], float] = 1.1,
) -> dict[str, Any]:
    seeds = _seeds()
    rows: list[dict[str, Any]] = []
    for workload in DEFAULT_WORKLOADS:
        regime = "skinny_gemv" if workload.m <= 8 else "hopper_gemm"
        configs = SKINNY_GEMV_CONFIGS if workload.m <= 8 else HOPPER_GEMM_CONFIGS
        workload_speedup = speedup(workload.key, regime) if callable(speedup) else speedup
        for config in configs:
            candidate_ms = 1.0
            torch_ms = candidate_ms * workload_speedup
            rows.append(
                {
                    "workload_key": workload.key,
                    "workload": workload.to_dict(),
                    "regime": regime,
                    "config_kind": regime,
                    "config_key": config.key,
                    "config": config.to_dict(),
                    "bank": 0,
                    "seed": seeds[workload.key],
                    "latency": _timing(candidate_ms),
                    "torch": _timing(torch_ms),
                    "correct": True,
                    "max_abs_error": 0.01,
                }
            )
    config_manifest = {
        "hopper_gemm": [config.to_dict() for config in HOPPER_GEMM_CONFIGS],
        "skinny_gemv": [config.to_dict() for config in SKINNY_GEMV_CONFIGS],
    }
    return {
        "schema_version": 1,
        "study_id": "hopper-h100-engineering-benchmark",
        "analysis_status": "post_hoc_exploratory",
        "gpu": "H100",
        "gpu_selector": "H100!",
        "hardware": {
            "gpu": "H100",
            "device_name": "NVIDIA H100 80GB HBM3",
            "compute_capability": [9, 0],
            "multiprocessor_count": 132,
            "total_memory_gb": 79.1788330078125,
            "cuda_version": "12.8",
            "torch_version": "2.8.0+cu128",
            "triton_version": "3.4.0",
        },
        "bank": 0,
        "protocol": {
            "warmup_ms": 25,
            "rep_ms": 100,
            "quantiles": [0.2, 0.5, 0.8],
            "candidate_policy": {
                "skinny_gemv": {
                    "condition": "m <= 8",
                    "config_set": "SKINNY_GEMV_CONFIGS",
                    "config_count": 48,
                },
                "hopper_gemm": {
                    "condition": "m > 8",
                    "config_set": "HOPPER_GEMM_CONFIGS",
                    "config_count": 23,
                },
            },
            "expected_workloads": 96,
            "expected_skinny_workloads": 32,
            "expected_hopper_workloads": 64,
            "expected_skinny_rows": 1536,
            "expected_hopper_rows": 1472,
            "expected_candidate_rows": 3008,
            "torch_measurements": 96,
        },
        "correctness_gate": {
            "artifact": "artifacts/hopper-correctness.json",
            "artifact_sha256": "a" * 64,
            "manifest": "artifacts/hopper-correctness.json.manifest.json",
            "manifest_sha256": "b" * 64,
        },
        "configs": config_manifest,
        "config_manifest_sha256": hashlib.sha256(
            strict_json_dumps(config_manifest, compact=True).encode("utf-8")
        ).hexdigest(),
        "workloads": [workload.to_dict() for workload in DEFAULT_WORKLOADS],
        "rows": rows,
        "verified": True,
    }


def _refresh_config_digest(artifact: dict[str, Any]) -> None:
    artifact["config_manifest_sha256"] = hashlib.sha256(
        strict_json_dumps(artifact["configs"], compact=True).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("compute_capability", 0), True),
        (("compute_capability", 0), 9.0),
        (("multiprocessor_count",), True),
        (("multiprocessor_count",), 132.0),
        (("total_memory_gb",), 140.0),
        (("device_name",), "NVIDIA H200"),
        (("cuda_version",), None),
        (("torch_version",), True),
        (("triton_version",), ""),
    ],
)
def test_invalid_or_incomplete_hardware_is_rejected(
    path: tuple[str, int] | tuple[str],
    replacement: object,
) -> None:
    artifact = _artifact()
    hardware = artifact["hardware"]
    if len(path) == 2:
        hardware[path[0]][path[1]] = replacement
    else:
        hardware[path[0]] = replacement

    with pytest.raises(ValueError):
        _ANALYZER.analyze_artifact(artifact)


def test_missing_hardware_field_is_rejected() -> None:
    artifact = _artifact()
    del artifact["hardware"]["triton_version"]

    with pytest.raises(ValueError, match="missing fields"):
        _ANALYZER.analyze_artifact(artifact)


@pytest.mark.parametrize(
    ("regime", "field"),
    [
        *(
            ("skinny_gemv", field)
            for field in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "split_k")
        ),
        *(
            ("hopper_gemm", field)
            for field in (
                "block_m",
                "block_n",
                "block_k",
                "num_warps",
                "num_stages",
                "group_m",
            )
        ),
    ],
)
def test_config_manifest_integer_fields_reject_true_with_recomputed_digest(
    regime: str,
    field: str,
) -> None:
    artifact = _artifact()
    artifact["configs"][regime][0][field] = True
    _refresh_config_digest(artifact)

    with pytest.raises(ValueError):
        _ANALYZER.analyze_artifact(artifact)


@pytest.mark.parametrize("field", ["epilogue_subtile", "warp_specialize"])
def test_config_manifest_boolean_fields_reject_equal_integers_with_recomputed_digest(
    field: str,
) -> None:
    artifact = _artifact()
    original = artifact["configs"]["hopper_gemm"][0][field]
    artifact["configs"]["hopper_gemm"][0][field] = int(original)
    _refresh_config_digest(artifact)

    with pytest.raises(ValueError):
        _ANALYZER.analyze_artifact(artifact)


@pytest.mark.parametrize("field", ["m", "n", "k"])
def test_workload_manifest_integer_fields_reject_true(field: str) -> None:
    artifact = _artifact()
    artifact["workloads"][0][field] = True

    with pytest.raises(ValueError):
        _ANALYZER.analyze_artifact(artifact)


@pytest.mark.parametrize("field", ["m", "n", "k"])
def test_embedded_row_workload_integer_fields_reject_true(field: str) -> None:
    artifact = _artifact()
    artifact["rows"][0]["workload"][field] = True

    with pytest.raises(ValueError):
        _ANALYZER.analyze_artifact(artifact)


@pytest.mark.parametrize(
    ("regime", "field"),
    [
        *(
            ("skinny_gemv", field)
            for field in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "split_k")
        ),
        *(
            ("hopper_gemm", field)
            for field in (
                "block_m",
                "block_n",
                "block_k",
                "num_warps",
                "num_stages",
                "group_m",
            )
        ),
    ],
)
def test_embedded_row_config_integer_fields_reject_true(regime: str, field: str) -> None:
    artifact = _artifact()
    row = next(candidate for candidate in artifact["rows"] if candidate["regime"] == regime)
    row["config"][field] = True

    with pytest.raises(ValueError):
        _ANALYZER.analyze_artifact(artifact)


@pytest.mark.parametrize("replacement", [True, 48.0])
def test_candidate_policy_config_count_requires_exact_integer(replacement: object) -> None:
    artifact = _artifact()
    artifact["protocol"]["candidate_policy"]["skinny_gemv"]["config_count"] = replacement

    with pytest.raises(ValueError):
        _ANALYZER.analyze_artifact(artifact)


def test_threshold_boundaries_are_inclusive() -> None:
    exact = _ANALYZER.analyze_artifact(_artifact(1.05))
    assert exact.regime("skinny_gemv").geometric_mean_speedup == pytest.approx(1.05)
    assert exact.regime("skinny_gemv").passes_gate
    assert exact.regime("hopper_gemm").passes_gate

    skinny_keys = [workload.key for workload in DEFAULT_WORKLOADS if workload.m <= 8]
    winner_keys = set(skinny_keys[:8])
    loser_speedup = math.exp((32 * math.log(1.050001) - 8 * math.log(1.2)) / 24)

    def boundary_speedup(workload_key: str, regime: str) -> float:
        if regime == "hopper_gemm":
            return 0.9
        return 1.2 if workload_key in winner_keys else loser_speedup

    boundary = _ANALYZER.analyze_artifact(_artifact(boundary_speedup))
    skinny = boundary.regime("skinny_gemv")
    assert skinny.workloads_at_least_five_percent_faster == 8
    assert skinny.percent_at_least_five_percent_faster == 25.0
    assert skinny.geometric_mean_speedup == pytest.approx(1.050001)
    assert skinny.passes_gate


def test_one_regime_can_pass_while_the_other_fails() -> None:
    artifact = _artifact(lambda _key, regime: 1.1 if regime == "skinny_gemv" else 0.9)
    result = _ANALYZER.analyze_artifact(artifact)

    assert result.regime("skinny_gemv").passes_gate
    assert not result.regime("hopper_gemm").passes_gate
    assert result.proceed
    report = _ANALYZER.format_report(result)
    assert "skinny_gemv: PROCEED" in report
    assert "hopper_gemm: STOP" in report
    assert "Global: PROCEED" in report
    assert "post-hoc exploratory" in report
    assert "not confirmatory" in report

    stopped = _ANALYZER.analyze_artifact(_artifact(0.9))
    assert not stopped.proceed
    assert "Global: STOP" in _ANALYZER.format_report(stopped)


def test_duplicate_workload_config_row_is_rejected() -> None:
    artifact = _artifact()
    artifact["rows"][-1] = copy.deepcopy(artifact["rows"][0])

    with pytest.raises(ValueError, match="duplicates workload/config"):
        _ANALYZER.analyze_artifact(artifact)


def test_missing_candidate_row_is_rejected() -> None:
    artifact = _artifact()
    artifact["rows"].pop()

    with pytest.raises(ValueError, match="exactly 3008"):
        _ANALYZER.analyze_artifact(artifact)


def test_wrong_regime_and_config_kind_are_rejected() -> None:
    artifact = _artifact()
    artifact["rows"][0]["regime"] = "hopper_gemm"

    with pytest.raises(ValueError, match="wrong regime/config_kind"):
        _ANALYZER.analyze_artifact(artifact)

    artifact = _artifact()
    artifact["rows"][0]["config_kind"] = "hopper_gemm"
    with pytest.raises(ValueError, match="wrong regime/config_kind"):
        _ANALYZER.analyze_artifact(artifact)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_nonfinite_or_nonpositive_medians_are_rejected(value: float) -> None:
    artifact = _artifact()
    artifact["rows"][0]["latency"]["median_ms"] = value

    with pytest.raises(ValueError, match="row 0 latency median_ms must be (finite|positive)"):
        _ANALYZER.analyze_artifact(artifact)


def test_speedup_ratio_direction_is_torch_over_candidate() -> None:
    archive_key = DEFAULT_WORKLOADS[0].key
    result = _ANALYZER.analyze_artifact(
        _artifact(1.2),
        archive_ratios={archive_key: 0.625},
    )
    first = result.workloads[0]

    assert first.torch_ms == pytest.approx(1.2)
    assert first.best_candidate_ms == pytest.approx(1.0)
    assert first.torch_over_best_candidate == pytest.approx(1.2)
    assert first.archive_torch_over_best_triton == pytest.approx(0.625)
    assert result.regime(first.regime).passes_gate
    assert "Ratios are torch_ms / best_candidate_ms" in _ANALYZER.format_report(result)
