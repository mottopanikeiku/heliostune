from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

import heliostune.v3_engine as engine
from heliostune.configs import (
    PARHELION_V3_CANDIDATE_CONFIGS,
    PARHELION_V3_OFFICIAL_CONFIG_KEYS,
    Workload,
)
from heliostune.schema import HardwareProfile, Measurement

_CONFIGS = tuple(sorted(PARHELION_V3_CANDIDATE_CONFIGS[:16], key=lambda item: item.key))
_WORKLOADS = (
    Workload(1, 32, 32, "alpha", "attention", "decode"),
    Workload(7, 64, 32, "alpha", "feedforward", "decode"),
    Workload(1, 32, 32, "beta", "attention", "decode"),
    Workload(31, 128, 32, "beta", "feedforward", "mixed"),
)
_HARDWARE = (
    HardwareProfile("L4", "NVIDIA L4", (8, 9), 58, 22.0),
    HardwareProfile("A10", "NVIDIA A10", (8, 6), 72, 22.0),
    HardwareProfile("A100-80GB", "NVIDIA A100-SXM4-80GB", (8, 0), 108, 80.0),
)
_CANONICALIZER = Path(__file__).resolve().parents[1] / "scripts/canonicalize_parhelion_v3_a100.py"


def _load_canonicalizer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("canonicalize_parhelion_v3_a100", _CANONICALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix() -> tuple[Measurement, ...]:
    rows: list[Measurement] = []
    for gpu_index, hardware in enumerate(_HARDWARE):
        for workload_index, workload in enumerate(_WORKLOADS):
            for config_index, config in enumerate(_CONFIGS):
                for bank in range(5):
                    base = 1.0 + workload_index * 0.1 + gpu_index * 0.05
                    factor = 0.8 + ((config_index * 7 + workload_index * 3) % 16) / 20
                    latency = base * factor * (1 + bank * 0.0005 * (config_index + 1))
                    torch_latency = base * 1.8 + bank * 0.01
                    rows.append(
                        Measurement(
                            hardware=hardware,
                            workload=workload,
                            config=config,
                            bank=bank,
                            latency_ms=latency,
                            torch_latency_ms=torch_latency,
                            correct=True,
                            max_abs_error=0.0,
                            latency_p20_ms=latency * 0.99,
                            latency_p80_ms=latency * 1.01,
                            torch_latency_p20_ms=torch_latency * 0.99,
                            torch_latency_p80_ms=torch_latency * 1.01,
                            compile_ms=0.5 + config_index / 100,
                            benchmark_wall_ms=100.0 + latency,
                            torch_benchmark_wall_ms=100.0 + torch_latency,
                        )
                    )
    return tuple(rows)


def test_mixed_a100_normalization_changes_device_name_only() -> None:
    canonicalizer = _load_canonicalizer()
    original = _matrix()
    mixed = tuple(
        replace(
            row,
            hardware=replace(row.hardware, device_name="NVIDIA A100 80GB PCIe"),
        )
        if row.hardware.gpu == "A100-80GB" and row.bank == 0
        else row
        for row in original
    )

    normalized, counts = canonicalizer.canonicalize_rows(mixed)

    assert counts == {
        "NVIDIA A100 80GB PCIe": 64,
        "NVIDIA A100-SXM4-80GB": 256,
    }
    for before, after in zip(mixed, normalized, strict=True):
        if before.hardware.gpu == "A100-80GB":
            assert after.hardware.device_name == canonicalizer._CANONICAL_DEVICE_NAME
            assert replace(after, hardware=before.hardware) == before
        else:
            assert after is before


@pytest.fixture
def prepared(monkeypatch: pytest.MonkeyPatch) -> engine.V3Prepared:
    monkeypatch.setattr(engine, "require_v3_runtime", lambda _protocol: None)
    return engine.prepare_v3(
        {},
        _matrix(),
        source_gpus=("L4", "A10"),
        target_gpu="A100-80GB",
        retained_config_keys=tuple(config.key for config in _CONFIGS),
        official_config_keys=tuple(
            config.key for config in _CONFIGS if config.key in PARHELION_V3_OFFICIAL_CONFIG_KEYS
        ),
        seeds=(0, 1),
    )


def test_v3_zero_strength_stream_invariants(prepared: engine.V3Prepared) -> None:
    retrieval = engine.evaluate_v3_retrieval(prepared, k=1, temperature=0.2)
    cold = engine.evaluate_v3_cold(prepared)
    pooled = engine.evaluate_v3_pooled(prepared, transfer_strength=0.0)
    parhelion = engine.evaluate_v3_parhelion(
        prepared,
        k=1,
        temperature=0.2,
        transfer_strength=0.0,
        retrieval_evaluation=retrieval,
    )
    no_transfer = engine.evaluate_v3_no_transfer(
        prepared,
        k=1,
        temperature=0.2,
        retrieval_evaluation=retrieval,
    )
    anchored = engine.evaluate_v3_anchored_cold(prepared, retrieval)

    assert pooled == engine.V3Evaluation(
        "pooled_source_thompson",
        cold.recommendations,
        cold.probes,
    )
    assert parhelion.recommendations == no_transfer.recommendations
    assert parhelion.probes == no_transfer.probes
    retrieval_curve = engine.evaluation_seed_curves(prepared, retrieval)[0]
    for evaluation in (parhelion, anchored):
        for curve in engine.evaluation_seed_curves(prepared, evaluation):
            assert curve[0] == pytest.approx(retrieval_curve[0])


def test_v3_sensitivity_banks_never_change_primary_trace(
    prepared: engine.V3Prepared,
) -> None:
    random = engine.evaluate_v3_random(prepared)
    primary_before = engine.evaluation_seed_curves(prepared, random, bank=2)
    sensitivity_3 = engine.evaluation_seed_curves(prepared, random, bank=3)
    sensitivity_4 = engine.evaluation_seed_curves(prepared, random, bank=4)
    primary_after = engine.evaluation_seed_curves(prepared, random, bank=2)

    assert primary_after == primary_before
    assert sensitivity_3 != sensitivity_4
    for seed_probes in random.probes:
        for fold_index, fold in enumerate(prepared.folds):
            for workload in fold.target_workloads:
                selected = [
                    dict(round_pairs)[workload.key] for round_pairs in seed_probes[fold_index]
                ]
                assert len(selected) == len(set(selected)) == 16


def test_v3_selection_has_exact_method_local_grid_sizes(
    prepared: engine.V3Prepared,
) -> None:
    selection = engine.select_v3_parameters(prepared)

    scores = selection["candidate_scores"]
    assert len(scores["multisource_retrieval"]) == 12
    assert len(scores["pooled_source_thompson"]) == 4
    assert len(scores["parhelion_thompson"]) == 48
    assert set(selection["selected"]) == {
        "multisource_retrieval",
        "pooled_source_thompson",
        "parhelion_thompson",
    }
