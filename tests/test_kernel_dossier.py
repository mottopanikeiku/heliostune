from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from heliostune.action_configs import SplitKConfig, candidates_for
from heliostune.configs import DEFAULT_WORKLOADS, HopperGemmConfig, KernelConfig, Workload
from heliostune.tooling.kernel_dossier import build_dossier, predict

_REPO = Path(__file__).resolve().parents[1]
_SUMMARY: dict[str, Any] = json.loads(
    (_REPO / "results/action-expansion-summary.json").read_text(encoding="utf-8")
)
_QWEN_DOWN = {
    w.m: w for w in DEFAULT_WORKLOADS if w.model == "qwen2.5-7b" and w.projection == "ffn-down"
}


def test_ordinary_tile_counts_padding_programs_and_logical_bytes_by_hand() -> None:
    # M=96 pads to one 128-row tile; N=3584 is 28 tiles of 128; K=18944 is 296 tiles of 64.
    workload = _QWEN_DOWN[96]
    p = predict(workload, KernelConfig(128, 128, 64, 4, 5), sms=132)
    assert (p["output_tiles"], p["matmul_programs"], p["kernel_launches"]) == (28, 28, 1)
    assert p["k_tiles_per_split"] == 296
    assert p["masked_row_fraction"] == pytest.approx(0.25)
    assert p["padded_dot_flops"] == 2 * 128 * 3584 * 18944
    assert p["useful_dot_fraction"] == pytest.approx(0.75)
    assert p["programs_per_sm"] == pytest.approx(28 / 132)
    assert p["persistent_tiles_per_program_max"] is None
    assert p["persistent_tile_balance"] is None
    assert p["workspace_bytes"] == 0 and p["reduction_programs"] == 0
    ideal = 2 * (96 * 18944 + 18944 * 3584 + 96 * 3584)
    assert p["ideal_tensor_bytes"] == ideal
    assert p["ideal_intensity_flops_per_byte"] == pytest.approx(2 * 96 * 3584 * 18944 / ideal)
    # Without inter-tile reuse, A is reloaded for each of 28 N tiles; B once for one M tile.
    assert p["no_cross_tile_reuse_bytes"] == 2 * (28 * 96 * 18944 + 18944 * 3584 + 96 * 3584)
    assert p["label"].startswith("INFERENCE")


def test_persistent_programs_are_capped_at_sms_with_tile_imbalance() -> None:
    # M=1024: 8 x 28 = 224 tiles on 132 programs, so 92 programs take two tiles.
    p = predict(_QWEN_DOWN[1024], HopperGemmConfig(128, 128, 64, 4, 4), sms=132)
    assert (p["output_tiles"], p["matmul_programs"]) == (224, 132)
    assert p["persistent_tiles_per_program_max"] == 2
    assert p["persistent_tile_balance"] == pytest.approx(224 / 264)
    small = predict(_QWEN_DOWN[96], HopperGemmConfig(128, 128, 64, 4, 4), sms=132)
    assert small["matmul_programs"] == 28 and small["persistent_tile_balance"] == 1.0


def test_split_k_counts_two_launches_workspace_and_padded_k() -> None:
    # K=18944 is 296 K tiles; 16 splits take ceil(296/16)=19 tiles, padding K to 19456.
    p = predict(_QWEN_DOWN[1], SplitKConfig(16, 64, 64, 4, 3, 16), sms=132)
    assert p["kernel_launches"] == 2
    assert (p["output_tiles"], p["matmul_programs"]) == (56, 896)
    assert p["k_tiles_per_split"] == 19
    assert p["reduction_programs"] == 14  # ceil(1 * 3584 / 256)
    assert p["workspace_bytes"] == 4 * 16 * 1 * 3584
    assert p["workspace_write_read_bytes"] == 8 * 16 * 3584
    assert p["masked_row_fraction"] == pytest.approx(15 / 16)
    assert p["padded_dot_flops"] == 2 * 16 * 3584 * 19456
    ideal = 2 * (18944 + 18944 * 3584 + 3584)
    assert p["ideal_plus_workspace_bytes"] == ideal + 8 * 16 * 3584
    assert p["workspace_adjusted_intensity_flops_per_byte"] == pytest.approx(
        2 * 3584 * 18944 / (ideal + 8 * 16 * 3584)
    )


def test_split_k_outside_the_frozen_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="frozen workload/config manifest"):
        predict(_QWEN_DOWN[96], SplitKConfig(16, 64, 64, 4, 3, 16), sms=132)
    with pytest.raises(ValueError, match="frozen workload/config manifest"):
        predict(_QWEN_DOWN[1], SplitKConfig(16, 64, 64, 4, 3, 8), sms=132)
    with pytest.raises(ValueError):
        predict(Workload(1, 8, 8, "m", "p", "r"), KernelConfig(16, 16, 16, 1, 1), sms=0)


def test_dossier_covers_every_m_and_links_only_bank1_finalists_to_measurements() -> None:
    dossier = build_dossier(_SUMMARY)
    assert dossier["evidence_label"] == "INFERENCE"
    assert dossier["sm_count"] == _SUMMARY["hardware"]["multiprocessor_count"]
    predictions = dossier["predictions"]
    expected = sum(1 + len(candidates_for(w)) for w in _QWEN_DOWN.values())
    assert len(predictions) == expected
    workloads = {w["workload_key"]: w for w in _SUMMARY["workloads"]}
    for m, workload in _QWEN_DOWN.items():
        rows = [p for p in predictions if p["workload_key"] == workload.key]
        assert sum(p["arm"] == "old" for p in rows) == 1, m
        selected = [p for p in rows if p["bank1_selected"]]
        assert sorted(p["arm"] for p in selected) == ["new", "old"]
        record = workloads[workload.key]
        for p in rows:
            measured = p["bank2_measurement"]
            if not p["bank1_selected"]:
                assert measured is None
                continue
            arm = p["arm"]
            assert p["config_key"] == record["selected"][arm]["config_key"]
            assert measured["median_us"] == record["bank2"][arm]["median_ms"] * 1000
            assert measured["torch_us"] == record["torch_ms"] * 1000
            assert measured["ratio_to_torch"] == record[f"{arm}_to_torch"]


def test_dossier_attributes_tutorial_and_repository_code() -> None:
    provenance = {p["family"]: p for p in build_dossier(_SUMMARY)["provenance"]}
    assert "Tutorial-derived" in provenance["persistent"]["origin"]
    assert "v3.4.0" in provenance["persistent"]["origin"]
    assert "not a new algorithm" in provenance["split_k"]["origin"]
    for entry in provenance.values():
        assert (_REPO / entry["source"]).is_file()
