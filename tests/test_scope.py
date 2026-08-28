from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import heliostune.scope as scope_module
from heliostune.errors import ArtifactError, SchemaError
from heliostune.scope import (
    DOMAIN_VOCABULARY,
    DTYPE_VOCABULARY,
    EXECUTABLE_TEMPLATE_IDS,
    Capability,
    DTypeSpec,
    GatedMLPSemantics,
    NumericContract,
    QuantizationSpec,
    RMSNormSemantics,
    ShapeConstraint,
    Suite,
    TensorSpec,
    load_plugin,
    load_suite,
    verify_plugin,
    verify_suite,
)

ROOT = Path(__file__).parents[1]
PLUGIN = ROOT / "benchmarks/plugins/fusion-reference-plugin-v1.json"
MLP = ROOT / "benchmarks/suites/gated-mlp-epilogue-v1.json"
RMS = ROOT / "benchmarks/suites/residual-rmsnorm-v1.json"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert type(value) is dict
    return value


def _reject(model: Any, value: object) -> None:
    with pytest.raises(SchemaError):
        model.from_dict(value)


def _dtype(name: str, usage: str, packing: object = None) -> dict[str, object]:
    return {"name": name, "usage": usage, "packing": packing}


def _advanced_contract() -> dict[str, object]:
    return {
        "id": "fp8-contract",
        "input": _dtype("fp8_e4m3fn", "input"),
        "storage": _dtype("fp8_e4m3fn", "storage"),
        "accumulation": _dtype("fp32", "compute"),
        "output": _dtype("bf16", "output"),
        "tf32": False,
        "quantization": {
            "scheme": "per_tensor",
            "scale_dtype": "fp32",
            "scale_layout": "scalar",
            "calibration": "static",
            "group_size": None,
        },
    }


def test_legacy_byte_preservation_snapshots() -> None:
    expected = {
        "benchmarks/methodology-protocol-v1-template.json": "69f082a8dd481935e66ec1830a0554d4fc0d06799c30297b50e8dfeff918e47e",
        "benchmarks/parhelion-v2-development-protocol.json": "ae544f798284528ed888a4c46d79b7419d5790cc8c967ad2897e9030f22374c8",
        "benchmarks/parhelion-v3-development-protocol.json": "755ea87959edbeb1d50f1d9a5dea46ed6cd5e1aa5f8f964416767546109139cb",
    }
    assert {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in expected
    } == expected


def test_exact_key_and_type_roundtrip() -> None:
    for path, loader in ((MLP, load_suite), (RMS, load_suite), (PLUGIN, load_plugin)):
        original = _json(path)
        assert loader(path).to_dict() == original
        unknown = copy.deepcopy(original)
        unknown["unknown"] = None
        _reject(Suite if path != PLUGIN else type(load_plugin(PLUGIN)), unknown)
    wrong_bool = _json(MLP)
    wrong_bool["revision"] = True
    _reject(Suite, wrong_bool)
    wrong_int = _json(MLP)
    tensors = wrong_int["tensors"]
    assert type(tensors) is list and type(tensors[0]) is dict
    tensors[0]["contiguous"] = 1
    _reject(Suite, wrong_int)


def test_dtype_cross_rules() -> None:
    for name in DTYPE_VOCABULARY:
        usage = "compute" if name == "tf32" else "storage"
        packing = (
            {"bits": 4, "axis": "k", "order": "low_nibble_first"}
            if name in {"int4", "uint4"}
            else None
        )
        assert DTypeSpec.from_dict(_dtype(name, usage, packing)).name == name
    _reject(DTypeSpec, _dtype("tf32", "storage"))
    _reject(
        DTypeSpec, _dtype("int4", "compute", {"bits": 4, "axis": "k", "order": "low_nibble_first"})
    )
    _reject(DTypeSpec, _dtype("int4", "storage"))
    _reject(
        DTypeSpec, _dtype("fp16", "storage", {"bits": 4, "axis": "k", "order": "low_nibble_first"})
    )
    fp8_without_scale = _advanced_contract()
    fp8_without_scale["quantization"] = None
    _reject(NumericContract, fp8_without_scale)
    tf32 = _advanced_contract()
    tf32.update(
        {
            "id": "tf32",
            "input": _dtype("fp16", "input"),
            "storage": _dtype("fp16", "storage"),
            "accumulation": _dtype("tf32", "compute"),
            "quantization": None,
            "tf32": True,
        }
    )
    assert NumericContract.from_dict(tf32).tf32 is True


def test_quantization_cross_rules() -> None:
    good = {
        "scheme": "per_group",
        "scale_dtype": "fp16",
        "scale_layout": "group",
        "calibration": "dynamic",
        "group_size": 128,
    }
    assert QuantizationSpec.from_dict(good).group_size == 128
    for bad in (
        {**good, "group_size": None},
        {**good, "scale_layout": "channel"},
        {**good, "group_size": True},
        {**good, "calibration": "observed"},
    ):
        _reject(QuantizationSpec, bad)
    packed = {
        "id": "q",
        "role": "parameter",
        "shape": ["n", "k"],
        "storage_dtype": "int4",
        "logical_dtype": "fp16",
        "layout": "packed",
        "contiguous": True,
        "alignment": 16,
        "quantization": good,
        "packing": {"bits": 4, "axis": "k", "order": "low_nibble_first"},
    }
    assert TensorSpec.from_dict(packed).layout == "packed"
    _reject(TensorSpec, {**packed, "layout": "row_major"})
    _reject(TensorSpec, {**packed, "quantization": None})
    _reject(TensorSpec, {**packed, "packing": None})
    _reject(
        TensorSpec,
        {**packed, "packing": {"bits": 4, "axis": "missing", "order": "low_nibble_first"}},
    )
    _reject(TensorSpec, {**packed, "logical_dtype": "int4"})
    _reject(
        TensorSpec,
        {
            **packed,
            "storage_dtype": "tf32",
            "layout": "row_major",
            "quantization": None,
            "packing": None,
        },
    )
    _reject(
        TensorSpec,
        {
            **packed,
            "storage_dtype": "fp8_e5m2",
            "layout": "row_major",
            "quantization": None,
            "packing": None,
        },
    )
    _reject(
        TensorSpec,
        {
            **packed,
            "storage_dtype": "fp16",
            "layout": "row_major",
            "packing": None,
        },
    )


def test_capability_evidence_states() -> None:
    zero = "0" * 64
    assert Capability.from_dict({"state": "unprobed", "evidence_sha256": None}).state == "unprobed"
    assert (
        Capability.from_dict({"state": "available", "evidence_sha256": zero}).evidence_sha256
        == zero
    )
    assert (
        Capability.from_dict({"state": "unavailable", "evidence_sha256": zero}).state
        == "unavailable"
    )
    for bad in (
        {"state": "available", "evidence_sha256": None},
        {"state": "unprobed", "evidence_sha256": zero},
        {"state": "available", "evidence_sha256": "A" * 64},
        {"state": "passing", "evidence_sha256": zero},
    ):
        _reject(Capability, bad)


def test_gated_mlp_semantics() -> None:
    semantics = _json(MLP)["cases"]
    assert type(semantics) is list and type(semantics[0]) is dict
    raw = semantics[0]["semantics"]
    parsed = GatedMLPSemantics.from_dict(raw)
    assert parsed.activation == "silu"
    assert parsed.residual is False
    assert type(raw) is dict
    _reject(GatedMLPSemantics, {**raw, "output_arity": 2})
    _reject(GatedMLPSemantics, {**raw, "bias": 1})
    _reject(GatedMLPSemantics, {**raw, "activation": "relu"})
    _reject(
        GatedMLPSemantics,
        {
            **raw,
            "fusion_boundary": [
                "gate_projection",
                "up_projection",
                "silu",
                "residual_add",
                "gating_multiply",
            ],
        },
    )
    suite = _json(MLP)
    suite_cases = suite["cases"]
    assert type(suite_cases) is list and type(suite_cases[0]) is dict
    suite_semantics = suite_cases[0]["semantics"]
    assert type(suite_semantics) is dict
    suite_semantics["residual"] = True
    boundary = suite_semantics["fusion_boundary"]
    assert type(boundary) is list
    boundary.append("residual_add")
    _reject(Suite, suite)
    packed = _json(MLP)
    packed_cases = packed["cases"]
    assert type(packed_cases) is list and type(packed_cases[0]) is dict
    packed_semantics = packed_cases[0]["semantics"]
    assert type(packed_semantics) is dict
    packed_semantics["gate_up_layout"] = "packed"
    _reject(Suite, packed)


def test_rmsnorm_semantics() -> None:
    cases = _json(RMS)["cases"]
    assert type(cases) is list and type(cases[0]) is dict and type(cases[0]["semantics"]) is dict
    raw = cases[0]["semantics"]
    assert RMSNormSemantics.from_dict(raw).output_arity == 1
    _reject(RMSNormSemantics, {**raw, "epsilon": 0})
    _reject(RMSNormSemantics, {**raw, "epsilon": float("inf")})
    _reject(RMSNormSemantics, {**raw, "output_arity": 3})
    _reject(RMSNormSemantics, {**raw, "residual_position": "around"})
    _reject(RMSNormSemantics, {**raw, "fusion_boundary": list(reversed(raw["fusion_boundary"]))})
    suite = _json(RMS)
    tensors = suite["tensors"]
    suite_cases = suite["cases"]
    assert (
        type(tensors) is list
        and type(suite_cases) is list
        and type(suite_cases[0]) is dict
        and type(suite_cases[0]["semantics"]) is dict
    )
    suite_cases[0]["semantics"]["output_arity"] = 2
    _reject(Suite, suite)
    missing_residual = _json(RMS)
    residual_tensors = missing_residual["tensors"]
    assert type(residual_tensors) is list
    missing_residual["tensors"] = [
        tensor
        for tensor in residual_tensors
        if type(tensor) is not dict or tensor.get("id") != "residual"
    ]
    _reject(Suite, missing_residual)
    no_gamma = _json(RMS)
    no_gamma_cases = no_gamma["cases"]
    assert (
        type(no_gamma_cases) is list
        and type(no_gamma_cases[0]) is dict
        and type(no_gamma_cases[0]["semantics"]) is dict
    )
    no_gamma_cases[0]["semantics"]["gamma"] = False
    no_gamma_cases[0]["semantics"]["fusion_boundary"] = [
        "residual_add",
        "rms_normalize",
    ]
    _reject(Suite, no_gamma)


def test_inline_shape_applicability() -> None:
    shape = {"m": 16, "n": 33}
    assert ShapeConstraint.from_dict({"dimension": "m", "op": "divisible_by", "value": 8}).applies(
        shape
    )
    assert ShapeConstraint.from_dict({"dimension": "n", "op": "min", "value": 32}).applies(shape)
    assert not ShapeConstraint.from_dict({"dimension": "n", "op": "max", "value": 32}).applies(
        shape
    )
    assert not ShapeConstraint.from_dict(
        {"dimension": "missing", "op": "equal", "value": 1}
    ).applies(shape)
    _reject(ShapeConstraint, {"dimension": "m", "op": "divisible_by", "value": True})
    suite = _json(MLP)
    cases = suite["cases"]
    assert type(cases) is list and type(cases[0]) is dict and type(cases[0]["shape"]) is dict
    cases[0]["shape"]["hidden"] = 4095
    _reject(Suite, suite)
    wrong_contract = _json(MLP)
    contracts = wrong_contract["numeric_contracts"]
    arms = wrong_contract["arms"]
    assert (
        type(contracts) is list
        and type(contracts[0]) is dict
        and type(arms) is list
        and type(arms[1]) is dict
    )
    other_contract = copy.deepcopy(contracts[0])
    other_contract["id"] = "other-contract"
    contracts.append(other_contract)
    arms[1]["numeric_contract_ids"] = ["other-contract"]
    _reject(Suite, wrong_contract)
    indivisible_reference = _json(MLP)
    arms = indivisible_reference["arms"]
    assert type(arms) is list and type(arms[1]) is dict
    arms[1]["constraints"] = [{"dimension": "hidden", "op": "divisible_by", "value": 4097}]
    _reject(Suite, indivisible_reference)
    unlisted_reference = _json(MLP)
    cells = unlisted_reference["expected_cells"]
    assert type(cells) is list
    unlisted_reference["expected_cells"] = [
        cell for cell in cells if type(cell) is not dict or cell.get("arm_id") != "mlp-reference"
    ]
    _reject(Suite, unlisted_reference)


def test_correctness_before_timing_static_plan() -> None:
    suite = _json(MLP)
    cells = suite["expected_cells"]
    assert type(cells) is list
    cells[0], cells[1] = cells[1], cells[0]
    _reject(Suite, suite)
    suite = _json(MLP)
    cells = suite["expected_cells"]
    assert type(cells) is list and type(cells[1]) is dict
    cells[1]["input_seed"] = 18
    _reject(Suite, suite)


def test_executor_observation_limitation_exposed() -> None:
    verified = verify_suite(MLP)
    assert (
        verified.suite.executor_rule == "timing_requires_retained_passing_correctness_observation"
    )
    assert all(cell.stage in {"correctness", "timing"} for cell in verified.suite.expected_cells)
    assert not any(
        "outcome" in cell.to_dict() or "passing" in cell.to_dict()
        for cell in verified.suite.expected_cells
    )
    changed = _json(MLP)
    changed["executor_rule"] = "static_correctness_implies_passing"
    _reject(Suite, changed)


def test_plugin_suite_digest_and_path_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = tmp_path / "plugins"
    suites = tmp_path / "suites"
    plugins.mkdir()
    suites.mkdir()
    (suites / MLP.name).write_bytes(MLP.read_bytes())
    (suites / RMS.name).write_bytes(RMS.read_bytes())
    target = plugins / PLUGIN.name
    target.write_bytes(PLUGIN.read_bytes())
    original_read_verified = scope_module._read_verified
    atomic_suite = tmp_path / "atomic-suite.json"
    mlp_payload = MLP.read_bytes()
    atomic_suite.write_bytes(mlp_payload)

    def swap_suite_after_read(path: str | Path, context: str) -> tuple[Path, bytes]:
        resolved, payload = original_read_verified(path, context)
        if resolved == atomic_suite.resolve():
            atomic_suite.write_bytes(RMS.read_bytes())
        return resolved, payload

    monkeypatch.setattr(scope_module, "_read_verified", swap_suite_after_read)
    atomic_verified = verify_suite(atomic_suite)
    assert atomic_verified.bytes == mlp_payload
    assert atomic_verified.sha256 == hashlib.sha256(mlp_payload).hexdigest()
    assert atomic_verified.suite.suite_id == "gated-mlp-epilogue-reference"
    monkeypatch.setattr(scope_module, "_read_verified", original_read_verified)
    plugin_payload = target.read_bytes()

    def swap_plugin_after_read(path: str | Path, context: str) -> tuple[Path, bytes]:
        resolved, payload = original_read_verified(path, context)
        if resolved == target.resolve():
            target.write_bytes(b"{}")
        return resolved, payload

    monkeypatch.setattr(scope_module, "_read_verified", swap_plugin_after_read)
    atomic_plugin = verify_plugin(target)
    assert atomic_plugin.bytes == plugin_payload
    assert atomic_plugin.sha256 == hashlib.sha256(plugin_payload).hexdigest()
    assert atomic_plugin.plugin.plugin_id == "fusion-reference-plugin"
    monkeypatch.setattr(scope_module, "_read_verified", original_read_verified)
    target.write_bytes(plugin_payload)
    verified = verify_plugin(target)
    assert [item.sha256 for item in verified.suites] == [
        ref.sha256 for ref in verified.plugin.suite_refs
    ]
    bad = _json(target)
    refs = bad["suite_refs"]
    assert type(refs) is list and type(refs[0]) is dict
    refs[0]["sha256"] = "0" * 64
    target.write_text(json.dumps(bad))
    with pytest.raises(ArtifactError, match="digest mismatch"):
        verify_plugin(target)
    refs[0]["sha256"] = "A" * 64
    target.write_text(json.dumps(bad))
    with pytest.raises(SchemaError, match="lowercase"):
        verify_plugin(target)
    refs[0]["sha256"] = hashlib.sha256(MLP.read_bytes()).hexdigest()
    refs[0]["path"] = "../../outside.json"
    target.write_text(json.dumps(bad))
    with pytest.raises(ArtifactError, match="escapes"):
        verify_plugin(target)
    refs[0]["path"] = "/absolute.json"
    target.write_text(json.dumps(bad))
    with pytest.raises(SchemaError, match="relative"):
        verify_plugin(target)
    refs[0]["path"] = f"../suites/{MLP.name}"
    suite_data = _json(MLP)
    suite_data["plugin_id"] = "different-plugin"
    suite_payload = json.dumps(suite_data).encode()
    (suites / MLP.name).write_bytes(suite_payload)
    refs[0]["sha256"] = hashlib.sha256(suite_payload).hexdigest()
    target.write_text(json.dumps(bad))
    with pytest.raises(ArtifactError, match="plugin identity mismatch"):
        verify_plugin(target)
    refs[0]["suite_id"] = "different-suite"
    target.write_text(json.dumps(bad))
    with pytest.raises(ArtifactError, match="suite identity mismatch"):
        verify_plugin(target)


def _suite_with_fp32_output() -> dict[str, object]:
    suite = _json(MLP)
    contracts = suite["numeric_contracts"]
    tensors = suite["tensors"]
    assert (
        type(contracts) is list
        and type(contracts[0]) is dict
        and type(tensors) is list
        and all(type(tensor) is dict for tensor in tensors)
    )
    contracts[0]["output"] = _dtype("fp32", "output")
    output = next(tensor for tensor in tensors if tensor["role"] == "output")
    output["storage_dtype"] = "fp32"
    output["logical_dtype"] = "fp32"
    return suite


def test_vocabulary_vs_execution_separation() -> None:
    assert set(DTYPE_VOCABULARY) == {
        "fp32",
        "tf32",
        "fp16",
        "bf16",
        "fp8_e4m3fn",
        "fp8_e5m2",
        "int8",
        "int4",
        "uint4",
    }
    assert set(DOMAIN_VOCABULARY) == {
        "dense_gemm",
        "fused_mlp",
        "rmsnorm_residual",
        "attention",
        "kv_cache",
        "moe",
        "quantized_linear",
    }
    assert EXECUTABLE_TEMPLATE_IDS == ("gated_mlp_epilogue.v1", "residual_rmsnorm.v1")
    advanced = NumericContract.from_dict(_advanced_contract())
    assert not advanced.is_initially_executable
    suite = _json(MLP)
    suite["numeric_contracts"] = [_advanced_contract()]
    arms = suite["arms"]
    cases = suite["cases"]
    assert (
        type(arms) is list
        and all(type(arm) is dict for arm in arms)
        and type(cases) is list
        and type(cases[0]) is dict
    )
    for arm in arms:
        arm["numeric_contract_ids"] = ["fp8-contract"]
    cases[0]["numeric_contract_id"] = "fp8-contract"
    _reject(Suite, suite)
    quantization = _advanced_contract()["quantization"]
    fp8_tensor_suite = _json(MLP)
    fp8_tensors = fp8_tensor_suite["tensors"]
    assert type(fp8_tensors) is list and type(fp8_tensors[0]) is dict
    fp8_tensors[0].update(
        {
            "storage_dtype": "fp8_e4m3fn",
            "logical_dtype": "fp8_e4m3fn",
            "quantization": copy.deepcopy(quantization),
        }
    )
    assert TensorSpec.from_dict(fp8_tensors[0]).storage_dtype == "fp8_e4m3fn"
    _reject(Suite, fp8_tensor_suite)
    int4_tensor_suite = _json(MLP)
    int4_tensors = int4_tensor_suite["tensors"]
    assert type(int4_tensors) is list and type(int4_tensors[0]) is dict
    int4_tensors[0].update(
        {
            "storage_dtype": "int4",
            "logical_dtype": "bf16",
            "layout": "packed",
            "quantization": copy.deepcopy(quantization),
            "packing": {
                "bits": 4,
                "axis": "hidden",
                "order": "low_nibble_first",
            },
        }
    )
    assert TensorSpec.from_dict(int4_tensors[0]).storage_dtype == "int4"
    _reject(Suite, int4_tensor_suite)

    fp32_output_suite = _suite_with_fp32_output()
    parsed = Suite.from_dict(fp32_output_suite)
    assert {
        (tensor.storage_dtype, tensor.logical_dtype)
        for tensor in parsed.tensors
        if tensor.role == "output"
    } == {("fp32", "fp32")}

    fp32_contracts = fp32_output_suite["numeric_contracts"]
    assert type(fp32_contracts) is list and type(fp32_contracts[0]) is dict
    fp32_contracts[0]["output"] = _dtype("bf16", "output")
    with pytest.raises(SchemaError, match="every applicable case numeric contract"):
        Suite.from_dict(fp32_output_suite)

    fp32_input_suite = _suite_with_fp32_output()
    fp32_input_tensors = fp32_input_suite["tensors"]
    assert type(fp32_input_tensors) is list and all(
        type(tensor) is dict for tensor in fp32_input_tensors
    )
    input_tensor = next(tensor for tensor in fp32_input_tensors if tensor["role"] == "input")
    input_tensor["storage_dtype"] = "fp32"
    input_tensor["logical_dtype"] = "fp32"
    with pytest.raises(SchemaError, match="input, parameter, and intermediate"):
        Suite.from_dict(fp32_input_suite)

    mixed_contract_suite = _suite_with_fp32_output()
    mixed_contracts = mixed_contract_suite["numeric_contracts"]
    mixed_cases = mixed_contract_suite["cases"]
    assert (
        type(mixed_contracts) is list
        and type(mixed_contracts[0]) is dict
        and type(mixed_cases) is list
        and type(mixed_cases[0]) is dict
    )
    bf16_contract = copy.deepcopy(mixed_contracts[0])
    bf16_contract["id"] = "bf16-output"
    bf16_contract["output"] = _dtype("bf16", "output")
    mixed_contracts.append(bf16_contract)
    bf16_case = copy.deepcopy(mixed_cases[0])
    bf16_case["id"] = "mlp-case-bf16-output"
    bf16_case["numeric_contract_id"] = "bf16-output"
    mixed_cases.append(bf16_case)
    with pytest.raises(SchemaError, match="every applicable case numeric contract"):
        Suite.from_dict(mixed_contract_suite)

    advanced_output_suite = _suite_with_fp32_output()
    advanced_output_tensors = advanced_output_suite["tensors"]
    assert type(advanced_output_tensors) is list and all(
        type(tensor) is dict for tensor in advanced_output_tensors
    )
    output = next(tensor for tensor in advanced_output_tensors if tensor["role"] == "output")
    output.update(
        {
            "storage_dtype": "fp8_e4m3fn",
            "logical_dtype": "fp8_e4m3fn",
            "quantization": copy.deepcopy(_advanced_contract()["quantization"]),
        }
    )
    assert TensorSpec.from_dict(output).storage_dtype == "fp8_e4m3fn"
    with pytest.raises(SchemaError, match="only fp16/bf16"):
        Suite.from_dict(advanced_output_suite)
