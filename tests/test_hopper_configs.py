from collections.abc import Callable
from dataclasses import replace

import pytest

from heliostune.configs import (
    DEFAULT_CONFIGS,
    DEFAULT_WORKLOADS,
    HOPPER_GEMM_CONFIGS,
    MODEL_SPECS,
    SKINNY_GEMV_CONFIGS,
    HopperGemmConfig,
    KernelConfig,
    ModelSpec,
    SkinnyGemvConfig,
    Workload,
)
from heliostune.errors import SchemaError


def test_committed_manifests_keep_their_frozen_sizes_and_record_types() -> None:
    assert len(DEFAULT_CONFIGS) == 36
    assert len(DEFAULT_WORKLOADS) == 96
    assert len(HOPPER_GEMM_CONFIGS) == 23
    assert len(SKINNY_GEMV_CONFIGS) == 48

    assert all(type(config) is KernelConfig for config in DEFAULT_CONFIGS)
    assert all(type(workload) is Workload for workload in DEFAULT_WORKLOADS)
    assert not any(
        isinstance(config, HopperGemmConfig | SkinnyGemvConfig) for config in DEFAULT_CONFIGS
    )
    assert not any(
        isinstance(workload, HopperGemmConfig | SkinnyGemvConfig) for workload in DEFAULT_WORKLOADS
    )


def test_all_serializable_committed_records_round_trip() -> None:
    assert all(KernelConfig.from_dict(config.to_dict()) == config for config in DEFAULT_CONFIGS)
    assert all(Workload.from_dict(workload.to_dict()) == workload for workload in DEFAULT_WORKLOADS)
    assert all(
        HopperGemmConfig.from_dict(config.to_dict()) == config for config in HOPPER_GEMM_CONFIGS
    )
    assert all(
        SkinnyGemvConfig.from_dict(config.to_dict()) == config for config in SKINNY_GEMV_CONFIGS
    )


def test_committed_config_keys_are_unique_and_sorted() -> None:
    for configs in (HOPPER_GEMM_CONFIGS, SKINNY_GEMV_CONFIGS):
        keys = [config.key for config in configs]
        assert keys == sorted(keys)
        assert len(keys) == len(set(keys))


@pytest.mark.parametrize("field", ("block_m", "block_n", "block_k"))
def test_hopper_dot_dimensions_reject_tiles_below_sixteen(field: str) -> None:
    payload = HOPPER_GEMM_CONFIGS[0].to_dict()
    payload[field] = 8
    with pytest.raises(ValueError, match=rf"^{field} must be at least 16$"):
        replace(HOPPER_GEMM_CONFIGS[0], **{field: 8})

    with pytest.raises(ValueError, match=rf"^{field} must be at least 16$"):
        HopperGemmConfig.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    ("block_m", "block_n", "block_k", "num_warps", "num_stages", "group_m"),
)
@pytest.mark.parametrize("invalid", (True, 16.0, "16"))
def test_hopper_integer_fields_require_exact_ints(field: str, invalid: object) -> None:
    payload = HOPPER_GEMM_CONFIGS[0].to_dict()
    payload[field] = invalid  # type: ignore[assignment]

    with pytest.raises(SchemaError, match=rf"^hopper gemm config {field} must be an integer$"):
        replace(HOPPER_GEMM_CONFIGS[0], **{field: invalid})

    with pytest.raises(SchemaError, match=rf"^hopper gemm config {field} must be an integer$"):
        HopperGemmConfig.from_dict(payload)


@pytest.mark.parametrize("field", ("epilogue_subtile", "warp_specialize"))
@pytest.mark.parametrize("invalid", (0, 1, "false", None))
def test_hopper_boolean_fields_require_exact_bools(field: str, invalid: object) -> None:
    payload = HOPPER_GEMM_CONFIGS[0].to_dict()
    payload[field] = invalid  # type: ignore[assignment]

    with pytest.raises(SchemaError, match=rf"^hopper gemm config {field} must be a boolean$"):
        HopperGemmConfig.from_dict(payload)
    with pytest.raises(SchemaError, match=rf"^hopper gemm config {field} must be a boolean$"):
        replace(HOPPER_GEMM_CONFIGS[0], **{field: invalid})


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value.update(block_m=24), "block_m must be a power of two"),
        (lambda value: value.update(num_warps=3), "num_warps must be one of 1, 2, 4, or 8"),
        (lambda value: value.pop("group_m"), "hopper gemm config has missing fields"),
        (lambda value: value.update(extra=1), "hopper gemm config has unknown fields"),
    ),
)
def test_hopper_config_rejects_invalid_shape_and_schema(
    mutate: Callable[[dict[str, int | bool]], object], message: str
) -> None:
    payload = HOPPER_GEMM_CONFIGS[0].to_dict()
    mutate(payload)

    with pytest.raises(SchemaError, match=message):
        HopperGemmConfig.from_dict(payload)


@pytest.mark.parametrize(
    "field", ("block_m", "block_n", "block_k", "num_warps", "num_stages", "split_k")
)
@pytest.mark.parametrize("invalid", (True, 16.0, "16"))
def test_skinny_integer_fields_require_exact_ints(field: str, invalid: object) -> None:
    payload = SKINNY_GEMV_CONFIGS[0].to_dict()
    payload[field] = invalid  # type: ignore[assignment]

    with pytest.raises(SchemaError, match=rf"^skinny gemv config {field} must be an integer$"):
        SkinnyGemvConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {
                "block_m": 3,
                "block_n": 32,
                "block_k": 32,
                "num_warps": 4,
                "num_stages": 3,
                "split_k": 1,
            },
            "block_m must be a power of two",
        ),
        (
            {
                "block_m": 1,
                "block_n": 32,
                "block_k": 32,
                "num_warps": 3,
                "num_stages": 3,
                "split_k": 1,
            },
            "num_warps must be one of 1, 2, 4, or 8",
        ),
        (
            {
                "block_m": 8,
                "block_n": 256,
                "block_k": 64,
                "num_warps": 8,
                "num_stages": 3,
                "split_k": 1,
            },
            r"block_m \* block_n \* block_k must not exceed 8192, got 131072",
        ),
    ),
)
def test_skinny_config_rejects_invalid_shapes(payload: dict[str, int], message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        SkinnyGemvConfig.from_dict(payload)


def test_skinny_config_rejects_missing_and_unknown_fields() -> None:
    payload = SKINNY_GEMV_CONFIGS[0].to_dict()
    payload.pop("split_k")
    with pytest.raises(SchemaError, match="skinny gemv config has missing fields"):
        SkinnyGemvConfig.from_dict(payload)

    payload = SKINNY_GEMV_CONFIGS[0].to_dict()
    payload["extra"] = 1
    with pytest.raises(SchemaError, match="skinny gemv config has unknown fields"):
        SkinnyGemvConfig.from_dict(payload)


def test_kernel_config_contract_and_invalid_fields() -> None:
    config = KernelConfig(16, 32, 64, 4, 3)
    assert config.key == "m16n32k64-w4s3g8"
    assert KernelConfig.from_dict(config.to_dict()) == config

    bool_payload = config.to_dict()
    bool_payload["block_m"] = True
    with pytest.raises(SchemaError, match="kernel config block_m must be an integer"):
        KernelConfig.from_dict(bool_payload)

    with pytest.raises(SchemaError, match="block_m must be a power of two"):
        KernelConfig(24, 32, 64, 4, 3)
    with pytest.raises(SchemaError, match="num_warps must be one of 1, 2, 4, or 8"):
        KernelConfig(16, 32, 64, 3, 3)
    with pytest.raises(SchemaError, match="kernel config has unknown fields"):
        KernelConfig.from_dict({**config.to_dict(), "extra": 1})


def test_workload_contract_and_invalid_fields() -> None:
    workload = Workload(7, 64, 32, "model", "projection", "decode")
    assert workload.key == "model-projection-decode-m7-n64-k32"
    assert workload.flops == 28_672
    assert Workload.from_dict(workload.to_dict()) == workload

    bool_payload = workload.to_dict()
    bool_payload["m"] = True
    with pytest.raises(SchemaError, match="workload m must be an integer"):
        Workload.from_dict(bool_payload)

    blank_payload = workload.to_dict()
    blank_payload["model"] = " "
    with pytest.raises(SchemaError, match="workload model must be nonblank"):
        Workload.from_dict(blank_payload)

    with pytest.raises(SchemaError, match="workload has missing fields"):
        Workload.from_dict({"m": 7})


def test_model_spec_contract_and_invalid_fields() -> None:
    model = ModelSpec("model", 64, 128, 8, 2, "https://example.test/config.json")
    assert model.key_value_size == 16
    assert all(spec.key_value_size > 0 for spec in MODEL_SPECS)

    with pytest.raises(SchemaError, match="model spec name must be nonblank"):
        ModelSpec("", 64, 128, 8, 2, "https://example.test/config.json")
    with pytest.raises(SchemaError, match="model spec hidden_size must be an integer"):
        ModelSpec("model", True, 128, 8, 2, "https://example.test/config.json")
    with pytest.raises(SchemaError, match="hidden_size must be divisible by attention_heads"):
        ModelSpec("model", 63, 128, 8, 2, "https://example.test/config.json")
    with pytest.raises(SchemaError, match="key_value_heads must not exceed attention_heads"):
        ModelSpec("model", 64, 128, 8, 9, "https://example.test/config.json")
