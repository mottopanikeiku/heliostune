from __future__ import annotations

import copy
import io
import json

import pytest

from heliostune.configs import KernelConfig, ModelSpec, Workload
from heliostune.errors import SchemaError
from heliostune.schema import HardwareProfile, Measurement, read_jsonl, write_jsonl
from heliostune.synthetic import synthetic_measurements


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        gpu="L4",
        device_name="NVIDIA L4",
        compute_capability=(8, 9),
        multiprocessor_count=58,
        total_memory_gb=22.0,
        cuda_version="12.8",
        torch_version="2.8.0",
        triton_version="3.4.0",
    )


def _workload() -> Workload:
    return Workload(7, 64, 32, "model", "projection", "decode-7")


def _config() -> KernelConfig:
    return KernelConfig(16, 32, 32, 4, 3, 8)


def _measurement() -> Measurement:
    return Measurement(
        hardware=_hardware(),
        workload=_workload(),
        config=_config(),
        bank=2,
        latency_ms=1.5,
        torch_latency_ms=2.5,
        correct=True,
        max_abs_error=0.01,
        latency_p20_ms=1.25,
        latency_p80_ms=1.75,
        torch_latency_p20_ms=2.25,
        torch_latency_p80_ms=2.75,
        compile_ms=4.0,
        benchmark_wall_ms=101.0,
        torch_benchmark_wall_ms=102.0,
    )


def _v1_payload(*, correct: object = True) -> dict[str, object]:
    row = _measurement().to_dict()
    row["schema_version"] = 1
    row["replicate"] = row.pop("bank")
    for field in (
        "torch_latency_p20_ms",
        "torch_latency_p80_ms",
        "benchmark_wall_ms",
        "torch_benchmark_wall_ms",
        "failure_stage",
    ):
        row.pop(field)
    row["correct"] = correct
    return row


def test_schema_v2_round_trip_uses_bank() -> None:
    row = _measurement()
    destination = io.StringIO()

    write_jsonl([row], destination)
    payload = destination.getvalue()

    assert payload.endswith("\n")
    assert '"schema_version":2' in payload
    assert '"bank":2' in payload
    assert '"replicate"' not in payload
    assert read_jsonl(io.StringIO(payload)) == [row]


@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_v1_boolean_is_not_coerced(bad: object) -> None:
    with pytest.raises(SchemaError, match="must be a boolean"):
        Measurement.from_dict(_v1_payload(correct=bad))


def test_v1_success_loads_and_serializes_as_v2() -> None:
    row = Measurement.from_dict(_v1_payload())

    assert row.bank == 2
    assert row.failure_stage is None
    assert row.to_dict()["schema_version"] == 2


def test_v1_failure_requires_explicit_classification_before_write() -> None:
    payload = _v1_payload(correct=False)
    payload["latency_ms"] = None
    payload["latency_p20_ms"] = None
    payload["latency_p80_ms"] = None
    payload["error"] = "legacy compile failure"

    row = Measurement.from_dict(payload)

    assert row.failure_stage == "legacy_unknown"
    assert not row.correctness_classified
    with pytest.raises(SchemaError, match="explicitly classified"):
        row.to_dict()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(extra=1), "unknown fields"),
        (lambda row: row.pop("latency_ms"), "missing fields"),
        (lambda row: row.update(bank=True), "must be an integer"),
        (lambda row: row.update(latency_ms="1.5"), "must be a number"),
        (lambda row: row.update(latency_ms=float("inf")), "must be finite"),
        (lambda row: row.update(error="contradiction"), "must not contain an error"),
        (lambda row: row.update(latency_p20_ms=2.0), "p20 <= median <= p80"),
    ],
)
def test_v2_rejects_malformed_rows(mutation: object, message: str) -> None:
    payload = _measurement().to_dict()
    mutation(payload)
    with pytest.raises(SchemaError, match=message):
        Measurement.from_dict(payload)


def test_nested_schemas_reject_coercion_whitespace_and_bad_capability() -> None:
    payload = _measurement().to_dict()
    bad_workload = copy.deepcopy(payload)
    bad_workload["workload"]["m"] = "7"
    with pytest.raises(SchemaError, match="workload m must be an integer"):
        Measurement.from_dict(bad_workload)

    bad_identifier = copy.deepcopy(payload)
    bad_identifier["workload"]["model"] = " model"
    with pytest.raises(SchemaError, match="surrounding whitespace"):
        Measurement.from_dict(bad_identifier)

    bad_capability = copy.deepcopy(payload)
    bad_capability["hardware"]["compute_capability"] = [8, True]
    with pytest.raises(SchemaError, match=r"compute_capability\[1\].*integer"):
        Measurement.from_dict(bad_capability)


def test_jsonl_rejects_duplicate_keys_constants_and_blank_lines_with_context() -> None:
    payload = json.dumps(_v1_payload())
    duplicate = payload[:-1] + ',"correct":true}'
    with pytest.raises(SchemaError, match=r"rows.jsonl:1.*duplicate.*correct"):
        read_jsonl(io.StringIO(duplicate + "\n"), source_name="rows.jsonl")

    nonfinite = payload.replace('"latency_ms": 1.5', '"latency_ms": NaN')
    with pytest.raises(SchemaError, match=r"rows.jsonl:1.*non-finite"):
        read_jsonl(io.StringIO(nonfinite + "\n"), source_name="rows.jsonl")

    with pytest.raises(SchemaError, match=r"rows.jsonl:1.*blank"):
        read_jsonl(io.StringIO("\n"), source_name="rows.jsonl")


def test_quantiles_and_failure_stage_are_structurally_consistent() -> None:
    with pytest.raises(SchemaError, match="present together"):
        Measurement(_hardware(), _workload(), _config(), 1.0, 2.0, True, latency_p20_ms=0.9)
    with pytest.raises(SchemaError, match="classified failure stage"):
        Measurement(
            _hardware(),
            _workload(),
            _config(),
            None,
            2.0,
            False,
            error="failed",
        )
    with pytest.raises(SchemaError, match="failed measurement must not contain"):
        Measurement(
            _hardware(),
            _workload(),
            _config(),
            1.0,
            2.0,
            False,
            error="failed",
            failure_stage="benchmark",
        )


def test_config_and_model_spec_invariants() -> None:
    with pytest.raises(SchemaError, match="power of two"):
        KernelConfig(24, 32, 32, 4, 3)
    with pytest.raises(SchemaError, match="divisible"):
        ModelSpec("model", 63, 128, 8, 4, "https://example.test/config.json")
    with pytest.raises(SchemaError, match="must not exceed"):
        ModelSpec("model", 64, 128, 8, 9, "https://example.test/config.json")


def test_synthetic_banks_are_order_independent_and_schema_v2_deterministic() -> None:
    forward = synthetic_measurements(seed=11, banks=(0, 2))
    reversed_rows = synthetic_measurements(seed=11, banks=(2, 0))

    def indexed(rows: list[Measurement]) -> dict[tuple[str, str, str, int], dict[str, object]]:
        return {
            (row.hardware.gpu, row.workload.key, row.config.key, row.bank): row.to_dict()
            for row in rows
        }

    assert indexed(forward) == indexed(reversed_rows)
    assert all(row.to_dict()["schema_version"] == 2 for row in forward)


@pytest.mark.parametrize("banks", [(), (0, 0), (True,), (-1,), ("0",)])
def test_synthetic_banks_must_be_unique_nonnegative_integers(banks: object) -> None:
    with pytest.raises(SchemaError):
        synthetic_measurements(banks=banks)
