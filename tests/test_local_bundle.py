from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

import heliostune.local_bundle as local_bundle
from heliostune.artifacts import strict_json_dumps, strict_json_loads, write_bytes_atomic
from heliostune.errors import ArtifactError, SchemaError
from heliostune.local_executor import (
    CapabilityProbe,
    CellObservation,
    CorrectnessObservation,
    LocalExecutionResult,
    TensorMaterialization,
    TimingObservation,
)
from heliostune.methodology import VerifiedBundle, verify_bundle_v1
from heliostune.scope import verify_plugin, verify_suite

_ROOT = Path(__file__).resolve().parents[1]
_PLUGIN = _ROOT / "benchmarks/plugins/fusion-reference-plugin-v1.json"

_DESCRIPTOR_PUBLICATION = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and all(
        function in os.supports_dir_fd
        for function in (os.open, os.mkdir, os.rename, os.stat, os.unlink, os.rmdir)
    )
)
_DESCRIPTOR_REASON = "requires O_NOFOLLOW, O_DIRECTORY, and dir_fd filesystem operations"
_MLP = _ROOT / "benchmarks/suites/gated-mlp-epilogue-v1.json"
_RMS = _ROOT / "benchmarks/suites/residual-rmsnorm-v1.json"


def _result(
    *,
    outcome: Literal["completed", "failed", "aborted"] = "completed",
    terminal: int = 4,
    unavailable: bool = False,
) -> LocalExecutionResult:
    verified = verify_suite(_MLP)
    cells = verified.suite.expected_cells
    attempts: list[dict[str, object]] = []
    observations: list[CellObservation] = []
    for index, cell in enumerate(cells[:terminal]):
        failed = outcome == "failed" and index == terminal - 1
        final_status = "failure" if failed else "success"
        final_state = "failed" if failed else "passed"
        attempts.extend(
            (
                {
                    "attempt_id": len(attempts) + 1,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": "running",
                    "from_state": "pending",
                    "to_state": "running",
                    "reason": None,
                },
                {
                    "attempt_id": len(attempts) + 2,
                    "cell_id": cell.id,
                    "stage": cell.stage,
                    "status": final_status,
                    "from_state": "running",
                    "to_state": final_state,
                    "reason": "injected failure" if failed else None,
                },
            )
        )
        key = local_bundle._correctness_key(verified.sha256, verified.suite, cell)
        if cell.stage == "correctness":
            correctness = CorrectnessObservation(
                "failed" if failed else "passed",
                key,
                "injected" if failed else None,
                "injected failure" if failed else None,
                {
                    "shape": [8, 11008],
                    "device": "cuda:0",
                    "dtype": "torch.bfloat16",
                    "layout": "torch.strided",
                    "contiguous": True,
                },
                True,
                True,
                True,
                not failed,
                1.0 if failed else 0.0,
            )
            timing = None
        else:
            correctness = None
            timing = TimingObservation(
                "failed" if failed else "passed",
                key,
                "injected" if failed else None,
                "injected failure" if failed else None,
                10,
                0 if failed else 50,
                () if failed else tuple(float(sample) for sample in range(1, 51)),
                None if failed else 25.5,
            )
        observations.append(
            CellObservation(
                cell.id,
                cell.case_id,
                cell.arm_id,
                cell.stage,
                "failed" if failed else "passed",
                correctness,
                timing,
            )
        )

    capability = CapabilityProbe(
        not unavailable,
        ("cuda_unavailable",) if unavailable else (),
        None if unavailable else "2.8.0",
        None if unavailable else "12.8",
        None,
        None if unavailable else 0,
        None if unavailable else "test CUDA device",
        None if unavailable else (9, 0),
        None if unavailable else True,
        None if unavailable else True,
        not unavailable,
        "CUDA is unavailable" if unavailable else None,
    )
    materialization: tuple[TensorMaterialization, ...] = ()
    if not unavailable:
        records: list[TensorMaterialization] = []
        case = verified.suite.cases[0]
        input_specs = tuple(tensor for tensor in verified.suite.tensors if tensor.role != "output")
        tensor_ids = tuple(tensor.id for tensor in input_specs)
        for cell in cells[:terminal]:
            if cell.stage != "correctness":
                continue
            descriptors = tuple(
                {
                    "tensor_id": tensor.id,
                    "role": tensor.role,
                    "shape": [case.shape_dict[name] for name in tensor.shape],
                    "draw": "normal_0_1_fp32_cpu",
                    "normal_scale": (
                        1.0 / math.sqrt(case.shape_dict["hidden"])
                        if tensor.role == "parameter"
                        else 1.0
                    ),
                    "normal_offset": 0.0,
                    "cpu_dtype": "float32",
                    "storage_dtype": "bfloat16",
                    "device": "cuda:0",
                    "contiguous": True,
                    "alignment_bytes": tensor.alignment,
                    "alignment_satisfied": True,
                    "storage_sha256": f"{index + 1:064x}",
                }
                for index, tensor in enumerate(input_specs)
            )
            records.append(
                TensorMaterialization(
                    verified.sha256,
                    cell.case_id,
                    cell.arm_id,
                    case.input_seed,
                    tensor_ids,
                    descriptors,
                )
            )
        materialization = tuple(records)
    candidate = next(arm for arm in verified.suite.arms if arm.role == "candidate")
    candidate_executed = any(
        cell.stage == "correctness" and cell.arm_id == candidate.id for cell in cells[:terminal]
    )
    compile_outcomes = (
        {
            candidate.id: {
                "case_id": verified.suite.cases[0].id,
                "arm_id": candidate.id,
                "entrypoint": candidate.entrypoint,
                "status": "compiled_and_first_call_completed",
                "error": None,
                "wrapper_create_ns": 1,
                "first_call_ns": 1,
                "eager_fallback": False,
                "backend_invoked": True,
                "callable_distinct": True,
                "autocast_policy": {
                    "device_type": "cuda",
                    "enabled": False,
                    "restore_ambient_state": True,
                },
            }
        }
        if candidate_executed and not unavailable
        else {}
    )
    return LocalExecutionResult(
        str(verified.path),
        verified.sha256,
        verified.bytes,
        verified.suite.suite_id,
        capability,
        materialization,
        tuple(observations),
        tuple(attempts),
        {"python": "test", "torch": None if unavailable else "2.8.0"},
        compile_outcomes,
        {"terminal": terminal, "fusion_claim": False},
        outcome,
    )


def _write(tmp_path: Path, result: LocalExecutionResult | None = None) -> VerifiedBundle:
    return local_bundle.write_local_bundle(
        _result() if result is None else result,
        plugin_path=_PLUGIN,
        output_dir=tmp_path / "bundle",
    )


def _root_dict(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"), source=path)
    assert type(value) is dict
    return value


def _plugin_fixture(tmp_path: Path, suite_indices: tuple[int, ...]) -> Path:
    plugin_raw = json.loads(_PLUGIN.read_text(encoding="utf-8"))
    refs = plugin_raw["suite_refs"]
    assert type(refs) is list
    plugin_raw["suite_refs"] = [refs[index] for index in suite_indices]
    suites = [verify_suite((_MLP, _RMS)[index]).suite for index in suite_indices]
    plugin_raw["domains"] = list(dict.fromkeys(suite.domain for suite in suites))
    plugin_raw["arm_ids"] = list(dict.fromkeys(arm.id for suite in suites for arm in suite.arms))
    plugin_dir = tmp_path / "declarations/plugins"
    suite_dir = tmp_path / "declarations/suites"
    plugin_dir.mkdir(parents=True)
    suite_dir.mkdir()
    plugin_path = plugin_dir / "plugin.json"
    plugin_path.write_text(strict_json_dumps(plugin_raw), encoding="utf-8")
    for suite_path in (_MLP, _RMS):
        (suite_dir / suite_path.name).write_bytes(suite_path.read_bytes())
    return plugin_path


def test_complete_bundle_preserves_transitive_sources_and_closes_all_cells(
    tmp_path: Path,
) -> None:
    verified = _write(tmp_path)
    bundle = verified.bundle

    assert bundle.lifecycle.state == "SEALED"
    assert bundle.lifecycle.outcome == "completed"
    assert bundle.coverage.expected_cells == 4
    assert bundle.coverage.terminal_cells == 4
    assert bundle.coverage.successes == 4
    assert bundle.coverage.failures == 0
    assert bundle.attempts.logical == bundle.attempts.physical == bundle.attempts.terminal == 4
    assert verified.limitations.plugin_suite_custody == "checked"
    assert verified.limitations.attempt_journal_hash_chain == "checked"
    assert verified.limitations.attempt_reconciliation == "checked"
    assert not verified.publication_eligible

    output = verified.root_path.parent
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "plugin.json").read_bytes() == _PLUGIN.read_bytes()
    assert (output / "plugin_suite_0.json").read_bytes() == _MLP.read_bytes()
    assert (output / "plugin_suite_1.json").read_bytes() == _RMS.read_bytes()
    assert not (output / "suite.json").exists()
    assert strict_json_loads(
        (output / "selected_suite.json").read_text(), source="selected suite"
    ) == {
        "schema": "heliostune.selected-suite/1",
        "plugin_suite_index": 0,
    }
    assert strict_json_loads(
        (output / "attempt_chain.json").read_text(), source="attempt chain"
    ) == {"schema": "heliostune.attempt-chain/1"}

    attempts_payload = (output / "attempts.jsonl").read_bytes()
    expected_rows: list[dict[str, str]] = []
    predecessor = hashlib.sha256(b"").hexdigest()
    for cell in verify_suite(_MLP).suite.expected_cells:
        for status in ("pending", "running", "success"):
            row = {
                "cell_id": cell.id,
                "predecessor_sha256": predecessor,
                "status": status,
            }
            expected_rows.append(row)
            predecessor = hashlib.sha256(
                (strict_json_dumps(row, compact=True) + "\n").encode()
            ).hexdigest()
    expected_attempts = "".join(
        strict_json_dumps(row, compact=True) + "\n" for row in expected_rows
    ).encode()
    assert attempts_payload == expected_attempts
    assert bundle.attempts.hash_chain_head == predecessor

    expected_roles = {
        *local_bundle._PROTOCOL_ROLES,
        *local_bundle._EXTRA_ROLES,
        "plugin_suite_0",
        "plugin_suite_1",
    }
    role_paths = {artifact.role: artifact.path for artifact in bundle.artifacts}
    assert set(role_paths) == expected_roles
    assert set(path.name for path in output.iterdir()) == {
        "bundle.json",
        "protocol.json",
        "attempts.jsonl",
        *role_paths.values(),
    }
    for role, path in role_paths.items():
        if role not in {"plugin", "plugin_suite_0", "plugin_suite_1", "observations"}:
            payload = (output / path).read_text(encoding="utf-8")
            assert payload == strict_json_dumps(strict_json_loads(payload, source=role))


def test_bundle_verifies_offline_after_all_source_declarations_are_removed(
    tmp_path: Path,
) -> None:
    plugin_path = _plugin_fixture(tmp_path, (0, 1))
    verified = local_bundle.write_local_bundle(
        _result(),
        plugin_path=plugin_path,
        output_dir=tmp_path / "bundle",
    )

    shutil.rmtree(tmp_path / "declarations")

    reopened = verify_bundle_v1(verified.root_path)
    assert reopened.limitations.plugin_suite_custody == "checked"
    assert (verified.root_path.parent / "plugin_suite_0.json").read_bytes() == _MLP.read_bytes()
    assert (verified.root_path.parent / "plugin_suite_1.json").read_bytes() == _RMS.read_bytes()


def test_selected_suite_descriptor_uses_plugin_reference_order(tmp_path: Path) -> None:
    plugin_path = _plugin_fixture(tmp_path, (1, 0))
    verified = local_bundle.write_local_bundle(
        _result(),
        plugin_path=plugin_path,
        output_dir=tmp_path / "bundle",
    )

    selected = strict_json_loads(
        (verified.root_path.parent / "selected_suite.json").read_text(),
        source="selected suite",
    )
    assert selected == {
        "schema": "heliostune.selected-suite/1",
        "plugin_suite_index": 1,
    }
    assert (verified.root_path.parent / "plugin_suite_0.json").read_bytes() == _RMS.read_bytes()
    assert (verified.root_path.parent / "plugin_suite_1.json").read_bytes() == _MLP.read_bytes()


def test_one_suite_plugin_emits_one_indexed_suite_without_selected_alias(
    tmp_path: Path,
) -> None:
    plugin_path = _plugin_fixture(tmp_path, (0,))
    verified = local_bundle.write_local_bundle(
        _result(),
        plugin_path=plugin_path,
        output_dir=tmp_path / "bundle",
    )
    output = verified.root_path.parent

    assert (output / "plugin_suite_0.json").read_bytes() == _MLP.read_bytes()
    assert not (output / "plugin_suite_1.json").exists()
    assert not (output / "suite.json").exists()
    assert verify_bundle_v1(verified.root_path).limitations.plugin_suite_custody == "checked"


def test_capability_unavailable_writes_valid_aborted_empty_prefix(tmp_path: Path) -> None:
    verified = _write(tmp_path, _result(outcome="aborted", terminal=0, unavailable=True))

    assert verified.bundle.lifecycle.outcome == "aborted"
    assert verified.bundle.coverage.terminal_cells == 0
    assert verified.bundle.attempts.logical == 0
    assert (verified.root_path.parent / "terminal_cells.json").read_text() == "[]\n"
    assert (verified.root_path.parent / "observations.jsonl").read_bytes() == b""
    assert (verified.root_path.parent / "attempts.jsonl").read_bytes() == b""
    assert verified.bundle.attempts.hash_chain_head == hashlib.sha256(b"").hexdigest()
    assert verified.limitations.plugin_suite_custody == "checked"
    assert verified.limitations.attempt_journal_hash_chain == "checked"
    assert verified.limitations.attempt_reconciliation == "checked"


def test_execution_summary_is_reconstructed_without_claims(tmp_path: Path) -> None:
    verified = _write(tmp_path)
    summary = strict_json_loads(
        (verified.root_path.parent / "execution_summary.json").read_text(),
        source="execution summary",
    )
    assert type(summary) is dict
    assert summary["claims"] == []
    assert summary["fusion_claim"] is False
    assert type(summary["summary"]) is dict
    assert summary["summary"]["claims"] == []
    assert summary["summary"]["fusion_claim"] is False
    assert summary["summary"]["terminal_cell_ids"] == [
        cell.id for cell in verify_suite(_MLP).suite.expected_cells
    ]


def test_wrong_observation_metadata_is_rejected_before_publication(tmp_path: Path) -> None:
    result = _result()
    wrong = replace(result.observations[0], case_id="wrong-case")
    result = replace(result, observations=(wrong, *result.observations[1:]))

    with pytest.raises(SchemaError, match="expected case_id"):
        _write(tmp_path, result)
    assert not (tmp_path / "bundle").exists()


def test_observation_correctness_key_must_bind_suite_contract_and_seed(tmp_path: Path) -> None:
    result = _result()
    observation = result.observations[0]
    assert observation.correctness is not None
    wrong_nested = replace(observation.correctness, correctness_key="0" * 64)
    wrong_observation = replace(observation, correctness=wrong_nested)
    result = replace(
        result,
        observations=(wrong_observation, *result.observations[1:]),
    )

    with pytest.raises(SchemaError, match="exact suite, case, arm, seed, and numeric contract"):
        _write(tmp_path, result)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("suite_sha256", "f" * 64, "suite SHA-256"),
        ("case_id", "wrong-case", "case/arm"),
        ("tensor_order", ("input",), "tensor order"),
    ],
)
def test_suite_materialization_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    result = _result()
    if field == "suite_sha256":
        assert isinstance(value, str)
        materialized = replace(result.materialization[0], suite_sha256=value)
    elif field == "case_id":
        assert isinstance(value, str)
        materialized = replace(result.materialization[0], case_id=value)
    else:
        assert isinstance(value, tuple)
        materialized = replace(result.materialization[0], tensor_order=value)
    result = replace(result, materialization=(materialized, *result.materialization[1:]))

    with pytest.raises((ArtifactError, SchemaError), match=message):
        _write(tmp_path, result)
    assert not (tmp_path / "bundle").exists()


def test_materialization_tensor_hash_must_be_sha256(tmp_path: Path) -> None:
    result = _result()
    record = result.materialization[0]
    descriptor = dict(record.tensors[0])
    descriptor["storage_sha256"] = "not-a-sha256"
    wrong_record = replace(record, tensors=(descriptor, *record.tensors[1:]))
    result = replace(
        result,
        materialization=(wrong_record, *result.materialization[1:]),
    )

    with pytest.raises(SchemaError, match="storage SHA-256"):
        _write(tmp_path, result)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples_ms", ("1.0", *tuple(float(sample) for sample in range(2, 51)))),
        ("samples_ms", (True, *tuple(float(sample) for sample in range(2, 51)))),
        ("samples_ms", (float("nan"), *tuple(float(sample) for sample in range(2, 51)))),
        ("samples_ms", (float("inf"), *tuple(float(sample) for sample in range(2, 51)))),
        ("samples_ms", (0.0, *tuple(float(sample) for sample in range(2, 51)))),
        ("median_ms", float("nan")),
        ("median_ms", float("inf")),
        ("median_ms", 99.0),
    ],
)
def test_passing_timing_requires_exact_finite_positive_samples_and_median(
    tmp_path: Path, field: str, value: object
) -> None:
    result = _result()
    index = next(
        index for index, observation in enumerate(result.observations) if observation.timing
    )
    observation = result.observations[index]
    assert observation.timing is not None
    changes: Any = {field: value}
    timing = replace(observation.timing, **changes)
    changed = replace(observation, timing=timing)
    observations = list(result.observations)
    observations[index] = changed

    with pytest.raises(SchemaError):
        _write(tmp_path, replace(result, observations=tuple(observations)))
    assert not (tmp_path / "bundle").exists()


def test_failed_timing_cannot_retain_positive_result_payload(tmp_path: Path) -> None:
    result = _result(outcome="failed", terminal=4)
    observation = result.observations[-1]
    assert observation.timing is not None
    timing = replace(observation.timing, repetitions=1, samples_ms=(1.0,), median_ms=1.0)
    observations = (*result.observations[:-1], replace(observation, timing=timing))

    with pytest.raises(SchemaError, match="positive timing results"):
        _write(tmp_path, replace(result, observations=observations))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "parameter"),
        ("shape", [1]),
        ("draw", "uniform"),
        ("normal_scale", 1),
        ("storage_dtype", "bf16"),
        ("device", "cpu"),
        ("contiguous", False),
        ("alignment_bytes", 1),
        ("alignment_satisfied", False),
        ("storage_sha256", "A" * 64),
    ],
)
def test_materialization_descriptor_must_match_exact_executor_and_suite_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    result = _result()
    record = result.materialization[0]
    descriptor = dict(record.tensors[0])
    descriptor[field] = value
    changed = replace(record, tensors=(descriptor, *record.tensors[1:]))

    with pytest.raises(SchemaError):
        _write(
            tmp_path,
            replace(result, materialization=(changed, *result.materialization[1:])),
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_materialization_descriptor_rejects_missing_or_extra_fields(
    tmp_path: Path, mutation: str
) -> None:
    result = _result()
    record = result.materialization[0]
    descriptor = dict(record.tensors[0])
    if mutation == "missing":
        del descriptor["role"]
    else:
        descriptor["storage_id"] = "not-in-the-producer-contract"
    changed = replace(record, tensors=(descriptor, *record.tensors[1:]))

    with pytest.raises(SchemaError, match="invalid evidence shape"):
        _write(
            tmp_path,
            replace(result, materialization=(changed, *result.materialization[1:])),
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_storage_unchanged", False),
        ("max_abs_error", float("nan")),
        ("max_abs_error", float("inf")),
        ("max_abs_error", -1.0),
    ],
)
def test_passing_correctness_requires_exact_pass_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    result = _result()
    observation = result.observations[0]
    assert observation.correctness is not None
    changes: Any = {field: value}
    correctness = replace(observation.correctness, **changes)
    observations = (replace(observation, correctness=correctness), *result.observations[1:])

    with pytest.raises(SchemaError):
        _write(tmp_path, replace(result, observations=observations))
    assert not (tmp_path / "bundle").exists()


def test_passing_correctness_output_descriptor_is_closed(tmp_path: Path) -> None:
    result = _result()
    observation = result.observations[0]
    assert observation.correctness is not None
    output = dict(observation.correctness.output or {})
    output["claim"] = "publish"
    correctness = replace(observation.correctness, output=output)
    observations = (replace(observation, correctness=correctness), *result.observations[1:])

    with pytest.raises(SchemaError, match="output descriptor"):
        _write(tmp_path, replace(result, observations=observations))
    assert not (tmp_path / "bundle").exists()


def test_compile_failed_cannot_accompany_passed_candidate_correctness(tmp_path: Path) -> None:
    result = _result()
    key, raw = next(iter(result.compile_outcomes.items()))
    failed = dict(raw)
    failed.update(
        status="compile_failed",
        error="injected compile failure",
        first_call_ns=None,
        backend_invoked=False,
        callable_distinct=False,
    )

    with pytest.raises(SchemaError, match="completed exact compile evidence"):
        _write(tmp_path, replace(result, compile_outcomes={key: failed}))
    assert not (tmp_path / "bundle").exists()


def test_lazy_first_call_compile_failure_is_retained_in_failed_bundle(tmp_path: Path) -> None:
    result = _result(outcome="failed", terminal=1)
    key, raw = next(iter(result.compile_outcomes.items()))
    failed = dict(raw)
    failed.update(
        status="compile_failed",
        error="injected lazy compile failure",
        first_call_ns=2,
        backend_invoked=True,
        callable_distinct=True,
    )

    verified = _write(tmp_path, replace(result, compile_outcomes={key: failed}))

    assert verified.bundle.lifecycle.outcome == "failed"
    assert verified.bundle.coverage.failures == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wrapper_create_ns", 0),
        ("wrapper_create_ns", 1.0),
        ("first_call_ns", True),
        ("backend_invoked", False),
        ("callable_distinct", False),
        ("eager_fallback", True),
        ("error", "hidden failure"),
        ("autocast_policy", {"enabled": False}),
    ],
)
def test_completed_compile_outcome_requires_exact_closed_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    result = _result()
    key, raw = next(iter(result.compile_outcomes.items()))
    changed = dict(raw)
    changed[field] = value

    with pytest.raises(SchemaError):
        _write(tmp_path, replace(result, compile_outcomes={key: changed}))
    assert not (tmp_path / "bundle").exists()


def test_compile_outcome_rejects_extra_claim_field(tmp_path: Path) -> None:
    result = _result()
    key, raw = next(iter(result.compile_outcomes.items()))
    changed = dict(raw)
    changed["fusion_claim"] = False

    with pytest.raises(SchemaError, match="invalid evidence shape"):
        _write(tmp_path, replace(result, compile_outcomes={key: changed}))
    assert not (tmp_path / "bundle").exists()


def test_unavailable_environment_cannot_claim_execution(tmp_path: Path) -> None:
    result = _result(outcome="aborted", terminal=0, unavailable=True)
    environment = dict(result.environment)
    environment["backend_invoked"] = True

    with pytest.raises(SchemaError, match="execution evidence"):
        _write(tmp_path, replace(result, environment=environment))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "summary",
    [
        {"claims": ["publish"]},
        {"fusion_claim": True},
        {"nested": {"publication_eligible": True}},
    ],
)
def test_summary_claim_and_publication_overrides_are_rejected(
    tmp_path: Path, summary: dict[str, object]
) -> None:
    with pytest.raises(SchemaError, match="may not override"):
        _write(tmp_path, replace(_result(), summary=summary))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "evidence", ["materialization", "compile_outcomes", "observations", "attempts"]
)
def test_unavailable_result_rejects_execution_evidence(tmp_path: Path, evidence: str) -> None:
    unavailable = _result(outcome="aborted", terminal=0, unavailable=True)
    available = _result()
    result = replace(unavailable, **{evidence: getattr(available, evidence)})

    with pytest.raises(SchemaError):
        _write(tmp_path, result)
    assert not (tmp_path / "bundle").exists()


def test_compile_outcome_must_bind_candidate_case_and_entrypoint(tmp_path: Path) -> None:
    result = _result()
    key, raw = next(iter(result.compile_outcomes.items()))
    mismatched = dict(raw)
    mismatched["entrypoint"] = "wrong.entrypoint"
    result = replace(result, compile_outcomes={key: mismatched})

    with pytest.raises(SchemaError, match="exact entrypoint"):
        _write(tmp_path, result)
    assert not (tmp_path / "bundle").exists()


def test_failed_timing_writes_sealed_exploratory_prefix_with_failure_journal(
    tmp_path: Path,
) -> None:
    result = _result(outcome="failed", terminal=2)
    failure = result.observations[-1]
    assert failure.timing is not None
    verified = _write(tmp_path, result)

    assert verified.bundle.lifecycle.state == "SEALED"
    assert verified.bundle.lifecycle.outcome == "failed"
    assert verified.bundle.coverage.terminal_cells == 2
    assert verified.bundle.coverage.successes == 1
    assert verified.bundle.coverage.failures == 1
    protocol = strict_json_loads(
        (verified.root_path.parent / "protocol.json").read_text(), source="protocol"
    )
    assert type(protocol) is dict
    assert protocol["evidence_class"] == "exploratory"
    assert not verified.publication_eligible
    expected = strict_json_loads(
        (verified.root_path.parent / "expected_cells.json").read_text(), source="expected"
    )
    terminal = strict_json_loads(
        (verified.root_path.parent / "terminal_cells.json").read_text(), source="terminal"
    )
    assert type(expected) is list
    assert terminal == expected[:2]
    observation_rows = [
        strict_json_loads(line, source="observations")
        for line in (verified.root_path.parent / "observations.jsonl").read_text().splitlines()
    ]
    assert observation_rows[-1] == failure.to_dict()
    attempt_rows = [
        strict_json_loads(line, source="attempts")
        for line in (verified.root_path.parent / "attempts.jsonl").read_text().splitlines()
    ]
    assert type(attempt_rows[-1]) is dict
    assert set(attempt_rows[-1]) == {"cell_id", "predecessor_sha256", "status"}
    assert attempt_rows[-1]["cell_id"] == failure.cell_id
    assert attempt_rows[-1]["status"] == "failure"
    assert verify_bundle_v1(verified.root_path).bundle == verified.bundle


def test_tampering_is_detected(tmp_path: Path) -> None:
    verified = _write(tmp_path)
    capability = verified.root_path.parent / "capability_probe.json"
    payload = capability.read_bytes()
    capability.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
        verify_bundle_v1(verified.root_path)


def test_escaping_artifact_path_is_rejected(tmp_path: Path) -> None:
    verified = _write(tmp_path)
    raw = _root_dict(verified.root_path)
    artifacts = raw["artifacts"]
    assert type(artifacts) is list and type(artifacts[0]) is dict
    artifacts[0]["path"] = "../plugin.json"
    write_bytes_atomic(verified.root_path, strict_json_dumps(raw).encode())

    with pytest.raises(SchemaError, match="normalized"):
        verify_bundle_v1(verified.root_path)


def test_symlink_alias_in_closed_inventory_is_rejected(tmp_path: Path) -> None:
    verified = _write(tmp_path)
    output = verified.root_path.parent
    capability = output / "capability_probe.json"
    capability.unlink()
    capability.symlink_to("plugin.json")

    with pytest.raises(ArtifactError):
        verify_bundle_v1(verified.root_path)


def test_plugin_suite_mismatch_fails_before_output_creation(tmp_path: Path) -> None:
    plugin_raw = json.loads(_PLUGIN.read_text(encoding="utf-8"))
    rms_ref = plugin_raw["suite_refs"][1]
    plugin_raw["domains"] = ["rmsnorm_residual"]
    plugin_raw["arm_ids"] = ["rmsnorm-candidate", "rmsnorm-reference"]
    plugin_raw["suite_refs"] = [rms_ref]
    plugin_dir = tmp_path / "standalone/plugins"
    suite_dir = tmp_path / "standalone/suites"
    plugin_dir.mkdir(parents=True)
    suite_dir.mkdir()
    mismatched_plugin = plugin_dir / "plugin.json"
    mismatched_plugin.write_text(strict_json_dumps(plugin_raw), encoding="utf-8")
    (suite_dir / _RMS.name).write_bytes(_RMS.read_bytes())
    assert len(verify_plugin(mismatched_plugin).suites) == 1

    output = tmp_path / "bundle"
    with pytest.raises(ArtifactError, match="not exactly one"):
        local_bundle.write_local_bundle(_result(), plugin_path=mismatched_plugin, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("existing_kind", ["directory", "file", "symlink"])
def test_existing_output_fails_closed(tmp_path: Path, existing_kind: str) -> None:
    output = tmp_path / "bundle"
    if existing_kind == "directory":
        output.mkdir()
    elif existing_kind == "file":
        output.write_text("occupied", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)

    with pytest.raises(ArtifactError, match="already exists"):
        local_bundle.write_local_bundle(_result(), plugin_path=_PLUGIN, output_dir=output)
    if existing_kind == "file":
        assert output.read_text(encoding="utf-8") == "occupied"


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason=_DESCRIPTOR_REASON)
def test_bundle_root_is_staged_last_and_root_failure_leaves_no_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = local_bundle._write_file_at
    writes: list[str] = []

    def fail_at_root(directory_fd: int, name: str, payload: bytes) -> None:
        writes.append(name)
        if name == "bundle.json":
            raise ArtifactError("injected root failure")
        real_write(directory_fd, name, payload)

    monkeypatch.setattr(local_bundle, "_write_file_at", fail_at_root)
    output = tmp_path / "bundle"
    with pytest.raises(ArtifactError, match="injected root failure"):
        local_bundle.write_local_bundle(_result(), plugin_path=_PLUGIN, output_dir=output)

    assert writes[-1] == "bundle.json"
    assert not output.exists()
    assert set(writes[:-1]) == {
        *(f"{role}.json" for role in local_bundle._PROTOCOL_ROLES),
        "plugin_suite_0.json",
        "plugin_suite_1.json",
        "selected_suite.json",
        "attempt_chain.json",
        "terminal_cells.json",
        "observations.jsonl",
        "capability_probe.json",
        "tensor_materialization.json",
        "execution_summary.json",
        "protocol.json",
        "attempts.jsonl",
    }
    assert not any(path.name.startswith(".heliostune-bundle-") for path in tmp_path.iterdir())


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason=_DESCRIPTOR_REASON)
@pytest.mark.parametrize(
    "name",
    ["plugin_suite_0.json", "plugin_suite_1.json", "selected_suite.json", "attempt_chain.json"],
)
@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_opt_in_custody_artifact_damage_prevents_publication_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mutation: str,
) -> None:
    real_write = local_bundle._write_file_at

    def damage_artifact(directory_fd: int, staged_name: str, payload: bytes) -> None:
        if staged_name == name:
            if mutation == "tampered":
                real_write(directory_fd, staged_name, payload + b" ")
            return
        real_write(directory_fd, staged_name, payload)

    monkeypatch.setattr(local_bundle, "_write_file_at", damage_artifact)
    output = tmp_path / "bundle"
    with pytest.raises(ArtifactError):
        local_bundle.write_local_bundle(_result(), plugin_path=_PLUGIN, output_dir=output)

    assert not output.exists()
    assert not any(path.name.startswith(".heliostune-bundle-") for path in tmp_path.iterdir())


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason=_DESCRIPTOR_REASON)
@pytest.mark.parametrize(
    "control",
    ["plugin_suite_custody", "attempt_journal_hash_chain", "attempt_reconciliation"],
)
def test_unchecked_required_control_prevents_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    real_verify = local_bundle.verify_bundle_v1_from_directory_fd

    def remove_control(
        directory_fd: int,
        root_relative_path: str = "bundle.json",
        *,
        diagnostic_directory: str | Path | None = None,
    ) -> VerifiedBundle:
        verified = real_verify(
            directory_fd,
            root_relative_path,
            diagnostic_directory=diagnostic_directory,
        )
        return replace(
            verified,
            limitations=replace(verified.limitations, **{control: "not_checked"}),
        )

    monkeypatch.setattr(
        local_bundle,
        "verify_bundle_v1_from_directory_fd",
        remove_control,
    )
    output = tmp_path / "bundle"
    with pytest.raises(ArtifactError, match="did not check required controls"):
        local_bundle.write_local_bundle(_result(), plugin_path=_PLUGIN, output_dir=output)

    assert not output.exists()
    assert not any(path.name.startswith(".heliostune-bundle-") for path in tmp_path.iterdir())


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason=_DESCRIPTOR_REASON)
def test_staged_verification_failure_leaves_no_final_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    real_verify = local_bundle.verify_bundle_v1_from_directory_fd
    saw_pinned_directory = False

    def fail_verification(
        directory_fd: int,
        root_relative_path: str = "bundle.json",
        *,
        diagnostic_directory: str | Path | None = None,
    ) -> VerifiedBundle:
        nonlocal saw_pinned_directory
        os.stat(root_relative_path, dir_fd=directory_fd, follow_symlinks=False)
        assert diagnostic_directory == output
        assert not output.exists()
        real_verify(
            directory_fd,
            root_relative_path,
            diagnostic_directory=diagnostic_directory,
        )
        saw_pinned_directory = True
        raise ArtifactError("injected staged verification failure")

    monkeypatch.setattr(
        local_bundle,
        "verify_bundle_v1_from_directory_fd",
        fail_verification,
    )
    with pytest.raises(ArtifactError, match="injected staged verification failure"):
        local_bundle.write_local_bundle(_result(), plugin_path=_PLUGIN, output_dir=output)

    assert saw_pinned_directory
    assert not output.exists()
    assert not any(path.name.startswith(".heliostune-bundle-") for path in tmp_path.iterdir())


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason=_DESCRIPTOR_REASON)
def test_staging_path_swap_cannot_redirect_pinned_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_verify = local_bundle.verify_bundle_v1_from_directory_fd
    swapped = tmp_path / "verified-staging"
    calls = 0

    def swap_path(
        directory_fd: int,
        root_relative_path: str = "bundle.json",
        *,
        diagnostic_directory: str | Path | None = None,
    ) -> VerifiedBundle:
        nonlocal calls
        calls += 1
        if calls != 1:
            return real_verify(
                directory_fd,
                root_relative_path,
                diagnostic_directory=diagnostic_directory,
            )
        staging = next(
            path for path in tmp_path.iterdir() if path.name.startswith(".heliostune-bundle-")
        )
        staging.rename(swapped)
        staging.mkdir()
        (staging / "bundle.json").write_bytes(b"unverified pathname bytes")
        try:
            return real_verify(
                directory_fd,
                root_relative_path,
                diagnostic_directory=diagnostic_directory,
            )
        finally:
            shutil.rmtree(staging)
            swapped.rename(staging)

    monkeypatch.setattr(
        local_bundle,
        "verify_bundle_v1_from_directory_fd",
        swap_path,
    )
    verified = _write(tmp_path)

    assert calls == 2
    assert verified.root_path.read_bytes() != b"unverified pathname bytes"
    assert verify_bundle_v1(verified.root_path).root_sha256 == verified.root_sha256


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason=_DESCRIPTOR_REASON)
def test_post_rename_tampering_is_reverified_and_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bundle"
    real_rename = local_bundle._rename_directory_noreplace

    def tamper_after_rename(parent_fd: int, source: str, destination: str) -> None:
        real_rename(parent_fd, source, destination)
        capability = output / "capability_probe.json"
        payload = capability.read_bytes()
        capability.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    monkeypatch.setattr(local_bundle, "_rename_directory_noreplace", tamper_after_rename)
    with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
        local_bundle.write_local_bundle(_result(), plugin_path=_PLUGIN, output_dir=output)

    assert not output.exists()
    assert not any(path.name.startswith(".heliostune-bundle-") for path in tmp_path.iterdir())


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason=_DESCRIPTOR_REASON)
def test_parent_rename_and_symlink_substitution_cannot_redirect_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    output = parent / "bundle"
    real_write = local_bundle._write_file_at
    substituted = False

    def substitute_parent(directory_fd: int, name: str, payload: bytes) -> None:
        nonlocal substituted
        real_write(directory_fd, name, payload)
        if not substituted:
            parent.rename(moved_parent)
            parent.symlink_to(replacement, target_is_directory=True)
            substituted = True

    monkeypatch.setattr(local_bundle, "_write_file_at", substitute_parent)
    with pytest.raises(ArtifactError, match="parent identity changed"):
        local_bundle.write_local_bundle(_result(), plugin_path=_PLUGIN, output_dir=output)

    assert substituted
    assert not output.exists()
    assert not (moved_parent / "bundle").exists()
    assert not any(path.name.startswith(".heliostune-bundle-") for path in moved_parent.iterdir())


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason=_DESCRIPTOR_REASON)
@pytest.mark.parametrize("substitute", ["before", "after"])
def test_parent_substitution_at_publication_linearization_never_returns_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitute: str,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    sentinel = replacement / "owned-by-replacement"
    sentinel.write_text("preserve", encoding="utf-8")
    output = parent / "bundle"
    real_rename = local_bundle._rename_directory_noreplace

    def substitute_parent(parent_fd: int, source: str, destination: str) -> None:
        if substitute == "after":
            real_rename(parent_fd, source, destination)
        parent.rename(moved_parent)
        parent.symlink_to(replacement, target_is_directory=True)
        if substitute == "before":
            real_rename(parent_fd, source, destination)

    monkeypatch.setattr(local_bundle, "_rename_directory_noreplace", substitute_parent)
    with pytest.raises(ArtifactError, match="parent identity changed"):
        local_bundle.write_local_bundle(_result(), plugin_path=_PLUGIN, output_dir=output)

    assert not output.exists()
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (replacement / "bundle").exists()


@pytest.mark.skipif(not _DESCRIPTOR_PUBLICATION, reason=_DESCRIPTOR_REASON)
def test_destination_inode_substitution_after_rename_is_never_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    renamed_original = tmp_path / "renamed-original"
    real_rename = local_bundle._rename_directory_noreplace

    def substitute_destination(parent_fd: int, source: str, destination: str) -> None:
        real_rename(parent_fd, source, destination)
        output.rename(renamed_original)
        output.mkdir()
        (output / "replacement-owned").write_text("preserve", encoding="utf-8")

    monkeypatch.setattr(local_bundle, "_rename_directory_noreplace", substitute_destination)
    with pytest.raises(ArtifactError, match="published bundle identity changed"):
        local_bundle.write_local_bundle(_result(), plugin_path=_PLUGIN, output_dir=output)

    assert (output / "replacement-owned").read_text(encoding="utf-8") == "preserve"
    assert renamed_original.is_dir()
