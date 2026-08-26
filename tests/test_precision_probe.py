from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS

_REPO = Path(__file__).resolve().parents[1]
_PROBE_MODULE = _REPO / "modal_precision_probe.py"
_WHEEL_NAME = "heliostune-0.4.0-py3-none-any.whl"
_ANALYZER_MODULE = _REPO / "scripts/analyze_precision_probe.py"


def _load_analyzer() -> ModuleType:
    name = "_test_analyze_precision_probe"
    spec = importlib.util.spec_from_file_location(name, _ANALYZER_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ANALYZER = _load_analyzer()
evaluate_explanation = _ANALYZER.evaluate_explanation
_ARCHIVE = _REPO / "benchmarks/data/parhelion-v2-measurements.jsonl.zst"


class _FakeImage:
    @classmethod
    def debian_slim(cls, **_kwargs: object) -> _FakeImage:
        return cls()

    def pip_install(self, *_args: object) -> _FakeImage:
        return self

    def add_local_file(self, *_args: object, **_kwargs: object) -> _FakeImage:
        return self

    def run_commands(self, *_args: object) -> _FakeImage:
        return self

    def env(self, *_args: object) -> _FakeImage:
        return self


class _FakeApp:
    def __init__(self, name: str) -> None:
        self.name = name

    def function(self, **_kwargs: object) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda function: function

    def local_entrypoint(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda function: function


def _source_digest(repository: Path) -> str:
    package = repository / "src/heliostune"
    digest = hashlib.sha256()
    paths = sorted(
        path for path in package.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    )
    for path in paths:
        digest.update(f"heliostune/{path.relative_to(package).as_posix()}".encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _wheel_manifest(*, wheel: Path, head: str, source_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "head_commit": head,
        "source_sha256": source_sha256,
        "wheel_filename": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "python_version": "3.11",
        "pip_dependencies": [
            "numpy==2.4.6",
            "rich==14.3.4",
            "zstandard==0.25.0",
            "torch==2.8.0",
            "triton==3.4.0",
        ],
        "build_dependencies": ["hatchling==1.32.0"],
        "build_tools": {"uv": "0.12.5", "hatchling": "1.32.0"},
        "wheel_install_args": ["--no-deps"],
    }


def _load_probe() -> ModuleType:
    modal = ModuleType("modal")
    modal.App = _FakeApp  # type: ignore[attr-defined]
    modal.Image = _FakeImage  # type: ignore[attr-defined]
    previous_modal = sys.modules.get("modal")
    previous_wheel = os.environ.get("HELIOSTUNE_MODAL_WHEEL")
    previous_run = subprocess.run
    head = "a" * 40
    with tempfile.TemporaryDirectory() as directory:
        wheel = Path(directory) / _WHEEL_NAME
        wheel.write_bytes(b"stand-in-wheel")
        wheel.with_name(f"{wheel.name}.manifest.json").write_text(
            json.dumps(
                _wheel_manifest(
                    wheel=wheel,
                    head=head,
                    source_sha256=_source_digest(_REPO),
                )
            ),
            encoding="utf-8",
        )

        def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
            stdout = "" if args[1:3] == ["status", "--porcelain"] else f"{head}\n"
            return SimpleNamespace(stdout=stdout)

        sys.modules["modal"] = modal
        os.environ["HELIOSTUNE_MODAL_WHEEL"] = str(wheel)
        subprocess.run = fake_run  # type: ignore[assignment]
        try:
            spec = importlib.util.spec_from_file_location(
                "_test_modal_precision_probe", _PROBE_MODULE
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            subprocess.run = previous_run
            if previous_modal is None:
                del sys.modules["modal"]
            else:
                sys.modules["modal"] = previous_modal
            if previous_wheel is None:
                del os.environ["HELIOSTUNE_MODAL_WHEEL"]
            else:
                os.environ["HELIOSTUNE_MODAL_WHEEL"] = previous_wheel
    return module


@pytest.fixture(scope="module")
def precision_probe() -> ModuleType:
    return _load_probe()


def _arm(*, reduced: bool) -> dict[str, object]:
    return {
        "latency_ms": 1.0,
        "latency_p20_ms": 0.9,
        "latency_p80_ms": 1.1,
        "benchmark_wall_ms": 2.0,
        "max_abs_error": 0.0,
        "allow_fp16_reduced_precision_reduction": reduced,
        "allow_tf32": False,
    }


def _payload(precision_probe: ModuleType, bank: int = 0) -> tuple[dict[str, object], str, str]:
    workload = DEFAULT_WORKLOADS[0]
    config = DEFAULT_CONFIGS[0]
    order_seed = precision_probe.probe_seed(
        purpose="arm-order", bank=bank, workload_key=workload.key
    )
    canonical = list(DEFAULT_WORKLOADS)
    random.Random(bank).shuffle(canonical)
    workload_index = canonical.index(workload)
    triton = {
        "latency_ms": 1.0,
        "latency_p20_ms": 0.9,
        "latency_p80_ms": 1.1,
        "benchmark_wall_ms": 2.0,
        "max_abs_error": 0.0,
        "compile_ms": 0.0,
        "config": config.to_dict(),
        "config_key": config.key,
    }
    row = {
        "bank": bank,
        "workload": workload.to_dict(),
        "workload_key": workload.key,
        "workload_index": workload_index,
        "tensor_seed": bank * 10_000 + workload_index,
        "arm_order_seed": order_seed,
        "arm_order": ["torch_strict", "triton", "torch_reduced"],
        "reference": {
            "dtype": "float32",
            "allow_tf32": False,
            "allow_fp16_reduced_precision_reduction": False,
        },
        "arms": {
            "torch_reduced": _arm(reduced=True),
            "torch_strict": _arm(reduced=False),
            "triton": triton,
        },
    }
    payload = {
        "probe": "h100-fp16-reduction-probe",
        "schema_version": 2,
        "gpu": "H100",
        "bank": bank,
        "warmup_ms": 25,
        "rep_ms": 100,
        "hardware": {
            "gpu": "H100",
            "device_name": "NVIDIA H100 80GB HBM3",
            "compute_capability": [9, 0],
            "multiprocessor_count": 120,
            "total_memory_gb": 79.0,
            "cuda_version": "12.8",
            "torch_version": "2.8.0",
            "triton_version": "3.4.0",
        },
        "rows": [row],
    }
    return payload, workload.key, config.key


def _validate_payload(precision_probe: ModuleType, payload: object, key: str, config: str) -> None:
    precision_probe._validated_payload(
        payload,
        bank=0,
        warmup_ms=25,
        rep_ms=100,
        workload_keys=(key,),
        triton_config_keys={key: config},
    )


def test_archive_baseline_selects_bank_one_and_scores_bank_two(
    precision_probe: ModuleType,
) -> None:
    baseline = precision_probe.archive_baseline(_ARCHIVE, "H100")
    assert baseline["selection_bank"] == 1
    assert baseline["evaluation_bank"] == 2
    assert baseline["banks"] == [1, 2]
    assert baseline["published_endpoint_torch_over_best_triton"] == 0.627266
    assert int(baseline["median_torch_over_best_triton"] * 1_000_000) / 1_000_000 == 0.627266
    assert {row["configs_considered"] for row in baseline["workloads"].values()} == {36}


@pytest.mark.parametrize("value", ["0", "0,1", "0,1,1", "0,1,3", "2,1,0"])
def test_probe_rejects_nonfrozen_bank_sets(precision_probe: ModuleType, value: str) -> None:
    with pytest.raises(ValueError):
        precision_probe._parse_banks(value)


def test_payload_validation_accepts_exact_payload(precision_probe: ModuleType) -> None:
    payload, key, config = _payload(precision_probe)
    data, rows = precision_probe._validated_payload(
        payload,
        bank=0,
        warmup_ms=25,
        rep_ms=100,
        workload_keys=(key,),
        triton_config_keys={key: config},
    )
    assert data["gpu"] == "H100"
    assert len(rows) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("gpu", "A100"),
        lambda payload: payload.__setitem__("warmup_ms", 24),
        lambda payload: payload["rows"][0]["arm_order"].__setitem__(0, "triton"),
        lambda payload: payload["rows"][0]["arms"]["triton"].__setitem__("latency_ms", -1.0),
        lambda payload: payload["rows"].append(copy.deepcopy(payload["rows"][0])),
    ],
)
def test_payload_validation_rejects_wrong_or_duplicate_data(
    precision_probe: ModuleType,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    payload, key, config = _payload(precision_probe)
    mutation(payload)
    with pytest.raises((TypeError, ValueError)):
        _validate_payload(precision_probe, payload, key, config)


def test_cross_product_must_be_exact(precision_probe: ModuleType) -> None:
    rows = [
        {"bank": bank, "workload_key": workload} for bank in (0, 1, 2) for workload in ("a", "b")
    ]
    precision_probe._validate_cross_product(rows, banks=(0, 1, 2), workload_keys=("a", "b"))
    with pytest.raises(ValueError, match="banks × requested workloads"):
        precision_probe._validate_cross_product(
            rows[:-1], banks=(0, 1, 2), workload_keys=("a", "b")
        )


def test_probe_output_rejects_symlinked_benchmark_path(
    precision_probe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    (tmp_path / "linked").symlink_to(benchmarks, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="benchmarks"):
        precision_probe._resolved_output("linked/probe.json", repository=tmp_path)
    with pytest.raises(ValueError):
        precision_probe._resolved_output("../probe.json", repository=tmp_path)
    with pytest.raises(ValueError):
        precision_probe._resolved_output(str(tmp_path / "probe.json"), repository=tmp_path)


def _gate_payload(precision_probe: ModuleType) -> dict[str, object]:
    checks = sorted(precision_probe._expected_gate_checks())
    results = [
        {
            "kernel": kernel,
            "config_key": config_key,
            "workload_key": workload_key,
            "correct": True,
            "max_abs_error": 0.0,
            "error": None,
        }
        for kernel, config_key, workload_key in checks
    ]
    grouped: dict[tuple[str, str], int] = {}
    for kernel, config_key, _workload_key in checks:
        grouped[(kernel, config_key)] = grouped.get((kernel, config_key), 0) + 1
    summaries = [
        {
            "kernel": kernel,
            "config_key": config_key,
            "check_count": check_count,
            "correct": True,
            "max_abs_error": 0.0,
            "errors": [],
        }
        for (kernel, config_key), check_count in sorted(grouped.items())
    ]
    configs = precision_probe._gate_config_manifest()
    return {
        "schema_version": 1,
        "gate": "hopper-candidate-correctness",
        "study_status": "post_hoc_exploratory",
        "analysis_status": "post_hoc_exploratory",
        "gpu_selector": "H100!",
        "gpu": "H100",
        "hardware": {
            "gpu": "H100",
            "device_name": "NVIDIA H100 80GB HBM3",
            "compute_capability": [9, 0],
            "multiprocessor_count": 120,
            "total_memory_gb": 79.0,
            "cuda_version": "12.8",
            "torch_version": "2.8.0",
            "triton_version": "3.4.0",
        },
        "config_counts": {
            "hopper_gemm": len(configs["hopper_gemm"]),
            "skinny_gemv": len(configs["skinny_gemv"]),
            "total": len(configs["hopper_gemm"]) + len(configs["skinny_gemv"]),
        },
        "config_manifest_sha256": precision_probe._gate_config_manifest_sha256(configs),
        "validation_workload_count": len({workload for _, _, workload in checks}),
        "validation_check_count": len(results),
        "candidate_summaries": summaries,
        "validation_results": results,
    }


def test_gate_payload_accepts_exact_complete_result(precision_probe: ModuleType) -> None:
    payload = _gate_payload(precision_probe)
    data, summaries, results = precision_probe._validated_gate_payload(payload)
    assert data["gpu_selector"] == "H100!"
    assert len(summaries) == 71
    assert len(results) == payload["validation_check_count"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["validation_results"].pop(),
        lambda payload: payload["validation_results"][0].__setitem__("correct", False),
        lambda payload: payload["validation_results"][0].__setitem__("config_key", "wrong"),
        lambda payload: payload["hardware"].__setitem__("device_name", "NVIDIA A100"),
        lambda payload: payload["hardware"].__setitem__("compute_capability", [8, 0]),
    ],
)
def test_gate_payload_rejects_incomplete_wrong_or_non_h100_results(
    precision_probe: ModuleType,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    payload = _gate_payload(precision_probe)
    mutation(payload)
    with pytest.raises((TypeError, ValueError)):
        precision_probe._validated_gate_payload(payload)


def test_gate_output_requires_artifacts_and_rejects_benchmark_symlink(
    precision_probe: ModuleType,
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    benchmarks = tmp_path / "benchmarks"
    artifacts.mkdir()
    benchmarks.mkdir()
    (artifacts / "linked").symlink_to(benchmarks, target_is_directory=True)

    expected = (artifacts / "hopper.json").resolve()
    assert (
        precision_probe._resolved_gate_output(
            "artifacts/hopper.json",
            repository=tmp_path,
        )
        == expected
    )
    for invalid in (
        "hopper.json",
        "benchmarks/hopper.json",
        "artifacts/linked/hopper.json",
        "../artifacts/hopper.json",
        str(expected),
    ):
        with pytest.raises(ValueError):
            precision_probe._resolved_gate_output(invalid, repository=tmp_path)


def test_gate_rejects_stale_wheel_manifest(
    precision_probe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    package = repository / "src/heliostune"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""tiny source."""\n', encoding="utf-8")
    wheel = tmp_path / _WHEEL_NAME
    wheel.write_bytes(b"wheel")
    wheel.with_name(f"{wheel.name}.manifest.json").write_text(
        json.dumps(
            _wheel_manifest(
                wheel=wheel,
                head="a" * 40,
                source_sha256=precision_probe._source_digest(repository),
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(precision_probe, "_git_head", lambda _repository: "b" * 40)

    with pytest.raises(RuntimeError, match="current HEAD"):
        precision_probe._validated_wheel_provenance(
            wheel,
            head_commit="b" * 40,
            repository=repository,
        )


class _GateCall:
    object_id = "fc-hopper-gate"

    def __init__(self, payload: object, journal: Path, *, failure: Exception | None = None) -> None:
        self.payload = payload
        self.journal = journal
        self.failure = failure
        self.get_count = 0

    def get(self) -> object:
        self.get_count += 1
        records = [json.loads(line) for line in self.journal.read_text().splitlines()]
        assert [(record["call_id"], record["status"]) for record in records] == [
            (self.object_id, "spawned")
        ]
        if self.failure is not None:
            raise self.failure
        return self.payload


class _GateSpawner:
    def __init__(self, call: _GateCall) -> None:
        self.call = call
        self.spawn_count = 0

    def spawn(self) -> _GateCall:
        self.spawn_count += 1
        return self.call


def _configure_gate_run(
    precision_probe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    payload: object,
    failure: Exception | None = None,
) -> tuple[Path, Path, _GateCall, _GateSpawner]:
    destination = tmp_path / "artifacts/hopper-correctness.json"
    journal = Path(f"{destination}.attempts.jsonl")
    wheel = tmp_path / _WHEEL_NAME
    wheel.write_bytes(b"wheel")
    wheel_manifest = wheel.with_name(f"{wheel.name}.manifest.json")
    wheel_manifest.write_text("{}", encoding="utf-8")
    call = _GateCall(payload, journal, failure=failure)
    spawner = _GateSpawner(call)
    monkeypatch.setattr(
        precision_probe,
        "_resolved_gate_output",
        lambda _output: destination,
    )
    monkeypatch.setattr(precision_probe, "_resolve_wheel", lambda _wheel: wheel)
    monkeypatch.setattr(
        precision_probe,
        "_git_identity",
        lambda: ("a" * 40, hashlib.sha256(("a" * 40).encode()).hexdigest()),
    )
    monkeypatch.setattr(
        precision_probe,
        "_validated_wheel_provenance",
        lambda *_args, **_kwargs: {
            "manifest_path": str(wheel_manifest),
            "manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
            "head_commit": "a" * 40,
            "source_sha256": "b" * 64,
            "wheel_sha256": hashlib.sha256(b"wheel").hexdigest(),
        },
    )
    monkeypatch.setattr(precision_probe, "hopper_correctness_h100", spawner)
    return destination, journal, call, spawner


def test_gate_failed_get_is_journaled_and_writes_no_artifact(
    precision_probe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination, journal, call, spawner = _configure_gate_run(
        precision_probe,
        monkeypatch,
        tmp_path,
        payload={},
        failure=RuntimeError("remote failed"),
    )
    with pytest.raises(RuntimeError, match="remote failed"):
        precision_probe.hopper_gate()

    assert spawner.spawn_count == 1
    assert call.get_count == 1
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [record["status"] for record in records] == ["spawned", "failed"]
    assert not destination.exists()
    assert not Path(f"{destination}.manifest.json").exists()


def test_gate_success_writes_digest_valid_artifact_and_sidecar(
    precision_probe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _gate_payload(precision_probe)
    destination, journal, call, spawner = _configure_gate_run(
        precision_probe,
        monkeypatch,
        tmp_path,
        payload=payload,
    )
    precision_probe.hopper_gate()

    assert spawner.spawn_count == 1
    assert call.get_count == 1
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [record["status"] for record in records] == ["spawned", "completed"]
    artifact = json.loads(destination.read_text())
    sidecar = json.loads(Path(f"{destination}.manifest.json").read_text())
    assert artifact["verified"] is True
    assert artifact["correctness_only"] is True
    assert artifact["performance_validated"] is False
    assert len(artifact["config_manifest"]["hopper_gemm"]) == 23
    assert len(artifact["config_manifest"]["skinny_gemv"]) == 48
    assert sidecar["data"]["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert sidecar["attempt_journal"]["sha256"] == hashlib.sha256(journal.read_bytes()).hexdigest()


class _FailedCall:
    object_id = "call-0"

    def get(self) -> object:
        raise RuntimeError("remote failure")


class _FailingSpawner:
    def __init__(self) -> None:
        self.banks: list[int] = []

    def spawn(self, **kwargs: object) -> _FailedCall:
        self.banks.append(int(kwargs["bank"]))
        return _FailedCall()


def test_failure_prevents_later_bank_spawns(
    precision_probe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workload = DEFAULT_WORKLOADS[0]
    config = DEFAULT_CONFIGS[0]
    archive = tmp_path / "archive.zst"
    archive.write_bytes(b"archive")
    wheel = tmp_path / _WHEEL_NAME
    wheel.write_bytes(b"wheel")
    baseline = {
        "workloads": {workload.key: {"best_triton_config_key": config.key}},
        "selection_bank": 1,
        "evaluation_bank": 2,
    }
    spawner = _FailingSpawner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(precision_probe, "archive_baseline", lambda *_args: baseline)
    monkeypatch.setattr(precision_probe, "_resolve_wheel", lambda _value: wheel)
    monkeypatch.setattr(precision_probe, "_git_identity", lambda: ("a" * 40, "b" * 64))
    monkeypatch.setattr(precision_probe, "probe_h100", spawner)
    with pytest.raises(RuntimeError, match="remote failure"):
        precision_probe.main(
            archive=str(archive),
            output="probe.json",
            workloads=workload.key,
        )
    assert spawner.banks == [0]
    journal = tmp_path / "probe.json.attempts.jsonl"
    statuses = [json.loads(line)["status"] for line in journal.read_text().splitlines()]
    assert statuses == ["spawned", "failed"]
    assert not (tmp_path / "probe.json").exists()


def _summary(
    *,
    reduced: float,
    strict: float,
    triton: float = 1.0,
    reduced_error: float = 0.0,
    strict_error: float = 0.0,
) -> Any:
    return _ANALYZER.WorkloadSummary(
        key="w",
        m=1,
        n=1,
        k=1,
        banks=(0, 1, 2),
        latencies={"torch_reduced": reduced, "torch_strict": strict, "triton": triton},
        errors={"torch_reduced": reduced_error, "torch_strict": strict_error, "triton": 0.0},
        archive_ratio=0.627266,
    )


def test_identical_reduced_and_strict_does_not_explain() -> None:
    verdict = evaluate_explanation([_summary(reduced=0.627266, strict=0.627266)])
    assert verdict.classification == "does not explain"


def test_material_effect_without_baseline_agreement_is_inconclusive() -> None:
    verdict = evaluate_explanation([_summary(reduced=0.9, strict=1.0)])
    assert verdict.classification == "inconclusive"
    assert not verdict.baseline_agrees


def test_material_effect_with_baseline_agreement_supports_explanation() -> None:
    verdict = evaluate_explanation([_summary(reduced=0.627266, strict=0.7)])
    assert verdict.classification == "supports explanation"
    assert verdict.parity_authorized


def test_accuracy_regression_prevents_parity_authorization() -> None:
    verdict = evaluate_explanation(
        [_summary(reduced=0.627266, strict=0.7, reduced_error=2.0, strict_error=1.0)]
    )
    assert verdict.classification == "supports explanation"
    assert verdict.accuracy_regression
    assert not verdict.parity_authorized
