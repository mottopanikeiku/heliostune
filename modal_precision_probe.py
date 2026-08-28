"""Probe whether FP16 reduced-precision reduction explains torch's H100 matmul lead.

Question: the frozen Parhelion v2 H100 matrix reports ``torch.matmul`` beating the
bank-1-selected, bank-2-scored best of 36 Triton launch configurations on 96/96
workloads (torch/best-Triton latency endpoint 0.627266). The Triton kernel in
``heliostune.kernel`` hard-codes an FP32 accumulator,
while PyTorch 2.8 defaults ``torch.backends.cuda.matmul.allow_fp16_reduced_precision_
reduction`` to ``True``, permitting FP16 split-K reduction. This probe times, on one
H100 and within a single run, ``torch.matmul`` with that flag enabled, ``torch.matmul``
with it disabled, and the archive-winning Triton configuration, and records the max
absolute error of all three against an FP32-output reference.

Status: post-hoc exploratory diagnostics. This is NOT a confirmatory Parhelion endpoint,
it is not covered by ``benchmarks/parhelion-v2-h100-freeze.json``, and nothing it emits
may be used to revise a published claim. It only tells us whether the reduction flag is
a plausible cause of the observed gap.

Run the timing probe with ``modal run modal_precision_probe.py::main``. Run the
correctness-only gate with ``modal run modal_precision_probe.py::hopper_gate``.
Failed remote calls are never retried.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import re
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import modal

# Keep the image definition in this entrypoint: Modal mounts only this file before the
# wheel is installed, so importing another repository entrypoint would make remote
# startup depend on a file that is not present in the container.
_WHEEL_FILENAME = re.compile(
    r"[A-Za-z0-9_.]+"
    r"-[A-Za-z0-9_.!+]+"
    r"(-[0-9][A-Za-z0-9_.]*)?"
    r"-[A-Za-z0-9_.]+"
    r"-[A-Za-z0-9_.]+"
    r"-[A-Za-z0-9_.]+"
    r"\.whl"
)
_REPO = Path(__file__).resolve().parent
_WHEEL_DIRECTORY = "artifacts/modal-wheel"
_WHEEL_MANIFEST_SCHEMA_VERSION = 1
_PYTHON_VERSION = "3.11"
_PIP_DEPENDENCIES = (
    "numpy==2.4.6",
    "rich==14.3.4",
    "zstandard==0.25.0",
    "torch==2.8.0",
    "triton==3.4.0",
)
_BUILD_DEPENDENCIES = ("hatchling==1.32.0",)
_BUILD_TOOLS = {"uv": "0.12.5", "hatchling": "1.32.0"}

_GPU = "H100"
_MODAL_SELECTOR = "H100!"
_PROBE_NAME = "h100-fp16-reduction-probe"
_PROBE_SCHEMA_VERSION = 2
_GATE_NAME = "hopper-candidate-correctness"
_GATE_SCHEMA_VERSION = 1
_GATE_STATUS = "post_hoc_exploratory"
_GATE_TIMEOUT_SECONDS = 20 * 60
_SEED_NAMESPACE = "heliostune-precision-probe-v1"
_ARM_NAMES = ("torch_reduced", "torch_strict", "triton")
# The frozen H100 protocol: benchmarks/parhelion-v2-h100-freeze.json
# final_evaluation.{warmup_ms,repetition_ms,replicates,banks} with the median of
# heliostune.kernel._timed_do_bench's [0.2, 0.5, 0.8] quantiles.
_FROZEN_WARMUP_MS = 25
_FROZEN_REP_MS = 100
_FROZEN_BANK_VALUES = (0, 1, 2)
_FROZEN_BANKS = "0,1,2"
_FROZEN_QUANTILES = (0.2, 0.5, 0.8)
_FROZEN_ARCHIVE_ENDPOINT = 0.627266
_DEFAULT_ARCHIVE = "benchmarks/data/parhelion-v2-measurements.jsonl.zst"
_DEFAULT_OUTPUT = "artifacts/h100-precision-probe.json"
_DEFAULT_GATE_OUTPUT = "artifacts/hopper-correctness.json"
_BENCHMARK_STUDY_ID = "hopper-h100-engineering-benchmark"
_BENCHMARK_SCHEMA_VERSION = 1
_BENCHMARK_TIMEOUT_SECONDS = 60 * 60
_BENCHMARK_BANK = 0
_BENCHMARK_WARMUP_MS = 25
_BENCHMARK_REP_MS = 100
_BENCHMARK_QUANTILES = (0.2, 0.5, 0.8)
_BENCHMARK_EXPECTED_ROWS = 32 * 48 + 64 * 23
_DEFAULT_BENCHMARK_OUTPUT = "artifacts/hopper-h100-engineering.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serialized_json_sha256(value: object) -> str:
    """Hash the exact bytes that ``write_json_atomic`` will publish."""
    from heliostune.artifacts import strict_json_dumps

    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _source_digest(repository: Path) -> str:
    package = repository / "src/heliostune"
    if not package.is_dir():
        raise RuntimeError(f"HeliosTune source directory does not exist: {package}")
    digest = hashlib.sha256()
    paths = sorted(
        path for path in package.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    )
    for path in paths:
        name = f"heliostune/{path.relative_to(package).as_posix()}"
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def wheel_manifest_path(wheel: Path) -> Path:
    return wheel.with_name(f"{wheel.name}.manifest.json")


def remote_wheel_path(wheel: Path) -> str:
    name = wheel.name
    if _WHEEL_FILENAME.fullmatch(name) is None:
        raise ValueError(
            f"Modal wheel is not a valid PEP 427 filename: {name}; expected "
            "distribution-version(-build)?-python-abi-platform.whl"
        )
    return f"/root/{name}"


def remote_wheel_manifest_path(wheel: Path) -> str:
    return f"{remote_wheel_path(wheel)}.manifest.json"


def _git_head(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("Modal wheel use requires a clean Git HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_object(path: Path) -> dict[str, object]:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"wheel manifest contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read Modal wheel manifest {path}: {exc}") from exc
    if type(value) is not dict:
        raise RuntimeError(f"Modal wheel manifest must be a JSON object: {path}")
    return cast(dict[str, object], value)


def validate_wheel_manifest(
    wheel: Path,
    *,
    repository: Path | None = None,
    remote: bool = False,
) -> Path:
    """Validate wheel bytes and, locally, bind them to the current HEAD and source."""
    configured_manifest = os.environ.get("HELIOSTUNE_MODAL_WHEEL_MANIFEST")
    if remote:
        if not configured_manifest:
            raise RuntimeError("HELIOSTUNE_MODAL_WHEEL_MANIFEST is required for a remote wheel")
        manifest = Path(configured_manifest)
        if manifest != wheel_manifest_path(wheel):
            raise RuntimeError("remote Modal wheel manifest must be adjacent to the wheel")
    else:
        manifest = wheel_manifest_path(wheel)
    if not manifest.is_file():
        raise RuntimeError(f"Modal wheel manifest does not exist: {manifest}")
    data = _manifest_object(manifest)
    expected_fields = {
        "schema_version",
        "head_commit",
        "source_sha256",
        "wheel_filename",
        "wheel_sha256",
        "python_version",
        "pip_dependencies",
        "build_dependencies",
        "build_tools",
        "wheel_install_args",
    }
    if set(data) != expected_fields:
        raise RuntimeError(
            "Modal wheel manifest fields differ: "
            f"missing={sorted(expected_fields - set(data))}, "
            f"unknown={sorted(set(data) - expected_fields)}"
        )
    expected_values: dict[str, object] = {
        "schema_version": _WHEEL_MANIFEST_SCHEMA_VERSION,
        "wheel_filename": wheel.name,
        "wheel_sha256": _sha256_file(wheel),
        "python_version": _PYTHON_VERSION,
        "pip_dependencies": list(_PIP_DEPENDENCIES),
        "build_dependencies": list(_BUILD_DEPENDENCIES),
        "build_tools": _BUILD_TOOLS,
        "wheel_install_args": ["--no-deps"],
    }
    for field, expected in expected_values.items():
        if data[field] != expected:
            raise RuntimeError(
                f"Modal wheel manifest {field} is {data[field]!r}, expected {expected!r}"
            )
    digest_lengths = {"head_commit": 40, "source_sha256": 64}
    for field, length in digest_lengths.items():
        value = data[field]
        if (
            type(value) is not str
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"Modal wheel manifest {field} is not a lowercase hex digest")
    if not remote:
        root = _REPO if repository is None else repository.resolve()
        head = _git_head(root)
        if data["head_commit"] != head:
            raise RuntimeError(
                f"Modal wheel was built at HEAD {data['head_commit']}, current HEAD is {head}"
            )
        if data["source_sha256"] != _source_digest(root):
            raise RuntimeError("Modal wheel source digest does not match the current source tree")
    return manifest


def configured_modal_wheel(root: Path | None = None) -> Path:
    base = _REPO if root is None else root.resolve()
    configured = os.environ.get("HELIOSTUNE_MODAL_WHEEL")
    if configured:
        wheel = Path(configured)
        if not wheel.is_file():
            raise RuntimeError(f"HELIOSTUNE_MODAL_WHEEL does not exist: {wheel}")
        remote = wheel.is_absolute() and wheel.parent == Path("/root")
        validate_wheel_manifest(wheel, repository=base, remote=remote)
        return wheel
    directory = base / _WHEEL_DIRECTORY
    wheels = tuple(sorted(directory.glob("heliostune-*.whl")))
    if len(wheels) != 1:
        raise RuntimeError(
            "run `uv run python scripts/build_modal_wheel.py` before Modal; "
            f"searched {directory}; found {[str(item) for item in wheels]}"
        )
    validate_wheel_manifest(wheels[0], repository=base)
    return wheels[0]


def build_image(wheel: Path) -> modal.Image:
    remote = wheel.is_absolute() and wheel.parent == Path("/root")
    manifest = validate_wheel_manifest(wheel, remote=remote)
    wheel_remote = remote_wheel_path(wheel)
    manifest_remote = remote_wheel_manifest_path(wheel)
    return (
        modal.Image.debian_slim(python_version=_PYTHON_VERSION)
        .pip_install(*_PIP_DEPENDENCIES)
        .add_local_file(wheel, remote_path=wheel_remote, copy=True)
        .add_local_file(manifest, remote_path=manifest_remote, copy=True)
        .run_commands(f"python -m pip install --no-deps {wheel_remote}")
        .env(
            {
                "HELIOSTUNE_MODAL_WHEEL": wheel_remote,
                "HELIOSTUNE_MODAL_WHEEL_MANIFEST": manifest_remote,
            }
        )
    )


app = modal.App("heliostune-precision-probe")
_MODAL_WHEEL = configured_modal_wheel()
image = build_image(_MODAL_WHEEL)


def probe_seed(*, purpose: str, bank: int, workload_key: str) -> int:
    """Return the deterministic 64-bit probe seed for one randomized decision.

    Deliberately outside ``heliostune.protocol.v3_seed``'s frozen purpose namespace: this
    is exploratory work and must not mint seeds inside a published protocol.
    """
    if not purpose or purpose != purpose.strip():
        raise ValueError("probe seed purpose must be a non-blank unpadded string")
    if type(bank) is not int or bank < 0:
        raise ValueError("probe seed bank must be a non-negative integer")
    if not workload_key or workload_key != workload_key.strip():
        raise ValueError("probe seed workload_key must be a non-blank unpadded string")
    preimage = "\0".join((_SEED_NAMESPACE, purpose, str(bank), workload_key)).encode()
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big")


def _remote_probe(
    gpu: str,
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    triton_config_keys: dict[str, str],
) -> dict[str, Any]:
    """Time the three arms per workload on one validated GPU, interleaved and seeded."""
    import random
    import time
    from functools import partial

    import torch

    from heliostune.configs import (
        DEFAULT_CONFIGS,
        DEFAULT_WORKLOADS,
        PARHELION_V3_CANDIDATE_CONFIGS,
        KernelConfig,
        Workload,
    )
    from heliostune.hardware import expectation_for_gpu, validate_hardware

    # _timed_do_bench is private but is the exact frozen timing helper; reproducing it
    # here instead would silently decouple the probe from the published protocol.
    from heliostune.kernel import _timed_do_bench, get_hardware_profile, matmul

    if not workload_keys:
        raise ValueError("the probe requires at least one workload key")
    if set(triton_config_keys) != set(workload_keys):
        raise ValueError("triton_config_keys must name exactly the requested workloads")

    configs_by_key = {
        config.key: config for config in (*DEFAULT_CONFIGS, *PARHELION_V3_CANDIDATE_CONFIGS)
    }
    try:
        selected_configs = {key: configs_by_key[triton_config_keys[key]] for key in workload_keys}
    except KeyError as exc:
        raise ValueError(f"remote manifest contains unknown config key {exc.args[0]!r}") from exc

    # This is deliberately the first CUDA operation. Identity is rejected before any
    # benchmark tensor allocation or paid timing work.
    profile = get_hardware_profile(gpu)
    validate_hardware(profile, expectation_for_gpu(gpu))

    # Reproduce heliostune.kernel.benchmark_measurements' legacy-bank ordering over the
    # full 96-workload manifest so every tensor is bit-identical to the committed archive,
    # then keep only the requested subset in that same order.
    canonical = list(DEFAULT_WORKLOADS)
    random.Random(bank).shuffle(canonical)
    requested = set(workload_keys)
    schedule = [(index, item) for index, item in enumerate(canonical) if item.key in requested]
    if len(schedule) != len(requested):
        known = {item.key for item in canonical}
        raise ValueError(f"unknown workload key(s): {sorted(requested - known)}")

    device = torch.device("cuda", torch.cuda.current_device())

    # One frame per workload: every tensor below is local to it, so the operand pair and
    # the FP32 reference are released before the next workload allocates. The three arm
    # helpers are closures over that frame rather than over the driving loop, which is
    # what keeps them from capturing a rebound loop variable.
    def probe_workload(
        workload_index: int,
        workload: Workload,
        config: KernelConfig,
    ) -> dict[str, Any]:
        tensor_seed = bank * 10_000 + workload_index
        torch.manual_seed(tensor_seed)
        a = torch.empty((workload.m, workload.k), device=device, dtype=torch.float16)
        b = torch.empty((workload.k, workload.n), device=device, dtype=torch.float16)
        a.uniform_(-1.0, 1.0)
        b.uniform_(-1.0, 1.0)

        ambient_tf32 = torch.backends.cuda.matmul.allow_tf32
        ambient_reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        try:
            reference = torch.mm(a, b, out_dtype=torch.float32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = ambient_tf32
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = ambient_reduced
        difference = torch.empty_like(reference)

        def max_abs_error(output: torch.Tensor) -> float:
            torch.sub(output, reference, out=difference)
            difference.abs_()
            return float(difference.max().item())

        def torch_arm(*, reduced: bool) -> dict[str, Any]:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
            try:
                error = max_abs_error(torch.matmul(a, b))
                quantiles, wall_ms = _timed_do_bench(
                    partial(torch.matmul, a, b),
                    warmup_ms=warmup_ms,
                    rep_ms=rep_ms,
                )
            finally:
                torch.backends.cuda.matmul.allow_tf32 = ambient_tf32
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = ambient_reduced
            p20, median, p80 = quantiles
            return {
                "latency_ms": median,
                "latency_p20_ms": p20,
                "latency_p80_ms": p80,
                "benchmark_wall_ms": wall_ms,
                "max_abs_error": error,
                "allow_fp16_reduced_precision_reduction": reduced,
                "allow_tf32": False,
            }

        def triton_arm() -> dict[str, Any]:
            torch.cuda.synchronize(device)
            compile_started = time.perf_counter()
            output = matmul(a, b, config)
            torch.cuda.synchronize(device)
            compile_ms = (time.perf_counter() - compile_started) * 1_000
            error = max_abs_error(output)
            del output
            quantiles, wall_ms = _timed_do_bench(
                lambda: matmul(a, b, config),
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
            )
            p20, median, p80 = quantiles
            return {
                "latency_ms": median,
                "latency_p20_ms": p20,
                "latency_p80_ms": p80,
                "benchmark_wall_ms": wall_ms,
                "max_abs_error": error,
                "compile_ms": compile_ms,
                "config": config.to_dict(),
                "config_key": config.key,
            }

        # Kill the ordering confound: the frozen collector times torch once, before a
        # shuffled config loop. Here all three arms are interleaved per workload in a
        # seeded random order that is recorded with the result.
        order_seed = probe_seed(purpose="arm-order", bank=bank, workload_key=workload.key)
        arm_order = list(_ARM_NAMES)
        random.Random(order_seed).shuffle(arm_order)
        arms: dict[str, dict[str, Any]] = {}
        for arm in arm_order:
            if arm == "torch_reduced":
                arms[arm] = torch_arm(reduced=True)
            elif arm == "torch_strict":
                arms[arm] = torch_arm(reduced=False)
            else:
                arms[arm] = triton_arm()

        return {
            "bank": bank,
            "workload": workload.to_dict(),
            "workload_key": workload.key,
            "workload_index": workload_index,
            "tensor_seed": tensor_seed,
            "arm_order_seed": order_seed,
            "arm_order": arm_order,
            "reference": {
                "dtype": "float32",
                "allow_tf32": False,
                "allow_fp16_reduced_precision_reduction": False,
            },
            "arms": arms,
        }

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for workload_index, workload in schedule:
            rows.append(probe_workload(workload_index, workload, selected_configs[workload.key]))

    return {
        "probe": _PROBE_NAME,
        "schema_version": _PROBE_SCHEMA_VERSION,
        "gpu": gpu,
        "bank": bank,
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
        "hardware": profile.to_dict(),
        "rows": rows,
    }


@app.function(image=image, gpu=_MODAL_SELECTOR, timeout=60 * 60)
def probe_h100(
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    triton_config_keys: dict[str, str],
) -> dict[str, Any]:
    return _remote_probe(_GPU, bank, warmup_ms, rep_ms, workload_keys, triton_config_keys)


@app.function(image=image, gpu=_MODAL_SELECTOR, timeout=_GATE_TIMEOUT_SECONDS)
def hopper_correctness_h100() -> dict[str, Any]:
    """Validate every Hopper candidate without executing a timing primitive."""
    from heliostune.artifacts import strict_json_dumps
    from heliostune.configs import HOPPER_GEMM_CONFIGS, SKINNY_GEMV_CONFIGS
    from heliostune.hopper_kernel import (
        assert_candidate_kernels_correct,
        validation_workloads,
    )
    from heliostune.kernel import get_hardware_profile

    workloads = validation_workloads()
    results = assert_candidate_kernels_correct(_GPU)
    serialized_configs = {
        "hopper_gemm": [config.to_dict() for config in HOPPER_GEMM_CONFIGS],
        "skinny_gemv": [config.to_dict() for config in SKINNY_GEMV_CONFIGS],
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    validation_results = [result.to_dict() for result in results]
    for result in validation_results:
        key = (str(result["kernel"]), str(result["config_key"]))
        grouped.setdefault(key, []).append(result)
    candidate_summaries = []
    for (kernel, config_key), checks in sorted(grouped.items()):
        errors = [str(check["error"]) for check in checks if check["error"] is not None]
        candidate_summaries.append(
            {
                "kernel": kernel,
                "config_key": config_key,
                "check_count": len(checks),
                "correct": all(check["correct"] is True for check in checks),
                "max_abs_error": max(cast(float, check["max_abs_error"]) for check in checks),
                "errors": errors,
            }
        )
    return {
        "schema_version": _GATE_SCHEMA_VERSION,
        "gate": _GATE_NAME,
        "study_status": _GATE_STATUS,
        "analysis_status": _GATE_STATUS,
        "gpu_selector": _MODAL_SELECTOR,
        "gpu": _GPU,
        "hardware": get_hardware_profile(_GPU).to_dict(),
        "config_counts": {
            "hopper_gemm": len(HOPPER_GEMM_CONFIGS),
            "skinny_gemv": len(SKINNY_GEMV_CONFIGS),
            "total": len(HOPPER_GEMM_CONFIGS) + len(SKINNY_GEMV_CONFIGS),
        },
        "config_manifest_sha256": _sha256_payload(
            strict_json_dumps(serialized_configs, compact=True)
        ),
        "validation_workload_count": len(workloads),
        "validation_check_count": len(validation_results),
        "candidate_summaries": candidate_summaries,
        "validation_results": validation_results,
    }


@app.function(image=image, gpu=_MODAL_SELECTOR, timeout=_BENCHMARK_TIMEOUT_SECONDS)
def hopper_benchmark_h100() -> dict[str, object]:
    """Benchmark the frozen one-bank Hopper candidate cross-product."""
    from heliostune.configs import DEFAULT_WORKLOADS
    from heliostune.hopper_benchmark import benchmark_hopper_candidates

    return benchmark_hopper_candidates(
        gpu=_GPU,
        bank=_BENCHMARK_BANK,
        warmup_ms=_BENCHMARK_WARMUP_MS,
        rep_ms=_BENCHMARK_REP_MS,
        workload_keys=tuple(workload.key for workload in DEFAULT_WORKLOADS),
    )


def _sha256_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def _gate_config_manifest() -> dict[str, list[dict[str, object]]]:
    from heliostune.configs import HOPPER_GEMM_CONFIGS, SKINNY_GEMV_CONFIGS

    return {
        "hopper_gemm": [
            cast(dict[str, object], config.to_dict()) for config in HOPPER_GEMM_CONFIGS
        ],
        "skinny_gemv": [
            cast(dict[str, object], config.to_dict()) for config in SKINNY_GEMV_CONFIGS
        ],
    }


def _gate_config_manifest_sha256(configs: dict[str, list[dict[str, object]]]) -> str:
    from heliostune.artifacts import strict_json_dumps

    return _sha256_payload(strict_json_dumps(configs, compact=True))


def _expected_gate_checks() -> set[tuple[str, str, str]]:
    from heliostune.configs import HOPPER_GEMM_CONFIGS, SKINNY_GEMV_CONFIGS
    from heliostune.hopper_spec import SKINNY_M_LIMIT, validation_workloads

    expected: set[tuple[str, str, str]] = set()
    for workload in validation_workloads():
        expected.update(
            ("hopper_matmul", config.key, workload.key) for config in HOPPER_GEMM_CONFIGS
        )
        if workload.m <= SKINNY_M_LIMIT:
            expected.update(
                ("skinny_gemv", config.key, workload.key) for config in SKINNY_GEMV_CONFIGS
            )
    return expected


def _validated_gate_payload(
    payload: object,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Validate the exact correctness result against local candidate manifests."""
    from heliostune.configs import HOPPER_GEMM_CONFIGS, SKINNY_GEMV_CONFIGS
    from heliostune.hardware import expectation_for_gpu, validate_hardware
    from heliostune.hopper_spec import validation_workloads
    from heliostune.schema import HardwareProfile
    from heliostune.validation import (
        exact_bool,
        exact_fields,
        exact_int,
        finite_float,
        nonblank_string,
    )

    data = exact_fields(
        payload,
        required=(
            "schema_version",
            "gate",
            "study_status",
            "analysis_status",
            "gpu_selector",
            "gpu",
            "hardware",
            "config_counts",
            "config_manifest_sha256",
            "validation_workload_count",
            "validation_check_count",
            "candidate_summaries",
            "validation_results",
        ),
        context="Hopper correctness payload",
    )
    if exact_int(data["schema_version"], context="gate schema_version", minimum=1) != (
        _GATE_SCHEMA_VERSION
    ):
        raise ValueError("Hopper correctness payload reports the wrong schema version")
    expected_scalars = {
        "gate": _GATE_NAME,
        "study_status": _GATE_STATUS,
        "analysis_status": _GATE_STATUS,
        "gpu_selector": _MODAL_SELECTOR,
        "gpu": _GPU,
    }
    for field, expected in expected_scalars.items():
        if data[field] != expected:
            raise ValueError(f"Hopper correctness payload reports the wrong {field}")

    hardware = HardwareProfile.from_dict(data["hardware"])
    if hardware.gpu != _GPU:
        raise ValueError("Hopper correctness payload hardware is not H100")
    validate_hardware(hardware, expectation_for_gpu(_GPU))

    expected_counts = {
        "hopper_gemm": len(HOPPER_GEMM_CONFIGS),
        "skinny_gemv": len(SKINNY_GEMV_CONFIGS),
        "total": len(HOPPER_GEMM_CONFIGS) + len(SKINNY_GEMV_CONFIGS),
    }
    counts = exact_fields(
        data["config_counts"],
        required=("hopper_gemm", "skinny_gemv", "total"),
        context="Hopper correctness config_counts",
    )
    for count_field, expected_count in expected_counts.items():
        if (
            exact_int(
                counts[count_field],
                context=f"config_counts {count_field}",
                minimum=0,
            )
            != expected_count
        ):
            raise ValueError(f"Hopper correctness payload reports the wrong {count_field} count")

    configs = _gate_config_manifest()
    manifest_sha256 = nonblank_string(
        data["config_manifest_sha256"], context="gate config_manifest_sha256"
    )
    if manifest_sha256 != _gate_config_manifest_sha256(configs):
        raise ValueError("Hopper correctness payload config manifest differs from local source")

    expected_workload_count = len(validation_workloads())
    if (
        exact_int(
            data["validation_workload_count"],
            context="validation_workload_count",
            minimum=1,
        )
        != expected_workload_count
    ):
        raise ValueError("Hopper correctness payload reports the wrong validation workload count")

    expected_checks = _expected_gate_checks()
    if exact_int(
        data["validation_check_count"],
        context="validation_check_count",
        minimum=1,
    ) != len(expected_checks):
        raise ValueError("Hopper correctness payload reports the wrong validation check count")

    raw_results = data["validation_results"]
    if type(raw_results) is not list:
        raise ValueError("Hopper correctness validation_results must be a list")
    seen_checks: set[tuple[str, str, str]] = set()
    results: list[dict[str, object]] = []
    grouped_errors: dict[tuple[str, str], list[float]] = {}
    for raw_result in raw_results:
        result = exact_fields(
            raw_result,
            required=(
                "kernel",
                "config_key",
                "workload_key",
                "correct",
                "max_abs_error",
                "error",
            ),
            context="Hopper correctness validation result",
        )
        kernel = nonblank_string(result["kernel"], context="validation result kernel")
        config_key = nonblank_string(result["config_key"], context="validation result config_key")
        workload_key = nonblank_string(
            result["workload_key"], context="validation result workload_key"
        )
        key = (kernel, config_key, workload_key)
        if key not in expected_checks:
            raise ValueError(f"Hopper correctness payload contains unexpected check {key}")
        if key in seen_checks:
            raise ValueError(f"Hopper correctness payload duplicates check {key}")
        seen_checks.add(key)
        if not exact_bool(result["correct"], context=f"{key} correct"):
            raise ValueError(f"Hopper correctness payload contains a failed check {key}")
        if result["error"] is not None:
            raise ValueError(f"Hopper correctness payload contains an error for {key}")
        maximum_error = finite_float(
            result["max_abs_error"], context=f"{key} max_abs_error", minimum=0
        )
        grouped_errors.setdefault((kernel, config_key), []).append(maximum_error)
        result["max_abs_error"] = maximum_error
        results.append(result)
    if seen_checks != expected_checks:
        raise ValueError(
            "Hopper correctness payload check set is incomplete: "
            f"missing={len(expected_checks - seen_checks)}"
        )

    raw_summaries = data["candidate_summaries"]
    if type(raw_summaries) is not list:
        raise ValueError("Hopper correctness candidate_summaries must be a list")
    seen_candidates: set[tuple[str, str]] = set()
    summaries: list[dict[str, object]] = []
    for raw_summary in raw_summaries:
        summary = exact_fields(
            raw_summary,
            required=(
                "kernel",
                "config_key",
                "check_count",
                "correct",
                "max_abs_error",
                "errors",
            ),
            context="Hopper correctness candidate summary",
        )
        candidate = (
            nonblank_string(summary["kernel"], context="candidate summary kernel"),
            nonblank_string(summary["config_key"], context="candidate summary config_key"),
        )
        expected_errors = grouped_errors.get(candidate)
        if expected_errors is None:
            raise ValueError(f"Hopper correctness payload contains unknown candidate {candidate}")
        if candidate in seen_candidates:
            raise ValueError(f"Hopper correctness payload duplicates candidate {candidate}")
        seen_candidates.add(candidate)
        if exact_int(summary["check_count"], context=f"{candidate} check_count", minimum=1) != len(
            expected_errors
        ):
            raise ValueError(f"Hopper correctness payload has wrong check count for {candidate}")
        if not exact_bool(summary["correct"], context=f"{candidate} correct"):
            raise ValueError(f"Hopper correctness payload marks candidate {candidate} incorrect")
        if summary["errors"] != []:
            raise ValueError(f"Hopper correctness payload reports errors for {candidate}")
        maximum_error = finite_float(
            summary["max_abs_error"], context=f"{candidate} max_abs_error", minimum=0
        )
        if maximum_error != max(expected_errors):
            raise ValueError(f"Hopper correctness payload has wrong max error for {candidate}")
        summary["max_abs_error"] = maximum_error
        summaries.append(summary)
    if seen_candidates != set(grouped_errors):
        raise ValueError("Hopper correctness payload candidate summaries are incomplete")

    results.sort(
        key=lambda row: (str(row["workload_key"]), str(row["kernel"]), str(row["config_key"]))
    )
    summaries.sort(key=lambda row: (str(row["kernel"]), str(row["config_key"])))
    data["hardware"] = hardware.to_dict()
    return data, summaries, results


def _parse_banks(value: str) -> tuple[int, ...]:
    items = tuple(value.split(","))
    if not items or any(not item or item != item.strip() for item in items):
        raise ValueError("banks must be a non-empty comma-separated list without whitespace")
    if any(not item.isascii() or not item.isdecimal() for item in items):
        raise ValueError("banks must contain only non-negative decimal integers")
    banks = tuple(int(item) for item in items)
    if len(set(banks)) != len(banks):
        raise ValueError("banks must not contain duplicates")
    if banks != _FROZEN_BANK_VALUES:
        raise ValueError(f"banks must be exactly {_FROZEN_BANKS}")
    return banks


def _git_identity() -> tuple[str, str]:
    repository = Path(__file__).resolve().parent
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("the precision probe requires a clean Git HEAD")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return head, _sha256_payload(head)


def _resolve_wheel(path: str) -> Path:
    wheel = Path(path) if path else _MODAL_WHEEL
    if not wheel.is_file():
        raise ValueError(f"Modal wheel does not exist: {wheel}")
    if wheel.resolve() != _MODAL_WHEEL.resolve():
        raise ValueError(
            "--wheel must match the wheel baked into the image; set "
            "HELIOSTUNE_MODAL_WHEEL before `modal run` to choose another wheel"
        )
    return wheel


def archive_baseline(archive: Path, gpu: str) -> dict[str, Any]:
    """Select on frozen bank 1 and score the selected configuration on bank 2."""
    import math

    from heliostune.artifacts import read_measurements
    from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS

    selection_bank = 1
    evaluation_bank = 2
    expected_configs = {config.key for config in DEFAULT_CONFIGS}
    expected_workloads = {workload.key for workload in DEFAULT_WORKLOADS}
    cells: dict[tuple[str, str, int], Any] = {}
    hardware: dict[str, object] | None = None
    for row in read_measurements(archive):
        if row.hardware.gpu != gpu or row.bank not in {selection_bank, evaluation_bank}:
            continue
        if row.workload.key not in expected_workloads or row.config.key not in expected_configs:
            raise ValueError(
                f"archive contains unexpected {gpu} cell "
                f"{row.workload.key}/{row.config.key}/bank-{row.bank}"
            )
        key = (row.workload.key, row.config.key, row.bank)
        if key in cells:
            raise ValueError(
                f"archive contains duplicate cell "
                f"{row.workload.key}/{row.config.key}/bank-{row.bank}"
            )
        if (
            not row.usable
            or row.latency_ms is None
            or not math.isfinite(row.latency_ms)
            or row.latency_ms <= 0.0
        ):
            raise ValueError(
                f"archive cell {row.workload.key}/{row.config.key}/bank-{row.bank} is unusable"
            )
        if (
            row.torch_latency_ms is None
            or not math.isfinite(row.torch_latency_ms)
            or row.torch_latency_ms <= 0.0
        ):
            raise ValueError(
                f"archive torch value {row.workload.key}/{row.config.key}/bank-{row.bank} "
                "must be finite and positive"
            )
        profile = row.hardware.to_dict()
        if hardware is None:
            hardware = profile
        elif hardware != profile:
            raise ValueError(f"archive {gpu} rows disagree on hardware identity")
        cells[key] = row

    expected_cells = {
        (workload, config, bank)
        for workload in expected_workloads
        for config in expected_configs
        for bank in (selection_bank, evaluation_bank)
    }
    missing = sorted(expected_cells - set(cells))
    if missing:
        preview = ", ".join(f"{w}/{c}/bank-{b}" for w, c, b in missing[:3])
        raise ValueError(
            f"archive {archive} lacks {len(missing)} required {gpu} cells"
            f"{': ' + preview if preview else ''}"
        )

    workloads: dict[str, Any] = {}
    ratios: list[float] = []
    for workload_key in sorted(expected_workloads):
        best_key = min(
            expected_configs,
            key=lambda config_key: (
                cells[(workload_key, config_key, selection_bank)].latency_ms,
                config_key,
            ),
        )
        selected = cells[(workload_key, best_key, evaluation_bank)]
        best_latency = selected.latency_ms
        # The published comparator is evaluation-only and is read from the canonical
        # first manifest cell, exactly as the frozen replay endpoint does.
        torch_latency = cells[
            (workload_key, DEFAULT_CONFIGS[0].key, evaluation_bank)
        ].torch_latency_ms
        ratio = torch_latency / best_latency
        ratios.append(ratio)
        workloads[workload_key] = {
            "best_triton_config_key": best_key,
            "selection_bank_latency_ms": cells[(workload_key, best_key, selection_bank)].latency_ms,
            "best_triton_latency_ms": best_latency,
            "torch_latency_ms": torch_latency,
            "torch_over_best_triton": ratio,
            "configs_considered": len(expected_configs),
        }
    return {
        "path": str(archive),
        "gpu": gpu,
        "banks": [selection_bank, evaluation_bank],
        "selection_bank": selection_bank,
        "evaluation_bank": evaluation_bank,
        "aggregation": "config selected on bank 1 and both latencies scored on bank 2",
        "median_torch_over_best_triton": statistics.median(ratios),
        "triton_wins": sum(1 for ratio in ratios if ratio > 1.0),
        "hardware": hardware,
        "published_endpoint_torch_over_best_triton": _FROZEN_ARCHIVE_ENDPOINT,
        "workloads": workloads,
    }


def _probe_protocol(
    *,
    banks: tuple[int, ...],
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "probe": _PROBE_NAME,
        "schema_version": _PROBE_SCHEMA_VERSION,
        "role": "post-hoc exploratory; not a confirmatory Parhelion endpoint",
        "gpu": _GPU,
        "modal_selector": _MODAL_SELECTOR,
        "banks": list(banks),
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
        "quantiles": list(_FROZEN_QUANTILES),
        "statistic": "median of triton.testing.do_bench quantiles [0.2, 0.5, 0.8]",
        "arms": list(_ARM_NAMES),
        "arm_order": "per-workload seeded random permutation of the three arms",
        "arm_order_seed": f"sha256('{_SEED_NAMESPACE}\\0arm-order\\0<bank>\\0<workload_key>')[:8]",
        "tensor_seed": "bank * 10000 + legacy-bank shuffled workload index",
        "tensor_seed_protocol": "legacy-bank, identical to the committed v2 archive",
        "reference": "torch.mm(out_dtype=float32) with TF32 and FP16 reduction disabled",
        "workload_count": len(workload_keys),
        "retry_policy": "none; a failed remote call is journaled and re-raised",
    }


def _validate_timing_fields(
    arm: dict[str, object],
    *,
    context: str,
) -> None:
    from heliostune.validation import finite_float

    p20 = finite_float(
        arm["latency_p20_ms"], context=f"{context} latency_p20_ms", strictly_positive=True
    )
    median = finite_float(
        arm["latency_ms"], context=f"{context} latency_ms", strictly_positive=True
    )
    p80 = finite_float(
        arm["latency_p80_ms"], context=f"{context} latency_p80_ms", strictly_positive=True
    )
    if not p20 <= median <= p80:
        raise ValueError(f"{context} timing quantiles are not ordered")
    finite_float(
        arm["benchmark_wall_ms"],
        context=f"{context} benchmark_wall_ms",
        strictly_positive=True,
    )
    finite_float(arm["max_abs_error"], context=f"{context} max_abs_error", minimum=0)


def _validated_payload(
    payload: object,
    *,
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    triton_config_keys: dict[str, str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return one exact remote payload and its fully validated unique rows."""
    import random

    from heliostune.configs import DEFAULT_WORKLOADS, KernelConfig, Workload
    from heliostune.schema import HardwareProfile
    from heliostune.validation import (
        exact_bool,
        exact_fields,
        exact_int,
        finite_float,
        nonblank_string,
    )

    data = exact_fields(
        payload,
        required=(
            "probe",
            "schema_version",
            "gpu",
            "bank",
            "warmup_ms",
            "rep_ms",
            "hardware",
            "rows",
        ),
        context=f"probe payload for bank {bank}",
    )
    if data["probe"] != _PROBE_NAME:
        raise ValueError(f"probe payload for bank {bank} reports the wrong probe")
    if exact_int(data["schema_version"], context="probe payload schema_version", minimum=1) != (
        _PROBE_SCHEMA_VERSION
    ):
        raise ValueError(f"probe payload for bank {bank} reports the wrong schema version")
    if data["gpu"] != _GPU:
        raise ValueError(f"probe payload for bank {bank} reports gpu {data['gpu']!r}, not H100")
    if exact_int(data["bank"], context="probe payload bank", minimum=0) != bank:
        raise ValueError(f"probe payload reports the wrong bank for bank {bank}")
    if exact_int(data["warmup_ms"], context="probe payload warmup_ms", minimum=0) != warmup_ms:
        raise ValueError(f"probe payload for bank {bank} reports the wrong warmup")
    if exact_int(data["rep_ms"], context="probe payload rep_ms", minimum=1) != rep_ms:
        raise ValueError(f"probe payload for bank {bank} reports the wrong repetition")
    hardware = HardwareProfile.from_dict(data["hardware"])
    if hardware.gpu != _GPU:
        raise ValueError(f"probe payload hardware for bank {bank} is not H100")

    raw_rows = data["rows"]
    if type(raw_rows) is not list or not raw_rows:
        raise ValueError(f"probe payload for bank {bank} carries no rows")
    expected_workloads = {item.key: item for item in DEFAULT_WORKLOADS}
    if set(workload_keys) != set(triton_config_keys):
        raise ValueError("selected Triton configs do not match the requested workloads")
    canonical = list(DEFAULT_WORKLOADS)
    random.Random(bank).shuffle(canonical)
    expected_indices = {workload.key: index for index, workload in enumerate(canonical)}
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        row = exact_fields(
            raw_row,
            required=(
                "bank",
                "workload",
                "workload_key",
                "workload_index",
                "tensor_seed",
                "arm_order_seed",
                "arm_order",
                "reference",
                "arms",
            ),
            context=f"probe row for bank {bank}",
        )
        if exact_int(row["bank"], context="probe row bank", minimum=0) != bank:
            raise ValueError(f"probe row reports the wrong bank for bank {bank}")
        workload_key = nonblank_string(row["workload_key"], context="probe row workload_key")
        if workload_key not in workload_keys:
            raise ValueError(f"probe row contains unrequested workload {workload_key}")
        if workload_key in seen:
            raise ValueError(f"probe payload bank {bank} duplicates workload {workload_key}")
        seen.add(workload_key)
        workload = Workload.from_dict(row["workload"])
        if workload != expected_workloads.get(workload_key) or workload.key != workload_key:
            raise ValueError(f"probe row workload body does not match {workload_key}")
        workload_index = exact_int(
            row["workload_index"], context=f"{workload_key} workload_index", minimum=0
        )
        if workload_index != expected_indices[workload_key]:
            raise ValueError(f"probe row {workload_key} reports the wrong workload index")
        if (
            exact_int(row["tensor_seed"], context=f"{workload_key} tensor_seed", minimum=0)
            != bank * 10_000 + workload_index
        ):
            raise ValueError(f"probe row {workload_key} reports the wrong tensor seed")
        expected_order_seed = probe_seed(purpose="arm-order", bank=bank, workload_key=workload_key)
        if (
            exact_int(row["arm_order_seed"], context=f"{workload_key} arm_order_seed", minimum=0)
            != expected_order_seed
        ):
            raise ValueError(f"probe row {workload_key} reports the wrong arm-order seed")
        arm_order = row["arm_order"]
        if (
            type(arm_order) is not list
            or len(arm_order) != len(_ARM_NAMES)
            or any(type(name) is not str for name in arm_order)
            or set(arm_order) != set(_ARM_NAMES)
        ):
            raise ValueError(f"probe row {workload_key} arm_order is not a permutation")
        reference = exact_fields(
            row["reference"],
            required=("dtype", "allow_tf32", "allow_fp16_reduced_precision_reduction"),
            context=f"{workload_key} reference",
        )
        if reference != {
            "dtype": "float32",
            "allow_tf32": False,
            "allow_fp16_reduced_precision_reduction": False,
        }:
            raise ValueError(f"probe row {workload_key} reports the wrong reference")
        arms = exact_fields(row["arms"], required=_ARM_NAMES, context=f"{workload_key} arms")
        for arm_name, reduced in (("torch_reduced", True), ("torch_strict", False)):
            arm = exact_fields(
                arms[arm_name],
                required=(
                    "latency_ms",
                    "latency_p20_ms",
                    "latency_p80_ms",
                    "benchmark_wall_ms",
                    "max_abs_error",
                    "allow_fp16_reduced_precision_reduction",
                    "allow_tf32",
                ),
                context=f"{workload_key}/{arm_name}",
            )
            _validate_timing_fields(arm, context=f"{workload_key}/{arm_name}")
            if exact_bool(
                arm["allow_fp16_reduced_precision_reduction"],
                context=f"{workload_key}/{arm_name} reduced flag",
            ) is not reduced or exact_bool(
                arm["allow_tf32"], context=f"{workload_key}/{arm_name} TF32 flag"
            ):
                raise ValueError(f"probe row {workload_key} has wrong flags for {arm_name}")
        triton = exact_fields(
            arms["triton"],
            required=(
                "latency_ms",
                "latency_p20_ms",
                "latency_p80_ms",
                "benchmark_wall_ms",
                "max_abs_error",
                "compile_ms",
                "config",
                "config_key",
            ),
            context=f"{workload_key}/triton",
        )
        _validate_timing_fields(triton, context=f"{workload_key}/triton")

        finite_float(triton["compile_ms"], context=f"{workload_key}/triton compile_ms", minimum=0)
        config_key = nonblank_string(
            triton["config_key"], context=f"{workload_key}/triton config_key"
        )
        config = KernelConfig.from_dict(triton["config"])
        if config.key != config_key or config_key != triton_config_keys[workload_key]:
            raise ValueError(f"probe row {workload_key} reports the wrong Triton config")
        rows.append(row)
    if seen != set(workload_keys):
        raise ValueError(
            f"probe payload bank {bank} workload set differs: "
            f"missing={sorted(set(workload_keys) - seen)}, "
            f"unknown={sorted(seen - set(workload_keys))}"
        )
    return data, rows


def _validate_cross_product(
    rows: list[dict[str, object]],
    *,
    banks: tuple[int, ...],
    workload_keys: tuple[str, ...],
) -> None:
    pairs = {(cast(int, row["bank"]), cast(str, row["workload_key"])) for row in rows}
    expected = {(bank, workload) for bank in banks for workload in workload_keys}
    if len(rows) != len(expected) or pairs != expected:
        raise ValueError("final probe rows are not exactly banks × requested workloads")


def _resolved_output(value: str, *, repository: Path | None = None) -> Path:
    """Resolve a relative output while protecting the repository benchmarks tree."""
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("output must be a relative path")
    if ".." in candidate.parts:
        raise ValueError("output must not contain '..'")
    destination = candidate.resolve()
    root = Path(__file__).resolve().parent if repository is None else repository.resolve()
    benchmarks = (root / "benchmarks").resolve()
    if destination == benchmarks or destination.is_relative_to(benchmarks):
        raise ValueError("exploratory probe output must never be written under benchmarks/")
    return destination


def _resolved_gate_output(value: str, *, repository: Path | None = None) -> Path:
    """Require a repository-relative artifacts/ destination, resolving symlinks."""
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("Hopper gate output must be a relative path under artifacts/")
    if ".." in candidate.parts:
        raise ValueError("Hopper gate output must not contain '..'")
    if not candidate.parts or candidate.parts[0] != "artifacts":
        raise ValueError("Hopper gate output must be under artifacts/")
    root = _REPO if repository is None else repository.resolve()
    artifacts = (root / "artifacts").resolve()
    benchmarks = (root / "benchmarks").resolve()
    destination = (root / candidate).resolve()
    if destination == artifacts or not destination.is_relative_to(artifacts):
        raise ValueError("Hopper gate output must resolve under artifacts/")
    if destination == benchmarks or destination.is_relative_to(benchmarks):
        raise ValueError("Hopper gate output must never resolve under benchmarks/")
    return destination


def _validated_wheel_provenance(
    wheel: Path,
    *,
    head_commit: str,
    repository: Path | None = None,
) -> dict[str, str]:
    root = _REPO if repository is None else repository.resolve()
    manifest_path_value = validate_wheel_manifest(wheel, repository=root)
    manifest = _manifest_object(manifest_path_value)
    if manifest["head_commit"] != head_commit:
        raise RuntimeError("Modal wheel manifest is stale for the clean gate HEAD")
    return {
        "manifest_path": str(manifest_path_value),
        "manifest_sha256": _sha256_file(manifest_path_value),
        "head_commit": head_commit,
        "source_sha256": cast(str, manifest["source_sha256"]),
        "wheel_sha256": cast(str, manifest["wheel_sha256"]),
    }


def _validated_correctness_gate(
    gate_path: Path,
    *,
    wheel_provenance: dict[str, str],
    head_sha256: str,
) -> dict[str, str]:
    """Validate and bind the completed correctness gate before any paid spawn."""
    from heliostune.artifacts import strict_json_dumps
    from heliostune.collection import AttemptJournal, attempt_journal_path, manifest_path
    from heliostune.configs import HOPPER_GEMM_CONFIGS, SKINNY_GEMV_CONFIGS
    from heliostune.validation import exact_bool, exact_fields, exact_int, nonblank_string

    if not gate_path.is_file():
        raise ValueError(f"Hopper correctness gate does not exist: {gate_path}")
    sidecar_path = manifest_path(gate_path)
    if not sidecar_path.is_file():
        raise ValueError(f"Hopper correctness gate manifest does not exist: {sidecar_path}")

    artifact = exact_fields(
        _manifest_object(gate_path),
        required=(
            "schema_version",
            "gate",
            "study_status",
            "analysis_status",
            "verified",
            "correctness_only",
            "performance_validated",
            "protocol",
            "gpu",
            "gpu_selector",
            "hardware",
            "config_counts",
            "config_manifest",
            "validation_workload_count",
            "validation_check_count",
            "candidate_summaries",
            "validation_results",
            "remote_call",
        ),
        context="Hopper correctness artifact",
    )
    if exact_int(artifact["schema_version"], context="gate schema_version", minimum=1) != 1:
        raise ValueError("Hopper correctness gate has the wrong schema version")
    if artifact["gate"] != _GATE_NAME:
        raise ValueError("Hopper correctness gate has the wrong gate name")
    if artifact["study_status"] != _GATE_STATUS or artifact["analysis_status"] != _GATE_STATUS:
        raise ValueError("Hopper correctness gate has the wrong analysis status")
    if not exact_bool(artifact["verified"], context="gate verified"):
        raise ValueError("Hopper correctness gate is not verified")
    if not exact_bool(artifact["correctness_only"], context="gate correctness_only"):
        raise ValueError("Hopper correctness gate is not correctness-only")
    if exact_bool(artifact["performance_validated"], context="gate performance_validated"):
        raise ValueError("Hopper correctness gate must not claim performance validation")
    if artifact["gpu"] != _GPU or artifact["gpu_selector"] != _MODAL_SELECTOR:
        raise ValueError("Hopper correctness gate is not bound to H100!")

    config_manifest = exact_fields(
        artifact["config_manifest"],
        required=("sha256", "hopper_gemm", "skinny_gemv"),
        context="Hopper correctness config manifest",
    )
    expected_configs = _gate_config_manifest()
    expected_config_sha256 = _gate_config_manifest_sha256(expected_configs)
    if config_manifest["sha256"] != expected_config_sha256:
        raise ValueError("Hopper correctness gate config digest differs from local source")
    if config_manifest["hopper_gemm"] != expected_configs["hopper_gemm"]:
        raise ValueError("Hopper correctness gate Hopper config manifest differs from local source")
    if config_manifest["skinny_gemv"] != expected_configs["skinny_gemv"]:
        raise ValueError("Hopper correctness gate skinny config manifest differs from local source")

    protocol = exact_fields(
        artifact["protocol"],
        required=(
            "schema_version",
            "gate",
            "study_status",
            "analysis_status",
            "correctness_only",
            "performance_validated",
            "timing_operations",
            "gpu",
            "gpu_selector",
            "remote_function",
            "remote_call_count",
            "timeout_seconds",
            "operator_sequence",
        ),
        context="Hopper correctness protocol",
    )
    expected_gate_protocol = {
        "schema_version": _GATE_SCHEMA_VERSION,
        "gate": _GATE_NAME,
        "study_status": _GATE_STATUS,
        "analysis_status": _GATE_STATUS,
        "correctness_only": True,
        "performance_validated": False,
        "timing_operations": 0,
        "gpu": _GPU,
        "gpu_selector": _MODAL_SELECTOR,
        "remote_function": "hopper_correctness_h100",
        "remote_call_count": 1,
        "timeout_seconds": _GATE_TIMEOUT_SECONDS,
        "operator_sequence": [
            "precision_probe",
            "hopper_correctness_gate",
            "full_collection",
        ],
    }
    if protocol != expected_gate_protocol:
        raise ValueError("Hopper correctness gate protocol differs from the frozen gate")

    remote_payload = {
        "schema_version": artifact["schema_version"],
        "gate": artifact["gate"],
        "study_status": artifact["study_status"],
        "analysis_status": artifact["analysis_status"],
        "gpu_selector": artifact["gpu_selector"],
        "gpu": artifact["gpu"],
        "hardware": artifact["hardware"],
        "config_counts": artifact["config_counts"],
        "config_manifest_sha256": config_manifest["sha256"],
        "validation_workload_count": artifact["validation_workload_count"],
        "validation_check_count": artifact["validation_check_count"],
        "candidate_summaries": artifact["candidate_summaries"],
        "validation_results": artifact["validation_results"],
    }
    _data, summaries, results = _validated_gate_payload(remote_payload)
    if len(summaries) != len(HOPPER_GEMM_CONFIGS) + len(SKINNY_GEMV_CONFIGS):
        raise ValueError("Hopper correctness gate must contain exactly 71 candidate summaries")
    if len(results) != 633:
        raise ValueError("Hopper correctness gate must contain exactly 633 validation checks")

    remote_call = exact_fields(
        artifact["remote_call"],
        required=("call_id", "payload_sha256"),
        context="Hopper correctness remote call",
    )
    nonblank_string(remote_call["call_id"], context="Hopper correctness call ID")
    nonblank_string(remote_call["payload_sha256"], context="Hopper correctness payload digest")

    artifact_sha256 = _sha256_file(gate_path)
    sidecar = exact_fields(
        _manifest_object(sidecar_path),
        required=(
            "schema_version",
            "gate",
            "verified",
            "request",
            "binding",
            "protocol",
            "data",
            "attempt_journal",
            "inputs",
            "facts",
            "analysis_runtime",
        ),
        context="Hopper correctness manifest",
    )
    if exact_int(sidecar["schema_version"], context="gate manifest schema_version", minimum=1) != 1:
        raise ValueError("Hopper correctness manifest has the wrong schema version")
    if sidecar["gate"] != _GATE_NAME or not exact_bool(
        sidecar["verified"], context="gate manifest verified"
    ):
        raise ValueError("Hopper correctness manifest is not verified")
    if sidecar["protocol"] != artifact["protocol"]:
        raise ValueError("Hopper correctness manifest protocol differs from the artifact")
    request = exact_fields(
        sidecar["request"],
        required=(
            "schema_version",
            "gate",
            "gpu",
            "gpu_selector",
            "remote_call_count",
            "resume",
            "retry",
        ),
        context="Hopper correctness manifest request",
    )
    expected_gate_request: dict[str, object] = {
        "schema_version": _GATE_SCHEMA_VERSION,
        "gate": _GATE_NAME,
        "gpu": _GPU,
        "gpu_selector": _MODAL_SELECTOR,
        "remote_call_count": 1,
        "resume": False,
        "retry": False,
    }
    if (
        exact_int(
            request["schema_version"],
            context="gate manifest request schema_version",
            minimum=1,
        )
        != _GATE_SCHEMA_VERSION
        or request["gate"] != _GATE_NAME
        or request["gpu"] != _GPU
        or request["gpu_selector"] != _MODAL_SELECTOR
        or exact_int(
            request["remote_call_count"],
            context="gate manifest request remote_call_count",
            minimum=1,
        )
        != 1
        or exact_bool(request["resume"], context="gate manifest request resume")
        or exact_bool(request["retry"], context="gate manifest request retry")
    ):
        raise ValueError("Hopper correctness manifest request differs from the frozen gate")

    binding = exact_fields(
        sidecar["binding"],
        required=(
            "protocol_sha256",
            "config_manifest_sha256",
            "wheel_sha256",
            "head_sha256",
        ),
        context="Hopper correctness manifest binding",
    )
    if binding["config_manifest_sha256"] != expected_config_sha256:
        raise ValueError("Hopper correctness manifest is stale for the config manifest")
    if binding["wheel_sha256"] != wheel_provenance["wheel_sha256"]:
        raise ValueError("Hopper correctness manifest is stale for the wheel")
    if binding["head_sha256"] != head_sha256:
        raise ValueError("Hopper correctness manifest is stale for the current HEAD")
    if binding["protocol_sha256"] != _sha256_payload(
        strict_json_dumps(expected_gate_protocol, compact=True)
    ):
        raise ValueError("Hopper correctness manifest protocol digest is invalid")

    data = exact_fields(
        sidecar["data"],
        required=("path", "sha256", "candidate_summaries", "validation_results"),
        context="Hopper correctness manifest data",
    )
    if data["sha256"] != artifact_sha256:
        raise ValueError("Hopper correctness artifact digest does not match its manifest")
    if (
        Path(nonblank_string(data["path"], context="gate data path")).resolve()
        != gate_path.resolve()
    ):
        raise ValueError("Hopper correctness manifest points at a different artifact")
    if exact_int(data["candidate_summaries"], context="gate summary count", minimum=0) != 71:
        raise ValueError("Hopper correctness manifest must report 71 candidate summaries")
    if exact_int(data["validation_results"], context="gate result count", minimum=0) != 633:
        raise ValueError("Hopper correctness manifest must report 633 validation checks")
    attempt = exact_fields(
        sidecar["attempt_journal"],
        required=("path", "sha256"),
        context="Hopper correctness attempt journal",
    )
    attempt_path = Path(nonblank_string(attempt["path"], context="gate attempt journal path"))
    canonical_attempt_path = attempt_journal_path(gate_path)
    if attempt_path != canonical_attempt_path:
        raise ValueError("Hopper correctness attempt journal path is not canonical")
    if not attempt_path.is_file() or attempt["sha256"] != _sha256_file(attempt_path):
        raise ValueError("Hopper correctness attempt journal digest is invalid")
    journal_records = AttemptJournal.load(attempt_path).records

    inputs = exact_fields(
        sidecar["inputs"],
        required=("wheel", "wheel_manifest", "source"),
        context="Hopper correctness manifest inputs",
    )
    wheel_input = exact_fields(
        inputs["wheel"], required=("path", "sha256"), context="gate wheel input"
    )
    wheel_manifest_input = exact_fields(
        inputs["wheel_manifest"],
        required=("path", "sha256"),
        context="gate wheel manifest input",
    )
    source_input = exact_fields(inputs["source"], required=("sha256",), context="gate source input")
    if wheel_input["sha256"] != wheel_provenance["wheel_sha256"]:
        raise ValueError("Hopper correctness gate input wheel digest is stale")
    if wheel_manifest_input["sha256"] != wheel_provenance["manifest_sha256"]:
        raise ValueError("Hopper correctness gate wheel manifest digest is stale")
    if source_input["sha256"] != wheel_provenance["source_sha256"]:
        raise ValueError("Hopper correctness gate source digest is stale")
    facts = exact_fields(
        sidecar["facts"],
        required=(
            "head_commit",
            "call_id",
            "operator_command",
            "correctness_only",
            "performance_validated",
            "modal",
            "python",
            "numpy",
            "rich",
            "zstandard",
            "torch",
            "triton",
        ),
        context="Hopper correctness manifest facts",
    )
    if facts["head_commit"] != wheel_provenance["head_commit"]:
        raise ValueError("Hopper correctness manifest facts are stale for the current HEAD")
    if facts["call_id"] != remote_call["call_id"]:
        raise ValueError("Hopper correctness manifest and artifact call IDs differ")
    if len(journal_records) != 2:
        raise ValueError("Hopper correctness attempt journal must contain exactly two records")
    spawned, completed = journal_records
    if (
        spawned.key != (_GPU, 0)
        or completed.key != spawned.key
        or spawned.status != "spawned"
        or completed.status != "completed"
    ):
        raise ValueError(
            "Hopper correctness attempt journal must contain one spawned/completed call"
        )
    expected_request_sha256 = _sha256_payload(
        strict_json_dumps(expected_gate_request, compact=True)
    )
    expected_journal_digests = {
        "request_sha256": expected_request_sha256,
        "protocol_sha256": binding["protocol_sha256"],
        "config_manifest_sha256": binding["config_manifest_sha256"],
        "wheel_sha256": binding["wheel_sha256"],
        "head_sha256": binding["head_sha256"],
    }
    for record in journal_records:
        if any(
            getattr(record, field) != expected
            for field, expected in expected_journal_digests.items()
        ):
            raise ValueError("Hopper correctness attempt journal request or binding digest differs")
        if record.call_id != remote_call["call_id"]:
            raise ValueError("Hopper correctness attempt journal call ID differs")
    if completed.chunk_sha256 != remote_call["payload_sha256"]:
        raise ValueError("Hopper correctness attempt journal payload digest differs")

    return {
        "artifact": str(gate_path),
        "artifact_sha256": artifact_sha256,
        "manifest": str(sidecar_path),
        "manifest_sha256": _sha256_file(sidecar_path),
    }


def _validated_benchmark_timing(value: object, *, context: str) -> dict[str, float]:
    from heliostune.validation import exact_fields, finite_float

    timing = exact_fields(
        value,
        required=("p20_ms", "median_ms", "p80_ms", "wall_ms"),
        context=context,
    )
    validated = {
        field: finite_float(timing[field], context=f"{context} {field}", minimum=0)
        for field in ("p20_ms", "median_ms", "p80_ms", "wall_ms")
    }
    if not (validated["p20_ms"] <= validated["median_ms"] <= validated["p80_ms"]):
        raise ValueError(f"{context} quantiles are not ordered")
    if validated["median_ms"] <= 0 or validated["wall_ms"] <= 0:
        raise ValueError(f"{context} median_ms and wall_ms must be positive")
    return validated


def _validated_benchmark_payload(
    payload: object,
) -> tuple[
    dict[str, object],
    dict[str, list[dict[str, object]]],
    list[dict[str, int | str]],
    list[dict[str, object]],
]:
    """Validate the exact one-bank remote benchmark result and cross-product."""
    from heliostune.configs import (
        DEFAULT_WORKLOADS,
        HOPPER_GEMM_CONFIGS,
        SKINNY_GEMV_CONFIGS,
    )
    from heliostune.hardware import expectation_for_gpu, validate_hardware
    from heliostune.schema import HardwareProfile
    from heliostune.validation import (
        exact_bool,
        exact_fields,
        exact_int,
        finite_float,
        nonblank_string,
    )

    data = exact_fields(
        payload,
        required=("hardware", "protocol", "configs", "workloads", "rows"),
        context="Hopper benchmark payload",
    )
    hardware = HardwareProfile.from_dict(data["hardware"])
    if hardware.gpu != _GPU:
        raise ValueError("Hopper benchmark payload hardware is not H100")
    validate_hardware(hardware, expectation_for_gpu(_GPU))

    protocol = exact_fields(
        data["protocol"],
        required=(
            "warmup_ms",
            "rep_ms",
            "quantiles",
            "candidate_policy",
            "expected_workloads",
            "expected_skinny_workloads",
            "expected_hopper_workloads",
            "expected_skinny_rows",
            "expected_hopper_rows",
            "expected_candidate_rows",
            "torch_measurements",
        ),
        context="Hopper benchmark protocol",
    )
    expected_protocol_scalars = {
        "warmup_ms": _BENCHMARK_WARMUP_MS,
        "rep_ms": _BENCHMARK_REP_MS,
        "expected_workloads": 96,
        "expected_skinny_workloads": 32,
        "expected_hopper_workloads": 64,
        "expected_skinny_rows": 32 * 48,
        "expected_hopper_rows": 64 * 23,
        "expected_candidate_rows": _BENCHMARK_EXPECTED_ROWS,
        "torch_measurements": 96,
    }
    for field, expected in expected_protocol_scalars.items():
        if exact_int(protocol[field], context=f"benchmark protocol {field}", minimum=0) != expected:
            raise ValueError(f"Hopper benchmark payload reports the wrong protocol {field}")
    if protocol["quantiles"] != list(_BENCHMARK_QUANTILES):
        raise ValueError("Hopper benchmark payload reports the wrong quantiles")
    expected_policy = {
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
    }
    if protocol["candidate_policy"] != expected_policy:
        raise ValueError("Hopper benchmark payload reports the wrong candidate policy")

    expected_configs = _gate_config_manifest()
    configs_data = exact_fields(
        data["configs"],
        required=("hopper_gemm", "skinny_gemv"),
        context="Hopper benchmark configs",
    )
    configs = {
        "hopper_gemm": cast(list[dict[str, object]], configs_data["hopper_gemm"]),
        "skinny_gemv": cast(list[dict[str, object]], configs_data["skinny_gemv"]),
    }
    if configs != expected_configs:
        raise ValueError("Hopper benchmark config manifest differs from local source")

    expected_workloads = [workload.to_dict() for workload in DEFAULT_WORKLOADS]
    if data["workloads"] != expected_workloads:
        raise ValueError("Hopper benchmark workload list differs from DEFAULT_WORKLOADS")
    workloads = cast(list[dict[str, int | str]], data["workloads"])
    workload_by_key = {workload.key: workload for workload in DEFAULT_WORKLOADS}
    shuffled_workloads = list(DEFAULT_WORKLOADS)
    random.Random(_BENCHMARK_BANK).shuffle(shuffled_workloads)
    expected_seed_by_workload = {
        workload.key: seed for seed, workload in enumerate(shuffled_workloads)
    }
    expected_pairs = {
        (workload.key, config.key)
        for workload in DEFAULT_WORKLOADS
        for config in (SKINNY_GEMV_CONFIGS if workload.m <= 8 else HOPPER_GEMM_CONFIGS)
    }

    raw_rows = data["rows"]
    if type(raw_rows) is not list:
        raise ValueError("Hopper benchmark rows must be a list")
    if len(raw_rows) != _BENCHMARK_EXPECTED_ROWS:
        raise ValueError(
            f"Hopper benchmark payload must contain exactly {_BENCHMARK_EXPECTED_ROWS} rows"
        )
    seen: set[tuple[str, str]] = set()
    torch_by_workload: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        row = exact_fields(
            raw_row,
            required=(
                "workload_key",
                "workload",
                "regime",
                "config_kind",
                "config_key",
                "config",
                "bank",
                "seed",
                "latency",
                "torch",
                "correct",
                "max_abs_error",
            ),
            context="Hopper benchmark row",
        )
        workload_key = nonblank_string(row["workload_key"], context="benchmark workload_key")
        workload = workload_by_key.get(workload_key)
        if workload is None or row["workload"] != workload.to_dict():
            raise ValueError(f"Hopper benchmark row has wrong workload {workload_key}")
        expected_regime = "skinny_gemv" if workload.m <= 8 else "hopper_gemm"
        regime = nonblank_string(row["regime"], context=f"{workload_key} regime")
        config_kind = nonblank_string(row["config_kind"], context=f"{workload_key} config_kind")
        if regime != expected_regime or config_kind != regime:
            raise ValueError(
                f"Hopper benchmark row has wrong or mismatched regime/config_kind for {workload_key}"
            )
        config_key = nonblank_string(row["config_key"], context=f"{workload_key} config_key")
        if regime == "skinny_gemv":
            config_body = next(
                (
                    candidate.to_dict()
                    for candidate in SKINNY_GEMV_CONFIGS
                    if candidate.key == config_key
                ),
                None,
            )
        else:
            config_body = next(
                (
                    candidate.to_dict()
                    for candidate in HOPPER_GEMM_CONFIGS
                    if candidate.key == config_key
                ),
                None,
            )
        if config_body is None or row["config"] != config_body:
            raise ValueError(f"Hopper benchmark row has wrong config {config_key}")
        pair = (workload_key, config_key)
        if pair in seen:
            raise ValueError(f"Hopper benchmark payload duplicates row {pair}")
        seen.add(pair)
        if exact_int(row["bank"], context=f"{pair} bank", minimum=0) != _BENCHMARK_BANK:
            raise ValueError(f"Hopper benchmark row has wrong bank for {pair}")
        if (
            exact_int(row["seed"], context=f"{pair} seed", minimum=0)
            != expected_seed_by_workload[workload_key]
        ):
            raise ValueError(f"Hopper benchmark row has wrong seed for {pair}")
        latency = _validated_benchmark_timing(row["latency"], context=f"{pair} latency")
        torch_timing = _validated_benchmark_timing(row["torch"], context=f"{pair} torch")
        prior_torch = torch_by_workload.setdefault(workload_key, torch_timing)
        if prior_torch != torch_timing:
            raise ValueError(
                f"Hopper benchmark repeats inconsistent torch timing for {workload_key}"
            )
        if not exact_bool(row["correct"], context=f"{pair} correct"):
            raise ValueError(f"Hopper benchmark contains an incorrect candidate {pair}")
        maximum_error = finite_float(
            row["max_abs_error"], context=f"{pair} max_abs_error", minimum=0
        )
        row["latency"] = latency
        row["torch"] = torch_timing
        row["max_abs_error"] = maximum_error
        rows.append(row)
    if seen != expected_pairs:
        raise ValueError("Hopper benchmark rows are not the exact workload/config cross-product")
    if len(torch_by_workload) != len(DEFAULT_WORKLOADS):
        raise ValueError("Hopper benchmark payload is missing a torch workload baseline")

    data["hardware"] = hardware.to_dict()
    return data, configs, workloads, rows


@app.local_entrypoint()
def main(
    archive: str = _DEFAULT_ARCHIVE,
    output: str = _DEFAULT_OUTPUT,
    banks: str = _FROZEN_BANKS,
    warmup_ms: int = _FROZEN_WARMUP_MS,
    rep_ms: int = _FROZEN_REP_MS,
    workloads: str = "",
    wheel: str = "",
) -> None:
    from heliostune.artifacts import strict_json_dumps, write_json_atomic
    from heliostune.collection import (
        AttemptRecord,
        AttemptStatus,
        CollectionBinding,
        CollectionRequest,
        RemoteCall,
        attempt_journal_path,
        manifest_path,
        preflight_collection,
    )
    from heliostune.configs import DEFAULT_WORKLOADS
    from heliostune.protocol import runtime_manifest
    from heliostune.v3_artifacts import sha256_file

    if type(warmup_ms) is not int or warmup_ms != _FROZEN_WARMUP_MS:
        raise ValueError(f"warmup-ms must be exactly {_FROZEN_WARMUP_MS}")
    if type(rep_ms) is not int or rep_ms != _FROZEN_REP_MS:
        raise ValueError(f"rep-ms must be exactly {_FROZEN_REP_MS}")
    bank_values = _parse_banks(banks)
    destination = _resolved_output(output)
    archive_path = Path(archive)
    if not archive_path.is_file():
        raise ValueError(f"archive does not exist: {archive_path}")

    known = {item.key for item in DEFAULT_WORKLOADS}
    if workloads:
        workload_keys = tuple(workloads.split(","))
        if any(not key or key != key.strip() for key in workload_keys):
            raise ValueError("workloads must be comma-separated without blanks or whitespace")
        unknown = sorted(set(workload_keys) - known)
        if unknown:
            raise ValueError(f"unknown workload key(s): {unknown}")
        if len(set(workload_keys)) != len(workload_keys):
            raise ValueError("workloads must not contain duplicates")
    else:
        workload_keys = tuple(item.key for item in DEFAULT_WORKLOADS)

    baseline = archive_baseline(archive_path, _GPU)
    missing = sorted(set(workload_keys) - set(baseline["workloads"]))
    if missing:
        raise ValueError(f"archive has no {_GPU} winner for workload(s): {missing}")
    triton_config_keys = {
        key: str(baseline["workloads"][key]["best_triton_config_key"]) for key in workload_keys
    }
    protocol = _probe_protocol(
        banks=bank_values,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        workload_keys=workload_keys,
    )

    request = CollectionRequest(
        gpus=(_GPU,),
        banks=bank_values,
        workload_keys=workload_keys,
        config_keys=tuple(sorted(set(triton_config_keys.values()))),
        warmup_ms=float(warmup_ms),
        repetition_ms=float(rep_ms),
        seed_protocol="legacy-bank",
    )
    wheel_path = _resolve_wheel(wheel)
    head_commit, head_sha256 = _git_identity()
    archive_sha256 = sha256_file(archive_path)
    binding = CollectionBinding(
        protocol_sha256=_sha256_payload(strict_json_dumps(protocol, compact=True)),
        config_manifest_sha256=_sha256_payload(
            strict_json_dumps(
                {"archive_sha256": archive_sha256, "triton_config_keys": triton_config_keys},
                compact=True,
            )
        ),
        wheel_sha256=sha256_file(wheel_path),
        head_sha256=head_sha256,
    )
    journal = preflight_collection(destination)

    def record(
        bank: int,
        call_id: str,
        status: AttemptStatus,
        *,
        chunk_sha256: str | None = None,
        error: str | None = None,
    ) -> AttemptRecord:
        return AttemptRecord(
            request_sha256=request.sha256,
            **binding.to_dict(),
            gpu=_GPU,
            bank=bank,
            call_id=call_id,
            status=status,
            timestamp_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            chunk_sha256=chunk_sha256,
            error=error,
        )

    banks_payload: list[dict[str, Any]] = []
    rows: list[dict[str, object]] = []
    hardware_identity: object | None = None
    for bank in bank_values:
        call: RemoteCall = probe_h100.spawn(
            bank=bank,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
            workload_keys=workload_keys,
            triton_config_keys=triton_config_keys,
        )
        call_id = call.object_id
        journal.append(record(bank, call_id, "spawned"))
        try:
            payload = call.get()
            bank_data, bank_rows = _validated_payload(
                payload,
                bank=bank,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
                workload_keys=workload_keys,
                triton_config_keys=triton_config_keys,
            )
            if hardware_identity is None:
                hardware_identity = bank_data["hardware"]
            elif hardware_identity != bank_data["hardware"]:
                raise ValueError("probe payloads disagree on H100 hardware identity")
        except Exception as exc:
            journal.append(
                record(bank, call_id, "failed", error=f"{type(exc).__name__}: {exc}".strip())
            )
            # No retry and no later spawn: at most this retrieved call could be orphaned.
            raise
        chunk_sha256 = _sha256_payload(strict_json_dumps(payload, compact=True))
        journal.append(record(bank, call_id, "completed", chunk_sha256=chunk_sha256))
        banks_payload.append(
            {
                "bank": bank,
                "call_id": call_id,
                "chunk_sha256": chunk_sha256,
                "hardware": bank_data["hardware"],
            }
        )
        rows.extend(bank_rows)

    _validate_cross_product(rows, banks=bank_values, workload_keys=workload_keys)
    rows.sort(key=lambda row: (str(row["workload_key"]), int(str(row["bank"]))))
    artifact = {
        "schema_version": _PROBE_SCHEMA_VERSION,
        "probe": _PROBE_NAME,
        "question": (
            "Does torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction "
            "explain torch.matmul's H100 lead over the best archived Triton config?"
        ),
        "role": "post-hoc exploratory; not a confirmatory Parhelion endpoint",
        "protocol": protocol,
        "banks": banks_payload,
        "archive_baseline": baseline,
        "rows": rows,
    }
    artifact_sha256 = _serialized_json_sha256(artifact)
    sidecar = manifest_path(destination)
    journal_path = attempt_journal_path(destination)
    write_json_atomic(
        sidecar,
        {
            "schema_version": _PROBE_SCHEMA_VERSION,
            "probe": _PROBE_NAME,
            "request": request.to_dict(),
            "binding": binding.to_dict(),
            "protocol": protocol,
            "triton_config_keys": triton_config_keys,
            # Keyed "data" and "attempt_journal" so heliostune.collection's preflight
            # recognises the pair and refuses to overwrite a digest-valid probe.
            "data": {
                "path": str(destination),
                "sha256": artifact_sha256,
                "rows": len(rows),
            },
            "attempt_journal": {
                "path": str(journal_path),
                "sha256": sha256_file(journal_path),
            },
            "inputs": {
                "archive": {"path": str(archive_path), "sha256": archive_sha256},
                "wheel": {"path": str(wheel_path), "sha256": binding.wheel_sha256},
            },
            "facts": {
                "head_commit": head_commit,
                "attempts": [
                    {"bank": item["bank"], "call_id": item["call_id"]} for item in banks_payload
                ],
                "modal": importlib.metadata.version("modal"),
                "python": "3.11",
                "numpy": "2.4.6",
                "rich": "14.3.4",
                "zstandard": "0.25.0",
                "torch": "2.8.0",
                "triton": "3.4.0",
            },
            "analysis_runtime": runtime_manifest(),
        },
    )
    write_json_atomic(destination, artifact)
    print(f"wrote {destination} ({sha256_file(destination)})")
    print(f"wrote {sidecar}")


@app.local_entrypoint()
def hopper_gate(
    output: str = _DEFAULT_GATE_OUTPUT,
    wheel: str = "",
) -> None:
    """Run the one-call correctness-only gate required before paid collection."""
    from heliostune.artifacts import strict_json_dumps, write_json_atomic
    from heliostune.collection import (
        AttemptRecord,
        AttemptStatus,
        CollectionBinding,
        RemoteCall,
        attempt_journal_path,
        manifest_path,
        preflight_collection,
    )
    from heliostune.protocol import runtime_manifest
    from heliostune.v3_artifacts import sha256_file
    from heliostune.validation import nonblank_string

    destination = _resolved_gate_output(output)
    wheel_path = _resolve_wheel(wheel)
    head_commit, head_sha256 = _git_identity()
    wheel_provenance = _validated_wheel_provenance(
        wheel_path,
        head_commit=head_commit,
    )
    config_manifest = _gate_config_manifest()
    config_manifest_sha256 = _gate_config_manifest_sha256(config_manifest)
    protocol = {
        "schema_version": _GATE_SCHEMA_VERSION,
        "gate": _GATE_NAME,
        "study_status": _GATE_STATUS,
        "analysis_status": _GATE_STATUS,
        "correctness_only": True,
        "performance_validated": False,
        "timing_operations": 0,
        "gpu": _GPU,
        "gpu_selector": _MODAL_SELECTOR,
        "remote_function": "hopper_correctness_h100",
        "remote_call_count": 1,
        "timeout_seconds": _GATE_TIMEOUT_SECONDS,
        "operator_sequence": [
            "precision_probe",
            "hopper_correctness_gate",
            "full_collection",
        ],
    }
    request = {
        "schema_version": _GATE_SCHEMA_VERSION,
        "gate": _GATE_NAME,
        "gpu": _GPU,
        "gpu_selector": _MODAL_SELECTOR,
        "remote_call_count": 1,
        "resume": False,
        "retry": False,
    }
    request_sha256 = _sha256_payload(strict_json_dumps(request, compact=True))
    binding = CollectionBinding(
        protocol_sha256=_sha256_payload(strict_json_dumps(protocol, compact=True)),
        config_manifest_sha256=config_manifest_sha256,
        wheel_sha256=wheel_provenance["wheel_sha256"],
        head_sha256=head_sha256,
    )
    journal = preflight_collection(destination)

    def record(
        call_id: str,
        status: AttemptStatus,
        *,
        chunk_sha256: str | None = None,
        error: str | None = None,
    ) -> AttemptRecord:
        return AttemptRecord(
            request_sha256=request_sha256,
            **binding.to_dict(),
            gpu=_GPU,
            # The journal schema requires a non-negative call-plan slot. This gate has
            # exactly one unbanked call, represented by slot zero.
            bank=0,
            call_id=call_id,
            status=status,
            timestamp_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            chunk_sha256=chunk_sha256,
            error=error,
        )

    call: RemoteCall = hopper_correctness_h100.spawn()
    call_id = nonblank_string(call.object_id, context="Hopper correctness call ID")
    # AttemptJournal.append flushes and fsyncs both the record and parent directory.
    journal.append(record(call_id, "spawned"))
    try:
        remote_payload = call.get()
        data, candidate_summaries, validation_results = _validated_gate_payload(remote_payload)
    except Exception as exc:
        journal.append(record(call_id, "failed", error=f"{type(exc).__name__}: {exc}".strip()))
        raise

    payload_sha256 = _sha256_payload(strict_json_dumps(remote_payload, compact=True))
    journal.append(record(call_id, "completed", chunk_sha256=payload_sha256))
    artifact = {
        "schema_version": _GATE_SCHEMA_VERSION,
        "gate": _GATE_NAME,
        "study_status": _GATE_STATUS,
        "analysis_status": _GATE_STATUS,
        "verified": True,
        "correctness_only": True,
        "performance_validated": False,
        "protocol": protocol,
        "gpu": _GPU,
        "gpu_selector": _MODAL_SELECTOR,
        "hardware": data["hardware"],
        "config_counts": data["config_counts"],
        "config_manifest": {
            "sha256": config_manifest_sha256,
            "hopper_gemm": config_manifest["hopper_gemm"],
            "skinny_gemv": config_manifest["skinny_gemv"],
        },
        "validation_workload_count": data["validation_workload_count"],
        "validation_check_count": data["validation_check_count"],
        "candidate_summaries": candidate_summaries,
        "validation_results": validation_results,
        "remote_call": {
            "call_id": call_id,
            "payload_sha256": payload_sha256,
        },
    }

    sidecar = manifest_path(destination)
    journal_path = attempt_journal_path(destination)
    write_json_atomic(
        sidecar,
        {
            "schema_version": _GATE_SCHEMA_VERSION,
            "gate": _GATE_NAME,
            "verified": True,
            "request": request,
            "binding": binding.to_dict(),
            "protocol": protocol,
            "data": {
                "path": str(destination),
                "sha256": _serialized_json_sha256(artifact),
                "candidate_summaries": len(candidate_summaries),
                "validation_results": len(validation_results),
            },
            "attempt_journal": {
                "path": str(journal_path),
                "sha256": sha256_file(journal_path),
            },
            "inputs": {
                "wheel": {
                    "path": str(wheel_path),
                    "sha256": wheel_provenance["wheel_sha256"],
                },
                "wheel_manifest": {
                    "path": wheel_provenance["manifest_path"],
                    "sha256": wheel_provenance["manifest_sha256"],
                },
                "source": {"sha256": wheel_provenance["source_sha256"]},
            },
            "facts": {
                "head_commit": head_commit,
                "call_id": call_id,
                "operator_command": "modal run modal_precision_probe.py::hopper_gate",
                "correctness_only": True,
                "performance_validated": False,
                "modal": importlib.metadata.version("modal"),
                "python": _PYTHON_VERSION,
                "numpy": "2.4.6",
                "rich": "14.3.4",
                "zstandard": "0.25.0",
                "torch": "2.8.0",
                "triton": "3.4.0",
            },
            "analysis_runtime": runtime_manifest(),
        },
    )
    write_json_atomic(destination, artifact)
    print(f"wrote {destination} ({sha256_file(destination)})")
    print(f"wrote {sidecar}")


@app.local_entrypoint()
def hopper_benchmark(
    output: str = _DEFAULT_BENCHMARK_OUTPUT,
    wheel: str = "",
) -> None:
    """Run the single-call, bank-zero H100 engineering benchmark."""
    from heliostune.artifacts import strict_json_dumps, write_json_atomic
    from heliostune.collection import (
        AttemptRecord,
        AttemptStatus,
        CollectionBinding,
        RemoteCall,
        attempt_journal_path,
        manifest_path,
        preflight_collection,
    )
    from heliostune.protocol import runtime_manifest
    from heliostune.v3_artifacts import sha256_file
    from heliostune.validation import nonblank_string

    destination = _resolved_gate_output(output)
    gate_path = _resolved_gate_output(_DEFAULT_GATE_OUTPUT)
    if destination == gate_path:
        raise ValueError("benchmark output must differ from the correctness gate artifact")
    wheel_path = _resolve_wheel(wheel)
    head_commit, head_sha256 = _git_identity()
    wheel_provenance = _validated_wheel_provenance(
        wheel_path,
        head_commit=head_commit,
    )
    correctness = _validated_correctness_gate(
        gate_path,
        wheel_provenance=wheel_provenance,
        head_sha256=head_sha256,
    )

    config_manifest = _gate_config_manifest()
    config_manifest_sha256 = _gate_config_manifest_sha256(config_manifest)
    request = {
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "study_id": _BENCHMARK_STUDY_ID,
        "analysis_status": _GATE_STATUS,
        "gpu": _GPU,
        "gpu_selector": _MODAL_SELECTOR,
        "bank": _BENCHMARK_BANK,
        "remote_call_count": 1,
        "resume": False,
        "retry": False,
    }
    protocol_binding = {
        "warmup_ms": _BENCHMARK_WARMUP_MS,
        "rep_ms": _BENCHMARK_REP_MS,
        "quantiles": list(_BENCHMARK_QUANTILES),
        "expected_workloads": 96,
        "expected_candidate_rows": _BENCHMARK_EXPECTED_ROWS,
    }
    request_sha256 = _sha256_payload(strict_json_dumps(request, compact=True))
    binding = CollectionBinding(
        protocol_sha256=_sha256_payload(strict_json_dumps(protocol_binding, compact=True)),
        config_manifest_sha256=config_manifest_sha256,
        wheel_sha256=wheel_provenance["wheel_sha256"],
        head_sha256=head_sha256,
    )
    journal = preflight_collection(destination)

    def record(
        call_id: str,
        status: AttemptStatus,
        *,
        chunk_sha256: str | None = None,
        error: str | None = None,
    ) -> AttemptRecord:
        return AttemptRecord(
            request_sha256=request_sha256,
            **binding.to_dict(),
            gpu=_GPU,
            bank=_BENCHMARK_BANK,
            call_id=call_id,
            status=status,
            timestamp_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            chunk_sha256=chunk_sha256,
            error=error,
        )

    call: RemoteCall = hopper_benchmark_h100.spawn()
    call_id = nonblank_string(call.object_id, context="Hopper benchmark call ID")
    # AttemptJournal.append flushes and fsyncs before retrieval.
    journal.append(record(call_id, "spawned"))
    try:
        remote_payload = call.get()
        data, configs, workloads, rows = _validated_benchmark_payload(remote_payload)
    except Exception as exc:
        journal.append(record(call_id, "failed", error=f"{type(exc).__name__}: {exc}".strip()))
        raise

    payload_sha256 = _sha256_payload(strict_json_dumps(remote_payload, compact=True))
    journal.append(record(call_id, "completed", chunk_sha256=payload_sha256))
    artifact = {
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "study_id": _BENCHMARK_STUDY_ID,
        "analysis_status": _GATE_STATUS,
        "gpu": _GPU,
        "gpu_selector": _MODAL_SELECTOR,
        "hardware": data["hardware"],
        "bank": _BENCHMARK_BANK,
        "protocol": data["protocol"],
        "correctness_gate": correctness,
        "config_manifest_sha256": config_manifest_sha256,
        "configs": configs,
        "workloads": workloads,
        "rows": rows,
        "verified": True,
    }

    sidecar = manifest_path(destination)
    journal_path = attempt_journal_path(destination)
    write_json_atomic(
        sidecar,
        {
            "schema_version": _BENCHMARK_SCHEMA_VERSION,
            "study_id": _BENCHMARK_STUDY_ID,
            "analysis_status": _GATE_STATUS,
            "verified": True,
            "request": request,
            "binding": binding.to_dict(),
            "protocol": data["protocol"],
            "data": {
                "path": str(destination),
                "sha256": _serialized_json_sha256(artifact),
                "rows": len(rows),
                "workloads": len(workloads),
            },
            "attempt_journal": {
                "path": str(journal_path),
                "sha256": sha256_file(journal_path),
            },
            "inputs": {
                "correctness_gate": correctness,
                "wheel": {
                    "path": str(wheel_path),
                    "sha256": wheel_provenance["wheel_sha256"],
                },
                "wheel_manifest": {
                    "path": wheel_provenance["manifest_path"],
                    "sha256": wheel_provenance["manifest_sha256"],
                },
                "source": {"sha256": wheel_provenance["source_sha256"]},
            },
            "remote_call": {
                "call_id": call_id,
                "payload_sha256": payload_sha256,
            },
            "facts": {
                "head_commit": head_commit,
                "operator_command": "modal run modal_precision_probe.py::hopper_benchmark",
                "remote_function": "hopper_benchmark_h100",
                "modal": importlib.metadata.version("modal"),
                "python": _PYTHON_VERSION,
                "numpy": "2.4.6",
                "rich": "14.3.4",
                "zstandard": "0.25.0",
                "torch": "2.8.0",
                "triton": "3.4.0",
            },
            "analysis_runtime": runtime_manifest(),
        },
    )
    write_json_atomic(destination, artifact)
    print(f"wrote {destination} ({sha256_file(destination)})")
    print(f"wrote {sidecar}")
