"""Collect HeliosTune matrices with durable paid-call journaling on Modal."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import modal

# PEP 427 wheel filename: distribution-version(-build)?-python-abi-platform.whl. No
# component may contain "-", so pip refuses names such as "heliostune.whl" or
# "heliostune-0.4.0.whl"; commit 0486126 fixed exactly that install failure by
# preserving the built filename inside the container.
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Return the required manifest adjacent to ``wheel``."""
    return wheel.with_name(f"{wheel.name}.manifest.json")


def remote_wheel_manifest_path(wheel: Path) -> str:
    """Return the in-container path of the manifest adjacent to ``wheel``."""
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
    """Validate the wheel's byte binding, and its source/HEAD binding when local."""
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
        source_sha256 = _source_digest(root)
        if data["source_sha256"] != source_sha256:
            raise RuntimeError("Modal wheel source digest does not match the current source tree")
    return manifest


def configured_modal_wheel(root: Path | None = None) -> Path:
    """Return the single manifest-validated wheel to bake into the Modal image."""
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


def remote_wheel_path(wheel: Path) -> str:
    """Return the in-container install path, preserving the wheel's own filename."""
    name = wheel.name
    if _WHEEL_FILENAME.fullmatch(name) is None:
        raise ValueError(
            f"Modal wheel is not a valid PEP 427 filename: {name}; expected "
            "distribution-version(-build)?-python-abi-platform.whl"
        )
    return f"/root/{name}"


def build_image(wheel: Path) -> modal.Image:
    """Return the pinned image after validating and copying wheel plus manifest."""
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


app = modal.App("heliostune-bench")
# Modal binds the image when `@app.function` decorates, so the wheel lookup cannot be
# deferred into the entrypoint. This single call is the module's only import-time file
# access, and it is module-relative rather than cwd-relative.
image = build_image(configured_modal_wheel())


def _remote_collect(
    gpu: str,
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    config_keys: tuple[str, ...],
    seed_protocol: str,
) -> list[dict[str, Any]]:
    from heliostune.configs import (
        DEFAULT_WORKLOADS,
        PARHELION_V3_CANDIDATE_CONFIGS,
    )
    from heliostune.hardware import expectation_for_gpu, validate_hardware
    from heliostune.kernel import collect_benchmarks, get_hardware_profile
    from heliostune.protocol import v3_seed

    workloads_by_key = {workload.key: workload for workload in DEFAULT_WORKLOADS}
    configs_by_key = {config.key: config for config in PARHELION_V3_CANDIDATE_CONFIGS}
    try:
        workloads = tuple(workloads_by_key[key] for key in workload_keys)
        configs = tuple(configs_by_key[key] for key in config_keys)
    except KeyError as exc:
        raise ValueError(f"remote manifest contains unknown key {exc.args[0]!r}") from exc

    # This is deliberately the first CUDA operation. Identity is rejected before
    # any benchmark tensor allocation or paid timing work.
    profile = get_hardware_profile(gpu)
    validate_hardware(profile, expectation_for_gpu(gpu))
    if seed_protocol == "parhelion-v3":
        workload_order_seed = v3_seed(
            purpose="collector-workload-order",
            gpu=gpu,
            bank=bank,
        )
        config_order_seeds = {
            workload.key: v3_seed(
                purpose="collector-config-order",
                gpu=gpu,
                bank=bank,
                workload_key=workload.key,
            )
            for workload in workloads
        }
        tensor_seeds = {
            workload.key: v3_seed(
                purpose="tensor",
                gpu=gpu,
                bank=bank,
                workload_key=workload.key,
            )
            for workload in workloads
        }
    else:
        workload_order_seed = None
        config_order_seeds = None
        tensor_seeds = None
    return collect_benchmarks(
        gpu,
        bank=bank,
        configs=configs,
        workloads=workloads,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
        hardware_profile=profile,
        workload_order_seed=workload_order_seed,
        config_order_seeds=config_order_seeds,
        tensor_seeds=tensor_seeds,
    )


@app.function(image=image, gpu="L4", timeout=60 * 60)
def benchmark_l4(
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    config_keys: tuple[str, ...],
    seed_protocol: str,
) -> list[dict[str, Any]]:
    return _remote_collect("L4", bank, warmup_ms, rep_ms, workload_keys, config_keys, seed_protocol)


@app.function(image=image, gpu="A10", timeout=60 * 60)
def benchmark_a10(
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    config_keys: tuple[str, ...],
    seed_protocol: str,
) -> list[dict[str, Any]]:
    return _remote_collect(
        "A10", bank, warmup_ms, rep_ms, workload_keys, config_keys, seed_protocol
    )


@app.function(image=image, gpu="T4", timeout=60 * 60)
def benchmark_t4(
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    config_keys: tuple[str, ...],
    seed_protocol: str,
) -> list[dict[str, Any]]:
    return _remote_collect("T4", bank, warmup_ms, rep_ms, workload_keys, config_keys, seed_protocol)


@app.function(image=image, gpu="H100!", timeout=60 * 60)
def benchmark_h100(
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    config_keys: tuple[str, ...],
    seed_protocol: str,
) -> list[dict[str, Any]]:
    return _remote_collect(
        "H100", bank, warmup_ms, rep_ms, workload_keys, config_keys, seed_protocol
    )


@app.function(image=image, gpu="A100-80GB", timeout=60 * 60)
def benchmark_a100_80gb(
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    config_keys: tuple[str, ...],
    seed_protocol: str,
) -> list[dict[str, Any]]:
    return _remote_collect(
        "A100-80GB",
        bank,
        warmup_ms,
        rep_ms,
        workload_keys,
        config_keys,
        seed_protocol,
    )


@app.function(image=image, gpu="H200", timeout=60 * 60)
def benchmark_h200(
    bank: int,
    warmup_ms: int,
    rep_ms: int,
    workload_keys: tuple[str, ...],
    config_keys: tuple[str, ...],
    seed_protocol: str,
) -> list[dict[str, Any]]:
    return _remote_collect(
        "H200", bank, warmup_ms, rep_ms, workload_keys, config_keys, seed_protocol
    )


def _strict_csv(value: str, *, label: str) -> tuple[str, ...]:
    values = tuple(value.split(","))
    if not values or any(not item or item != item.strip() for item in values):
        raise ValueError(f"{label} must be a non-empty comma-separated list without whitespace")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def _strict_banks(value: str) -> tuple[int, ...]:
    raw = _strict_csv(value, label="banks")
    if any(not item.isascii() or not item.isdecimal() for item in raw):
        raise ValueError("banks must contain only non-negative decimal integers")
    banks = tuple(int(item) for item in raw)
    if len(set(banks)) != len(banks):
        raise ValueError("banks must not contain duplicates")
    return banks


def _sha256_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_identity() -> tuple[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("Modal collection requires a clean Git HEAD")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return head, _sha256_payload(head)


def _resolve_wheel(path: str) -> Path:
    baked = configured_modal_wheel()
    wheel = Path(path) if path else baked
    if not wheel.is_file():
        raise ValueError(f"Modal wheel does not exist: {wheel}")
    if wheel.resolve() != baked.resolve():
        raise ValueError(
            "--wheel must match the wheel baked into the image; set "
            "HELIOSTUNE_MODAL_WHEEL before `modal run` to choose another wheel"
        )
    return wheel


def _resolved_output(value: str, *, repository: Path = _REPO) -> Path:
    """Resolve a relative destination while protecting the committed benchmarks tree."""
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("output must be a relative path")
    if ".." in candidate.parts:
        raise ValueError("output must not contain '..'")
    destination = candidate.resolve()
    benchmarks = (repository / "benchmarks").resolve()
    if destination == benchmarks or destination.is_relative_to(benchmarks):
        raise ValueError("collection output must never be written under benchmarks/")
    return destination


def _selected_manifests(
    pilot: bool,
    protocol_path: Path | None,
    config_path: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    from heliostune.artifacts import read_json
    from heliostune.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS
    from heliostune.protocol import (
        V3_PILOT_CONFIG_KEYS,
        V3_PILOT_WORKLOAD_KEYS,
        load_v3_protocol,
        require_v3_runtime,
    )
    from heliostune.validation import exact_object

    if protocol_path is None:
        configs = DEFAULT_CONFIGS[:3] if pilot else DEFAULT_CONFIGS
        workloads = DEFAULT_WORKLOADS[:2] if pilot else DEFAULT_WORKLOADS
        return (
            tuple(workload.key for workload in workloads),
            tuple(config.key for config in configs),
            "legacy-bank",
        )
    protocol = load_v3_protocol(protocol_path)
    require_v3_runtime(protocol)
    if pilot:
        if config_path is not None:
            raise ValueError("v3 pilot cannot use a config manifest")
        return V3_PILOT_WORKLOAD_KEYS, V3_PILOT_CONFIG_KEYS, "parhelion-v3"
    workload_rows = protocol.get("workloads")
    config_rows = protocol.get("candidate_configs")
    if not isinstance(workload_rows, list) or not isinstance(config_rows, list):
        raise ValueError("v3 protocol must serialize workloads and candidate_configs")
    workload_keys = tuple(
        str(exact_object(row, context="v3 protocol workload")["key"]) for row in workload_rows
    )
    if config_path is None:
        config_keys = tuple(
            str(exact_object(row, context="v3 protocol config")["key"]) for row in config_rows
        )
    else:
        manifest = exact_object(read_json(config_path), context="v3 config manifest")
        retained = manifest.get("retained_config_keys")
        if not isinstance(retained, list):
            raise ValueError("v3 config manifest must contain retained_config_keys")
        config_keys = tuple(str(key) for key in retained)
    return workload_keys, config_keys, "parhelion-v3"


@app.local_entrypoint()
def main(
    output: str = "measurements.jsonl.zst",
    warmup_ms: int = 25,
    rep_ms: int = 100,
    banks: str = "0,1,2",
    pilot: bool = False,
    gpus: str = "L4,A10",
    resume_attempts: str = "",
    protocol: str = "",
    config_manifest: str = "",
    wheel: str = "",
) -> None:
    from heliostune.artifacts import strict_json_dumps
    from heliostune.collection import (
        CallPlanItem,
        CollectionBinding,
        CollectionRequest,
        RemoteCall,
        commit_chunks,
        execute_call_plan,
        preflight_collection,
        sha256_file,
    )

    gpu_names = _strict_csv(gpus, label="gpus")
    bank_values = _strict_banks(banks)
    functions = {
        "L4": benchmark_l4,
        "A10": benchmark_a10,
        "T4": benchmark_t4,
        "H100": benchmark_h100,
        "A100-80GB": benchmark_a100_80gb,
        "H200": benchmark_h200,
    }
    unknown = [gpu for gpu in gpu_names if gpu not in functions]
    if unknown:
        raise ValueError(
            f"unknown GPU selector(s): {', '.join(unknown)}; choose from {', '.join(functions)}"
        )
    if type(warmup_ms) is not int or warmup_ms < 0:
        raise ValueError("warmup-ms must be a non-negative integer")
    if type(rep_ms) is not int or rep_ms <= 0:
        raise ValueError("rep-ms must be a positive integer")

    protocol_path = Path(protocol) if protocol else None
    config_path = Path(config_manifest) if config_manifest else None
    if protocol_path is not None and not protocol_path.is_file():
        raise ValueError(f"protocol does not exist: {protocol_path}")
    if config_path is not None and not config_path.is_file():
        raise ValueError(f"config manifest does not exist: {config_path}")
    if (
        protocol_path is not None
        and not pilot
        and any(bank != 0 for bank in bank_values)
        and config_path is None
    ):
        raise ValueError("v3 banks 1-4 require the frozen retained config manifest")
    workload_keys, config_keys, seed_protocol = _selected_manifests(
        pilot,
        protocol_path,
        config_path,
    )
    request = CollectionRequest(
        gpus=gpu_names,
        banks=bank_values,
        workload_keys=workload_keys,
        config_keys=config_keys,
        warmup_ms=float(warmup_ms),
        repetition_ms=float(rep_ms),
        pilot=pilot,
        seed_protocol=seed_protocol,
    )
    wheel_path = _resolve_wheel(wheel)
    head_commit, head_sha256 = _git_identity()
    protocol_sha256 = (
        sha256_file(protocol_path)
        if protocol_path is not None
        else _sha256_payload(
            strict_json_dumps(
                {"protocol": "stage1-default", "request": request.to_dict()},
                compact=True,
            )
        )
    )
    config_sha256 = (
        sha256_file(config_path)
        if config_path is not None
        else _sha256_payload(strict_json_dumps({"config_keys": list(config_keys)}, compact=True))
    )
    binding = CollectionBinding(
        protocol_sha256=protocol_sha256,
        config_manifest_sha256=config_sha256,
        wheel_sha256=sha256_file(wheel_path),
        head_sha256=head_sha256,
    )
    destination = _resolved_output(output)
    journal = preflight_collection(
        destination,
        resume_attempts=(resume_attempts or None),
    )

    def spawn(item: CallPlanItem) -> RemoteCall:
        return functions[item.gpu].spawn(
            bank=item.bank,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
            workload_keys=request.workload_keys,
            config_keys=request.config_keys,
            seed_protocol=request.seed_protocol,
        )

    chunks = execute_call_plan(
        request,
        binding,
        journal,
        spawn=spawn,
        restore=(modal.FunctionCall.from_id if resume_attempts else None),
    )
    commit_chunks(
        destination,
        request,
        binding,
        journal,
        chunks,
        facts={
            "head_commit": head_commit,
            "protocol_path": None if protocol_path is None else str(protocol_path),
            "config_manifest_path": None if config_path is None else str(config_path),
            "seed_protocol": request.seed_protocol,
            "wheel_path": str(wheel_path),
            "python": "3.11",
            "numpy": "2.4.6",
            "rich": "14.3.4",
            "zstandard": "0.25.0",
            "torch": "2.8.0",
            "triton": "3.4.0",
            "modal": importlib.metadata.version("modal"),
            "hardware_cache": None,
            "nvidia_smi": None,
        },
    )
