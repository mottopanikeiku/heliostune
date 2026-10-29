"""Collect the frozen old/new action comparison on a Modal H100 under an explicit run ID.

Each run ID gets its own checkpoint directory in the shared Modal volume, so a fresh
repeat needs a new ``--run-id``, not a code change. The measurement itself is
``heliostune.action_expansion.collect_on_device``; ``scripts/collect_action_expansion.py``
runs the same code on any local CUDA machine.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

from heliostune.action_collection import (
    RunSpec,
    checkpoint_path,
    default_output,
    file_sha256,
    git_identity,
    package_source_sha256,
    plan_manifest,
)
from heliostune.tooling.artifacts import strict_json_dumps, write_bytes_atomic_noreplace
from modal_bench import build_image, configured_modal_wheel

_REPO = Path(__file__).resolve().parent
_CHECKPOINTS = Path("/data/runs")

app = modal.App("heliostune-day-actions")
volume = modal.Volume.from_name("heliostune-day-actions", create_if_missing=True)
image = (
    build_image(configured_modal_wheel())
    .env({"TRITON_CACHE_DIR": "/data/triton-cache"})
    .add_local_file("modal_bench.py", remote_path="/root/modal_bench.py")
)


def _collect(fields: dict[str, Any]) -> dict[str, Any]:
    import os

    from heliostune.action_expansion import collect_on_device

    spec = RunSpec(**fields)
    manifest = json.loads(
        Path(os.environ["HELIOSTUNE_MODAL_WHEEL_MANIFEST"]).read_text(encoding="utf-8")
    )
    if manifest["source_sha256"] != spec.source_sha256:
        raise ValueError("the installed wheel does not match the run's source digest")
    return collect_on_device(spec, checkpoint_path(_CHECKPOINTS, spec), after_save=volume.commit)


@app.function(
    image=image,
    gpu="H100!",
    cpu=1,
    memory=4096,
    timeout=240,
    max_containers=1,
    volumes={"/data": volume},
)
def pilot_collect(fields: dict[str, Any]) -> dict[str, Any]:
    return _collect(fields)


@app.function(
    image=image,
    gpu="H100!",
    cpu=1,
    memory=4096,
    timeout=1620,
    max_containers=1,
    volumes={"/data": volume},
)
def full_collect(fields: dict[str, Any]) -> dict[str, Any]:
    return _collect(fields)


@app.local_entrypoint()
def main(run_id: str, pilot: bool = False, output: str = "", dry_run: bool = False) -> None:
    commit, plan_commit, clean = git_identity(_REPO)
    if not clean and not dry_run:
        raise ValueError("commit the action set and analysis plan before collecting measurements")
    spec = RunSpec(
        run_id=run_id,
        runner="modal",
        gpu="H100",
        pilot=pilot,
        source_commit=commit,
        plan_commit=plan_commit,
        collector_sha256=file_sha256(Path(__file__)),
        source_sha256=package_source_sha256(_REPO / "src/heliostune"),
    )
    if dry_run:
        print(
            strict_json_dumps(
                plan_manifest(spec, checkpoint_path(_CHECKPOINTS, spec), clean_tree=clean)
            ),
            end="",
        )
        return
    destination = Path(output) if output else _REPO / default_output(spec)
    if destination.exists():
        raise ValueError(f"{destination} exists; choose a new output path or run ID")
    fields = asdict(spec)
    result = pilot_collect.remote(fields) if pilot else full_collect.remote(fields)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic_noreplace(destination, strict_json_dumps(result).encode("utf-8"))
    print(f"Wrote {len(result['rows'])} rows to {destination}")
