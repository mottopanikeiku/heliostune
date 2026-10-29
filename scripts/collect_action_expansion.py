"""Run or plan the frozen action-expansion collection on this machine's CUDA GPU.

The measurement code is the same as the Modal collector's. ``--dry-run`` writes the
run plan (identity, protocol, unit order and seeds) without importing Torch, so a
run can be checked on CPU before any GPU time is spent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINTS = Path.home() / ".cache/heliostune/action-runs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", required=True, help="new name for a fresh run, or the name to resume"
    )
    parser.add_argument("--gpu", required=True, help="hardware label, e.g. H100 or A100-80GB")
    parser.add_argument(
        "--allow-unlisted-gpu",
        action="store_true",
        help="accept a label with no hardware expectation; the exact observed profile is still bound to the run",
    )
    parser.add_argument(
        "--pilot", action="store_true", help="measure only the four pilot workloads"
    )
    parser.add_argument("--checkpoint-root", type=Path, default=_DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, help="raw artifact path; must not exist yet")
    parser.add_argument("--dry-run", action="store_true", help="write the run plan and stop")
    parser.add_argument(
        "--plan-output", type=Path, help="with --dry-run, write the plan here instead of stdout"
    )
    return parser


def main(argv: list[str] | None = None, *, repository: Path = _REPO) -> int:
    args = _parser().parse_args(argv)
    commit, plan_commit, clean = git_identity(repository)
    spec = RunSpec(
        run_id=args.run_id,
        runner="local",
        gpu=args.gpu,
        pilot=args.pilot,
        source_commit=commit,
        plan_commit=plan_commit,
        collector_sha256=file_sha256(Path(__file__)),
        source_sha256=package_source_sha256(repository / "src/heliostune"),
        allow_unlisted_gpu=args.allow_unlisted_gpu,
    )
    checkpoint = checkpoint_path(args.checkpoint_root, spec)
    if args.dry_run:
        plan = strict_json_dumps(plan_manifest(spec, checkpoint, clean_tree=clean))
        if args.plan_output is None:
            sys.stdout.write(plan)
        else:
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            write_bytes_atomic_noreplace(args.plan_output, plan.encode("utf-8"))
            print(f"Wrote the plan for run {spec.run_id} to {args.plan_output}")
        if not clean:
            print(
                "warning: the working tree has uncommitted changes; a real run would refuse",
                file=sys.stderr,
            )
        return 0
    if not clean:
        raise SystemExit("commit every tracked change before collecting measurements")
    output = args.output if args.output is not None else repository / default_output(spec)
    if output.exists():
        raise SystemExit(f"{output} exists; choose a new output path or run ID")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Torch is not installed; install the gpu extra (uv sync --extra gpu) on the CUDA machine"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device is available; use --dry-run to check the plan on CPU")
    from heliostune.action_expansion import collect_on_device

    result = collect_on_device(spec, checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic_noreplace(output, strict_json_dumps(result).encode("utf-8"))
    print(f"Wrote {len(result['rows'])} rows for run {spec.run_id} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
