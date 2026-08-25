"""Generate or strictly verify the HeliosTune research artifact catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from heliostune.artifacts import write_json_atomic
from heliostune.catalog import build_research_catalog, verify_research_catalog

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT = _REPO / "benchmarks/research-artifact-manifest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", nargs="?", type=Path, default=_DEFAULT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        write_json_atomic(args.catalog, build_research_catalog(_REPO))
    facts = verify_research_catalog(args.catalog)
    print(
        f"measurement_rows={facts['measurement_rows']} "
        f"json_artifacts={facts['json_artifacts']} "
        f"html_reports={facts['html_reports']} "
        f"file_artifacts={facts['file_artifacts']} aliases={facts['aliases']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
