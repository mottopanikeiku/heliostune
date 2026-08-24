"""Validate that one release tag exactly matches installed package metadata."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


def check_tag(tag: str) -> bool:
    version = importlib.metadata.version("heliostune")
    return tag == f"v{version}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    args = parser.parse_args(argv)
    if check_tag(args.tag):
        return 0
    expected = f"v{importlib.metadata.version('heliostune')}"
    print(f"release tag must be exactly {expected}, got {args.tag!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
