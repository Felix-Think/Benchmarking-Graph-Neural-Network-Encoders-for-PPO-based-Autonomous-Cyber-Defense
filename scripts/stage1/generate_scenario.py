"""Generate a frozen CAGE-2-style scenario YAML for one node scale."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from acd_benchmark.cyber.scalable_scenario import (  # noqa: E402
    allocate_subnet_counts,
    write_scalable_scenario,
)


def main() -> None:
    """Write the requested deterministic scenario file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = write_scalable_scenario(args.output, args.nodes)
    print(f"Saved {args.nodes}-node scenario to {path}")
    print(f"Subnet counts: {allocate_subnet_counts(args.nodes)}")


if __name__ == "__main__":
    main()
