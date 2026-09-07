import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analysis.summarize_canonical_qualification import _paired_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a paired full-versus-static observation gate."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--full-file", default="evaluation_stochastic.json"
    )
    parser.add_argument(
        "--static-file",
        default="observation_gate_static_only_stochastic.json",
    )
    parser.add_argument(
        "--output-file", default="observation_use_gate_statistics.json"
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_821)
    return parser


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seed_dir = args.run_dir / f"seed_{args.seed}"
    full = _load(seed_dir / args.full_file)
    static = _load(seed_dir / args.static_file)
    comparison = _paired_comparison(
        full,
        static,
        np.random.default_rng(args.bootstrap_seed),
    )
    result = {
        "protocol": {
            "full_file": args.full_file,
            "static_file": args.static_file,
            "paired_episodes": len(full["episode_returns"]),
            "bootstrap_resamples": 20_000,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "full": {
            "mean": full["mean_return"],
            "std": full["std_return"],
        },
        "static_only": {
            "mean": static["mean_return"],
            "std": static["std_return"],
        },
        "full_minus_static_only": comparison,
        "strict_gate_passed": comparison["bootstrap_ci95"][0] > 0,
    }
    output_path = seed_dir / args.output_file
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
