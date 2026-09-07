import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analysis.summarize_canonical_qualification import _paired_comparison


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _effect(
    full: dict[str, Any],
    ablated: dict[str, Any],
) -> dict[str, list[float]]:
    full_returns = np.asarray(full["episode_returns"], dtype=np.float64)
    ablated_returns = np.asarray(ablated["episode_returns"], dtype=np.float64)
    if full_returns.shape != ablated_returns.shape:
        raise ValueError("Feature gates must use the same episode count")
    return {"episode_returns": (full_returns - ablated_returns).tolist()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired canonical feature-gate effects by encoder.",
    )
    parser.add_argument("--gcn-seed-dir", type=Path, required=True)
    parser.add_argument("--flat-seed-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filenames = {
        "full": "evaluation_stochastic.json",
        "static": "observation_gate_static_only_stochastic.json",
        "topology": "observation_gate_topology_only_stochastic.json",
        "topology_no_adjacency": (
            "observation_gate_topology_no_adjacency_stochastic.json"
        ),
    }
    gcn = {name: _load(args.gcn_seed_dir / path) for name, path in filenames.items()}
    flat = {
        name: _load(args.flat_seed_dir / path)
        for name, path in filenames.items()
    }
    gcn_config = _load(args.gcn_seed_dir / "config.json")
    flat_config = _load(args.flat_seed_dir / "config.json")

    gcn_dynamic = _effect(gcn["full"], gcn["static"])
    flat_dynamic = _effect(flat["full"], flat["static"])
    gcn_adjacency = _effect(gcn["topology"], gcn["topology_no_adjacency"])
    flat_adjacency = _effect(flat["topology"], flat["topology_no_adjacency"])

    rng = np.random.default_rng(20_260_821)
    comparisons = {
        "gcn_minus_flat_full_return": _paired_comparison(
            gcn["full"], flat["full"], rng
        ),
        "gcn_minus_flat_static_return": _paired_comparison(
            gcn["static"], flat["static"], rng
        ),
        "dynamic_use_interaction_gcn_minus_flat": _paired_comparison(
            gcn_dynamic, flat_dynamic, rng
        ),
        "adjacency_use_interaction_gcn_minus_flat": _paired_comparison(
            gcn_adjacency, flat_adjacency, rng
        ),
    }
    result = {
        "protocol": {
            "gcn_requested_training_steps": int(
                gcn_config["total_simulator_steps"]
            ),
            "flat_requested_training_steps": int(
                flat_config["total_simulator_steps"]
            ),
            "gcn_parameter_count": int(gcn_config["parameter_count"]),
            "flat_parameter_count": int(flat_config["parameter_count"]),
            "paired_episode_count": len(gcn["full"]["episode_returns"]),
            "bootstrap_resamples": 20_000,
        },
        "comparisons": comparisons,
    }
    json_path = args.gcn_seed_dir / "encoder_gate_comparison_with_flat.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    def row(name: str) -> str:
        comparison = comparisons[name]
        lower, upper = comparison["bootstrap_ci95"]
        return (
            f"| `{name}` | {comparison['mean_delta']:.2f} | "
            f"[{lower:.2f}, {upper:.2f}] |"
        )

    markdown = f"""# GCN versus Flat Feature-Gate Diagnostic

## Protocol warning

This is not a fair final encoder benchmark. GCN used
{result['protocol']['gcn_requested_training_steps']} requested simulator steps,
while Flat used {result['protocol']['flat_requested_training_steps']}. The
parameter counts are approximately matched ({result['protocol']['gcn_parameter_count']}
versus {result['protocol']['flat_parameter_count']}). All comparisons use the
same 30 held-out environment seeds.

| Comparison | Mean paired delta | Bootstrap 95% CI |
|---|---:|---:|
{row('gcn_minus_flat_full_return')}
{row('gcn_minus_flat_static_return')}
{row('dynamic_use_interaction_gcn_minus_flat')}
{row('adjacency_use_interaction_gcn_minus_flat')}

The dynamic interaction is
`(GCN full - GCN static) - (Flat full - Flat static)`. The adjacency interaction
uses the analogous topology versus topology-without-adjacency difference.
"""
    markdown_path = args.gcn_seed_dir / "ENCODER_GATE_COMPARISON_WITH_FLAT.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"statistics={json_path}")
    print(f"summary={markdown_path}")
    for name, comparison in comparisons.items():
        print(
            f"{name}: {comparison['mean_delta']:.2f} "
            f"{comparison['bootstrap_ci95']}"
        )


if __name__ == "__main__":
    main()
