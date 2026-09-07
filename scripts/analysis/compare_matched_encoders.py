import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analysis.summarize_canonical_qualification import _paired_comparison


CONTROL_FIELDS = (
    "canonical_variant",
    "evidence_age_horizon",
    "environment_distribution",
    "total_simulator_steps",
    "episode_steps",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "update_epochs",
    "minibatch_size",
    "reward_scale",
    "clip_coef",
    "value_coef",
    "entropy_coef",
    "max_grad_norm",
    "target_kl",
    "seed",
    "device",
    "latent_dim",
    "head_hidden_dim",
    "observation_dim",
    "num_actions",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare matched Flat and GCN held-out evaluations."
    )
    parser.add_argument("--flat-run-dir", type=Path, required=True)
    parser.add_argument("--gcn-run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/matched_encoder_comparison.json"),
    )
    return parser


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _effect(full: dict[str, Any], static: dict[str, Any]) -> dict[str, list[float]]:
    full_returns = np.asarray(full["episode_returns"], dtype=np.float64)
    static_returns = np.asarray(static["episode_returns"], dtype=np.float64)
    if full_returns.shape != static_returns.shape:
        raise ValueError("Full and static evaluations must be paired")
    return {"episode_returns": (full_returns - static_returns).tolist()}


def _action_type_proportions(evaluation: dict[str, Any]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for label, count in evaluation["action_counts"].items():
        action_label = label.split(": ", maxsplit=1)[-1]
        counts[action_label.split(maxsplit=1)[0]] += int(count)
    total = sum(counts.values())
    return {
        action_type: count / total
        for action_type, count in sorted(counts.items())
    }


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    action_types = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(action, 0.0) - right.get(action, 0.0))
        for action in action_types
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    flat_dir = args.flat_run_dir / f"seed_{args.seed}"
    gcn_dir = args.gcn_run_dir / f"seed_{args.seed}"

    flat_config = _load(flat_dir / "config.json")
    gcn_config = _load(gcn_dir / "config.json")
    mismatches = {
        field: {"flat": flat_config.get(field), "gcn": gcn_config.get(field)}
        for field in CONTROL_FIELDS
        if flat_config.get(field) != gcn_config.get(field)
    }
    if mismatches:
        raise ValueError(f"Matched-control mismatch: {mismatches}")

    flat_full = _load(flat_dir / "evaluation_stochastic.json")
    gcn_full = _load(gcn_dir / "evaluation_stochastic.json")
    flat_static = _load(flat_dir / "observation_gate_static_only_stochastic.json")
    gcn_static = _load(gcn_dir / "observation_gate_static_only_stochastic.json")
    flat_eval_protocol = _load(flat_dir / "evaluation_comparison.json")
    gcn_eval_protocol = _load(gcn_dir / "evaluation_comparison.json")
    for field in ("environment_seed", "stochastic_action_seed"):
        if flat_eval_protocol[field] != gcn_eval_protocol[field]:
            raise ValueError(f"Evaluation protocol mismatch for {field}")

    flat_dynamic = _effect(flat_full, flat_static)
    gcn_dynamic = _effect(gcn_full, gcn_static)
    rng = np.random.default_rng(20_260_821)
    comparisons = {
        "gcn_minus_flat_full": _paired_comparison(gcn_full, flat_full, rng),
        "gcn_minus_flat_static": _paired_comparison(
            gcn_static, flat_static, rng
        ),
        "dynamic_use_interaction_gcn_minus_flat": _paired_comparison(
            gcn_dynamic, flat_dynamic, rng
        ),
    }

    action_proportions = {
        "flat_full": _action_type_proportions(flat_full),
        "flat_static": _action_type_proportions(flat_static),
        "gcn_full": _action_type_proportions(gcn_full),
        "gcn_static": _action_type_proportions(gcn_static),
    }
    result = {
        "protocol": {
            "matched_controls": list(CONTROL_FIELDS),
            "environment_seed": flat_eval_protocol["environment_seed"],
            "stochastic_action_seed": flat_eval_protocol[
                "stochastic_action_seed"
            ],
            "paired_episodes": len(flat_full["episode_returns"]),
            "flat_parameters": int(flat_config["parameter_count"]),
            "gcn_parameters": int(gcn_config["parameter_count"]),
            "relative_parameter_difference": abs(
                int(flat_config["parameter_count"])
                - int(gcn_config["parameter_count"])
            )
            / int(flat_config["parameter_count"]),
        },
        "returns": {
            "flat_full": {
                "mean": flat_full["mean_return"],
                "std": flat_full["std_return"],
            },
            "flat_static": {
                "mean": flat_static["mean_return"],
                "std": flat_static["std_return"],
            },
            "gcn_full": {
                "mean": gcn_full["mean_return"],
                "std": gcn_full["std_return"],
            },
            "gcn_static": {
                "mean": gcn_static["mean_return"],
                "std": gcn_static["std_return"],
            },
        },
        "comparisons": comparisons,
        "action_type_proportions": action_proportions,
        "action_type_total_variation": {
            "flat_full_vs_static": _total_variation(
                action_proportions["flat_full"],
                action_proportions["flat_static"],
            ),
            "gcn_full_vs_static": _total_variation(
                action_proportions["gcn_full"],
                action_proportions["gcn_static"],
            ),
            "gcn_vs_flat_full": _total_variation(
                action_proportions["gcn_full"],
                action_proportions["flat_full"],
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
