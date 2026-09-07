import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analysis.summarize_canonical_qualification import _paired_comparison


EVALUATION_FILES = {
    "full": "evaluation_stochastic.json",
    "static_only": "observation_gate_static_only_stochastic.json",
    "zero_all": "observation_gate_zero_all_stochastic.json",
    "zero_all_no_action_mask": (
        "observation_gate_zero_all_no_action_mask_stochastic.json"
    ),
    "mission_only": "observation_gate_mission_only_stochastic.json",
    "node_identity_only": "observation_gate_node_identity_only_stochastic.json",
    "structure_only": "observation_gate_structure_only_stochastic.json",
    "topology_only": "observation_gate_topology_only_stochastic.json",
    "topology_no_adjacency": (
        "observation_gate_topology_no_adjacency_stochastic.json"
    ),
    "topology_mission": "observation_gate_topology_mission_stochastic.json",
    "full_no_action_mask": (
        "observation_gate_full_no_action_mask_stochastic.json"
    ),
}

INCREMENTAL_COMPARISONS = {
    "action_mask_given_zero_observation": ("zero_all", "zero_all_no_action_mask"),
    "mission_given_zero_observation": ("mission_only", "zero_all"),
    "node_identity_given_zero_observation": ("node_identity_only", "zero_all"),
    "structure_given_zero_observation": ("structure_only", "zero_all"),
    "topology_static_given_zero_observation": ("topology_only", "zero_all"),
    "adjacency_given_identity_and_membership": (
        "topology_only",
        "topology_no_adjacency",
    ),
    "mission_given_topology_static": ("topology_mission", "topology_only"),
    "policy_firewall_globals_given_topology_mission": (
        "static_only",
        "topology_mission",
    ),
    "dynamic_features_given_static_context": ("full", "static_only"),
    "action_mask_given_full_observation": ("full", "full_no_action_mask"),
}


def _load_evaluation(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _action_type_counts(evaluation: dict[str, Any]) -> Counter[str]:
    result: Counter[str] = Counter()
    for label, count in evaluation["action_counts"].items():
        action_label = label.split(": ", maxsplit=1)[-1]
        action_type = action_label.split(maxsplit=1)[0]
        result[action_type] += int(count)
    return result


def _action_type_proportions(evaluation: dict[str, Any]) -> dict[str, float]:
    counts = _action_type_counts(evaluation)
    total = sum(counts.values())
    return {
        action_type: count / total
        for action_type, count in sorted(counts.items())
    }


def _total_variation(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    action_types = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(action_type, 0.0) - right.get(action_type, 0.0))
        for action_type in action_types
    )


def _format_ci(comparison: dict[str, Any]) -> str:
    lower, upper = comparison["bootstrap_ci95"]
    return f"{comparison['mean_delta']:.2f} [{lower:.2f}, {upper:.2f}]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize canonical V1 feature-group evaluation gates.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_dir = args.run_dir / f"seed_{args.seed}"
    evaluations = {
        name: _load_evaluation(seed_dir / filename)
        for name, filename in EVALUATION_FILES.items()
    }
    config = _load_evaluation(seed_dir / "config.json")

    rng = np.random.default_rng(20_260_821 + args.seed)
    full_comparisons = {
        f"full_minus_{name}": _paired_comparison(
            evaluations["full"],
            evaluation,
            rng,
        )
        for name, evaluation in evaluations.items()
        if name != "full"
    }
    incremental_comparisons = {
        name: _paired_comparison(
            evaluations[left],
            evaluations[right],
            rng,
        )
        for name, (left, right) in INCREMENTAL_COMPARISONS.items()
    }

    action_proportions = {
        name: _action_type_proportions(evaluation)
        for name, evaluation in evaluations.items()
    }
    action_tv_from_full = {
        name: round(
            _total_variation(action_proportions["full"], proportions),
            4,
        )
        for name, proportions in action_proportions.items()
    }

    summary = {
        "protocol": {
            "checkpoint": "checkpoint.pt",
            "policy_type": config.get("policy_type", "flat"),
            "training_simulator_steps": int(config["total_simulator_steps"]),
            "evaluation_mode": "stochastic",
            "paired_episode_count": len(evaluations["full"]["episode_returns"]),
            "shared_environment_seed_start": 10_000,
            "shared_action_sampling_seed": 10_000,
            "bootstrap_resamples": 20_000,
        },
        "returns": {
            name: {
                "mean": round(float(evaluation["mean_return"]), 2),
                "std": round(float(evaluation["std_return"]), 2),
            }
            for name, evaluation in evaluations.items()
        },
        "full_comparisons": full_comparisons,
        "incremental_comparisons": incremental_comparisons,
        "action_type_proportions": action_proportions,
        "action_type_total_variation_from_full": action_tv_from_full,
    }
    json_path = seed_dir / "feature_group_gate_statistics.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return_rows = "\n".join(
        f"| `{name}` | {values['mean']:.2f} | {values['std']:.2f} | "
        f"{action_tv_from_full[name]:.4f} |"
        for name, values in summary["returns"].items()
    )
    comparison_rows = "\n".join(
        f"| `{name}` | {_format_ci(values)} | "
        f"{values['wins_ties_losses'][0]}/{values['wins_ties_losses'][1]}/"
        f"{values['wins_ties_losses'][2]} |"
        for name, values in incremental_comparisons.items()
    )
    markdown = f"""# Canonical V1 Feature-Group Gate Summary

## Protocol

- Policy: `{summary['protocol']['policy_type']}`.
- Checkpoint: `checkpoint.pt` after
  {summary['protocol']['training_simulator_steps']} requested simulator steps.
- Evaluation: stochastic policy sampling, 30 paired held-out episodes.
- Environment seeds: 10000-10029; action-sampling seed: 10000.
- Uncertainty: 20,000 paired bootstrap resamples.
- `topology_only` is not pure adjacency. It retains structural adjacency, host
  membership, and static node identity/type/existence/control features.
- No-action-mask gates replace the mask with all-valid entries. Padded action
  indices can therefore be converted to `Sleep` by the fixed-action wrapper.

## Returns

| Gate | Mean return | Std | Action-type TV from full |
|---|---:|---:|---:|
{return_rows}

## Incremental Paired Comparisons

Positive deltas favor the feature group named before `given`.

| Comparison | Mean delta [paired bootstrap 95% CI] | W/T/L |
|---|---:|---:|
{comparison_rows}

## Interpretation Boundary

These are evaluation-time interventions on one trained seed. They identify
feature groups that are sufficient or behaviorally influential for this
checkpoint, but they do not establish unique causal attribution because the
groups can interact nonlinearly. Replication across training seeds remains
necessary before making a paper-level claim.
"""
    markdown_path = seed_dir / "FEATURE_GROUP_GATE_SUMMARY.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"statistics={json_path}")
    print(f"summary={markdown_path}")
    for name, values in incremental_comparisons.items():
        print(f"{name}: {_format_ci(values)}")


if __name__ == "__main__":
    main()
