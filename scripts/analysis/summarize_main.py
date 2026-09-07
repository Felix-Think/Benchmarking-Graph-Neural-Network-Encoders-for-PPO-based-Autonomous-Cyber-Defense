"""Summarize final stochastic return with training seeds as the replicate unit."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


MATCHED_FIELDS = (
    "total_simulator_steps",
    "episode_steps",
    "episodes_per_update",
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
    "canonical_schema",
    "canonical_parser_version",
    "canonical_variant",
    "passive_memory_ttl",
    "analysis_memory_ttl",
    "semantic_action_mask",
    "environment_distribution",
    "observation_condition",
    "action_catalog",
    "node_embedding_dim",
    "context_dim",
    "head_hidden_dim",
    "eval_episodes",
    "eval_seed",
    "parameter_count",
)


def summarize(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    std = float(values.std(ddof=1)) if n > 1 else None
    half_width = float(t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else None
    mean = float(values.mean())
    return {
        "training_seeds": n,
        "mean": mean,
        "sample_std": std,
        "t_ci95": [mean - half_width, mean + half_width] if n > 1 else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Training seeds must be nonempty and distinct")
    rows = []
    for seed in args.seeds:
        row = {"seed": seed}
        reference_config = None
        reference_steps = None
        for encoder in ("nodemlp", "gcn"):
            directory = args.run_root / encoder / f"seed_{seed}"
            config = json.loads((directory / "config.json").read_text())
            summary = json.loads((directory / "training_summary.json").read_text())
            evaluation = json.loads(
                (directory / "evaluation_stochastic.json").read_text()
            )
            if (
                config["policy_type"] != f"structured_{encoder}"
                or int(config["seed"]) != seed
            ):
                raise ValueError(f"Incorrect run identity: {directory}")
            if evaluation["deterministic"]:
                raise ValueError(f"Expected stochastic evaluation: {directory}")
            if reference_config is not None:
                differing = [
                    key
                    for key in MATCHED_FIELDS
                    if reference_config[key] != config[key]
                ]
                if differing or summary["total_simulator_steps"] != reference_steps:
                    raise ValueError(
                        f"Unmatched pair seed={seed}: {differing}, check actual budgets"
                    )
            reference_config = config
            reference_steps = summary["total_simulator_steps"]
            row[encoder] = float(evaluation["mean_return"])
        row["gcn_minus_nodemlp"] = row["gcn"] - row["nodemlp"]
        rows.append(row)
    result = {
        "scope": "Final checkpoints, raw team return, paired training seeds; not convergence",
        "rows": rows,
        "summary": {
            key: summarize([row[key] for row in rows])
            for key in ("nodemlp", "gcn", "gcn_minus_nodemlp")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
