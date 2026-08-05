"""Re-evaluate one Stage 1 checkpoint from its frozen run configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import torch  # noqa: E402
import yaml  # noqa: E402

from acd_benchmark.evaluation.operational_metrics import (  # noqa: E402
    evaluate_policy_with_operational_metrics,
)
from acd_benchmark.rl.runtime import resolve_device  # noqa: E402
from acd_benchmark.rl.stage1_factory import (  # noqa: E402
    build_stage1_policy,
    make_stage1_env,
    resolve_model_hidden_dim,
)


def main() -> None:
    """Load, reconstruct, and evaluate a saved policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(
        (args.run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    run = config["run"]
    scenario = config["scenario"]
    training = config["training"]
    device = resolve_device(str(config["compute"]["device"]))
    env = make_stage1_env(
        str(run["model"]),
        scenario_path=args.run_dir / "scenario.yaml",
        node_count=int(scenario["node_count"]),
        observation_probability=float(run["observation_probability"]),
        max_steps=int(scenario["max_episode_steps"]),
    )
    observation, _ = env.reset(seed=int(run["seed"]))
    policy = build_stage1_policy(
        str(run["model"]),
        observation,
        env,
        hidden_dim=resolve_model_hidden_dim(str(run["model"]), config["model"]),
    ).to(device)
    checkpoint_path = args.checkpoint or args.run_dir / "checkpoints" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    result = evaluate_policy_with_operational_metrics(
        env,
        policy,
        episodes=args.episodes
        or int(training.get("final_evaluation_episodes", training["evaluation_episodes"])),
        seed=args.seed
        if args.seed is not None
        else int(
            training.get(
                "final_evaluation_base_seed",
                training["evaluation_base_seed"],
            )
        ),
        device=device,
    )
    output = args.output or args.run_dir / "reevaluation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved reevaluation to {output}")
    print(
        f"episode_return={result['mean_episode_return']:.4f} "
        f"availability={result['mean_mission_availability']:.4f}"
    )


if __name__ == "__main__":
    main()
