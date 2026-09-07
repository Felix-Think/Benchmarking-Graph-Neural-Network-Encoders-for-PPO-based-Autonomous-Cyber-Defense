import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch

from acd_benchmark.cyber.cc4_flat_env import CC4FlatEnv
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.ppo.evaluation import (
    evaluate_policy_modes,
    load_policy_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Flat PPO checkpoint with deterministic and stochastic "
            "actions on identical CAGE 4 episode seeds."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/flat_cc4"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--stochastic-action-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="evaluation",
        help="Prefix for the three evaluation JSON files.",
    )
    return parser


def run_evaluation(
    args: argparse.Namespace,
    env_factory: Callable[..., object] = CC4FlatEnv,
    policy_factory: Callable[[dict], torch.nn.Module] | None = None,
) -> None:
    run_dir = args.run_dir / f"seed_{args.seed}"
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    checkpoint_path = args.checkpoint_path or run_dir / "checkpoint.pt"

    if policy_factory is None:
        policy = FlatActorCritic(
            observation_dim=int(config["observation_dim"]),
            num_actions=int(config["num_actions"]),
            encoder_hidden_dim=int(config["encoder_hidden_dim"]),
            latent_dim=int(config["latent_dim"]),
            head_hidden_dim=int(config["head_hidden_dim"]),
        )
    else:
        policy = policy_factory(config)
    load_policy_checkpoint(
        policy,
        checkpoint_path,
        device=args.device,
    )

    deterministic_env = env_factory(
        episode_steps=int(config["episode_steps"]),
        seed=args.eval_seed,
    )
    stochastic_env = env_factory(
        episode_steps=int(config["episode_steps"]),
        seed=args.eval_seed,
    )
    comparison = evaluate_policy_modes(
        deterministic_env,
        stochastic_env,
        policy,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        stochastic_action_seed=args.stochastic_action_seed,
    )

    output_files = {
        "deterministic": run_dir / f"{args.output_prefix}_deterministic.json",
        "stochastic": run_dir / f"{args.output_prefix}_stochastic.json",
        "comparison": run_dir / f"{args.output_prefix}_comparison.json",
    }
    output_files["deterministic"].write_text(
        json.dumps(comparison.deterministic.to_dict(), indent=2),
        encoding="utf-8",
    )
    output_files["stochastic"].write_text(
        json.dumps(comparison.stochastic.to_dict(), indent=2),
        encoding="utf-8",
    )
    output_files["comparison"].write_text(
        json.dumps(comparison.to_dict(), indent=2),
        encoding="utf-8",
    )

    print(
        "deterministic "
        f"mean_return={comparison.deterministic.mean_return:.2f} "
        f"std_return={comparison.deterministic.std_return:.2f}"
    )
    print(
        "stochastic "
        f"mean_return={comparison.stochastic.mean_return:.2f} "
        f"std_return={comparison.stochastic.std_return:.2f}"
    )
    print(
        "paired_return_difference "
        "definition=stochastic-deterministic "
        f"mean={comparison.mean_paired_return_difference:.2f} "
        f"std={comparison.std_paired_return_difference:.2f}"
    )
    for name, path in output_files.items():
        print(f"{name}_file={path}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_evaluation(args)


if __name__ == "__main__":
    main()
