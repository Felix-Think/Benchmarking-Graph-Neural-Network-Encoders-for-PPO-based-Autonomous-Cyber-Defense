from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.cyber.cc4_distribution import VALID_ENVIRONMENT_DISTRIBUTIONS
from acd_benchmark.cyber.dynamic_host_observation_mask import (
    VALID_MASK_PROCESSES,
    FixedBudgetDynamicHostMask,
)
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.policies.gcn import build_gcn_policy_from_config
from acd_benchmark.rl.policies.graphsage import build_graphsage_policy_from_config
from acd_benchmark.rl.ppo.evaluation import evaluate_policy, load_policy_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure zero-shot PPO degradation under fixed-budget dynamic-host "
            "observation retention."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-type", choices=("flat", "gcn", "graphsage"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--stochastic-action-seed", type=int, default=10_000)
    parser.add_argument("--mask-seed", type=int, default=30_000)
    parser.add_argument(
        "--mask-process",
        choices=VALID_MASK_PROCESSES,
        default="iid",
    )
    parser.add_argument(
        "--retention-levels",
        type=float,
        nargs="+",
        default=(0.75, 0.50, 0.25),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--environment-distribution",
        choices=VALID_ENVIRONMENT_DISTRIBUTIONS,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _build_policy(config: dict, policy_type: str):
    if policy_type == "gcn":
        return build_gcn_policy_from_config(config)
    if policy_type == "graphsage":
        return build_graphsage_policy_from_config(config)
    return FlatActorCritic(
        observation_dim=int(config["observation_dim"]),
        num_actions=int(config["num_actions"]),
        encoder_hidden_dim=int(config["encoder_hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        head_hidden_dim=int(config["head_hidden_dim"]),
    )


def _retention_label(retention: float) -> str:
    return f"q{int(round(retention * 100)):03d}"


def _output_prefix(mask_process: str) -> str:
    if mask_process == "persistent":
        return "persistent_dynamic_host_retention"
    return "dynamic_host_retention"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if any(not 0.0 <= q <= 1.0 for q in args.retention_levels):
        raise ValueError("Every retention level must be in [0, 1]")

    seed_dir = args.run_dir / f"seed_{args.seed}"
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("canonical_schema", "v1") != "v1":
        raise ValueError("The current degradation protocol is defined for v1 only")

    policy_type = args.policy_type or config.get("policy_type", "flat")
    policy = _build_policy(config, policy_type)
    checkpoint_path = args.checkpoint_path or seed_dir / "checkpoint.pt"
    load_policy_checkpoint(policy, checkpoint_path, device=args.device)
    environment_distribution = (
        args.environment_distribution
        or config.get("environment_distribution", "official")
    )

    for retention in args.retention_levels:
        output_path = seed_dir / (
            f"{_output_prefix(args.mask_process)}_"
            f"{_retention_label(retention)}_stochastic.json"
        )
        if output_path.exists() and not args.overwrite:
            print(f"SKIP existing retention={retention:.2f} file={output_path}")
            continue

        base_env = CC4CanonicalFlatEnv(
            episode_steps=int(config["episode_steps"]),
            seed=args.eval_seed,
            canonical_variant=config["canonical_variant"],
            evidence_age_horizon=int(config["evidence_age_horizon"]),
            environment_distribution=environment_distribution,
            canonical_schema="v1",
            passive_memory_ttl=int(config.get("passive_memory_ttl", 20)),
            analysis_memory_ttl=int(config.get("analysis_memory_ttl", 20)),
            semantic_action_mask=bool(config.get("semantic_action_mask", False)),
        )
        env = FixedBudgetDynamicHostMask(
            base_env,
            retention=retention,
            mask_seed=args.mask_seed,
            mask_process=args.mask_process,
        )

        torch.manual_seed(args.stochastic_action_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.stochastic_action_seed)
        result = evaluate_policy(
            env,
            policy,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            deterministic=False,
        )
        record = {
            "protocol": {
                "name": (
                    "zero_shot_persistent_dynamic_host_retention_curve"
                    if args.mask_process == "persistent"
                    else "zero_shot_dynamic_host_retention_curve"
                ),
                "mask_process": env.process_name,
                "retention": retention,
                "mask_seed": args.mask_seed,
                "environment_seed": args.eval_seed,
                "stochastic_action_seed": args.stochastic_action_seed,
                "episodes": args.eval_episodes,
                "checkpoint_path": str(checkpoint_path),
                "policy_type": policy_type,
                "environment_distribution": environment_distribution,
                "semantic_action_mask": bool(
                    config.get("semantic_action_mask", False)
                ),
                "masked_content": "dynamic host-node features only",
                "preserved_content": (
                    "static node identity, topology, communication/firewall "
                    "state, global features and action mask"
                ),
            },
            "evaluation": result.to_dict(),
        }
        output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(
            f"retention={retention:.2f} mean_return={result.mean_return:.2f} "
            f"std_return={result.std_return:.2f} file={output_path}"
        )


if __name__ == "__main__":
    main()
