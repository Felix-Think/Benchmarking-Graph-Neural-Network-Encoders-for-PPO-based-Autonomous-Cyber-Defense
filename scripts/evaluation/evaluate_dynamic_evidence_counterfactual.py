from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from acd_benchmark.cyber.cc4_canonical_env import (
    CC4CanonicalFlatEnv,
    require_current_canonical_parser,
)
from acd_benchmark.cyber.cc4_structured_action_env import CC4StructuredActionEnv
from acd_benchmark.cyber.dynamic_evidence_counterfactual import (
    COUNTERFACTUAL_TRANSFORMS,
    DynamicEvidenceCounterfactual,
)
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.policies.gcn import build_gcn_policy_from_config
from acd_benchmark.rl.policies.graphsage import build_graphsage_policy_from_config
from acd_benchmark.rl.policies.structured_action import (
    build_structured_gcn_policy_from_config,
    build_structured_nodemlp_policy_from_config,
)
from acd_benchmark.rl.ppo.evaluation import evaluate_policy, load_policy_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical-v1 dynamic-evidence counterfactuals."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--policy-type",
        choices=(
            "flat",
            "gcn",
            "graphsage",
            "structured_nodemlp",
            "structured_gcn",
        ),
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--transform", choices=COUNTERFACTUAL_TRANSFORMS, required=True)
    parser.add_argument("--time-shift-lag", type=int, default=5)
    parser.add_argument("--intervention-seed", type=int, default=40_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--stochastic-action-seed", type=int, default=10_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _build_policy(config: dict, policy_type: str):
    if policy_type == "structured_nodemlp":
        return build_structured_nodemlp_policy_from_config(config)
    if policy_type == "structured_gcn":
        return build_structured_gcn_policy_from_config(config)
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


def _output_name(transform: str, lag: int) -> str:
    if transform == "time_shift":
        return f"counterfactual_time_shift_lag{lag}_stochastic.json"
    return f"counterfactual_{transform}_stochastic.json"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seed_dir = args.run_dir / f"seed_{args.seed}"
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    require_current_canonical_parser(config)

    policy_type = args.policy_type or config.get("policy_type", "flat")
    policy = _build_policy(config, policy_type)
    checkpoint_path = args.checkpoint_path or seed_dir / "checkpoint.pt"
    load_policy_checkpoint(policy, checkpoint_path, device=args.device)
    output_path = seed_dir / _output_name(args.transform, args.time_shift_lag)
    if output_path.exists() and not args.overwrite:
        print(f"SKIP existing file={output_path}")
        return

    base_env = CC4CanonicalFlatEnv(
        episode_steps=int(config["episode_steps"]),
        seed=args.eval_seed,
        canonical_variant=config["canonical_variant"],
        evidence_age_horizon=int(config["evidence_age_horizon"]),
        environment_distribution=config["environment_distribution"],
        canonical_schema=config.get("canonical_schema", "v1"),
        passive_memory_ttl=int(config.get("passive_memory_ttl", 20)),
        analysis_memory_ttl=int(config.get("analysis_memory_ttl", 20)),
        semantic_action_mask=bool(config.get("semantic_action_mask", False)),
    )
    action_env = (
        CC4StructuredActionEnv(base_env)
        if policy_type in ("structured_nodemlp", "structured_gcn")
        else base_env
    )
    env = DynamicEvidenceCounterfactual(
        action_env,
        transform=args.transform,
        intervention_seed=args.intervention_seed,
        time_shift_lag=args.time_shift_lag,
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
            "name": "dynamic_evidence_counterfactual",
            "transform": args.transform,
            "process_name": env.process_name,
            "time_shift_lag": args.time_shift_lag,
            "intervention_seed": args.intervention_seed,
            "environment_seed": args.eval_seed,
            "stochastic_action_seed": args.stochastic_action_seed,
            "episodes": args.eval_episodes,
            "checkpoint_path": str(checkpoint_path),
            "policy_type": policy_type,
            "action_mask": "unchanged official mask",
            "canonical_schema": config.get("canonical_schema", "v1"),
            "evidence_node_features": list(env.evidence_node_features),
            "preserved_dynamic_node_features": ["decoy_active"],
        },
        "evaluation": result.to_dict(),
    }
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        f"transform={args.transform} mean_return={result.mean_return:.2f} "
        f"std_return={result.std_return:.2f} file={output_path}"
    )


if __name__ == "__main__":
    main()
