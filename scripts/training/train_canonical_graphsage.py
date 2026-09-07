import argparse
from pathlib import Path

from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.cyber.canonical_observation_condition import (
    CanonicalObservationCondition,
)
from acd_benchmark.rl.policies.graphsage import GraphSAGEActorCritic
from scripts.training.train_canonical_flat import build_parser as build_flat_parser
from scripts.training.train_flat import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = build_flat_parser()
    parser.description = (
        "Train a shared GraphSAGE-PPO policy on the matched canonical CAGE 4 "
        "observation."
    )
    parser.add_argument("--sage-hidden-dim", type=int, default=28)
    parser.add_argument("--sage-layers", type=int, default=2)
    parser.set_defaults(
        run_dir=Path("runs/canonical_graphsage_cc4"),
        policy_type="graphsage",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    def env_factory(episode_steps: int, seed: int | None = None):
        base_env = CC4CanonicalFlatEnv(
            episode_steps=episode_steps,
            seed=seed,
            canonical_variant=args.canonical_variant,
            evidence_age_horizon=args.evidence_age_horizon,
            environment_distribution=args.environment_distribution,
            canonical_schema=args.canonical_schema,
            passive_memory_ttl=args.passive_memory_ttl,
            analysis_memory_ttl=args.analysis_memory_ttl,
            semantic_action_mask=args.semantic_action_mask,
        )
        return CanonicalObservationCondition(
            base_env,
            condition=args.observation_condition,
        )

    def policy_factory(
        observation_dim: int,
        num_actions: int,
        _: argparse.Namespace,
    ) -> GraphSAGEActorCritic:
        return GraphSAGEActorCritic(
            observation_dim=observation_dim,
            num_actions=num_actions,
            sage_hidden_dim=args.sage_hidden_dim,
            sage_layers=args.sage_layers,
            latent_dim=args.latent_dim,
            head_hidden_dim=args.head_hidden_dim,
            canonical_schema=args.canonical_schema,
        )

    run_training(
        args,
        env_factory=env_factory,
        policy_factory=policy_factory,
    )


if __name__ == "__main__":
    main()
