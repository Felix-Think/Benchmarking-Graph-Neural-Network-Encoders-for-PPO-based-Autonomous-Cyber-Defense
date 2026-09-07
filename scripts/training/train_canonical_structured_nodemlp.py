import argparse
from pathlib import Path

from acd_benchmark.cyber.canonical_observation_condition import (
    CanonicalObservationCondition,
)
from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.cyber.cc4_structured_action_env import (
    NUM_STRUCTURED_ACTIONS,
    CC4StructuredActionEnv,
)
from acd_benchmark.rl.policies.structured_action import (
    StructuredNodeMLPActorCritic,
)
from scripts.training.train_canonical_flat import build_parser as build_flat_parser
from scripts.training.train_flat import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = build_flat_parser()
    parser.description = (
        "Train canonical PPO with a shared node/edge/global action scorer and "
        "a per-node MLP encoder without message passing."
    )
    parser.add_argument("--node-hidden-dim", type=int, default=128)
    parser.add_argument("--node-embedding-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.set_defaults(
        run_dir=Path("runs/canonical_structured_nodemlp_cc4"),
        policy_type="structured_nodemlp",
        canonical_schema="v2",
        action_catalog="host_edge_global_v1",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    def env_factory(episode_steps: int, seed: int | None = None):
        canonical_env = CC4CanonicalFlatEnv(
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
        structured_env = CC4StructuredActionEnv(canonical_env)
        return CanonicalObservationCondition(
            structured_env,
            condition=args.observation_condition,
        )

    def policy_factory(
        observation_dim: int,
        num_actions: int,
        _: argparse.Namespace,
    ) -> StructuredNodeMLPActorCritic:
        if num_actions != NUM_STRUCTURED_ACTIONS:
            raise ValueError(f"Unexpected structured action count: {num_actions}")
        return StructuredNodeMLPActorCritic(
            observation_dim=observation_dim,
            node_hidden_dim=args.node_hidden_dim,
            node_embedding_dim=args.node_embedding_dim,
            context_dim=args.context_dim,
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
