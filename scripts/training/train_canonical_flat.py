import argparse
from pathlib import Path

from acd_benchmark.cyber.canonical_observation import VALID_VARIANTS
from acd_benchmark.cyber.canonical_observation_v2 import CANONICAL_SCHEMAS
from acd_benchmark.cyber.canonical_observation_condition import (
    VALID_OBSERVATION_CONDITIONS,
    CanonicalObservationCondition,
)
from acd_benchmark.cyber.cc4_canonical_env import (
    CANONICAL_PARSER_VERSION,
    CC4CanonicalFlatEnv,
)
from acd_benchmark.cyber.cc4_distribution import VALID_ENVIRONMENT_DISTRIBUTIONS
from scripts.training.train_flat import build_parser as build_flat_parser
from scripts.training.train_flat import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = build_flat_parser()
    parser.description = (
        "Train the existing shared Flat PPO policy on the matched CAGE 4 "
        "canonical observation."
    )
    parser.add_argument(
        "--canonical-variant",
        choices=VALID_VARIANTS,
        default="action_aware",
    )
    parser.add_argument("--evidence-age-horizon", type=int, default=50)
    parser.add_argument(
        "--canonical-schema",
        choices=CANONICAL_SCHEMAS,
        default="v1",
    )
    parser.add_argument("--passive-memory-ttl", type=int, default=20)
    parser.add_argument("--analysis-memory-ttl", type=int, default=20)
    parser.add_argument("--semantic-action-mask", action="store_true")
    parser.add_argument(
        "--observation-condition",
        choices=VALID_OBSERVATION_CONDITIONS,
        default="full",
        help=(
            "Canonical information available during both training and "
            "evaluation. The official action mask is always retained."
        ),
    )
    parser.add_argument(
        "--environment-distribution",
        choices=VALID_ENVIRONMENT_DISTRIBUTIONS,
        default="official",
    )
    parser.set_defaults(
        run_dir=Path("runs/canonical_flat_cc4"),
        periodic_eval_mode="stochastic",
        canonical_parser_version=CANONICAL_PARSER_VERSION,
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

    run_training(args, env_factory=env_factory)


if __name__ == "__main__":
    main()
