import argparse
from pathlib import Path

from acd_benchmark.cyber.cc4_canonical_env import (
    CC4CanonicalFlatEnv,
    require_current_canonical_parser,
)
from acd_benchmark.cyber.canonical_observation_condition import (
    CanonicalObservationCondition,
)
from acd_benchmark.cyber.cc4_distribution import VALID_ENVIRONMENT_DISTRIBUTIONS
from scripts.evaluation.evaluate_flat import build_parser as build_flat_parser
from scripts.evaluation.evaluate_flat import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = build_flat_parser()
    parser.description = (
        "Evaluate a canonical Flat PPO checkpoint in deterministic and "
        "stochastic modes."
    )
    parser.add_argument(
        "--environment-distribution",
        choices=VALID_ENVIRONMENT_DISTRIBUTIONS,
        default=None,
        help="Override the distribution stored in the training config.",
    )
    parser.set_defaults(run_dir=Path("runs/canonical_flat_cc4"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir / f"seed_{args.seed}"

    import json

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    require_current_canonical_parser(config)

    def env_factory(episode_steps: int, seed: int | None = None):
        base_env = CC4CanonicalFlatEnv(
            episode_steps=episode_steps,
            seed=seed,
            canonical_variant=config["canonical_variant"],
            evidence_age_horizon=int(config["evidence_age_horizon"]),
            environment_distribution=(
                args.environment_distribution
                or config.get("environment_distribution", "official")
            ),
            canonical_schema=config.get("canonical_schema", "v1"),
            passive_memory_ttl=int(config.get("passive_memory_ttl", 20)),
            analysis_memory_ttl=int(config.get("analysis_memory_ttl", 20)),
            semantic_action_mask=bool(config.get("semantic_action_mask", False)),
        )
        return CanonicalObservationCondition(
            base_env,
            condition=config.get("observation_condition", "full"),
        )

    run_evaluation(args, env_factory=env_factory)


if __name__ == "__main__":
    main()
