import argparse
import json
from pathlib import Path

from acd_benchmark.cyber.canonical_observation import VALID_VARIANTS
from acd_benchmark.cyber.canonical_observation_v2 import CANONICAL_SCHEMAS
from acd_benchmark.cyber.canonical_observation_condition import (
    VALID_OBSERVATION_CONDITIONS,
    CanonicalObservationCondition,
)
from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.cyber.cc4_structured_action_env import CC4StructuredActionEnv
from acd_benchmark.cyber.cc4_distribution import VALID_ENVIRONMENT_DISTRIBUTIONS
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.policies.structured_action import (
    StructuredNodeMLPActorCritic,
)
from acd_benchmark.rl.ppo.evaluation import (
    evaluate_policy,
    evaluate_policy_modes,
    evaluate_policy_with_preserved_rng,
)
from acd_benchmark.rl.ppo.training import seed_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a reproducibly initialized, untrained Flat policy on "
            "the canonical CAGE 4 observation."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/canonical_flat_cc4"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument(
        "--canonical-variant",
        choices=VALID_VARIANTS,
        default="action_aware",
    )
    parser.add_argument("--evidence-age-horizon", type=int, default=50)
    parser.add_argument("--canonical-schema", choices=CANONICAL_SCHEMAS, default="v1")
    parser.add_argument("--passive-memory-ttl", type=int, default=20)
    parser.add_argument("--analysis-memory-ttl", type=int, default=20)
    parser.add_argument("--semantic-action-mask", action="store_true")
    parser.add_argument(
        "--environment-distribution",
        choices=VALID_ENVIRONMENT_DISTRIBUTIONS,
        default="official",
    )
    parser.add_argument(
        "--observation-condition",
        choices=VALID_OBSERVATION_CONDITIONS,
        default="full",
    )
    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--policy-type",
        choices=("flat", "structured_nodemlp"),
        default="flat",
    )
    parser.add_argument("--node-hidden-dim", type=int, default=128)
    parser.add_argument("--node-embedding-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--mode",
        choices=("both", "deterministic", "stochastic"),
        default="both",
    )
    parser.add_argument("--output-prefix", type=str, default="untrained_evaluation")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seed_runtime(args.seed)

    def make_env():
        base_env = CC4CanonicalFlatEnv(
            episode_steps=args.episode_steps,
            seed=args.eval_seed,
            canonical_variant=args.canonical_variant,
            evidence_age_horizon=args.evidence_age_horizon,
            environment_distribution=args.environment_distribution,
            canonical_schema=args.canonical_schema,
            passive_memory_ttl=args.passive_memory_ttl,
            analysis_memory_ttl=args.analysis_memory_ttl,
            semantic_action_mask=args.semantic_action_mask,
        )
        action_env = (
            CC4StructuredActionEnv(base_env)
            if args.policy_type == "structured_nodemlp"
            else base_env
        )
        return CanonicalObservationCondition(
            action_env,
            condition=args.observation_condition,
        )

    deterministic_env = make_env()
    stochastic_env = make_env()
    reference_agent = deterministic_env.possible_agents[0]
    observation_dim = int(
        deterministic_env.observation_space(reference_agent).shape[0]
    )
    num_actions = int(deterministic_env.action_space(reference_agent).n)
    if args.policy_type == "structured_nodemlp":
        policy = StructuredNodeMLPActorCritic(
            observation_dim=observation_dim,
            node_hidden_dim=args.node_hidden_dim,
            node_embedding_dim=args.node_embedding_dim,
            context_dim=args.context_dim,
            head_hidden_dim=args.head_hidden_dim,
            canonical_schema=args.canonical_schema,
        )
    else:
        policy = FlatActorCritic(
            observation_dim=observation_dim,
            num_actions=num_actions,
            encoder_hidden_dim=args.encoder_hidden_dim,
            latent_dim=args.latent_dim,
            head_hidden_dim=args.head_hidden_dim,
        )
    policy = policy.to(args.device)

    output_dir = args.run_dir / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, Path] = {}
    if args.mode == "both":
        comparison = evaluate_policy_modes(
            deterministic_env,
            stochastic_env,
            policy,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            stochastic_action_seed=args.eval_seed,
        )
        output_files = {
            "deterministic": (
                output_dir / f"{args.output_prefix}_deterministic.json"
            ),
            "stochastic": output_dir / f"{args.output_prefix}_stochastic.json",
            "comparison": output_dir / f"{args.output_prefix}_comparison.json",
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
            "untrained_deterministic "
            f"mean_return={comparison.deterministic.mean_return:.2f} "
            f"std_return={comparison.deterministic.std_return:.2f}"
        )
        print(
            "untrained_stochastic "
            f"mean_return={comparison.stochastic.mean_return:.2f} "
            f"std_return={comparison.stochastic.std_return:.2f}"
        )
    else:
        deterministic = args.mode == "deterministic"
        if deterministic:
            result = evaluate_policy(
                deterministic_env,
                policy,
                episodes=args.eval_episodes,
                seed=args.eval_seed,
                deterministic=True,
            )
        else:
            result = evaluate_policy_with_preserved_rng(
                stochastic_env,
                policy,
                episodes=args.eval_episodes,
                seed=args.eval_seed,
                deterministic=False,
                action_seed=args.eval_seed,
            )
        output_files[args.mode] = (
            output_dir / f"{args.output_prefix}_{args.mode}.json"
        )
        output_files[args.mode].write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        print(
            f"untrained_{args.mode} "
            f"mean_return={result.mean_return:.2f} "
            f"std_return={result.std_return:.2f}"
        )

    for name, path in output_files.items():
        print(f"{name}_file={path}")


if __name__ == "__main__":
    main()
