import argparse
import json
from pathlib import Path

import numpy as np
import torch

from acd_benchmark.cyber.canonical_observation import (
    GLOBAL_FEATURES,
    MAX_LOCAL_HOSTS,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
)
from acd_benchmark.cyber.canonical_observation_condition import (
    VALID_OBSERVATION_CONDITIONS,
    CanonicalObservationCondition,
)
from acd_benchmark.cyber.cc4_canonical_env import (
    CC4CanonicalFlatEnv,
    require_current_canonical_parser,
)
from acd_benchmark.cyber.cc4_distribution import VALID_ENVIRONMENT_DISTRIBUTIONS
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.policies.gcn import build_gcn_policy_from_config
from acd_benchmark.rl.policies.graphsage import (
    build_graphsage_policy_from_config,
)
from acd_benchmark.rl.ppo.evaluation import evaluate_policy, load_policy_checkpoint


DYNAMIC_NODE_FEATURES = (
    "observed_now",
    "process_event_now",
    "connection_event_now",
    "analysis_result_now",
    "max_file_density_now",
    "process_event_memory",
    "connection_event_memory",
    "analysis_memory",
    "max_file_density_memory",
    "decoy_active",
    "evidence_age_normalized",
)

OBSERVATION_TRANSFORMS = (
    "static_only",
    "zero_all",
    "zero_all_no_action_mask",
    "mission_only",
    "node_identity_only",
    "structure_only",
    "topology_only",
    "topology_no_adjacency",
    "topology_mission",
    "full_no_action_mask",
)
ACTION_MASK_ABLATIONS = (
    "zero_all_no_action_mask",
    "full_no_action_mask",
)


class CanonicalObservationGate:
    def __init__(self, env: CC4CanonicalFlatEnv, transform: str):
        self.env = env
        self.transform = transform
        self.possible_agents = env.possible_agents
        self.node_feature_names = getattr(env, "node_feature_names", NODE_FEATURES)
        self.node_feature_index = {
            name: index for index, name in enumerate(self.node_feature_names)
        }
        static_features = {
            "exists",
            "is_subnet",
            "is_host",
            "is_user",
            "is_server",
            "controlled",
        }
        self.dynamic_node_features = tuple(
            name for name in self.node_feature_names if name not in static_features
        )

    def _transform(self, observations: dict[str, np.ndarray]):
        transformed = {}
        for agent, observation in observations.items():
            observation = observation.copy()
            mission_phase = observation[-len(GLOBAL_FEATURES) : -4].copy()
            node_size = NUM_NODES * len(self.node_feature_names)
            membership_size = MAX_LOCAL_HOSTS * NUM_SUBNET_NODES
            matrix_size = NUM_SUBNET_NODES * NUM_SUBNET_NODES
            membership_start = node_size
            structural_start = membership_start + membership_size
            communication_policy_start = structural_start + matrix_size
            if self.transform in ("zero_all", "zero_all_no_action_mask"):
                observation.fill(0.0)
            elif self.transform == "mission_only":
                observation.fill(0.0)
                observation[-len(GLOBAL_FEATURES) : -4] = mission_phase
            elif self.transform == "node_identity_only":
                nodes = observation[:node_size].reshape(
                    NUM_NODES, len(self.node_feature_names)
                )
                nodes[
                    :,
                    [
                        self.node_feature_index[name]
                        for name in self.dynamic_node_features
                    ],
                ] = 0.0
                observation[node_size:] = 0.0
            elif self.transform == "structure_only":
                structure = observation[
                    membership_start:communication_policy_start
                ].copy()
                observation.fill(0.0)
                observation[
                    membership_start:communication_policy_start
                ] = structure
            elif self.transform in (
                "static_only",
                "topology_only",
                "topology_no_adjacency",
                "topology_mission",
            ):
                nodes = observation[:node_size].reshape(
                    NUM_NODES, len(self.node_feature_names)
                )
                nodes[
                    :,
                    [
                        self.node_feature_index[name]
                        for name in self.dynamic_node_features
                    ],
                ] = 0.0
                if self.transform != "static_only":
                    observation[communication_policy_start:] = 0.0
                    if self.transform == "topology_no_adjacency":
                        observation[
                            structural_start:communication_policy_start
                        ] = 0.0
                    if self.transform == "topology_mission":
                        observation[-len(GLOBAL_FEATURES) : -4] = mission_phase
            elif self.transform == "full_no_action_mask":
                pass
            else:
                raise ValueError(f"Unknown transform: {self.transform}")
            transformed[agent] = observation
        return transformed

    def _transform_infos(self, infos: dict[str, dict]) -> dict[str, dict]:
        if self.transform not in ACTION_MASK_ABLATIONS:
            return infos
        transformed = {}
        for agent, info in infos.items():
            info = dict(info)
            action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
            info["action_mask"] = np.ones_like(action_mask)
            transformed[agent] = info
        return transformed

    def reset(self, seed: int | None = None):
        observations, infos = self.env.reset(seed=seed)
        return self._transform(observations), self._transform_infos(infos)

    def step(self, actions: dict[str, int]):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return (
            self._transform(observations),
            rewards,
            terminated,
            truncated,
            self._transform_infos(infos),
        )

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_labels(self, agent: str):
        return self.env.action_labels(agent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a canonical Flat policy uses observations."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--stochastic-action-seed", type=int, default=10_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--policy-type",
        choices=("flat", "gcn", "graphsage"),
        default=None,
        help="Defaults to policy_type stored in the run config.",
    )
    parser.add_argument(
        "--transform",
        choices=(
            *OBSERVATION_TRANSFORMS,
            *VALID_OBSERVATION_CONDITIONS,
            "all",
            "qualification_all",
        ),
        default="static_only",
    )
    parser.add_argument(
        "--mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
    )
    parser.add_argument(
        "--environment-distribution",
        choices=VALID_ENVIRONMENT_DISTRIBUTIONS,
        default=None,
        help="Override the distribution stored in the training config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir / f"seed_{args.seed}"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    require_current_canonical_parser(config)
    checkpoint_path = args.checkpoint_path or run_dir / "best_checkpoint.pt"
    environment_distribution = (
        args.environment_distribution
        or config.get("environment_distribution", "official")
    )

    policy_type = args.policy_type or config.get("policy_type", "flat")
    if policy_type == "gcn":
        policy = build_gcn_policy_from_config(config)
    elif policy_type == "graphsage":
        policy = build_graphsage_policy_from_config(config)
    else:
        policy = FlatActorCritic(
            observation_dim=int(config["observation_dim"]),
            num_actions=int(config["num_actions"]),
            encoder_hidden_dim=int(config["encoder_hidden_dim"]),
            latent_dim=int(config["latent_dim"]),
            head_hidden_dim=int(config["head_hidden_dim"]),
        )
    load_policy_checkpoint(policy, checkpoint_path, device=args.device)

    deterministic = args.mode == "deterministic"
    if args.transform == "all":
        transforms = OBSERVATION_TRANSFORMS
    elif args.transform == "qualification_all":
        transforms = VALID_OBSERVATION_CONDITIONS
    else:
        transforms = (args.transform,)
    for transform in transforms:
        base_env = CC4CanonicalFlatEnv(
            episode_steps=int(config["episode_steps"]),
            seed=args.eval_seed,
            canonical_variant=config["canonical_variant"],
            evidence_age_horizon=int(config["evidence_age_horizon"]),
            environment_distribution=environment_distribution,
            canonical_schema=config.get("canonical_schema", "v1"),
            passive_memory_ttl=int(config.get("passive_memory_ttl", 20)),
            analysis_memory_ttl=int(config.get("analysis_memory_ttl", 20)),
            semantic_action_mask=bool(config.get("semantic_action_mask", False)),
        )
        if not deterministic:
            torch.manual_seed(args.stochastic_action_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.stochastic_action_seed)

        evaluation_env = (
            CanonicalObservationCondition(base_env, condition=transform)
            if transform in VALID_OBSERVATION_CONDITIONS
            else CanonicalObservationGate(base_env, transform=transform)
        )
        result = evaluate_policy(
            evaluation_env,
            policy,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            deterministic=deterministic,
        )
        distribution_suffix = (
            f"_{environment_distribution}"
            if args.environment_distribution is not None
            else ""
        )
        output_path = run_dir / (
            f"observation_gate_{transform}{distribution_suffix}_{args.mode}.json"
        )
        output_path.write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        print(
            f"transform={transform} mode={args.mode} "
            f"mean_return={result.mean_return:.2f} "
            f"std_return={result.std_return:.2f}"
        )
        print(f"output_file={output_path}")


if __name__ == "__main__":
    main()
