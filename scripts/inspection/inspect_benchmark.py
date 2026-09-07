"""Print the native -> canonical -> structured-action contract without training."""

import argparse
import numpy as np

from acd_benchmark.cyber.benchmark_env import make_benchmark_env
from acd_benchmark.cyber.observation_schema import NODE_FEATURES
from acd_benchmark.rl.policies.factory import build_policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args(argv)
    env = make_benchmark_env({}, seed=args.seed)
    observations, infos = env.reset(seed=args.seed)
    for agent in env.possible_agents:
        print(
            agent,
            "observation=",
            observations[agent].shape,
            "actions=",
            env.action_space(agent).n,
            "valid=",
            int(infos[agent]["action_mask"].sum()),
        )
    print("node_feature_count=", len(NODE_FEATURES))
    for index, name in enumerate(NODE_FEATURES):
        print(index, name)
    for encoder in ("nodemlp", "gcn"):
        policy = build_policy(
            dict(
                policy_type=f"structured_{encoder}",
                canonical_schema="v2",
                observation_dim=5185,
                num_actions=242,
                node_hidden_dim=128,
                gcn_hidden_dim=128,
                gcn_layers=2,
                node_embedding_dim=64,
                context_dim=128,
                head_hidden_dim=128,
            )
        )
        print(encoder, "parameters=", sum(p.numel() for p in policy.parameters()))
        del policy
    rng = np.random.default_rng(args.seed)
    for tick in range(args.steps):
        actions = {
            agent: int(rng.choice(np.flatnonzero(infos[agent]["action_mask"])))
            for agent in env.possible_agents
            if infos[agent]["decision_ready"]
        }
        observations, rewards, terminated, truncated, infos = env.step(actions)
        print(
            "tick=",
            tick + 1,
            "decisions=",
            len(actions),
            "team_reward=",
            rewards[env.possible_agents[0]],
        )
        if all(terminated[a] or truncated[a] for a in env.possible_agents):
            break


if __name__ == "__main__":
    main()
