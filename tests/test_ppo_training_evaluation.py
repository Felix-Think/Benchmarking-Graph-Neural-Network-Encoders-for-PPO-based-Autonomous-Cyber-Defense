import math

import numpy as np
import torch

from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.ppo.evaluation import (
    evaluate_policy,
    load_policy_checkpoint,
)
from acd_benchmark.rl.ppo.training import PPOTrainingConfig, train_policy


class TinyEpisodeEnv:
    possible_agents = ("blue_0", "blue_1")

    def __init__(self, horizon: int = 3):
        self.horizon = horizon
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        observations = {
            agent: np.zeros(4, dtype=np.float32)
            for agent in self.possible_agents
        }
        infos = {
            agent: {
                "seed": seed,
                "decision_ready": True,
                "action_mask": np.ones(2, dtype=np.bool_),
            }
            for agent in self.possible_agents
        }
        return observations, infos

    def step(self, actions):
        assert set(actions) == set(self.possible_agents)
        self.step_count += 1
        observations = {
            agent: np.full(4, self.step_count, dtype=np.float32)
            for agent in self.possible_agents
        }
        rewards = {agent: 1.0 for agent in self.possible_agents}
        terminated = {agent: False for agent in self.possible_agents}
        truncated = {
            agent: self.step_count >= self.horizon
            for agent in self.possible_agents
        }
        infos = {
            agent: {
                "decision_ready": True,
                "action_mask": np.ones(2, dtype=np.bool_),
            }
            for agent in self.possible_agents
        }
        return observations, rewards, terminated, truncated, infos

    @staticmethod
    def action_labels(agent):
        return (f"{agent}: action_0", f"{agent}: action_1")


def test_train_policy_tracks_updates_episodes_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    env = TinyEpisodeEnv(horizon=3)
    policy = FlatActorCritic(
        observation_dim=4,
        num_actions=2,
        encoder_hidden_dim=8,
        latent_dim=8,
        head_hidden_dim=8,
    )
    checkpoint_path = tmp_path / "flat_checkpoint.pt"
    config = PPOTrainingConfig(
        total_simulator_steps=10,
        update_epochs=2,
        minibatch_size=2,
        checkpoint_path=str(checkpoint_path),
    )

    result = train_policy(env, policy, config)

    assert result.total_simulator_steps == 12
    assert result.total_updates == 4
    assert result.episode_returns == [3.0, 3.0, 3.0, 3.0]
    assert result.episode_lengths == [3, 3, 3, 3]
    assert [row["optimizer_steps"] for row in result.history] == [6, 6, 6, 6]
    assert checkpoint_path.is_file()

    for row in result.history:
        for name, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), name

    restored_policy = FlatActorCritic(
        observation_dim=4,
        num_actions=2,
        encoder_hidden_dim=8,
        latent_dim=8,
        head_hidden_dim=8,
    )
    checkpoint = load_policy_checkpoint(
        restored_policy,
        checkpoint_path,
    )
    assert checkpoint["total_simulator_steps"] == 12
    for trained, restored in zip(
        policy.parameters(),
        restored_policy.parameters(),
    ):
        assert torch.equal(trained, restored)


def test_train_policy_combines_multiple_complete_episodes_per_update():
    torch.manual_seed(0)
    env = TinyEpisodeEnv(horizon=3)
    policy = FlatActorCritic(
        observation_dim=4,
        num_actions=2,
        encoder_hidden_dim=8,
        latent_dim=8,
        head_hidden_dim=8,
    )
    config = PPOTrainingConfig(
        total_simulator_steps=10,
        episodes_per_update=2,
        update_epochs=1,
        minibatch_size=4,
    )

    result = train_policy(env, policy, config)

    assert result.total_simulator_steps == 12
    assert result.total_episodes == 4
    assert result.total_updates == 2
    assert [row["completed_episodes"] for row in result.history] == [2, 2]
    assert [row["policy_decisions"] for row in result.history] == [12, 12]
    assert result.history[0]["episode_return_mean"] == 3.0
    assert result.history[0]["episode_return_std"] == 0.0


def test_evaluate_policy_reports_returns_lengths_and_actions():
    torch.manual_seed(0)
    env = TinyEpisodeEnv(horizon=3)
    policy = FlatActorCritic(
        observation_dim=4,
        num_actions=2,
        encoder_hidden_dim=8,
        latent_dim=8,
        head_hidden_dim=8,
    )

    result = evaluate_policy(
        env,
        policy,
        episodes=4,
        seed=100,
        deterministic=True,
    )

    assert result.episode_returns == [3.0, 3.0, 3.0, 3.0]
    assert result.episode_lengths == [3, 3, 3, 3]
    assert result.mean_return == 3.0
    assert result.std_return == 0.0
    assert sum(result.action_counts.values()) == 24
    assert policy.training
