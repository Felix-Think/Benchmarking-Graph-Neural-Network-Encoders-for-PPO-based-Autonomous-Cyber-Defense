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
    def __init__(self, horizon: int = 3):
        self.horizon = horizon
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        observation = np.zeros((13, 4), dtype=np.float32)
        return observation, {"seed": seed}

    def step(self, action):
        self.step_count += 1
        observation = np.full(
            (13, 4),
            self.step_count,
            dtype=np.float32,
        )
        reward = 1.0
        terminated = False
        truncated = self.step_count >= self.horizon
        return observation, reward, terminated, truncated, {}


def test_train_policy_tracks_updates_episodes_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    env = TinyEpisodeEnv(horizon=3)
    policy = FlatActorCritic()
    checkpoint_path = tmp_path / "flat_checkpoint.pt"
    config = PPOTrainingConfig(
        total_timesteps=10,
        rollout_steps=4,
        update_epochs=2,
        minibatch_size=2,
        checkpoint_path=str(checkpoint_path),
    )

    result = train_policy(env, policy, config)

    assert result.total_timesteps == 10
    assert result.total_updates == 3
    assert result.episode_returns == [3.0, 3.0, 3.0]
    assert result.episode_lengths == [3, 3, 3]
    assert [row["optimizer_steps"] for row in result.history] == [4, 4, 2]
    assert checkpoint_path.is_file()

    for row in result.history:
        for name, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), name

    restored_policy = FlatActorCritic()
    checkpoint = load_policy_checkpoint(
        restored_policy,
        checkpoint_path,
    )
    assert checkpoint["total_timesteps"] == 10
    for trained, restored in zip(
        policy.parameters(),
        restored_policy.parameters(),
    ):
        assert torch.equal(trained, restored)


def test_evaluate_policy_reports_returns_lengths_and_actions():
    torch.manual_seed(0)
    env = TinyEpisodeEnv(horizon=3)
    policy = FlatActorCritic()

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
    assert sum(result.action_counts.values()) == 12
    assert policy.training
