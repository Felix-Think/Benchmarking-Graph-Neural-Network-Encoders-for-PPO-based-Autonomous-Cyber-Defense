import math

import numpy as np
import pytest
import torch

from acd_benchmark.metrics.cyber import CyberMetrics
from acd_benchmark.rl.policies.actor_critic import CategoricalActorCritic
from acd_benchmark.rl.ppo.evaluation import (
    evaluate_policy,
    load_policy_checkpoint,
)
from acd_benchmark.rl.ppo.training import PPOTrainingConfig, train_policy
from dataclasses import replace
from contextlib import contextmanager
from acd_benchmark.metrics.compute import ComputeMetrics
from acd_benchmark.rl.ppo.evaluation import preserved_evaluation_rng


class TinyPolicy(CategoricalActorCritic):
    """Small test-only network; exercises PPO without the simulator."""

    def __init__(
        self,
        observation_dim,
        num_actions,
        encoder_hidden_dim,
        latent_dim,
        head_hidden_dim,
    ):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, encoder_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder_hidden_dim, latent_dim),
            torch.nn.ReLU(),
        )
        self.actor = torch.nn.Linear(latent_dim, num_actions)
        self.critic = torch.nn.Linear(latent_dim, 1)

    def forward(self, observation):
        latent = self.encoder(observation)
        return self.actor(latent), self.critic(latent)


class TinyEpisodeEnv:
    possible_agents = ("blue_0", "blue_1")

    def __init__(self, horizon: int = 3):
        self.horizon = horizon
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        self.metrics = CyberMetrics()
        self.metrics.reset({
            "test_host": {"subnet": "test_zone", "has_ot": True, "protected": True},
        })
        observations = {
            agent: np.zeros(4, dtype=np.float32) for agent in self.possible_agents
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
        self.metrics.update(0, {"test_host"}, set())
        observations = {
            agent: np.full(4, self.step_count, dtype=np.float32)
            for agent in self.possible_agents
        }
        rewards = {agent: 1.0 for agent in self.possible_agents}
        terminated = {agent: False for agent in self.possible_agents}
        truncated = {
            agent: self.step_count >= self.horizon for agent in self.possible_agents
        }
        infos = {
            agent: {
                "decision_ready": True,
                "action_mask": np.ones(2, dtype=np.bool_),
            }
            for agent in self.possible_agents
        }
        return observations, rewards, terminated, truncated, infos

    def cyber_summary(self):
        return self.metrics.summary()

    @staticmethod
    def action_labels(agent):
        return (f"{agent}: action_0", f"{agent}: action_1")


def test_resource_instrumentation_does_not_change_learning(monkeypatch):
    def policy():
        torch.manual_seed(7)
        return TinyPolicy(4, 2, 8, 8, 8)

    config = PPOTrainingConfig(total_simulator_steps=10, episodes_per_update=2, update_epochs=1, minibatch_size=4)
    measured = policy()
    with_metrics = train_policy(TinyEpisodeEnv(), measured, config)

    @contextmanager
    def unmeasured(self, stage):
        yield

    monkeypatch.setattr(ComputeMetrics, "measure", unmeasured)
    reference = policy()
    without_metrics = train_policy(TinyEpisodeEnv(), reference, config)
    assert with_metrics.episode_returns == without_metrics.episode_returns
    for left, right in zip(measured.parameters(), reference.parameters(), strict=True):
        assert torch.equal(left, right)
    for left, right in zip(with_metrics.history, without_metrics.history, strict=True):
        assert {k: v for k, v in left.items() if not k.startswith("compute/")} == {
            k: v for k, v in right.items() if not k.startswith("compute/")
        }


def test_milestones_do_not_change_training_trajectory(tmp_path):
    def policy():
        torch.manual_seed(123)
        return TinyPolicy(4, 2, 8, 8, 8)

    config = PPOTrainingConfig(
        total_simulator_steps=10, episodes_per_update=2,
        update_epochs=1, minibatch_size=4,
    )
    reference = policy()
    train_policy(TinyEpisodeEnv(), reference, config)
    observed = []

    def milestone(result, requested):
        observed.append((requested, result.total_simulator_steps))
        path = tmp_path / "milestones" / f"step_{requested}" / "checkpoint.pt"
        record = torch.load(path)
        assert record["total_simulator_steps"] == result.total_simulator_steps
        assert "optimizer_state_dict" in record
        with preserved_evaluation_rng():
            torch.rand(100)
            np.random.random(10)

    candidate = policy()
    train_policy(
        TinyEpisodeEnv(), candidate,
        replace(config, checkpoint_path=str(tmp_path / "checkpoint.pt"), checkpoint_steps=(5, 10)),
        on_milestone=milestone,
    )
    assert observed == [(5, 6), (10, 12)]
    for left, right in zip(reference.parameters(), candidate.parameters(), strict=True):
        assert torch.equal(left, right)


def test_train_policy_tracks_updates_episodes_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    env = TinyEpisodeEnv(horizon=3)
    policy = TinyPolicy(
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

    restored_policy = TinyPolicy(
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
    policy = TinyPolicy(
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
    policy = TinyPolicy(
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
    assert [row["ticks"] for row in result.episode_cyber_metrics] == result.episode_lengths


def test_evaluate_policy_rejects_mismatched_metric_ticks_and_restores_mode():
    class BadMetricsEnv(TinyEpisodeEnv):
        def cyber_summary(self):
            summary = super().cyber_summary()
            summary["ticks"] += 1
            return summary

    policy = TinyPolicy(4, 2, 8, 8, 8)
    policy.train()
    with pytest.raises(RuntimeError, match="tick count"):
        evaluate_policy(BadMetricsEnv(), policy, episodes=1)
    assert policy.training
