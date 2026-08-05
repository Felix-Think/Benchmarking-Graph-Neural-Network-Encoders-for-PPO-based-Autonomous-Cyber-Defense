from pathlib import Path

import numpy as np
import torch

from acd_benchmark.cyber.scalable_scenario import write_scalable_scenario
from acd_benchmark.rl.policies import (
    FlatActorCritic,
    GraphSAGEActorCritic,
    infer_action_head_spec,
)
from acd_benchmark.rl.ppo.advantages import compute_vector_gae
from acd_benchmark.rl.ppo.config import PPOConfig
from acd_benchmark.rl.ppo.optimizer import make_optimizer
from acd_benchmark.rl.ppo.training import (
    train_update,
    train_vector_update,
)
from acd_benchmark.rl.rollout import (
    EpisodeReturnTracker,
    collect_rollout,
    collect_vector_rollout,
)
from acd_benchmark.wrappers.stage1.env import Stage1FlatEnv, Stage1GraphEnv


def _scenario(tmp_path: Path) -> Path:
    return write_scalable_scenario(tmp_path / "cage2_16.yaml", 16)


class _ConstantRewardEnv:
    """Minimal environment for rollout-boundary return tests."""

    def __init__(self, episode_steps: int):
        self.episode_steps = episode_steps
        self.current_step = 0

    def reset(self, seed=None):
        del seed
        self.current_step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: int):
        del action
        self.current_step += 1
        truncated = self.current_step >= self.episode_steps
        return np.zeros(2, dtype=np.float32), 1.0, False, truncated, {}


def test_graphsage_matches_stage1_action_space_and_batches(tmp_path: Path):
    env = Stage1GraphEnv(_scenario(tmp_path), observation_probability=0.25, max_steps=5)
    observation, _ = env.reset(seed=1)
    action_head = infer_action_head_spec(env.action_specs, observation.hostnames)
    policy = GraphSAGEActorCritic(
        node_feature_dim=observation.node_features.shape[1],
        hidden_dim=16,
        action_head=action_head,
    )
    prepared = policy.prepare_observation(observation, torch.device("cpu"))
    logits, value = policy.forward_prepared(prepared)
    batch_logits, batch_values = policy.forward_batch([prepared, prepared])

    assert action_head.action_count == 178
    assert logits.shape == (178,)
    assert value.shape == ()
    assert batch_logits.shape == (2, 178)
    assert batch_values.shape == (2,)
    assert torch.allclose(batch_logits[0], logits, atol=1e-6)


def test_flat_and_graphsage_use_shared_ppo_update(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    graph_env = Stage1GraphEnv(scenario_path, observation_probability=1.0, max_steps=5)
    flat_env = Stage1FlatEnv(scenario_path, observation_probability=1.0, max_steps=5)
    graph_observation, _ = graph_env.reset(seed=2)
    flat_observation, _ = flat_env.reset(seed=2)
    action_head = infer_action_head_spec(graph_env.action_specs, graph_observation.hostnames)
    graph_policy = GraphSAGEActorCritic(
        node_feature_dim=graph_observation.node_features.shape[1],
        hidden_dim=16,
        action_head=action_head,
    )
    flat_policy = FlatActorCritic(
        input_dim=flat_observation.shape[0],
        hidden_dim=16,
        action_count=flat_env.action_count,
    )
    config = PPOConfig(n_steps=4, batch_size=2, ppo_epochs=1)

    _, graph_losses = train_update(
        graph_env,
        graph_policy,
        make_optimizer(graph_policy, config),
        graph_observation,
        config,
        torch.device("cpu"),
    )
    _, flat_losses = train_update(
        flat_env,
        flat_policy,
        make_optimizer(flat_policy, config),
        flat_observation,
        config,
        torch.device("cpu"),
    )

    expected = {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "gradient_norm",
        "learning_rate",
        "mean_completed_episode_return",
    }
    assert set(graph_losses) >= expected
    assert set(flat_losses) >= expected


def test_vector_gae_keeps_environment_trajectories_separate():
    rewards = torch.tensor([[1.0, 10.0], [1.0, 10.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    advantages, _ = compute_vector_gae(
        rewards,
        values,
        dones,
        next_values=torch.zeros(2),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.equal(advantages[:, 0], torch.tensor([2.0, 1.0]))
    assert torch.equal(advantages[:, 1], torch.tensor([10.0, 10.0]))


def test_two_environment_vector_update_runs(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    envs = [
        Stage1FlatEnv(
            scenario_path,
            observation_probability=0.25,
            max_steps=3,
        )
        for _ in range(2)
    ]
    observations = [env.reset(seed=10 + index)[0] for index, env in enumerate(envs)]
    policy = FlatActorCritic(
        input_dim=observations[0].shape[0],
        hidden_dim=16,
        action_count=envs[0].action_count,
    )
    config = PPOConfig(n_steps=2, batch_size=2, ppo_epochs=1)
    optimizer = make_optimizer(policy, config)

    next_observations, losses = train_vector_update(
        envs,
        policy,
        optimizer,
        observations,
        config,
        torch.device("cpu"),
        steps_per_env=2,
    )

    assert len(next_observations) == 2
    assert all(observation.shape == (496,) for observation in next_observations)
    assert set(losses) >= {"policy_loss", "value_loss", "entropy"}


def test_single_rollout_preserves_episode_return_across_segments():
    env = _ConstantRewardEnv(episode_steps=3)
    observation, _ = env.reset()
    policy = FlatActorCritic(input_dim=2, hidden_dim=4, action_count=1)
    config = PPOConfig(n_steps=2, batch_size=2, ppo_epochs=1)
    tracker = EpisodeReturnTracker.for_envs(1)

    first = collect_rollout(
        env,
        policy,
        observation,
        config,
        torch.device("cpu"),
        episode_return_tracker=tracker,
    )
    second = collect_rollout(
        env,
        policy,
        first.next_observation,
        config,
        torch.device("cpu"),
        episode_return_tracker=tracker,
    )

    assert first.episode_returns == []
    assert second.episode_returns == [3.0]
    assert tracker.running_returns == [1.0]


def test_vector_rollout_preserves_each_environment_return_across_segments():
    envs = [_ConstantRewardEnv(episode_steps=3), _ConstantRewardEnv(episode_steps=4)]
    observations = [env.reset()[0] for env in envs]
    policy = FlatActorCritic(input_dim=2, hidden_dim=4, action_count=1)
    tracker = EpisodeReturnTracker.for_envs(2)

    first = collect_vector_rollout(
        envs,
        policy,
        observations,
        steps_per_env=2,
        device=torch.device("cpu"),
        episode_return_tracker=tracker,
    )
    second = collect_vector_rollout(
        envs,
        policy,
        first.next_observations,
        steps_per_env=2,
        device=torch.device("cpu"),
        episode_return_tracker=tracker,
    )

    assert first.episode_returns == []
    assert second.episode_returns == [3.0, 4.0]
    assert tracker.running_returns == [1.0, 0.0]
