"""End-to-end single and vector PPO training updates."""

from typing import Any

import numpy as np
import torch

from acd_benchmark.rl.policies.base import ActorCriticPolicy
from acd_benchmark.rl.ppo.advantages import compute_gae, compute_vector_gae
from acd_benchmark.rl.ppo.config import PPOConfig
from acd_benchmark.rl.ppo.update import update_policy
from acd_benchmark.rl.rollout.collection import collect_rollout, collect_vector_rollout
from acd_benchmark.rl.rollout.types import EpisodeReturnTracker, Rollout


def train_update(
    env: Any,
    policy: ActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    observation: Any,
    config: PPOConfig,
    device: torch.device,
    *,
    episode_return_tracker: EpisodeReturnTracker | None = None,
) -> tuple[Any, dict[str, float]]:
    """Collect one single-env rollout, estimate advantages, and update PPO."""
    rollout = collect_rollout(
        env,
        policy,
        observation,
        config,
        device,
        episode_return_tracker=episode_return_tracker,
    )
    next_prepared = policy.prepare_observation(rollout.next_observation, device)
    with torch.no_grad():
        _, next_value = policy.forward_prepared(next_prepared)
    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        next_value=next_value,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    losses = update_policy(policy, optimizer, rollout, advantages, returns, config)
    losses["mean_completed_episode_return"] = (
        float(np.mean(rollout.episode_returns))
        if rollout.episode_returns
        else float("nan")
    )
    losses["completed_episodes"] = float(len(rollout.episode_returns))
    return rollout.next_observation, losses


def train_vector_update(
    envs: list[Any],
    policy: ActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    observations: list[Any],
    config: PPOConfig,
    device: torch.device,
    *,
    steps_per_env: int,
    episode_return_tracker: EpisodeReturnTracker | None = None,
) -> tuple[list[Any], dict[str, float]]:
    """Collect vector rollouts, estimate advantages, and update PPO."""
    vector_rollout = collect_vector_rollout(
        envs,
        policy,
        observations,
        steps_per_env=steps_per_env,
        device=device,
        episode_return_tracker=episode_return_tracker,
    )
    next_prepared = [
        policy.prepare_observation(observation, device)
        for observation in vector_rollout.next_observations
    ]
    with torch.no_grad():
        _, next_values = policy.forward_batch(next_prepared)
    advantages, returns = compute_vector_gae(
        rewards=vector_rollout.rewards,
        values=vector_rollout.values,
        dones=vector_rollout.dones,
        next_values=next_values,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    flat_rollout = Rollout(
        observations=vector_rollout.observations,
        actions=vector_rollout.actions.reshape(-1),
        old_log_probs=vector_rollout.old_log_probs.reshape(-1),
        values=vector_rollout.values.reshape(-1),
        rewards=vector_rollout.rewards.reshape(-1),
        dones=vector_rollout.dones.reshape(-1),
        next_observation=vector_rollout.next_observations,
        episode_returns=vector_rollout.episode_returns,
    )
    losses = update_policy(
        policy,
        optimizer,
        flat_rollout,
        advantages.reshape(-1),
        returns.reshape(-1),
        config,
    )
    losses["mean_completed_episode_return"] = (
        float(np.mean(vector_rollout.episode_returns))
        if vector_rollout.episode_returns
        else float("nan")
    )
    losses["completed_episodes"] = float(len(vector_rollout.episode_returns))
    return vector_rollout.next_observations, losses
