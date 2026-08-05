"""Collect on-policy experience without performing PPO mathematics."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributions import Categorical

from acd_benchmark.rl.policies.base import ActorCriticPolicy, PreparedObservation
from acd_benchmark.rl.ppo.config import PPOConfig
from acd_benchmark.rl.rollout.types import EpisodeReturnTracker, Rollout, VectorRollout


def collect_rollout(
    env: Any,
    policy: ActorCriticPolicy,
    observation: Any,
    config: PPOConfig,
    device: torch.device,
    *,
    episode_return_tracker: EpisodeReturnTracker | None = None,
) -> Rollout:
    """Collect one fixed-length trajectory segment from a single environment."""
    policy.eval()
    observations: list[PreparedObservation] = []
    actions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    dones: list[float] = []
    completed_episode_returns: list[float] = []
    return_tracker = episode_return_tracker or EpisodeReturnTracker.for_envs(1)
    if len(return_tracker.running_returns) != 1:
        raise ValueError("Single-environment rollout requires one return accumulator.")
    current_observation = observation

    for _ in range(config.n_steps):
        prepared = policy.prepare_observation(current_observation, device)
        with torch.no_grad():
            logits, value = policy.forward_prepared(prepared)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

        next_observation, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = bool(terminated or truncated)
        completed_return = return_tracker.record(0, float(reward), done)

        observations.append(prepared)
        actions.append(action.detach())
        old_log_probs.append(log_prob.detach())
        values.append(value.detach())
        rewards.append(float(reward))
        dones.append(float(done))

        if done:
            if completed_return is None:
                raise RuntimeError("Done transition did not complete an episode return.")
            completed_episode_returns.append(completed_return)
            current_observation, _ = env.reset()
        else:
            current_observation = next_observation

    return Rollout(
        observations=observations,
        actions=torch.stack(actions),
        old_log_probs=torch.stack(old_log_probs),
        values=torch.stack(values),
        rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
        dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
        next_observation=current_observation,
        episode_returns=completed_episode_returns,
    )


def collect_vector_rollout(
    envs: list[Any],
    policy: ActorCriticPolicy,
    observations: list[Any],
    *,
    steps_per_env: int,
    device: torch.device,
    episode_return_tracker: EpisodeReturnTracker | None = None,
) -> VectorRollout:
    """Collect synchronous trajectory segments with batched policy inference."""
    if len(envs) == 0 or len(envs) != len(observations):
        raise ValueError("envs and observations must have the same non-zero length.")
    policy.eval()
    prepared_observations: list[PreparedObservation] = []
    actions_by_time: list[torch.Tensor] = []
    log_probs_by_time: list[torch.Tensor] = []
    values_by_time: list[torch.Tensor] = []
    rewards_by_time: list[torch.Tensor] = []
    dones_by_time: list[torch.Tensor] = []
    completed_episode_returns: list[float] = []
    return_tracker = episode_return_tracker or EpisodeReturnTracker.for_envs(len(envs))
    if len(return_tracker.running_returns) != len(envs):
        raise ValueError("Return accumulator count must match the environment count.")
    current_observations = list(observations)

    for _ in range(steps_per_env):
        prepared = [
            policy.prepare_observation(observation, device)
            for observation in current_observations
        ]
        with torch.no_grad():
            logits, values = policy.forward_batch(prepared)
            distributions = Categorical(logits=logits)
            actions = distributions.sample()
            log_probs = distributions.log_prob(actions)

        step_rewards = []
        step_dones = []
        next_observations = []
        for env_index, env in enumerate(envs):
            next_observation, reward, terminated, truncated, _ = env.step(
                int(actions[env_index].item())
            )
            done = bool(terminated or truncated)
            completed_return = return_tracker.record(env_index, float(reward), done)
            step_rewards.append(float(reward))
            step_dones.append(float(done))
            if done:
                if completed_return is None:
                    raise RuntimeError("Done transition did not complete an episode return.")
                completed_episode_returns.append(completed_return)
                next_observation, _ = env.reset()
            next_observations.append(next_observation)

        prepared_observations.extend(prepared)
        actions_by_time.append(actions.detach())
        log_probs_by_time.append(log_probs.detach())
        values_by_time.append(values.detach())
        rewards_by_time.append(
            torch.as_tensor(step_rewards, dtype=torch.float32, device=device)
        )
        dones_by_time.append(
            torch.as_tensor(step_dones, dtype=torch.float32, device=device)
        )
        current_observations = next_observations

    return VectorRollout(
        observations=prepared_observations,
        actions=torch.stack(actions_by_time),
        old_log_probs=torch.stack(log_probs_by_time),
        values=torch.stack(values_by_time),
        rewards=torch.stack(rewards_by_time),
        dones=torch.stack(dones_by_time),
        next_observations=current_observations,
        episode_returns=completed_episode_returns,
    )
