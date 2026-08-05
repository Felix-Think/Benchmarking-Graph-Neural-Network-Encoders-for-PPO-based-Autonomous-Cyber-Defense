"""Deterministic policy evaluation."""

from typing import Any

import numpy as np
import torch

from acd_benchmark.rl.policies.base import ActorCriticPolicy


def evaluate_policy(
    env: Any,
    policy: ActorCriticPolicy,
    *,
    episodes: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    """Evaluate a learned policy with deterministic argmax actions."""
    policy.eval()
    episode_returns: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed * 1000 + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        while not terminated and not truncated:
            prepared = policy.prepare_observation(observation, device)
            with torch.no_grad():
                logits, _ = policy.forward_prepared(prepared)
                action = torch.argmax(logits).item()
            observation, reward, terminated, truncated, _ = env.step(int(action))
            episode_return += float(reward)
        episode_returns.append(episode_return)
    return (
        float(np.mean(episode_returns)),
        float(np.std(episode_returns)),
        episode_returns,
    )
