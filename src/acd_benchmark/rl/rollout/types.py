"""Data containers produced by on-policy rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from acd_benchmark.rl.policies.base import PreparedObservation


@dataclass
class EpisodeReturnTracker:
    """Accumulate complete episode returns across rollout boundaries.

    PPO rollouts are fixed-length trajectory segments and do not generally end
    on an episode boundary. Keeping this state outside one ``Rollout`` prevents
    the training log from dropping rewards collected in a previous segment.
    """

    running_returns: list[float]

    @classmethod
    def for_envs(cls, env_count: int) -> "EpisodeReturnTracker":
        """Create zeroed accumulators for one or more environments."""
        if env_count <= 0:
            raise ValueError("env_count must be positive.")
        return cls(running_returns=[0.0] * env_count)

    def record(self, env_index: int, reward: float, done: bool) -> float | None:
        """Record one transition and return a completed episode total, if any."""
        if not 0 <= env_index < len(self.running_returns):
            raise IndexError(f"Environment index out of range: {env_index}")
        self.running_returns[env_index] += float(reward)
        if not done:
            return None
        completed_return = self.running_returns[env_index]
        self.running_returns[env_index] = 0.0 # reset if Done
        return completed_return


@dataclass
class Rollout:
    """One single-environment trajectory segment consumed by PPO."""

    observations: list[PreparedObservation]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_observation: Any
    episode_returns: list[float]


@dataclass
class VectorRollout:
    """Time-major trajectory segments from synchronous environments."""

    observations: list[PreparedObservation]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_observations: list[Any]
    episode_returns: list[float]
