"""Optimizer construction for shared PPO training."""

import torch

from acd_benchmark.rl.policies.base import ActorCriticPolicy
from acd_benchmark.rl.ppo.config import PPOConfig


def make_optimizer(
    policy: ActorCriticPolicy,
    config: PPOConfig,
) -> torch.optim.Optimizer:
    """Create the Adam optimizer shared by flat and graph PPO."""
    return torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
