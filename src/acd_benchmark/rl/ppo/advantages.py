"""Generalized advantage estimation and return targets for PPO."""

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute single-environment GAE advantages and bootstrapped returns."""
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, device=rewards.device)
    running_next_value = next_value.detach()
    for index in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * running_next_value * non_terminal - values[index]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[index] = gae
        running_next_value = values[index]
    return advantages, advantages + values


def compute_vector_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independent GAE trajectories for ``[time, env]`` tensors."""
    if rewards.ndim != 2 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("Vector GAE expects matching [time, env] tensors.")
    if next_values.shape != (rewards.shape[1],):
        raise ValueError("next_values must contain one value per environment.")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)
    running_next_values = next_values.detach()
    for timestep in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[timestep]
        delta = rewards[timestep] + gamma * running_next_values * non_terminal - values[timestep]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[timestep] = gae
        running_next_values = values[timestep]
    return advantages, advantages + values
