"""PPO objective calculation and minibatch optimization."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from acd_benchmark.rl.policies.base import ActorCriticPolicy
from acd_benchmark.rl.ppo.config import PPOConfig
from acd_benchmark.rl.rollout.types import Rollout


def update_policy(
    policy: ActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> dict[str, float]:
    """Run clipped PPO minibatch updates over one on-policy rollout."""
    policy.train()
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    rollout_size = len(rollout.actions)
    indices = np.arange(rollout_size)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []
    clip_fractions: list[float] = []
    gradient_norms: list[float] = []

    returns_variance = torch.var(returns, unbiased=False)
    if float(returns_variance.item()) > 1e-8:
        explained_variance = 1.0 - torch.var(
            returns - rollout.values,
            unbiased=False,
        ) / returns_variance
        explained_variance_value = float(explained_variance.item())
    else:
        explained_variance_value = float("nan")

    for _ in range(config.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, rollout_size, config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            batch_observations = [
                rollout.observations[index]
                for index in batch_indices
            ]
            logits, new_values = policy.forward_batch(batch_observations)
            distribution = Categorical(logits=logits)
            action_indices = torch.as_tensor(
                batch_indices,
                device=rollout.actions.device,
            )
            new_log_probs = distribution.log_prob(rollout.actions[action_indices])
            entropy = distribution.entropy().mean()

            ratio = (new_log_probs - rollout.old_log_probs[action_indices]).exp()
            log_ratio = new_log_probs - rollout.old_log_probs[action_indices]
            batch_advantages = advantages[action_indices].detach()
            unclipped_objective = ratio * batch_advantages
            clipped_objective = torch.clamp(
                ratio,
                min=1.0 - config.ppo_clip,
                max=1.0 + config.ppo_clip,
            ) * batch_advantages
            policy_loss = -torch.min(
                unclipped_objective,
                clipped_objective,
            ).mean()
            value_loss = F.smooth_l1_loss(
                new_values,
                returns[action_indices].detach(),
            )
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.max_grad_norm,
            )
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            approximate_kls.append(
                float(((ratio - 1.0) - log_ratio).mean().detach().item())
            )
            clip_fractions.append(
                float(((ratio - 1.0).abs() > config.ppo_clip).float().mean().item())
            )
            gradient_norms.append(float(gradient_norm.item()))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approximate_kls)),
        "clip_fraction": float(np.mean(clip_fractions)),
        "explained_variance": explained_variance_value,
        "gradient_norm": float(np.mean(gradient_norms)),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
