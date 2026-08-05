"""Shared PPO hyperparameter contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters shared exactly by flat and graph PPO."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    n_steps: int = 2048
    batch_size: int = 64
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
