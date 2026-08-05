"""Flat MLP actor-critic policy."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from acd_benchmark.rl.policies.base import ActorCriticPolicy, PreparedObservation
from acd_benchmark.rl.policies.common import init_linear_weights


class FlatActorCritic(ActorCriticPolicy):
    """Two-layer MLP actor-critic for flattened matched observations."""

    def __init__(self, input_dim: int, hidden_dim: int, action_count: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_count)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_linear_weights)

    def prepare_observation(self, observation: Any, device: torch.device) -> torch.Tensor:
        """Convert one flat NumPy observation to a float tensor."""
        return torch.as_tensor(observation, dtype=torch.float32, device=device)

    def forward_prepared(
        self,
        observation: PreparedObservation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and value for one flat observation."""
        if not isinstance(observation, torch.Tensor):
            raise TypeError("FlatActorCritic expects one tensor observation.")
        encoded = self.encoder(observation)
        return self.actor(encoded), self.critic(encoded).squeeze(-1)

    def forward_batch(
        self,
        observations: list[PreparedObservation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorize an entire flat PPO minibatch."""
        if not all(isinstance(observation, torch.Tensor) for observation in observations):
            raise TypeError("FlatActorCritic expects tensor observations.")
        batch = torch.stack(observations)  # type: ignore[arg-type]
        encoded = self.encoder(batch)
        return self.actor(encoded), self.critic(encoded).squeeze(-1)
