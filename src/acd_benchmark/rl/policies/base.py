"""Common actor-critic interface used by rollout and PPO code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


PreparedObservation = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class ActorCriticPolicy(nn.Module, ABC):
    """Interface that lets one PPO implementation train any encoder."""

    @abstractmethod
    def prepare_observation(
        self,
        observation: Any,
        device: torch.device,
    ) -> PreparedObservation:
        """Convert one environment observation to device tensors."""

    @abstractmethod
    def forward_prepared(
        self,
        observation: PreparedObservation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one action-logit vector and one scalar value."""

    def forward_batch(
        self,
        observations: list[PreparedObservation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return batched logits and values; subclasses may vectorize this."""
        outputs = [self.forward_prepared(observation) for observation in observations]
        return (
            torch.stack([logits for logits, _ in outputs]),
            torch.stack([value for _, value in outputs]),
        )
