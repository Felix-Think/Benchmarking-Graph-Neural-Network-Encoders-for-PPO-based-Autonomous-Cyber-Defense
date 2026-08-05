"""Shared ordered-readout actor-critic implementation for GNN encoders."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from acd_benchmark.rl.policies.action_head import ActionHeadSpec
from acd_benchmark.rl.policies.base import ActorCriticPolicy, PreparedObservation
from acd_benchmark.rl.policies.common import init_linear_weights
from acd_benchmark.wrappers.shared.host_obs import GraphObservation


class OrderedGraphActorCritic(ActorCriticPolicy):
    """Shared GNN policy whose readout preserves canonical host order."""

    def __init__(
        self,
        *,
        conv1: nn.Module,
        conv2: nn.Module,
        hidden_dim: int,
        action_head: ActionHeadSpec,
        dropout: float,
    ) -> None:
        super().__init__()
        self.action_head = action_head
        self.conv1 = conv1
        self.conv2 = conv2
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * action_head.num_hosts, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_head.action_count)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_linear_weights)

    def prepare_observation(
        self,
        observation: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert one GraphObservation to node and edge tensors."""
        if not isinstance(observation, GraphObservation):
            raise TypeError(f"{self.__class__.__name__} expects GraphObservation input.")
        return graph_to_tensors(observation, device)

    def _encode(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured two-layer message-passing encoder."""
        x = F.relu(self.conv1(node_features, edge_index))
        x = self.dropout(x)
        return F.relu(self.conv2(x, edge_index))

    def forward_prepared(
        self,
        observation: PreparedObservation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and value for one graph."""
        if not isinstance(observation, tuple):
            raise TypeError(
                f"{self.__class__.__name__} expects a (node_features, edge_index) tuple."
            )
        node_features, edge_index = observation
        x = self._encode(node_features, edge_index)
        encoded = self.output_projection(x.reshape(-1))
        return self.actor(encoded), self.critic(encoded).squeeze(-1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the one-graph call surface used by inspection and tests."""
        return self.forward_prepared((node_features, edge_index))

    def forward_batch(
        self,
        observations: list[PreparedObservation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch fixed-size graphs for one pair of message-passing passes."""
        graph_data: list[Data] = []
        for observation in observations:
            if not isinstance(observation, tuple):
                raise TypeError(f"{self.__class__.__name__} expects graph tensor tuples.")
            node_features, edge_index = observation
            if node_features.shape[0] != self.action_head.num_hosts:
                raise ValueError("Every graph must preserve the configured host count.")
            graph_data.append(Data(x=node_features, edge_index=edge_index))

        graph_batch = Batch.from_data_list(graph_data)
        x = self._encode(graph_batch.x, graph_batch.edge_index)# graph_batch.x : [Bx16, 15], graph_batch.edge_index: [2, BxE], ngoài ra còn có graph_batch.batch:[B×16]
        encoded = self.output_projection(x.reshape(len(observations), -1))
        return self.actor(encoded), self.critic(encoded).squeeze(-1)


def graph_to_tensors(
    observation: GraphObservation,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one graph observation to PyTorch tensors."""
    node_features = torch.as_tensor(
        observation.node_features,
        dtype=torch.float32,
        device=device,
    )
    edge_index = torch.as_tensor(
        observation.edge_index,
        dtype=torch.long,
        device=device,
    )
    return node_features, edge_index
