from __future__ import annotations

import torch
import torch.nn as nn

from acd_benchmark.cyber.observation_schema import (
    CANONICAL_SCHEMA,
    GLOBAL_FEATURES,
    MAX_LOCAL_HOSTS,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
    flat_observation_dim,
)
from acd_benchmark.cyber.cc4_structured_action_env import (
    EDGE_ACTIONS,
    GLOBAL_ACTIONS,
    HOST_ACTIONS,
    NUM_STRUCTURED_ACTIONS,
    STRUCTURED_EDGE_SLOTS,
    STRUCTURED_HOST_SLOTS,
)
from acd_benchmark.rl.policies.actor_critic import CategoricalActorCritic
from acd_benchmark.rl.ppo.init_weight import init_weight


class StructuredActionActorCritic(CategoricalActorCritic):
    """Common host/edge/global action scorer for canonical encoders."""

    def __init__(
        self,
        observation_dim: int,
        node_embedding_dim: int,
        context_dim: int = 128,
        head_hidden_dim: int = 128,
        canonical_schema: str = "v2",
    ):
        super().__init__()
        if canonical_schema != CANONICAL_SCHEMA:
            raise ValueError("Legacy schemas are available on the archive branch")
        self.canonical_schema = canonical_schema
        self.node_feature_names = NODE_FEATURES
        expected_observation_dim = flat_observation_dim()
        if observation_dim != expected_observation_dim:
            raise ValueError(
                f"Expected canonical dimension {expected_observation_dim}, "
                f"received {observation_dim}"
            )
        self.observation_dim = observation_dim
        self.node_embedding_dim = node_embedding_dim
        self.context_dim = context_dim
        self.head_hidden_dim = head_hidden_dim
        self._exists_index = self.node_feature_names.index("exists")
        self._controlled_index = self.node_feature_names.index("controlled")

        self.context_encoder = nn.Sequential(
            nn.Linear(node_embedding_dim + len(GLOBAL_FEATURES), context_dim),
            nn.ReLU(),
        )
        self.host_action_head = nn.Sequential(
            nn.Linear(node_embedding_dim + context_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, len(HOST_ACTIONS)),
        )
        self.edge_action_head = nn.Sequential(
            nn.Linear(
                2 * node_embedding_dim + context_dim + 3,
                head_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, len(EDGE_ACTIONS)),
        )
        self.global_action_head = nn.Sequential(
            nn.Linear(context_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, len(GLOBAL_ACTIONS)),
        )
        self.critic = nn.Sequential(
            nn.Linear(context_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1),
        )
        self.context_encoder.apply(init_weight)
        self.host_action_head.apply(init_weight)
        self.edge_action_head.apply(init_weight)
        self.global_action_head.apply(init_weight)
        self.critic.apply(init_weight)

    def _slices(self) -> dict[str, slice]:
        node_size = NUM_NODES * len(self.node_feature_names)
        membership_size = MAX_LOCAL_HOSTS * NUM_SUBNET_NODES
        matrix_size = NUM_SUBNET_NODES * NUM_SUBNET_NODES
        membership_start = node_size
        structural_start = membership_start + membership_size
        communication_start = structural_start + matrix_size
        firewall_start = communication_start + matrix_size
        global_start = firewall_start + matrix_size
        return {
            "nodes": slice(0, node_size),
            "membership": slice(membership_start, structural_start),
            "structural": slice(structural_start, communication_start),
            "communication": slice(communication_start, firewall_start),
            "firewall": slice(firewall_start, global_start),
            "global": slice(global_start, self.observation_dim),
        }

    def _unpack(self, observation: torch.Tensor):
        slices = self._slices()
        batch_size = observation.shape[0]
        nodes = observation[:, slices["nodes"]].reshape(
            batch_size,
            NUM_NODES,
            len(self.node_feature_names),
        )
        membership = observation[:, slices["membership"]].reshape(
            batch_size,
            MAX_LOCAL_HOSTS,
            NUM_SUBNET_NODES,
        )
        matrices = {
            name: observation[:, slices[name]].reshape(
                batch_size,
                NUM_SUBNET_NODES,
                NUM_SUBNET_NODES,
            )
            for name in ("structural", "communication", "firewall")
        }
        return (
            nodes,
            membership,
            matrices["structural"],
            matrices["communication"],
            matrices["firewall"],
            observation[:, slices["global"]],
        )

    def encode_nodes(
        self,
        nodes: torch.Tensor,
        membership: torch.Tensor,
        structural: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _controlled_subnet_slots(
        nodes: torch.Tensor,
        controlled_index: int,
    ) -> torch.Tensor:
        subnet_indices = torch.arange(
            NUM_SUBNET_NODES,
            device=nodes.device,
        ).unsqueeze(0)
        controlled = nodes[:, :NUM_SUBNET_NODES, controlled_index] > 0.5 # 9 nodes đâu la subnet và lấy cột controlled_index
        # Controlled subnet indices come first in ascending canonical order.
        # Remaining entries only pad single-subnet agents and are action-masked.
        scores = controlled.float() * 100.0 - subnet_indices.float()
        return torch.argsort(scores, dim=1, descending=True)[:, :3]

    def _edge_action_features(
        self,
        nodes: torch.Tensor,
        node_embeddings: torch.Tensor,
        context: torch.Tensor,
        structural: torch.Tensor,
        communication: torch.Tensor,
        firewall: torch.Tensor,
    ) -> torch.Tensor:
        controlled_slots = self._controlled_subnet_slots(
            nodes,
            self._controlled_index,
        )
        batch_size = nodes.shape[0]
        subnet_indices = torch.arange(
            NUM_SUBNET_NODES,
            device=nodes.device,
        ).view(1, 1, NUM_SUBNET_NODES)
        all_sources = subnet_indices.expand(batch_size, 3, -1)
        destination_grid = controlled_slots.unsqueeze(-1).expand(-1, -1, 8)
        sources = all_sources[all_sources != controlled_slots.unsqueeze(-1)].reshape(
            batch_size, STRUCTURED_EDGE_SLOTS
        )
        destinations = destination_grid.reshape(
            batch_size,
            STRUCTURED_EDGE_SLOTS,
        )
        batch_indices = torch.arange(
            batch_size,
            device=nodes.device,
        ).unsqueeze(1)

        source_embeddings = node_embeddings[batch_indices, sources]
        destination_embeddings = node_embeddings[batch_indices, destinations]
        relations = torch.stack(
            (
                structural[batch_indices, sources, destinations],
                communication[batch_indices, sources, destinations],
                firewall[batch_indices, sources, destinations],
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                source_embeddings,
                destination_embeddings,
                context.unsqueeze(1).expand(-1, STRUCTURED_EDGE_SLOTS, -1),
                relations,
            ),
            dim=-1,
        )
        if features.shape[1] != STRUCTURED_EDGE_SLOTS:
            raise AssertionError(f"Unexpected edge slots: {features.shape}")
        return features

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                "Structured canonical observation must have shape "
                f"(batch, {self.observation_dim}), received "
                f"{tuple(observation.shape)}"
            )

        observation = observation.float()
        (
            nodes,
            membership,
            structural,
            communication,
            firewall,
            global_features,
        ) = self._unpack(observation)
        node_embeddings = self.encode_nodes(nodes, membership, structural)
        exists = nodes[:, :, self._exists_index].unsqueeze(-1)
        pooled = (node_embeddings * exists).sum(dim=1) / exists.sum(dim=1).clamp_min( # nhan exist là broadcast để lọc những node ko tồn tài (exist = 0) trong obs này
            1.0
        ) # masked mean pooling
        context = self.context_encoder(torch.cat((pooled, global_features), dim=-1))

        host_embeddings = node_embeddings[
            :,
            NUM_SUBNET_NODES : NUM_SUBNET_NODES + STRUCTURED_HOST_SLOTS,
        ]
        host_context = context.unsqueeze(1).expand(
            -1,
            STRUCTURED_HOST_SLOTS,
            -1,
        )
        host_logits = self.host_action_head(
            torch.cat((host_embeddings, host_context), dim=-1)
        )
        host_logits = host_logits.transpose(1, 2).reshape(
            observation.shape[0],
            -1,
        )

        edge_features = self._edge_action_features(
            nodes,
            node_embeddings,
            context,
            structural,
            communication,
            firewall,
        )
        edge_logits = self.edge_action_head(edge_features)
        edge_logits = edge_logits.transpose(1, 2).reshape(
            observation.shape[0],
            -1,
        )
        global_logits = self.global_action_head(context)
        logits = torch.cat((host_logits, edge_logits, global_logits), dim=-1)
        if logits.shape[-1] != NUM_STRUCTURED_ACTIONS:
            raise AssertionError(f"Unexpected action logits: {logits.shape}")
        return logits, self.critic(context)
