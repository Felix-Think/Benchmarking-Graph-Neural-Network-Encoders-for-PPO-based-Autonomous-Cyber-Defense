from collections.abc import Mapping

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from acd_benchmark.cyber.canonical_observation import (
    GLOBAL_FEATURES,
    MAX_LOCAL_HOSTS,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
    flat_observation_dim,
)
from acd_benchmark.cyber.canonical_observation_v2 import (
    CANONICAL_SCHEMAS,
    V2_NODE_FEATURES,
    v2_flat_observation_dim,
)
from acd_benchmark.rl.policies.actor_critic import ActorCritic
from acd_benchmark.rl.ppo.init_weight import init_weight


class GCNActorCritic(ActorCritic):
    """GCN encoder for the fixed-size canonical CAGE 4 representation.

    Host membership and physical subnet adjacency define message-passing edges.
    Node embeddings remain ordered before projection because the current fixed
    action catalogue contains host-specific action logits. Communication policy,
    firewall state and global features are retained as graph-level context.
    """

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        gcn_hidden_dim: int = 32,
        gcn_layers: int = 2,
        latent_dim: int = 128,
        head_hidden_dim: int = 128,
        canonical_schema: str = "v1",
    ):
        if canonical_schema not in CANONICAL_SCHEMAS:
            raise ValueError(f"canonical_schema must be one of {CANONICAL_SCHEMAS}")
        node_feature_names = (
            NODE_FEATURES if canonical_schema == "v1" else V2_NODE_FEATURES
        )
        expected_observation_dim = (
            flat_observation_dim()
            if canonical_schema == "v1"
            else v2_flat_observation_dim()
        )
        if observation_dim != expected_observation_dim:
            raise ValueError(
                f"GCN requires canonical dimension {expected_observation_dim}, "
                f"received {observation_dim}"
            )
        if gcn_hidden_dim <= 0:
            raise ValueError("gcn_hidden_dim must be positive")
        if gcn_layers <= 0:
            raise ValueError("gcn_layers must be positive")

        super().__init__(
            latent_dim=latent_dim,
            num_actions=num_actions,
            head_hidden_dim=head_hidden_dim,
        )
        self.observation_dim = observation_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gcn_layers = gcn_layers
        self.canonical_schema = canonical_schema
        self.node_feature_names = node_feature_names

        convolutions = [GCNConv(len(node_feature_names), gcn_hidden_dim)]
        convolutions.extend(
            GCNConv(gcn_hidden_dim, gcn_hidden_dim)
            for _ in range(gcn_layers - 1)
        )
        self.convolutions = nn.ModuleList(convolutions)

        context_dim = (
            2 * NUM_SUBNET_NODES * NUM_SUBNET_NODES
            + len(GLOBAL_FEATURES)
        )
        self.encoder_projection = nn.Sequential(
            nn.Linear(NUM_NODES * gcn_hidden_dim + context_dim, latent_dim),
            nn.ReLU(),
        )
        self.encoder_projection.apply(init_weight)

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
        structural = observation[:, slices["structural"]].reshape(
            batch_size,
            NUM_SUBNET_NODES,
            NUM_SUBNET_NODES,
        )
        context = torch.cat(
            (
                observation[:, slices["communication"]],
                observation[:, slices["firewall"]],
                observation[:, slices["global"]],
            ),
            dim=-1,
        )
        return nodes, membership, structural, context

    @staticmethod
    def _batched_edge_index(
        membership: torch.Tensor,
        structural: torch.Tensor,
    ) -> torch.Tensor:
        structural_edges = (structural > 0).nonzero(as_tuple=False)
        membership_edges = (membership > 0).nonzero(as_tuple=False)
        sources: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []

        if structural_edges.numel() > 0:
            offsets = structural_edges[:, 0] * NUM_NODES
            sources.append(offsets + structural_edges[:, 1])
            destinations.append(offsets + structural_edges[:, 2])

        if membership_edges.numel() > 0:
            offsets = membership_edges[:, 0] * NUM_NODES
            host_nodes = offsets + NUM_SUBNET_NODES + membership_edges[:, 1]
            subnet_nodes = offsets + membership_edges[:, 2]
            sources.extend((host_nodes, subnet_nodes))
            destinations.extend((subnet_nodes, host_nodes))

        if not sources:
            return torch.empty(
                (2, 0),
                dtype=torch.long,
                device=membership.device,
            )
        return torch.stack(
            (torch.cat(sources), torch.cat(destinations)),
            dim=0,
        ).long()

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2:
            raise ValueError(
                "GCN observation must have shape (observation_dim,) or "
                f"(batch, observation_dim), received {tuple(observation.shape)}"
            )
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"received {observation.shape[-1]}"
            )

        observation = observation.float()
        nodes, membership, structural, context = self._unpack(observation)
        edge_index = self._batched_edge_index(membership, structural)
        node_embeddings = nodes.reshape(-1, len(self.node_feature_names))
        for convolution in self.convolutions:
            node_embeddings = torch.relu(
                convolution(node_embeddings, edge_index)
            )
        node_embeddings = node_embeddings.reshape(
            observation.shape[0],
            NUM_NODES * self.gcn_hidden_dim,
        )
        return self.encoder_projection(
            torch.cat((node_embeddings, context), dim=-1)
        )


def build_gcn_policy_from_config(config: Mapping[str, object]) -> GCNActorCritic:
    return GCNActorCritic(
        observation_dim=int(config["observation_dim"]),
        num_actions=int(config["num_actions"]),
        gcn_hidden_dim=int(config["gcn_hidden_dim"]),
        gcn_layers=int(config["gcn_layers"]),
        latent_dim=int(config["latent_dim"]),
        head_hidden_dim=int(config["head_hidden_dim"]),
        canonical_schema=str(config.get("canonical_schema", "v1")),
    )
