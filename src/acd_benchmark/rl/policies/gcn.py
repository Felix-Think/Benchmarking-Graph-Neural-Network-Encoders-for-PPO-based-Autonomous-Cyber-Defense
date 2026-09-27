import torch
from torch import nn
from torch_geometric.nn import GCNConv

from acd_benchmark.cyber.observation_schema import NUM_NODES
from acd_benchmark.rl.policies.graph_edges import batched_edge_index
from acd_benchmark.rl.policies.structured_action import StructuredActionActorCritic


class StructuredGCNActorCritic(StructuredActionActorCritic):
    """GCN encoder with the same structured action scorer as NodeMLP."""

    def __init__(
        self,
        observation_dim: int,
        gcn_hidden_dim: int = 128,
        gcn_layers: int = 2,
        node_embedding_dim: int = 64,
        context_dim: int = 128,
        head_hidden_dim: int = 128,
        canonical_schema: str = "v2",
    ):
        super().__init__(
            observation_dim=observation_dim,
            node_embedding_dim=node_embedding_dim,
            context_dim=context_dim,
            head_hidden_dim=head_hidden_dim,
            canonical_schema=canonical_schema,
        )
        if gcn_layers <= 0:
            raise ValueError("gcn_layers must be positive")
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gcn_layers = gcn_layers
        dimensions = [
            len(self.node_feature_names),
            *([gcn_hidden_dim] * (gcn_layers - 1)),
            node_embedding_dim,
        ]
        self.convolutions = nn.ModuleList(
            GCNConv(input_dim, output_dim)
            for input_dim, output_dim in zip(
                dimensions[:-1],
                dimensions[1:],
                strict=True,
            )
        )

    _batched_edge_index = staticmethod(batched_edge_index)

    def encode_nodes(
        self,
        nodes: torch.Tensor,
        membership: torch.Tensor,
        structural: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = self._batched_edge_index(membership, structural)
        embeddings = nodes.reshape(-1, len(self.node_feature_names))
        for convolution in self.convolutions:
            embeddings = torch.relu(convolution(embeddings, edge_index))
        return embeddings.reshape(
            nodes.shape[0],
            NUM_NODES,
            self.node_embedding_dim,
        )
