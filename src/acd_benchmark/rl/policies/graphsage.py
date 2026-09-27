import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from acd_benchmark.cyber.observation_schema import NUM_NODES
from acd_benchmark.rl.policies.graph_edges import batched_edge_index
from acd_benchmark.rl.policies.structured_action import StructuredActionActorCritic


class StructuredGraphSAGEActorCritic(StructuredActionActorCritic):
    """Mean GraphSAGE with the same pooling, context and action heads as GCN."""

    def __init__(
        self,
        observation_dim: int,
        sage_hidden_dim: int = 64,
        sage_layers: int = 2,
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
        if sage_layers <= 0:
            raise ValueError("sage_layers must be positive")
        dimensions = [
            len(self.node_feature_names),
            *([sage_hidden_dim] * (sage_layers - 1)),
            node_embedding_dim,
        ]
        self.convolutions = nn.ModuleList(
            SAGEConv(source, target, aggr="mean", normalize=False, root_weight=True)
            for source, target in zip(dimensions[:-1], dimensions[1:], strict=True)
        )

    def encode_nodes(self, nodes, membership, structural):
        edge_index = batched_edge_index(membership, structural)
        embeddings = nodes.reshape(-1, len(self.node_feature_names))
        for convolution in self.convolutions:
            embeddings = torch.relu(convolution(embeddings, edge_index))
        return embeddings.reshape(nodes.shape[0], NUM_NODES, self.node_embedding_dim)
