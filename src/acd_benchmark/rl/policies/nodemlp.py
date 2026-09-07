import torch
from torch import nn

from acd_benchmark.rl.policies.structured_action import StructuredActionActorCritic
from acd_benchmark.rl.ppo.init_weight import init_weight


class StructuredNodeMLPActorCritic(StructuredActionActorCritic):
    """Shared per-node MLP control with no graph message passing."""

    def __init__(
        self,
        observation_dim: int,
        node_hidden_dim: int = 128,
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
        self.node_hidden_dim = node_hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(len(self.node_feature_names), node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, node_embedding_dim),
            nn.ReLU(),
        )
        self.node_encoder.apply(init_weight)

    def encode_nodes(
        self,
        nodes: torch.Tensor,
        membership: torch.Tensor,
        structural: torch.Tensor,
    ) -> torch.Tensor:
        del membership, structural
        return self.node_encoder(nodes)
