"""GCN actor-critic policy."""

from torch_geometric.nn import GCNConv

from acd_benchmark.rl.policies.action_head import ActionHeadSpec
from acd_benchmark.rl.policies.graph_base import OrderedGraphActorCritic


class GCNActorCritic(OrderedGraphActorCritic):
    """Two-layer GCN encoder with an ordered host readout."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        action_head: ActionHeadSpec,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            conv1=GCNConv(node_feature_dim, hidden_dim),
            conv2=GCNConv(hidden_dim, hidden_dim),
            hidden_dim=hidden_dim,
            action_head=action_head,
            dropout=dropout,
        )
