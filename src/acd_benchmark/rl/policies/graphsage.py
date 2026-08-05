"""GraphSAGE actor-critic policy."""

from torch_geometric.nn import SAGEConv

from acd_benchmark.rl.policies.action_head import ActionHeadSpec
from acd_benchmark.rl.policies.graph_base import OrderedGraphActorCritic


class GraphSAGEActorCritic(OrderedGraphActorCritic):
    """Two-layer GraphSAGE encoder with an ordered host readout."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        action_head: ActionHeadSpec,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            conv1=SAGEConv(node_feature_dim, hidden_dim),
            conv2=SAGEConv(hidden_dim, hidden_dim),
            hidden_dim=hidden_dim,
            action_head=action_head,
            dropout=dropout,
        )
