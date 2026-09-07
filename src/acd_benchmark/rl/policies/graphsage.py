from collections.abc import Mapping

import torch.nn as nn
from torch_geometric.nn import SAGEConv

from acd_benchmark.rl.policies.gcn import GCNActorCritic


class GraphSAGEActorCritic(GCNActorCritic):
    """GraphSAGE encoder over the matched canonical CAGE 4 graph.

    Graph construction, ordered node output, context projection and PPO heads
    are inherited from GCNActorCritic. Only the message-passing operator changes
    from GCNConv to SAGEConv, keeping the encoder comparison controlled.
    """

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        sage_hidden_dim: int = 28,
        sage_layers: int = 2,
        latent_dim: int = 128,
        head_hidden_dim: int = 128,
        canonical_schema: str = "v1",
    ):
        super().__init__(
            observation_dim=observation_dim,
            num_actions=num_actions,
            gcn_hidden_dim=sage_hidden_dim,
            gcn_layers=sage_layers,
            latent_dim=latent_dim,
            head_hidden_dim=head_hidden_dim,
            canonical_schema=canonical_schema,
        )
        self.sage_hidden_dim = sage_hidden_dim
        self.sage_layers = sage_layers
        self.convolutions = nn.ModuleList(
            [SAGEConv(len(self.node_feature_names), sage_hidden_dim)]
            + [
                SAGEConv(sage_hidden_dim, sage_hidden_dim)
                for _ in range(sage_layers - 1)
            ]
        )


def build_graphsage_policy_from_config(
    config: Mapping[str, object],
) -> GraphSAGEActorCritic:
    return GraphSAGEActorCritic(
        observation_dim=int(config["observation_dim"]),
        num_actions=int(config["num_actions"]),
        sage_hidden_dim=int(config["sage_hidden_dim"]),
        sage_layers=int(config["sage_layers"]),
        latent_dim=int(config["latent_dim"]),
        head_hidden_dim=int(config["head_hidden_dim"]),
        canonical_schema=str(config.get("canonical_schema", "v1")),
    )
