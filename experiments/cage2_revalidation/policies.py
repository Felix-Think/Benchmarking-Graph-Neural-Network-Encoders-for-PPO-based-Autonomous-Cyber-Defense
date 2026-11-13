from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.nn import GCNConv, SAGEConv

from acd_benchmark.cyber.blue_table_env import HOST_NAMES
from acd_benchmark.cyber.cc2_env import get_default_scenario_path
from acd_benchmark.rl.policies.actor_critic import ActorCritic
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.ppo.init_weight import init_weight


def public_graph(path: Path | None = None) -> dict:
    """Use the vendored public scenario, never runtime attacker state."""
    path = path or get_default_scenario_path()
    scenario = yaml.safe_load(path.read_text())
    subnets = tuple(sorted(scenario["Subnets"]))
    node_names = HOST_NAMES + tuple(f"{name}_router" for name in subnets)
    index = {name: position for position, name in enumerate(node_names)}
    edges = set()
    for name in subnets:
        data = scenario["Subnets"][name]
        router = index[f"{name}_router"]
        for host in data["Hosts"]:
            edges.add((index[host], router))
            edges.add((router, index[host]))
        # These are the physical router links created by the native generator.
        targets = subnets if "all" in data["NACLs"] else data["NACLs"]
        for other in targets:
            if other != name:
                edges.add((router, index[f"{other}_router"]))
    return {
        "node_names": list(node_names),
        "host_names": list(HOST_NAMES),
        "router_features": "four zeros; no runtime telemetry",
        "edge_index": torch.tensor(sorted(edges), dtype=torch.long).t().tolist(),
        "source": str(path),
        "dynamic_firewall_edges": False,
    }


class GraphActorCritic(ActorCritic):
    """Two PyG layers and an ordered host readout into main's existing heads."""

    def __init__(self, model: str, graph: dict, hidden_dim: int = 8):
        super().__init__(latent_dim=64, num_actions=41)
        conv = {"gcn": GCNConv, "graphsage": SAGEConv}[model]
        self.conv1 = conv(4, hidden_dim)
        self.conv2 = conv(hidden_dim, hidden_dim)
        self.projection = nn.Linear(len(HOST_NAMES) * hidden_dim, 64)
        self.projection.apply(init_weight)
        self.router_count = len(graph["node_names"]) - len(HOST_NAMES)
        self.register_buffer(
            "edge_index", torch.tensor(graph["edge_index"], dtype=torch.long)
        )

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)
        if observation.shape[1:] != (len(HOST_NAMES), 4):
            raise ValueError(
                "Expected BlueTable observations with shape [batch, 13, 4]"
            )
        routers = observation.new_zeros((len(observation), self.router_count, 4))
        nodes = torch.cat((observation, routers), dim=1)
        nodes = F.relu(self.conv1(nodes, self.edge_index))
        nodes = F.relu(self.conv2(nodes, self.edge_index))
        hosts = nodes[:, : len(HOST_NAMES)].flatten(start_dim=1)
        return torch.tanh(self.projection(hosts))


def make_policy(model: str, graph: dict) -> ActorCritic:
    if model == "flat":
        return FlatActorCritic()
    return GraphActorCritic(model, graph)
