import argparse

import numpy as np

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from acd_benchmark.cyber.env_factory import make_cc2_env  # noqa: E402
from acd_benchmark.wrappers.shared.flat_obs import encode_flat_observation  # noqa: E402
from acd_benchmark.wrappers.shared.graph_obs import (  # noqa: E402
    build_graph_encoder,
    encode_graph_observation,
)
from CybORG.Agents import BlueReactRemoveAgent  # noqa: E402


def _format_value(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _print_graph_observation(graph_obs, max_hosts: int, max_edges: int) -> None:
    print("=== graph_obs output ===")
    print(f"node_features shape: {list(graph_obs.node_features.shape)}")
    print(f"adjacency shape:     {list(graph_obs.adjacency.shape)}")
    print(f"edge_index shape:    {list(graph_obs.edge_index.shape)}")
    print(f"host count:          {len(graph_obs.hostnames)}")
    print(f"feature count:       {len(graph_obs.feature_names)}")
    print()

    print("Hosts:")
    for index, hostname in enumerate(graph_obs.hostnames):
        print(f"  [{index}] {hostname}")
    print()

    dynamic_features = [
        graph_obs.feature_names[index]
        for index in graph_obs.dynamic_feature_indices
    ]
    print("Dynamic feature columns:")
    for name in dynamic_features:
        print(f"  - {name}")
    print()

    print(f"Non-zero features for first {min(max_hosts, len(graph_obs.hostnames))} hosts:")
    for host_index, hostname in enumerate(graph_obs.hostnames[:max_hosts]):
        print(f"  [{host_index}] {hostname}")
        row = graph_obs.node_features[host_index]
        for feature_index, value in enumerate(row):
            if value != 0:
                feature_name = graph_obs.feature_names[feature_index]
                print(f"    {feature_index:02d} {feature_name}: {_format_value(float(value))}")
    print()

    src_nodes, dst_nodes = graph_obs.edge_index
    edge_count = graph_obs.edge_index.shape[1]
    print(f"Edges shown: {min(max_edges, edge_count)} / {edge_count}")
    for edge_pos, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
        if edge_pos >= max_edges:
            break
        src_name = graph_obs.hostnames[int(src)]
        dst_name = graph_obs.hostnames[int(dst)]
        print(f"  {int(src):02d}:{src_name} -> {int(dst):02d}:{dst_name}")
    print()


def _print_flat_observation(flat_obs: np.ndarray, graph_obs, max_values: int) -> None:
    num_hosts, feature_dim = graph_obs.node_features.shape
    node_part_len = num_hosts * feature_dim
    adjacency_part_len = num_hosts * num_hosts
    node_part = flat_obs[:node_part_len]
    adjacency_part = flat_obs[node_part_len:]

    print("=== flat_obs output ===")
    print(f"flat shape:             {list(flat_obs.shape)}")
    print(f"node feature part:      {node_part_len} values")
    print(f"adjacency part:         {adjacency_part_len} values")
    print(f"node part shape view:   {[num_hosts, feature_dim]}")
    print(f"adjacency shape view:   {[num_hosts, num_hosts]}")
    print(
        "node part matches graph_obs.node_features: "
        f"{np.array_equal(node_part.reshape(num_hosts, feature_dim), graph_obs.node_features)}"
    )
    print(
        "adjacency part matches graph_obs.adjacency: "
        f"{np.array_equal(adjacency_part.reshape(num_hosts, num_hosts), graph_obs.adjacency)}"
    )
    print()

    print(f"First {min(max_values, len(flat_obs))} flat values:")
    for index, value in enumerate(flat_obs[:max_values]):
        print(f"  flat[{index:03d}] = {_format_value(float(value))}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect graph_obs and flat_obs outputs from the raw Blue host encoder.",
    )
    parser.add_argument("--steps", type=int, default=0, help="Blue/Red steps to run before printing.")
    parser.add_argument("--max-hosts", type=int, default=3, help="How many hosts to show in detail.")
    parser.add_argument("--max-edges", type=int, default=20, help="How many graph edges to print.")
    parser.add_argument("--max-flat-values", type=int, default=40, help="How many flat values to print.")
    args = parser.parse_args()

    cyborg = make_cc2_env()
    blue_agent = BlueReactRemoveAgent()

    result = cyborg.reset("Blue")
    observation = result.observation
    encoder = build_graph_encoder(observation)
    graph_obs = encode_graph_observation(encoder, observation)

    for _ in range(args.steps):
        action = blue_agent.get_action(observation, cyborg.get_action_space("Blue"))
        result = cyborg.step("Blue", action)
        observation = result.observation
        graph_obs = encode_graph_observation(encoder, observation)

    flat_obs = encode_flat_observation(graph_obs)

    print(f"Printed after steps: {args.steps}")
    print()
    _print_graph_observation(graph_obs, max_hosts=args.max_hosts, max_edges=args.max_edges)
    _print_flat_observation(flat_obs, graph_obs, max_values=args.max_flat_values)


if __name__ == "__main__":
    main()
