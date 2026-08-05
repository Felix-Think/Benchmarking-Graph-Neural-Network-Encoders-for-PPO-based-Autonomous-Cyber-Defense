"""Matched Stage 1 observations with known topology and masked host state."""

from __future__ import annotations

import numpy as np

from acd_benchmark.cyber.scalable_scenario import ROLES, SUBNETS, ScalableTopology
from acd_benchmark.wrappers.shared.host_obs import (
    SEMANTIC_FEATURE_NAMES,
    GraphObservation,
    select_node_features,
)
from acd_benchmark.wrappers.shared.partial_obs import (
    apply_host_visibility_mask,
    sample_host_visibility,
)


STATIC_FEATURE_NAMES = (
    *(f"static.role={role}" for role in ROLES),
    *(f"static.subnet={subnet}" for subnet in SUBNETS),
    "static.is_critical",
)
STAGE1_FEATURE_NAMES = STATIC_FEATURE_NAMES + SEMANTIC_FEATURE_NAMES + (
    "visibility.host_visible",
)


def _edge_index(adjacency: np.ndarray) -> np.ndarray:
    """Convert the frozen dense adjacency matrix to PyG edge indices."""
    source, destination = np.nonzero(adjacency)
    if len(source) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.vstack([source, destination]).astype(np.int64)


def _static_features(topology: ScalableTopology) -> np.ndarray:
    """Encode role, subnet, and criticality without simulator true state."""
    rows = []
    critical_hosts = set(topology.critical_hosts)
    for hostname in topology.hostnames:
        role = topology.role_by_host[hostname]
        subnet = topology.subnet_by_host[hostname]
        rows.append(
            [float(role == category) for category in ROLES]
            + [float(subnet == category) for category in SUBNETS]
            + [float(hostname in critical_hosts)]
        )
    return np.asarray(rows, dtype=np.float32)


def build_stage1_base_observation(
    raw_graph_observation: GraphObservation,
    topology: ScalableTopology,
) -> GraphObservation:
    """Build the shared static/dynamic host representation before masking.

    Dynamic security features come only from raw Blue observations. Static
    role/subnet/criticality and adjacency come from the frozen benchmark
    topology and are never masked.
    """
    raw_index = {
        hostname: index
        for index, hostname in enumerate(raw_graph_observation.hostnames)
    }
    if set(raw_index) != set(topology.hostnames):
        raise ValueError("Raw Blue hosts do not match the frozen topology.")
    semantic = select_node_features(
        raw_graph_observation,
        selected_feature_names=SEMANTIC_FEATURE_NAMES,
    )
    semantic = semantic[
        [raw_index[hostname] for hostname in topology.hostnames],
        :,
    ]
    static = _static_features(topology)
    dynamic_start = static.shape[1]
    return GraphObservation(
        node_features=np.concatenate([static, semantic], axis=1),
        edge_index=_edge_index(topology.adjacency),
        adjacency=topology.adjacency.copy(),
        hostnames=topology.hostnames,
        feature_names=STATIC_FEATURE_NAMES + SEMANTIC_FEATURE_NAMES,
        dynamic_feature_indices=tuple(
            range(dynamic_start, dynamic_start + len(SEMANTIC_FEATURE_NAMES))
        ),
    )


def build_stage1_observation(
    raw_graph_observation: GraphObservation,
    topology: ScalableTopology,
    observation_probability: float,
    rng: np.random.Generator,
    forced_visible_hosts: tuple[str, ...] = (),
) -> GraphObservation:
    """Build one Stage 1 Bernoulli-masked graph observation."""
    base = build_stage1_base_observation(raw_graph_observation, topology)
    topology_index = {
        hostname: index
        for index, hostname in enumerate(topology.hostnames)
    }
    forced_indices = tuple(
        topology_index[hostname]
        for hostname in forced_visible_hosts
        if hostname in topology_index
    )
    visibility = sample_host_visibility(
        host_count=topology.node_count,
        observation_probability=observation_probability,
        rng=rng,
        forced_visible_indices=forced_indices,
    )
    return apply_host_visibility_mask(base, visibility=visibility, fill_value=0.0)
