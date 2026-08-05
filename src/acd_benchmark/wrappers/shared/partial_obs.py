"""Partial-observation masking for host-level graph observations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from acd_benchmark.wrappers.shared.host_obs import GraphObservation


def mask_feature_groups(
    graph_observation: GraphObservation,
    groups: tuple[str, ...] = ("sessions", "processes", "user_info"),
    fill_value: float = -1.0,
) -> GraphObservation:
    """Mask selected feature groups while preserving host count and topology."""
    prefixes = tuple(f"{group}." for group in groups)
    mask_indices = [
        index
        for index, name in enumerate(graph_observation.feature_names)
        if name.startswith(prefixes)
    ]
    masked_features = graph_observation.node_features.copy()
    if mask_indices:
        masked_features[:, mask_indices] = fill_value
    return replace(graph_observation, node_features=masked_features)


def mask_dynamic_features(
    graph_observation: GraphObservation,
    fill_value: float = -1.0,
) -> GraphObservation:
    """Mask the dynamic features recorded in the shared host schema."""
    masked_features = graph_observation.node_features.copy()
    dynamic_indices = list(graph_observation.dynamic_feature_indices)
    if dynamic_indices:
        masked_features[:, dynamic_indices] = fill_value
    return replace(graph_observation, node_features=masked_features)


def observed_fraction(graph_observation: GraphObservation, fill_value: float = -1.0) -> float:
    """Return the fraction of node-feature values not masked."""
    total = graph_observation.node_features.size
    if total == 0:
        return 0.0
    return float(np.mean(graph_observation.node_features != fill_value))


def sample_host_visibility(
    host_count: int,
    observation_probability: float,
    rng: np.random.Generator,
    forced_visible_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Sample the proposal's i.i.d. Bernoulli per-host visibility mask."""
    if not 0.0 <= observation_probability <= 1.0:
        raise ValueError("observation_probability must be in [0, 1].")
    visibility = rng.random(host_count) < observation_probability
    if forced_visible_indices:
        visibility[list(forced_visible_indices)] = True
    return visibility


def apply_host_visibility_mask(
    graph_observation: GraphObservation,
    visibility: np.ndarray,
    fill_value: float = 0.0,
    feature_name: str = "visibility.host_visible",
) -> GraphObservation:
    """Mask dynamic host state and append an explicit visibility bit.

    Static node features and graph topology are preserved. The visibility bit
    is appended even at 100% observation so all severity levels share one
    fixed tensor schema.
    """
    host_count = graph_observation.node_features.shape[0]
    if visibility.shape != (host_count,):
        raise ValueError(f"visibility must have shape ({host_count},)")
    masked_features = graph_observation.node_features.copy()
    dynamic_indices = list(graph_observation.dynamic_feature_indices)
    hidden_rows = np.array(
        [
            index
            for index, is_visibility in enumerate(visibility)
            if not is_visibility # tim cac ham False trong visibility(False = bi che)
        ]
    )
    #hidden_rows = np.flatnonzero(~visibility.astype(bool))
    if dynamic_indices and len(hidden_rows):
        masked_features[np.ix_(hidden_rows, dynamic_indices)] = fill_value
    masked_features = np.concatenate(
        [masked_features, visibility.astype(np.float32).reshape(-1, 1)],
        axis=1,
    )
    return GraphObservation(
        node_features=masked_features,
        edge_index=graph_observation.edge_index,
        adjacency=graph_observation.adjacency,
        hostnames=graph_observation.hostnames,
        feature_names=graph_observation.feature_names + (feature_name,),
        dynamic_feature_indices=graph_observation.dynamic_feature_indices,
    )
