"""Flat observations sharing the same raw host features as graph observations."""

from __future__ import annotations

import numpy as np

from acd_benchmark.wrappers.shared.host_obs import GraphObservation, flatten_graph_observation


def encode_flat_observation(
    graph_observation: GraphObservation,
    include_adjacency: bool = True,
    selected_feature_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Flatten graph features for the matched MLP/PPO baseline.

    Keeping the adjacency in the flat vector gives the MLP baseline the same
    topology information that the graph encoder receives as edges. A named
    feature subset can be used for controlled diagnostic views.
    """
    return flatten_graph_observation(
        graph_observation,
        include_adjacency=include_adjacency,
        selected_feature_names=selected_feature_names,
    )
