"""Host-level graph observations built from raw Blue observations."""

from __future__ import annotations

from typing import Any

from acd_benchmark.wrappers.shared.host_obs import GraphObservation, RawBlueHostEncoder


def build_graph_encoder(initial_blue_observation: dict[str, Any]) -> RawBlueHostEncoder:
    """Create the benchmark's stateful raw-observation host encoder."""
    return RawBlueHostEncoder.from_initial_observation(initial_blue_observation)


def encode_graph_observation(
    encoder: RawBlueHostEncoder,
    blue_observation: dict[str, Any],
) -> GraphObservation:
    """Update the encoder cache and return a host-level graph observation."""
    return encoder.update(blue_observation)
