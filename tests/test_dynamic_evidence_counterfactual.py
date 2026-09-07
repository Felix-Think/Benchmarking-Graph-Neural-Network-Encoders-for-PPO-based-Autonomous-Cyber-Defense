from __future__ import annotations

import numpy as np

from acd_benchmark.cyber.canonical_observation import (
    CanonicalObservation,
    FEATURE_INDEX,
    GLOBAL_FEATURES,
    MAX_LOCAL_HOSTS,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
)
from acd_benchmark.cyber.dynamic_evidence_counterfactual import (
    DynamicEvidenceCounterfactual,
    EVIDENCE_NODE_FEATURES,
    V2_EVIDENCE_NODE_FEATURES,
)
from acd_benchmark.cyber.canonical_observation_v2 import (
    V2_FEATURE_INDEX,
    V2_NODE_FEATURES,
)


class DummyEnv:
    possible_agents = ("blue_agent_0",)
    node_feature_names = NODE_FEATURES
    last_canonical = {}


class DummyV2Env:
    possible_agents = ("blue_agent_0",)
    node_feature_names = V2_NODE_FEATURES
    last_canonical = {}


def _observation(offset: float = 0.0) -> np.ndarray:
    nodes = np.zeros((NUM_NODES, len(NODE_FEATURES)), dtype=np.float32)
    membership = np.zeros(
        (MAX_LOCAL_HOSTS, NUM_SUBNET_NODES), dtype=np.float32
    )
    evidence_indices = [FEATURE_INDEX[name] for name in EVIDENCE_NODE_FEATURES]
    for host_row in range(8):
        node = NUM_SUBNET_NODES + host_row
        nodes[node, FEATURE_INDEX["exists"]] = 1.0
        nodes[node, FEATURE_INDEX["is_host"]] = 1.0
        nodes[node, evidence_indices] = offset + host_row + 1.0
        nodes[node, FEATURE_INDEX["decoy_active"]] = host_row % 2
        membership[host_row, host_row % 2] = 1.0
    return CanonicalObservation(
        node_features=nodes,
        host_membership=membership,
        structural_adjacency=np.eye(NUM_SUBNET_NODES, dtype=np.float32),
        communication_policy=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        firewall_blocks=np.zeros(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        global_features=np.ones(len(GLOBAL_FEATURES), dtype=np.float32),
    ).flat_view()


def _evidence(observation: np.ndarray) -> np.ndarray:
    nodes = observation[: NUM_NODES * len(NODE_FEATURES)].reshape(
        NUM_NODES, len(NODE_FEATURES)
    )
    host_nodes = np.arange(NUM_SUBNET_NODES, NUM_SUBNET_NODES + 8)
    evidence_indices = [FEATURE_INDEX[name] for name in EVIDENCE_NODE_FEATURES]
    return nodes[np.ix_(host_nodes, evidence_indices)]


def _decoy(observation: np.ndarray) -> np.ndarray:
    nodes = observation[: NUM_NODES * len(NODE_FEATURES)].reshape(
        NUM_NODES, len(NODE_FEATURES)
    )
    host_nodes = np.arange(NUM_SUBNET_NODES, NUM_SUBNET_NODES + 8)
    return nodes[host_nodes, FEATURE_INDEX["decoy_active"]]


def test_shuffle_preserves_values_but_changes_host_alignment() -> None:
    original = _observation()
    gate = DynamicEvidenceCounterfactual(
        DummyEnv(), transform="shuffle_hosts", intervention_seed=123
    )
    shuffled = gate._transform_agent("blue_agent_0", 0, original)
    np.testing.assert_array_equal(
        np.sort(_evidence(shuffled), axis=0),
        np.sort(_evidence(original), axis=0),
    )
    assert not np.array_equal(_evidence(shuffled), _evidence(original))
    np.testing.assert_array_equal(_decoy(shuffled), _decoy(original))


def test_time_shift_uses_exact_lag_and_resets_initial_ticks() -> None:
    gate = DynamicEvidenceCounterfactual(
        DummyEnv(), transform="time_shift", time_shift_lag=2
    )
    first = gate._transform_agent("blue_agent_0", 0, _observation(0.0))
    gate._tick = 1
    second = gate._transform_agent("blue_agent_0", 0, _observation(10.0))
    gate._tick = 2
    third = gate._transform_agent("blue_agent_0", 0, _observation(20.0))
    assert np.count_nonzero(_evidence(first)) == 0
    assert np.count_nonzero(_evidence(second)) == 0
    np.testing.assert_array_equal(_evidence(third), _evidence(_observation(0.0)))
    np.testing.assert_array_equal(_decoy(third), _decoy(_observation(20.0)))


def test_zero_evidence_preserves_decoy_and_non_node_blocks() -> None:
    original = _observation()
    gate = DynamicEvidenceCounterfactual(DummyEnv(), transform="zero_evidence")
    transformed = gate._transform_agent("blue_agent_0", 0, original)
    assert np.count_nonzero(_evidence(transformed)) == 0
    np.testing.assert_array_equal(_decoy(transformed), _decoy(original))
    node_size = NUM_NODES * len(NODE_FEATURES)
    np.testing.assert_array_equal(transformed[node_size:], original[node_size:])


def test_v2_zero_evidence_preserves_action_state() -> None:
    nodes = np.zeros((NUM_NODES, len(V2_NODE_FEATURES)), dtype=np.float32)
    host = NUM_SUBNET_NODES
    nodes[host, V2_FEATURE_INDEX["exists"]] = 1.0
    nodes[host, V2_FEATURE_INDEX["is_host"]] = 1.0
    for feature in V2_EVIDENCE_NODE_FEATURES:
        nodes[host, V2_FEATURE_INDEX[feature]] = 1.0
    nodes[host, V2_FEATURE_INDEX["decoy_active"]] = 1.0
    nodes[host, V2_FEATURE_INDEX["analyse_memory_active"]] = 1.0
    observation = CanonicalObservation(
        node_features=nodes,
        host_membership=np.zeros(
            (MAX_LOCAL_HOSTS, NUM_SUBNET_NODES), dtype=np.float32
        ),
        structural_adjacency=np.eye(NUM_SUBNET_NODES, dtype=np.float32),
        communication_policy=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        firewall_blocks=np.zeros(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        global_features=np.ones(len(GLOBAL_FEATURES), dtype=np.float32),
    ).flat_view()

    gate = DynamicEvidenceCounterfactual(
        DummyV2Env(),
        transform="zero_evidence",
    )
    transformed = gate._transform_agent("blue_agent_0", 0, observation)
    transformed_nodes = transformed[
        : NUM_NODES * len(V2_NODE_FEATURES)
    ].reshape(NUM_NODES, len(V2_NODE_FEATURES))
    for feature in V2_EVIDENCE_NODE_FEATURES:
        assert transformed_nodes[host, V2_FEATURE_INDEX[feature]] == 0.0
    assert transformed_nodes[host, V2_FEATURE_INDEX["decoy_active"]] == 1.0
    assert (
        transformed_nodes[host, V2_FEATURE_INDEX["analyse_memory_active"]]
        == 1.0
    )
