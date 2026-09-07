import numpy as np

from acd_benchmark.cyber.canonical_observation import (
    FEATURE_INDEX,
    GLOBAL_FEATURES,
    MAX_LOCAL_HOSTS,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
    CanonicalObservation,
    deserialize_flat_view,
)
from acd_benchmark.cyber.canonical_observation_v2 import V2_NODE_FEATURES
from scripts.evaluation.evaluate_canonical_observation_gate import (
    DYNAMIC_NODE_FEATURES,
    CanonicalObservationGate,
)


AGENT = "blue_agent_0"


class StaticGateEnv:
    possible_agents = [AGENT]

    def __init__(self, observation: np.ndarray):
        self.observation = observation

    def reset(self, seed=None):
        return (
            {AGENT: self.observation.copy()},
            {AGENT: {"action_mask": np.array([True, False, True])}},
        )


def _canonical() -> CanonicalObservation:
    node_features = np.ones((NUM_NODES, len(NODE_FEATURES)), dtype=np.float32)
    return CanonicalObservation(
        node_features=node_features,
        host_membership=np.ones(
            (MAX_LOCAL_HOSTS, NUM_SUBNET_NODES), dtype=np.float32
        ),
        structural_adjacency=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        communication_policy=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        firewall_blocks=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        global_features=np.ones(len(GLOBAL_FEATURES), dtype=np.float32),
    )


def _reset(transform: str):
    canonical = _canonical()
    gate = CanonicalObservationGate(StaticGateEnv(canonical.flat_view()), transform)
    observations, infos = gate.reset(seed=123)
    return canonical, observations[AGENT], infos[AGENT]["action_mask"]


def test_zero_all_preserves_action_mask():
    _, observation, action_mask = _reset("zero_all")
    assert not observation.any()
    np.testing.assert_array_equal(action_mask, [True, False, True])


def test_zero_all_no_action_mask_keeps_only_policy_biases():
    _, observation, action_mask = _reset("zero_all_no_action_mask")
    assert not observation.any()
    np.testing.assert_array_equal(action_mask, [True, True, True])


def test_mission_only_preserves_only_phase_features_and_action_mask():
    _, observation, action_mask = _reset("mission_only")
    assert observation.sum() == 3
    np.testing.assert_array_equal(
        observation[-len(GLOBAL_FEATURES) :],
        [1, 1, 1, 0, 0, 0, 0],
    )
    np.testing.assert_array_equal(action_mask, [True, False, True])


def test_topology_only_removes_dynamic_policy_firewall_and_global_features():
    _, observation, _ = _reset("topology_only")
    transformed = deserialize_flat_view(observation)
    dynamic_indices = [FEATURE_INDEX[name] for name in DYNAMIC_NODE_FEATURES]
    static_indices = [
        index for index in range(len(NODE_FEATURES)) if index not in dynamic_indices
    ]

    assert transformed.node_features[:, dynamic_indices].sum() == 0
    assert transformed.node_features[:, static_indices].all()
    assert transformed.host_membership.all()
    assert transformed.structural_adjacency.all()
    assert not transformed.communication_policy.any()
    assert not transformed.firewall_blocks.any()
    assert not transformed.global_features.any()


def test_node_identity_only_removes_all_graph_matrices():
    _, observation, _ = _reset("node_identity_only")
    transformed = deserialize_flat_view(observation)
    dynamic_indices = [FEATURE_INDEX[name] for name in DYNAMIC_NODE_FEATURES]
    static_indices = [
        index for index in range(len(NODE_FEATURES)) if index not in dynamic_indices
    ]

    assert transformed.node_features[:, static_indices].all()
    assert not transformed.node_features[:, dynamic_indices].any()
    assert not transformed.host_membership.any()
    assert not transformed.structural_adjacency.any()
    assert not transformed.communication_policy.any()
    assert not transformed.firewall_blocks.any()
    assert not transformed.global_features.any()


def test_structure_only_preserves_membership_and_adjacency():
    _, observation, _ = _reset("structure_only")
    transformed = deserialize_flat_view(observation)

    assert not transformed.node_features.any()
    assert transformed.host_membership.all()
    assert transformed.structural_adjacency.all()
    assert not transformed.communication_policy.any()
    assert not transformed.firewall_blocks.any()
    assert not transformed.global_features.any()


def test_topology_no_adjacency_preserves_identity_and_membership():
    _, observation, _ = _reset("topology_no_adjacency")
    transformed = deserialize_flat_view(observation)
    dynamic_indices = [FEATURE_INDEX[name] for name in DYNAMIC_NODE_FEATURES]
    static_indices = [
        index for index in range(len(NODE_FEATURES)) if index not in dynamic_indices
    ]

    assert transformed.node_features[:, static_indices].all()
    assert not transformed.node_features[:, dynamic_indices].any()
    assert transformed.host_membership.all()
    assert not transformed.structural_adjacency.any()
    assert not transformed.communication_policy.any()
    assert not transformed.firewall_blocks.any()
    assert not transformed.global_features.any()


def test_topology_mission_adds_phase_to_topology_only():
    _, observation, _ = _reset("topology_mission")
    transformed = deserialize_flat_view(observation)
    np.testing.assert_array_equal(
        transformed.global_features,
        [1, 1, 1, 0, 0, 0, 0],
    )
    assert transformed.structural_adjacency.all()
    assert not transformed.communication_policy.any()


def test_full_no_action_mask_preserves_observation_and_removes_mask_constraint():
    canonical, observation, action_mask = _reset("full_no_action_mask")
    np.testing.assert_array_equal(observation, canonical.flat_view())
    np.testing.assert_array_equal(action_mask, [True, True, True])


def test_static_gate_supports_v2_node_feature_layout():
    canonical = CanonicalObservation(
        node_features=np.ones(
            (NUM_NODES, len(V2_NODE_FEATURES)), dtype=np.float32
        ),
        host_membership=np.ones(
            (MAX_LOCAL_HOSTS, NUM_SUBNET_NODES), dtype=np.float32
        ),
        structural_adjacency=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        communication_policy=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        firewall_blocks=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        ),
        global_features=np.ones(len(GLOBAL_FEATURES), dtype=np.float32),
    )
    env = StaticGateEnv(canonical.flat_view())
    env.node_feature_names = V2_NODE_FEATURES
    gate = CanonicalObservationGate(env, "node_identity_only")
    observations, _ = gate.reset(seed=123)
    nodes = observations[AGENT][: NUM_NODES * len(V2_NODE_FEATURES)].reshape(
        NUM_NODES, len(V2_NODE_FEATURES)
    )
    static_features = {
        "exists",
        "is_subnet",
        "is_host",
        "is_user",
        "is_server",
        "controlled",
    }

    for index, name in enumerate(V2_NODE_FEATURES):
        if name in static_features:
            assert nodes[:, index].all()
        else:
            assert not nodes[:, index].any()
