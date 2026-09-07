import numpy as np
import pytest

from acd_benchmark.cyber.observation_schema import (
    FEATURE_INDEX,
    GLOBAL_FEATURES,
    MAX_LOCAL_HOSTS,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
    CanonicalObservation,
    deserialize_flat_view,
)
from acd_benchmark.cyber.canonical_observation_condition import (
    SENSOR_EVIDENCE_FEATURES,
    CanonicalObservationCondition,
)
from acd_benchmark.cyber.observation_schema import (
    PASSIVE_MEMORY_FEATURES,
    PASSIVE_NOW_FEATURES,
    flat_observation_dim,
)


AGENT = "blue_agent_0"


class StaticCanonicalEnv:
    possible_agents = (AGENT,)
    node_feature_names = NODE_FEATURES

    def __init__(self, observation: np.ndarray):
        self.observation = observation
        self.last_canonical = {AGENT: None}

    def reset(self, seed=None):
        return (
            {AGENT: self.observation.copy()},
            {
                AGENT: {
                    "action_mask": np.asarray([True, False, True]),
                    "decision_ready": True,
                }
            },
        )

    def step(self, actions):
        observations, infos = self.reset()
        return (
            observations,
            {AGENT: 0.0},
            {AGENT: False},
            {AGENT: True},
            infos,
        )

    @staticmethod
    def observation_space(agent):
        return None

    @staticmethod
    def action_space(agent):
        return None

    @staticmethod
    def action_labels(agent):
        return ("a", "b", "c")


def _canonical() -> CanonicalObservation:
    return CanonicalObservation(
        node_features=np.ones(
            (NUM_NODES, len(NODE_FEATURES)),
            dtype=np.float32,
        ),
        host_membership=np.ones(
            (MAX_LOCAL_HOSTS, NUM_SUBNET_NODES),
            dtype=np.float32,
        ),
        structural_adjacency=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES),
            dtype=np.float32,
        ),
        communication_policy=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES),
            dtype=np.float32,
        ),
        firewall_blocks=np.ones(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES),
            dtype=np.float32,
        ),
        global_features=np.ones(len(GLOBAL_FEATURES), dtype=np.float32),
    )


def _condition(name: str):
    canonical = _canonical()
    env = CanonicalObservationCondition(
        StaticCanonicalEnv(canonical.flat_view()),
        condition=name,
    )
    observations, infos = env.reset(seed=123)
    return canonical, deserialize_flat_view(observations[AGENT]), infos[AGENT]


@pytest.mark.parametrize(
    "name",
    (
        "full",
        "no_evidence",
        "action_state_only",
        "topology_only",
        "zero_with_mask",
    ),
)
def test_all_conditions_preserve_official_action_mask(name):
    _, _, info = _condition(name)
    np.testing.assert_array_equal(info["action_mask"], [True, False, True])
    assert info["observation_condition"] == name


def test_full_preserves_every_canonical_block():
    canonical, transformed, _ = _condition("full")
    np.testing.assert_array_equal(transformed.flat_view(), canonical.flat_view())


def test_no_evidence_preserves_action_state_and_topology():
    _, transformed, _ = _condition("no_evidence")
    evidence_indices = [FEATURE_INDEX[name] for name in SENSOR_EVIDENCE_FEATURES]

    assert not transformed.node_features[:, evidence_indices].any()
    assert transformed.node_features[:, FEATURE_INDEX["decoy_active"]].all()
    assert transformed.node_features[:, FEATURE_INDEX["analyse_completed_now"]].all()
    assert transformed.node_features[:, FEATURE_INDEX["analyse_memory_active"]].all()
    assert transformed.host_membership.all()
    assert transformed.structural_adjacency.all()
    assert transformed.communication_policy.all()
    assert transformed.firewall_blocks.all()
    assert transformed.global_features.all()


def test_action_state_only_keeps_exact_requested_blocks():
    _, transformed, _ = _condition("action_state_only")
    decoy_index = FEATURE_INDEX["decoy_active"]

    assert transformed.node_features[:, decoy_index].all()
    assert transformed.node_features.sum() == NUM_NODES
    assert not transformed.host_membership.any()
    assert not transformed.structural_adjacency.any()
    assert not transformed.communication_policy.any()
    assert transformed.firewall_blocks.all()
    assert transformed.global_features.all()


def test_topology_only_keeps_static_identity_and_physical_structure():
    _, transformed, _ = _condition("topology_only")
    static = {
        "exists",
        "is_subnet",
        "is_host",
        "is_user",
        "is_server",
        "controlled",
    }

    for index, name in enumerate(NODE_FEATURES):
        if name in static:
            assert transformed.node_features[:, index].all()
        else:
            assert not transformed.node_features[:, index].any()
    assert transformed.host_membership.all()
    assert transformed.structural_adjacency.all()
    assert not transformed.communication_policy.any()
    assert not transformed.firewall_blocks.any()
    assert not transformed.global_features.any()


def test_zero_with_mask_removes_observation_but_not_mask():
    _, transformed, info = _condition("zero_with_mask")
    assert not transformed.flat_view().any()
    np.testing.assert_array_equal(info["action_mask"], [True, False, True])


def test_v2_no_evidence_removes_all_relational_now_and_memory_features():
    observation = np.ones(flat_observation_dim(), dtype=np.float32)
    env = StaticCanonicalEnv(observation)
    env.node_feature_names = NODE_FEATURES
    conditioned = CanonicalObservationCondition(env, condition="no_evidence")

    observations, infos = conditioned.reset(seed=123)
    nodes = observations[AGENT][: NUM_NODES * len(NODE_FEATURES)].reshape(
        NUM_NODES, len(NODE_FEATURES)
    )
    evidence_features = (
        *PASSIVE_NOW_FEATURES,
        *PASSIVE_MEMORY_FEATURES,
        "passive_memory_active",
        "passive_memory_age_normalized",
    )

    for feature in evidence_features:
        assert not nodes[:, FEATURE_INDEX[feature]].any()
    assert nodes[:, FEATURE_INDEX["decoy_active"]].all()
    assert nodes[:, FEATURE_INDEX["analyse_completed_now"]].all()
    np.testing.assert_array_equal(
        infos[AGENT]["action_mask"],
        [True, False, True],
    )
