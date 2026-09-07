import numpy as np

from acd_benchmark.cyber.canonical_observation import (
    FEATURE_INDEX,
    NODE_FEATURES,
    NUM_NODES,
    flat_observation_dim,
)
from acd_benchmark.cyber.canonical_observation_v2 import (
    V2_FEATURE_INDEX,
    V2_NODE_FEATURES,
    v2_flat_observation_dim,
)
from scripts.analysis.audit_dynamic_evidence_action_reward import (
    _enrich_episode,
    _zero_dynamic_features,
    _zero_node_feature_group,
)


def test_zero_node_feature_group_changes_only_requested_features():
    observation = np.ones(flat_observation_dim(), dtype=np.float32)
    transformed = _zero_node_feature_group(
        observation,
        ("process_event_now", "connection_event_now"),
    )
    nodes = transformed[: NUM_NODES * len(NODE_FEATURES)].reshape(
        NUM_NODES, len(NODE_FEATURES)
    )

    assert np.all(nodes[:, FEATURE_INDEX["process_event_now"]] == 0.0)
    assert np.all(nodes[:, FEATURE_INDEX["connection_event_now"]] == 0.0)
    assert np.all(nodes[:, FEATURE_INDEX["process_event_memory"]] == 1.0)
    assert np.all(
        transformed[NUM_NODES * len(NODE_FEATURES) :] == 1.0
    )
    assert np.all(observation == 1.0)


def test_enrich_episode_uses_post_decision_reward_and_same_host_timing():
    decisions = [
        {"tick": 2, "agent": "blue_agent_0", "target_host": "host_a", "action_type": "Analyse"},
        {"tick": 6, "agent": "blue_agent_0", "target_host": "host_b", "action_type": "Restore"},
    ]
    events = [{"tick": 1, "agent": "blue_agent_0", "hostname": "host_a"}]
    ticks = [
        {"reward": value, "blue_reward_machine": value, "action_cost": 0.0}
        for value in (-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0)
    ]

    _enrich_episode(decisions, events, ticks)

    assert decisions[0]["reward_next_1"] == -3.0
    assert decisions[0]["reward_next_5"] == -25.0
    assert events[0]["first_decision_latency"] == 1
    assert events[0]["first_same_host_action_latency"] == 1
    assert events[0]["same_host_response_within_1"]
    assert not events[0]["same_host_response_within_0"]


def test_v2_audit_zeroing_uses_v2_node_offsets():
    observation = np.ones(v2_flat_observation_dim(), dtype=np.float32)
    transformed = _zero_dynamic_features(observation, V2_NODE_FEATURES)
    nodes = transformed[: NUM_NODES * len(V2_NODE_FEATURES)].reshape(
        NUM_NODES, len(V2_NODE_FEATURES)
    )

    assert nodes[:, V2_FEATURE_INDEX["exists"]].all()
    assert nodes[:, V2_FEATURE_INDEX["controlled"]].all()
    assert not nodes[:, V2_FEATURE_INDEX["process_event_count_now"]].any()
    assert not nodes[:, V2_FEATURE_INDEX["passive_memory_active"]].any()
    assert np.all(
        transformed[NUM_NODES * len(V2_NODE_FEATURES) :] == 1.0
    )
