from __future__ import annotations

import numpy as np

from acd_benchmark.cyber.observation_schema import (
    CanonicalObservation,
    FEATURE_INDEX,
    GLOBAL_FEATURES,
    MAX_LOCAL_HOSTS,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
)
from acd_benchmark.cyber.dynamic_host_observation_mask import (
    DYNAMIC_NODE_FEATURES,
    FixedBudgetDynamicHostMask,
)


class DummyCanonicalEnv:
    possible_agents = ("blue_agent_0",)
    node_feature_names = NODE_FEATURES
    last_canonical = {}

    def action_space(self, agent):
        return None

    def observation_space(self, agent):
        return None

    def action_labels(self, agent):
        return ("Sleep",)


def _observation(host_count: int = 8) -> np.ndarray:
    nodes = np.zeros((NUM_NODES, len(NODE_FEATURES)), dtype=np.float32)
    membership = np.zeros((MAX_LOCAL_HOSTS, NUM_SUBNET_NODES), dtype=np.float32)
    for subnet in range(NUM_SUBNET_NODES):
        nodes[subnet, FEATURE_INDEX["exists"]] = 1.0
        nodes[subnet, FEATURE_INDEX["is_subnet"]] = 1.0
    dynamic_indices = [FEATURE_INDEX[name] for name in DYNAMIC_NODE_FEATURES]
    for host_row in range(host_count):
        node = NUM_SUBNET_NODES + host_row
        nodes[node, FEATURE_INDEX["exists"]] = 1.0
        nodes[node, FEATURE_INDEX["is_host"]] = 1.0
        nodes[node, FEATURE_INDEX["controlled"]] = 1.0
        nodes[node, dynamic_indices] = np.arange(
            1, len(dynamic_indices) + 1, dtype=np.float32
        )
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


def test_full_retention_is_identity() -> None:
    original = _observation()
    wrapper = FixedBudgetDynamicHostMask(
        DummyCanonicalEnv(), retention=1.0, mask_seed=123
    )
    transformed, visible, host_count = wrapper._mask_observation(
        original, agent_index=0
    )
    np.testing.assert_array_equal(transformed, original)
    assert host_count == 8
    assert int(visible.sum()) == 8


def test_fixed_budget_masks_only_dynamic_host_features() -> None:
    original = _observation()
    wrapper = FixedBudgetDynamicHostMask(
        DummyCanonicalEnv(), retention=0.5, mask_seed=123
    )
    transformed, visible, host_count = wrapper._mask_observation(
        original, agent_index=0
    )
    assert host_count == 8
    assert int(visible.sum()) == 4

    node_size = NUM_NODES * len(NODE_FEATURES)
    original_nodes = original[:node_size].reshape(NUM_NODES, len(NODE_FEATURES))
    transformed_nodes = transformed[:node_size].reshape(NUM_NODES, len(NODE_FEATURES))
    static_indices = [
        FEATURE_INDEX[name]
        for name in (
            "exists",
            "is_subnet",
            "is_host",
            "is_user",
            "is_server",
            "controlled",
        )
    ]
    dynamic_indices = [FEATURE_INDEX[name] for name in DYNAMIC_NODE_FEATURES]
    np.testing.assert_array_equal(
        transformed_nodes[:, static_indices], original_nodes[:, static_indices]
    )
    np.testing.assert_array_equal(transformed[node_size:], original[node_size:])

    host_nodes = np.arange(NUM_SUBNET_NODES, NUM_SUBNET_NODES + host_count)
    hidden = host_nodes[~visible[host_nodes]]
    shown = host_nodes[visible[host_nodes]]
    assert np.count_nonzero(transformed_nodes[np.ix_(hidden, dynamic_indices)]) == 0
    np.testing.assert_array_equal(
        transformed_nodes[np.ix_(shown, dynamic_indices)],
        original_nodes[np.ix_(shown, dynamic_indices)],
    )


def test_visibility_is_reproducible_for_paired_seed_and_tick() -> None:
    original = _observation()
    first = FixedBudgetDynamicHostMask(
        DummyCanonicalEnv(), retention=0.5, mask_seed=456
    )
    second = FixedBudgetDynamicHostMask(
        DummyCanonicalEnv(), retention=0.5, mask_seed=456
    )
    first._episode_seed = second._episode_seed = 10_000
    first._tick = second._tick = 17
    _, first_visible, _ = first._mask_observation(original, agent_index=0)
    _, second_visible, _ = second._mask_observation(original, agent_index=0)
    np.testing.assert_array_equal(first_visible, second_visible)


def test_persistent_visibility_does_not_change_between_ticks() -> None:
    original = _observation()
    wrapper = FixedBudgetDynamicHostMask(
        DummyCanonicalEnv(),
        retention=0.5,
        mask_seed=789,
        mask_process="persistent",
    )
    wrapper._episode_seed = 10_000
    _, first_visible, _ = wrapper._mask_observation(original, agent_index=0)
    wrapper._tick = 399
    _, last_visible, _ = wrapper._mask_observation(original, agent_index=0)
    np.testing.assert_array_equal(first_visible, last_visible)


def test_iid_visibility_changes_across_ticks() -> None:
    original = _observation(host_count=16)
    wrapper = FixedBudgetDynamicHostMask(
        DummyCanonicalEnv(),
        retention=0.5,
        mask_seed=789,
        mask_process="iid",
    )
    wrapper._episode_seed = 10_000
    _, first_visible, _ = wrapper._mask_observation(original, agent_index=0)
    wrapper._tick = 1
    _, next_visible, _ = wrapper._mask_observation(original, agent_index=0)
    assert not np.array_equal(first_visible, next_visible)
