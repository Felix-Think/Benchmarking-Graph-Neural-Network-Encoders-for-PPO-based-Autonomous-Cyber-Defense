import numpy as np

from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions.AbstractActions import Analyse, Remove, Restore
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    AllowTrafficZone,
    BlockTrafficZone,
)
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DeployDecoy import (
    DeployDecoy,
)

from acd_benchmark.cyber.canonical_observation import (
    FEATURE_INDEX,
    MAX_SERVERS,
    NUM_SUBNET_NODES,
    SUBNET_INDEX,
    HostCompressedBuilder,
    ObservedHostEvents,
    deserialize_flat_view,
    flat_observation_dim,
)


AGENT = "blue_agent_0"
HOST = "restricted_zone_a_subnet_user_host_0"
ROUTER = "restricted_zone_a_subnet_router"
HOST_NODE = NUM_SUBNET_NODES + MAX_SERVERS
RESTRICTED_A = SUBNET_INDEX["restricted_zone_a_subnet"]
OPERATIONAL_A = SUBNET_INDEX["operational_zone_a_subnet"]
ADMIN = SUBNET_INDEX["admin_network_subnet"]
OFFICE = SUBNET_INDEX["office_network_subnet"]


def _builder() -> HostCompressedBuilder:
    builder = HostCompressedBuilder(
        agent=AGENT,
        episode_steps=9,
        variant="action_aware",
    )
    builder.reset({"success": TernaryEnum.UNKNOWN, ROUTER: {}, HOST: {}})
    return builder


def test_process_properties_are_not_misclassified_as_process_events():
    builder = _builder()
    service_inventory = builder.update(
        {
            "success": TernaryEnum.TRUE,
            HOST: {"Processes": [{"Properties": ["rfi"]}]},
        },
        step=1,
    )
    assert service_inventory.node_features[
        HOST_NODE, FEATURE_INDEX["process_event_now"]
    ] == 0

    event = builder.update(
        {"success": TernaryEnum.TRUE, HOST: {}},
        step=2,
        event_indicators={
            HOST: ObservedHostEvents(
                process_creation=True,
                network_connection=True,
            )
        },
    )
    for feature in (
        "process_event_now",
        "connection_event_now",
        "process_event_memory",
        "connection_event_memory",
    ):
        assert event.node_features[HOST_NODE, FEATURE_INDEX[feature]] == 1

    no_new_event = builder.update(
        {"success": TernaryEnum.UNKNOWN, HOST: {}},
        step=3,
    )
    assert no_new_event.node_features[
        HOST_NODE, FEATURE_INDEX["process_event_now"]
    ] == 0
    assert no_new_event.node_features[
        HOST_NODE, FEATURE_INDEX["connection_event_now"]
    ] == 0
    assert no_new_event.node_features[
        HOST_NODE, FEATURE_INDEX["process_event_memory"]
    ] == 1
    assert no_new_event.node_features[
        HOST_NODE, FEATURE_INDEX["connection_event_memory"]
    ] == 1


def test_remove_preserves_belief_but_restore_clears_it():
    builder = _builder()
    builder.update(
        {"success": TernaryEnum.TRUE, HOST: {}},
        step=1,
        event_indicators={HOST: ObservedHostEvents(process_creation=True)},
    )
    analysed = builder.update(
        {
            "success": TernaryEnum.TRUE,
            "action": Analyse(session=0, agent=AGENT, hostname=HOST),
            HOST: {"Files": [{"Density": 0.92}, {"Density": 0.25}]},
        },
        step=2,
    )
    assert analysed.node_features[
        HOST_NODE, FEATURE_INDEX["analysis_result_now"]
    ] == 1
    assert analysed.node_features[HOST_NODE, FEATURE_INDEX["analysis_memory"]] == 1
    assert np.isclose(
        analysed.node_features[
            HOST_NODE, FEATURE_INDEX["max_file_density_memory"]
        ],
        0.92,
    )

    removed = builder.update(
        {
            "success": TernaryEnum.TRUE,
            "action": Remove(session=0, agent=AGENT, hostname=HOST),
        },
        step=3,
    )
    for feature in (
        "process_event_memory",
        "analysis_memory",
        "max_file_density_memory",
    ):
        assert removed.node_features[HOST_NODE, FEATURE_INDEX[feature]] > 0

    builder.update(
        {
            "success": TernaryEnum.TRUE,
            "action": DeployDecoy(session=0, agent=AGENT, hostname=HOST),
        },
        step=4,
    )
    restored = builder.update(
        {
            "success": TernaryEnum.TRUE,
            "action": Restore(session=0, agent=AGENT, hostname=HOST),
        },
        step=5,
    )
    for feature in (
        "process_event_memory",
        "connection_event_memory",
        "analysis_memory",
        "max_file_density_memory",
        "decoy_active",
    ):
        assert restored.node_features[HOST_NODE, FEATURE_INDEX[feature]] == 0


def test_structure_policy_and_firewall_are_independent_relations():
    builder = _builder()
    blocked = builder.update(
        {
            "success": TernaryEnum.TRUE,
            "action": BlockTrafficZone(
                session=0,
                agent=AGENT,
                from_subnet="restricted_zone_a_subnet",
                to_subnet="operational_zone_a_subnet",
            ),
        },
        step=1,
    )
    assert blocked.structural_adjacency[RESTRICTED_A, OPERATIONAL_A] == 1
    assert blocked.communication_policy[RESTRICTED_A, OPERATIONAL_A] == 1
    assert blocked.firewall_blocks[RESTRICTED_A, OPERATIONAL_A] == 1
    assert blocked.firewall_blocks[OPERATIONAL_A, RESTRICTED_A] == 1

    phase_one = builder.update({"success": TernaryEnum.UNKNOWN}, step=4)
    np.testing.assert_array_equal(
        phase_one.structural_adjacency,
        blocked.structural_adjacency,
    )
    assert phase_one.communication_policy[RESTRICTED_A, OPERATIONAL_A] == 0
    assert phase_one.firewall_blocks[RESTRICTED_A, OPERATIONAL_A] == 1

    allowed = builder.update(
        {
            "success": TernaryEnum.TRUE,
            "action": AllowTrafficZone(
                session=0,
                agent=AGENT,
                from_subnet="restricted_zone_a_subnet",
                to_subnet="operational_zone_a_subnet",
            ),
        },
        step=5,
    )
    assert allowed.firewall_blocks[RESTRICTED_A, OPERATIONAL_A] == 0
    assert allowed.structural_adjacency[RESTRICTED_A, OPERATIONAL_A] == 1

    edge_index = blocked.graph_edge_index()
    edges = set(zip(edge_index[0], edge_index[1], strict=True))
    assert (RESTRICTED_A, OPERATIONAL_A) in edges
    assert (ADMIN, OFFICE) not in edges


def test_flat_view_is_exactly_recoverable():
    builder = HostCompressedBuilder(
        agent=AGENT,
        episode_steps=100,
        variant="belief",
    )
    canonical = builder.reset(
        {"success": TernaryEnum.UNKNOWN, ROUTER: {}, HOST: {}}
    )
    flat = canonical.flat_view()
    recovered = deserialize_flat_view(flat)

    assert flat.shape == (flat_observation_dim(),)
    assert flat.dtype == np.float32
    np.testing.assert_array_equal(recovered.node_features, canonical.node_features)
    np.testing.assert_array_equal(recovered.host_membership, canonical.host_membership)
    np.testing.assert_array_equal(
        recovered.structural_adjacency,
        canonical.structural_adjacency,
    )
    np.testing.assert_array_equal(
        recovered.communication_policy,
        canonical.communication_policy,
    )
    np.testing.assert_array_equal(recovered.firewall_blocks, canonical.firewall_blocks)
    np.testing.assert_array_equal(recovered.global_features, canonical.global_features)
