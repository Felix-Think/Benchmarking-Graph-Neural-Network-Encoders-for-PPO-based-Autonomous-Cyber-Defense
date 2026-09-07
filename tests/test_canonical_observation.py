import numpy as np
import pytest

from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions.AbstractActions import Analyse, Restore
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    AllowTrafficZone,
    BlockTrafficZone,
)

from acd_benchmark.cyber.observation_schema import (
    MAX_SERVERS,
    NUM_SUBNET_NODES,
    RELATIONAL_EVENT_NOW_FEATURES,
    FEATURE_INDEX,
    flat_observation_dim,
)
from acd_benchmark.cyber.canonical_observation import (
    HostCompressedBuilder,
    ObservedHostEventDetails,
)
from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv


AGENT = "blue_agent_0"
HOST = "restricted_zone_a_subnet_user_host_0"
ROUTER = "restricted_zone_a_subnet_router"
HOST_NODE = NUM_SUBNET_NODES + MAX_SERVERS


def _builder() -> HostCompressedBuilder:
    return HostCompressedBuilder(
        agent=AGENT,
        episode_steps=20,
        variant="action_aware",
        passive_memory_ttl=4,
        analysis_memory_ttl=4,
    )


def test_keeps_observable_event_attributes_and_expires_passive_memory():
    relational_values = tuple(
        1.0 if name == "connection_inbound_now" else 0.0
        for name in RELATIONAL_EVENT_NOW_FEATURES
    )
    builder = _builder()
    details = ObservedHostEventDetails(
        process_count=5,
        process_has_pid=True,
        process_has_connection=True,
        connection_count=2,
        connection_has_remote_address=True,
        connection_has_remote_port=True,
        relational_values=relational_values,
    )
    initial = builder.reset(
        {"success": TernaryEnum.UNKNOWN, ROUTER: {}, HOST: {}},
        event_details={HOST: details},
    )

    assert (
        initial.node_features[HOST_NODE, FEATURE_INDEX["process_event_count_now"]]
        == 1.0
    )
    assert (
        initial.node_features[HOST_NODE, FEATURE_INDEX["process_has_pid_memory"]] == 1.0
    )
    assert (
        initial.node_features[HOST_NODE, FEATURE_INDEX["passive_memory_active"]] == 1.0
    )
    assert (
        initial.node_features[HOST_NODE, FEATURE_INDEX["connection_inbound_now"]] == 1.0
    )
    assert (
        initial.node_features[HOST_NODE, FEATURE_INDEX["connection_inbound_memory"]]
        == 1.0
    )

    aged = builder.update({"success": TernaryEnum.UNKNOWN}, step=1)
    assert (
        aged.node_features[HOST_NODE, FEATURE_INDEX["process_event_count_now"]] == 0.0
    )
    assert np.isclose(
        aged.node_features[HOST_NODE, FEATURE_INDEX["process_event_count_memory"]],
        0.75,
    )
    assert np.isclose(
        aged.node_features[HOST_NODE, FEATURE_INDEX["passive_memory_age_normalized"]],
        0.25,
    )
    assert np.isclose(
        aged.node_features[HOST_NODE, FEATURE_INDEX["connection_inbound_memory"]],
        0.75,
    )

    builder.update({"success": TernaryEnum.UNKNOWN}, step=2)
    builder.update({"success": TernaryEnum.UNKNOWN}, step=3)
    expired = builder.update({"success": TernaryEnum.UNKNOWN}, step=4)
    for feature in (
        "process_event_count_memory",
        "process_has_pid_memory",
        "passive_memory_active",
        "passive_memory_age_normalized",
        "connection_inbound_memory",
    ):
        assert expired.node_features[HOST_NODE, FEATURE_INDEX[feature]] == 0.0


def test_refresh_replaces_old_passive_snapshot_instead_of_extending_maximum():
    builder = _builder()
    builder.reset(
        {"success": TernaryEnum.UNKNOWN, ROUTER: {}, HOST: {}},
        event_details={
            HOST: ObservedHostEventDetails(
                process_count=5,
                process_has_pid=True,
                process_has_connection=True,
            )
        },
    )
    refreshed = builder.update(
        {"success": TernaryEnum.UNKNOWN, HOST: {}},
        step=1,
        event_details={HOST: ObservedHostEventDetails(process_count=1)},
    )

    assert np.isclose(
        refreshed.node_features[HOST_NODE, FEATURE_INDEX["process_event_count_memory"]],
        0.2,
    )
    assert (
        refreshed.node_features[HOST_NODE, FEATURE_INDEX["process_has_pid_memory"]]
        == 0.0
    )
    assert (
        refreshed.node_features[
            HOST_NODE, FEATURE_INDEX["process_has_connection_memory"]
        ]
        == 0.0
    )
    assert (
        refreshed.node_features[
            HOST_NODE, FEATURE_INDEX["passive_memory_age_normalized"]
        ]
        == 0.0
    )


def test_separates_analyse_history_and_restore_clears_both_memories():
    builder = _builder()
    builder.reset({"success": TernaryEnum.UNKNOWN, ROUTER: {}, HOST: {}})
    builder.update(
        {"success": TernaryEnum.UNKNOWN, HOST: {}},
        step=1,
        event_details={
            HOST: ObservedHostEventDetails(
                connection_count=1,
                relational_values=tuple(
                    1.0 if name == "connection_inbound_now" else 0.0
                    for name in RELATIONAL_EVENT_NOW_FEATURES
                ),
            )
        },
    )
    analysed = builder.update(
        {
            "success": TernaryEnum.TRUE,
            "action": Analyse(session=0, agent=AGENT, hostname=HOST),
            HOST: {"Files": [{"Density": 0.8}]},
        },
        step=2,
    )

    assert (
        analysed.node_features[HOST_NODE, FEATURE_INDEX["analyse_completed_now"]] == 1.0
    )
    assert (
        analysed.node_features[HOST_NODE, FEATURE_INDEX["analyse_memory_active"]] == 1.0
    )
    assert np.isclose(
        analysed.node_features[HOST_NODE, FEATURE_INDEX["max_file_density_memory"]],
        0.8,
    )
    assert (
        analysed.node_features[HOST_NODE, FEATURE_INDEX["passive_memory_active"]] == 1.0
    )
    assert (
        analysed.node_features[HOST_NODE, FEATURE_INDEX["connection_inbound_memory"]]
        > 0.0
    )

    reanalysed = builder.update(
        {
            "success": TernaryEnum.TRUE,
            "action": Analyse(session=0, agent=AGENT, hostname=HOST),
            HOST: {"Files": [{"Density": 0.2}]},
        },
        step=3,
    )
    assert np.isclose(
        reanalysed.node_features[HOST_NODE, FEATURE_INDEX["max_file_density_memory"]],
        0.2,
    )

    restored = builder.update(
        {
            "success": TernaryEnum.TRUE,
            "action": Restore(session=0, agent=AGENT, hostname=HOST),
        },
        step=4,
    )
    for feature in (
        "passive_memory_active",
        "analyse_memory_active",
        "max_file_density_memory",
        "connection_inbound_memory",
    ):
        assert restored.node_features[HOST_NODE, FEATURE_INDEX[feature]] == 0.0


@pytest.mark.parametrize("canonical_schema", ["v2"])
def test_semantic_mask_tracks_successful_firewall_transitions_independently_of_v2(
    canonical_schema,
):
    env = CC4CanonicalFlatEnv(
        episode_steps=20,
        seed=123,
        canonical_schema=canonical_schema,
        semantic_action_mask=True,
    )
    _, infos = env.reset(seed=123)
    agent = env.possible_agents[0]
    actions = env.env.actions(agent)
    block_index = next(
        index
        for index, action in enumerate(actions)
        if isinstance(action, BlockTrafficZone)
    )
    block = actions[block_index]
    allow_index = next(
        index
        for index, action in enumerate(actions)
        if isinstance(action, AllowTrafficZone)
        and action.from_subnet == block.from_subnet
        and action.to_subnet == block.to_subnet
    )

    assert infos[agent]["action_mask"][block_index]
    assert not infos[agent]["action_mask"][allow_index]

    _, _, _, _, infos = env.step({agent: block_index})
    assert not infos[agent]["action_mask"][block_index]
    assert infos[agent]["action_mask"][allow_index]

    _, _, _, _, infos = env.step({agent: allow_index})
    assert infos[agent]["action_mask"][block_index]
    assert not infos[agent]["action_mask"][allow_index]


def test_can_use_official_mask_without_semantic_intervention():
    v2_off = CC4CanonicalFlatEnv(
        episode_steps=20,
        seed=123,
        canonical_schema="v2",
        semantic_action_mask=False,
    )
    observations, official_infos = v2_off.reset(seed=123)
    agent = v2_off.possible_agents[0]
    allow_index = next(
        index
        for index, action in enumerate(v2_off.env.actions(agent))
        if isinstance(action, AllowTrafficZone)
    )
    assert observations[agent].shape == (flat_observation_dim(),)
    assert official_infos[agent]["action_mask"][allow_index]
