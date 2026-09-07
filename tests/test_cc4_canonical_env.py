import numpy as np
import pytest

from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DeployDecoy import (
    DeployDecoy,
)

from acd_benchmark.cyber.observation_schema import (
    FEATURE_INDEX,
    NUM_SUBNET_NODES,
    SUBNET_INDEX,
    flat_observation_dim,
)
from acd_benchmark.cyber.observation_schema import (
    PASSIVE_NOW_FEATURES,
    RELATIONAL_EVENT_NOW_FEATURES,
)
from acd_benchmark.cyber.cc4_canonical_env import (
    CANONICAL_PARSER_VERSION,
    CC4CanonicalFlatEnv,
    require_current_canonical_parser,
)


def test_event_details_are_parsed_from_blue_visible_processes():
    raw_observation = {
        "success": "TRUE",
        "host_process": {
            "Processes": [
                {
                    "PID": 17,
                    "PPID": 1,
                    "service_name": "apache2",
                    "Properties": ["rfi"],
                }
            ]
        },
        "host_connection": {
            "Processes": [
                {
                    "Connections": [
                        {
                            "local_port": 22,
                            "remote_address": "10.0.0.2",
                            "remote_port": 49152,
                            "pid": 21,
                        }
                    ]
                }
            ]
        },
    }

    details = CC4CanonicalFlatEnv._observed_host_event_details(raw_observation)

    assert details["host_process"].process_count == 1
    assert details["host_process"].process_has_pid
    assert details["host_process"].connection_count == 0
    assert details["host_connection"].process_count == 0
    assert details["host_connection"].connection_count == 1
    assert details["host_connection"].connection_has_remote_address
    assert details["host_connection"].connection_has_remote_port
    assert details["host_connection"].connection_has_pid


def test_relational_connection_features_preserve_direction_port_and_subnet():
    remote_address = "10.0.0.2"
    raw_observation = {
        "host_inbound": {
            "Processes": [
                {
                    "Connections": [
                        {
                            "local_port": 22,
                            "remote_address": remote_address,
                            "remote_port": 50_000,
                        }
                    ]
                }
            ]
        },
        "host_outbound": {
            "Processes": [
                {
                    "Connections": [
                        {
                            "local_port": 50_001,
                            "remote_address": "203.0.113.7",
                            "remote_port": 443,
                        }
                    ]
                }
            ]
        },
    }

    details = CC4CanonicalFlatEnv._observed_host_event_details(
        raw_observation,
        visible_ip_to_subnet={remote_address: 4},
    )
    inbound = dict(
        zip(
            RELATIONAL_EVENT_NOW_FEATURES,
            details["host_inbound"].relational_values,
            strict=True,
        )
    )
    outbound = dict(
        zip(
            RELATIONAL_EVENT_NOW_FEATURES,
            details["host_outbound"].relational_values,
            strict=True,
        )
    )

    assert inbound["connection_inbound_now"] == 1.0
    assert inbound["local_port_ssh_now"] == 1.0
    assert inbound["remote_port_ephemeral_now"] == 1.0
    assert inbound["remote_address_known_host_now"] == 1.0
    assert inbound["connection_remote_subnet_4_now"] == 1.0
    assert outbound["connection_outbound_now"] == 1.0
    assert outbound["local_port_ephemeral_now"] == 1.0
    assert outbound["remote_port_service_now"] == 1.0
    assert outbound["remote_address_unknown_now"] == 1.0


def test_visible_ip_maps_do_not_share_observations_between_blue_agents():
    env = CC4CanonicalFlatEnv(
        episode_steps=5,
        seed=123,
        canonical_schema="v2",
    )
    first_agent, second_agent = env.possible_agents[:2]
    env._visible_ip_to_subnet = {agent: {} for agent in env.possible_agents}
    env._refresh_visible_ip_maps(
        {
            first_agent: {
                "restricted_zone_a_subnet_router": {
                    "Interface": [{"ip_address": "10.0.0.1"}]
                }
            },
            second_agent: {
                "operational_zone_a_subnet_router": {
                    "Interface": [{"ip_address": "10.0.1.1"}]
                }
            },
        }
    )

    assert "10.0.0.1" in env._visible_ip_to_subnet[first_agent]
    assert "10.0.0.1" not in env._visible_ip_to_subnet[second_agent]
    assert "10.0.1.1" in env._visible_ip_to_subnet[second_agent]
    assert "10.0.1.1" not in env._visible_ip_to_subnet[first_agent]


def test_reset_clears_agent_local_visible_ip_memory():
    env = CC4CanonicalFlatEnv(
        episode_steps=5,
        seed=123,
        canonical_schema="v2",
    )
    env.reset(seed=123)
    agent = env.possible_agents[0]
    env._visible_ip_to_subnet[agent]["stale-address"] = 0

    env.reset(seed=124)

    assert all(
        "stale-address" not in visible_map
        for visible_map in env._visible_ip_to_subnet.values()
    )


def test_checkpoint_parser_guard_rejects_old_or_missing_versions():
    require_current_canonical_parser(
        {"canonical_parser_version": CANONICAL_PARSER_VERSION}
    )

    with pytest.raises(ValueError, match="incompatible"):
        require_current_canonical_parser(
            {
                "canonical_parser_version": (
                    "raw_blue_observation_v3_decoy_feedback_filtered"
                )
            }
        )
    with pytest.raises(ValueError, match="incompatible"):
        require_current_canonical_parser({})


def test_successful_decoy_action_feedback_is_not_passive_evidence():
    target = "restricted_zone_a_subnet_server_host_0"
    action = DeployDecoy(session=0, agent="blue_agent_0", hostname=target)
    raw_observation = {
        "success": TernaryEnum.TRUE,
        "action": action,
        target: {
            "Processes": [
                {
                    "PID": 17,
                    "service_name": "apache2",
                    # Real DeployDecoy payloads commonly expose service/exploit
                    # properties without an explicit "decoy" marker.
                    "Properties": ["rfi"],
                }
            ]
        },
        "restricted_zone_a_subnet_user_host_0": {
            "Processes": [{"PID": 21, "Properties": ["rfi"]}]
        },
    }

    details = CC4CanonicalFlatEnv._observed_host_event_details(raw_observation)

    assert target not in details
    assert details["restricted_zone_a_subnet_user_host_0"].process_count == 1


def test_successful_decoy_feedback_keeps_monitor_connection_evidence():
    target = "restricted_zone_a_subnet_server_host_0"
    action = DeployDecoy(session=0, agent="blue_agent_0", hostname=target)
    raw_observation = {
        "success": TernaryEnum.TRUE,
        "action": action,
        target: {
            "Processes": [
                {
                    "PID": 17,
                    "service_name": "apache2",
                    "Properties": ["rfi"],
                    "Connections": [
                        {
                            "local_port": 80,
                            "remote_address": "10.0.0.2",
                            "remote_port": 49152,
                            "pid": 17,
                        }
                    ],
                }
            ]
        },
    }

    details = CC4CanonicalFlatEnv._observed_host_event_details(raw_observation)

    assert details[target].process_count == 0
    assert details[target].connection_count == 1
    assert details[target].connection_has_remote_address


def test_reset_inventory_is_not_encoded_as_passive_evidence():
    env = CC4CanonicalFlatEnv(episode_steps=20, seed=123)
    env.reset(seed=123)
    for canonical in env.last_canonical.values():
        indices = (
            FEATURE_INDEX["process_event_count_now"],
            FEATURE_INDEX["connection_event_count_now"],
        )
        assert np.count_nonzero(canonical.node_features[:, indices]) == 0


def test_canonical_env_matches_current_ppo_contract():
    env = CC4CanonicalFlatEnv(
        episode_steps=20,
        seed=123,
        canonical_variant="action_aware",
    )
    observations, infos = env.reset(seed=123)

    action_counts = {env.action_space(agent).n for agent in env.possible_agents}
    assert len(action_counts) == 1

    for agent in env.possible_agents:
        observation = observations[agent]
        mask = np.asarray(infos[agent]["action_mask"], dtype=bool)
        assert observation.shape == (flat_observation_dim(),)
        assert observation.dtype == np.float32
        assert env.observation_space(agent).contains(observation)
        assert mask.shape == (env.action_space(agent).n,)
        assert infos[agent]["decision_ready"]
        np.testing.assert_array_equal(
            observation,
            env.last_canonical[agent].flat_view(),
        )

    sleep_actions = {
        agent: env.action_labels(agent).index("Sleep") for agent in env.possible_agents
    }
    next_observations, rewards, terminated, truncated, next_infos = env.step(
        sleep_actions
    )
    assert set(rewards) == set(env.possible_agents)
    for agent in env.possible_agents:
        assert next_observations[agent].shape == (flat_observation_dim(),)
        assert isinstance(terminated[agent], bool)
        assert isinstance(truncated[agent], bool)
        assert "decision_ready" in next_infos[agent]


def test_canonical_structure_matches_cage4_data_links():
    env = CC4CanonicalFlatEnv(episode_steps=20, seed=123)
    env.reset(seed=123)
    state = env.env.env.environment_controller.state
    expected = np.zeros((NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32)

    for source, destination in state.link_diagram.edges:
        source_subnet = state.hostname_subnet_map[source]
        destination_subnet = state.hostname_subnet_map[destination]
        source_name = getattr(source_subnet, "value", source_subnet)
        destination_name = getattr(destination_subnet, "value", destination_subnet)
        if source_name == destination_name:
            continue
        source_index = SUBNET_INDEX[source_name]
        destination_index = SUBNET_INDEX[destination_name]
        expected[source_index, destination_index] = 1.0
        expected[destination_index, source_index] = 1.0

    for canonical in env.last_canonical.values():
        np.testing.assert_array_equal(canonical.structural_adjacency, expected)


def test_v2_event_features_match_raw_blue_observation():
    env = CC4CanonicalFlatEnv(
        episode_steps=30,
        seed=123,
        canonical_schema="v2",
    )
    env.reset(seed=123)
    sleep_actions = {
        agent: env.action_labels(agent).index("Sleep") for agent in env.possible_agents
    }
    observed_events = 0

    for _ in range(20):
        env.step(sleep_actions)
        for agent in env.possible_agents:
            raw_observation = env.last_raw_observations[agent]
            details_by_host = env._observed_host_event_details(
                raw_observation,
                env._visible_ip_to_subnet[agent],
            )
            canonical = env.last_canonical[agent]
            builder = env._builders[agent]
            for hostname, details in details_by_host.items():
                node_index = builder._node_index(hostname)
                if node_index is None:
                    continue
                node = canonical.node_features[node_index]
                expected = details.normalized_values()
                actual = node[[FEATURE_INDEX[name] for name in PASSIVE_NOW_FEATURES]]
                np.testing.assert_array_equal(actual, expected)
                observed_events += int(details.has_passive_event)

    assert observed_events > 0
