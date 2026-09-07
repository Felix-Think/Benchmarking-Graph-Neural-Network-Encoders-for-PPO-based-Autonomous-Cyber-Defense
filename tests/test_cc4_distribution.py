import numpy as np

from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.cyber.cc4_distribution import LATERAL_MOVEMENT_EDGES


def _generator(env: CC4CanonicalFlatEnv):
    return env.env.env.scenario_generator


def test_random_pivots_are_existing_hosts_and_reproduce_by_seed():
    first = CC4CanonicalFlatEnv(
        episode_steps=20,
        seed=123,
        environment_distribution="random_host",
    )
    second = CC4CanonicalFlatEnv(
        episode_steps=20,
        seed=123,
        environment_distribution="random_host",
    )

    first.reset(seed=123)
    second.reset(seed=123)
    first_assignments = _generator(first).last_pivot_assignments
    second_assignments = _generator(second).last_pivot_assignments

    assert first_assignments == second_assignments
    state = first.env.env.environment_controller.state
    for subnet_name, hostname in first_assignments.items():
        assert hostname in state.hosts
        assert "_router" not in hostname
        assert state.hostname_subnet_map[hostname].value == subnet_name


def test_random_pivot_links_match_logical_cage4_links():
    env = CC4CanonicalFlatEnv(
        episode_steps=20,
        seed=321,
        environment_distribution="random_host",
    )
    env.reset(seed=321)
    state = env.env.env.environment_controller.state
    assignments = _generator(env).last_pivot_assignments

    expected_links = {hostname: set() for hostname in assignments.values()}
    for source_subnet, destination_subnet in LATERAL_MOVEMENT_EDGES:
        source_host = assignments[source_subnet]
        destination_host = assignments[destination_subnet]
        expected_links[source_host].add(destination_host)
        expected_links[destination_host].add(source_host)

    for hostname, expected in expected_links.items():
        assert set(state.hosts[hostname].info) == expected

    non_pivots = set(state.hosts) - set(assignments.values())
    assert all(not state.hosts[hostname].info for hostname in non_pivots)


def test_random_pivot_distribution_changes_slots_across_seeds():
    assignments = []
    for seed in range(100, 106):
        env = CC4CanonicalFlatEnv(
            episode_steps=10,
            seed=seed,
            environment_distribution="random_host",
        )
        env.reset(seed=seed)
        assignments.append(tuple(sorted(_generator(env).last_pivot_assignments.items())))

    assert len(set(assignments)) > 1
    selected_hosts = {
        hostname
        for assignment in assignments
        for _, hostname in assignment
    }
    assert any("_user_host_" in hostname for hostname in selected_hosts)
    assert any(not hostname.endswith("_server_host_0") for hostname in selected_hosts)


def test_official_distribution_keeps_server_zero_pivots():
    env = CC4CanonicalFlatEnv(
        episode_steps=10,
        seed=789,
        environment_distribution="official",
    )
    env.reset(seed=789)
    state = env.env.env.environment_controller.state

    for source_subnet, destination_subnet in LATERAL_MOVEMENT_EDGES:
        source = f"{source_subnet}_server_host_0"
        destination = f"{destination_subnet}_server_host_0"
        assert destination in state.hosts[source].info
        assert source in state.hosts[destination].info


def test_distribution_keeps_canonical_and_action_contract_fixed():
    official = CC4CanonicalFlatEnv(
        episode_steps=20,
        seed=456,
        environment_distribution="official",
    )
    randomized = CC4CanonicalFlatEnv(
        episode_steps=20,
        seed=456,
        environment_distribution="random_host",
    )
    official_observations, official_infos = official.reset(seed=456)
    random_observations, random_infos = randomized.reset(seed=456)

    for agent in official.possible_agents:
        assert official.observation_space(agent) == randomized.observation_space(agent)
        assert official.action_space(agent).n == randomized.action_space(agent).n
        assert official.action_labels(agent) == randomized.action_labels(agent)
        np.testing.assert_array_equal(
            official_infos[agent]["action_mask"],
            random_infos[agent]["action_mask"],
        )
        assert official_observations[agent].shape == random_observations[agent].shape
