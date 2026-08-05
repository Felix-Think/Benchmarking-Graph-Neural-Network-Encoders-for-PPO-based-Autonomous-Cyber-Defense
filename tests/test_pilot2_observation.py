from pathlib import Path

import numpy as np
import pytest

from acd_benchmark.cyber.scalable_scenario import SUBNETS, write_scalable_scenario
from acd_benchmark.wrappers.pilot2.env import Pilot2FlatEnv, Pilot2GraphEnv
from acd_benchmark.wrappers.pilot2.observation import allocate_subnet_quotas


def _scenario(tmp_path: Path) -> Path:
    return write_scalable_scenario(tmp_path / "cage2_16.yaml", 16)


@pytest.mark.parametrize(
    ("q", "expected_total"),
    ((1.0, 16), (0.75, 12), (0.5, 8), (0.25, 4)),
)
def test_fixed_budget_retention_is_exact_and_reproducible(
    tmp_path: Path,
    q: float,
    expected_total: int,
):
    scenario_path = _scenario(tmp_path)
    env_a = Pilot2GraphEnv(scenario_path, q_values=(q,), fixed_q=q)
    env_b = Pilot2GraphEnv(scenario_path, q_values=(q,), fixed_q=q)

    obs_a, info_a = env_a.reset(seed=17)
    obs_b, info_b = env_b.reset(seed=17)

    assert info_a["visible_host_count"] == expected_total
    assert info_a["intended_visible_host_count"] == expected_total
    assert sum(info_a["visible_count_by_subnet"].values()) == expected_total
    assert np.array_equal(obs_a.node_features, obs_b.node_features)
    assert info_a["visible_hosts"] == info_b["visible_hosts"]


def test_largest_remainder_quota_is_stratified(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    env = Pilot2GraphEnv(scenario_path)
    quotas = allocate_subnet_quotas(
        env.topology,
        retention_q=0.25,
        tie_break_order=SUBNETS,
    )

    assert quotas == {"Enterprise": 1, "Operational": 1, "User": 2}


def test_target_action_does_not_force_host_visibility(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    env = Pilot2GraphEnv(
        scenario_path,
        q_values=(0.0,),
        fixed_q=0.0,
        max_steps=3,
    )
    observation, _ = env.reset(seed=3)
    user_index = observation.hostnames.index("User0")
    analyse_index = env.action_names.index("Analyse(User0)")

    observation, *_ = env.step(analyse_index)

    assert observation.node_features[user_index, -1] == 0.0


def test_flat_pilot2_excludes_adjacency_by_default(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    graph_env = Pilot2GraphEnv(scenario_path, q_values=(1.0,), fixed_q=1.0)
    flat_env = Pilot2FlatEnv(scenario_path, q_values=(1.0,), fixed_q=1.0)

    graph, _ = graph_env.reset(seed=9)
    flat, _ = flat_env.reset(seed=9)

    assert flat.shape == (graph.node_features.size,)
    assert np.array_equal(flat.reshape(graph.node_features.shape), graph.node_features)


def test_mixed_q_exposure_is_reproducible_and_auditable(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    q_values = (1.0, 0.75, 0.5, 0.25)
    env_a = Pilot2GraphEnv(scenario_path, q_values=q_values, max_steps=1)
    env_b = Pilot2GraphEnv(scenario_path, q_values=q_values, max_steps=1)

    sequences = []
    for env in (env_a, env_b):
        observed_q = []
        for episode in range(32):
            env.reset(seed=41 if episode == 0 else None)
            observed_q.append(env.retention_q)
            sleep_index = env.action_names.index("Sleep")
            env.step(sleep_index)
        sequences.append(observed_q)

    assert sequences[0] == sequences[1]
    assert set(sequences[0]) == set(q_values)
    assert sum(env_a.q_episode_counts.values()) == 32
    assert sum(env_a.q_step_counts.values()) == 32
    assert all(count > 0 for count in env_a.q_episode_counts.values())
