from pathlib import Path

import numpy as np

from acd_benchmark.cyber.scalable_scenario import write_scalable_scenario
from acd_benchmark.wrappers.stage1.env import Stage1FlatEnv, Stage1GraphEnv
from acd_benchmark.wrappers.stage1.observation import STATIC_FEATURE_NAMES, STAGE1_FEATURE_NAMES


def _scenario(tmp_path: Path) -> Path:
    return write_scalable_scenario(tmp_path / "cage2_16.yaml", 16)


def test_stage1_full_and_flat_observations_match(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    graph_env = Stage1GraphEnv(scenario_path, observation_probability=1.0)
    flat_env = Stage1FlatEnv(scenario_path, observation_probability=1.0)

    graph, graph_info = graph_env.reset(seed=7)
    flat, flat_info = flat_env.reset(seed=7)

    assert graph.node_features.shape == (16, 15)
    assert graph.feature_names == STAGE1_FEATURE_NAMES
    assert graph.edge_index.shape == (2, 84)
    assert np.all(graph.node_features[:, -1] == 1.0)
    assert flat.shape == (496,)
    assert np.array_equal(flat[: 16 * 15].reshape(16, 15), graph.node_features)
    assert np.array_equal(flat[16 * 15 :].reshape(16, 16), graph.adjacency)
    assert graph_info["visible_host_count"] == flat_info["visible_host_count"] == 16


def test_masking_is_reproducible_and_preserves_static_topology(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    full_env = Stage1GraphEnv(scenario_path, observation_probability=1.0)
    masked_env_a = Stage1GraphEnv(scenario_path, observation_probability=0.25)
    masked_env_b = Stage1GraphEnv(scenario_path, observation_probability=0.25)

    full, _ = full_env.reset(seed=11)
    masked_a, _ = masked_env_a.reset(seed=11)
    masked_b, _ = masked_env_b.reset(seed=11)
    static_count = len(STATIC_FEATURE_NAMES)

    assert np.array_equal(masked_a.node_features, masked_b.node_features)
    assert np.array_equal(masked_a.node_features[:, :static_count], full.node_features[:, :static_count])
    assert np.array_equal(masked_a.adjacency, full.adjacency)
    hidden = masked_a.node_features[:, -1] == 0.0
    dynamic = masked_a.node_features[:, static_count:-1]
    assert np.all(dynamic[hidden] == 0.0)


def test_action_feedback_forces_visibility_for_current_and_previous_step(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    env = Stage1GraphEnv(scenario_path, observation_probability=0.0, max_steps=5)
    observation, _ = env.reset(seed=3)
    user_index = observation.hostnames.index("User0")
    analyse_index = env.action_names.index("Analyse(User0)")
    sleep_index = env.action_names.index("Sleep")

    observation, *_ = env.step(analyse_index)
    assert observation.node_features[user_index, -1] == 1.0
    observation, *_ = env.step(sleep_index)
    assert observation.node_features[user_index, -1] == 1.0
    observation, *_ = env.step(sleep_index)
    assert observation.node_features[user_index, -1] == 0.0
