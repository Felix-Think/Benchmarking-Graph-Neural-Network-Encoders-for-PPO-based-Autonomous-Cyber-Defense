from pathlib import Path

import numpy as np

from acd_benchmark.cyber.action_space import build_blue_discrete_action_specs
from acd_benchmark.cyber.env_factory import make_scalable_cage2_env
from acd_benchmark.cyber.scalable_scenario import (
    allocate_subnet_counts,
    build_static_topology,
    write_scalable_scenario,
)


def test_16_node_topology_contract():
    topology = build_static_topology(16)

    assert allocate_subnet_counts(16) == {
        "Enterprise": 5,
        "Operational": 5,
        "User": 6,
    }
    assert len(topology.hostnames) == 16
    assert topology.critical_hosts == ("Op_Server0",)
    assert np.array_equal(topology.adjacency, topology.adjacency.T)
    assert not np.any(np.diag(topology.adjacency))
    assert int(topology.adjacency.sum()) == 84


def test_generated_16_node_scenario_resets_and_steps(tmp_path: Path):
    scenario_path = write_scalable_scenario(tmp_path / "cage2_16.yaml", 16)
    env = make_scalable_cage2_env(scenario_path)
    reset_result = env.reset("Blue", seed=1)
    hostnames = build_static_topology(16).hostnames
    action_specs = build_blue_discrete_action_specs(
        env.get_action_space("Blue"),
        hostnames,
    )

    assert len([value for value in reset_result.observation.values() if isinstance(value, dict)]) == 16
    assert len(action_specs) == 178
    step_result = env.step("Blue", action_specs[0].instantiate())
    assert isinstance(step_result.reward, float)
