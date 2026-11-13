import pytest
import torch
import numpy as np
from collections import Counter

from acd_benchmark.cyber.action_space import ACTION_SPACE
from acd_benchmark.cyber.blue_table_env import BlueTableEnv, HOST_NAMES
from acd_benchmark.rl.ppo.training import seed_runtime
from experiments.cage2_revalidation.policies import make_policy, public_graph
from experiments.cage2_revalidation.run import (
    action_id,
    concentration,
    intervention_observations,
)


def test_graph_uses_only_public_scenario_links():
    graph = public_graph()
    assert graph["node_names"][:13] == list(HOST_NAMES)
    assert len(graph["node_names"]) == 16
    edges = set(zip(*graph["edge_index"]))
    assert len(edges) == 32
    assert all(source >= 13 or target >= 13 for source, target in edges)
    assert all((target, source) in edges for source, target in edges)


def test_blue_table_host_order_matches_action_catalogue():
    env = BlueTableEnv(max_step=3)
    observation, _ = env.reset(seed=0)
    rows = env.env._create_blue_table(success=None)._rows
    assert tuple(row[2] for row in rows) == HOST_NAMES
    assert observation.shape == (13, 4)
    assert len(ACTION_SPACE) == 41


@pytest.mark.parametrize("model", ("flat", "gcn", "graphsage"))
def test_policy_batching_gradients_and_initial_shared_heads(model):
    seed_runtime(0)
    reference = make_policy("flat", public_graph())
    seed_runtime(0)
    policy = make_policy(model, public_graph())
    for name in ("actor", "critic"):
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                getattr(reference, name).parameters(),
                getattr(policy, name).parameters(),
            )
        )
    observation = torch.rand(5, 13, 4)
    logits, values = policy(observation)
    assert logits.shape == (5, 41)
    assert values.shape == (5, 1)
    for index in range(5):
        single_logits, single_value = policy(observation[index])
        assert torch.allclose(logits[index], single_logits[0], atol=1e-6)
        assert torch.allclose(values[index], single_value[0], atol=1e-6)
    (logits.square().mean() + values.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in policy.parameters())
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in policy.parameters()
    )


def test_graph_checkpoint_roundtrip(tmp_path):
    policy = make_policy("gcn", public_graph())
    path = tmp_path / "graph.pt"
    torch.save(policy.state_dict(), path)
    restored = make_policy("gcn", public_graph())
    restored.load_state_dict(torch.load(path, weights_only=True))
    observation = torch.rand(13, 4)
    assert torch.equal(policy(observation)[0], restored(observation)[0])


def test_action_concentration_distinguishes_restore_and_target_share():
    counts = Counter(
        {
            action_id("Restore", "Op_Server0"): 60,
            action_id("Analyse", "User0"): 30,
            action_id("Monitor"): 10,
        }
    )
    result = concentration(counts)
    assert result["restore_share"] == pytest.approx(0.6)
    assert result["top_action"] == "Restore(Op_Server0)"
    assert result["top_action_share"] == pytest.approx(0.6)
    assert result["dominant_target_share_of_targeted_actions"] == pytest.approx(2 / 3)
    assert sum(result["action_counts"].values()) == 100


def test_observation_probes_do_not_mutate_source():
    observation = np.arange(52, dtype=np.float32).reshape(13, 4)
    original = observation.copy()
    probes = intervention_observations(observation)
    assert np.array_equal(observation, original)
    assert not probes["zero_all"].any()
    assert not probes["zero_activity"][:, :2].any()
    assert np.array_equal(probes["zero_activity"][:, 2:], observation[:, 2:])
    assert not probes["zero_compromise"][:, 2:].any()
    assert np.array_equal(probes["zero_compromise"][:, :2], observation[:, :2])
