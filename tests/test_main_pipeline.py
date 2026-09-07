import json

import numpy as np
import pytest
import torch

from acd_benchmark.cyber.benchmark_env import make_benchmark_env
from acd_benchmark.cyber.dynamic_host_observation_mask import FixedBudgetDynamicHostMask
from acd_benchmark.cyber.observation_schema import (
    flat_observation_dim,
    deserialize_flat_view,
)
from acd_benchmark.rl.policies.factory import build_policy
from acd_benchmark.rl.ppo.advantages import compute_smdp_gae
from acd_benchmark.rl.ppo.evaluation import evaluate_policy_with_preserved_rng
from scripts.training.train import build_parser, run_training
from scripts.evaluation.evaluate import main as evaluate


def policy_config(encoder):
    return dict(
        policy_type=f"structured_{encoder}",
        canonical_schema="v2",
        observation_dim=5185,
        num_actions=242,
        node_hidden_dim=128,
        gcn_hidden_dim=128,
        gcn_layers=2,
        node_embedding_dim=64,
        context_dim=128,
        head_hidden_dim=128,
    )


def test_matched_default_parameter_budget():
    assert flat_observation_dim() == 5185
    for encoder in ("nodemlp", "gcn"):
        policy = build_policy(policy_config(encoder))
        assert sum(p.numel() for p in policy.parameters()) == 119881


@pytest.mark.parametrize(
    "condition",
    ["full", "no_evidence", "action_state_only", "topology_only", "zero_with_mask"],
)
def test_gcn_supports_all_information_conditions_even_with_no_edges(condition):
    env = make_benchmark_env({"episode_steps": 10, "observation_condition": condition})
    observations, infos = env.reset(seed=123)
    policy = build_policy(policy_config("gcn"))
    batch = torch.tensor(np.stack(list(observations.values())))
    masks = torch.tensor(
        np.stack([infos[a]["action_mask"] for a in env.possible_agents])
    )
    actions, log_prob, entropy, value = policy.get_action_and_value(
        batch, action_mask=masks
    )
    assert masks.gather(1, actions[:, None]).all()
    assert torch.isfinite(log_prob).all() and torch.isfinite(value).all()
    assert torch.isfinite(entropy).all()


def test_gae_uses_duration_and_stops_trace_but_not_bootstrap_at_truncation():
    advantages, returns = compute_smdp_gae(
        rewards=torch.tensor([1.0, 100.0]),
        values=torch.zeros(2),
        next_values=torch.tensor([4.0, 999.0]),
        durations=torch.tensor([2, 1]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([True, False]),
        gamma=0.5,
        gae_lambda=0.9,
    )
    torch.testing.assert_close(advantages, torch.tensor([2.0, 100.0]))
    torch.testing.assert_close(returns, advantages)


def test_schema_round_trip_and_persistent_mask_preserve_native_mask():
    env = FixedBudgetDynamicHostMask(
        make_benchmark_env({"episode_steps": 10}),
        retention=0.5,
        mask_process="persistent",
        mask_seed=321,
    )
    observations, infos = env.reset(seed=123)
    original = {a: visible.copy() for a, visible in env.last_visibility.items()}
    for agent, observation in observations.items():
        np.testing.assert_array_equal(
            deserialize_flat_view(observation).flat_view(), observation
        )
        assert infos[agent]["dynamic_hosts_visible"] == int(
            np.floor(0.5 * infos[agent]["dynamic_hosts_total"] + 0.5)
        )
    for _ in range(3):
        actions = {
            a: int(np.flatnonzero(infos[a]["action_mask"])[0])
            for a in env.possible_agents
            if infos[a]["decision_ready"]
        }
        _, _, _, _, infos = env.step(actions)
        for agent in env.possible_agents:
            np.testing.assert_array_equal(env.last_visibility[agent], original[agent])


def test_periodic_evaluation_preserves_torch_rng_and_training_mode():
    env = make_benchmark_env({"episode_steps": 10})
    policy = build_policy(policy_config("nodemlp"))
    before = torch.random.get_rng_state().clone()
    result1 = evaluate_policy_with_preserved_rng(
        env, policy, episodes=1, seed=123, deterministic=False, action_seed=456
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert policy.training
    result2 = evaluate_policy_with_preserved_rng(
        env, policy, episodes=1, seed=123, deterministic=False, action_seed=456
    )
    assert result1.episode_returns == result2.episode_returns
    assert result1.action_counts == result2.action_counts


@pytest.mark.parametrize("encoder", ["nodemlp", "gcn"])
def test_train_eval_commands_save_final_best_and_diagnostics(tmp_path, encoder):
    root = tmp_path / encoder
    args = build_parser().parse_args(
        [
            "--encoder",
            encoder,
            "--run-dir",
            str(root),
            "--total-simulator-steps",
            "10",
            "--episode-steps",
            "10",
            "--episodes-per-update",
            "2",
            "--update-epochs",
            "1",
            "--eval-every-updates",
            "1",
            "--periodic-eval-episodes",
            "1",
            "--eval-episodes",
            "1",
            "--print-every-updates",
            "1",
        ]
    )
    run_training(args)
    seed_dir = root / "seed_42"
    assert (seed_dir / "checkpoint.pt").is_file()
    assert (seed_dir / "best_checkpoint.pt").is_file()
    assert list((seed_dir / "tensorboard").glob("events.out.tfevents.*"))
    config = json.loads((seed_dir / "config.json").read_text())
    assert config["parameter_count"] == 119881
    assert (
        config["canonical_parser_version"]
        == "raw_blue_observation_v4_relational_events"
    )
    for checkpoint in ("final", "best"):
        output = tmp_path / f"{encoder}_{checkpoint}.json"
        evaluate(
            [
                "--run-dir",
                str(root),
                "--checkpoint",
                checkpoint,
                "--episodes",
                "1",
                "--output",
                str(output),
                "--counterfactual",
                "zero_evidence",
            ]
        )
        report = json.loads(output.read_text())
        assert report["deterministic"]["deterministic"] is True
        assert report["stochastic"]["deterministic"] is False
    with pytest.raises(FileExistsError):
        run_training(args)
