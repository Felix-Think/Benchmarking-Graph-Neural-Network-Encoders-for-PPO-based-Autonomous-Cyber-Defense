from pathlib import Path

import torch

from acd_benchmark.cyber.scalable_scenario import write_scalable_scenario
from acd_benchmark.rl.policies import GCNActorCritic, infer_action_head_spec
from acd_benchmark.rl.policies.graph_base import graph_to_tensors
from acd_benchmark.rl.ppo.config import PPOConfig
from acd_benchmark.rl.ppo.optimizer import make_optimizer
from acd_benchmark.rl.ppo.training import train_update
from acd_benchmark.rl.stage1_factory import (
    build_stage1_policy,
    make_stage1_env,
    resolve_model_hidden_dim,
)
from acd_benchmark.wrappers.stage1.env import Stage1GraphEnv


def _scenario(tmp_path: Path) -> Path:
    return write_scalable_scenario(tmp_path / "cage2_16.yaml", 16)


def _gcn_env(tmp_path: Path) -> Stage1GraphEnv:
    return Stage1GraphEnv(
        _scenario(tmp_path),
        observation_probability=1.0,
        max_steps=5,
    )


def test_gcn_policy_matches_stage1_action_space(tmp_path: Path):
    env = _gcn_env(tmp_path)
    observation, _ = env.reset(seed=1)
    action_head = infer_action_head_spec(env.action_specs, observation.hostnames)
    policy = GCNActorCritic(
        node_feature_dim=observation.node_features.shape[1],
        hidden_dim=16,
        action_head=action_head,
    )

    node_features, edge_index = graph_to_tensors(observation, torch.device("cpu"))
    logits, value = policy(node_features, edge_index)

    assert observation.node_features.shape == (16, 15)
    assert action_head.action_count == 178
    assert logits.shape == (178,)
    assert value.shape == ()


def test_gcn_ppo_single_update_runs(tmp_path: Path):
    env = _gcn_env(tmp_path)
    observation, _ = env.reset(seed=1)
    action_head = infer_action_head_spec(env.action_specs, observation.hostnames)
    policy = GCNActorCritic(
        node_feature_dim=observation.node_features.shape[1],
        hidden_dim=16,
        action_head=action_head,
    )
    config = PPOConfig(n_steps=4, batch_size=2, ppo_epochs=1)

    next_observation, losses = train_update(
        env=env,
        policy=policy,
        optimizer=make_optimizer(policy, config),
        observation=observation,
        config=config,
        device=torch.device("cpu"),
    )

    assert next_observation.node_features.shape == observation.node_features.shape
    assert set(losses) >= {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "gradient_norm",
        "mean_completed_episode_return",
    }


def test_gcn_batched_forward_matches_individual_forward(tmp_path: Path):
    env = _gcn_env(tmp_path)
    observation, _ = env.reset(seed=1)
    action_head = infer_action_head_spec(env.action_specs, observation.hostnames)
    policy = GCNActorCritic(
        node_feature_dim=observation.node_features.shape[1],
        hidden_dim=16,
        action_head=action_head,
    )
    prepared = policy.prepare_observation(observation, torch.device("cpu"))

    individual_logits, individual_value = policy.forward_prepared(prepared)
    batch_logits, batch_values = policy.forward_batch([prepared, prepared])

    assert batch_logits.shape == (2, 178)
    assert batch_values.shape == (2,)
    assert torch.allclose(batch_logits[0], individual_logits, atol=1e-6)
    assert torch.allclose(batch_values[0], individual_value, atol=1e-6)


def test_gcn_is_connected_to_stage1_factory(tmp_path: Path):
    scenario_path = _scenario(tmp_path)
    env = make_stage1_env(
        "gcn",
        scenario_path=scenario_path,
        node_count=16,
        observation_probability=1.0,
        max_steps=3,
    )
    observation, _ = env.reset(seed=1)
    hidden_dim = resolve_model_hidden_dim(
        "gcn",
        {"hidden_dim": 32, "hidden_dim_by_model": {"gcn": 16}},
    )
    policy = build_stage1_policy(
        "gcn",
        observation,
        env,
        hidden_dim=hidden_dim,
    )

    assert isinstance(env, Stage1GraphEnv)
    assert isinstance(policy, GCNActorCritic)
    assert hidden_dim == 16
