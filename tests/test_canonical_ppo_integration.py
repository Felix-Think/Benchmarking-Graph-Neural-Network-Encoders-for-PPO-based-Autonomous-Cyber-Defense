import math

import pytest
import torch

from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.policies.gcn import GCNActorCritic
from acd_benchmark.rl.ppo.update import ppo_update
from acd_benchmark.rl.rollout.rollout import collect_rollout


def test_current_ppo_collects_a_complete_canonical_rollout():
    env = CC4CanonicalFlatEnv(
        episode_steps=10,
        seed=321,
        canonical_variant="action_aware",
    )
    agent = env.possible_agents[0]
    policy = FlatActorCritic(
        observation_dim=env.observation_space(agent).shape[0],
        num_actions=env.action_space(agent).n,
        encoder_hidden_dim=16,
        latent_dim=16,
        head_hidden_dim=16,
    )

    batch, _, simulator_steps = collect_rollout(
        env=env,
        policy=policy,
        seed=321,
        reward_scale=0.01,
    )

    assert simulator_steps == 9
    assert batch.observations.ndim == 2
    assert batch.observations.shape[1] == env.observation_space(agent).shape[0]
    assert batch.actions.shape[0] == batch.observations.shape[0]
    assert batch.action_masks.shape[0] == batch.observations.shape[0]


@pytest.mark.parametrize("policy_type", ["flat", "gcn"])
def test_v2_semantic_mask_completes_rollout_and_ppo_update(policy_type):
    env = CC4CanonicalFlatEnv(
        episode_steps=10,
        seed=654,
        canonical_variant="action_aware",
        canonical_schema="v2",
        passive_memory_ttl=4,
        analysis_memory_ttl=4,
        semantic_action_mask=True,
    )
    agent = env.possible_agents[0]
    common = {
        "observation_dim": env.observation_space(agent).shape[0],
        "num_actions": env.action_space(agent).n,
        "latent_dim": 16,
        "head_hidden_dim": 16,
    }
    if policy_type == "flat":
        policy = FlatActorCritic(encoder_hidden_dim=16, **common)
    else:
        policy = GCNActorCritic(
            gcn_hidden_dim=8,
            gcn_layers=2,
            canonical_schema="v2",
            **common,
        )

    batch, _, simulator_steps = collect_rollout(
        env=env,
        policy=policy,
        seed=654,
        reward_scale=0.01,
    )
    selected_actions_are_valid = batch.action_masks.gather(
        1, batch.actions.unsqueeze(1)
    ).squeeze(1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    metrics = ppo_update(
        policy=policy,
        batch=batch,
        optimizer=optimizer,
        update_epochs=2,
        minibatch_size=16,
        target_kl=0.05,
    )

    assert simulator_steps == 9
    assert selected_actions_are_valid.all()
    assert metrics["optimizer_steps"] > 0
    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "normalized_entropy",
        "approx_kl",
        "grad_norm",
        "total_loss",
    ):
        assert math.isfinite(metrics[name])
    assert 0.0 <= metrics["normalized_entropy"] <= 1.0 + 1e-6
    assert 1.0 <= metrics["valid_action_count_mean"] <= env.action_space(agent).n
