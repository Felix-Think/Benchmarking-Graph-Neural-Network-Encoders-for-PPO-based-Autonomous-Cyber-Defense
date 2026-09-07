import numpy as np
import torch

from acd_benchmark.cyber.canonical_observation import (
    FEATURE_INDEX,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
)
from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.cyber.canonical_observation_v2 import V2_NODE_FEATURES
from acd_benchmark.rl.policies.gcn import GCNActorCritic
from acd_benchmark.rl.rollout.rollout import collect_rollout


def _environment_and_policy(episode_steps: int = 20):
    env = CC4CanonicalFlatEnv(
        episode_steps=episode_steps,
        seed=123,
        canonical_variant="action_aware",
    )
    agent = env.possible_agents[0]
    policy = GCNActorCritic(
        observation_dim=env.observation_space(agent).shape[0],
        num_actions=env.action_space(agent).n,
        gcn_hidden_dim=8,
        gcn_layers=2,
        latent_dim=16,
        head_hidden_dim=16,
    )
    return env, policy


def test_gcn_builds_canonical_membership_and_structural_edges():
    env, policy = _environment_and_policy()
    observations, _ = env.reset(seed=123)
    observation = torch.as_tensor(
        observations[env.possible_agents[0]],
        dtype=torch.float32,
    ).unsqueeze(0)

    nodes, membership, structural, context = policy._unpack(observation)
    edge_index = policy._batched_edge_index(membership, structural)

    assert nodes.shape == (1, NUM_NODES, len(NODE_FEATURES))
    assert context.shape == (1, 169)
    expected_edges = int(structural.count_nonzero()) + 2 * int(
        membership.count_nonzero()
    )
    assert edge_index.shape == (2, expected_edges)
    assert int(edge_index.min()) >= 0
    assert int(edge_index.max()) < NUM_NODES


def test_gcn_single_and_batch_outputs_respect_action_masks():
    env, policy = _environment_and_policy()
    observations, infos = env.reset(seed=123)
    agent = env.possible_agents[0]
    observation = torch.as_tensor(observations[agent], dtype=torch.float32)
    action_mask = torch.as_tensor(infos[agent]["action_mask"], dtype=torch.bool)

    action, log_prob, entropy, value = policy.get_action_and_value(
        observation,
        action_mask=action_mask,
    )
    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1,)
    assert action_mask[action.item()]

    observation_batch = torch.stack(
        [
            torch.as_tensor(observations[name], dtype=torch.float32)
            for name in env.possible_agents
        ]
    )
    mask_batch = torch.stack(
        [
            torch.as_tensor(infos[name]["action_mask"], dtype=torch.bool)
            for name in env.possible_agents
        ]
    )
    actions, _, _, values = policy.get_action_and_value(
        observation_batch,
        action_mask=mask_batch,
    )
    assert actions.shape == (5,)
    assert values.shape == (5,)
    assert mask_batch.gather(1, actions.unsqueeze(1)).all()


def test_gcn_encoding_changes_when_node_evidence_changes():
    env, policy = _environment_and_policy()
    observations, _ = env.reset(seed=123)
    agent = env.possible_agents[0]
    base = np.asarray(observations[agent], dtype=np.float32)
    changed = base.copy()

    canonical = env.last_canonical[agent]
    host_rows = np.flatnonzero(canonical.host_membership.sum(axis=1) > 0)
    assert host_rows.size > 0
    node_index = NUM_SUBNET_NODES + int(host_rows[0])
    flat_index = node_index * len(NODE_FEATURES) + FEATURE_INDEX["process_event_now"]
    changed[flat_index] = 1.0 - changed[flat_index]

    with torch.no_grad():
        base_latent = policy.encode(torch.from_numpy(base))
        changed_latent = policy.encode(torch.from_numpy(changed))
    assert not torch.allclose(base_latent, changed_latent)


def test_current_ppo_collects_a_complete_gcn_rollout():
    env, policy = _environment_and_policy(episode_steps=10)
    batch, _, simulator_steps = collect_rollout(
        env=env,
        policy=policy,
        seed=321,
        reward_scale=0.01,
    )

    assert simulator_steps == 9
    assert batch.observations.ndim == 2
    assert batch.actions.shape[0] == batch.observations.shape[0]
    assert batch.action_masks.shape[0] == batch.observations.shape[0]


def test_gcn_supports_v2_canonical_schema():
    env = CC4CanonicalFlatEnv(
        episode_steps=10,
        seed=123,
        canonical_schema="v2",
        semantic_action_mask=True,
    )
    agent = env.possible_agents[0]
    policy = GCNActorCritic(
        observation_dim=env.observation_space(agent).shape[0],
        num_actions=env.action_space(agent).n,
        gcn_hidden_dim=8,
        gcn_layers=2,
        latent_dim=16,
        head_hidden_dim=16,
        canonical_schema="v2",
    )
    observations, infos = env.reset(seed=123)
    observation = torch.as_tensor(observations[agent], dtype=torch.float32)
    nodes, _, _, _ = policy._unpack(observation.unsqueeze(0))
    action_mask = torch.as_tensor(infos[agent]["action_mask"], dtype=torch.bool)
    action, _, _, value = policy.get_action_and_value(
        observation,
        action_mask=action_mask,
    )

    assert nodes.shape == (1, NUM_NODES, len(V2_NODE_FEATURES))
    assert action_mask[action.item()]
    assert value.shape == (1,)
