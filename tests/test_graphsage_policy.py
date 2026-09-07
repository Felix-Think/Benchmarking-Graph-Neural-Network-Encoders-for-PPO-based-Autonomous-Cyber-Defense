import numpy as np
import torch

from acd_benchmark.cyber.canonical_observation import (
    FEATURE_INDEX,
    NODE_FEATURES,
    NUM_SUBNET_NODES,
)
from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.policies.gcn import GCNActorCritic
from acd_benchmark.rl.policies.graphsage import GraphSAGEActorCritic
from acd_benchmark.rl.rollout.rollout import collect_rollout


def _environment_and_policy(episode_steps: int = 20):
    env = CC4CanonicalFlatEnv(
        episode_steps=episode_steps,
        seed=123,
        canonical_variant="action_aware",
    )
    agent = env.possible_agents[0]
    policy = GraphSAGEActorCritic(
        observation_dim=env.observation_space(agent).shape[0],
        num_actions=env.action_space(agent).n,
        sage_hidden_dim=28,
        sage_layers=2,
        latent_dim=128,
        head_hidden_dim=128,
    )
    return env, policy


def _parameter_count(policy: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in policy.parameters())


def test_graphsage_is_parameter_matched_to_flat_and_gcn():
    env, sage = _environment_and_policy()
    agent = env.possible_agents[0]
    observation_dim = env.observation_space(agent).shape[0]
    num_actions = env.action_space(agent).n
    flat = FlatActorCritic(observation_dim, num_actions)
    gcn = GCNActorCritic(
        observation_dim,
        num_actions,
        gcn_hidden_dim=28,
        gcn_layers=2,
    )

    flat_count = _parameter_count(flat)
    assert abs(_parameter_count(gcn) - flat_count) / flat_count < 0.01
    assert abs(_parameter_count(sage) - flat_count) / flat_count < 0.01


def test_graphsage_outputs_respect_action_masks_and_dynamic_evidence():
    env, policy = _environment_and_policy()
    observations, infos = env.reset(seed=123)
    agent = env.possible_agents[0]
    base = np.asarray(observations[agent], dtype=np.float32)
    action_mask = torch.as_tensor(infos[agent]["action_mask"], dtype=torch.bool)

    action, log_prob, entropy, value = policy.get_action_and_value(
        torch.from_numpy(base),
        action_mask=action_mask,
    )
    assert action.shape == log_prob.shape == entropy.shape == value.shape == (1,)
    assert action_mask[action.item()]

    changed = base.copy()
    canonical = env.last_canonical[agent]
    host_rows = np.flatnonzero(canonical.host_membership.sum(axis=1) > 0)
    node_index = NUM_SUBNET_NODES + int(host_rows[0])
    flat_index = node_index * len(NODE_FEATURES) + FEATURE_INDEX["process_event_now"]
    changed[flat_index] = 1.0 - changed[flat_index]
    with torch.no_grad():
        assert not torch.allclose(
            policy.encode(torch.from_numpy(base)),
            policy.encode(torch.from_numpy(changed)),
        )


def test_current_ppo_collects_a_complete_graphsage_rollout():
    env, policy = _environment_and_policy(episode_steps=10)
    batch, _, simulator_steps = collect_rollout(
        env=env,
        policy=policy,
        seed=321,
        reward_scale=0.01,
    )

    assert simulator_steps == 9
    assert batch.actions.shape[0] == batch.observations.shape[0]
    assert batch.action_masks.shape[0] == batch.observations.shape[0]
