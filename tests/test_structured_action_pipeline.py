import numpy as np
import torch

from acd_benchmark.cyber.canonical_observation import (
    AGENT_SUBNETS,
    NUM_SUBNET_NODES,
    SUBNET_INDEX,
)
from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.cyber.cc4_structured_action_env import (
    HOST_ACTIONS,
    NUM_STRUCTURED_ACTIONS,
    STRUCTURED_HOST_SLOTS,
    CC4StructuredActionEnv,
    _official_action_key,
)
from acd_benchmark.rl.policies.structured_action import (
    StructuredGCNActorCritic,
    StructuredNodeMLPActorCritic,
)


def _make_env(seed: int = 123):
    return CC4StructuredActionEnv(
        CC4CanonicalFlatEnv(
            episode_steps=30,
            seed=seed,
            canonical_schema="v2",
        )
    )


def test_structured_catalog_maps_to_same_native_action_semantics():
    env = _make_env()
    _, infos = env.reset(seed=123)

    for agent in env.possible_agents:
        assert env.action_space(agent).n == NUM_STRUCTURED_ACTIONS == 242
        assert len(env.action_labels(agent)) == NUM_STRUCTURED_ACTIONS
        mask = np.asarray(infos[agent]["action_mask"], dtype=np.bool_)
        mapping = env._structured_to_native[agent]
        native_labels = env.env.action_labels(agent)
        for action_id in np.flatnonzero(mask):
            native_id = int(mapping[action_id])
            assert native_id >= 0
            assert _official_action_key(native_labels[native_id]) == env._keys[agent][
                action_id
            ]


def test_structured_catalog_executes_one_valid_action_per_agent():
    env = _make_env()
    _, infos = env.reset(seed=123)
    actions = {
        agent: int(np.flatnonzero(infos[agent]["action_mask"])[0])
        for agent in env.possible_agents
    }

    observations, rewards, terminated, truncated, next_infos = env.step(actions)

    assert set(observations) == set(env.possible_agents)
    assert set(rewards) == set(env.possible_agents)
    assert not any(terminated.values())
    assert not any(truncated.values())
    for agent in env.possible_agents:
        assert next_infos[agent]["action_mask"].shape == (
            NUM_STRUCTURED_ACTIONS,
        )


def test_structured_policies_match_ppo_shapes_and_action_mask():
    env = _make_env()
    observations, infos = env.reset(seed=123)
    observation_batch = torch.as_tensor(
        np.stack([observations[agent] for agent in env.possible_agents]),
        dtype=torch.float32,
    )
    mask_batch = torch.as_tensor(
        np.stack([infos[agent]["action_mask"] for agent in env.possible_agents]),
        dtype=torch.bool,
    )
    policies = (
        StructuredNodeMLPActorCritic(
            observation_dim=observation_batch.shape[-1],
        ),
        StructuredGCNActorCritic(
            observation_dim=observation_batch.shape[-1],
        ),
    )

    for policy in policies:
        actions, log_probs, entropy, values = policy.get_action_and_value(
            observation_batch,
            action_mask=mask_batch,
        )
        assert actions.shape == (5,)
        assert log_probs.shape == (5,)
        assert entropy.shape == (5,)
        assert values.shape == (5,)
        assert mask_batch.gather(1, actions.unsqueeze(1)).all()


def test_nodemlp_host_scorer_is_slot_equivariant():
    env = _make_env()
    observations, _ = env.reset(seed=123)
    observation = torch.as_tensor(
        observations[env.possible_agents[4]],
        dtype=torch.float32,
    )
    policy = StructuredNodeMLPActorCritic(
        observation_dim=observation.shape[-1],
    )
    swapped = observation.clone()
    node_feature_count = len(policy.node_feature_names)
    nodes = swapped[: 57 * node_feature_count].reshape(57, node_feature_count)
    first = NUM_SUBNET_NODES
    second = NUM_SUBNET_NODES + 1
    original_first = nodes[first].clone()
    nodes[first] = nodes[second]
    nodes[second] = original_first

    original_logits, _ = policy(observation)
    swapped_logits, _ = policy(swapped)
    for action_index in range(len(HOST_ACTIONS)):
        offset = action_index * STRUCTURED_HOST_SLOTS
        torch.testing.assert_close(
            swapped_logits[0, offset],
            original_logits[0, offset + 1],
        )
        torch.testing.assert_close(
            swapped_logits[0, offset + 1],
            original_logits[0, offset],
        )


def test_controlled_subnet_slots_follow_structured_catalog_order():
    env = _make_env()
    observations, _ = env.reset(seed=123)
    policy = StructuredNodeMLPActorCritic(
        observation_dim=next(iter(observations.values())).shape[-1],
    )
    node_feature_count = len(policy.node_feature_names)

    for agent in env.possible_agents:
        observation = torch.as_tensor(observations[agent], dtype=torch.float32)
        nodes = observation[: 57 * node_feature_count].reshape(
            1,
            57,
            node_feature_count,
        )
        slots = policy._controlled_subnet_slots(
            nodes,
            policy._controlled_index,
        )[0]
        expected = torch.as_tensor(
            [SUBNET_INDEX[subnet] for subnet in AGENT_SUBNETS[agent]],
        )
        torch.testing.assert_close(slots[: expected.numel()].cpu(), expected)
