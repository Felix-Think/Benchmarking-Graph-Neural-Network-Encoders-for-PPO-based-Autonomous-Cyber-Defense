"""Opt-in byte-level refactor check for the pinned local CPU runtime.
Run with PYTHONHASHSEED=42, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1.
Use --before only when PYTHONPATH points to the archive source tree.
This is deliberately not a default cross-platform pytest assertion.
"""
import hashlib
import json
import sys
import numpy as np
import torch
from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.cyber.cc4_structured_action_env import CC4StructuredActionEnv
from acd_benchmark.rl.ppo.training import PPOTrainingConfig, seed_runtime, train_policy

if "--before" in sys.argv:
    from acd_benchmark.rl.policies.structured_action import StructuredNodeMLPActorCritic, StructuredGCNActorCritic
else:
    from acd_benchmark.rl.policies.nodemlp import StructuredNodeMLPActorCritic
    from acd_benchmark.rl.policies.gcn import StructuredGCNActorCritic

def digest_arrays(arrays):
    result = hashlib.sha256()
    for value in arrays:
        result.update(np.asarray(value).tobytes())
    return result.hexdigest()

def make_env():
    return CC4StructuredActionEnv(CC4CanonicalFlatEnv(
        seed=123, episode_steps=80, canonical_schema="v2",
        environment_distribution="random_host", passive_memory_ttl=5,
        analysis_memory_ttl=20,
    ))

torch.set_num_threads(1)
env = make_env()
observations, infos = env.reset(seed=123)
arrays = []
rng = np.random.default_rng(456)
for tick in range(60):
    arrays.extend(observations[a] for a in env.possible_agents)
    arrays.extend(infos[a]["action_mask"] for a in env.possible_agents)
    actions = {a: int(rng.choice(np.flatnonzero(infos[a]["action_mask"])))
               for a in env.possible_agents if infos[a]["decision_ready"]}
    observations, rewards, terminated, truncated, infos = env.step(actions)
    arrays.append([rewards[a] for a in env.possible_agents])

output = {"trajectory": digest_arrays(arrays)}
obs_batch = torch.tensor(np.stack(list(observations.values())))
for name, cls, kwargs in [
    ("nodemlp", StructuredNodeMLPActorCritic, {"node_hidden_dim": 128}),
    ("gcn", StructuredGCNActorCritic, {"gcn_hidden_dim": 128}),
]:
    seed_runtime(42)
    policy = cls(observation_dim=5185, **kwargs)
    with torch.no_grad():
        logits, values = policy(obs_batch)
    output[name] = {"parameters": sum(p.numel() for p in policy.parameters()),
                    "forward": digest_arrays([logits.numpy(), values.numpy()])}
    result = train_policy(make_env(), policy, PPOTrainingConfig(
        total_simulator_steps=80, episodes_per_update=2, update_epochs=1,
        minibatch_size=64, seed=42, entropy_coef=0.001, target_kl=0.015,
    ))
    output[name]["trained_weights"] = digest_arrays(
        [v.detach().cpu().numpy() for v in policy.state_dict().values()])
    output[name]["returns"] = result.episode_returns
print(json.dumps(output, sort_keys=True))
