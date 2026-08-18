import numpy as np

from acd_benchmark.cyber.cc4_flat_env import CC4FlatEnv


env = CC4FlatEnv(episode_steps=20, seed=123)
observations, info = env.reset(seed=123)

print(observations.keys()) #dict_keys(['blue_agent_0', 'blue_agent_1', 'blue_agent_2', 'blue_agent_3', 'blue_agent_4'])

for agent, observation in observations.items():
    mask = np.asarray(info[agent]["action_mask"], dtype=bool)

    print(
        agent,
        observation.shape,
        env.observation_space(agent),
        env.action_space(agent),
        f"valid_actions={mask.sum()}",
    )

    assert observation.shape == env.observation_space(agent).shape
    assert len(mask) == env.action_space(agent).n
    assert mask.any()

rng = np.random.default_rng(123)

for step in range(20):
    actions = {
        agent: int(
            rng.choice(
                np.flatnonzero(info[agent]["action_mask"])
            )
        )
        for agent in observations
    }

    observations, rewards, terminated, truncated, info = env.step(
        actions
    )

    print(step, actions, rewards)

    if all(
        terminated.get(agent, False)
        or truncated.get(agent, False)
        for agent in env.possible_agents
    ):
        break



