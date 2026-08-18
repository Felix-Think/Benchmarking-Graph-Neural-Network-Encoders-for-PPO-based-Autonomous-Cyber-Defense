import numpy as np


from acd_benchmark.cyber.cc4_env import BLUE_AGENTS, make_cc4_env

from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Shared.Enums import TernaryEnum

class CC4FlatEnv:
    def __init__(
        self,
        episode_steps: int = 100,
        seed: int | None = None,
    ):
        cyborg = make_cc4_env(
            episode_steps=episode_steps,
            seed=seed,
        )
        self.env = BlueFlatWrapper(
            cyborg,
            pad_spaces=True,
        )
        self.possible_agents = BLUE_AGENTS

    def _normalise_episode_end(
        self,
        raw_terminated,
        raw_truncated,
    ):
        raw_done = {
            agent: bool(
                raw_terminated.get(agent, False)
                or raw_truncated.get(agent, False)
            )
            for agent in self.possible_agents
        }

        # CAGE 4 hiện chỉ kết thúc do đạt episode_steps.
        terminated = {
            agent: False
            for agent in self.possible_agents
        }
        truncated = raw_done

        return terminated, truncated

    def _add_decision_status(self, infos):
        for agent in self.possible_agents:
            raw_observation = self.env.env.get_observation(agent)

            infos[agent]["decision_ready"] = (raw_observation.get("success") != TernaryEnum.IN_PROGRESS)

        return infos
    
    def reset(self, seed: int | None = None):
        observations, info = self.env.reset(seed=seed)
        observations = {
            agent: np.asarray(observation, dtype=np.float32)
            for agent, observation in observations.items()
        }
        return observations, self._add_decision_status(info)

    def step(self, actions: dict[str, int]):
        observations, rewards, raw_terminated, raw_truncated, info = (
            self.env.step(actions=actions)
        )
        
        terminated, truncated = self._normalise_episode_end(raw_terminated, raw_truncated)
        
        observations = {
            agent: np.asarray(observation, dtype=np.float32)
            for agent, observation in observations.items()
        }
        infos = self._add_decision_status(info)
        return observations, rewards, terminated, truncated, infos

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_labels(self, agent: str):
        return self.env.action_labels(agent)

if __name__ == "__main__":
    env = CC4FlatEnv()
    observations, info = env.reset(seed=123)

    for agent in BLUE_AGENTS:
        observation = observations[agent]
        action_mask = np.asarray(
            info[agent]["action_mask"],
            dtype=np.bool_,
        )
        labels = env.action_labels(agent)
        action_count = env.action_space(agent).n

        print(f"\nAgent: {agent}")
        print(f"Observation shape: {observation.shape}")
        print(f"Observation dtype: {observation.dtype}")
        print(f"Action count: {action_count}")
        print(f"Action-mask shape: {action_mask.shape}")
        print(f"Valid actions: {action_mask.sum()}")
        print(f"Action-label count: {len(labels)}")

        assert len(labels) == action_count
        assert action_mask.shape == (action_count,)