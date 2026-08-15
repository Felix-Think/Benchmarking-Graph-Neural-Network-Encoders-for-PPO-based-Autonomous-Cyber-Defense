from CybORG.Agents.Wrappers import BlueTableWrapper

from acd_benchmark.cyber.cc2_env import (
    get_default_scenario_path,
    make_cc2_env
)
import numpy as np

# Define 13 hosts

HOST_NAMES = (
    "Defender",
    "Enterprise0", "Enterprise1", "Enterprise2",
    "Op_Host0", "Op_Host1", "Op_Host2", "Op_Server0",
    "User0", "User1", "User2", "User3", "User4",
)

class BlueTableEnv:
    def __init__(self, max_step : int = 200 ):
        scenario_path = get_default_scenario_path()
        cyborg = make_cc2_env(scenario_path, red_agent = None)
        self.env = BlueTableWrapper(cyborg, output_mode = "vector")
        self.step_count = 0
        self.max_step = max_step

    @staticmethod 
    def _to_node_observation(vector):
        vector = np.array(vector, dtype = np.float32)
        return vector.reshape(len(HOST_NAMES), 4)

    def reset(self, seed : int = None):
        self.step_count = 0
        if seed is not None:
            self.env.set_seed(seed)
        raw = self.env.reset(agent = "Blue")
        obs = self._to_node_observation(raw.observation)
        return obs, raw
    
    def step(self, action):
        self.step_count += 1
        raw = self.env.step(agent = "Blue", action = action)
        obs = self._to_node_observation(raw.observation)
        terminated = bool(raw.done)
        truncated = self.step_count >= self.max_step and not terminated
                      
        return obs, raw.reward, terminated, truncated, raw 

    def get_last_action(self, agent):
        "Agent is either Red or Blue"
        return self.env.get_last_action(agent)
    
if __name__ == "__main__":
    env = BlueTableEnv()
    observation, result = env.reset(seed=123)

    print(observation)
    print(observation.shape)
    print(observation.dtype)

    assert observation.shape == (13, 4)
    assert observation.dtype == np.float32