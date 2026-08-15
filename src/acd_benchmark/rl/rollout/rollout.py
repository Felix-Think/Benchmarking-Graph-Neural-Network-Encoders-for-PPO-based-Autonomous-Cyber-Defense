from dataclasses import dataclass

import torch

from acd_benchmark.rl.ppo.advantages import comput_gae
from acd_benchmark.cyber.action_space import build_action

@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    def __init__(self, device : torch.device):
        self.clear()
        self.device = device
    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.next_values = []
        self.terminated = []
        self.truncated = []


    def __len__(self):
        return len(self.rewards)

    def add(
        self,
        observation,
        action,
        log_prob,
        reward,
        value,
        next_value,
        terminated,
        truncated,
    ):
        self.observations.append(
            observation.detach().to(self.device)
        )
        self.actions.append(
            action.detach().to(self.device).reshape(())
        )
        self.log_probs.append(
            log_prob.detach().to(self.device).reshape(())
        )

        self.rewards.append(float(reward))

        self.values.append(
            value.detach().to(self.device).reshape(())
        )
        self.next_values.append(
            next_value.detach().to(self.device).reshape(())
        )

        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    def build(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> RolloutBatch:
        observations = torch.stack(
            self.observations
        ).to(self.device)

        actions = torch.stack(
            self.actions
        ).long().to(self.device)

        old_log_probs = torch.stack(
            self.log_probs
        ).float().to(self.device)

        rewards = torch.tensor(
            self.rewards,
            dtype=torch.float32,
            device=self.device,
        )

        values = torch.stack(
            self.values
        ).float().to(self.device)

        next_values = torch.stack(
            self.next_values
        ).float().to(self.device)

        terminated = torch.tensor(
            self.terminated,
            dtype=torch.bool,
            device=self.device,
        )

        truncated = torch.tensor(
            self.truncated,
            dtype=torch.bool,
            device=self.device,
        )

        advantages, returns = comput_gae(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        return RolloutBatch(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            advantages=advantages,
            returns=returns,
        )

def collect_rollout(env, policy, initial_observation, rollout_steps, gamma=0.99, gae_lambda=0.95):
    device = next(policy.parameters()).device
    buffer = RolloutBuffer(device = device)
    observation = initial_observation

    for _ in range(rollout_steps):
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            action, log_prob, _, value = (policy.get_action_and_value(observation_tensor))

        action_id = int(action.item())
        cyborg_action = build_action(action_id)

        next_observation, reward, terminated, truncated, _ = (env.step(cyborg_action))

        next_observation_tensor = torch.as_tensor(next_observation, dtype=torch.float32, device=device,)

        with torch.no_grad():
            next_value = policy.get_value(next_observation_tensor)

        buffer.add(
            observation=observation_tensor,
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            next_value=next_value,
            terminated=terminated,
            truncated=truncated,
        )

        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation

    return buffer.build(gamma = gamma, gae_lambda = gae_lambda), observation

if __name__ == "__main__":
    from acd_benchmark.cyber.action_space import ACTION_SPACE
    from acd_benchmark.cyber.blue_table_env import BlueTableEnv
    from acd_benchmark.rl.policies.flat import FlatActorCritic
    
    env = BlueTableEnv()

    policy = FlatActorCritic(
        num_actions=len(ACTION_SPACE),
    )

    initial_observation, reset_result = env.reset(seed=123)

    batch, next_observation = collect_rollout(
        env=env,
        policy=policy,
        initial_observation=initial_observation,
        rollout_steps=128,
    )

    print(batch.observations.shape)   # (128, 13, 4)
    print(batch.actions.shape)        # (128,)
    print(batch.old_log_probs.shape)  # (128,)
    print(batch.advantages.shape)     # (128,)
    print(batch.returns.shape)        # (128,) 