from dataclasses import dataclass

import torch

from acd_benchmark.rl.ppo.advantages import compute_smdp_gae

@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    durations: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

@dataclass
class PendingTransition:
    observation: torch.Tensor
    action: torch.Tensor
    action_mask: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    discounted_reward: float = 0.0
    duration: int = 0
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
        self.action_mask = []
        self.durations = []

    def __len__(self):
        return len(self.rewards)

    def add(
        self,
        observation,
        action,
        action_mask,
        log_prob,
        reward,
        value,
        next_value,
        duration,
        terminated,
        truncated,
    ):
        self.observations.append(observation.detach().to(self.device))
        self.actions.append(action.detach().to(self.device).reshape(()))
        self.action_mask.append(action_mask.detach().bool().to(self.device))
        self.log_probs.append(log_prob.detach().to(self.device).reshape(()))
        self.rewards.append(float(reward))
        self.values.append(value.detach().to(self.device).reshape(()))
        self.next_values.append(next_value.detach().to(self.device).reshape(()))
        self.durations.append(int(duration))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))


    def build(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        reward_scale=0.01,
    ) -> RolloutBatch:
        observations = torch.stack(self.observations).to(self.device)

        actions = torch.stack(self.actions).long().to(self.device)

        action_mask = torch.stack(self.action_mask).to(self.device)

        old_log_probs = torch.stack(self.log_probs).float().to(self.device)

        raw_rewards = torch.tensor(self.rewards,dtype=torch.float32,device=self.device,)

        values = torch.stack(self.values).float().to(self.device)

        next_values = torch.stack(self.next_values).float().to(self.device)

        durations = torch.tensor(self.durations, dtype=torch.float32, device=self.device,)

        terminated = torch.tensor(self.terminated,dtype=torch.bool,device=self.device,)

        truncated = torch.tensor(self.truncated,dtype=torch.bool,device=self.device,)

        scaled_rewards = raw_rewards * reward_scale

        advantages, returns = compute_smdp_gae(
            rewards=scaled_rewards,
            values=values,
            next_values=next_values,
            durations=durations,
            terminated=terminated,
            truncated=truncated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        return RolloutBatch(
            observations=observations,
            actions=actions,
            action_masks= action_mask,
            old_log_probs=old_log_probs,
            rewards=scaled_rewards,
            values=values,
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
            durations=durations,
            advantages=advantages,
            returns=returns,
        )

def collect_rollout(env, policy, seed, gamma=0.99, gae_lambda = 0.95, reward_scale=0.01,
                    progress_every_ticks=0, committed_ticks=0):
    device = next(policy.parameters()).device
    observations, infos = env.reset(seed=seed)

    buffers = {
        agent: RolloutBuffer(device)
        for agent in env.possible_agents
    }
    pending = {}

    team_return = 0.0
    simulator_steps = 0
    reward_agent = env.possible_agents[0]

    while True:
        actions = {}

        # Chỉ tạo quyết định cho những agent đang rảnh.
        for agent in env.possible_agents:
            if not infos[agent]["decision_ready"]:
                continue

            assert agent not in pending

            observation = torch.as_tensor(
                observations[agent],
                dtype=torch.float32,
                device=device,
            )
            action_mask = torch.as_tensor(  
                infos[agent]["action_mask"],
                dtype=torch.bool,
                device=device,
            )

            with torch.no_grad():
                action, log_prob, _, value = (
                    policy.get_action_and_value(
                        observation,
                        action_mask=action_mask,
                    )
                )

            actions[agent] = int(action.item())
            pending[agent] = PendingTransition(
                observation=observation,
                action_mask=action_mask,
                action=action,
                log_prob=log_prob,
                value=value,
            )

        (next_observations, rewards, terminated, truncated, next_infos,) = env.step(actions)

        simulator_steps += 1

        # CAGE 4 wrapper hiện gán cùng done cho cả hai loại.
        episode_done = all(
            terminated.get(agent, False)
            or truncated.get(agent, False)
            for agent in env.possible_agents
        )

        # Cập nhật reward cho action đang chạy của từng agent.
        for agent, transition in list(pending.items()):
            transition.discounted_reward += (
                gamma ** transition.duration
            ) * float(rewards[agent])
            transition.duration += 1

            action_completed = next_infos[agent]["decision_ready"]

            if action_completed or episode_done:
                next_observation = torch.as_tensor(
                    next_observations[agent],
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    next_value = policy.get_value(next_observation)

                buffers[agent].add(
                    observation=transition.observation,
                    action_mask=transition.action_mask,
                    action=transition.action,
                    log_prob=transition.log_prob,
                    reward=transition.discounted_reward,
                    value=transition.value,
                    next_value=next_value,
                    duration=transition.duration,
                    terminated=terminated.get(agent, False),
                    truncated=truncated.get(agent, False),
                )

                del pending[agent]

        # Reward là team reward và giống nhau cho mọi Blue agent.
        # Không sum rewards.values(), nếu không score bị nhân số agent.
        team_return += float(rewards[reward_agent])

        if progress_every_ticks > 0 and (simulator_steps % progress_every_ticks == 0 or episode_done):
            metrics = env.cyber_summary()
            recoveries = metrics.get("ot_recovery_events", [])
            ttr = (sum(event["ttr_ticks"] for event in recoveries) / len(recoveries)
                   if recoveries else None)
            def fmt(value):
                return "NA" if value is None else f"{value:.4f}"
            print(
                f"rollout episode_seed={seed} episode_tick={simulator_steps} "
                f"committed_ticks={committed_ticks} "
                f"reward={float(rewards[reward_agent]):.2f} return_so_far={team_return:.2f} "
                f"availability={fmt(metrics.get('availability'))} "
                f"compromise_burden={fmt(metrics.get('compromise_burden'))} "
                f"compromised_hosts={metrics.get('compromised_host_count')} "
                f"impact_successes={metrics.get('impact_successes')} "
                f"ttr_completed={fmt(ttr)} "
                f"unrecovered={len(metrics.get('ot_unrecovery_events', []))}",
                flush=True,
            )

        if episode_done:
            break

        observations = next_observations
        infos = next_infos

    agent_batches = [
        buffers[agent].build(
            gamma=gamma,
            gae_lambda=gae_lambda,
            reward_scale=reward_scale,
            )
            for agent in env.possible_agents
        ]

    combined_batch = concatenate_batches(agent_batches)

    return combined_batch, team_return, simulator_steps

def concatenate_batches(batches):
    return RolloutBatch(
        observations=torch.cat([b.observations for b in batches]),
        action_masks=torch.cat([b.action_masks for b in batches]),
        actions=torch.cat([b.actions for b in batches]),
        old_log_probs=torch.cat([b.old_log_probs for b in batches]),
        rewards=torch.cat([b.rewards for b in batches]),
        values=torch.cat([b.values for b in batches]),
        next_values=torch.cat([b.next_values for b in batches]),
        durations=torch.cat([b.durations for b in batches]),
        terminated=torch.cat([b.terminated for b in batches]),
        truncated=torch.cat([b.truncated for b in batches]),
        advantages=torch.cat([b.advantages for b in batches]),
        returns=torch.cat([b.returns for b in batches]),
    )

if __name__ == "__main__":
    pass
