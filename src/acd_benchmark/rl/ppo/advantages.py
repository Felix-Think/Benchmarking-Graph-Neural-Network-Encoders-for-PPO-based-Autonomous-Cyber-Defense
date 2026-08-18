import torch 
@torch.no_grad()

def compute_smdp_gae(
    rewards,
    values,
    next_values,
    durations,
    terminated,
    truncated,
    gamma=0.99,
    gae_lambda=0.95,
):
    expected_shape = rewards.shape

    for name, tensor in {
        "values": values,
        "next_values": next_values,
        "durations": durations,
        "terminated": terminated,
        "truncated": truncated,
    }.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} shape {tensor.shape} != rewards shape {expected_shape}"
            )

    if (durations < 1).any():
        raise ValueError("Every SMDP transition must span at least one tick.")

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device,)

    for t in reversed(range(len(rewards))):
        duration = durations[t].to(rewards.dtype)

        bootstrap_discount = gamma ** duration
        trace_discount = (gamma * gae_lambda) ** duration

        bootstrap_mask = (~terminated[t]).to(rewards.dtype)
        continuation_mask = (
            ~(terminated[t] | truncated[t])
        ).to(rewards.dtype)

        delta = (rewards[t] + bootstrap_discount * bootstrap_mask * next_values[t] - values[t])

        last_advantage = (delta + trace_discount * continuation_mask * last_advantage)
        advantages[t] = last_advantage

    return advantages, advantages + values
