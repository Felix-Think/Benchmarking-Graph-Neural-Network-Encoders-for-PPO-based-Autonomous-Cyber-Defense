import torch 
@torch.no_grad()

def comput_gae(
    rewards: torch.Tensor,
    values: torch.Tensor, 
    next_values: torch.Tensor, 
    terminated: torch.Tensor, 
    truncated: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(
        (),
        dtype=rewards.dtype,
        device=rewards.device,
    )
    for t in reversed(range(len(rewards))):
        # True terminal: không bootstrap từ next state.
        bootstrap_mask = (~terminated[t]).to(rewards.dtype)

        # Cả terminal và time-limit đều kết thúc trajectory hiện tại.
        continuation_mask = (
            ~(terminated[t] | truncated[t])
        ).to(rewards.dtype)

        delta = (
            rewards[t]
            + gamma * next_values[t] * bootstrap_mask
            - values[t]
        )

        last_advantage = (
            delta
            + gamma
            * gae_lambda
            * continuation_mask
            * last_advantage
        )

        advantages[t] = last_advantage

    returns = advantages + values

    return advantages, returns           