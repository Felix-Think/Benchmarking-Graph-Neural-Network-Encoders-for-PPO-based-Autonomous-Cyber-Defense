import torch
import torch.nn as nn
from torch.distributions import Categorical
from acd_benchmark.rl.ppo.init_weight import init_weight

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, head_hidden_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, output_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        
class ActorCritic(nn.Module):
    def __init__(self, latent_dim : int, num_actions : int, head_hidden_dim: int = 128,):
        super().__init__()

        self.actor = MLP(input_dim = latent_dim, output_dim = num_actions, head_hidden_dim = head_hidden_dim)
        self.critic = MLP(input_dim = latent_dim, output_dim = 1, head_hidden_dim = head_hidden_dim)

        self.actor.apply(init_weight)
        self.critic.apply(init_weight)

    def encode(self, observation : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, observation: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(observation)
        logits = self.actor(latent)
        value = self.critic(latent)
        return logits, value
    
    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(observation)

        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device,)

            if (
                action_mask.ndim == 1
                and logits.ndim == 2
                and logits.shape[0] == 1
            ):
                action_mask = action_mask.unsqueeze(0)

            if action_mask.shape != logits.shape:
                raise ValueError(
                    "action_mask must have the same shape as actor logits: "
                    f"mask={tuple(action_mask.shape)}, "
                    f"logits={tuple(logits.shape)}"
                )

            if not action_mask.any(dim=-1).all().item():
                raise ValueError(
                    "Every observation must have at least one valid action."
                )

            invalid_logit = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~action_mask, invalid_logit)

        distribution = Categorical(logits=logits)

        if action is None:
            action = (
                logits.argmax(dim=-1)
                if deterministic
                else distribution.sample()
            )
        else:
            action = torch.as_tensor(
                action,
                dtype=torch.long,
                device=logits.device,
            )

            expected_shape = logits.shape[:-1]
            if action.shape != expected_shape:
                raise ValueError(
                    "action shape must match the batch dimensions of logits: "
                    f"action={tuple(action.shape)}, "
                    f"expected={tuple(expected_shape)}"
                )

            if action_mask is not None:
                selected_action_is_valid = action_mask.gather(
                    dim=-1,
                    index=action.unsqueeze(-1),
                ).squeeze(-1)
                if not selected_action_is_valid.all().item():
                    raise ValueError("A supplied action is masked as invalid.")

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


    def get_value(self, observation: torch.Tensor):
        _, value = self.forward(observation)
        return value.squeeze(-1)
    
