import torch 
from torch.nn.utils import clip_grad_norm_

from acd_benchmark.rl.rollout.rollout import RolloutBatch

def ppo_update(policy, 
               batch: RolloutBatch, 
               optimizer : torch.optim.Optimizer, 
               update_epochs : int = 4,
               minibatch_size : int = 64,
               clip_coef: float = 0.2,
               value_coef: float = 0.5,
               entropy_coef: float = 0.01,
               max_grad_norm: float = 0.5,
               target_kl: float | None = None) -> dict:

    policy.train()

    device = next(policy.parameters()).device 

    observations = batch.observations.to(device)
    actions = batch.actions.to(device)
    old_log_probs = batch.old_log_probs.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    action_masks = batch.action_masks.to(device)
    #normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    batch_size = actions.shape[0]
    metric_sums = {
        "policy_loss": 0.0,
        "value_loss" : 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
        "total_loss": 0.0,
    }
    samples_processed = 0
    optimizer_steps = 0
    early_stopped = False 

    for _ in range(update_epochs):
        indices = torch.randperm(batch_size, device = device)

        epoch_kl_sum = 0.0
        epoch_samples = 0

        for start in range(0, batch_size, minibatch_size):
            minibatch_indices = indices[start: start+ minibatch_size]

            _, new_log_probs, entropy, new_values = policy.get_action_and_value(
                observations[minibatch_indices],
                action=actions[minibatch_indices],
                action_mask=action_masks[minibatch_indices],
            )

            log_ratio = (new_log_probs - old_log_probs[minibatch_indices])
            ratio = log_ratio.exp()

            minibatch_advantages = advantages[minibatch_indices]
            policy_loss_unclipped = ratio * minibatch_advantages
            policy_loss_clipped = minibatch_advantages * torch.clamp(ratio, min = 1.0 - clip_coef, max = 1.0 + clip_coef)

            policy_loss = -torch.min(policy_loss_clipped, policy_loss_unclipped).mean()
            value_loss = 0.5 * (new_values - returns[minibatch_indices]).pow(2).mean()

            entropy_mean = entropy.mean()

            total_loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy_mean)

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            grad_norm = clip_grad_norm_(
                policy.parameters(), max_grad_norm,
            )
            optimizer.step()

            minibatch_count = len(minibatch_indices)

            values_to_log = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_mean,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "grad_norm": grad_norm,
                "total_loss": total_loss,
            }
            for name, value in values_to_log.items():
                metric_sums[name] += (
                    float(value.detach())
                    * minibatch_count
                )

            samples_processed += minibatch_count
            optimizer_steps += 1
            epoch_kl_sum += (
                float(approx_kl) * minibatch_count
            )
            epoch_samples += minibatch_count

        mean_epoch_kl = epoch_kl_sum / epoch_samples

        if(target_kl is not None and mean_epoch_kl > target_kl):
            early_stopped = True
            break
    metrics = {
        name: total / samples_processed
        for name, total in metric_sums.items()
    }

    metrics["optimizer_steps"] = optimizer_steps
    metrics["early_stopped"] = early_stopped

    return metrics