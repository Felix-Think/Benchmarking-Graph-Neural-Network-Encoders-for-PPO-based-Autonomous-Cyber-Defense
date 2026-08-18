from dataclasses import asdict, is_dataclass
import json
from numbers import Real
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.tensorboard import SummaryWriter


SCALAR_TAGS = {
    "update": "progress/update",
    "rollout_reward_mean": "training/reward_per_step",
    "scaled_transition_reward_mean": "training/scaled_transition_reward_mean",
    "completed_episodes": "training/completed_episodes",
    "episode_return_mean": "training/episode_return",
    "episode_length_mean": "training/episode_length",
    "policy_decisions": "training/policy_decisions",
    "decision_rate": "training/decisions_per_simulator_step",
    "duration_mean": "training/action_duration_mean",
    "duration_min": "training/action_duration_min",
    "duration_max": "training/action_duration_max",
    "policy_loss": "loss/policy",
    "value_loss": "loss/value",
    "total_loss": "loss/total",
    "entropy": "diagnostics/entropy",
    "approx_kl": "diagnostics/approx_kl",
    "clip_fraction": "diagnostics/clip_fraction",
    "grad_norm": "diagnostics/grad_norm_before_clipping",
    "optimizer_steps": "diagnostics/optimizer_steps",
    "early_stopped": "diagnostics/early_stopped",
}


class TensorBoardLogger:
    """Write PPO update, evaluation and optional parameter metrics."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        config=None,
        flush_secs: int = 10,
    ):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs,
        )
        if config is not None:
            self.log_config(config)

    def log_config(self, config) -> None:
        if is_dataclass(config):
            config = asdict(config)
        elif isinstance(config, Mapping):
            config = dict(config)
        else:
            raise TypeError("config must be a dataclass or mapping")

        formatted = json.dumps(
            config,
            indent=2,
            sort_keys=True,
            default=str,
        )
        self.writer.add_text(
            "config/hyperparameters",
            f"```json\n{formatted}\n```",
            global_step=0,
        )

    def __call__(self, metrics: Mapping[str, object]) -> None:
        """Allow the logger to be passed directly as train_policy.on_update."""
        global_step = int(metrics["global_step"])

        for metric_name, tag in SCALAR_TAGS.items():
            value = metrics.get(metric_name)
            scalar = self._to_scalar(value)
            if scalar is not None:
                self.writer.add_scalar(tag, scalar, global_step)

    def log_evaluation(
        self,
        result,
        *,
        global_step: int,
        namespace: str = "evaluation",
    ) -> None:
        namespace = namespace.strip("/")
        if not namespace:
            raise ValueError("evaluation namespace must not be empty")

        evaluation_scalars = {
            f"{namespace}/return_mean": result.mean_return,
            f"{namespace}/return_std": result.std_return,
            f"{namespace}/return_min": result.min_return,
            f"{namespace}/return_max": result.max_return,
            f"{namespace}/episode_length_mean": result.mean_episode_length,
        }
        for tag, value in evaluation_scalars.items():
            self.writer.add_scalar(tag, value, global_step)

        self.writer.add_histogram(
            f"{namespace}/episode_returns",
            torch.tensor(result.episode_returns, dtype=torch.float32),
            global_step,
        )

        total_actions = sum(result.action_counts.values())
        if total_actions > 0:
            for action_name, count in result.action_counts.items():
                self.writer.add_scalar(
                    f"{namespace}/action_fraction/{action_name}",
                    count / total_actions,
                    global_step,
                )

    def log_parameter_histograms(
        self,
        policy: torch.nn.Module,
        *,
        global_step: int,
        include_gradients: bool = False,
    ) -> None:
        """Log model distributions occasionally, not at every PPO update."""
        for parameter_name, parameter in policy.named_parameters():
            tag_name = parameter_name.replace(".", "/")
            self.writer.add_histogram(
                f"parameters/{tag_name}",
                parameter.detach().cpu(),
                global_step,
            )
            if include_gradients and parameter.grad is not None:
                self.writer.add_histogram(
                    f"gradients/{tag_name}",
                    parameter.grad.detach().cpu(),
                    global_step,
                )

    @staticmethod
    def _to_scalar(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, Real):
            return float(value)
        if torch.is_tensor(value) and value.numel() == 1:
            return float(value.detach().cpu())
        return None

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def write_checkpoint_history_to_tensorboard(
    checkpoint_path: str | Path,
    log_dir: str | Path,
) -> int:
    """Rebuild TensorBoard scalar events from a saved training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint.get("history")
    if not isinstance(history, list):
        raise ValueError("checkpoint does not contain a training history list")

    with TensorBoardLogger(
        log_dir,
        config=checkpoint.get("config"),
    ) as logger:
        for metrics in history:
            logger(metrics)

    return len(history)
