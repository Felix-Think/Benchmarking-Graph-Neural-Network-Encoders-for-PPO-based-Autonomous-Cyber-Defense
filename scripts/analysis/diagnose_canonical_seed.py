import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from acd_benchmark.cyber.canonical_observation import (
    FEATURE_INDEX,
    NODE_FEATURES,
    NUM_NODES,
)
from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.ppo.evaluation import load_policy_checkpoint
from scripts.evaluation.evaluate_canonical_observation_gate import (
    DYNAMIC_NODE_FEATURES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose checkpoint-selection noise and observation use.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trajectory-episodes", type=int, default=10)
    parser.add_argument("--trajectory-seed", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_checkpoint_selection(
    periodic: list[dict[str, Any]],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    updates = np.asarray([record["update"] for record in periodic])
    returns = np.asarray(
        [record["episode_returns"] for record in periodic],
        dtype=np.float64,
    )
    if returns.ndim != 2:
        raise ValueError("Periodic episode returns must form a 2D matrix")

    checkpoint_count, episode_count = returns.shape
    observed_means = returns.mean(axis=1)
    observed_best_index = int(observed_means.argmax())
    observed_order = np.argsort(observed_means)[::-1]
    runner_up_index = int(observed_order[1])

    selection_counts = np.zeros(checkpoint_count, dtype=np.int64)
    selected_mean_samples = np.empty(bootstrap_samples, dtype=np.float64)
    fixed_winner_mean_samples = np.empty(bootstrap_samples, dtype=np.float64)
    chunk_size = 5_000
    cursor = 0
    while cursor < bootstrap_samples:
        current_size = min(chunk_size, bootstrap_samples - cursor)
        indices = rng.integers(
            0,
            episode_count,
            size=(current_size, episode_count),
        )
        sampled_means = returns[:, indices].mean(axis=2).T
        selected_indices = sampled_means.argmax(axis=1)
        selection_counts += np.bincount(
            selected_indices,
            minlength=checkpoint_count,
        )
        selected_mean_samples[cursor : cursor + current_size] = sampled_means[
            np.arange(current_size),
            selected_indices,
        ]
        fixed_winner_mean_samples[cursor : cursor + current_size] = (
            sampled_means[:, observed_best_index]
        )
        cursor += current_size

    paired_difference = (
        returns[observed_best_index] - returns[runner_up_index]
    )
    paired_indices = rng.integers(
        0,
        episode_count,
        size=(bootstrap_samples, episode_count),
    )
    paired_bootstrap = paired_difference[paired_indices].mean(axis=1)

    leave_one_out_choices = []
    leave_one_out_returns = []
    for held_out_index in range(episode_count):
        training_indices = np.arange(episode_count) != held_out_index
        chosen_index = int(returns[:, training_indices].mean(axis=1).argmax())
        leave_one_out_choices.append(int(updates[chosen_index]))
        leave_one_out_returns.append(float(returns[chosen_index, held_out_index]))

    selection_probabilities = [
        {
            "update": int(update),
            "observed_mean": round(float(mean), 2),
            "bootstrap_selection_probability": round(
                float(selection_counts[index] / bootstrap_samples),
                6,
            ),
        }
        for index, (update, mean) in enumerate(zip(updates, observed_means, strict=True))
    ]
    selection_probabilities.sort(
        key=lambda record: record["bootstrap_selection_probability"],
        reverse=True,
    )

    return {
        "checkpoint_count": checkpoint_count,
        "selection_episode_count": episode_count,
        "observed_best_update": int(updates[observed_best_index]),
        "observed_best_mean": round(float(observed_means[observed_best_index]), 2),
        "runner_up_update": int(updates[runner_up_index]),
        "runner_up_mean": round(float(observed_means[runner_up_index]), 2),
        "observed_margin": round(
            float(observed_means[observed_best_index] - observed_means[runner_up_index]),
            2,
        ),
        "best_minus_runner_up_paired_bootstrap_ci95": [
            round(float(value), 2)
            for value in np.quantile(paired_bootstrap, [0.025, 0.975])
        ],
        "observed_winner_bootstrap_selection_probability": round(
            float(selection_counts[observed_best_index] / bootstrap_samples),
            6,
        ),
        "bootstrap_unique_selected_checkpoints": int((selection_counts > 0).sum()),
        "top_bootstrap_selection_probabilities": selection_probabilities[:10],
        "bootstrap_winner_optimism_estimate": round(
            float((selected_mean_samples - fixed_winner_mean_samples).mean()),
            2,
        ),
        "leave_one_out_selected_updates": dict(
            Counter(leave_one_out_choices).most_common()
        ),
        "leave_one_out_selected_return_mean": round(
            float(np.mean(leave_one_out_returns)),
            2,
        ),
        "leave_one_out_selected_return_std": round(
            float(np.std(leave_one_out_returns)),
            2,
        ),
    }


def _held_out_sampling_noise(
    held_out: dict[str, Any],
    observed_periodic_mean: float,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    returns = np.asarray(held_out["episode_returns"], dtype=np.float64)
    episode_count = returns.size
    periodic_episode_count = 10
    indices = rng.integers(
        0,
        episode_count,
        size=(bootstrap_samples, periodic_episode_count),
    )
    ten_episode_means = returns[indices].mean(axis=1)
    return {
        "held_out_episode_count": episode_count,
        "held_out_mean": round(float(returns.mean()), 2),
        "held_out_std": round(float(returns.std()), 2),
        "estimated_standard_error_by_episode_count": {
            str(count): round(float(returns.std() / np.sqrt(count)), 2)
            for count in (10, 30, 50, 100)
        },
        "bootstrap_10_episode_mean_ci95": [
            round(float(value), 2)
            for value in np.quantile(ten_episode_means, [0.025, 0.975])
        ],
        "probability_10_episode_mean_at_least_as_high_as_periodic_winner": round(
            float((ten_episode_means >= observed_periodic_mean).mean()),
            6,
        ),
    }


def _training_dynamics(seed_dir: Path) -> dict[str, Any]:
    checkpoint = torch.load(seed_dir / "checkpoint.pt", map_location="cpu")
    history = checkpoint["history"]
    returns = np.asarray(checkpoint["episode_returns"], dtype=np.float64)
    window = min(20, len(history))
    first = history[:window]
    last = history[-window:]

    def metric_mean(records: list[dict[str, Any]], name: str) -> float:
        return round(float(np.mean([record[name] for record in records])), 6)

    return {
        "window_episodes": window,
        "training_return_first_window": round(float(returns[:window].mean()), 2),
        "training_return_last_window": round(float(returns[-window:].mean()), 2),
        "entropy_first_window": metric_mean(first, "entropy"),
        "entropy_last_window": metric_mean(last, "entropy"),
        "decision_rate_first_window": metric_mean(first, "decision_rate"),
        "decision_rate_last_window": metric_mean(last, "decision_rate"),
        "approx_kl_all_updates_mean": metric_mean(history, "approx_kl"),
        "early_stop_rate": round(
            float(np.mean([record["early_stopped"] for record in history])),
            6,
        ),
        "grad_norm_all_updates_mean": metric_mean(history, "grad_norm"),
    }


def _static_only(observation: np.ndarray) -> np.ndarray:
    transformed = observation.copy()
    nodes = transformed[: NUM_NODES * len(NODE_FEATURES)].reshape(
        NUM_NODES,
        len(NODE_FEATURES),
    )
    nodes[:, [FEATURE_INDEX[name] for name in DYNAMIC_NODE_FEATURES]] = 0.0
    return transformed


def _masked_probabilities(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(
        ~action_mask,
        torch.finfo(logits.dtype).min,
    )
    return torch.softmax(masked_logits, dim=-1)


def _counterfactual_metrics(
    policy: FlatActorCritic,
    observation: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, float], int, int, int]:
    full = torch.as_tensor(observation, dtype=torch.float32, device=device)
    static = torch.as_tensor(
        _static_only(observation),
        dtype=torch.float32,
        device=device,
    )
    zero = torch.zeros_like(full)
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)

    with torch.no_grad():
        full_logits, full_value = policy(full)
        static_logits, static_value = policy(static)
        zero_logits, zero_value = policy(zero)
        full_latent = policy.encode(full)
        static_latent = policy.encode(static)

    full_prob = _masked_probabilities(full_logits, mask)
    static_prob = _masked_probabilities(static_logits, mask)
    zero_prob = _masked_probabilities(zero_logits, mask)
    epsilon = torch.finfo(full_prob.dtype).eps
    full_action = int(full_prob.argmax(dim=-1).item())
    static_action = int(static_prob.argmax(dim=-1).item())
    zero_action = int(zero_prob.argmax(dim=-1).item())

    full_static_tv = 0.5 * torch.abs(full_prob - static_prob).sum()
    full_zero_tv = 0.5 * torch.abs(full_prob - zero_prob).sum()
    full_static_kl = (
        full_prob * (torch.log(full_prob + epsilon) - torch.log(static_prob + epsilon))
    ).sum()
    static_norm = torch.linalg.vector_norm(static_latent)

    node_features = observation[: NUM_NODES * len(NODE_FEATURES)].reshape(
        NUM_NODES,
        len(NODE_FEATURES),
    )
    dynamic_values = node_features[
        :,
        [FEATURE_INDEX[name] for name in DYNAMIC_NODE_FEATURES],
    ]
    metrics = {
        "full_static_tv": float(full_static_tv.item()),
        "full_zero_tv": float(full_zero_tv.item()),
        "full_static_kl": float(full_static_kl.item()),
        "latent_relative_l2": float(
            (
                torch.linalg.vector_norm(full_latent - static_latent)
                / (static_norm + 1e-8)
            ).item()
        ),
        "value_absolute_change": float(
            torch.abs(full_value - static_value).item()
        ),
        "dynamic_nonzero_fraction": float(np.count_nonzero(dynamic_values) / dynamic_values.size),
        "dynamic_l1": float(np.abs(dynamic_values).sum()),
    }
    return metrics, full_action, static_action, zero_action


def _summarize_metric(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 8),
        "median": round(float(np.median(array)), 8),
        "p90": round(float(np.quantile(array, 0.9)), 8),
        "max": round(float(array.max()), 8),
    }


def _trajectory_sensitivity(
    seed_dir: Path,
    seed: int,
    episodes: int,
    episode_seed: int,
    device_name: str,
) -> dict[str, Any]:
    config = _load(seed_dir / "config.json")
    env = CC4CanonicalFlatEnv(
        episode_steps=int(config["episode_steps"]),
        seed=episode_seed,
        canonical_variant=config["canonical_variant"],
        evidence_age_horizon=int(config["evidence_age_horizon"]),
    )
    policy = FlatActorCritic(
        observation_dim=int(config["observation_dim"]),
        num_actions=int(config["num_actions"]),
        encoder_hidden_dim=int(config["encoder_hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        head_hidden_dim=int(config["head_hidden_dim"]),
    )
    device = torch.device(device_name)
    load_policy_checkpoint(policy, seed_dir / "best_checkpoint.pt", device=device)
    policy.eval()

    metrics: dict[str, list[float]] = defaultdict(list)
    agent_metrics: dict[str, dict[str, list[float]]] = {
        agent: defaultdict(list) for agent in env.possible_agents
    }
    action_changes = Counter()
    action_types = Counter()
    total_decisions = 0

    for episode_index in range(episodes):
        observations, infos = env.reset(seed=episode_seed + episode_index)
        labels = {
            agent: env.action_labels(agent) for agent in env.possible_agents
        }
        while True:
            actions: dict[str, int] = {}
            for agent in env.possible_agents:
                if not infos[agent]["decision_ready"]:
                    continue
                values, full_action, static_action, zero_action = (
                    _counterfactual_metrics(
                        policy,
                        observations[agent],
                        infos[agent]["action_mask"],
                        device,
                    )
                )
                for name, value in values.items():
                    metrics[name].append(value)
                    agent_metrics[agent][name].append(value)
                action_changes["full_vs_static"] += full_action != static_action
                action_changes["full_vs_zero"] += full_action != zero_action
                action_changes["static_vs_zero"] += static_action != zero_action
                action_types[labels[agent][full_action].split()[0]] += 1
                actions[agent] = full_action
                total_decisions += 1

            observations, _, terminated, truncated, infos = env.step(actions)
            if all(
                terminated.get(agent, False) or truncated.get(agent, False)
                for agent in env.possible_agents
            ):
                break

    return {
        "training_seed": seed,
        "trajectory_episode_seeds": (
            f"{episode_seed}-{episode_seed + episodes - 1}"
        ),
        "trajectory_policy": "deterministic best checkpoint using full observation",
        "episodes": episodes,
        "total_policy_decisions": total_decisions,
        "aggregate_metrics": {
            name: _summarize_metric(values) for name, values in metrics.items()
        },
        "greedy_action_change_rates": {
            name: round(count / total_decisions, 8)
            for name, count in action_changes.items()
        },
        "full_policy_action_type_counts": dict(action_types.most_common()),
        "per_agent": {
            agent: {
                name: _summarize_metric(values)
                for name, values in measurements.items()
            }
            for agent, measurements in agent_metrics.items()
        },
    }


def _action_type_counts(evaluation: dict[str, Any]) -> Counter:
    counts = Counter()
    for key, count in evaluation["action_counts"].items():
        action_label = key.split(": ", maxsplit=1)[1]
        counts[action_label.split()[0]] += int(count)
    return counts


def _paired_return_summary(
    left: dict[str, Any],
    right: dict[str, Any],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    differences = np.asarray(left["episode_returns"], dtype=np.float64) - np.asarray(
        right["episode_returns"],
        dtype=np.float64,
    )
    indices = rng.integers(
        0,
        differences.size,
        size=(bootstrap_samples, differences.size),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    return {
        "mean": round(float(differences.mean()), 2),
        "bootstrap_ci95": [
            round(float(value), 2)
            for value in np.quantile(bootstrap_means, [0.025, 0.975])
        ],
        "identical_episode_returns": int((differences == 0).sum()),
        "full_better": int((differences > 0).sum()),
        "static_only_better": int((differences < 0).sum()),
    }


def _behavior_gate_summary(
    seed_dir: Path,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    full = _load(seed_dir / "best_evaluation_deterministic.json")
    static = _load(seed_dir / "observation_gate_static_only_deterministic.json")
    full_counts = _action_type_counts(full)
    static_counts = _action_type_counts(static)
    action_types = sorted(full_counts.keys() | static_counts.keys())
    full_total = sum(full_counts.values())
    static_total = sum(static_counts.values())
    action_type_table = {
        action_type: {
            "full_fraction": round(full_counts[action_type] / full_total, 6),
            "static_only_fraction": round(
                static_counts[action_type] / static_total,
                6,
            ),
        }
        for action_type in action_types
    }
    result = {
        "deterministic_full_minus_static_only": _paired_return_summary(
            full,
            static,
            rng,
            bootstrap_samples,
        ),
        "action_type_fractions": action_type_table,
    }
    stochastic_static_path = (
        seed_dir / "observation_gate_static_only_stochastic.json"
    )
    if stochastic_static_path.exists():
        stochastic_full = _load(seed_dir / "best_evaluation_stochastic.json")
        stochastic_static = _load(stochastic_static_path)
        result["stochastic_full_minus_static_only"] = _paired_return_summary(
            stochastic_full,
            stochastic_static,
            rng,
            bootstrap_samples,
        )
    return result


def main() -> None:
    args = parse_args()
    seed_dir = args.run_dir / f"seed_{args.seed}"
    periodic = _load(seed_dir / "periodic_evaluation.json")
    if not isinstance(periodic, list):
        raise TypeError("Expected periodic_evaluation.json to contain a list")

    rng = np.random.default_rng(20_260_821 + args.seed)
    selection = _bootstrap_checkpoint_selection(
        periodic,
        rng,
        args.bootstrap_samples,
    )
    best_held_out = _load(seed_dir / "best_evaluation_deterministic.json")
    selection["selected_checkpoint_held_out_mean"] = round(
        float(best_held_out["mean_return"]),
        2,
    )
    selection["periodic_minus_held_out_gap"] = round(
        selection["observed_best_mean"] - float(best_held_out["mean_return"]),
        2,
    )
    selection["held_out_resampling_noise"] = _held_out_sampling_noise(
        best_held_out,
        selection["observed_best_mean"],
        rng,
        args.bootstrap_samples,
    )

    payload = {
        "checkpoint_selection": selection,
        "training_dynamics": _training_dynamics(seed_dir),
        "behavior_level_observation_gate": _behavior_gate_summary(
            seed_dir,
            rng,
            args.bootstrap_samples,
        ),
        "on_trajectory_counterfactual_sensitivity": _trajectory_sensitivity(
            seed_dir,
            args.seed,
            args.trajectory_episodes,
            args.trajectory_seed,
            args.device,
        ),
        "limitations": [
            "Only best and final model states were retained; unselected checkpoints cannot be re-evaluated on 100 held-out episodes.",
            "Bootstrap selection probabilities condition on the ten periodic episode seeds and do not recover the unknown population ranking.",
            "Static-only and zero-all observations are counterfactual inputs and may be outside the training distribution.",
        ],
    }
    output_path = seed_dir / "seed_diagnostics.json"
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(f"output_file={output_path}")


if __name__ == "__main__":
    main()
