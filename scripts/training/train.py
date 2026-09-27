import argparse
import json
import os
from pathlib import Path

import torch
from time import perf_counter
from acd_benchmark.metrics.resources import runtime_metadata

from acd_benchmark.cyber.benchmark_env import make_benchmark_env
from acd_benchmark.cyber.cc4_canonical_env import CANONICAL_PARSER_VERSION
from acd_benchmark.cyber.canonical_observation_condition import (
    VALID_OBSERVATION_CONDITIONS,
)
from acd_benchmark.cyber.cc4_distribution import VALID_ENVIRONMENT_DISTRIBUTIONS
from acd_benchmark.rl.rollout.parallel import ParallelEpisodeCollector, make_training_env
from acd_benchmark.metrics.agregate import aggregate_episodes
from acd_benchmark.rl.policies.factory import build_policy
from acd_benchmark.rl.ppo.evaluation import (
    evaluate_policy_modes,
    evaluate_policy_with_preserved_rng,
)
from scripts.evaluation.retention_curve import evaluate_retention_curve
from acd_benchmark.rl.ppo.training import (
    PPOTrainingConfig,
    save_best_evaluation_checkpoint,
    seed_runtime,
    train_policy,
)
from acd_benchmark.rl.tensorboard_logger import TensorBoardLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train structured NodeMLP, GCN or GraphSAGE with shared PPO on CAGE 4.",
    )
    parser.add_argument(
        "--total-simulator-steps",
        "--total-timesteps",
        dest="total_simulator_steps",
        type=int,
        default=500_000,
        help="Training budget measured in CAGE 4 simulator ticks.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=4,
        help=(
            "Number of complete episodes collected before each "
            "PPO update."
        ),
    )
    parser.add_argument("--rollout-workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--action-seed-protocol", choices=("global", "per_episode_v1"), default="global")
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.01,
        help="Fixed multiplier for PPO rewards before GAE.",
    )
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--episode-steps",
        "--episode-horizon",
        dest="episode_steps",
        type=int,
        default=500,
        help="CAGE 4 EnterpriseScenarioGenerator steps parameter.",
    )

    parser.add_argument("--encoder", choices=("nodemlp", "gcn", "graphsage"), required=True)
    parser.add_argument("--node-hidden-dim", type=int, default=128)
    parser.add_argument("--gcn-hidden-dim", type=int, default=128)
    parser.add_argument("--gcn-layers", type=int, default=2)
    parser.add_argument("--sage-hidden-dim", type=int, default=64)
    parser.add_argument("--sage-layers", type=int, default=2)
    parser.add_argument("--train-regime", choices=("full", "mixed-q"), default="full")
    parser.add_argument("--train-retentions", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--q-seed", type=int, default=40_000)
    parser.add_argument("--train-mask-seed", type=int, default=30_000)
    parser.add_argument("--eval-mask-seed", type=int, default=50_000)
    parser.add_argument("--eval-retentions", type=float, nargs="+", default=[])
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[])
    parser.add_argument("--node-embedding-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--passive-memory-ttl", type=int, default=5)
    parser.add_argument("--analysis-memory-ttl", type=int, default=20)
    parser.add_argument("--semantic-action-mask", action="store_true")
    parser.add_argument(
        "--observation-condition", choices=VALID_OBSERVATION_CONDITIONS, default="full"
    )
    parser.add_argument(
        "--environment-distribution",
        choices=VALID_ENVIRONMENT_DISTRIBUTIONS,
        default="random_host",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.set_defaults(
        canonical_schema="v2",
        canonical_variant="action_aware",
        canonical_parser_version=CANONICAL_PARSER_VERSION,
        action_catalog="host_edge_global_v1",
    )
    parser.add_argument("--head-hidden-dim", type=int, default=128)

    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=25,
        help="Run fixed-seed evaluation every N PPO updates; 0 disables it.",
    )
    parser.add_argument("--periodic-eval-episodes", type=int, default=10)
    parser.add_argument("--periodic-eval-seed", type=int, default=20_000)
    parser.add_argument(
        "--periodic-eval-mode",
        choices=("deterministic", "stochastic"),
        default="stochastic",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--best-checkpoint-path", type=Path, default=None)
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--log-parameter-histograms", action="store_true")
    parser.add_argument("--print-every-updates", type=int, default=25)
    return parser


def _json_ready_args(args: argparse.Namespace) -> dict:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def _print_update(metrics: dict, every_updates: int) -> None:
    update = int(metrics["update"])
    if every_updates <= 0 or update % every_updates != 0:
        return

    print(
        f"update= {update} "
        f"simulator_step= {int(metrics['global_step'])} "
        f"decisions= {int(metrics['policy_decisions'])} "
        f"return= {metrics['episode_return_mean']:.2f} "
        f"policy_loss= {metrics['policy_loss']:.4f} "
        f"value_loss= {metrics['value_loss']:.4f} "
        f"entropy= {metrics['entropy']:.4f} "
        f"approx_kl= {metrics['approx_kl']:.6f}"
    )


def run_training(args: argparse.Namespace) -> None:
    from acd_benchmark.rl.runtime import learner_device, cpu_inference_policy, cpu_state_dict

    args.policy_type = f"structured_{args.encoder}"
    torch.set_num_threads(args.torch_threads)

    def env_factory(episode_steps: int, seed: int | None = None):
        return make_benchmark_env(
            {**vars(args), "episode_steps": episode_steps}, seed=seed
        )

    if args.eval_every_updates < 0:
        raise ValueError("--eval-every-updates must be non-negative.")
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive.")
    if args.episodes_per_update <= 0:
        raise ValueError("--episodes-per-update must be positive.")
    device = learner_device(args.device)
    if (args.rollout_workers > 1 or device.type == "cuda") and args.action_seed_protocol != "per_episode_v1":
        raise ValueError("CPU workers require --action-seed-protocol per_episode_v1")
    if args.episode_steps < 2:
        raise ValueError("--episode-steps must be at least 2")
    if any(not 0 <= q <= 1 for q in args.eval_retentions + args.train_retentions):
        raise ValueError("Retention levels must lie in [0, 1]")

    run_dir = args.run_dir / f"seed_{args.seed}"
    if (run_dir / "config.json").exists():
        raise FileExistsError(f"Run already exists: {run_dir}; choose a new --run-dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_started = perf_counter()
    checkpoint_path = args.checkpoint_path or run_dir / "checkpoint.pt"
    best_checkpoint_path = args.best_checkpoint_path or run_dir / "best_checkpoint.pt"
    tensorboard_log_dir = args.tensorboard_log_dir or run_dir / "tensorboard"

    config = PPOTrainingConfig(
        total_simulator_steps=args.total_simulator_steps,
        episodes_per_update=args.episodes_per_update,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        reward_scale=args.reward_scale,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        seed=args.seed,
        device=args.device,
        checkpoint_path=str(checkpoint_path),
        checkpoint_steps=tuple(args.checkpoint_steps),
        rollout_workers=args.rollout_workers,
        action_seed_protocol=args.action_seed_protocol,
    )

    seed_runtime(config.seed)
    train_env = make_training_env(vars(args))
    reference_agent = train_env.possible_agents[0]
    observation_dim = int(train_env.observation_space(reference_agent).shape[0])
    num_actions = int(train_env.action_space(reference_agent).n)

    policy = build_policy(
        {**vars(args), "observation_dim": observation_dim, "num_actions": num_actions}
    )
    evaluation_policy = cpu_inference_policy(policy) if device.type == "cuda" else policy
    policy.to(device)

    def synced_evaluation_policy():
        if evaluation_policy is not policy:
            evaluation_policy.load_state_dict(cpu_state_dict(policy))
        return evaluation_policy
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
    )
    parameter_count = int(sum(parameter.numel() for parameter in policy.parameters()))

    run_config = {
        **_json_ready_args(args),
        "compute_metadata": runtime_metadata(config.device),
        "execution_pipeline": "cpu_rollout_cuda_learner_v1" if device.type == "cuda" else "cpu",
        "rollout_device": "cpu",
        "evaluation_device": "cpu",
        "learner_precision": "float32",
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_selection_metric": (
            f"periodic_{args.periodic_eval_mode}_mean_return"
        ),
        "tensorboard_log_dir": str(tensorboard_log_dir),
        "observation_dim": observation_dim,
        "num_actions": num_actions,
        "parameter_count": parameter_count,
        "training_observation_process": (
            "persistent_fixed_budget_dynamic_host" if args.train_regime == "mixed-q"
            else "native_canonical_no_additional_mask"
        ),
        "effective_train_q_seed": args.q_seed + args.seed,
        "effective_train_mask_seed": args.train_mask_seed + args.seed,
        "policy_type": getattr(args, "policy_type", "flat"),
        "runtime_seed_controls": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    periodic_eval_env = (
        env_factory(
            episode_steps=args.episode_steps,
            seed=args.periodic_eval_seed,
        )
        if args.eval_every_updates > 0
        else None
    )
    periodic_evaluations: list[dict] = []
    periodic_evaluation_path = run_dir / "periodic_evaluation.json"
    training_history_path = run_dir / "training_history.jsonl"
    best_mean_return: float | None = None
    best_periodic_record: dict | None = None
    tensorboard_logger = None

    if not args.no_tensorboard:
        tensorboard_logger = TensorBoardLogger(
            tensorboard_log_dir,
            config=run_config,
        )

    def on_update(metrics: dict) -> None:
        nonlocal best_mean_return, best_periodic_record

        with training_history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(metrics, allow_nan=False) + "\n")

        if tensorboard_logger is not None:
            tensorboard_logger(metrics)
        _print_update(metrics, args.print_every_updates)

        update = int(metrics["update"])
        if periodic_eval_env is None or update % args.eval_every_updates != 0:
            return

        periodic_result = evaluate_policy_with_preserved_rng(
            periodic_eval_env,
            synced_evaluation_policy(),
            episodes=args.periodic_eval_episodes,
            seed=args.periodic_eval_seed,
            deterministic=args.periodic_eval_mode == "deterministic",
            action_seed=args.periodic_eval_seed,
        )
        is_best = (
            best_mean_return is None or periodic_result.mean_return > best_mean_return
        )
        if is_best:
            best_mean_return = periodic_result.mean_return

        periodic_record = {
            "update": update,
            "global_step": int(metrics["global_step"]),
            "is_best": is_best,
            "best_mean_return_so_far": best_mean_return,
            **periodic_result.to_dict(),
        }
        periodic_evaluations.append(periodic_record)
        periodic_evaluation_path.write_text(
            json.dumps(periodic_evaluations, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        if is_best:
            best_periodic_record = periodic_record
            save_best_evaluation_checkpoint(
                path=best_checkpoint_path,
                policy=policy,
                optimizer=optimizer,
                config=config,
                update=update,
                global_step=int(metrics["global_step"]),
                training_metrics=metrics,
                evaluation=periodic_record,
                metric_name=(f"periodic_{args.periodic_eval_mode}_mean_return"),
                metric_value=periodic_result.mean_return,
            )

        if tensorboard_logger is not None:
            tensorboard_logger.log_evaluation(
                periodic_result,
                global_step=int(metrics["global_step"]),
                namespace="evaluation",
            )
        print(
            f"periodic_evaluation mode={args.periodic_eval_mode} "
            f"update={update} "
            f"mean_return={periodic_result.mean_return:.2f} "
            f"std_return={periodic_result.std_return:.2f} "
            f"is_best={is_best}"
        )

    def on_milestone(result, requested_step):
        milestone_dir = checkpoint_path.parent / "milestones" / f"step_{requested_step}"
        metadata = {
            "requested_step": requested_step,
            "actual_simulator_steps": result.total_simulator_steps,
            "total_updates": result.total_updates,
            "total_episodes": result.total_episodes,
            "metrics_summary": aggregate_episodes(result.episode_returns, result.episode_cyber_metrics),
            "compute_metrics": result.compute_metrics,
        }
        (milestone_dir / "training_summary.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8",
        )
        print(
            f"milestone_saved requested={requested_step} actual={result.total_simulator_steps}",
            flush=True,
        )
        if args.eval_retentions:
            curve = evaluate_retention_curve(
                env_factory, synced_evaluation_policy(), episode_steps=args.episode_steps,
                retentions=args.eval_retentions, episodes=args.periodic_eval_episodes,
                seed=args.periodic_eval_seed, mask_seed=args.eval_mask_seed,
                global_step=result.total_simulator_steps, logger=tensorboard_logger,
            )
            (milestone_dir / "validation_retention.json").write_text(
                json.dumps(curve, indent=2, allow_nan=False), encoding="utf-8",
            )
        if tensorboard_logger is not None:
            tensorboard_logger.flush()

    episode_collector = None
    try:
        training_started = perf_counter()
        if args.rollout_workers > 1 or device.type == "cuda":
            episode_collector = ParallelEpisodeCollector(run_config, args.rollout_workers)
        training_result = train_policy(
            train_env,
            policy,
            config,
            optimizer=optimizer,
            on_update=on_update,
            **({"on_milestone": on_milestone} if args.checkpoint_steps else {}),
            **({"episode_collector": episode_collector} if episode_collector is not None else {}),
        )
        if episode_collector is not None:
            episode_collector.close()
            episode_collector = None
        training_with_validation_seconds = perf_counter() - training_started

        # Preserve training results even if final evaluation fails.
        training_summary = {
            "total_simulator_steps": training_result.total_simulator_steps,
            "total_policy_decisions": training_result.total_policy_decisions,
            "total_episodes": training_result.total_episodes,
            "total_updates": training_result.total_updates,
            "episode_returns": training_result.episode_returns,
            "episode_lengths": training_result.episode_lengths,
            "episode_cyber_metrics": training_result.episode_cyber_metrics,
            "metrics_summary": aggregate_episodes(
                training_result.episode_returns,
                training_result.episode_cyber_metrics,
            ),
            "compute_metrics": training_result.compute_metrics,
            "parameter_count": parameter_count,
            "best_checkpoint_path": (
                str(best_checkpoint_path) if best_periodic_record is not None else None
            ),
            "best_periodic_evaluation": best_periodic_record,
        }
        (run_dir / "training_summary.json").write_text(
            json.dumps(training_summary, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        final_evaluation_started = perf_counter()
        deterministic_eval_env = env_factory(
            episode_steps=args.episode_steps,
            seed=args.eval_seed,
        )
        stochastic_eval_env = env_factory(
            episode_steps=args.episode_steps,
            seed=args.eval_seed,
        )
        evaluation_comparison = evaluate_policy_modes(
            deterministic_eval_env,
            stochastic_eval_env,
            synced_evaluation_policy(),
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            stochastic_action_seed=args.eval_seed,
        )
        deterministic_evaluation = evaluation_comparison.deterministic
        stochastic_evaluation = evaluation_comparison.stochastic

        if tensorboard_logger is not None:
            tensorboard_logger.log_evaluation(
                deterministic_evaluation,
                global_step=training_result.total_simulator_steps,
                namespace="final_evaluation/deterministic",
            )
            tensorboard_logger.log_evaluation(
                stochastic_evaluation,
                global_step=training_result.total_simulator_steps,
                namespace="final_evaluation/stochastic",
            )
            if args.log_parameter_histograms:
                tensorboard_logger.log_parameter_histograms(
                    policy,
                    global_step=training_result.total_simulator_steps,
                )

        deterministic_evaluation_path = run_dir / "evaluation_deterministic.json"
        deterministic_evaluation_path.write_text(
            json.dumps(deterministic_evaluation.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        stochastic_evaluation_path = run_dir / "evaluation_stochastic.json"
        stochastic_evaluation_path.write_text(
            json.dumps(stochastic_evaluation.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        evaluation_comparison_path = run_dir / "evaluation_comparison.json"
        evaluation_comparison_path.write_text(
            json.dumps(evaluation_comparison.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )

        if args.eval_retentions:
            curve = evaluate_retention_curve(
                env_factory, synced_evaluation_policy(), episode_steps=args.episode_steps,
                retentions=args.eval_retentions, episodes=args.eval_episodes,
                seed=args.eval_seed, mask_seed=args.eval_mask_seed,
                modes=("stochastic", "deterministic"),
                global_step=training_result.total_simulator_steps,
                logger=tensorboard_logger, namespace="final_retention",
                reference_results={
                    "stochastic": stochastic_evaluation,
                    "deterministic": deterministic_evaluation,
                },
            )
            (run_dir / "evaluation_retention.json").write_text(
                json.dumps(curve, indent=2, allow_nan=False), encoding="utf-8",
            )

        run_compute = {
            "status": "completed",
            "compute_metadata": run_config["compute_metadata"],
            "training_with_validation_seconds": training_with_validation_seconds,
            "final_evaluation_wall_seconds": perf_counter() - final_evaluation_started,
            "run_wall_seconds": perf_counter() - run_started,
            "training_compute_metrics": training_result.compute_metrics,
        }
        (run_dir / "compute_summary.json").write_text(
            json.dumps(run_compute, indent=2, allow_nan=False), encoding="utf-8",
        )
        if tensorboard_logger is not None:
            for name in ("training_with_validation_seconds", "final_evaluation_wall_seconds", "run_wall_seconds"):
                tensorboard_logger.writer.add_scalar(
                    f"compute_run/{name}", run_compute[name], training_result.total_simulator_steps,
                )

        print(
            f"training_complete simulator_steps="
            f"{training_result.total_simulator_steps} "
            f"decisions={training_result.total_policy_decisions} "
            f"updates={training_result.total_updates} "
            f"episodes={training_result.total_episodes}"
        )
        print(
            "deterministic_evaluation "
            f"mean_return={deterministic_evaluation.mean_return:.2f} "
            f"std_return={deterministic_evaluation.std_return:.2f}"
        )
        print(
            "stochastic_evaluation "
            f"mean_return={stochastic_evaluation.mean_return:.2f} "
            f"std_return={stochastic_evaluation.std_return:.2f}"
        )
        print(
            "paired_return_difference "
            "definition=stochastic-deterministic "
            f"mean={evaluation_comparison.mean_paired_return_difference:.2f}"
        )
        print(f"parameters={parameter_count}")
        print(f"checkpoint={checkpoint_path}")
        print(f"training_history_file={training_history_path}")
        print(f"training_summary_file={run_dir / 'training_summary.json'}")
        if best_periodic_record is not None:
            print(
                f"best_checkpoint={best_checkpoint_path} "
                f"update={best_periodic_record['update']} "
                f"mean_return={best_periodic_record['mean_return']:.2f}"
            )
        print(f"deterministic_evaluation_file={deterministic_evaluation_path}")
        print(f"stochastic_evaluation_file={stochastic_evaluation_path}")
        print(f"evaluation_comparison_file={evaluation_comparison_path}")
        if periodic_evaluations:
            print(f"periodic_evaluation_file={periodic_evaluation_path}")
        if tensorboard_logger is not None:
            print(f"tensorboard_log_dir={tensorboard_log_dir}")
    finally:
        if episode_collector is not None:
            episode_collector.close()
        if tensorboard_logger is not None:
            tensorboard_logger.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
