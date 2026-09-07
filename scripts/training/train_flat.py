import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

import torch

from acd_benchmark.cyber.cc4_flat_env import CC4FlatEnv
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.ppo.evaluation import (
    evaluate_policy_modes,
    evaluate_policy_with_preserved_rng,
)
from acd_benchmark.rl.ppo.training import (
    PPOTrainingConfig,
    save_best_evaluation_checkpoint,
    seed_runtime,
    train_policy,
)
from acd_benchmark.rl.tensorboard_logger import TensorBoardLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the shared Flat PPO policy on CAGE 4.",
    )
    parser.add_argument(
        "--total-simulator-steps",
        "--total-timesteps",
        dest="total_simulator_steps",
        type=int,
        default=100_000,
        help="Training budget measured in CAGE 4 simulator ticks.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=1,
        help=(
            "Number of complete episodes collected sequentially before each "
            "PPO update."
        ),
    )
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--reward-scale",type=float,default=0.01,help="Fixed multiplier for PPO rewards before GAE.",)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--episode-steps",
        "--episode-horizon",
        dest="episode_steps",
        type=int,
        default=500,
        help="CAGE 4 EnterpriseScenarioGenerator steps parameter.",
    )

    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=128)

    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=10,
        help="Run fixed-seed evaluation every N PPO updates; 0 disables it.",
    )
    parser.add_argument("--periodic-eval-episodes", type=int, default=10)
    parser.add_argument("--periodic-eval-seed", type=int, default=20_000)
    parser.add_argument(
        "--periodic-eval-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/flat"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--best-checkpoint-path", type=Path, default=None)
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--log-parameter-histograms", action="store_true")
    parser.add_argument("--print-every-updates", type=int, default=1)
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


def run_training(
    args: argparse.Namespace,
    env_factory: Callable[..., object] = CC4FlatEnv,
    policy_factory: Callable[[int, int, argparse.Namespace], torch.nn.Module]
    | None = None,
) -> None:
    if args.eval_every_updates < 0:
        raise ValueError("--eval-every-updates must be non-negative.")
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive.")
    if args.episodes_per_update <= 0:
        raise ValueError("--episodes-per-update must be positive.")

    run_dir = args.run_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_path or run_dir / "checkpoint.pt"
    best_checkpoint_path = (
        args.best_checkpoint_path or run_dir / "best_checkpoint.pt"
    )
    tensorboard_log_dir = (
        args.tensorboard_log_dir or run_dir / "tensorboard"
    )

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
    )

    seed_runtime(config.seed)
    train_env = env_factory(
        episode_steps=args.episode_steps,
        seed=args.seed,
    )
    reference_agent = train_env.possible_agents[0]
    observation_dim = int(train_env.observation_space(reference_agent).shape[0])
    num_actions = int(train_env.action_space(reference_agent).n)

    if policy_factory is None:
        policy = FlatActorCritic(
            observation_dim=observation_dim,
            num_actions=num_actions,
            encoder_hidden_dim=args.encoder_hidden_dim,
            latent_dim=args.latent_dim,
            head_hidden_dim=args.head_hidden_dim,
        )
    else:
        policy = policy_factory(observation_dim, num_actions, args)
    policy.to(torch.device(config.device))
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
    )
    parameter_count = int(
        sum(parameter.numel() for parameter in policy.parameters())
    )

    run_config = {
        **_json_ready_args(args),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_selection_metric": (
            f"periodic_{args.periodic_eval_mode}_mean_return"
        ),
        "tensorboard_log_dir": str(tensorboard_log_dir),
        "observation_dim": observation_dim,
        "num_actions": num_actions,
        "parameter_count": parameter_count,
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

        if tensorboard_logger is not None:
            tensorboard_logger(metrics)
        _print_update(metrics, args.print_every_updates)

        update = int(metrics["update"])
        if (
            periodic_eval_env is None
            or update % args.eval_every_updates != 0
        ):
            return

        periodic_result = evaluate_policy_with_preserved_rng(
            periodic_eval_env,
            policy,
            episodes=args.periodic_eval_episodes,
            seed=args.periodic_eval_seed,
            deterministic=args.periodic_eval_mode == "deterministic",
            action_seed=args.periodic_eval_seed,
        )
        is_best = (
            best_mean_return is None
            or periodic_result.mean_return > best_mean_return
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
            json.dumps(periodic_evaluations, indent=2),
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
                metric_name=(
                    f"periodic_{args.periodic_eval_mode}_mean_return"
                ),
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

    try:
        training_result = train_policy(
            train_env,
            policy,
            config,
            optimizer=optimizer,
            on_update=on_update,
        )

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
            policy,
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
            json.dumps(deterministic_evaluation.to_dict(), indent=2),
            encoding="utf-8",
        )
        stochastic_evaluation_path = run_dir / "evaluation_stochastic.json"
        stochastic_evaluation_path.write_text(
            json.dumps(stochastic_evaluation.to_dict(), indent=2),
            encoding="utf-8",
        )
        evaluation_comparison_path = run_dir / "evaluation_comparison.json"
        evaluation_comparison_path.write_text(
            json.dumps(evaluation_comparison.to_dict(), indent=2),
            encoding="utf-8",
        )

        training_summary = {
            "total_simulator_steps": training_result.total_simulator_steps,
            "total_policy_decisions": training_result.total_policy_decisions,
            "total_episodes": training_result.total_episodes,
            "total_updates": training_result.total_updates,
            "parameter_count": parameter_count,
            "best_checkpoint_path": (
                str(best_checkpoint_path)
                if best_periodic_record is not None
                else None
            ),
            "best_periodic_evaluation": best_periodic_record,
        }
        (run_dir / "training_summary.json").write_text(
            json.dumps(training_summary, indent=2),
            encoding="utf-8",
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
        if tensorboard_logger is not None:
            tensorboard_logger.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
