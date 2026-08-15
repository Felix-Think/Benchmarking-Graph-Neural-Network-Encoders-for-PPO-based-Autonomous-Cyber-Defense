import argparse
import json
from pathlib import Path

from acd_benchmark.cyber.action_space import ACTION_SPACE
from acd_benchmark.cyber.blue_table_env import BlueTableEnv
from acd_benchmark.rl.policies.flat import FlatActorCritic
from acd_benchmark.rl.ppo.evaluation import evaluate_policy
from acd_benchmark.rl.ppo.training import (
    PPOTrainingConfig,
    seed_runtime,
    train_policy,
)
from acd_benchmark.rl.tensorboard_logger import TensorBoardLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Flat PPO baseline on CAGE-2.",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episode-horizon", type=int, default=200)

    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=10,
        help="Run diagnostic evaluation every N PPO updates; 0 disables it.",
    )
    parser.add_argument("--periodic-eval-episodes", type=int, default=10)
    parser.add_argument("--periodic-eval-seed", type=int, default=20_000)
    parser.add_argument(
        "--stochastic-evaluation",
        action="store_true",
        help="Sample evaluation actions instead of using deterministic argmax.",
    )

    parser.add_argument("--run-dir", type=Path, default=Path("runs/flat"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--log-parameter-histograms", action="store_true")
    parser.add_argument("--print-every-updates", type=int, default=1)
    return parser


def _print_update(metrics: dict, every_updates: int) -> None:
    update = int(metrics["update"])
    if every_updates <= 0 or update % every_updates != 0:
        return

    episode_return = metrics["episode_return_mean"]
    episode_text = (
        "n/a" if episode_return is None else f"{episode_return:.2f}"
    )
    print(
        f"update={update} "
        f"step={int(metrics['global_step'])} "
        f"episode_return={episode_text} "
        f"policy_loss={metrics['policy_loss']:.4f} "
        f"value_loss={metrics['value_loss']:.4f} "
        f"entropy={metrics['entropy']:.4f} "
        f"approx_kl={metrics['approx_kl']:.6f}"
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    run_dir = args.run_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_path or run_dir / "checkpoint.pt"
    tensorboard_log_dir = (
        args.tensorboard_log_dir or run_dir / "tensorboard"
    )

    config = PPOTrainingConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        seed=args.seed,
        device=args.device,
        checkpoint_path=str(checkpoint_path),
    )

    # Seed before policy construction so initial weights are reproducible.
    seed_runtime(config.seed)
    policy = FlatActorCritic(num_actions=len(ACTION_SPACE))
    train_env = BlueTableEnv(max_step=args.episode_horizon)
    periodic_eval_env = (
        BlueTableEnv(max_step=args.episode_horizon)
        if args.eval_every_updates > 0
        else None
    )
    periodic_evaluations: list[dict] = []
    periodic_evaluation_path = run_dir / "periodic_evaluation.json"
    tensorboard_logger = None

    if not args.no_tensorboard:
        tensorboard_logger = TensorBoardLogger(
            tensorboard_log_dir,
            config={
                **vars(args),
                "checkpoint_path": str(checkpoint_path),
                "tensorboard_log_dir": str(tensorboard_log_dir),
            },
        )

    def on_update(metrics: dict) -> None:
        if tensorboard_logger is not None:
            tensorboard_logger(metrics)
        _print_update(metrics, args.print_every_updates)

        update = int(metrics["update"])
        if (
            periodic_eval_env is None
            or update % args.eval_every_updates != 0
        ):
            return

        periodic_result = evaluate_policy(
            periodic_eval_env,
            policy,
            episodes=args.periodic_eval_episodes,
            seed=args.periodic_eval_seed,
            deterministic=True,
        )
        periodic_record = {
            "update": update,
            "global_step": int(metrics["global_step"]),
            **periodic_result.to_dict(),
        }
        periodic_evaluations.append(periodic_record)
        periodic_evaluation_path.write_text(
            json.dumps(periodic_evaluations, indent=2),
            encoding="utf-8",
        )

        if tensorboard_logger is not None:
            tensorboard_logger.log_evaluation(
                periodic_result,
                global_step=int(metrics["global_step"]),
                namespace="evaluation",
            )
        print(
            f"periodic_evaluation update={update} "
            f"mean_return={periodic_result.mean_return:.2f} "
            f"std_return={periodic_result.std_return:.2f}"
        )

    try:
        training_result = train_policy(
            train_env,
            policy,
            config,
            on_update=on_update,
        )

        evaluation = evaluate_policy(
            BlueTableEnv(max_step=args.episode_horizon),
            policy,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            deterministic=not args.stochastic_evaluation,
        )

        if tensorboard_logger is not None:
            tensorboard_logger.log_evaluation(
                evaluation,
                global_step=training_result.total_timesteps,
                namespace="final_evaluation",
            )
            if args.log_parameter_histograms:
                tensorboard_logger.log_parameter_histograms(
                    policy,
                    global_step=training_result.total_timesteps,
                )

        evaluation_path = run_dir / "evaluation.json"
        evaluation_path.write_text(
            json.dumps(evaluation.to_dict(), indent=2),
            encoding="utf-8",
        )

        print(
            f"training_complete steps={training_result.total_timesteps} "
            f"updates={training_result.total_updates} "
            f"episodes={len(training_result.episode_returns)}"
        )
        print(
            f"evaluation mean_return={evaluation.mean_return:.2f} "
            f"std_return={evaluation.std_return:.2f}"
        )
        print(f"checkpoint={checkpoint_path}")
        print(f"evaluation_file={evaluation_path}")
        if periodic_evaluations:
            print(f"periodic_evaluation_file={periodic_evaluation_path}")
        if tensorboard_logger is not None:
            print(f"tensorboard_log_dir={tensorboard_log_dir}")
    finally:
        if tensorboard_logger is not None:
            tensorboard_logger.close()


if __name__ == "__main__":
    main()
