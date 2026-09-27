"""Evaluate final/best checkpoints, with optional observation-only diagnostics."""

import argparse
import json
from pathlib import Path

import torch

from acd_benchmark.cyber.benchmark_env import make_benchmark_env
from acd_benchmark.cyber.cc4_canonical_env import require_current_canonical_parser
from acd_benchmark.cyber.canonical_observation_condition import (
    VALID_OBSERVATION_CONDITIONS,
)
from acd_benchmark.cyber.dynamic_evidence_counterfactual import (
    DynamicEvidenceCounterfactual,
    COUNTERFACTUAL_TRANSFORMS,
)
from acd_benchmark.cyber.dynamic_host_observation_mask import FixedBudgetDynamicHostMask
from acd_benchmark.rl.policies.factory import build_policy
from acd_benchmark.rl.ppo.evaluation import (
    evaluate_policy_modes,
    load_policy_checkpoint,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Parent of seed_N")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--action-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON file; existing results are never overwritten",
    )
    parser.add_argument(
        "--condition", choices=VALID_OBSERVATION_CONDITIONS, default=None
    )
    intervention = parser.add_mutually_exclusive_group()
    intervention.add_argument("--counterfactual", choices=COUNTERFACTUAL_TRANSFORMS)
    intervention.add_argument("--retention", type=float, default=None)
    parser.add_argument(
        "--mask-process", choices=("iid", "persistent"), default="persistent"
    )
    parser.add_argument("--intervention-seed", type=int, default=30_000)
    parser.add_argument("--time-shift-lag", type=int, default=6)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Evaluation already exists: {args.output}")
    if args.condition is not None and (
        args.counterfactual is not None or args.retention is not None
    ):
        raise ValueError(
            "Use one diagnostic at a time, not condition plus another intervention"
        )
    torch.set_num_threads(args.torch_threads)
    run_dir = args.run_dir / f"seed_{args.seed}"
    training_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    require_current_canonical_parser(training_config)
    policy = build_policy(training_config).to(args.device)
    filename = "checkpoint.pt" if args.checkpoint == "final" else "best_checkpoint.pt"
    path = Path(
        training_config.get(
            "checkpoint_path" if args.checkpoint == "final" else "best_checkpoint_path",
            str(run_dir / filename),
        )
    )
    # A copied/archived run keeps its original config paths. Prefer the local copy.
    if (run_dir / filename).is_file():
        path = run_dir / filename
    load_policy_checkpoint(policy, path, device=args.device)
    config = dict(training_config)
    if args.condition is not None:
        config["observation_condition"] = args.condition
    if (args.counterfactual is not None or args.retention is not None) and config.get(
        "observation_condition", "full"
    ) != "full":
        raise ValueError(
            "Counterfactual/retention diagnostics require a Full-trained policy"
        )

    def env_factory():
        env = make_benchmark_env(config, seed=args.eval_seed)
        if args.counterfactual is not None:
            return DynamicEvidenceCounterfactual(
                env,
                transform=args.counterfactual,
                intervention_seed=args.intervention_seed,
                time_shift_lag=args.time_shift_lag,
            )
        if args.retention is not None:
            return FixedBudgetDynamicHostMask(
                env,
                retention=args.retention,
                mask_process=args.mask_process,
                mask_seed=args.intervention_seed,
            )
        return env

    action_seed = args.eval_seed if args.action_seed is None else args.action_seed
    result = evaluate_policy_modes(
        env_factory(),
        env_factory(),
        policy,
        episodes=args.episodes,
        seed=args.eval_seed,
        stochastic_action_seed=action_seed,
    )
    record = {
        "checkpoint": str(path),
        "training_seed": args.seed,
        "training_config": training_config,
        "evaluation_config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        **result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"deterministic mean_return={result.deterministic.mean_return:.2f}")
    print(f"stochastic mean_return={result.stochastic.mean_return:.2f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
