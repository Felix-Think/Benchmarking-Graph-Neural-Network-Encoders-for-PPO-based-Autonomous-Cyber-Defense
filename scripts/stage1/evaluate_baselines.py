"""Evaluate Sleep, Monitor, and uniform-random Stage 1 baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts._bootstrap import ensure_src_on_path, repository_root

ensure_src_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from acd_benchmark.evaluation.operational_metrics import (  # noqa: E402
    evaluate_action_selector_with_operational_metrics,
)
from acd_benchmark.rl.stage1_factory import make_stage1_env  # noqa: E402
from acd_benchmark.utils.reproducibility import (  # noqa: E402
    collect_reproducibility_metadata,
    sha256_file,
)


BASELINES = ("sleep", "monitor", "random")


def _fixed_action_index(env: Any, action_name: str) -> int:
    """Resolve one globally applicable Blue action by class name."""
    matches = [
        index
        for index, spec in enumerate(env.action_specs)
        if spec.action_cls.__name__.lower() == action_name and spec.hostname is None
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one global {action_name} action, found {matches}.")
    return matches[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a rectangular CSV when rows are available."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run reproducible non-learning baselines under the final evaluation seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_16.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--policy-seed", type=int, default=2027)
    parser.add_argument("--observation-levels", nargs="+", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scenario = config["scenario"]
    training = config["training"]
    levels = args.observation_levels or [
        int(level) for level in config["matrix"]["observation_levels"]
    ]
    episodes = args.episodes or int(
        training.get("final_evaluation_episodes", training["evaluation_episodes"])
    )
    environment_seed = args.seed if args.seed is not None else int(
        training.get("final_evaluation_base_seed", training["evaluation_base_seed"])
    )
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    output_dir = args.output_dir or Path(config["output_root"]) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for level in levels:
        for baseline in BASELINES:
            env = make_stage1_env(
                "flat",
                scenario_path=Path(str(scenario["path"])),
                node_count=int(scenario["node_count"]),
                observation_probability=level / 100.0,
                max_steps=int(scenario["max_episode_steps"]),
            )
            # Action specs are materialized from CybORG's reset-time action space.
            env.reset(seed=environment_seed * 1000)
            if baseline == "random":
                rng = np.random.default_rng(args.policy_seed)

                def select_action(_observation: Any, *, _rng=rng, _env=env) -> int:
                    return int(_rng.integers(len(_env.action_specs)))

            else:
                action_index = _fixed_action_index(env, baseline)

                def select_action(_observation: Any, *, _action=action_index) -> int:
                    return _action

            metrics = evaluate_action_selector_with_operational_metrics(
                env,
                select_action,
                episodes=episodes,
                seed=environment_seed,
            )
            aggregate = {
                "baseline": baseline,
                "observation_level": level,
                **{key: value for key, value in metrics.items() if key != "episodes"},
            }
            aggregate_rows.append(aggregate)
            episode_rows.extend(
                {
                    "baseline": baseline,
                    "observation_level": level,
                    **episode,
                }
                for episode in metrics["episodes"]
            )
            results.append({**aggregate, "episodes": metrics["episodes"]})
            print(
                f"{baseline} obs={level}: "
                f"episode_return={metrics['mean_episode_return']:.4f} "
                f"availability={metrics['mean_mission_availability']:.4f}",
                flush=True,
            )

    metadata = {
        "protocol": config["protocol"],
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "scenario_sha256": sha256_file(Path(str(scenario["path"]))),
        "episodes": episodes,
        "environment_base_seed": environment_seed,
        "random_policy_seed": args.policy_seed,
        "reproducibility": collect_reproducibility_metadata(
            repository_root(),
            device=torch.device("cpu"),
        ),
        "results": results,
    }
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "baseline_aggregate.csv", aggregate_rows)
    _write_csv(output_dir / "baseline_per_episode.csv", episode_rows)


if __name__ == "__main__":
    main()
