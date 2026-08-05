"""Evaluate non-learning baselines at every fixed Pilot v2 retention level."""

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
from acd_benchmark.rl.pilot2_factory import make_pilot2_env  # noqa: E402
from acd_benchmark.utils.reproducibility import (  # noqa: E402
    collect_reproducibility_metadata,
    sha256_file,
)


BASELINES = ("sleep", "monitor", "random")


def _fixed_action_index(env: Any, action_name: str) -> int:
    """Resolve one global action by its CybORG class name."""
    matches = [
        index
        for index, spec in enumerate(env.action_specs)
        if spec.action_cls.__name__.lower() == action_name and spec.hostname is None
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one global {action_name} action, found {matches}.")
    return matches[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a rectangular CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run paired Sleep, Monitor, and random baselines for each q."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilot2_observation_16.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--policy-seed", type=int, default=2027)
    parser.add_argument("--retention-levels", nargs="+", type=float)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scenario = config["scenario"]
    training = config["training"]
    q_values = tuple(float(q) for q in config["observation"]["q_values"])
    retention_levels = args.retention_levels or [
        float(q) for q in config["evaluation"]["retention_levels"]
    ]
    episodes = args.episodes or int(training["final_evaluation_episodes"])
    environment_seed = (
        args.seed
        if args.seed is not None
        else int(training["final_evaluation_base_seed"])
    )
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    output_dir = args.output_dir or Path(config["output_root"]) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for q in retention_levels:
        for baseline in BASELINES:
            env = make_pilot2_env(
                "flat",
                scenario_path=Path(str(scenario["path"])),
                node_count=int(scenario["node_count"]),
                q_values=q_values,
                fixed_q=q,
                max_steps=int(scenario["max_episode_steps"]),
                flat_include_adjacency=bool(
                    config["observation"]["flat_include_adjacency"]
                ),
            )
            env.reset(seed=environment_seed * 1000)
            if baseline == "random":
                # Reinitialize at every q so random action sequences are paired.
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
                "retention_q": q,
                **{key: value for key, value in metrics.items() if key != "episodes"},
            }
            aggregate_rows.append(aggregate)
            episode_rows.extend(
                {"baseline": baseline, "retention_q": q, **episode}
                for episode in metrics["episodes"]
            )
            results.append({**aggregate, "episodes": metrics["episodes"]})
            print(
                f"{baseline} q={q:.2f}: "
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
        "paired_across_q": True,
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
