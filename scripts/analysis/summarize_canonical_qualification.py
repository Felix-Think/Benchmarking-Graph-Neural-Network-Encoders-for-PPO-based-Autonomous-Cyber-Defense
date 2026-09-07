import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_comparison(
    left: dict[str, Any],
    right: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    left_returns = np.asarray(left["episode_returns"], dtype=np.float64)
    right_returns = np.asarray(right["episode_returns"], dtype=np.float64)
    if left_returns.shape != right_returns.shape:
        raise ValueError("Paired evaluations must contain the same number of episodes")

    delta = left_returns - right_returns
    sample_count = delta.size
    bootstrap_indices = rng.integers(
        0,
        sample_count,
        size=(20_000, sample_count),
    )
    bootstrap_means = delta[bootstrap_indices].mean(axis=1)
    delta_std = delta.std(ddof=1)

    result = {
        "mean_delta": round(float(delta.mean()), 2),
        "bootstrap_ci95": [
            round(float(value), 2)
            for value in np.quantile(bootstrap_means, [0.025, 0.975])
        ],
        "paired_dz": (
            round(float(delta.mean() / delta_std), 4)
            if delta_std > 0
            else None
        ),
        "wins_ties_losses": [
            int((delta > 0).sum()),
            int((delta == 0).sum()),
            int((delta < 0).sum()),
        ],
        "paired_t_p": float(stats.ttest_rel(left_returns, right_returns).pvalue),
    }
    if np.any(delta != 0):
        result["wilcoxon_p"] = float(stats.wilcoxon(delta).pvalue)
    return result


def _mean_std(evaluation: dict[str, Any]) -> dict[str, float]:
    return {
        "mean": round(float(evaluation["mean_return"]), 2),
        "std": round(float(evaluation["std_return"]), 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one canonical-observation qualification seed.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--flat-reference", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_dir = args.run_dir / f"seed_{args.seed}"

    def load(name: str) -> dict[str, Any]:
        value = _load_json(seed_dir / name)
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object in {name}")
        return value

    untrained = load("untrained_heldout_evaluation_deterministic.json")
    static_only = load("observation_gate_static_only_deterministic.json")
    final = load("evaluation_deterministic.json")
    best = load("best_evaluation_deterministic.json")
    final_stochastic = load("evaluation_stochastic.json")
    best_stochastic = load("best_evaluation_stochastic.json")
    config = load("config.json")
    periodic = _load_json(seed_dir / "periodic_evaluation.json")
    if not isinstance(periodic, list):
        raise TypeError("Expected periodic_evaluation.json to contain a list")

    checkpoint = torch.load(seed_dir / "checkpoint.pt", map_location="cpu")
    rng = np.random.default_rng(20_260_820 + args.seed)
    comparisons = {
        "best_minus_untrained": _paired_comparison(best, untrained, rng),
        "best_minus_static_only": _paired_comparison(best, static_only, rng),
        "static_only_minus_untrained": _paired_comparison(
            static_only,
            untrained,
            rng,
        ),
        "final_minus_untrained": _paired_comparison(final, untrained, rng),
    }

    flat_reference = None
    if args.flat_reference is not None:
        value = _load_json(args.flat_reference)
        if not isinstance(value, dict):
            raise TypeError("Expected the Flat reference to contain a JSON object")
        flat_reference = value
        comparisons["best_minus_flat_reference"] = _paired_comparison(
            best,
            flat_reference,
            rng,
        )

    learns = comparisons["best_minus_untrained"]["bootstrap_ci95"][0] > 0
    uses_dynamic = (
        comparisons["best_minus_static_only"]["bootstrap_ci95"][0] > 0
    )
    stable = comparisons["final_minus_untrained"]["bootstrap_ci95"][0] > 0

    deterministic = {
        "untrained": _mean_std(untrained),
        "static_only": _mean_std(static_only),
        "final": _mean_std(final),
        "best": _mean_std(best),
    }
    qualification = {
        f"learns_from_canonical_seed_{args.seed}": learns,
        f"uses_dynamic_canonical_evidence_seed_{args.seed}": uses_dynamic,
        "stable_at_final_checkpoint": stable,
        "paper_level_multi_seed_evidence": False,
    }
    if flat_reference is not None:
        deterministic["flat_seed_42_context_only"] = _mean_std(flat_reference)
        qualification["better_than_flat_seed_42_context_only"] = (
            comparisons["best_minus_flat_reference"]["bootstrap_ci95"][0] > 0
        )

    payload = {
        "protocol": {
            "training_seed": args.seed,
            "requested_simulator_steps": config["total_simulator_steps"],
            "actual_simulator_steps": checkpoint["total_simulator_steps"],
            "updates": checkpoint["total_updates"],
            "episode_steps": config["episode_steps"],
            "canonical_variant": config.get("canonical_variant", "action_aware"),
            "observation_dim": config.get("observation_dim"),
            "parameter_count": config.get("parameter_count"),
            "periodic_episode_seeds": "20000-20009",
            "held_out_episode_seeds": "10000-10099",
            "held_out_episodes": len(best["episode_returns"]),
        },
        "deterministic_held_out": deterministic,
        "stochastic_held_out": {
            "final": _mean_std(final_stochastic),
            "best": _mean_std(best_stochastic),
        },
        "paired_comparisons": comparisons,
        "qualification": qualification,
    }
    (seed_dir / "qualification_statistics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    best_periodic = max(periodic, key=lambda record: record["mean_return"])
    learning = comparisons["best_minus_untrained"]
    observation_use = comparisons["best_minus_static_only"]
    summary = f"""# Canonical qualification - seed {args.seed}

- Best periodic checkpoint: update {best_periodic['update']}, step {best_periodic['global_step']}, mean return {best_periodic['mean_return']:.2f} on 10 selection episodes.
- Held-out deterministic best: {best['mean_return']:.2f} +/- {best['std_return']:.2f}.
- Held-out deterministic untrained: {untrained['mean_return']:.2f} +/- {untrained['std_return']:.2f}.
- Held-out deterministic static-only: {static_only['mean_return']:.2f} +/- {static_only['std_return']:.2f}.
- Learning delta (best - untrained): {learning['mean_delta']:.2f}, bootstrap 95% CI {learning['bootstrap_ci95']}.
- Observation-use delta (best - static-only): {observation_use['mean_delta']:.2f}, bootstrap 95% CI {observation_use['bootstrap_ci95']}.
- Per-seed learning gate: {'PASS' if learns else 'FAIL'}.
- Per-seed dynamic-observation-use gate: {'PASS' if uses_dynamic else 'FAIL'}.
- This is one training seed; it is not paper-level evidence by itself.
"""
    (seed_dir / "QUALIFICATION_SUMMARY.md").write_text(
        summary,
        encoding="utf-8",
    )

    print(json.dumps(comparisons, indent=2))
    print(json.dumps(qualification, indent=2))


if __name__ == "__main__":
    main()
