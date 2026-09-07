from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FEATURE_GROUPS = (
    "passive_event_now",
    "analysis_now",
    "passive_event_memory",
    "analysis_memory",
    "observed_now",
    "evidence_age",
    "decoy",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Flat and GCN dynamic-evidence audits by episode."
    )
    parser.add_argument("--flat-prefix", type=Path, required=True)
    parser.add_argument("--gcn-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_822)
    return parser


def _read_csv(prefix: Path, suffix: str) -> list[dict[str, str]]:
    path = prefix.with_name(prefix.name + suffix)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    return value == "True"


def _episode_groups(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["episode"])].append(row)
    return dict(grouped)


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else np.nan


def _rate(rows: list[dict[str, str]], field: str) -> float:
    return float(np.mean([_bool(row[field]) for row in rows])) if rows else np.nan


def _episode_metrics(prefix: Path) -> dict[str, list[float]]:
    decisions = _episode_groups(_read_csv(prefix, "_decisions.csv.gz"))
    events = _episode_groups(_read_csv(prefix, "_events.csv.gz"))
    ticks = _episode_groups(_read_csv(prefix, "_ticks.csv.gz"))
    episode_ids = sorted(ticks)
    output = defaultdict(list)

    for episode in episode_ids:
        episode_decisions = decisions[episode]
        episode_events = events[episode]
        episode_ticks = ticks[episode]
        current_context = [
            row
            for row in episode_decisions
            if row["target_host"] and _bool(row["current_any"])
        ]
        memory_context = [
            row
            for row in episode_decisions
            if row["target_host"] and _bool(row["memory_any"])
        ]

        output["return"].append(
            float(sum(float(row["reward"]) for row in episode_ticks))
        )
        output["blue_reward_machine"].append(
            float(
                sum(float(row["blue_reward_machine"]) for row in episode_ticks)
            )
        )
        output["action_cost"].append(
            float(sum(float(row["action_cost"]) for row in episode_ticks))
        )
        output["current_evidence_decision_fraction"].append(
            _rate(episode_decisions, "current_any")
        )
        output["memory_evidence_decision_fraction"].append(
            _rate(episode_decisions, "memory_any")
        )
        output["evidence_visible_when_decision_ready"].append(
            _rate(episode_events, "decision_ready")
        )
        output["evidence_first_decision_within_1"].append(
            float(
                np.mean(
                    [
                        row["first_decision_latency"] != ""
                        and int(row["first_decision_latency"]) <= 1
                        for row in episode_events
                    ]
                )
            )
        )
        output["same_host_response_within_10"].append(
            _rate(episode_events, "same_host_response_within_10")
        )
        output["current_host_alignment_excess_over_uniform"].append(
            _rate(current_context, "target_current_evidence")
            - _mean(current_context, "uniform_host_current_chance")
        )
        output["memory_host_alignment_excess_over_uniform"].append(
            _rate(memory_context, "target_memory_evidence")
            - _mean(memory_context, "uniform_host_memory_chance")
        )
        output["full_static_total_variation"].append(
            _mean(episode_decisions, "full_static_total_variation")
        )
        for group in FEATURE_GROUPS:
            present = [
                row
                for row in episode_decisions
                if _bool(row[f"{group}_present"])
            ]
            output[f"tv_zero_{group}_when_present"].append(
                _mean(present, f"tv_zero_{group}")
            )

    return dict(output)


def _bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrapped = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrapped, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def compare(args: argparse.Namespace) -> dict:
    flat = _episode_metrics(args.flat_prefix)
    gcn = _episode_metrics(args.gcn_prefix)
    if flat.keys() != gcn.keys():
        raise ValueError("Flat and GCN audit schemas differ")

    rng = np.random.default_rng(args.bootstrap_seed)
    estimates = {"flat": {}, "gcn": {}, "gcn_minus_flat": {}}
    for metric in flat:
        flat_values = np.asarray(flat[metric], dtype=np.float64)
        gcn_values = np.asarray(gcn[metric], dtype=np.float64)
        if flat_values.shape != gcn_values.shape:
            raise ValueError(f"Episode count mismatch for {metric}")
        estimates["flat"][metric] = _bootstrap_mean(
            flat_values, samples=args.bootstrap_samples, rng=rng
        )
        estimates["gcn"][metric] = _bootstrap_mean(
            gcn_values, samples=args.bootstrap_samples, rng=rng
        )
        estimates["gcn_minus_flat"][metric] = _bootstrap_mean(
            gcn_values - flat_values,
            samples=args.bootstrap_samples,
            rng=rng,
        )

    result = {
        "unit_of_analysis": "held-out episode",
        "paired_by": "environment seed",
        "training_seed_count": 1,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "estimates": estimates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = compare(args)
    estimates = result["estimates"]
    for metric in (
        "return",
        "current_host_alignment_excess_over_uniform",
        "same_host_response_within_10",
        "tv_zero_passive_event_now_when_present",
        "tv_zero_passive_event_memory_when_present",
        "tv_zero_evidence_age_when_present",
    ):
        difference = estimates["gcn_minus_flat"][metric]
        print(
            f"{metric}: gcn-flat={difference['mean']:.6f} "
            f"CI=[{difference['ci_low']:.6f}, {difference['ci_high']:.6f}]"
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
