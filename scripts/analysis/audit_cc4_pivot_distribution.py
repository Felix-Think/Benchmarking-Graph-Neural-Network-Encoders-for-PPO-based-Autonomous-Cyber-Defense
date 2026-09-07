import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the random-pivot CAGE 4 distribution breaks a "
            "fixed host-slot shortcut."
        )
    )
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--train-episodes", type=int, default=251)
    parser.add_argument("--test-seed", type=int, default=10_000)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--episode-steps", type=int, default=400)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cc4_random_pivot_distribution_audit.json"),
    )
    return parser


def _generator(env: CC4CanonicalFlatEnv):
    return env.env.env.scenario_generator


def _collect_assignments(
    env: CC4CanonicalFlatEnv,
    *,
    first_seed: int,
    episodes: int,
) -> list[dict[str, str]]:
    records = []
    for episode_index in range(episodes):
        env.reset(seed=first_seed + episode_index)
        records.append(dict(_generator(env).last_pivot_assignments))
    return records


def _normalised_entropy(counts: Counter[str]) -> float:
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    probabilities = np.asarray(
        [count / total for count in counts.values()], dtype=np.float64
    )
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(counts))


def _summarise(records: list[dict[str, str]]) -> dict:
    subnets = sorted(records[0])
    by_subnet = {}
    for subnet in subnets:
        counts = Counter(record[subnet] for record in records)
        most_common_host, most_common_count = counts.most_common(1)[0]
        server_zero = f"{subnet}_server_host_0"
        by_subnet[subnet] = {
            "unique_pivot_hosts": len(counts),
            "most_common_host": most_common_host,
            "most_common_fraction": most_common_count / len(records),
            "official_server_host_0_fraction": counts[server_zero] / len(records),
            "user_host_fraction": sum(
                count
                for hostname, count in counts.items()
                if "_user_host_" in hostname
            )
            / len(records),
            "normalised_entropy_over_observed_slots": _normalised_entropy(counts),
            "counts": dict(sorted(counts.items())),
        }

    assignment_maps = [tuple(sorted(record.items())) for record in records]
    return {
        "episodes": len(records),
        "unique_assignment_maps": len(set(assignment_maps)),
        "per_subnet": by_subnet,
    }


def _fixed_slot_test_hit_rate(
    train_records: list[dict[str, str]],
    test_records: list[dict[str, str]],
) -> dict:
    subnets = sorted(train_records[0])
    learned_slots = {
        subnet: Counter(record[subnet] for record in train_records).most_common(1)[0][0]
        for subnet in subnets
    }
    matches = sum(
        record[subnet] == learned_slots[subnet]
        for record in test_records
        for subnet in subnets
    )
    official_matches = sum(
        record[subnet] == f"{subnet}_server_host_0"
        for record in test_records
        for subnet in subnets
    )
    denominator = len(test_records) * len(subnets)
    return {
        "definition": (
            "Fraction of held-out subnet pivots matched by one fixed host slot "
            "per subnet; this is a shortcut proxy, not policy return."
        ),
        "train_modal_slots": learned_slots,
        "train_modal_slot_test_hit_rate": matches / denominator,
        "official_server_host_0_test_hit_rate": official_matches / denominator,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    env = CC4CanonicalFlatEnv(
        episode_steps=args.episode_steps,
        seed=args.train_seed,
        environment_distribution="random_host",
    )
    train_records = _collect_assignments(
        env,
        first_seed=args.train_seed,
        episodes=args.train_episodes,
    )
    test_records = _collect_assignments(
        env,
        first_seed=args.test_seed,
        episodes=args.test_episodes,
    )

    train_maps = {tuple(sorted(record.items())) for record in train_records}
    test_maps = {tuple(sorted(record.items())) for record in test_records}
    report = {
        "distribution": "random_host",
        "train_seed_range": [
            args.train_seed,
            args.train_seed + args.train_episodes - 1,
        ],
        "test_seed_range": [
            args.test_seed,
            args.test_seed + args.test_episodes - 1,
        ],
        "seed_overlap": bool(
            set(range(args.train_seed, args.train_seed + args.train_episodes))
            & set(range(args.test_seed, args.test_seed + args.test_episodes))
        ),
        "exact_assignment_map_overlap": len(train_maps & test_maps),
        "train": _summarise(train_records),
        "test": _summarise(test_records),
        "fixed_slot_shortcut": _fixed_slot_test_hit_rate(
            train_records, test_records
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report["fixed_slot_shortcut"], indent=2, sort_keys=True))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
