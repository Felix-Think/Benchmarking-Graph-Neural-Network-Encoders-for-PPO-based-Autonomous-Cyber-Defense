"""Audit Pilot v2 integrity and expose baseline margins without overclaiming."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


def _expected_visible_count(q: float, node_count: int) -> int:
    """Apply the protocol's positive-value round-half-up rule."""
    return math.floor(q * node_count + 0.5)


def _finite_training_log(path: Path) -> bool:
    """Require optimization metrics to be finite on every update."""
    if not path.exists():
        return False
    required_finite = {
        "update_seconds",
        "train_steps_per_second",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "gradient_norm",
        "learning_rate",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not required_finite <= set(record):
            return False
        for key in required_finite:
            value = record[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return False
    return True


def _baseline_margins(
    reports: dict[tuple[str, int], dict[str, Any]],
    baseline_path: Path,
    q_values: list[float],
) -> list[dict[str, Any]]:
    """Expose directional margins for human scientific review."""
    if not baseline_path.exists():
        return []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    by_q: dict[float, list[dict[str, Any]]] = {}
    for result in baseline["results"]:
        by_q.setdefault(float(result["retention_q"]), []).append(result)

    rows = []
    models = sorted({model for model, _seed in reports})
    for model in models:
        model_reports = [
            report for (candidate, _seed), report in reports.items() if candidate == model
        ]
        for q in q_values:
            q_label = f"q{int(round(q * 100)):03d}"
            metrics = [
                report["final_evaluation"]["by_q"][q_label]
                for report in model_reports
                if q_label in report["final_evaluation"].get("by_q", {})
            ]
            baselines = by_q.get(q, [])
            if not metrics or not baselines:
                continue
            model_availability = mean(
                float(item["mean_mission_availability"]) for item in metrics
            )
            model_return = mean(float(item["mean_episode_return"]) for item in metrics)
            model_privileged = mean(
                float(item["mean_privileged_compromised_hosts"]) for item in metrics
            )
            rows.append(
                {
                    "model": model,
                    "retention_q": q,
                    "seed_count": len(metrics),
                    "availability_margin_vs_best_baseline": (
                        model_availability
                        - max(
                            float(item["mean_mission_availability"])
                            for item in baselines
                        )
                    ),
                    "return_margin_vs_best_baseline": (
                        model_return
                        - max(float(item["mean_episode_return"]) for item in baselines)
                    ),
                    "privileged_burden_improvement_vs_best_baseline": (
                        min(
                            float(item["mean_privileged_compromised_hosts"])
                            for item in baselines
                        )
                        - model_privileged
                    ),
                    "interpretation": (
                        "Positive margins favor the learned model. Review paired episode "
                        "uncertainty before a scientific go/no decision."
                    ),
                }
            )
    return rows


def main() -> None:
    """Fail only on protocol-integrity errors; report science separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/pilot2_observation_16"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilot2_observation_16.yaml"),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = [str(model) for model in config["matrix"]["models"]]
    seeds = [int(seed) for seed in config["matrix"]["seeds"]]
    q_values = [float(q) for q in config["evaluation"]["retention_levels"]]
    expected_timesteps = int(config["training"]["total_timesteps"])
    node_count = int(config["scenario"]["node_count"])
    checks: list[dict[str, Any]] = []
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    training_q_exposure: dict[str, Any] = {}

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "masking_contract",
        config["observation"]["masking"]
        == "fixed_budget_stratified_per_host_per_timestep",
        str(config["observation"]["masking"]),
    )
    check(
        "mixed_q_sampling_contract",
        config["observation"]["q_sampling"] == "uniform_per_episode",
        str(config["observation"]["q_sampling"]),
    )
    check(
        "no_target_visibility_leakage",
        config["observation"]["force_target_visible"] is False,
        "force_target_visible must be false; unit tests verify behavior.",
    )
    check(
        "no_action_validity_mask",
        config["action"]["validity_mask"] == "none",
        str(config["action"]["validity_mask"]),
    )
    check(
        "action_result_excluded_symmetrically",
        config["action"]["policy_observes_result"] is False,
        "Pilot 2-A uses a host-only policy schema; native action result is logged only.",
    )
    check(
        "no_learned_normalizer",
        config["observation"]["learned_normalizer"] == "none",
        str(config["observation"]["learned_normalizer"]),
    )
    check(
        "flat_has_no_adjacency",
        config["observation"]["flat_include_adjacency"] is False,
        "Flat is the canonical vector baseline; topology exposure is documented.",
    )
    check(
        "topology_claim_blocked",
        config["generalisation"]["status"]
        == "blocked_until_graph_and_simulator_reachability_are_coupled",
        str(config["generalisation"]["status"]),
    )

    expected_eval_seed = int(config["training"]["final_evaluation_base_seed"])
    expected_episode_count = int(config["training"]["final_evaluation_episodes"])
    expected_environment_seeds = [
        expected_eval_seed * 1000 + episode
        for episode in range(expected_episode_count)
    ]
    expected_labels = {f"q{int(round(q * 100)):03d}" for q in q_values}

    for model in models:
        for seed in seeds:
            run_dir = args.run_root / model / "obs_mixed" / f"seed_{seed}"
            report_path = run_dir / "report.json"
            exists = report_path.exists()
            check(
                f"report_exists:{model}:seed_{seed}",
                exists,
                str(report_path),
            )
            if not exists:
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            reports[(model, seed)] = report
            check(
                f"timesteps_complete:{model}:seed_{seed}",
                int(report.get("completed_timesteps", -1)) == expected_timesteps,
                f"observed={report.get('completed_timesteps')} expected={expected_timesteps}",
            )
            check(
                f"training_metrics_finite:{model}:seed_{seed}",
                _finite_training_log(run_dir / "training_metrics.jsonl"),
                str(run_dir / "training_metrics.jsonl"),
            )
            exposure = report.get("training_q_exposure", {})
            episode_counts = exposure.get("episode_counts", {})
            timestep_counts = exposure.get("timestep_counts", {})
            training_q_exposure[f"{model}:seed_{seed}"] = exposure
            check(
                f"mixed_q_all_levels_seen:{model}:seed_{seed}",
                set(episode_counts) == expected_labels
                and all(int(episode_counts[label]) > 0 for label in expected_labels),
                f"episode_counts={episode_counts}",
            )
            check(
                f"mixed_q_timestep_accounting:{model}:seed_{seed}",
                set(timestep_counts) == expected_labels
                and sum(int(value) for value in timestep_counts.values())
                == expected_timesteps,
                f"timestep_counts={timestep_counts} expected_total={expected_timesteps}",
            )
            final = report.get("final_evaluation", {})
            by_q = final.get("by_q", {})
            check(
                f"fixed_q_curve_complete:{model}:seed_{seed}",
                set(by_q) == expected_labels,
                f"observed={sorted(by_q)} expected={sorted(expected_labels)}",
            )
            for q in q_values:
                q_label = f"q{int(round(q * 100)):03d}"
                if q_label not in by_q:
                    continue
                episodes = by_q[q_label].get("episodes", [])
                expected_visible = _expected_visible_count(q, node_count)
                visibility_ok = len(episodes) == expected_episode_count and all(
                    episode.get("retention_q") == q
                    and episode.get("intended_visible_host_count") == expected_visible
                    and episode.get("min_visible_host_count") == expected_visible
                    and episode.get("max_visible_host_count") == expected_visible
                    for episode in episodes
                )
                check(
                    f"exact_visibility:{model}:seed_{seed}:{q_label}",
                    visibility_ok,
                    f"expected {expected_visible}/{node_count} visible hosts per step",
                )
                observed_seeds = [
                    int(episode["environment_seed"])
                    for episode in episodes
                    if "environment_seed" in episode
                ]
                check(
                    f"paired_environment_seeds:{model}:seed_{seed}:{q_label}",
                    observed_seeds == expected_environment_seeds,
                    f"expected base seed {expected_eval_seed}",
                )

    integrity_pass = all(item["passed"] for item in checks)
    scientific_margins = _baseline_margins(
        reports,
        args.run_root / "baselines" / "baseline_summary.json",
        [q for q in q_values if q in {1.0, 0.5}],
    )
    audit = {
        "protocol": config["protocol"],
        "protocol_integrity_pass": integrity_pass,
        "checks": checks,
        "training_q_exposure": training_q_exposure,
        "scientific_gate_status": (
            "manual_review_required" if scientific_margins else "baselines_missing"
        ),
        "scientific_gate_margins": scientific_margins,
        "scientific_gate_rule": (
            "Use q=1.0 as required learning qualification and q=0.5 as preferred "
            "partial-observation qualification. Interpret availability, return, and "
            "privileged-compromise margins jointly with paired episode uncertainty. "
            "q=0.25 is not a hard gate."
        ),
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "pilot2_protocol_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    failed = [item for item in checks if not item["passed"]]
    lines = [
        "# Pilot v2 Protocol Audit",
        "",
        f"Protocol integrity pass: **{integrity_pass}**",
        f"Scientific gate: **{audit['scientific_gate_status']}**",
        "",
        "The script fails automatically only for protocol-integrity errors. Scientific go/no requires paired uncertainty review.",
        "",
    ]
    if failed:
        lines.extend(["## Failed Checks", ""])
        lines.extend(f"- `{item['name']}`: {item['detail']}" for item in failed)
        lines.append("")
    if scientific_margins:
        lines.extend(
            [
                "## Baseline Margins",
                "",
                "Positive values favor the learned policy for all three columns.",
                "",
                "| Model | q | Availability | Return | Privileged burden |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in scientific_margins:
            lines.append(
                f"| {row['model']} | {row['retention_q']:.2f} | "
                f"{row['availability_margin_vs_best_baseline']:.4f} | "
                f"{row['return_margin_vs_best_baseline']:.4f} | "
                f"{row['privileged_burden_improvement_vs_best_baseline']:.4f} |"
            )
        lines.append("")
    (args.run_root / "pilot2_protocol_audit.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"Pilot v2 protocol integrity pass: {integrity_pass}")
    if not integrity_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
