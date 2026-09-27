from math import isfinite 
from statistics import mean, pstdev, stdev 

def _summary(values, *, sample_std = False) -> dict: 
    values = list(values) # dict -> list 
    valid = [float(value) for value in values if value is not None]
    if any(not isfinite(value) for value in valid):
        raise ValueError("Metric values must be finite or None.")

    minimum_count = 2 if sample_std else 1 
    std_function = stdev if sample_std else pstdev 

    return {
        "n": len(valid),
        "missing": len(values) - len(valid),
        "mean": mean(valid) if valid else None,
        "std": std_function(valid) if len(valid) >= minimum_count else None, 
    }

def aggregate_episodes(episode_returns, cyber_summaries) -> dict:
    """Weight episodes equally; TTR is conditional on completed recoveries.

    Each episode contributes its mean completed TTR, or None if none completed.
    Unrecovered outages stay separate; their observation time is not a TTR.
    """
    if not episode_returns or len(episode_returns) != len(cyber_summaries):
        raise ValueError("Provide one cyber summary per episode_returns.")
    result = {"episode_return": _summary(episode_returns)}

    for metric in (
        "availability",
        "compromise_burden",
        "compromised_host_count",
        "unique_compromised_host_count",
        "successful_impacts_per_tick",
        "impact_attempts",
        "impact_successes",
        "ticks",
    ):
        result[metric] = _summary(row[metric] for row in cyber_summaries)

    for action_name in ("GreenLocalWork", "GreenAccessService"):
        rates = []
        for row in cyber_summaries:
            tasks = [task for task in row["green_task_success"] if task["action_name"] == action_name]
            attempts = sum(task["attempts"] for task in tasks)
            successes = sum(task["success"] for task in tasks)
            rates.append(successes / attempts if attempts > 0 else None)

        result[f"{action_name}_success_rate"] = _summary(rates)

    recovery_metrics = {
        "ot_ttr_completed_ticks": [],
        "ot_recovery_count": [],
        "ot_unrecovered_count": [],
        "ot_recovery_rate": [],
    }
    for row in cyber_summaries:
        completed = row["ot_recovery_events"]
        unrecovered = row["ot_unrecovery_events"]
        outage_count = len(completed) + len(unrecovered)
        recovery_metrics["ot_ttr_completed_ticks"].append(
            mean(event["ttr_ticks"] for event in completed) if completed else None
        )
        recovery_metrics["ot_recovery_count"].append(len(completed))
        recovery_metrics["ot_unrecovered_count"].append(len(unrecovered))
        recovery_metrics["ot_recovery_rate"].append(
            len(completed) / outage_count if outage_count else None
        )
    result.update({name: _summary(values) for name, values in recovery_metrics.items()})
    return result

def aggregate_seeds(seed_summaries: dict[int, dict]) -> dict:
    if not seed_summaries:
        raise ValueError("Provide at least one trainig seed")
    first_seed_summary = next(iter(seed_summaries.values()))
    metrics = list(first_seed_summary.keys())

    if any(set(row) != set(metrics) for row in seed_summaries.values()):
        raise ValueError("All training seeds must have the same metric keys.")
    return {
        metric : _summary(
            [row[metric]["mean"] for row in seed_summaries.values()],
            sample_std= True,
        )
        for metric in metrics
    }
