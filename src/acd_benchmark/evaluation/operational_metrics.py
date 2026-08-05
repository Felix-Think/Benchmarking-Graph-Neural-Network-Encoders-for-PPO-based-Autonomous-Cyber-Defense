"""Ground-truth operational metrics that are never exposed to the policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from acd_benchmark.rl.policies.base import ActorCriticPolicy


DETECTION_ACTIONS = {"Analyse", "Remove", "Restore"}


@dataclass(frozen=True)
class GroundTruthSnapshot:
    """Evaluation-only summary of the simulator's hidden state."""

    red_presence_hosts: frozenset[str]
    red_privileged_hosts: frozenset[str]
    available_critical_hosts: frozenset[str]


@dataclass(frozen=True)
class EpisodeOperationalMetrics:
    """One episode's reward, resilience, latency, and inference metrics."""

    episode: int
    environment_seed: int
    episode_return: float
    steps: int
    mission_availability: float
    mean_privileged_compromised_hosts: float
    mean_red_presence_hosts: float
    detection_latency: float | None
    detection_observed_count: int
    detection_censored_count: int
    recovery_latency: float | None
    recovery_observed_count: int
    recovery_censored_count: int
    steps_to_operational_server: int | None
    operational_server_reached: bool
    mean_inference_ms: float
    p95_inference_ms: float
    retention_q: float | None
    masking_process: str | None
    intended_visible_host_count: int | None
    mean_visible_host_count: float | None
    min_visible_host_count: int | None
    max_visible_host_count: int | None
    action_success_count: int
    action_failure_count: int
    action_unknown_count: int


def snapshot_ground_truth(env: Any) -> GroundTruthSnapshot:
    """Read hidden CybORG state strictly for evaluation logging.

    Callers must never append this object or its fields to an observation.
    """
    state = env.cyborg.environment_controller.state
    red_sessions = tuple(state.sessions.get("Red", {}).values())
    red_presence = frozenset(
        session.hostname
        for session in red_sessions
        if session.active
    )
    privileged = set()
    for session in red_sessions:
        if not session.active:
            continue
        host = state.hosts[session.hostname]
        username = str(session.username).lower()
        os_name = getattr(host.os_type, "name", str(host.os_type)).lower()
        if (os_name == "linux" and username == "root") or (
            os_name == "windows" and username == "system"
        ):
            privileged.add(session.hostname)

    available = set()
    for hostname in env.topology.critical_hosts:
        host = state.hosts[hostname]
        service = host.services.get("OTService")
        if service is None:
            continue
        service_pid = service.get("process")
        process_exists = any(process.pid == service_pid for process in host.processes)
        if bool(service.get("active", True)) and process_exists:
            available.add(hostname)
    return GroundTruthSnapshot(
        red_presence_hosts=red_presence,
        red_privileged_hosts=frozenset(privileged),
        available_critical_hosts=frozenset(available),
    )


class EpisodeMetricTracker:
    """Track first compromise, detection, recovery, and service availability."""

    def __init__(
        self,
        *,
        episode: int,
        environment_seed: int,
        critical_hosts: tuple[str, ...],
        initial_snapshot: GroundTruthSnapshot,
        reset_info: dict[str, Any] | None = None,
    ) -> None:
        self.episode = episode
        self.environment_seed = environment_seed
        self.critical_hosts = frozenset(critical_hosts)
        self.first_compromise = {
            hostname: 0
            for hostname in initial_snapshot.red_presence_hosts
        }
        self.first_detection: dict[str, int] = {}
        self.first_recovery: dict[str, int] = {}
        self.was_present = set(initial_snapshot.red_presence_hosts)
        self.steps_to_operational_server = (
            0
            if self.critical_hosts & initial_snapshot.red_presence_hosts
            else None
        )
        self.availability_samples: list[float] = []
        self.privileged_samples: list[float] = []
        self.presence_samples: list[float] = []
        self.inference_ms: list[float] = []
        reset_info = reset_info or {}
        raw_q = reset_info.get(
            "retention_q",
            reset_info.get("observation_probability"),
        )
        self.retention_q = float(raw_q) if raw_q is not None else None
        self.masking_process = reset_info.get("masking_process")
        raw_intended_count = reset_info.get("intended_visible_host_count")
        self.intended_visible_host_count = (
            int(raw_intended_count) if raw_intended_count is not None else None
        )
        self.visible_host_counts: list[int] = []
        self.action_success_count = 0
        self.action_failure_count = 0
        self.action_unknown_count = 0

    def record_step(
        self,
        *,
        timestep: int,
        action_name: str,
        action_hostname: str | None,
        snapshot: GroundTruthSnapshot,
        inference_ms: float,
        step_info: dict[str, Any] | None = None,
    ) -> None:
        """Record state after one joint Blue/Red environment step."""
        for hostname in snapshot.red_presence_hosts:
            self.first_compromise.setdefault(hostname, timestep)
        if (
            action_hostname is not None
            and action_name in DETECTION_ACTIONS
            and action_hostname in self.first_compromise
        ):
            self.first_detection.setdefault(action_hostname, timestep)

        recovered = self.was_present - set(snapshot.red_presence_hosts)
        for hostname in recovered:
            if hostname in self.first_compromise:
                self.first_recovery.setdefault(hostname, timestep)
        self.was_present = set(snapshot.red_presence_hosts)

        if (
            self.steps_to_operational_server is None
            and self.critical_hosts & snapshot.red_presence_hosts
        ):
            self.steps_to_operational_server = timestep

        available_count = len(self.critical_hosts & snapshot.available_critical_hosts)
        self.availability_samples.append(available_count / max(len(self.critical_hosts), 1))
        self.privileged_samples.append(float(len(snapshot.red_privileged_hosts)))
        self.presence_samples.append(float(len(snapshot.red_presence_hosts)))
        self.inference_ms.append(inference_ms)
        step_info = step_info or {}
        visible_host_count = step_info.get("visible_host_count")
        if visible_host_count is not None:
            self.visible_host_counts.append(int(visible_host_count))
        action_success = str(step_info.get("action_success", "UNKNOWN")).upper()
        if action_success in {"TRUE", "SUCCESS", "SUCCEEDED"}:
            self.action_success_count += 1
        elif action_success in {"FALSE", "FAILURE", "FAILED"}:
            self.action_failure_count += 1
        else:
            self.action_unknown_count += 1

    def finalize(self, episode_return: float) -> EpisodeOperationalMetrics:
        """Finalize metrics with explicit horizon censoring for missing events."""
        steps = len(self.availability_samples)
        detection_latencies = []
        recovery_latencies = []
        detection_observed = 0
        recovery_observed = 0
        for hostname, compromise_step in self.first_compromise.items():
            if hostname in self.first_detection:
                detection_latencies.append(self.first_detection[hostname] - compromise_step)
                detection_observed += 1
            else:
                detection_latencies.append(max(steps - compromise_step, 0))
            if hostname in self.first_recovery:
                recovery_latencies.append(self.first_recovery[hostname] - compromise_step)
                recovery_observed += 1
            else:
                recovery_latencies.append(max(steps - compromise_step, 0))

        compromise_count = len(self.first_compromise)
        return EpisodeOperationalMetrics(
            episode=self.episode,
            environment_seed=self.environment_seed,
            episode_return=episode_return,
            steps=steps,
            mission_availability=float(np.mean(self.availability_samples)) if steps else 0.0,
            mean_privileged_compromised_hosts=(
                float(np.mean(self.privileged_samples)) if steps else 0.0
            ),
            mean_red_presence_hosts=float(np.mean(self.presence_samples)) if steps else 0.0,
            detection_latency=(
                float(np.mean(detection_latencies)) if detection_latencies else None
            ),
            detection_observed_count=detection_observed,
            detection_censored_count=compromise_count - detection_observed,
            recovery_latency=(
                float(np.mean(recovery_latencies)) if recovery_latencies else None
            ),
            recovery_observed_count=recovery_observed,
            recovery_censored_count=compromise_count - recovery_observed,
            steps_to_operational_server=self.steps_to_operational_server,
            operational_server_reached=self.steps_to_operational_server is not None,
            mean_inference_ms=float(np.mean(self.inference_ms)) if self.inference_ms else 0.0,
            p95_inference_ms=float(np.percentile(self.inference_ms, 95)) if self.inference_ms else 0.0,
            retention_q=self.retention_q,
            masking_process=self.masking_process,
            intended_visible_host_count=self.intended_visible_host_count,
            mean_visible_host_count=(
                float(np.mean(self.visible_host_counts))
                if self.visible_host_counts
                else None
            ),
            min_visible_host_count=(
                min(self.visible_host_counts) if self.visible_host_counts else None
            ),
            max_visible_host_count=(
                max(self.visible_host_counts) if self.visible_host_counts else None
            ),
            action_success_count=self.action_success_count,
            action_failure_count=self.action_failure_count,
            action_unknown_count=self.action_unknown_count,
        )


def _optional_mean(values: list[float | None]) -> float | None:
    """Mean only defined latency values without converting missing to zero."""
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def aggregate_operational_metrics(
    episodes: list[EpisodeOperationalMetrics],
) -> dict[str, Any]:
    """Aggregate episode records while retaining raw records for CSV output."""
    episode_returns = [episode.episode_return for episode in episodes]
    known_action_results = sum(
        x.action_success_count + x.action_failure_count for x in episodes
    )
    masking_processes = sorted(
        {x.masking_process for x in episodes if x.masking_process is not None}
    )
    return {
        "episode_count": len(episodes),
        "mean_episode_return": float(np.mean(episode_returns)),
        "std_episode_return": float(np.std(episode_returns)),
        "mean_mission_availability": float(np.mean([x.mission_availability for x in episodes])),
        "mean_privileged_compromised_hosts": float(
            np.mean([x.mean_privileged_compromised_hosts for x in episodes])
        ),
        "mean_red_presence_hosts": float(np.mean([x.mean_red_presence_hosts for x in episodes])),
        "mean_detection_latency": _optional_mean([x.detection_latency for x in episodes]),
        "detection_observed_count": sum(x.detection_observed_count for x in episodes),
        "detection_censored_count": sum(x.detection_censored_count for x in episodes),
        "mean_recovery_latency": _optional_mean([x.recovery_latency for x in episodes]),
        "recovery_observed_count": sum(x.recovery_observed_count for x in episodes),
        "recovery_censored_count": sum(x.recovery_censored_count for x in episodes),
        "operational_server_reach_rate": float(
            np.mean([x.operational_server_reached for x in episodes])
        ),
        "mean_steps_to_operational_server": _optional_mean(
            [
                float(x.steps_to_operational_server)
                if x.steps_to_operational_server is not None
                else None
                for x in episodes
            ]
        ),
        "mean_inference_ms": float(np.mean([x.mean_inference_ms for x in episodes])),
        "p95_episode_inference_ms": float(np.percentile([x.p95_inference_ms for x in episodes], 95)),
        "retention_q": _optional_mean([x.retention_q for x in episodes]),
        "masking_processes": masking_processes,
        "mean_visible_host_count": _optional_mean(
            [x.mean_visible_host_count for x in episodes]
        ),
        "action_success_count": sum(x.action_success_count for x in episodes),
        "action_failure_count": sum(x.action_failure_count for x in episodes),
        "action_unknown_count": sum(x.action_unknown_count for x in episodes),
        "action_success_rate_among_known": (
            sum(x.action_success_count for x in episodes) / known_action_results
            if known_action_results
            else None
        ),
        "episodes": [asdict(episode) for episode in episodes],
    }


def evaluate_policy_with_operational_metrics(
    env: Any,
    policy: ActorCriticPolicy,
    *,
    episodes: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate argmax policy actions and log hidden state separately."""
    policy.eval()

    def select_action(observation: Any) -> int:
        prepared = policy.prepare_observation(observation, device)
        with torch.no_grad():
            logits, _ = policy.forward_prepared(prepared)
        return int(torch.argmax(logits).item())

    return evaluate_action_selector_with_operational_metrics(
        env,
        select_action,
        episodes=episodes,
        seed=seed,
    )


def evaluate_action_selector_with_operational_metrics(
    env: Any,
    select_action: Callable[[Any], int],
    *,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate any action selector under the same hidden-state metric protocol."""
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    records: list[EpisodeOperationalMetrics] = []
    for episode in range(episodes):
        environment_seed = seed * 1000 + episode
        observation, reset_info = env.reset(seed=environment_seed)
        tracker = EpisodeMetricTracker(
            episode=episode,
            environment_seed=environment_seed,
            critical_hosts=env.topology.critical_hosts,
            initial_snapshot=snapshot_ground_truth(env),
            reset_info=reset_info,
        )
        terminated = False
        truncated = False
        episode_return = 0.0
        timestep = 0
        while not terminated and not truncated:
            started = perf_counter()
            action = int(select_action(observation))
            inference_ms = (perf_counter() - started) * 1000.0
            if not 0 <= action < len(env.action_specs):
                raise IndexError(f"Action selector returned invalid index: {action}")
            action_spec = env.action_specs[action]
            observation, reward, terminated, truncated, step_info = env.step(action)
            timestep += 1
            episode_return += float(reward)
            tracker.record_step(
                timestep=timestep,
                action_name=action_spec.action_cls.__name__,
                action_hostname=action_spec.hostname,
                snapshot=snapshot_ground_truth(env),
                inference_ms=inference_ms,
                step_info=step_info,
            )
        records.append(tracker.finalize(episode_return))
    return aggregate_operational_metrics(records)
