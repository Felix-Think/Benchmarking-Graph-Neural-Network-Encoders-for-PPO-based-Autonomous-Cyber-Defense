"""Pilot v2 environments with mixed-q fixed-budget host retention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from acd_benchmark.cyber.action_space import BlueActionSpec, build_blue_discrete_action_specs
from acd_benchmark.cyber.env_factory import make_scalable_cage2_env
from acd_benchmark.cyber.scalable_scenario import SUBNETS, ScalableTopology, build_static_topology
from acd_benchmark.wrappers.pilot2.observation import FixedBudgetMask, build_pilot2_observation
from acd_benchmark.wrappers.shared.flat_obs import encode_flat_observation
from acd_benchmark.wrappers.shared.graph_obs import build_graph_encoder, encode_graph_observation
from acd_benchmark.wrappers.shared.host_obs import GraphObservation, RawBlueHostEncoder


def _q_label(q: float) -> str:
    """Use the same stable labels as Pilot v2 evaluation artifacts."""
    return f"q{int(round(q * 100)):03d}"


class Pilot2GraphEnv:
    """Fixed-scale graph environment for mixed-q observation qualification."""

    def __init__(
        self,
        scenario_path: Path,
        *,
        node_count: int = 16,
        q_values: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
        fixed_q: float | None = None,
        max_steps: int = 100,
    ) -> None:
        if not q_values:
            raise ValueError("q_values cannot be empty.")
        if any(not 0.0 <= q <= 1.0 for q in q_values):
            raise ValueError("Every q value must be in [0, 1].")
        if fixed_q is not None and fixed_q not in q_values:
            raise ValueError("fixed_q must be one of q_values.")
        self.scenario_path = scenario_path
        self.topology: ScalableTopology = build_static_topology(node_count)
        self.q_values = tuple(float(q) for q in q_values)
        self.fixed_q = fixed_q
        self.max_steps = max_steps
        self.cyborg = make_scalable_cage2_env(scenario_path)
        self.encoder: RawBlueHostEncoder | None = None
        self.step_count = 0
        self.retention_q = self.q_values[0]
        self._action_specs: tuple[BlueActionSpec, ...] = ()
        self._mask_rng = np.random.default_rng()
        self._episode_rng = np.random.default_rng()
        self._rng_initialized = False
        self._tie_break_order = SUBNETS
        self._latest_mask: FixedBudgetMask | None = None
        self._q_episode_counts = {q: 0 for q in self.q_values}
        self._q_step_counts = {q: 0 for q in self.q_values}

    @property
    def action_specs(self) -> tuple[BlueActionSpec, ...]:
        return self._action_specs

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._action_specs)

    @property
    def action_count(self) -> int:
        return len(self._action_specs)

    @property
    def q_episode_counts(self) -> dict[str, int]:
        """Return auditable mixed-q episode exposure for this environment."""
        return {
            _q_label(q): self._q_episode_counts[q]
            for q in self.q_values
        }

    @property
    def q_step_counts(self) -> dict[str, int]:
        """Return auditable mixed-q timestep exposure for this environment."""
        return {
            _q_label(q): self._q_step_counts[q]
            for q in self.q_values
        }

    def reset(self, seed: int | None = None) -> tuple[GraphObservation, dict[str, Any]]:
        """Reset simulator and sample one q plus one subnet tie order per episode."""
        result = self.cyborg.reset("Blue", seed=seed)
        self.step_count = 0
        self.encoder = build_graph_encoder(result.observation)
        if seed is not None:
            self._episode_rng = np.random.default_rng(seed + 1_000_003)
            self._mask_rng = np.random.default_rng(seed + 2_000_003)
            self._rng_initialized = True
        elif not self._rng_initialized:
            self._episode_rng = np.random.default_rng()
            self._mask_rng = np.random.default_rng()
            self._rng_initialized = True

        self.retention_q = (
            float(self.fixed_q)
            if self.fixed_q is not None
            else float(self._episode_rng.choice(self.q_values))
        )
        self._q_episode_counts[self.retention_q] += 1
        self._tie_break_order = tuple(
            str(value) for value in self._episode_rng.permutation(SUBNETS)
        )
        raw_graph = encode_graph_observation(self.encoder, result.observation)
        observation = self._encode(raw_graph)
        self._action_specs = build_blue_discrete_action_specs(
            self.cyborg.get_action_space("Blue"),
            self.topology.hostnames,
        )
        return observation, self._info(observation)

    def step(self, action: int) -> tuple[GraphObservation, float, bool, bool, dict[str, Any]]:
        """Apply one action without forcing its target host to become visible."""
        if self.encoder is None:
            raise RuntimeError("Environment must be reset before stepping.")
        action_index = int(action)
        action_spec = self._action_specs[action_index]
        result = self.cyborg.step(
            "Blue",
            action_spec.instantiate(session=0, agent="Blue"),
        )
        self._q_step_counts[self.retention_q] += 1
        self.step_count += 1
        self.encoder.apply_blue_action(
            action_name=action_spec.action_cls.__name__,
            hostname=action_spec.hostname,
        )
        raw_graph = encode_graph_observation(self.encoder, result.observation)
        observation = self._encode(raw_graph)
        terminated = bool(result.done)
        truncated = self.step_count >= self.max_steps and not terminated
        info = self._info(observation)
        success = result.observation.get("success")
        info.update(
            {
                "action_index": action_index,
                "action_name": action_spec.name,
                "action_hostname": action_spec.hostname,
                "action_success": getattr(success, "name", str(success)),
            }
        )
        return observation, float(result.reward), terminated, truncated, info

    def _encode(self, raw_graph: GraphObservation) -> GraphObservation:
        observation, mask = build_pilot2_observation(
            raw_graph_observation=raw_graph,
            topology=self.topology,
            retention_q=self.retention_q,
            rng=self._mask_rng,
            tie_break_order=self._tie_break_order,
        )
        self._latest_mask = mask
        return observation

    def _info(self, observation: GraphObservation) -> dict[str, Any]:
        if self._latest_mask is None:
            raise RuntimeError("Mask metadata is unavailable before encoding.")
        visible_hosts = tuple(
            hostname
            for hostname, visible in zip(
                observation.hostnames,
                self._latest_mask.visibility,
            )
            if visible
        )
        return {
            "step_count": self.step_count,
            "node_count": self.topology.node_count,
            "action_count": len(self._action_specs),
            "hostnames": observation.hostnames,
            "feature_names": observation.feature_names,
            "retention_q": self.retention_q,
            "masking_process": "fixed_budget_stratified_per_host_per_timestep",
            "intended_visible_host_count": self._latest_mask.intended_visible_count,
            "visible_host_count": int(self._latest_mask.visibility.sum()),
            "visible_host_fraction": float(self._latest_mask.visibility.mean()),
            "visible_count_by_subnet": self._latest_mask.visible_count_by_subnet,
            "visible_hosts": visible_hosts,
            "subnet_quota_tie_break_order": self._tie_break_order,
            "edge_count": int(observation.edge_index.shape[1]),
            "edge_rule": self.topology.edge_rule,
        }


class Pilot2FlatEnv:
    """Flatten Pilot v2 host features with configurable adjacency exposure."""

    def __init__(self, *args, include_adjacency: bool = False, **kwargs) -> None:
        self.graph_env = Pilot2GraphEnv(*args, **kwargs)
        self.include_adjacency = include_adjacency

    @property
    def cyborg(self):
        return self.graph_env.cyborg

    @property
    def topology(self) -> ScalableTopology:
        return self.graph_env.topology

    @property
    def action_specs(self) -> tuple[BlueActionSpec, ...]:
        return self.graph_env.action_specs

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.graph_env.action_names

    @property
    def action_count(self) -> int:
        return self.graph_env.action_count

    @property
    def q_episode_counts(self) -> dict[str, int]:
        return self.graph_env.q_episode_counts

    @property
    def q_step_counts(self) -> dict[str, int]:
        return self.graph_env.q_step_counts

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.graph_env.reset(seed=seed)
        return (
            encode_flat_observation(
                observation,
                include_adjacency=self.include_adjacency,
            ),
            info,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.graph_env.step(action)
        return (
            encode_flat_observation(
                observation,
                include_adjacency=self.include_adjacency,
            ),
            reward,
            terminated,
            truncated,
            info,
        )
