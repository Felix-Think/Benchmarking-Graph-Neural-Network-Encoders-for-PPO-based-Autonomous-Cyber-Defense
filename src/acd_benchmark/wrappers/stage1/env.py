"""Stage 1 graph and matched flat environments for scalable CAGE-2."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from acd_benchmark.cyber.action_space import BlueActionSpec, build_blue_discrete_action_specs
from acd_benchmark.cyber.env_factory import make_scalable_cage2_env
from acd_benchmark.cyber.scalable_scenario import ScalableTopology, build_static_topology
from acd_benchmark.wrappers.shared.flat_obs import encode_flat_observation
from acd_benchmark.wrappers.shared.graph_obs import build_graph_encoder, encode_graph_observation
from acd_benchmark.wrappers.shared.host_obs import GraphObservation, RawBlueHostEncoder
from acd_benchmark.wrappers.stage1.observation import build_stage1_observation


class Stage1GraphEnv:
    """16-node environment with reproducible Bernoulli host masking."""

    def __init__(
        self,
        scenario_path: Path,
        *,
        node_count: int = 16,
        observation_probability: float = 1.0,
        max_steps: int = 100,
    ) -> None:
        if not 0.0 <= observation_probability <= 1.0:
            raise ValueError("observation_probability must be in [0, 1].")
        self.scenario_path = scenario_path
        self.topology: ScalableTopology = build_static_topology(node_count)
        self.observation_probability = observation_probability
        self.max_steps = max_steps
        self.cyborg = make_scalable_cage2_env(scenario_path)
        self.encoder: RawBlueHostEncoder | None = None
        self.step_count = 0
        self._action_specs: tuple[BlueActionSpec, ...] = ()
        self._mask_rng = np.random.default_rng()
        self._mask_rng_initialized = False
        self._feedback_history: deque[str | None] = deque(maxlen=2)

    @property
    def action_specs(self) -> tuple[BlueActionSpec, ...]:
        """Return the fixed scalable discrete action mapping."""
        return self._action_specs

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return readable actions in discrete order."""
        return tuple(spec.name for spec in self._action_specs)

    @property
    def action_count(self) -> int:
        """Return the fixed action count."""
        return len(self._action_specs)

    def reset(self, seed: int | None = None) -> tuple[GraphObservation, dict[str, Any]]:
        """Reset environment, raw encoder, mask RNG, and feedback history."""
        result = self.cyborg.reset("Blue", seed=seed)
        self.step_count = 0
        self.encoder = build_graph_encoder(result.observation)
        if seed is not None:
            self._mask_rng = np.random.default_rng(seed + 1_000_003)
            self._mask_rng_initialized = True
        elif not self._mask_rng_initialized:
            self._mask_rng = np.random.default_rng()
            self._mask_rng_initialized = True
        self._feedback_history.clear()
        raw_graph = encode_graph_observation(self.encoder, result.observation)
        observation = self._encode(raw_graph)
        self._action_specs = build_blue_discrete_action_specs(
            self.cyborg.get_action_space("Blue"),
            self.topology.hostnames,
        )
        return observation, self._info(observation)

    def step(self, action: int) -> tuple[GraphObservation, float, bool, bool, dict[str, Any]]:
        """Apply one Blue action and mask only the resulting dynamic state."""
        if self.encoder is None:
            raise RuntimeError("Environment must be reset before stepping.")
        action_index = int(action)
        action_spec = self._action_specs[action_index]
        result = self.cyborg.step(
            "Blue",
            action_spec.instantiate(session=0, agent="Blue"),
        )
        self.step_count += 1
        self.encoder.apply_blue_action(
            action_name=action_spec.action_cls.__name__,
            hostname=action_spec.hostname,
        )
        self._feedback_history.append(action_spec.hostname)
        raw_graph = encode_graph_observation(self.encoder, result.observation)
        observation = self._encode(raw_graph)
        terminated = bool(result.done)
        truncated = self.step_count >= self.max_steps and not terminated
        info = self._info(observation)
        info.update(
            {
                "action_index": action_index,
                "action_name": action_spec.name,
                "action_hostname": action_spec.hostname,
                "success": result.observation.get("success"),
            }
        )
        return observation, float(result.reward), terminated, truncated, info

    def _encode(self, raw_graph: GraphObservation) -> GraphObservation:
        """Apply static topology, dynamic feature selection, and masking."""
        forced_hosts = tuple(
            hostname
            for hostname in self._feedback_history
            if hostname is not None
        )
        return build_stage1_observation(
            raw_graph_observation=raw_graph,
            topology=self.topology,
            observation_probability=self.observation_probability,
            rng=self._mask_rng,
            forced_visible_hosts=forced_hosts,
        )

    def _info(self, observation: GraphObservation) -> dict[str, Any]:
        """Expose audit metadata without exposing ground truth to the policy."""
        visibility = observation.node_features[:, -1]
        return {
            "step_count": self.step_count,
            "node_count": self.topology.node_count,
            "action_count": len(self._action_specs),
            "hostnames": observation.hostnames,
            "feature_names": observation.feature_names,
            "observation_probability": self.observation_probability,
            "visible_host_count": int(visibility.sum()),
            "visible_host_fraction": float(visibility.mean()),
            "edge_count": int(observation.edge_index.shape[1]),
            "edge_rule": self.topology.edge_rule,
        }


class Stage1FlatEnv:
    """Flatten exactly the same masked graph data and static adjacency."""

    def __init__(self, *args, **kwargs) -> None:
        self.graph_env = Stage1GraphEnv(*args, **kwargs)

    @property
    def cyborg(self):
        """Expose CybORG only for evaluation-only metrics."""
        return self.graph_env.cyborg

    @property
    def topology(self) -> ScalableTopology:
        """Expose the same frozen topology metadata."""
        return self.graph_env.topology

    @property
    def action_specs(self) -> tuple[BlueActionSpec, ...]:
        """Return the graph environment's action mapping."""
        return self.graph_env.action_specs

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return readable action names."""
        return self.graph_env.action_names

    @property
    def action_count(self) -> int:
        """Return the fixed action count."""
        return self.graph_env.action_count

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and flatten the Stage 1 graph observation."""
        observation, info = self.graph_env.reset(seed=seed)
        return encode_flat_observation(observation, include_adjacency=True), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step once and flatten the same masked graph observation."""
        observation, reward, terminated, truncated, info = self.graph_env.step(action)
        return (
            encode_flat_observation(observation, include_adjacency=True),
            reward,
            terminated,
            truncated,
            info,
        )
