from __future__ import annotations

import numpy as np

from acd_benchmark.cyber.canonical_observation import NUM_NODES
from acd_benchmark.cyber.canonical_observation_v2 import (
    PASSIVE_MEMORY_FEATURES,
    PASSIVE_NOW_FEATURES,
)

EVIDENCE_NODE_FEATURES = (
    "observed_now",
    "process_event_now",
    "connection_event_now",
    "analysis_result_now",
    "max_file_density_now",
    "process_event_memory",
    "connection_event_memory",
    "analysis_memory",
    "max_file_density_memory",
    "evidence_age_normalized",
)
V2_EVIDENCE_NODE_FEATURES = (
    "observed_now",
    *PASSIVE_NOW_FEATURES,
    *PASSIVE_MEMORY_FEATURES,
    "passive_memory_active",
    "passive_memory_age_normalized",
    "max_file_density_now",
    "max_file_density_memory",
)
ALL_EVIDENCE_NODE_FEATURES = tuple(
    dict.fromkeys((*EVIDENCE_NODE_FEATURES, *V2_EVIDENCE_NODE_FEATURES))
)
COUNTERFACTUAL_TRANSFORMS = ("shuffle_hosts", "time_shift", "zero_evidence")


class DynamicEvidenceCounterfactual:
    """Apply a paired counterfactual to canonical-v1 host evidence features."""

    def __init__(
        self,
        env,
        *,
        transform: str,
        intervention_seed: int = 40_000,
        time_shift_lag: int = 5,
    ):
        if transform not in COUNTERFACTUAL_TRANSFORMS:
            raise ValueError(f"transform must be one of {COUNTERFACTUAL_TRANSFORMS}")
        if time_shift_lag <= 0:
            raise ValueError("time_shift_lag must be positive")
        self.node_feature_names = tuple(getattr(env, "node_feature_names", ()))
        if not self.node_feature_names:
            raise ValueError("Environment must expose node_feature_names")
        self._feature_index = {
            name: index for index, name in enumerate(self.node_feature_names)
        }
        for required in ("exists", "is_host"):
            if required not in self._feature_index:
                raise ValueError(f"Canonical schema is missing {required!r}")
        self.evidence_node_features = tuple(
            name
            for name in ALL_EVIDENCE_NODE_FEATURES
            if name in self._feature_index
        )
        if not self.evidence_node_features:
            raise ValueError("Canonical schema exposes no recognized evidence features")

        self.env = env
        self.transform = transform
        self.intervention_seed = int(intervention_seed)
        self.time_shift_lag = int(time_shift_lag)
        self.possible_agents = env.possible_agents
        self._evidence_indices = np.asarray(
            [self._feature_index[name] for name in self.evidence_node_features],
            dtype=np.int64,
        )
        self._episode_seed = 0
        self._tick = 0
        self._history: dict[str, list[np.ndarray]] = {}

    @property
    def last_canonical(self):
        return self.env.last_canonical

    @property
    def process_name(self) -> str:
        if self.transform == "time_shift":
            return f"time_shift_evidence_lag_{self.time_shift_lag}"
        return f"{self.transform}_host_evidence"

    def _host_nodes(self, nodes: np.ndarray) -> np.ndarray:
        return np.flatnonzero(
            (nodes[:, self._feature_index["exists"]] > 0.5)
            & (nodes[:, self._feature_index["is_host"]] > 0.5)
        ).astype(np.int64)

    def _rng(self, agent_index: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence(
                (
                    self.intervention_seed,
                    self._episode_seed,
                    self._tick,
                    agent_index,
                )
            )
        )

    def _transform_agent(
        self,
        agent: str,
        agent_index: int,
        observation: np.ndarray,
    ) -> np.ndarray:
        transformed = np.asarray(observation, dtype=np.float32).copy()
        nodes = transformed[
            : NUM_NODES * len(self.node_feature_names)
        ].reshape(
            NUM_NODES,
            len(self.node_feature_names),
        )
        host_nodes = self._host_nodes(nodes)
        current_evidence = nodes[:, self._evidence_indices].copy()

        if self.transform == "zero_evidence":
            nodes[np.ix_(host_nodes, self._evidence_indices)] = 0.0
        elif self.transform == "shuffle_hosts":
            permutation = self._rng(agent_index).permutation(host_nodes)
            nodes[np.ix_(host_nodes, self._evidence_indices)] = current_evidence[
                permutation
            ]
        else:
            history = self._history.setdefault(agent, [])
            if len(history) < self.time_shift_lag:
                nodes[np.ix_(host_nodes, self._evidence_indices)] = 0.0
            else:
                delayed = history[-self.time_shift_lag]
                nodes[np.ix_(host_nodes, self._evidence_indices)] = delayed[
                    host_nodes
                ]
            history.append(current_evidence)

        return transformed

    def _transform(
        self,
        observations: dict[str, np.ndarray],
        infos: dict[str, dict],
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        transformed_observations = {
            agent: self._transform_agent(agent, index, observations[agent])
            for index, agent in enumerate(self.possible_agents)
        }
        transformed_infos = {}
        for agent in self.possible_agents:
            info = dict(infos[agent])
            info["dynamic_evidence_counterfactual"] = self.process_name
            transformed_infos[agent] = info
        return transformed_observations, transformed_infos

    def reset(self, seed: int | None = None):
        observations, infos = self.env.reset(seed=seed)
        self._episode_seed = 0 if seed is None else int(seed)
        self._tick = 0
        self._history = {}
        return self._transform(observations, infos)

    def step(self, actions: dict[str, int]):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self._tick += 1
        observations, infos = self._transform(observations, infos)
        return observations, rewards, terminated, truncated, infos

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_labels(self, agent: str):
        return self.env.action_labels(agent)
