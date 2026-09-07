from __future__ import annotations

import numpy as np

from acd_benchmark.cyber.canonical_observation import (
    MAX_LOCAL_HOSTS,
    NUM_NODES,
    NUM_SUBNET_NODES,
)
from acd_benchmark.cyber.canonical_observation_v2 import (
    PASSIVE_MEMORY_FEATURES,
    PASSIVE_NOW_FEATURES,
)


VALID_OBSERVATION_CONDITIONS = (
    "full",
    "no_evidence",
    "action_state_only",
    "topology_only",
    "zero_with_mask",
)

STATIC_NODE_FEATURES = (
    "exists",
    "is_subnet",
    "is_host",
    "is_user",
    "is_server",
    "controlled",
)

# These are sensor outputs or beliefs derived from sensor outputs. Analyse
# completion/history is deliberately excluded: it is action feedback, while
# file density is the information returned by that action.
V1_SENSOR_EVIDENCE_FEATURES = (
    "observed_now",
    "process_event_now",
    "connection_event_now",
    "max_file_density_now",
    "process_event_memory",
    "connection_event_memory",
    "max_file_density_memory",
    "evidence_age_normalized",
)

V2_SENSOR_EVIDENCE_FEATURES = (
    "observed_now",
    *PASSIVE_NOW_FEATURES,
    *PASSIVE_MEMORY_FEATURES,
    "passive_memory_active",
    "passive_memory_age_normalized",
    "max_file_density_now",
    "max_file_density_memory",
)


class CanonicalObservationCondition:
    """Apply a fixed canonical-information condition during train and eval.

    Every condition preserves the observation dimension and the environment's
    official action mask. This allows matched policies to differ only in the
    information available through the canonical observation.
    """

    def __init__(self, env, condition: str = "full"):
        if condition not in VALID_OBSERVATION_CONDITIONS:
            raise ValueError(
                "condition must be one of "
                f"{VALID_OBSERVATION_CONDITIONS}, received {condition!r}"
            )

        self.env = env
        self.condition = condition
        self.possible_agents = env.possible_agents
        self.node_feature_names = tuple(env.node_feature_names)
        self._feature_index = {
            name: index for index, name in enumerate(self.node_feature_names)
        }
        self._node_size = NUM_NODES * len(self.node_feature_names)
        self._membership_size = MAX_LOCAL_HOSTS * NUM_SUBNET_NODES
        self._matrix_size = NUM_SUBNET_NODES * NUM_SUBNET_NODES
        self._membership_start = self._node_size
        self._structural_start = self._membership_start + self._membership_size
        self._communication_start = self._structural_start + self._matrix_size
        self._firewall_start = self._communication_start + self._matrix_size
        self._global_start = self._firewall_start + self._matrix_size

        sensor_candidates = (
            set(V1_SENSOR_EVIDENCE_FEATURES)
            | set(V2_SENSOR_EVIDENCE_FEATURES)
        )
        self._sensor_indices = np.asarray(
            [
                index
                for index, name in enumerate(self.node_feature_names)
                if name in sensor_candidates
            ],
            dtype=np.int64,
        )
        self._static_indices = np.asarray(
            [
                self._feature_index[name]
                for name in STATIC_NODE_FEATURES
                if name in self._feature_index
            ],
            dtype=np.int64,
        )
        self._decoy_index = self._feature_index.get("decoy_active")

    @property
    def last_canonical(self):
        return self.env.last_canonical

    @property
    def process_name(self) -> str:
        return self.condition

    def _nodes(self, observation: np.ndarray) -> np.ndarray:
        return observation[: self._node_size].reshape(
            NUM_NODES,
            len(self.node_feature_names),
        )

    def _apply(self, observation: np.ndarray) -> np.ndarray:
        transformed = np.asarray(observation, dtype=np.float32).copy()

        if self.condition == "full":
            return transformed

        if self.condition == "zero_with_mask":
            transformed.fill(0.0)
            return transformed

        nodes = self._nodes(transformed)

        if self.condition == "no_evidence":
            nodes[:, self._sensor_indices] = 0.0
            return transformed

        if self.condition == "action_state_only":
            decoy_state = (
                nodes[:, self._decoy_index].copy()
                if self._decoy_index is not None
                else None
            )
            firewall_state = transformed[
                self._firewall_start : self._global_start
            ].copy()
            global_state = transformed[self._global_start :].copy()
            transformed.fill(0.0)
            if decoy_state is not None:
                self._nodes(transformed)[:, self._decoy_index] = decoy_state
            transformed[self._firewall_start : self._global_start] = firewall_state
            transformed[self._global_start :] = global_state
            return transformed

        if self.condition == "topology_only":
            static_nodes = nodes[:, self._static_indices].copy()
            membership = transformed[
                self._membership_start : self._structural_start
            ].copy()
            structural = transformed[
                self._structural_start : self._communication_start
            ].copy()
            transformed.fill(0.0)
            self._nodes(transformed)[:, self._static_indices] = static_nodes
            transformed[
                self._membership_start : self._structural_start
            ] = membership
            transformed[
                self._structural_start : self._communication_start
            ] = structural
            return transformed

        raise AssertionError(f"Unhandled observation condition: {self.condition}")

    def _transform_observations(
        self,
        observations: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return {
            agent: self._apply(observation)
            for agent, observation in observations.items()
        }

    def _annotate_infos(self, infos: dict[str, dict]) -> dict[str, dict]:
        transformed = {}
        for agent, info in infos.items():
            info = dict(info)
            info["observation_condition"] = self.condition
            transformed[agent] = info
        return transformed

    def reset(self, seed: int | None = None):
        observations, infos = self.env.reset(seed=seed)
        return self._transform_observations(observations), self._annotate_infos(infos)

    def step(self, actions: dict[str, int]):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return (
            self._transform_observations(observations),
            rewards,
            terminated,
            truncated,
            self._annotate_infos(infos),
        )

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_labels(self, agent: str):
        return self.env.action_labels(agent)
