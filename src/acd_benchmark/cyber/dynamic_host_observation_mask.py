from __future__ import annotations

import numpy as np

from acd_benchmark.cyber.observation_schema import (
    FEATURE_INDEX,
    MAX_LOCAL_HOSTS,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
)


STATIC_NODE_FEATURES = (
    "exists",
    "is_subnet",
    "is_host",
    "is_user",
    "is_server",
    "controlled",
)
DYNAMIC_NODE_FEATURES = tuple(
    name for name in NODE_FEATURES if name not in STATIC_NODE_FEATURES
)
VALID_MASK_PROCESSES = ("iid", "persistent")
MASK_PROCESS_NAMES = {
    "iid": "iid_fixed_budget_dynamic_host",
    "persistent": "persistent_fixed_budget_dynamic_host",
}


class FixedBudgetDynamicHostMask:
    """Retain dynamic features for an exact, subnet-stratified host budget.

    Static node identity, graph structure, global features, rewards and action
    masks are unchanged. IID mode samples a new subset every simulator tick;
    persistent mode samples once at reset and reuses it for the whole episode.
    This is an output-level zero-shot stress test, not a native CAGE 4 sensor
    model and not a topology-removal intervention.
    """

    def __init__(
        self,
        env,
        *,
        retention: float,
        mask_seed: int = 30_000,
        mask_process: str = "iid",
    ):
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention must be in [0, 1]")
        if tuple(getattr(env, "node_feature_names", ())) != NODE_FEATURES:
            raise ValueError("Host masking requires the current canonical schema")
        if mask_process not in VALID_MASK_PROCESSES:
            raise ValueError(f"mask_process must be one of {VALID_MASK_PROCESSES}")

        self.env = env
        self.retention = float(retention)
        self.mask_seed = int(mask_seed)
        self.mask_process = mask_process
        self.process_name = MASK_PROCESS_NAMES[mask_process]
        self.possible_agents = env.possible_agents
        self.node_feature_names = NODE_FEATURES
        self._dynamic_indices = np.asarray(
            [FEATURE_INDEX[name] for name in DYNAMIC_NODE_FEATURES],
            dtype=np.int64,
        )
        self._episode_seed = 0
        self._tick = 0
        self.last_visibility: dict[str, np.ndarray] = {}
        self._persistent_visibility: dict[int, np.ndarray] = {}
        self._persistent_existing_hosts: dict[int, np.ndarray] = {}

    @property
    def last_canonical(self):
        return self.env.last_canonical

    def _rng(self, agent_index: int) -> np.random.Generator:
        selection_tick = self._tick if self.mask_process == "iid" else 0
        seed_sequence = np.random.SeedSequence(
            (self.mask_seed, self._episode_seed, selection_tick, agent_index)
        )
        return np.random.default_rng(seed_sequence)

    def _target_count(self, host_count: int) -> int:
        return min(
            host_count,
            max(0, int(np.floor(self.retention * host_count + 0.5))),
        )

    def _stratified_visible_hosts(
        self,
        existing_host_nodes: np.ndarray,
        host_membership: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        target = self._target_count(len(existing_host_nodes))
        if target == 0:
            return np.empty(0, dtype=np.int64)
        if target == len(existing_host_nodes):
            return existing_host_nodes.copy()

        groups: dict[int, list[int]] = {}
        for node_index in existing_host_nodes:
            host_row = int(node_index) - NUM_SUBNET_NODES
            subnet_columns = np.flatnonzero(host_membership[host_row] > 0.5)
            stratum = int(subnet_columns[0]) if len(subnet_columns) else -1
            groups.setdefault(stratum, []).append(int(node_index))

        strata = sorted(groups)
        sizes = np.asarray([len(groups[stratum]) for stratum in strata], dtype=int)
        ideal = self.retention * sizes.astype(np.float64)
        quotas = np.floor(ideal).astype(int)
        remaining = target - int(quotas.sum())

        if remaining > 0:
            # Random tie-breaking is deterministic under the paired mask seed.
            tie_break = rng.random(len(strata))
            order = np.lexsort((tie_break, -(ideal - quotas)))
            for index in order[:remaining]:
                quotas[index] += 1

        selected: list[int] = []
        for stratum, quota in zip(strata, quotas, strict=True):
            candidates = np.asarray(groups[stratum], dtype=np.int64)
            if quota:
                chosen = rng.choice(candidates, size=int(quota), replace=False)
                selected.extend(int(node) for node in chosen)
        return np.asarray(sorted(selected), dtype=np.int64)

    def _mask_observation(
        self,
        observation: np.ndarray,
        *,
        agent_index: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        transformed = np.asarray(observation, dtype=np.float32).copy()
        node_size = NUM_NODES * len(NODE_FEATURES)
        membership_size = MAX_LOCAL_HOSTS * NUM_SUBNET_NODES
        nodes = transformed[:node_size].reshape(NUM_NODES, len(NODE_FEATURES))
        membership = transformed[node_size : node_size + membership_size].reshape(
            MAX_LOCAL_HOSTS, NUM_SUBNET_NODES
        )

        existing_host_nodes = np.flatnonzero(
            (nodes[:, FEATURE_INDEX["exists"]] > 0.5)
            & (nodes[:, FEATURE_INDEX["is_host"]] > 0.5)
        ).astype(np.int64)
        if (
            self.mask_process == "persistent"
            and agent_index in self._persistent_visibility
        ):
            expected_hosts = self._persistent_existing_hosts[agent_index]
            if not np.array_equal(existing_host_nodes, expected_hosts):
                raise RuntimeError(
                    "Existing host slots changed within a persistent-mask episode"
                )
            visible_host_nodes = self._persistent_visibility[agent_index]
        else:
            visible_host_nodes = self._stratified_visible_hosts(
                existing_host_nodes,
                membership,
                self._rng(agent_index),
            )
            if self.mask_process == "persistent":
                self._persistent_visibility[agent_index] = visible_host_nodes.copy()
                self._persistent_existing_hosts[agent_index] = (
                    existing_host_nodes.copy()
                )

        visible = np.zeros(NUM_NODES, dtype=np.bool_)
        visible[visible_host_nodes] = True
        hidden_host_nodes = existing_host_nodes[~visible[existing_host_nodes]]
        if len(hidden_host_nodes):
            nodes[np.ix_(hidden_host_nodes, self._dynamic_indices)] = 0.0
        return transformed, visible, len(existing_host_nodes)

    def _transform(
        self,
        observations: dict[str, np.ndarray],
        infos: dict[str, dict],
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        transformed_observations = {}
        transformed_infos = {}
        self.last_visibility = {}

        for agent_index, agent in enumerate(self.possible_agents):
            observation, visibility, host_count = self._mask_observation(
                observations[agent],
                agent_index=agent_index,
            )
            transformed_observations[agent] = observation
            self.last_visibility[agent] = visibility

            info = dict(infos[agent])
            info["observation_mask_process"] = self.process_name
            info["dynamic_host_retention"] = self.retention
            info["dynamic_hosts_total"] = host_count
            info["dynamic_hosts_visible"] = int(visibility.sum())
            transformed_infos[agent] = info

        return transformed_observations, transformed_infos

    def reset(self, seed: int | None = None):
        observations, infos = self.env.reset(seed=seed)
        self._episode_seed = 0 if seed is None else int(seed)
        self._tick = 0
        self._persistent_visibility = {}
        self._persistent_existing_hosts = {}
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
