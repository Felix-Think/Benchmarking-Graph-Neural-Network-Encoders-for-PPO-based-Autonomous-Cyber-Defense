from __future__ import annotations

import re

import numpy as np
from gymnasium.spaces import Discrete

from acd_benchmark.cyber.observation_schema import (
    AGENT_SUBNETS,
    MAX_SERVERS,
    MAX_SUBNETS_PER_AGENT,
    MAX_USERS,
    SUBNETS,
)


HOST_ACTIONS = ("Analyse", "DeployDecoy", "Remove", "Restore")
EDGE_ACTIONS = ("AllowTrafficZone", "BlockTrafficZone")
GLOBAL_ACTIONS = ("Monitor", "Sleep")
HOST_SLOTS_PER_SUBNET = MAX_SERVERS + MAX_USERS
STRUCTURED_HOST_SLOTS = MAX_SUBNETS_PER_AGENT * HOST_SLOTS_PER_SUBNET
STRUCTURED_EDGE_SLOTS = MAX_SUBNETS_PER_AGENT * (len(SUBNETS) - 1)
NUM_STRUCTURED_ACTIONS = (
    len(HOST_ACTIONS) * STRUCTURED_HOST_SLOTS
    + len(EDGE_ACTIONS) * STRUCTURED_EDGE_SLOTS
    + len(GLOBAL_ACTIONS)
)

_EDGE_LABEL = re.compile(
    r"^(AllowTrafficZone|BlockTrafficZone) "
    r"(?P<destination>\S+) \([^)]*\) <- "
    r"(?P<source>\S+) \([^)]*\)$"
)


def _hostnames(agent: str) -> list[str | None]:
    hostnames: list[str | None] = []
    controlled_subnets = AGENT_SUBNETS[agent]
    for local_slot in range(MAX_SUBNETS_PER_AGENT):
        if local_slot >= len(controlled_subnets):
            hostnames.extend([None] * HOST_SLOTS_PER_SUBNET)
            continue
        subnet = controlled_subnets[local_slot]
        hostnames.extend(
            f"{subnet}_server_host_{index}" for index in range(MAX_SERVERS)
        )
        hostnames.extend(f"{subnet}_user_host_{index}" for index in range(MAX_USERS))
    return hostnames


def _edge_slots(agent: str) -> list[tuple[str, str] | None]:
    edges: list[tuple[str, str] | None] = []
    controlled_subnets = AGENT_SUBNETS[agent]
    for local_slot in range(MAX_SUBNETS_PER_AGENT):
        if local_slot >= len(controlled_subnets):
            edges.extend([None] * (len(SUBNETS) - 1))
            continue
        destination = controlled_subnets[local_slot]
        edges.extend(
            (source, destination) for source in SUBNETS if source != destination
        )
    return edges


def structured_action_keys(agent: str) -> list[tuple]:
    keys = [
        ("host", action, hostname)
        for action in HOST_ACTIONS
        for hostname in _hostnames(agent)
    ]
    keys.extend(
        ("edge", action, edge) for action in EDGE_ACTIONS for edge in _edge_slots(agent)
    )
    keys.extend(("global", action) for action in GLOBAL_ACTIONS)
    if len(keys) != NUM_STRUCTURED_ACTIONS:
        raise AssertionError(f"Unexpected structured action count: {len(keys)}")
    return keys


def _official_action_key(label: str) -> tuple | None:
    if label.startswith("[Padding]"):
        return None
    label = label.removeprefix("[Invalid] ")
    if label in GLOBAL_ACTIONS:
        return ("global", label)

    edge_match = _EDGE_LABEL.match(label)
    if edge_match is not None:
        action = edge_match.group(1)
        edge = (
            edge_match.group("source"),
            edge_match.group("destination"),
        )
        return ("edge", action, edge)

    action, separator, hostname = label.partition(" ")
    if separator and action in HOST_ACTIONS:
        return ("host", action, hostname)
    return None


def structured_action_label(key: tuple) -> str:
    category = key[0]
    if category == "global":
        return str(key[1])
    if category == "host":
        action, hostname = key[1:]
        return f"{action} {hostname}" if hostname is not None else "[Invalid host slot]"
    action, edge = key[1:]
    if edge is None:
        return "[Invalid edge slot]"
    source, destination = edge
    return f"{action} {destination} <- {source}"


class CC4StructuredActionEnv:
    """Reindex the native CAGE 4 action catalogue by graph semantics.

    The wrapper changes only action ordering. Every valid structured action is
    translated back to the corresponding native ``BlueFixedActionWrapper``
    index before the simulator step. Native validity and semantic masks remain
    authoritative.
    """

    def __init__(self, env):
        self.env = env
        self.possible_agents = env.possible_agents
        self._action_space = Discrete(NUM_STRUCTURED_ACTIONS)
        self._keys = {
            agent: structured_action_keys(agent) for agent in self.possible_agents
        }
        self._labels = {
            agent: [structured_action_label(key) for key in self._keys[agent]]
            for agent in self.possible_agents
        }
        self._structured_to_native: dict[str, np.ndarray] = {}
        self._current_masks: dict[str, np.ndarray] = {}

    @property
    def node_feature_names(self):
        return self.env.node_feature_names

    @property
    def last_canonical(self):
        return self.env.last_canonical

    @property
    def last_raw_observations(self):
        return self.env.last_raw_observations

    @property
    def process_name(self) -> str:
        return "structured_action_catalog"

    def _build_mapping(self, agent: str) -> np.ndarray:
        native_keys = {
            key: index
            for index, label in enumerate(self.env.action_labels(agent))
            if (key := _official_action_key(label)) is not None
        }
        return np.asarray(
            [native_keys.get(key, -1) for key in self._keys[agent]],
            dtype=np.int64,
        )

    def _transform_infos(self, infos: dict[str, dict]) -> dict[str, dict]:
        transformed = {}
        for agent in self.possible_agents:
            mapping = self._structured_to_native[agent]
            native_mask = np.asarray(infos[agent]["action_mask"], dtype=np.bool_)
            structured_mask = np.zeros(NUM_STRUCTURED_ACTIONS, dtype=np.bool_)
            mapped = mapping >= 0
            structured_mask[mapped] = native_mask[mapping[mapped]]
            if not structured_mask.any():
                raise RuntimeError(f"No structured action is valid for {agent}")
            self._current_masks[agent] = structured_mask
            info = dict(infos[agent])
            info["action_mask"] = structured_mask
            info["action_catalog"] = "host_edge_global_v1"
            transformed[agent] = info
        return transformed

    def reset(self, seed: int | None = None):
        observations, infos = self.env.reset(seed=seed)
        self._structured_to_native = {
            agent: self._build_mapping(agent) for agent in self.possible_agents
        }
        return observations, self._transform_infos(infos)

    def step(self, actions: dict[str, int]):
        native_actions = {}
        for agent, structured_action in actions.items():
            action_id = int(structured_action)
            if not 0 <= action_id < NUM_STRUCTURED_ACTIONS:
                raise IndexError(f"Structured action out of range: {action_id}")
            if not self._current_masks[agent][action_id]:
                raise ValueError(
                    f"Structured action {action_id} is invalid for {agent}"
                )
            native_actions[agent] = int(self._structured_to_native[agent][action_id])

        observations, rewards, terminated, truncated, infos = self.env.step(
            native_actions
        )
        return (
            observations,
            rewards,
            terminated,
            truncated,
            self._transform_infos(infos),
        )

    def action_space(self, agent: str):
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return self._action_space

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_labels(self, agent: str):
        return self._labels[agent]
