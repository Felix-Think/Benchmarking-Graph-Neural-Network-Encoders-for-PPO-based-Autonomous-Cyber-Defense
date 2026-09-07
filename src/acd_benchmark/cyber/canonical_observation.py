from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions.AbstractActions import Analyse, Restore
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    AllowTrafficZone,
    BlockTrafficZone,
)
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DeployDecoy import (
    DeployDecoy,
)


SUBNETS = (
    "admin_network_subnet",
    "contractor_network_subnet",
    "internet_subnet",
    "office_network_subnet",
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "public_access_zone_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet",
)
SUBNET_INDEX = {name: index for index, name in enumerate(SUBNETS)}

AGENT_SUBNETS = {
    "blue_agent_0": ("restricted_zone_a_subnet",),
    "blue_agent_1": ("operational_zone_a_subnet",),
    "blue_agent_2": ("restricted_zone_b_subnet",),
    "blue_agent_3": ("operational_zone_b_subnet",),
    "blue_agent_4": (
        "admin_network_subnet",
        "office_network_subnet",
        "public_access_zone_subnet",
    ),
}

MAX_SUBNETS_PER_AGENT = 3
MAX_SERVERS = 6
MAX_USERS = 10
MAX_HOSTS_PER_SUBNET = MAX_SERVERS + MAX_USERS
MAX_LOCAL_HOSTS = MAX_SUBNETS_PER_AGENT * MAX_HOSTS_PER_SUBNET
NUM_SUBNET_NODES = len(SUBNETS)
NUM_NODES = NUM_SUBNET_NODES + MAX_LOCAL_HOSTS

NODE_FEATURES = (
    "exists",
    "is_subnet",
    "is_host",
    "is_user",
    "is_server",
    "controlled",
    "observed_now",
    "process_event_now",
    "connection_event_now",
    "analysis_result_now",
    "max_file_density_now",
    "process_event_memory",
    "connection_event_memory",
    "analysis_memory",
    "max_file_density_memory",
    "decoy_active",
    "evidence_age_normalized",
)
FEATURE_INDEX = {name: index for index, name in enumerate(NODE_FEATURES)}

GLOBAL_FEATURES = (
    "mission_phase_0",
    "mission_phase_1",
    "mission_phase_2",
    "success_unknown",
    "success_false",
    "success_true",
    "success_in_progress",
)

VALID_VARIANTS = ("instant", "belief", "action_aware")


@dataclass(frozen=True)
class CanonicalObservation:
    node_features: np.ndarray
    host_membership: np.ndarray
    structural_adjacency: np.ndarray
    communication_policy: np.ndarray
    firewall_blocks: np.ndarray
    global_features: np.ndarray

    def flat_view(self) -> np.ndarray:
        return np.concatenate(
            (
                self.node_features.reshape(-1),
                self.host_membership.reshape(-1),
                self.structural_adjacency.reshape(-1),
                self.communication_policy.reshape(-1),
                self.firewall_blocks.reshape(-1),
                self.global_features.reshape(-1),
            )
        ).astype(np.float32, copy=False)

    def graph_edge_index(self) -> np.ndarray:
        sources: list[int] = []
        destinations: list[int] = []

        host_rows, subnet_columns = np.nonzero(self.host_membership)
        for host_row, subnet_column in zip(host_rows, subnet_columns, strict=True):
            host_node = NUM_SUBNET_NODES + int(host_row)
            subnet_node = int(subnet_column)
            sources.extend((host_node, subnet_node))
            destinations.extend((subnet_node, host_node))

        subnet_sources, subnet_destinations = np.nonzero(self.structural_adjacency)
        sources.extend(subnet_sources.tolist())
        destinations.extend(subnet_destinations.tolist())
        return np.asarray((sources, destinations), dtype=np.int64)


def flat_observation_dim() -> int:
    return (
        NUM_NODES * len(NODE_FEATURES)
        + MAX_LOCAL_HOSTS * NUM_SUBNET_NODES
        + 3 * NUM_SUBNET_NODES * NUM_SUBNET_NODES
        + len(GLOBAL_FEATURES)
    )


def deserialize_flat_view(flat: np.ndarray) -> CanonicalObservation:
    flat = np.asarray(flat, dtype=np.float32)
    node_size = NUM_NODES * len(NODE_FEATURES)
    membership_size = MAX_LOCAL_HOSTS * NUM_SUBNET_NODES
    matrix_size = NUM_SUBNET_NODES * NUM_SUBNET_NODES
    expected = flat_observation_dim()
    if flat.shape != (expected,):
        raise ValueError(f"Expected flat shape ({expected},), received {flat.shape}")

    cursor = 0
    node_features = flat[cursor : cursor + node_size].reshape(
        NUM_NODES, len(NODE_FEATURES)
    )
    cursor += node_size
    host_membership = flat[cursor : cursor + membership_size].reshape(
        MAX_LOCAL_HOSTS, NUM_SUBNET_NODES
    )
    cursor += membership_size
    structural_adjacency = flat[cursor : cursor + matrix_size].reshape(
        NUM_SUBNET_NODES, NUM_SUBNET_NODES
    )
    cursor += matrix_size
    communication_policy = flat[cursor : cursor + matrix_size].reshape(
        NUM_SUBNET_NODES, NUM_SUBNET_NODES
    )
    cursor += matrix_size
    firewall_blocks = flat[cursor : cursor + matrix_size].reshape(
        NUM_SUBNET_NODES, NUM_SUBNET_NODES
    )
    cursor += matrix_size
    return CanonicalObservation(
        node_features=node_features,
        host_membership=host_membership,
        structural_adjacency=structural_adjacency,
        communication_policy=communication_policy,
        firewall_blocks=firewall_blocks,
        global_features=flat[cursor:],
    )


def mission_phase(step: int, episode_steps: int) -> int:
    # CAGE 4 evaluates the phase using its pre-increment simulator step.
    schedule_step = max(step - 1, 0)
    quotient, remainder = divmod(episode_steps, 3)
    lengths = [quotient, quotient, quotient]
    for index in range(remainder):
        lengths[index] += 1
    if schedule_step < lengths[0]:
        return 0
    if schedule_step < lengths[0] + lengths[1]:
        return 1
    return 2


def structural_subnet_adjacency() -> np.ndarray:
    """Return the fixed physical router-level topology of CAGE 4."""
    adjacency = np.zeros((NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32)
    edges = (
        ("internet_subnet", "restricted_zone_a_subnet"),
        ("internet_subnet", "restricted_zone_b_subnet"),
        ("internet_subnet", "contractor_network_subnet"),
        ("internet_subnet", "public_access_zone_subnet"),
        ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
        ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
        ("public_access_zone_subnet", "admin_network_subnet"),
        ("public_access_zone_subnet", "office_network_subnet"),
    )
    for source, destination in edges:
        adjacency[SUBNET_INDEX[source], SUBNET_INDEX[destination]] = 1.0
        adjacency[SUBNET_INDEX[destination], SUBNET_INDEX[source]] = 1.0
    return adjacency


def desired_communication_adjacency(phase: int) -> np.ndarray:
    """Return the public mission policy, where one means communication is desired."""
    adjacency = np.zeros((NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32)
    lan = (
        "internet_subnet",
        "admin_network_subnet",
        "office_network_subnet",
        "public_access_zone_subnet",
        "contractor_network_subnet",
        "restricted_zone_a_subnet",
        "restricted_zone_b_subnet",
    )
    for source in lan:
        for destination in lan:
            if source != destination:
                adjacency[SUBNET_INDEX[source], SUBNET_INDEX[destination]] = 1.0

    for source, destination in (
        ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
        ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
    ):
        adjacency[SUBNET_INDEX[source], SUBNET_INDEX[destination]] = 1.0
        adjacency[SUBNET_INDEX[destination], SUBNET_INDEX[source]] = 1.0

    removed = ()
    if phase == 1:
        removed = (
            ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
            ("restricted_zone_a_subnet", "contractor_network_subnet"),
            ("restricted_zone_a_subnet", "restricted_zone_b_subnet"),
            ("restricted_zone_a_subnet", "internet_subnet"),
        )
    elif phase == 2:
        removed = (
            ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
            ("restricted_zone_b_subnet", "contractor_network_subnet"),
            ("restricted_zone_b_subnet", "restricted_zone_a_subnet"),
            ("restricted_zone_b_subnet", "internet_subnet"),
        )
    for source, destination in removed:
        adjacency[SUBNET_INDEX[source], SUBNET_INDEX[destination]] = 0.0
        adjacency[SUBNET_INDEX[destination], SUBNET_INDEX[source]] = 0.0
    return adjacency


@dataclass(frozen=True)
class ObservedHostEvents:
    """Binary event indicators exposed by CAGE 4's automatic Monitor action."""

    process_creation: bool = False
    network_connection: bool = False


class HostCompressedBuilder:
    """Build a fixed graph from Blue-visible evidence and public CAGE 4 rules."""

    def __init__(
        self,
        agent: str,
        episode_steps: int,
        variant: str = "action_aware",
        evidence_age_horizon: int = 50,
    ):
        if agent not in AGENT_SUBNETS:
            raise ValueError(f"Unknown Blue agent: {agent}")
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}")
        if evidence_age_horizon <= 0:
            raise ValueError("evidence_age_horizon must be positive")
        self.agent = agent
        self.episode_steps = episode_steps
        self.variant = variant
        self.evidence_age_horizon = evidence_age_horizon
        self.node_features = np.zeros(
            (NUM_NODES, len(NODE_FEATURES)), dtype=np.float32
        )
        self.host_membership = np.zeros(
            (MAX_LOCAL_HOSTS, NUM_SUBNET_NODES), dtype=np.float32
        )
        self.firewall_blocked_edges: set[tuple[str, str]] = set()
        self._age_steps = np.zeros(NUM_NODES, dtype=np.int32)

    def reset(
        self,
        observation: dict,
        event_indicators: dict[str, ObservedHostEvents] | None = None,
    ) -> CanonicalObservation:
        self.node_features.fill(0.0)
        self.host_membership.fill(0.0)
        self.firewall_blocked_edges.clear()
        self._age_steps.fill(0)
        event_indicators = {} if event_indicators is None else event_indicators

        for subnet, subnet_index in SUBNET_INDEX.items():
            self.node_features[subnet_index, FEATURE_INDEX["exists"]] = 1.0
            self.node_features[subnet_index, FEATURE_INDEX["is_subnet"]] = 1.0
            if subnet in AGENT_SUBNETS[self.agent]:
                self.node_features[subnet_index, FEATURE_INDEX["controlled"]] = 1.0

        for local_subnet_index, _ in enumerate(AGENT_SUBNETS[self.agent]):
            for host_offset in range(MAX_HOSTS_PER_SUBNET):
                host_row = local_subnet_index * MAX_HOSTS_PER_SUBNET + host_offset
                node_index = NUM_SUBNET_NODES + host_row
                self.node_features[node_index, FEATURE_INDEX["is_host"]] = 1.0
                self.node_features[node_index, FEATURE_INDEX["controlled"]] = 1.0
                host_type = "is_server" if host_offset < MAX_SERVERS else "is_user"
                self.node_features[node_index, FEATURE_INDEX[host_type]] = 1.0

        for hostname in observation:
            node_index = self._node_index(hostname)
            if node_index is None:
                continue
            self.node_features[node_index, FEATURE_INDEX["exists"]] = 1.0
            self.node_features[node_index, FEATURE_INDEX["observed_now"]] = 1.0
            if node_index >= NUM_SUBNET_NODES:
                host_row = node_index - NUM_SUBNET_NODES
                subnet = self._subnet_from_hostname(hostname)
                self.host_membership[host_row, SUBNET_INDEX[subnet]] = 1.0

            self._update_node(
                node_index,
                observation.get(hostname, {}),
                event_indicators.get(hostname, ObservedHostEvents()),
            )

        return self._snapshot(observation, step=0)

    def update(
        self,
        observation: dict,
        step: int,
        event_indicators: dict[str, ObservedHostEvents] | None = None,
    ) -> CanonicalObservation:
        event_indicators = {} if event_indicators is None else event_indicators
        current_features = (
            "observed_now",
            "process_event_now",
            "connection_event_now",
            "analysis_result_now",
            "max_file_density_now",
        )
        self.node_features[
            :, [FEATURE_INDEX[feature] for feature in current_features]
        ] = 0.0

        if self.variant != "instant":
            has_memory = (
                self.node_features[
                    :,
                    [
                        FEATURE_INDEX["process_event_memory"],
                        FEATURE_INDEX["connection_event_memory"],
                        FEATURE_INDEX["analysis_memory"],
                    ],
                ].max(axis=1)
                > 0
            )
            self._age_steps[has_memory] += 1

        success = observation.get("success", TernaryEnum.UNKNOWN)
        action = observation.get("action")
        if success == TernaryEnum.TRUE:
            self._apply_successful_action(action)

        observed_hosts = {
            hostname
            for hostname in observation
            if hostname not in ("success", "message", "action")
        }
        observed_hosts.update(
            hostname
            for hostname, indicators in event_indicators.items()
            if indicators.process_creation or indicators.network_connection
        )
        for hostname in observed_hosts:
            if hostname in ("success", "message", "action"):
                continue
            node_index = self._node_index(hostname)
            if node_index is not None:
                self._update_node(
                    node_index,
                    observation.get(hostname, {}),
                    event_indicators.get(hostname, ObservedHostEvents()),
                )

        return self._snapshot(observation, step)

    def _update_node(
        self,
        node_index: int,
        host_data: dict,
        event_indicators: ObservedHostEvents,
    ) -> None:
        process_now = event_indicators.process_creation
        connection_now = event_indicators.network_connection
        files = host_data.get("Files", [])
        densities = [
            float(file_data["Density"])
            for file_data in files
            if file_data.get("Density") is not None
        ]
        max_density = max(densities, default=0.0)

        values = {
            "observed_now": 1.0,
            "process_event_now": float(process_now),
            "connection_event_now": float(connection_now),
            "max_file_density_now": float(np.clip(max_density, 0.0, 1.0)),
        }
        for feature, value in values.items():
            self.node_features[node_index, FEATURE_INDEX[feature]] = value

        if self.variant == "instant":
            return
        saw_evidence = False
        for feature, active in (
            ("process_event_memory", process_now),
            ("connection_event_memory", connection_now),
        ):
            if active:
                self.node_features[node_index, FEATURE_INDEX[feature]] = 1.0
                saw_evidence = True
        analysis_now = bool(
            self.node_features[node_index, FEATURE_INDEX["analysis_result_now"]]
        )
        if analysis_now:
            self.node_features[node_index, FEATURE_INDEX["analysis_memory"]] = 1.0
            density_memory_index = FEATURE_INDEX["max_file_density_memory"]
            self.node_features[node_index, density_memory_index] = max(
                self.node_features[node_index, density_memory_index],
                values["max_file_density_now"],
            )
            saw_evidence = True
        if saw_evidence:
            self._age_steps[node_index] = 0

    def _apply_successful_action(self, action) -> None:
        if isinstance(action, BlockTrafficZone):
            self.firewall_blocked_edges.add((action.from_subnet, action.to_subnet))
        elif isinstance(action, AllowTrafficZone):
            self.firewall_blocked_edges.discard((action.from_subnet, action.to_subnet))

        target = getattr(action, "hostname", None)
        node_index = self._node_index(target) if target is not None else None
        if isinstance(action, Analyse) and node_index is not None:
            self.node_features[node_index, FEATURE_INDEX["observed_now"]] = 1.0
            self.node_features[node_index, FEATURE_INDEX["analysis_result_now"]] = 1.0
            if self.variant != "instant":
                self.node_features[node_index, FEATURE_INDEX["analysis_memory"]] = 1.0
                self._age_steps[node_index] = 0

        if self.variant != "action_aware" or action is None:
            return
        if node_index is None:
            return

        # Remove intentionally has no branch: success does not prove the host is clean.
        if isinstance(action, Restore):
            for feature in (
                "process_event_memory",
                "connection_event_memory",
                "analysis_memory",
                "max_file_density_memory",
                "decoy_active",
            ):
                self.node_features[node_index, FEATURE_INDEX[feature]] = 0.0
            self._age_steps[node_index] = 0
        elif isinstance(action, DeployDecoy):
            self.node_features[node_index, FEATURE_INDEX["decoy_active"]] = 1.0

    def _snapshot(self, observation: dict, step: int) -> CanonicalObservation:
        if self.variant != "instant":
            ages = np.minimum(
                self._age_steps / float(self.evidence_age_horizon), 1.0
            )
            self.node_features[:, FEATURE_INDEX["evidence_age_normalized"]] = ages

        phase = mission_phase(step, self.episode_steps)
        structural_adjacency = structural_subnet_adjacency()
        communication_policy = desired_communication_adjacency(phase)
        firewall_blocks = np.zeros(
            (NUM_SUBNET_NODES, NUM_SUBNET_NODES), dtype=np.float32
        )
        for source, destination in self.firewall_blocked_edges:
            if source in SUBNET_INDEX and destination in SUBNET_INDEX:
                source_index = SUBNET_INDEX[source]
                destination_index = SUBNET_INDEX[destination]
                firewall_blocks[source_index, destination_index] = 1.0
                firewall_blocks[destination_index, source_index] = 1.0

        global_features = np.zeros(len(GLOBAL_FEATURES), dtype=np.float32)
        global_features[phase] = 1.0
        success = observation.get("success", TernaryEnum.UNKNOWN)
        if success == TernaryEnum.FALSE:
            success_index = 4
        elif success == TernaryEnum.TRUE:
            success_index = 5
        elif success == TernaryEnum.IN_PROGRESS:
            success_index = 6
        else:
            success_index = 3
        global_features[success_index] = 1.0

        return CanonicalObservation(
            node_features=self.node_features.copy(),
            host_membership=self.host_membership.copy(),
            structural_adjacency=structural_adjacency,
            communication_policy=communication_policy,
            firewall_blocks=firewall_blocks,
            global_features=global_features,
        )

    def _node_index(self, hostname: str | None) -> int | None:
        if not hostname:
            return None
        subnet = self._subnet_from_hostname(hostname)
        if subnet is None:
            return None
        if hostname.endswith("_router"):
            return SUBNET_INDEX[subnet]
        if subnet not in AGENT_SUBNETS[self.agent]:
            return None

        local_subnet_index = AGENT_SUBNETS[self.agent].index(subnet)
        if "_server_host_" in hostname:
            host_offset = int(hostname.rsplit("_", 1)[-1])
        elif "_user_host_" in hostname:
            host_offset = MAX_SERVERS + int(hostname.rsplit("_", 1)[-1])
        else:
            return None
        if not 0 <= host_offset < MAX_HOSTS_PER_SUBNET:
            return None
        host_row = local_subnet_index * MAX_HOSTS_PER_SUBNET + host_offset
        return NUM_SUBNET_NODES + host_row

    @staticmethod
    def _subnet_from_hostname(hostname: str) -> str | None:
        for subnet in SUBNETS:
            if hostname == f"{subnet}_router" or hostname.startswith(f"{subnet}_"):
                return subnet
        return None
