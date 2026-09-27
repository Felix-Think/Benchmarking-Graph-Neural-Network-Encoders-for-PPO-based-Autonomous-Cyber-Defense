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

from acd_benchmark.cyber.observation_schema import (
    AGENT_SUBNETS,
    CURRENT_FEATURES,
    FEATURE_INDEX,
    GLOBAL_FEATURES,
    MAX_EVENT_COUNT,
    MAX_HOSTS_PER_SUBNET,
    MAX_LOCAL_HOSTS,
    MAX_SERVERS,
    NODE_FEATURES,
    NUM_NODES,
    NUM_SUBNET_NODES,
    PASSIVE_MEMORY_FEATURES,
    PASSIVE_NOW_FEATURES,
    RELATIONAL_EVENT_NOW_FEATURES,
    SUBNET_INDEX,
    SUBNETS,
    VALID_VARIANTS,
    CanonicalObservation,
)
from acd_benchmark.cyber.topology import (
    desired_communication_adjacency,
    mission_phase,
    structural_subnet_adjacency,
)


@dataclass(frozen=True)
class ObservedHostEventDetails:
    process_count: int = 0
    process_has_pid: bool = False
    process_has_connection: bool = False
    connection_count: int = 0
    connection_has_remote_address: bool = False
    connection_has_remote_port: bool = False
    connection_has_pid: bool = False
    relational_values: tuple[float, ...] = (0.0,) * len(RELATIONAL_EVENT_NOW_FEATURES)

    @property
    def has_passive_event(self) -> bool:
        return self.process_count > 0 or self.connection_count > 0

    def normalized_values(self) -> np.ndarray:
        if len(self.relational_values) != len(RELATIONAL_EVENT_NOW_FEATURES):
            raise ValueError(
                "relational_values must match RELATIONAL_EVENT_NOW_FEATURES"
            )
        return np.asarray(
            (
                min(self.process_count, MAX_EVENT_COUNT) / MAX_EVENT_COUNT,
                float(self.process_has_pid),
                float(self.process_has_connection),
                min(self.connection_count, MAX_EVENT_COUNT) / MAX_EVENT_COUNT,
                float(self.connection_has_remote_address),
                float(self.connection_has_remote_port),
                float(self.connection_has_pid),
                *self.relational_values,
            ),
            dtype=np.float32,
        )


class HostCompressedBuilder:
    """Build typed observable alerts and bounded memories for one Blue agent."""

    def __init__(
        self,
        agent: str,
        episode_steps: int,
        variant: str = "action_aware",
        passive_memory_ttl: int = 20,
        analysis_memory_ttl: int = 20,
    ):
        if agent not in AGENT_SUBNETS:
            raise ValueError(f"Unknown Blue agent: {agent}")
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}")
        if passive_memory_ttl <= 0 or analysis_memory_ttl <= 0:
            raise ValueError("V2 memory TTL values must be positive")
        self.agent = agent
        self.episode_steps = episode_steps
        self.variant = variant
        self.passive_memory_ttl = passive_memory_ttl
        self.analysis_memory_ttl = analysis_memory_ttl
        self.node_features = np.zeros((NUM_NODES, len(NODE_FEATURES)), dtype=np.float32)
        self.host_membership = np.zeros(
            (MAX_LOCAL_HOSTS, NUM_SUBNET_NODES), dtype=np.float32
        )
        self.firewall_blocked_edges: set[tuple[str, str]] = set()
        self._passive_memory = np.zeros(
            (NUM_NODES, len(PASSIVE_NOW_FEATURES)), dtype=np.float32
        )
        self._passive_age = np.zeros(NUM_NODES, dtype=np.int32)
        self._passive_active = np.zeros(NUM_NODES, dtype=np.bool_)
        self._analysis_density = np.zeros(NUM_NODES, dtype=np.float32)
        self._analysis_age = np.zeros(NUM_NODES, dtype=np.int32)
        self._analysis_active = np.zeros(NUM_NODES, dtype=np.bool_)

    def reset(
        self,
        observation: dict,
        event_details: dict[str, ObservedHostEventDetails] | None = None,
    ) -> CanonicalObservation:
        self.node_features.fill(0.0)
        self.host_membership.fill(0.0)
        self.firewall_blocked_edges.clear()
        self._passive_memory.fill(0.0)
        self._passive_age.fill(0)
        self._passive_active.fill(False)
        self._analysis_density.fill(0.0)
        self._analysis_age.fill(0)
        self._analysis_active.fill(False)
        event_details = {} if event_details is None else event_details

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

        observed_hosts = {
            hostname
            for hostname in observation
            if hostname not in ("success", "message", "action")
        }
        observed_hosts.update(
            hostname
            for hostname, details in event_details.items()
            if details.has_passive_event
        )
        for hostname in observed_hosts:
            node_index = self._node_index(hostname)
            if node_index is None:
                continue
            self._mark_host_exists(hostname, node_index)
            self._update_node(
                node_index,
                observation.get(hostname, {}),
                event_details.get(hostname, ObservedHostEventDetails()),
            )

        return self._snapshot(observation, step=0)

    def update(
        self,
        observation: dict,
        step: int,
        event_details: dict[str, ObservedHostEventDetails] | None = None,
    ) -> CanonicalObservation:
        event_details = {} if event_details is None else event_details
        self.node_features[:, [FEATURE_INDEX[name] for name in CURRENT_FEATURES]] = 0.0
        if self.variant != "instant":
            self._advance_memories()

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
            for hostname, details in event_details.items()
            if details.has_passive_event
        )
        for hostname in observed_hosts:
            node_index = self._node_index(hostname)
            if node_index is None:
                continue
            self._mark_host_exists(hostname, node_index)
            self._update_node(
                node_index,
                observation.get(hostname, {}),
                event_details.get(hostname, ObservedHostEventDetails()),
            )

        return self._snapshot(observation, step)

    def _mark_host_exists(self, hostname: str, node_index: int) -> None:
        self.node_features[node_index, FEATURE_INDEX["exists"]] = 1.0
        self.node_features[node_index, FEATURE_INDEX["observed_now"]] = 1.0
        if node_index < NUM_SUBNET_NODES:
            return
        host_row = node_index - NUM_SUBNET_NODES
        subnet = self._subnet_from_hostname(hostname)
        self.host_membership[host_row, SUBNET_INDEX[subnet]] = 1.0

    def _update_node(
        self,
        node_index: int,
        host_data: dict,
        event_details: ObservedHostEventDetails,
    ) -> None:
        passive_values = event_details.normalized_values()
        for feature, value in zip(PASSIVE_NOW_FEATURES, passive_values, strict=True):
            self.node_features[node_index, FEATURE_INDEX[feature]] = value

        files = host_data.get("Files", [])
        densities = [
            float(file_data["Density"])
            for file_data in files
            if file_data.get("Density") is not None
        ]
        density = float(np.clip(max(densities, default=0.0), 0.0, 1.0))
        self.node_features[node_index, FEATURE_INDEX["max_file_density_now"]] = density

        if self.variant == "instant":
            return
        if event_details.has_passive_event:
            self._passive_memory[node_index] = passive_values
            self._passive_active[node_index] = True
            self._passive_age[node_index] = 0

        if self.node_features[node_index, FEATURE_INDEX["analyse_completed_now"]] > 0:
            self._analysis_active[node_index] = True
            self._analysis_age[node_index] = 0
            self._analysis_density[node_index] = density

    def _advance_memories(self) -> None:
        self._passive_age[self._passive_active] += 1
        passive_expired = self._passive_active & (
            self._passive_age >= self.passive_memory_ttl
        )
        self._passive_active[passive_expired] = False
        self._passive_age[passive_expired] = 0
        self._passive_memory[passive_expired] = 0.0

        self._analysis_age[self._analysis_active] += 1
        analysis_expired = self._analysis_active & (
            self._analysis_age >= self.analysis_memory_ttl
        )
        self._analysis_active[analysis_expired] = False
        self._analysis_age[analysis_expired] = 0
        self._analysis_density[analysis_expired] = 0.0

    def _apply_successful_action(self, action) -> None:
        if isinstance(action, BlockTrafficZone):
            self.firewall_blocked_edges.add((action.from_subnet, action.to_subnet))
        elif isinstance(action, AllowTrafficZone):
            self.firewall_blocked_edges.discard((action.from_subnet, action.to_subnet))

        target = getattr(action, "hostname", None)
        node_index = self._node_index(target) if target is not None else None
        if isinstance(action, Analyse) and node_index is not None:
            self.node_features[node_index, FEATURE_INDEX["observed_now"]] = 1.0
            self.node_features[node_index, FEATURE_INDEX["analyse_completed_now"]] = 1.0
            if self.variant != "instant":
                self._analysis_active[node_index] = True
                self._analysis_age[node_index] = 0

        if self.variant != "action_aware" or node_index is None:
            return
        if isinstance(action, Restore):
            self._clear_host_memory(node_index)
            self.node_features[node_index, FEATURE_INDEX["decoy_active"]] = 0.0
        elif isinstance(action, DeployDecoy):
            self.node_features[node_index, FEATURE_INDEX["decoy_active"]] = 1.0

    def _clear_host_memory(self, node_index: int) -> None:
        self._passive_memory[node_index] = 0.0
        self._passive_age[node_index] = 0
        self._passive_active[node_index] = False
        self._analysis_density[node_index] = 0.0
        self._analysis_age[node_index] = 0
        self._analysis_active[node_index] = False

    def _snapshot(self, observation: dict, step: int) -> CanonicalObservation:
        memory_indices = [FEATURE_INDEX[name] for name in PASSIVE_MEMORY_FEATURES]
        self.node_features[:, memory_indices] = 0.0
        self.node_features[:, FEATURE_INDEX["passive_memory_active"]] = (
            self._passive_active.astype(np.float32)
        )
        self.node_features[:, FEATURE_INDEX["passive_memory_age_normalized"]] = 0.0
        if self.variant != "instant":
            passive_decay = np.zeros(NUM_NODES, dtype=np.float32)
            passive_decay[self._passive_active] = 1.0 - (
                self._passive_age[self._passive_active] / float(self.passive_memory_ttl)
            )
            self.node_features[:, memory_indices] = (
                self._passive_memory * passive_decay[:, None]
            )
            self.node_features[
                self._passive_active,
                FEATURE_INDEX["passive_memory_age_normalized"],
            ] = self._passive_age[self._passive_active] / float(self.passive_memory_ttl)

        self.node_features[:, FEATURE_INDEX["analyse_memory_active"]] = (
            self._analysis_active.astype(np.float32)
        )
        self.node_features[:, FEATURE_INDEX["max_file_density_memory"]] = 0.0
        self.node_features[:, FEATURE_INDEX["analyse_memory_age_normalized"]] = 0.0
        if self.variant != "instant":
            analysis_decay = np.zeros(NUM_NODES, dtype=np.float32)
            analysis_decay[self._analysis_active] = 1.0 - (
                self._analysis_age[self._analysis_active]
                / float(self.analysis_memory_ttl)
            )
            self.node_features[:, FEATURE_INDEX["max_file_density_memory"]] = (
                self._analysis_density * analysis_decay
            )
            self.node_features[
                self._analysis_active,
                FEATURE_INDEX["analyse_memory_age_normalized"],
            ] = self._analysis_age[self._analysis_active] / float(
                self.analysis_memory_ttl
            )

        phase = mission_phase(step, self.episode_steps)
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
            structural_adjacency=structural_subnet_adjacency(),
            communication_policy=desired_communication_adjacency(phase),
            firewall_blocks=firewall_blocks,
            global_features=global_features,
        )

    def _node_index(self, hostname: str | None) -> int | None: # Builder for each agent has shape [57, 79]
        if not hostname:
            return None
        subnet = self._subnet_from_hostname(hostname)
        if subnet is None:
            return None
        if hostname.endswith("_router"):
            return SUBNET_INDEX[subnet]
        if subnet not in AGENT_SUBNETS[self.agent]:
            return None

        local_subnet_index = AGENT_SUBNETS[self.agent].index(subnet)  #[0,1,2]
        if "_server_host_" in hostname: #[0...5]
            host_offset = int(hostname.rsplit("_", 1)[-1]) 
        elif "_user_host_" in hostname:
            host_offset = MAX_SERVERS + int(hostname.rsplit("_", 1)[-1]) # [6...15]
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
