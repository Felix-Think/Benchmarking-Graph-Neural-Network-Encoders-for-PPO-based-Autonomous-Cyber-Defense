"""Fixed canonical schema used by the main benchmark (schema v2, parser v4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


CANONICAL_SCHEMA = "v2"
MAX_EVENT_COUNT = 5

RELATIONAL_EVENT_NOW_FEATURES = (
    "connection_inbound_now",
    "connection_outbound_now",
    "connection_direction_ambiguous_now",
    "local_port_ssh_now",
    "local_port_web_now",
    "local_port_mail_now",
    "local_port_remote_admin_now",
    "local_port_other_service_now",
    "local_port_ephemeral_now",
    "local_port_missing_now",
    "remote_port_service_now",
    "remote_port_ephemeral_now",
    "remote_port_missing_now",
    "remote_address_known_host_now",
    "remote_address_unknown_now",
    "remote_address_missing_now",
    *(f"connection_remote_subnet_{index}_now" for index in range(NUM_SUBNET_NODES)),
)
RELATIONAL_EVENT_MEMORY_FEATURES = tuple(
    name.replace("_now", "_memory") for name in RELATIONAL_EVENT_NOW_FEATURES
)

NODE_FEATURES = (
    "exists",
    "is_subnet",
    "is_host",
    "is_user",
    "is_server",
    "controlled",
    "observed_now",
    "process_event_count_now",
    "process_has_pid_now",
    "process_has_connection_now",
    "connection_event_count_now",
    "connection_has_remote_address_now",
    "connection_has_remote_port_now",
    "connection_has_pid_now",
    *RELATIONAL_EVENT_NOW_FEATURES,
    "process_event_count_memory",
    "process_has_pid_memory",
    "process_has_connection_memory",
    "connection_event_count_memory",
    "connection_has_remote_address_memory",
    "connection_has_remote_port_memory",
    "connection_has_pid_memory",
    *RELATIONAL_EVENT_MEMORY_FEATURES,
    "passive_memory_active",
    "passive_memory_age_normalized",
    "analyse_completed_now",
    "max_file_density_now",
    "analyse_memory_active",
    "max_file_density_memory",
    "analyse_memory_age_normalized",
    "decoy_active",
)
FEATURE_INDEX = {name: index for index, name in enumerate(NODE_FEATURES)}

PASSIVE_NOW_FEATURES = (
    "process_event_count_now",
    "process_has_pid_now",
    "process_has_connection_now",
    "connection_event_count_now",
    "connection_has_remote_address_now",
    "connection_has_remote_port_now",
    "connection_has_pid_now",
    *RELATIONAL_EVENT_NOW_FEATURES,
)
PASSIVE_MEMORY_FEATURES = tuple(
    name.replace("_now", "_memory") for name in PASSIVE_NOW_FEATURES
)
CURRENT_FEATURES = (
    "observed_now",
    *PASSIVE_NOW_FEATURES,
    "analyse_completed_now",
    "max_file_density_now",
)


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
