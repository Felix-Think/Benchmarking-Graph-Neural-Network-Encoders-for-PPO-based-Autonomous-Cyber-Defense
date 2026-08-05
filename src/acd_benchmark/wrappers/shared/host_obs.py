"""Raw Blue observation encoding shared by flat and graph policies.

This module keeps CybORG's raw Blue observation as the single data source,
then builds two matched views from it:

* a host-level graph view for GNN encoders;
* a flattened view for MLP/PPO baselines.

The central design rule is feature parity: graph and flat baselines must see
the same host features, with topology exposed to flat via the adjacency matrix.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from ipaddress import IPv4Address, IPv4Network
from typing import Any, Iterable

import numpy as np


UNKNOWN = "__unknown__"
ACTIVITY_NONE = "None"
ACTIVITY_SCAN = "Scan"
ACTIVITY_EXPLOIT = "Exploit"
COMPROMISED_NO = "No"
COMPROMISED_UNKNOWN = "Unknown"
COMPROMISED_USER = "User"
COMPROMISED_PRIVILEGED = "Privileged"
SEMANTIC_FEATURE_NAMES = (
    "activity.is_scan",
    "activity.is_exploit",
    "compromised.is_unknown",
    "compromised.is_user",
    "compromised.is_privileged",
)


@dataclass(frozen=True)
class HostFeatureSchema:
    """Fixed feature contract learned from the first Blue observation.

    The schema stores the stable host order, feature order, categorical
    vocabularies, and dynamic feature indices. It is intentionally separated
    from step observations so that every timestep is encoded with the same
    dimensions and meanings.
    """

    hostnames: tuple[str, ...]
    feature_names: tuple[str, ...]
    dynamic_feature_indices: tuple[int, ...]
    subnets: tuple[str, ...]
    category_maps: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class GraphObservation:
    """Encoded host-level observation for graph and flat consumers.

    ``node_features`` and ``edge_index`` are the natural GNN inputs.
    ``adjacency`` is retained for debugging and for the flat baseline so that
    topology is not exclusive to the graph encoder.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    adjacency: np.ndarray
    hostnames: tuple[str, ...]
    feature_names: tuple[str, ...]
    dynamic_feature_indices: tuple[int, ...]


def normalize_value(value: Any) -> Any:
    """Convert CybORG enum/IP/type objects into stable scalar values.

    Raw CybORG observations may contain Enum values and ipaddress objects.
    Feature schemas and JSON summaries need stable string-like values instead.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (IPv4Address, IPv4Network)):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    return value


def normalize_string(value: Any) -> str:
    """Normalize a potentially missing raw value into a category string."""
    if value is None:
        return UNKNOWN
    return str(normalize_value(value))


def split_blue_observation(
    observation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Split top-level metadata such as ``success`` from host observations.

    In CC2 raw Blue observations, host entries are top-level dictionaries while
    metadata such as the previous action success flag is not a dictionary. The
    host encoder only consumes the host dictionary portion.
    """
    metadata: dict[str, Any] = {}
    hosts: dict[str, dict[str, Any]] = {}
    for key, value in observation.items():
        if isinstance(value, dict):
            hosts[str(key)] = value
        else:
            metadata[str(key)] = value
    return metadata, hosts


def _first_dict(items: Any) -> dict[str, Any]:
    """Return the first dictionary from a CybORG list field.

    Fields such as ``Interface`` are represented as ``list[dict]`` even when a
    host has only one observed interface. Returning an empty dict keeps callers
    safe when the field is absent, empty, or malformed.
    """
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0]
    return {}


def _as_list(value: Any) -> list[Any]:
    """Return ``value`` only when it is a list, otherwise return an empty list."""
    return value if isinstance(value, list) else []


def _is_privileged_username(username: Any) -> bool:
    """Detect common high-privilege OS accounts from observed usernames."""
    normalized = normalize_string(username).lower()
    return normalized in {"root", "administrator", "system"}


def _has_default_password(user_info: Iterable[dict[str, Any]]) -> bool:
    """Detect known default username/password pairs in observed user records.

    This is a small CC2-oriented heuristic, not a complete credential audit.
    It creates a stable host-level risk flag from raw ``User Info`` entries.
    """
    default_pairs = {
        ("pi", "raspberry"),
        ("vagrant", "vagrant"),
    }
    for user in user_info:
        username = normalize_string(user.get("Username")).lower()
        password = normalize_string(user.get("Password")).lower()
        if (username, password) in default_pairs:
            return True
    return False


def _process_identity(process: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Build a stable, observation-only process identity for change tracking.

    Process dictionaries can contain nested, changing fields and cannot be put
    directly into a set. This function keeps only observed identifying fields
    and selected connection fields, then converts them into a sorted tuple so
    current processes can be compared against the reset-time baseline.
    """
    identity_parts = [
        (str(key), normalize_string(value))
        for key, value in process.items()
        if key in {"PID", "Username", "Process Name", "Process Type", "Path"}
    ]
    for connection_index, connection in enumerate(_as_list(process.get("Connections"))):
        if not isinstance(connection, dict):
            continue
        for key in ("local_port", "remote_port", "local_address", "remote_address"):
            if key in connection:
                identity_parts.append(
                    (
                        f"Connections[{connection_index}].{key}",
                        normalize_string(connection.get(key)),
                    )
                )
    return tuple(sorted(identity_parts))


def _port_value(value: Any) -> int | None:
    """Convert an observed port value into an int when possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _activity_from_process_anomalies(processes: Iterable[dict[str, Any]]) -> str:
    """Classify anomalous process connection patterns as Scan or Exploit.

    This mirrors the observation-only heuristic used by CybORG's
    ``BlueTableWrapper``: many local ports suggest scanning, reverse-shell-like
    traffic on remote port 4444 or many connections to one port suggests an
    exploit. The raw process dictionaries remain the only data source.
    """
    process_list = list(processes)
    num_connections = len(process_list)
    local_ports: set[int] = set()
    remote_ports: set[int] = set()
    for process in process_list:
        connections = _as_list(process.get("Connections"))
        if not connections or not isinstance(connections[0], dict):
            continue
        first_connection = connections[0]
        if (local_port := _port_value(first_connection.get("local_port"))) is not None:
            local_ports.add(local_port)
        if (remote_port := _port_value(first_connection.get("remote_port"))) is not None:
            remote_ports.add(remote_port)

    if num_connections >= 3 and len(local_ports) >= 3:
        return ACTIVITY_SCAN
    if 4444 in remote_ports:
        return ACTIVITY_EXPLOIT
    if num_connections >= 3 and len(local_ports) == 1:
        return ACTIVITY_EXPLOIT
    return ACTIVITY_SCAN


def _has_malware_file(files: Iterable[dict[str, Any]]) -> bool:
    """Detect file anomalies that CybORG marks as high-density malware."""
    for file_info in files:
        try:
            if float(file_info.get("Density", 0.0)) >= 0.9:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _host_anomalies_from_baseline(
    host_update: dict[str, Any],
    host_baseline: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Return Files/Processes in a sparse update that were absent at reset."""
    anomalies: dict[str, list[dict[str, Any]]] = {}
    if "Files" in host_update:
        baseline_files = _as_list(host_baseline.get("Files"))
        anomalous_files = [
            file_info
            for file_info in _as_list(host_update.get("Files"))
            if isinstance(file_info, dict) and file_info not in baseline_files
        ]
        if anomalous_files:
            anomalies["Files"] = anomalous_files

    if "Processes" in host_update:
        baseline_processes = _as_list(host_baseline.get("Processes"))
        anomalous_processes = [
            process
            for process in _as_list(host_update.get("Processes"))
            if isinstance(process, dict) and process not in baseline_processes
        ]
        if anomalous_processes:
            anomalies["Processes"] = anomalous_processes
    return anomalies


def _merge_host_observation(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Merge a sparse host observation into the latest-known host cache.

    CybORG step observations can omit fields that were known at reset. Dict
    fields are merged recursively so partial updates do not erase old keys.
    ``Interface`` is handled specially because it is ``list[dict]`` and its
    subnet must be preserved for graph topology when later step observations
    only report an IP address.

    Dynamic list fields such as ``Sessions`` and ``Processes`` are currently
    replaced rather than merged to avoid accumulating stale activity.
    """
    merged = deepcopy(existing)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_host_observation(merged[key], value)
        elif key == "Interface" and isinstance(value, list) and isinstance(merged.get(key), list):
            merged_interfaces = deepcopy(merged[key])
            for index, interface_update in enumerate(value):
                if index >= len(merged_interfaces) or not isinstance(interface_update, dict):
                    continue
                if isinstance(merged_interfaces[index], dict):
                    merged_interfaces[index] = _merge_host_observation(
                        merged_interfaces[index],
                        interface_update,
                    )
            merged[key] = merged_interfaces
        else:
            merged[key] = deepcopy(value)
    return merged


def _collect_categories(
    hosts: dict[str, dict[str, Any]],
    system_fields: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Collect categorical vocabularies used for one-hot host features.

    Only selected static categorical fields are collected here: chosen
    ``System info`` fields and the first observed interface subnet. Dynamic
    list fields such as sessions, processes, and user info are summarized later
    with counts/flags instead of one-hotting whole dictionaries.

    ``UNKNOWN`` is always appended as a final category so missing or unseen
    values still map to a fixed feature position.
    """
    category_maps: dict[str, tuple[str, ...]] = {}

    for field in system_fields:
        values = {
            normalize_string(host.get("System info", {}).get(field))
            for host in hosts.values()
            if isinstance(host.get("System info"), dict)
        }
        values.discard(UNKNOWN)
        category_maps[f"system.{field}"] = tuple(sorted(values) + [UNKNOWN])

    subnet_values = {
        normalize_string(_first_dict(host.get("Interface")).get("Subnet"))
        for host in hosts.values()
    }
    subnet_values.discard(UNKNOWN)
    category_maps["interface.Subnet"] = tuple(sorted(subnet_values) + [UNKNOWN])
    return category_maps


def _schema_feature_names(category_maps: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Materialize the exact host feature order used by every timestep.

    These names are the contract between the encoder, the graph wrapper, the
    flat wrapper, tests, and debugging output. The numeric vectors produced by
    ``_encode_host`` must follow this order exactly.
    """
    names: list[str] = ["known"]
    for field in ("Architecture", "OSType", "OSDistribution", "OSVersion"):
        names.extend(f"system.{field}={category}" for category in category_maps[f"system.{field}"])
    names.append("interface.has_interface")
    names.extend(f"interface.Subnet={category}" for category in category_maps["interface.Subnet"])
    names.extend(
        [
            "sessions.count_norm",
            "sessions.blue_count_norm",
            "sessions.red_count_norm",
            "sessions.has_velociraptor_client",
            "sessions.has_privileged_user",
            "processes.count_norm",
            "processes.unique_user_count_norm",
            "processes.has_privileged_user",
            "processes.new_count_norm",
            "activity.is_scan",
            "activity.is_exploit",
            "compromised.is_unknown",
            "compromised.is_user",
            "compromised.is_privileged",
            "user_info.count_norm",
            "user_info.has_default_password",
            "user_info.has_privileged_user",
            "user_info.has_green_agent",
            "user_info.has_password_exposed",
        ]
    )
    return tuple(names)


def _one_hot(value: Any, categories: tuple[str, ...]) -> list[float]:
    """Encode one categorical value using a fixed category tuple."""
    normalized = normalize_string(value)
    if normalized not in categories:
        normalized = UNKNOWN
    return [1.0 if category == normalized else 0.0 for category in categories]


def _count_norm(count: int, cap: int = 10) -> float:
    """Normalize a count into ``[0, 1]`` with a simple saturation cap."""
    return min(float(count) / float(cap), 1.0)


class RawBlueHostEncoder:
    """Stateful host-level encoder for sparse raw Blue observations.

    The encoder initializes a fixed schema from the first Blue observation.
    Later sparse observations update the latest-known host cache instead of
    deleting hosts that are absent from a step observation.
    """

    def __init__(self, schema: HostFeatureSchema, initial_hosts: dict[str, dict[str, Any]]):
        """Store the fixed schema, host cache, baseline, and security state."""
        self.schema = schema
        self.latest_hosts = deepcopy(initial_hosts)
        self.baseline_hosts = deepcopy(initial_hosts)
        self.baseline_processes = {
            hostname: {
                _process_identity(process)
                for process in _as_list(host.get("Processes"))
                if isinstance(process, dict)
            }
            for hostname, host in initial_hosts.items()
        }
        self.activity_by_host = {
            hostname: ACTIVITY_NONE
            for hostname in schema.hostnames
        }
        self.compromised_by_host = {
            hostname: COMPROMISED_NO
            for hostname in schema.hostnames
        }

    def apply_blue_action(self, action_name: str, hostname: str | None) -> None:
        """Update persistent compromise state from Blue's last host action.

        This mirrors BlueTableWrapper's state transition without importing the
        wrapper itself: Restore clears a host, while Remove turns a previously
        suspected host into Unknown because Blue no longer has direct evidence
        that the host is still compromised or clean.
        """
        if hostname not in self.compromised_by_host:
            return
        if action_name == "Restore":
            self.compromised_by_host[hostname] = COMPROMISED_NO
        elif action_name == "Remove" and self.compromised_by_host[hostname] != COMPROMISED_NO:
            self.compromised_by_host[hostname] = COMPROMISED_UNKNOWN

    @classmethod
    def from_initial_observation(cls, observation: dict[str, Any]) -> "RawBlueHostEncoder":
        """Create an encoder and fixed schema from the first raw Blue observation.

        Reset observations define the host order, categorical vocabularies, and
        dynamic feature indices. Later step observations update values but do
        not change the schema shape.
        """
        _, hosts = split_blue_observation(observation)
        hostnames = tuple(hosts.keys())
        system_fields = ("Architecture", "OSType", "OSDistribution", "OSVersion")
        category_maps = _collect_categories(hosts, system_fields)
        feature_names = _schema_feature_names(category_maps)
        dynamic_indices = tuple(
            index
            for index, name in enumerate(feature_names)
            if name.startswith(("sessions.", "processes.", "activity.", "compromised.", "user_info."))
        )
        schema = HostFeatureSchema(
            hostnames=hostnames,
            feature_names=feature_names,
            dynamic_feature_indices=dynamic_indices,
            subnets=category_maps["interface.Subnet"],
            category_maps=category_maps,
        )
        return cls(schema=schema, initial_hosts=hosts)

    def update(self, observation: dict[str, Any]) -> GraphObservation:
        """Update the host cache with a sparse observation and encode it.

        Unknown hostnames are ignored to keep graph and flat tensor shapes fixed
        for a fixed-shape benchmark observation.
        """
        _, hosts = split_blue_observation(observation)
        self._update_security_state(hosts)
        for hostname, host_obs in hosts.items():
            if hostname in self.latest_hosts:
                self.latest_hosts[hostname] = _merge_host_observation(
                    self.latest_hosts[hostname],
                    host_obs,
                )
        return self.encode_current()

    def _update_security_state(self, hosts: dict[str, dict[str, Any]]) -> None:
        """Update per-host activity and compromise states from raw updates.

        Activity is a current-step signal and resets to None every update.
        Compromise is persistent because CybORG's sparse observations may stop
        repeating the original exploit or malware evidence on later steps.
        """
        for hostname in self.schema.hostnames:
            self.activity_by_host[hostname] = ACTIVITY_NONE

        for hostname, host_obs in hosts.items():
            if hostname not in self.baseline_hosts:
                continue
            anomalies = _host_anomalies_from_baseline(
                host_update=host_obs,
                host_baseline=self.baseline_hosts[hostname],
            )
            if "Processes" in anomalies:
                activity = _activity_from_process_anomalies(anomalies["Processes"])
                self.activity_by_host[hostname] = activity
                if activity == ACTIVITY_EXPLOIT:
                    self.compromised_by_host[hostname] = COMPROMISED_USER
            if "Files" in anomalies and _has_malware_file(anomalies["Files"]):
                self.compromised_by_host[hostname] = COMPROMISED_PRIVILEGED

    def encode_current(self) -> GraphObservation:
        """Encode the latest-known host cache into graph tensors."""
        node_features = np.array(
            [self._encode_host(hostname, self.latest_hosts.get(hostname, {})) for hostname in self.schema.hostnames],
            dtype=np.float32,
        )
        adjacency = self._build_adjacency()
        edge_index = _adjacency_to_edge_index(adjacency)
        return GraphObservation(
            node_features=node_features,
            edge_index=edge_index,
            adjacency=adjacency,
            hostnames=self.schema.hostnames,
            feature_names=self.schema.feature_names,
            dynamic_feature_indices=self.schema.dynamic_feature_indices,
        )

    def _encode_host(self, hostname: str, host: dict[str, Any]) -> list[float]:
        """Encode one raw host dictionary into the fixed host feature vector.

        Static categorical fields become one-hot blocks. Dynamic raw list
        fields are reduced into host-level counts and boolean flags so every
        host has the same feature dimension.
        The first feature is ``known``: whether this host has cached data, not
        whether it appeared in the latest sparse step observation.
        """
        system_info = host.get("System info", {}) if isinstance(host.get("System info"), dict) else {}
        interface = _first_dict(host.get("Interface"))
        sessions = [item for item in _as_list(host.get("Sessions")) if isinstance(item, dict)]
        processes = [item for item in _as_list(host.get("Processes")) if isinstance(item, dict)]
        user_info = [item for item in _as_list(host.get("User Info")) if isinstance(item, dict)]

        features: list[float] = [1.0 if host else 0.0]
        for field in ("Architecture", "OSType", "OSDistribution", "OSVersion"):
            categories = self.schema.category_maps[f"system.{field}"]
            features.extend(_one_hot(system_info.get(field), categories))

        features.append(1.0 if interface else 0.0)
        features.extend(_one_hot(interface.get("Subnet"), self.schema.category_maps["interface.Subnet"]))

        blue_sessions = [s for s in sessions if normalize_string(s.get("Agent")) == "Blue"]
        red_sessions = [s for s in sessions if normalize_string(s.get("Agent")) == "Red"]
        features.extend(
            [
                _count_norm(len(sessions)),
                _count_norm(len(blue_sessions)),
                _count_norm(len(red_sessions)),
                float(any(normalize_string(s.get("Type")) == "VELOCIRAPTOR_CLIENT" for s in sessions)),
                float(any(_is_privileged_username(s.get("Username")) for s in sessions)),
            ]
        )

        process_users = {normalize_string(p.get("Username")) for p in processes}
        baseline = self.baseline_processes.get(hostname, set())
        new_processes = [
            process
            for process in processes
            if _process_identity(process) not in baseline
        ]
        features.extend(
            [
                _count_norm(len(processes)),
                _count_norm(len(process_users)),
                float(any(_is_privileged_username(p.get("Username")) for p in processes)),
                _count_norm(len(new_processes)),
            ]
        )

        activity = self.activity_by_host.get(hostname, ACTIVITY_NONE)
        compromised = self.compromised_by_host.get(hostname, COMPROMISED_NO)
        features.extend(
            [
                float(activity == ACTIVITY_SCAN),
                float(activity == ACTIVITY_EXPLOIT),
                float(compromised == COMPROMISED_UNKNOWN),
                float(compromised == COMPROMISED_USER),
                float(compromised == COMPROMISED_PRIVILEGED),
            ]
        )

        features.extend(
            [
                _count_norm(len(user_info)),
                float(_has_default_password(user_info)),
                float(any(_is_privileged_username(user.get("Username")) for user in user_info)),
                float(any(normalize_string(user.get("Username")) == "GreenAgent" for user in user_info)),
                float(any("Password" in user or "Password Hash" in user for user in user_info)),
            ]
        )
        return features

    def _build_adjacency(self) -> np.ndarray:
        """Build directed same-subnet host edges from latest known interfaces.

        The graph currently connects every ordered pair of distinct hosts that
        share a known subnet. This is a local topology rule, not a full router
        or inter-subnet network model.
        """
        host_subnets = [
            normalize_string(_first_dict(self.latest_hosts.get(hostname, {}).get("Interface")).get("Subnet"))
            for hostname in self.schema.hostnames
        ]
        size = len(self.schema.hostnames)
        adjacency = np.zeros((size, size), dtype=np.float32)
        for src in range(size):
            for dst in range(size):
                if src == dst:
                    continue
                if host_subnets[src] != UNKNOWN and host_subnets[src] == host_subnets[dst]:
                    adjacency[src, dst] = 1.0
        return adjacency


def _adjacency_to_edge_index(adjacency: np.ndarray) -> np.ndarray:
    """Convert a dense adjacency matrix into a ``[2, num_edges]`` edge index."""
    src, dst = np.nonzero(adjacency)
    if len(src) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.vstack([src, dst]).astype(np.int64)


def select_node_features(
    graph_observation: GraphObservation,
    selected_feature_names: Iterable[str] | None = None,
) -> np.ndarray:
    """Return all node features or a named subset in schema order."""
    if selected_feature_names is None:
        return graph_observation.node_features
    name_to_index = {
        name: index
        for index, name in enumerate(graph_observation.feature_names)
    }
    indices = [name_to_index[name] for name in selected_feature_names]
    return graph_observation.node_features[:, indices]


def select_graph_observation_features(
    graph_observation: GraphObservation,
    selected_feature_names: Iterable[str] | None = None,
) -> GraphObservation:
    """Return a graph observation with all features or a named feature subset."""
    if selected_feature_names is None:
        return graph_observation
    selected_names = tuple(selected_feature_names)
    selected_features = select_node_features(
        graph_observation=graph_observation,
        selected_feature_names=selected_names,
    )
    old_index_by_name = {
        name: index
        for index, name in enumerate(graph_observation.feature_names)
    }
    old_dynamic_indices = set(graph_observation.dynamic_feature_indices)
    selected_dynamic_indices = tuple(
        new_index
        for new_index, name in enumerate(selected_names)
        if old_index_by_name[name] in old_dynamic_indices
    )
    return GraphObservation(
        node_features=selected_features,
        edge_index=graph_observation.edge_index,
        adjacency=graph_observation.adjacency,
        hostnames=graph_observation.hostnames,
        feature_names=selected_names,
        dynamic_feature_indices=selected_dynamic_indices,
    )


def flatten_graph_observation(
    graph_observation: GraphObservation,
    include_adjacency: bool = True,
    selected_feature_names: Iterable[str] | None = None,
) -> np.ndarray:
    """Flatten graph data into the matched MLP/PPO baseline observation.

    By default the flat vector uses all node features. Passing
    ``selected_feature_names`` creates a compact view, useful for isolating the
    effect of semantic raw-derived security state from high-dimensional static
    host features. Including adjacency keeps topology available to the flat
    baseline as well.
    """
    node_features = select_node_features(
        graph_observation=graph_observation,
        selected_feature_names=selected_feature_names,
    )
    parts = [node_features.reshape(-1)]
    if include_adjacency:
        parts.append(graph_observation.adjacency.reshape(-1))
    return np.concatenate(parts).astype(np.float32)
