"""Generate and describe CAGE-2-style scalable benchmark scenarios."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from acd_benchmark.cyber.import_path import ensure_cyborg_on_path

ensure_cyborg_on_path()

from CybORG.Simulator.Actions import Monitor  # noqa: E402
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator  # noqa: E402

from acd_benchmark.cyber.cyborg_cc2 import get_cc2_scenario_path  # noqa: E402


SUBNETS = ("Enterprise", "Operational", "User")
ROLES = (
    "defender",
    "enterprise_host",
    "operational_host",
    "operational_server",
    "user_host",
)


@dataclass(frozen=True)
class ScalableTopology:
    """Frozen static information that is safe to expose to Blue."""

    node_count: int
    hostnames: tuple[str, ...]
    subnet_by_host: dict[str, str]
    role_by_host: dict[str, str]
    critical_hosts: tuple[str, ...]
    adjacency: np.ndarray
    edge_rule: str


class ScalableCAGE2ScenarioGenerator(FileReaderScenarioGenerator):
    """Load generated YAML while preserving official CAGE-2 controller flags."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.scenario.operational_firewall = True
        for agent_name, agent_data in self.scenario.agents.items():
            if "blue" in agent_name.lower():
                agent_data.default_actions = (Monitor, {"session": 0, "agent": agent_name})
            else:
                agent_data.internal_only = True


def allocate_subnet_counts(node_count: int) -> dict[str, int]:
    """Preserve the official 4:4:5 Enterprise/Operational/User ratio."""
    if node_count < 13:
        raise ValueError("Scalable CAGE-2 topologies require at least 13 nodes.")
    weights = np.array([4.0, 4.0, 5.0], dtype=np.float64) / 13.0
    raw_counts = weights * node_count
    counts = np.floor(raw_counts).astype(int)
    remainder = node_count - int(counts.sum())
    order = np.argsort(-(raw_counts - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return dict(zip(SUBNETS, (int(value) for value in counts)))


def canonical_hosts(node_count: int) -> dict[str, tuple[str, ...]]:
    """Return deterministic host names for every subnet and scale."""
    counts = allocate_subnet_counts(node_count)
    enterprise = ("Defender",) + tuple(
        f"Enterprise{index}" for index in range(counts["Enterprise"] - 1)
    )
    operational = ("Op_Server0",) + tuple(
        f"Op_Host{index}" for index in range(counts["Operational"] - 1)
    )
    user = tuple(f"User{index}" for index in range(counts["User"]))
    return {
        "Enterprise": enterprise,
        "Operational": operational,
        "User": user,
    }


def _int_host_entry() -> dict[str, str]:
    """Return the standard initial-knowledge declaration for one host."""
    return {
        "Interfaces": "All",
        "System info": "All",
        "User info": "All",
    }


def _clone_host(base_hosts: dict, hostname: str, subnet: str) -> dict:
    """Clone a role-compatible official host image for a new scalable host."""
    if subnet == "Enterprise":
        host = deepcopy(base_hosts["Enterprise1"])
        host["info"] = {hostname: {"Interfaces": "All"}}
    elif subnet == "Operational":
        host = deepcopy(base_hosts["Op_Host1"])
        host["info"] = {hostname: {"Interfaces": "All"}}
    elif subnet == "User":
        host = deepcopy(base_hosts["User4"])
        host["info"] = {hostname: {"Interfaces": "All"}}
    else:
        raise ValueError(f"Unsupported subnet: {subnet}")
    return host


def _velociraptor_username(hostname: str, existing_sessions: dict[str, dict]) -> str:
    """Reuse official usernames and choose role-compatible defaults for clones."""
    if hostname in existing_sessions:
        return str(existing_sessions[hostname]["username"])
    if hostname.startswith("Enterprise"):
        return "SYSTEM"
    return "ubuntu"


def build_scalable_scenario_dict(node_count: int) -> dict:
    """Expand official Scenario2 while preserving its defensive semantics."""
    with get_cc2_scenario_path().open("r", encoding="utf-8") as source:
        scenario = yaml.safe_load(source)

    hosts_by_subnet = canonical_hosts(node_count)
    ordered_hostnames = tuple(
        hostname
        for subnet in SUBNETS
        for hostname in hosts_by_subnet[subnet]
    )
    base_hosts = scenario["Hosts"]
    expanded_hosts = {}
    for subnet in SUBNETS:
        for hostname in hosts_by_subnet[subnet]:
            expanded_hosts[hostname] = (
                deepcopy(base_hosts[hostname])
                if hostname in base_hosts
                else _clone_host(base_hosts, hostname, subnet)
            )
    scenario["Hosts"] = expanded_hosts

    for subnet in SUBNETS:
        scenario["Subnets"][subnet]["Hosts"] = list(hosts_by_subnet[subnet])
        scenario["Subnets"][subnet]["Size"] = len(hosts_by_subnet[subnet])

    blue = scenario["Agents"]["Blue"]
    green = scenario["Agents"]["Green"]
    blue["INT"]["Hosts"] = {hostname: _int_host_entry() for hostname in ordered_hostnames}
    green["INT"]["Hosts"] = {hostname: _int_host_entry() for hostname in ordered_hostnames}

    existing_blue_sessions = {
        session["hostname"]: session
        for session in blue["starting_sessions"]
        if session.get("type") == "VelociraptorClient"
    }
    blue_clients = [
        {
            "hostname": hostname,
            "name": f"Velo{hostname}",
            "parent": "VeloServer",
            "type": "VelociraptorClient",
            "username": _velociraptor_username(hostname, existing_blue_sessions),
        }
        for hostname in ordered_hostnames
    ]
    blue_server = next(
        deepcopy(session)
        for session in blue["starting_sessions"]
        if session.get("type") == "VelociraptorServer"
    )
    blue_server["num_children_sessions"] = node_count
    blue["starting_sessions"] = blue_clients + [blue_server]

    green_hosts = hosts_by_subnet["User"] + tuple(
        hostname
        for hostname in hosts_by_subnet["Operational"]
        if hostname != "Op_Server0"
    )
    green["starting_sessions"] = [
        {
            "hostname": hostname,
            "name": "GreenSession",
            "type": "green_session",
            "username": "GreenAgent",
        }
        for hostname in green_hosts
    ]
    return scenario


def write_scalable_scenario(path: Path, node_count: int) -> Path:
    """Materialize a generated scenario as a reproducibility artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            build_scalable_scenario_dict(node_count),
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    return path


def build_static_topology(node_count: int) -> ScalableTopology:
    """Build the connected host-level topology known to both encoders.

    Hosts in each subnet form a directed clique. User hosts connect to the
    Enterprise0 gateway, and Enterprise0 connects to the Op_Host0 gateway.
    Every conceptual undirected link is represented in both directions.
    """
    hosts_by_subnet = canonical_hosts(node_count)
    hostnames = tuple(
        hostname
        for subnet in SUBNETS
        for hostname in hosts_by_subnet[subnet]
    )
    subnet_by_host = {
        hostname: subnet
        for subnet, hosts in hosts_by_subnet.items()
        for hostname in hosts
    }
    role_by_host = {}
    for hostname in hostnames:
        if hostname == "Defender":
            role = "defender"
        elif hostname == "Op_Server0":
            role = "operational_server"
        elif hostname.startswith("Enterprise"):
            role = "enterprise_host"
        elif hostname.startswith("Op_Host"):
            role = "operational_host"
        else:
            role = "user_host"
        role_by_host[hostname] = role

    index = {hostname: position for position, hostname in enumerate(hostnames)}
    adjacency = np.zeros((node_count, node_count), dtype=np.float32)

    def connect(left: str, right: str) -> None:
        adjacency[index[left], index[right]] = 1.0
        adjacency[index[right], index[left]] = 1.0

    for hosts in hosts_by_subnet.values():
        for left_position, left in enumerate(hosts):
            for right in hosts[left_position + 1 :]:
                connect(left, right)
    for user_host in hosts_by_subnet["User"]:
        connect(user_host, "Enterprise0")
    connect("Enterprise0", "Op_Host0")

    return ScalableTopology(
        node_count=node_count,
        hostnames=hostnames,
        subnet_by_host=subnet_by_host,
        role_by_host=role_by_host,
        critical_hosts=("Op_Server0",),
        adjacency=adjacency,
        edge_rule="subnet_cliques_plus_User-to-Enterprise0-to-Op_Host0_gateways",
    )
