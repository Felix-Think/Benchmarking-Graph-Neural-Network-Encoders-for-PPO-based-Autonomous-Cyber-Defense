from __future__ import annotations

import hashlib

import numpy as np

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


VALID_ENVIRONMENT_DISTRIBUTIONS = ("official", "random_host")

# These are the logical cross-zone discovery links defined by CAGE 4's
# EnterpriseScenarioGenerator._between_subnet_links. The official generator
# attaches every link to server_host_0 in each subnet.
LATERAL_MOVEMENT_EDGES = (
    ("contractor_network_subnet", "restricted_zone_a_subnet"),
    ("contractor_network_subnet", "restricted_zone_b_subnet"),
    ("contractor_network_subnet", "public_access_zone_subnet"),
    ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
    ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
    ("public_access_zone_subnet", "admin_network_subnet"),
    ("public_access_zone_subnet", "office_network_subnet"),
)


class RandomPivotEnterpriseScenarioGenerator(EnterpriseScenarioGenerator):
    """Randomize lateral-movement pivot hosts without changing CAGE 4 spaces.

    The base generator always exposes cross-subnet host information through
    ``server_host_0``. This creates a stable slot shortcut. This subclass first
    creates an otherwise standard CAGE 4 scenario, then assigns each subnet's
    cross-subnet information links to one uniformly sampled existing non-router
    host.

    A local RNG is derived from the completed scenario. It does not consume the
    simulator RNG, so the intervention does not silently alter later Red or
    Green action sampling.
    """

    def __init__(self, *args, pivot_salt: str = "cc4-random-pivot-v1", **kwargs):
        super().__init__(*args, **kwargs)
        self.pivot_salt = pivot_salt
        self.last_pivot_assignments: dict[str, str] = {}

    def create_scenario(self, np_random):
        scenario = super().create_scenario(np_random)
        pivot_rng = np.random.default_rng(self._scenario_fingerprint(scenario))
        assignments = self._sample_pivot_assignments(scenario, pivot_rng)
        self._apply_pivot_assignments(scenario, assignments)
        self.last_pivot_assignments = assignments
        return scenario

    def _scenario_fingerprint(self, scenario) -> int:
        host_records = []
        for hostname, host in sorted(scenario.hosts.items()):
            addresses = ",".join(
                sorted(str(interface.ip_address) for interface in host.interfaces)
            )
            host_records.append(f"{hostname}={addresses}")
        payload = f"{self.pivot_salt}|{'|'.join(host_records)}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def _subnets_by_name(scenario) -> dict[str, object]:
        return {subnet.name: subnet for subnet in scenario.subnets.values()}

    def _sample_pivot_assignments(self, scenario, rng) -> dict[str, str]:
        subnets = self._subnets_by_name(scenario) # 9 host có cả internet_subnet
        participating_subnets = sorted(
            set(
                subnet for edge in LATERAL_MOVEMENT_EDGES for subnet in edge
            ) # có thể dùng {}(set comprehension) để thay thế cho set 
        )
        assignments = {}
        for subnet_name in participating_subnets:
            candidates = sorted(
                hostname
                for hostname in subnets[subnet_name].hosts
                if "_router" not in hostname
            )
            if not candidates:
                raise RuntimeError(f"No pivot candidates in {subnet_name}")
            assignments[subnet_name] = str(rng.choice(candidates))
        return assignments

    @staticmethod
    def _apply_pivot_assignments(scenario, assignments: dict[str, str]) -> None:
        participating_subnets = {
            subnet for edge in LATERAL_MOVEMENT_EDGES for subnet in edge
        }
        for subnet_name in participating_subnets:
            subnet = next(
                subnet
                for subnet in scenario.subnets.values()
                if subnet.name == subnet_name
            )
            for hostname in subnet.hosts:
                scenario.hosts[hostname].info = {} # xóa infor của tất cả các host trong subnet

        pivot_links = {hostname: [] for hostname in assignments.values()}
        for source_subnet, destination_subnet in LATERAL_MOVEMENT_EDGES:
            source_host = assignments[source_subnet]
            destination_host = assignments[destination_subnet]
            pivot_links[source_host].append(destination_host)
            pivot_links[destination_host].append(source_host)

        for hostname, linked_hosts in pivot_links.items():
            scenario.hosts[hostname].info = {
                linked_host: {"Interfaces": "ip_address"}
                for linked_host in sorted(set(linked_hosts))
            }
