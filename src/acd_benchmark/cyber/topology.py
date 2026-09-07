"""Public CAGE 4 physical topology and mission communication schedule."""

import numpy as np

from acd_benchmark.cyber.observation_schema import NUM_SUBNET_NODES, SUBNET_INDEX


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
