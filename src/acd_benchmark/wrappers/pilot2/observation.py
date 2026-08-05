"""Exact fixed-budget host-retention masking for Pilot v2."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np

from acd_benchmark.cyber.scalable_scenario import SUBNETS, ScalableTopology
from acd_benchmark.wrappers.shared.host_obs import GraphObservation
from acd_benchmark.wrappers.shared.partial_obs import apply_host_visibility_mask
from acd_benchmark.wrappers.stage1.observation import build_stage1_base_observation


@dataclass(frozen=True)
class FixedBudgetMask:
    """One reproducible host-retention realization and its audit metadata."""

    visibility: np.ndarray
    intended_visible_count: int
    visible_count_by_subnet: dict[str, int]


def round_half_up(value: float) -> int:
    """Round a non-negative quota without Python's ties-to-even behavior."""
    if value < 0:
        raise ValueError("round_half_up only accepts non-negative values.")
    return int(floor(value + 0.5))


def allocate_subnet_quotas(
    topology: ScalableTopology,
    retention_q: float,
    tie_break_order: tuple[str, ...],
) -> dict[str, int]:
    """Allocate an exact global host budget using largest remainders."""
    if not 0.0 <= retention_q <= 1.0:
        raise ValueError("retention_q must be in [0, 1].")
    if set(tie_break_order) != set(SUBNETS):
        raise ValueError("tie_break_order must contain every benchmark subnet once.")

    counts = {
        subnet: sum(
            topology.subnet_by_host[hostname] == subnet
            for hostname in topology.hostnames
        )
        for subnet in SUBNETS
    }
    target = round_half_up(retention_q * topology.node_count)
    raw = {subnet: retention_q * count for subnet, count in counts.items()}
    quotas = {subnet: int(floor(value)) for subnet, value in raw.items()}
    remainder = target - sum(quotas.values())
    tie_rank = {subnet: index for index, subnet in enumerate(tie_break_order)}
    ranked = sorted(
        SUBNETS,
        key=lambda subnet: (-(raw[subnet] - quotas[subnet]), tie_rank[subnet]),
    )
    for subnet in ranked[:remainder]:
        quotas[subnet] += 1

    if sum(quotas.values()) != target:
        raise RuntimeError("Subnet quotas do not match the global retention budget.")
    if any(quotas[subnet] > counts[subnet] for subnet in SUBNETS):
        raise RuntimeError("Subnet quota exceeds the number of available host slots.")
    return quotas


def sample_fixed_budget_visibility(
    topology: ScalableTopology,
    retention_q: float,
    rng: np.random.Generator,
    tie_break_order: tuple[str, ...],
) -> FixedBudgetMask:
    """Sample exactly K=round(q*N) host slots, stratified by subnet."""
    quotas = allocate_subnet_quotas(topology, retention_q, tie_break_order)
    visibility = np.zeros(topology.node_count, dtype=bool)
    for subnet in SUBNETS:
        candidates = np.asarray(
            [
                index
                for index, hostname in enumerate(topology.hostnames)
                if topology.subnet_by_host[hostname] == subnet
            ],
            dtype=np.int64,
        )
        quota = quotas[subnet]
        if quota:
            selected = rng.choice(candidates, size=quota, replace=False)
            visibility[selected] = True
    return FixedBudgetMask(
        visibility=visibility,
        intended_visible_count=round_half_up(retention_q * topology.node_count),
        visible_count_by_subnet=quotas,
    )


def build_pilot2_observation(
    raw_graph_observation: GraphObservation,
    topology: ScalableTopology,
    retention_q: float,
    rng: np.random.Generator,
    tie_break_order: tuple[str, ...],
) -> tuple[GraphObservation, FixedBudgetMask]:
    """Apply Pilot v2 masking after native Blue observation encoding."""
    base = build_stage1_base_observation(raw_graph_observation, topology)
    mask = sample_fixed_budget_visibility(
        topology=topology,
        retention_q=retention_q,
        rng=rng,
        tie_break_order=tie_break_order,
    )
    return (
        apply_host_visibility_mask(base, visibility=mask.visibility, fill_value=0.0),
        mask,
    )
