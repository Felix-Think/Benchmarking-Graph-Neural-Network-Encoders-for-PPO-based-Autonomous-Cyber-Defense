"""Construct Pilot v2 environments while reusing the shared policy builders."""

from __future__ import annotations

from pathlib import Path

from acd_benchmark.wrappers.pilot2 import Pilot2FlatEnv, Pilot2GraphEnv


def make_pilot2_env(
    model: str,
    *,
    scenario_path: Path,
    node_count: int,
    q_values: tuple[float, ...],
    fixed_q: float | None,
    max_steps: int,
    flat_include_adjacency: bool,
):
    """Build a mixed-q training env or a fixed-q paired evaluation env."""
    common = {
        "scenario_path": scenario_path,
        "node_count": node_count,
        "q_values": q_values,
        "fixed_q": fixed_q,
        "max_steps": max_steps,
    }
    if model == "flat":
        return Pilot2FlatEnv(
            **common,
            include_adjacency=flat_include_adjacency,
        )
    if model in {"gcn", "graphsage"}:
        return Pilot2GraphEnv(**common)
    raise ValueError(f"Unsupported Pilot v2 model: {model}")
