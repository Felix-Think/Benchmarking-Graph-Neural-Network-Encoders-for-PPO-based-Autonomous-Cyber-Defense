"""Discrete action-layout validation shared by graph policies."""

from __future__ import annotations

from dataclasses import dataclass

from acd_benchmark.cyber.action_space import BlueActionSpec


@dataclass(frozen=True)
class ActionHeadSpec:
    """Global and per-host dimensions of the fixed discrete action list."""

    num_global_actions: int
    num_hosts: int
    num_host_actions: int

    @property
    def action_count(self) -> int:
        """Return the total number of discrete actions."""
        return self.num_global_actions + self.num_hosts * self.num_host_actions


def infer_action_head_spec(
    action_specs: tuple[BlueActionSpec, ...],# BlueActionSpec[action, hostname]
    hostnames: tuple[str, ...], #sorted hostname
) -> ActionHeadSpec:
    """Validate the action order and infer its global/per-host dimensions."""
    num_global_actions = sum(spec.hostname is None for spec in action_specs)
    host_action_count = len(action_specs) - num_global_actions
    if len(hostnames) == 0 or host_action_count % len(hostnames) != 0:
        raise ValueError("Action list is not host-factorized.")
    num_host_actions = host_action_count // len(hostnames)

    cursor = num_global_actions
    for hostname in hostnames:
        host_slice = action_specs[cursor : cursor + num_host_actions]
        if len(host_slice) != num_host_actions or any(
            spec.hostname != hostname for spec in host_slice
        ): # check xem thu tu action[host_name] có đúng thư tự như hostnames không, phải cùng 1 format để encoder
            raise ValueError("Action list order does not match host order.")
        cursor += num_host_actions
    return ActionHeadSpec(
        num_global_actions=num_global_actions,
        num_hosts=len(hostnames),
        num_host_actions=num_host_actions,
    )
