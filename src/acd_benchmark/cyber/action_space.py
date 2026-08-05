"""Action-space inspection and adaptation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from acd_benchmark.cyber.import_path import ensure_cyborg_on_path

ensure_cyborg_on_path()

from CybORG.Simulator.Actions.Action import Sleep  # noqa: E402
from CybORG.Simulator.Actions.AbstractActions import Analyse, Misinform, Monitor, Remove, Restore  # noqa: E402
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions import (  # noqa: E402
    DecoyApache,
    DecoyFemitter,
    DecoyHarakaSMPT,
    DecoySmss,
    DecoySSHD,
    DecoySvchost,
    DecoyTomcat,
    DecoyVsftpd,
)


GLOBAL_BLUE_ACTIONS = (Sleep, Monitor)
HOST_BLUE_ACTIONS = (
    Analyse,
    Remove,
    Restore,
    DecoyApache,
    DecoyFemitter,
    DecoyHarakaSMPT,
    DecoySmss,
    DecoySSHD,
    DecoySvchost,
    DecoyTomcat,
    DecoyVsftpd,
)


@dataclass(frozen=True)
class BlueActionSpec:
    """One entry in the benchmark's fixed discrete Blue action list.

    CybORG expects action objects with keyword arguments, while the benchmark
    policy emits an integer. This object is the reversible bridge between them.
    """

    action_cls: type
    hostname: str | None = None

    @property
    def name(self) -> str:
        """Human-readable action name used in reports/debugging."""
        if self.hostname is None:
            return self.action_cls.__name__
        return f"{self.action_cls.__name__}({self.hostname})"

    def instantiate(self, session: int = 0, agent: str = "Blue") -> Any:
        """Create the concrete CybORG action object for one env step."""
        if self.action_cls is Sleep:
            return self.action_cls()
        if self.action_cls is Monitor:
            return self.action_cls(session=session, agent=agent)
        if self.hostname is None:
            raise ValueError(f"{self.action_cls.__name__} requires a hostname")
        return self.action_cls(session=session, agent=agent, hostname=self.hostname)


def build_blue_discrete_action_specs(
    action_space: dict[str, Any],
    hostnames: tuple[str, ...],
    include_misinform: bool = False,
) -> tuple[BlueActionSpec, ...]:
    """Build a stable flattened Blue action list from the native action space.

    The default list mirrors the common CAGE2/ChallengeWrapper-style action
    family: two global actions plus host-parameterized analyse/remove/restore
    and decoy actions. Native CC2 also exposes ``Misinform``; it remains an
    explicit option because the benchmark's ChallengeWrapper-style action
    surface excludes it by default.
    """
    enabled_actions = {
        action_cls
        for action_cls, enabled in action_space.get("action", {}).items()
        if enabled
    }
    host_action_classes = list(HOST_BLUE_ACTIONS)
    if include_misinform:
        host_action_classes.insert(2, Misinform)

    specs: list[BlueActionSpec] = [
        BlueActionSpec(action_cls)
        for action_cls in GLOBAL_BLUE_ACTIONS
        if action_cls in enabled_actions
    ]
    for hostname in hostnames:
        for action_cls in host_action_classes:
            if action_cls in enabled_actions:
                specs.append(BlueActionSpec(action_cls=action_cls, hostname=hostname))
    return tuple(specs)
