from __future__ import annotations

from copy import deepcopy

import numpy as np
from gymnasium.spaces import Box

from acd_benchmark.cyber.cc4_env import BLUE_AGENTS, make_cc4_env
from acd_benchmark.cyber.cc4_distribution import VALID_ENVIRONMENT_DISTRIBUTIONS

from CybORG.Agents.Wrappers import BlueFixedActionWrapper
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    AllowTrafficZone,
    BlockTrafficZone,
)
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DeployDecoy import (
    DeployDecoy,
)

from acd_benchmark.cyber.canonical_observation import (
    SUBNET_INDEX,
    SUBNETS,
    VALID_VARIANTS,
    CanonicalObservation,
    HostCompressedBuilder,
    ObservedHostEvents,
    flat_observation_dim,
)
from acd_benchmark.cyber.canonical_observation_v2 import (
    CANONICAL_SCHEMAS,
    RELATIONAL_EVENT_NOW_FEATURES,
    V2_NODE_FEATURES,
    HostCompressedV2Builder,
    ObservedHostEventDetails,
    v2_flat_observation_dim,
)


CANONICAL_PARSER_VERSION = "raw_blue_observation_v4_relational_events"


def require_current_canonical_parser(config: dict) -> None:
    configured = config.get("canonical_parser_version")
    if configured != CANONICAL_PARSER_VERSION:
        raise ValueError(
            "Checkpoint canonical parser is incompatible with the current "
            f"environment: checkpoint={configured!r}, "
            f"current={CANONICAL_PARSER_VERSION!r}. Retrain the checkpoint."
        )


class CC4CanonicalFlatEnv:
    """Expose one matched Flat serialization of the canonical CAGE 4 graph."""

    def __init__(
        self,
        episode_steps: int = 500,
        seed: int | None = None,
        canonical_variant: str = "action_aware",
        evidence_age_horizon: int = 50,
        environment_distribution: str = "official",
        canonical_schema: str = "v1",
        passive_memory_ttl: int = 20,
        analysis_memory_ttl: int = 20,
        semantic_action_mask: bool = False,
    ):
        if canonical_variant not in VALID_VARIANTS:
            raise ValueError(f"canonical_variant must be one of {VALID_VARIANTS}")
        if environment_distribution not in VALID_ENVIRONMENT_DISTRIBUTIONS:
            raise ValueError(
                "environment_distribution must be one of "
                f"{VALID_ENVIRONMENT_DISTRIBUTIONS}"
            )
        if canonical_schema not in CANONICAL_SCHEMAS:
            raise ValueError(f"canonical_schema must be one of {CANONICAL_SCHEMAS}")
        cyborg = make_cc4_env(
            episode_steps=episode_steps,
            seed=seed,
            environment_distribution=environment_distribution,
        )
        self.env = BlueFixedActionWrapper(cyborg, pad_spaces=True)
        self.possible_agents = BLUE_AGENTS
        self.episode_steps = episode_steps
        self.canonical_variant = canonical_variant
        self.evidence_age_horizon = evidence_age_horizon
        self.environment_distribution = environment_distribution
        self.canonical_schema = canonical_schema
        self.passive_memory_ttl = passive_memory_ttl
        self.analysis_memory_ttl = analysis_memory_ttl
        self.semantic_action_mask = semantic_action_mask
        self._step = 0
        observation_dim = (
            flat_observation_dim()
            if canonical_schema == "v1"
            else v2_flat_observation_dim()
        )
        self._observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self._builders = self._new_builders()
        self.last_canonical: dict[str, CanonicalObservation] = {}
        self.last_raw_observations: dict[str, dict] = {}
        self._visible_ip_to_subnet: dict[str, dict[str, int]] = {
            agent: {} for agent in self.possible_agents
        }

    def _new_builders(
        self,
    ) -> dict[str, HostCompressedBuilder | HostCompressedV2Builder]:
        if self.canonical_schema == "v2":
            return {
                agent: HostCompressedV2Builder(
                    agent=agent,
                    episode_steps=self.episode_steps,
                    variant=self.canonical_variant,
                    passive_memory_ttl=self.passive_memory_ttl,
                    analysis_memory_ttl=self.analysis_memory_ttl,
                )
                for agent in self.possible_agents
            }
        return {
            agent: HostCompressedBuilder(
                agent=agent,
                episode_steps=self.episode_steps,
                variant=self.canonical_variant,
                evidence_age_horizon=self.evidence_age_horizon,
            )
            for agent in self.possible_agents
        }

    @property
    def node_feature_names(self) -> tuple[str, ...]:
        if self.canonical_schema == "v2":
            return V2_NODE_FEATURES
        from acd_benchmark.cyber.canonical_observation import NODE_FEATURES

        return NODE_FEATURES

    @staticmethod
    def _observed_host_event_details(
        observation: dict,
        visible_ip_to_subnet: dict[str, int] | None = None,
    ) -> dict[str, ObservedHostEventDetails]:
        """Parse passive alert attributes from the Blue-visible observation.

        CAGE 4 combines the selected Blue action result with the automatic
        Monitor result. Reading ``Host.events`` after the step is unsafe because
        another end-turn Monitor can clear or replace those queues. The combined
        observation is the interface contract actually exposed to the policy.
        """
        details = {}
        visible_ip_to_subnet = (
            {} if visible_ip_to_subnet is None else visible_ip_to_subnet
        )
        action = observation.get("action")
        successful_decoy_target = (
            getattr(action, "hostname", None)
            if isinstance(action, DeployDecoy)
            and observation.get("success") == TernaryEnum.TRUE
            else None
        )
        for hostname, host_data in observation.items():
            if hostname in ("success", "message", "action"):
                continue
            if not isinstance(host_data, dict):
                continue

            processes = host_data.get("Processes", [])
            process_events = []
            connections = []
            for process in processes:
                process_connections = process.get("Connections", [])
                connections.extend(process_connections)
                # DeployDecoy returns the process Blue just created in the same
                # combined observation as the automatic Monitor result. Its
                # Properties normally describe the emulated service rather than
                # containing a reliable "decoy" marker. Treating that action
                # feedback as a passive alert leaks Blue's own action result into
                # the sensor-evidence channel. Connections are retained because
                # DeployDecoy itself does not create connection observations.
                is_decoy_process_feedback = hostname == successful_decoy_target
                if not is_decoy_process_feedback and (
                    not process_connections
                    or any(
                    process.get(key) is not None
                    for key in (
                        "PID",
                        "PPID",
                        "service_name",
                        "username",
                        "Properties",
                    )
                    )
                ):
                    process_events.append(process)

            if not process_events and not connections:
                continue
            relational = {
                name: 0.0 for name in RELATIONAL_EVENT_NOW_FEATURES
            }
            for connection in connections:
                local_port = connection.get("local_port")
                remote_port = connection.get("remote_port")
                local_class = CC4CanonicalFlatEnv._port_class(local_port)
                remote_class = CC4CanonicalFlatEnv._port_class(remote_port)
                if local_class != "ephemeral" and remote_class == "ephemeral":
                    relational["connection_inbound_now"] = 1.0
                elif local_class == "ephemeral" and remote_class != "ephemeral":
                    relational["connection_outbound_now"] = 1.0
                else:
                    relational["connection_direction_ambiguous_now"] = 1.0

                local_feature = CC4CanonicalFlatEnv._local_port_feature(local_port)
                relational[local_feature] = 1.0
                remote_feature = (
                    f"remote_port_{remote_class}_now"
                    if remote_class in {"ephemeral", "missing"}
                    else "remote_port_service_now"
                )
                relational[remote_feature] = 1.0

                remote_address = connection.get("remote_address")
                if remote_address is None:
                    relational["remote_address_missing_now"] = 1.0
                else:
                    remote_subnet = visible_ip_to_subnet.get(str(remote_address))
                    if remote_subnet is None:
                        relational["remote_address_unknown_now"] = 1.0
                    else:
                        relational["remote_address_known_host_now"] = 1.0
                        relational[
                            f"connection_remote_subnet_{remote_subnet}_now"
                        ] = 1.0
            details[hostname] = ObservedHostEventDetails(
                process_count=len(process_events),
                process_has_pid=any(
                    event.get("PID") is not None for event in process_events
                ),
                process_has_connection=any(
                    bool(event.get("Connections")) for event in process_events
                ),
                connection_count=len(connections),
                connection_has_remote_address=any(
                    connection.get("remote_address") is not None
                    for connection in connections
                ),
                connection_has_remote_port=any(
                    connection.get("remote_port") is not None
                    for connection in connections
                ),
                connection_has_pid=any(
                    connection.get("PID", connection.get("pid")) is not None
                    for connection in connections
                ),
                relational_values=tuple(
                    relational[name]
                    for name in RELATIONAL_EVENT_NOW_FEATURES
                ),
            )
        return details

    @staticmethod
    def _port_class(value) -> str:
        if value is None:
            return "missing"
        port = int(value)
        if port >= 49_152:
            return "ephemeral"
        if port < 1_024:
            return "well_known"
        return "registered"

    @staticmethod
    def _local_port_feature(value) -> str:
        if value is None:
            return "local_port_missing_now"
        port = int(value)
        if port >= 49_152:
            return "local_port_ephemeral_now"
        if port == 22:
            return "local_port_ssh_now"
        if port in {80, 443}:
            return "local_port_web_now"
        if port == 25:
            return "local_port_mail_now"
        if port in {3389, 3390}:
            return "local_port_remote_admin_now"
        return "local_port_other_service_now"

    @staticmethod
    def _hostname_subnet_index(hostname: str) -> int | None:
        for subnet in SUBNETS:
            if hostname.startswith(f"{subnet}_"):
                return SUBNET_INDEX[subnet]
        return None

    def _refresh_visible_ip_maps(self, raw_observations: dict) -> None:
        for agent, observation in raw_observations.items():
            visible_ip_to_subnet = self._visible_ip_to_subnet[agent]
            for hostname, host_data in observation.items():
                if not isinstance(host_data, dict):
                    continue
                subnet_index = self._hostname_subnet_index(hostname)
                if subnet_index is None:
                    continue
                for interface in host_data.get("Interface", []):
                    address = interface.get("ip_address")
                    if address is not None:
                        visible_ip_to_subnet[str(address)] = subnet_index

    @classmethod
    def _observed_host_events(
        cls,
        observation: dict,
    ) -> dict[str, ObservedHostEvents]:
        details = cls._observed_host_event_details(observation)
        return {
            hostname: ObservedHostEvents(
                process_creation=event.process_count > 0,
                network_connection=event.connection_count > 0,
            )
            for hostname, event in details.items()
        }

    def _add_decision_status(self, infos: dict, raw_observations: dict) -> dict:
        for agent in self.possible_agents:
            infos[agent]["decision_ready"] = (
                raw_observations[agent].get("success")
                != TernaryEnum.IN_PROGRESS
            )
        return infos

    def _normalise_episode_end(self, raw_terminated, raw_truncated):
        raw_done = {
            agent: bool(
                raw_terminated.get(agent, False)
                or raw_truncated.get(agent, False)
            )
            for agent in self.possible_agents
        }
        terminated = {agent: False for agent in self.possible_agents}
        return terminated, raw_done

    def _apply_semantic_action_masks(self, infos: dict) -> dict:
        if not self.semantic_action_mask:
            return infos
        transformed = {}
        for agent in self.possible_agents:
            info = dict(infos[agent])
            action_mask = np.asarray(info["action_mask"], dtype=np.bool_).copy()
            blocked_edges = self._builders[agent].firewall_blocked_edges
            for action_index, action in enumerate(self.env.actions(agent)):
                if not action_mask[action_index]:
                    continue
                edge = (
                    getattr(action, "from_subnet", None),
                    getattr(action, "to_subnet", None),
                )
                if isinstance(action, AllowTrafficZone):
                    action_mask[action_index] = edge in blocked_edges
                elif isinstance(action, BlockTrafficZone):
                    action_mask[action_index] = edge not in blocked_edges
            if not action_mask.any():
                raise RuntimeError(f"Semantic action mask removed every action for {agent}")
            info["action_mask"] = action_mask
            transformed[agent] = info
        return transformed

    def _build_reset_canonical(self, agent: str, observation: dict):
        if self.canonical_schema == "v2":
            return self._builders[agent].reset(
                observation,
                # Reset observations describe static host/service inventory.
                # They are not passive security events.
                event_details={},
            )
        return self._builders[agent].reset(
            observation,
            event_indicators={},
        )

    def _build_step_canonical(self, agent: str, observation: dict):
        if self.canonical_schema == "v2":
            return self._builders[agent].update(
                observation,
                step=self._step,
                event_details=self._observed_host_event_details(
                    observation,
                    self._visible_ip_to_subnet[agent],
                ),
            )
        return self._builders[agent].update(
            observation,
            step=self._step,
            event_indicators=self._observed_host_events(observation),
        )

    def reset(self, seed: int | None = None):
        raw_observations, infos = self.env.reset(seed=seed)
        self.last_raw_observations = deepcopy(raw_observations)
        self._visible_ip_to_subnet = {
            agent: {} for agent in self.possible_agents
        }
        self._refresh_visible_ip_maps(self.last_raw_observations)
        self._step = 0
        self._builders = self._new_builders()
        self.last_canonical = {
            agent: self._build_reset_canonical(
                agent,
                self.last_raw_observations[agent],
            )
            for agent in self.possible_agents
        }
        observations = {
            agent: canonical.flat_view()
            for agent, canonical in self.last_canonical.items()
        }
        infos = self._add_decision_status(infos, raw_observations)
        return observations, self._apply_semantic_action_masks(infos)

    def step(self, actions: dict[str, int]):
        (
            raw_observations,
            rewards,
            raw_terminated,
            raw_truncated,
            infos,
        ) = self.env.step(actions=actions)
        self.last_raw_observations = deepcopy(raw_observations)
        self._refresh_visible_ip_maps(self.last_raw_observations)
        self._step += 1
        self.last_canonical = {
            agent: self._build_step_canonical(
                agent,
                self.last_raw_observations[agent],
            )
            for agent in self.possible_agents
        }
        observations = {
            agent: canonical.flat_view()
            for agent, canonical in self.last_canonical.items()
        }
        terminated, truncated = self._normalise_episode_end(
            raw_terminated, raw_truncated
        )
        infos = self._add_decision_status(infos, raw_observations)
        infos = self._apply_semantic_action_masks(infos)
        return observations, rewards, terminated, truncated, infos

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return self._observation_space

    def action_labels(self, agent: str):
        return self.env.action_labels(agent)
