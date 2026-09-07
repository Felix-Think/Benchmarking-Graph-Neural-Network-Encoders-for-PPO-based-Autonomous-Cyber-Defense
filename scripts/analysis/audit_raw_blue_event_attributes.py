from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DeployDecoy import (
    DeployDecoy,
)

from acd_benchmark.cyber.cc4_canonical_env import (
    CC4CanonicalFlatEnv,
    require_current_canonical_parser,
)
from acd_benchmark.cyber.cc4_structured_action_env import CC4StructuredActionEnv
from acd_benchmark.rl.policies.structured_action import (
    build_structured_gcn_policy_from_config,
    build_structured_nodemlp_policy_from_config,
)
from acd_benchmark.rl.ppo.evaluation import load_policy_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure which process, connection and file attributes are actually "
            "present in strict raw Blue observations."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--action-seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _build_policy(config: dict):
    policy_type = config.get("policy_type")
    if policy_type == "structured_nodemlp":
        return build_structured_nodemlp_policy_from_config(config)
    if policy_type == "structured_gcn":
        return build_structured_gcn_policy_from_config(config)
    raise ValueError(f"Unsupported policy type: {policy_type!r}")


def _port_class(value) -> str:
    if value is None:
        return "missing"
    port = int(value)
    if port >= 49_152:
        return "ephemeral"
    if port < 1_024:
        return "well_known"
    return "registered"


def _connection_direction(local_port, remote_port) -> str:
    local_class = _port_class(local_port)
    remote_class = _port_class(remote_port)
    if local_class != "ephemeral" and remote_class == "ephemeral":
        return "inbound"
    if local_class == "ephemeral" and remote_class != "ephemeral":
        return "outbound"
    return "ambiguous"


def _update_ip_map(raw_observations: dict, ip_to_host: dict[str, str]) -> None:
    for observation in raw_observations.values():
        for hostname, host_data in observation.items():
            if not isinstance(host_data, dict):
                continue
            for interface in host_data.get("Interface", []):
                address = interface.get("ip_address")
                if address is not None:
                    ip_to_host[str(address)] = hostname


def _record_attributes(
    canonical_env: CC4CanonicalFlatEnv,
    counters: Counter,
    values: dict[str, Counter],
    ip_to_host: dict[str, str],
) -> None:
    _update_ip_map(canonical_env.last_raw_observations, ip_to_host)
    for agent, observation in canonical_env.last_raw_observations.items():
        action = observation.get("action")
        successful_decoy_target = (
            getattr(action, "hostname", None)
            if isinstance(action, DeployDecoy)
            and observation.get("success") == TernaryEnum.TRUE
            else None
        )
        counters["agent_ticks"] += 1
        for hostname, host_data in observation.items():
            if not isinstance(host_data, dict):
                continue
            counters["host_payloads"] += 1
            processes = host_data.get("Processes", [])
            files = host_data.get("Files", [])
            for process in processes:
                connections = process.get("Connections", [])
                for key in process:
                    values["process_keys"][str(key)] += 1
                if hostname != successful_decoy_target:
                    counters["process_records"] += 1
                    values["process_service_name"][
                        str(process.get("service_name", "<missing>"))
                    ] += 1
                    values["process_name"][
                        str(process.get("process_name", "<missing>"))
                    ] += 1
                    for property_value in process.get("Properties", ()) or ():
                        values["process_properties"][str(property_value)] += 1
                    counters["process_with_pid"] += int(
                        process.get("PID", process.get("pid")) is not None
                    )
                else:
                    counters["filtered_decoy_process_records"] += 1

                for connection in connections:
                    counters["connection_records"] += 1
                    for key in connection:
                        values["connection_keys"][str(key)] += 1
                    local_port = connection.get("local_port")
                    remote_port = connection.get("remote_port")
                    values["local_port_class"][_port_class(local_port)] += 1
                    values["remote_port_class"][_port_class(remote_port)] += 1
                    values["connection_direction"][
                        _connection_direction(local_port, remote_port)
                    ] += 1
                    if local_port is not None:
                        values["local_ports"][str(local_port)] += 1
                    if remote_port is not None:
                        values["remote_ports"][str(remote_port)] += 1
                    remote_address = connection.get("remote_address")
                    if remote_address is None:
                        remote_scope = "missing"
                    elif str(remote_address) in ip_to_host:
                        remote_scope = "known_host"
                    else:
                        remote_scope = "unknown_address"
                    values["remote_scope"][remote_scope] += 1
                    counters["connection_with_pid"] += int(
                        connection.get("PID", connection.get("pid")) is not None
                    )

            for file_data in files:
                counters["file_records"] += 1
                for key in file_data:
                    values["file_keys"][str(key)] += 1
                counters["file_with_density"] += int(
                    file_data.get("Density") is not None
                )
                values["file_signed"][str(file_data.get("Signed", "<missing>"))] += 1


def run_audit(args: argparse.Namespace) -> dict:
    seed_dir = args.run_dir / f"seed_{args.seed}"
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    require_current_canonical_parser(config)
    policy = _build_policy(config)
    checkpoint_path = args.checkpoint_path or seed_dir / "checkpoint.pt"
    load_policy_checkpoint(policy, checkpoint_path, device=args.device)
    policy.eval()
    device = next(policy.parameters()).device

    canonical_env = CC4CanonicalFlatEnv(
        episode_steps=int(config["episode_steps"]),
        seed=args.eval_seed,
        canonical_variant=config["canonical_variant"],
        evidence_age_horizon=int(config["evidence_age_horizon"]),
        environment_distribution=config["environment_distribution"],
        canonical_schema=config.get("canonical_schema", "v1"),
        passive_memory_ttl=int(config.get("passive_memory_ttl", 20)),
        analysis_memory_ttl=int(config.get("analysis_memory_ttl", 20)),
        semantic_action_mask=bool(config.get("semantic_action_mask", False)),
    )
    env = CC4StructuredActionEnv(canonical_env)
    torch.manual_seed(args.action_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.action_seed)

    counters: Counter = Counter()
    values: dict[str, Counter] = {
        name: Counter()
        for name in (
            "process_keys",
            "process_service_name",
            "process_name",
            "process_properties",
            "connection_keys",
            "local_port_class",
            "remote_port_class",
            "connection_direction",
            "local_ports",
            "remote_ports",
            "remote_scope",
            "file_keys",
            "file_signed",
        )
    }
    ip_to_host: dict[str, str] = {}
    returns = []
    for episode in range(args.episodes):
        observations, infos = env.reset(seed=args.eval_seed + episode)
        _update_ip_map(canonical_env.last_raw_observations, ip_to_host)
        episode_return = 0.0
        while True:
            if canonical_env._step > 0:
                _record_attributes(canonical_env, counters, values, ip_to_host)
            actions = {}
            for agent in env.possible_agents:
                if not infos[agent]["decision_ready"]:
                    continue
                observation = torch.as_tensor(
                    observations[agent], dtype=torch.float32, device=device
                )
                action_mask = torch.as_tensor(
                    infos[agent]["action_mask"], dtype=torch.bool, device=device
                )
                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(
                        observation, action_mask=action_mask
                    )
                actions[agent] = int(action.item())
            observations, rewards, terminated, truncated, infos = env.step(actions)
            episode_return += float(next(iter(rewards.values())))
            if all(
                terminated.get(agent, False) or truncated.get(agent, False)
                for agent in env.possible_agents
            ):
                break
        returns.append(episode_return)

    payload = {
        "protocol": {
            "training_seed": args.seed,
            "episodes": args.episodes,
            "eval_seed": args.eval_seed,
            "action_seed": args.action_seed,
            "checkpoint_path": str(checkpoint_path),
            "observation_boundary": "strict raw combined Blue observation",
            "reset_inventory_counted_as_event": False,
            "deploy_decoy_process_feedback_counted_as_event": False,
        },
        "evaluation": {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
        },
        "counts": dict(sorted(counters.items())),
        "attribute_values": {
            name: dict(counter.most_common())
            for name, counter in values.items()
        },
        "known_ip_addresses_from_blue_observations": len(ip_to_host),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_audit(args)
    print(json.dumps(payload["counts"], indent=2))
    for name in (
        "connection_direction",
        "local_port_class",
        "remote_port_class",
        "remote_scope",
        "file_signed",
    ):
        print(name, json.dumps(payload["attribute_values"][name], indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
