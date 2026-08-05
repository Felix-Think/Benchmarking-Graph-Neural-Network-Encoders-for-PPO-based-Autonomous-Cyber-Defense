import argparse
from pathlib import Path
from typing import Any

import numpy as np

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from acd_benchmark.cyber.env_factory import make_cc2_env  # noqa: E402
from acd_benchmark.wrappers.shared.flat_obs import encode_flat_observation  # noqa: E402
from acd_benchmark.wrappers.shared.graph_obs import (  # noqa: E402
    build_graph_encoder,
    encode_graph_observation,
)
from acd_benchmark.wrappers.shared.host_obs import (  # noqa: E402
    ACTIVITY_EXPLOIT,
    ACTIVITY_SCAN,
    COMPROMISED_PRIVILEGED,
    COMPROMISED_UNKNOWN,
    COMPROMISED_USER,
    UNKNOWN,
    _as_list,
    _first_dict,
    _has_default_password,
    _is_privileged_username,
    _process_identity,
    normalize_string,
    split_blue_observation,
)
from CybORG.Agents import BlueReactRemoveAgent  # noqa: E402


def _count_norm(count: int, cap: int = 10) -> float:
    return min(float(count) / float(cap), 1.0)


def _feature_category_groups(feature_names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for name in feature_names:
        if "=" not in name:
            continue
        prefix, category = name.split("=", maxsplit=1)
        groups.setdefault(prefix, []).append(category)
    return {prefix: tuple(categories) for prefix, categories in groups.items()}


def _one_hot_expected(value: Any, category: str, categories: tuple[str, ...]) -> float:
    normalized = normalize_string(value)
    if normalized not in categories:
        normalized = UNKNOWN
    return float(normalized == category)


def _expected_host_features(
    *,
    hostname: str,
    host: dict[str, Any],
    feature_names: tuple[str, ...],
    category_groups: dict[str, tuple[str, ...]],
    baseline_processes: dict[str, set[tuple[tuple[str, str], ...]]],
    activity_by_host: dict[str, str],
    compromised_by_host: dict[str, str],
) -> dict[str, float]:
    system_info = host.get("System info", {}) if isinstance(host.get("System info"), dict) else {}
    interface = _first_dict(host.get("Interface"))
    sessions = [item for item in _as_list(host.get("Sessions")) if isinstance(item, dict)]
    processes = [item for item in _as_list(host.get("Processes")) if isinstance(item, dict)]
    user_info = [item for item in _as_list(host.get("User Info")) if isinstance(item, dict)]

    blue_sessions = [session for session in sessions if normalize_string(session.get("Agent")) == "Blue"]
    red_sessions = [session for session in sessions if normalize_string(session.get("Agent")) == "Red"]
    process_users = {normalize_string(process.get("Username")) for process in processes}
    baseline = baseline_processes.get(hostname, set())
    new_processes = [
        process for process in processes if _process_identity(process) not in baseline
    ]

    exact_features = {
        "known": float(bool(host)),
        "interface.has_interface": float(bool(interface)),
        "sessions.count_norm": _count_norm(len(sessions)),
        "sessions.blue_count_norm": _count_norm(len(blue_sessions)),
        "sessions.red_count_norm": _count_norm(len(red_sessions)),
        "sessions.has_velociraptor_client": float(
            any(normalize_string(session.get("Type")) == "VELOCIRAPTOR_CLIENT" for session in sessions)
        ),
        "sessions.has_privileged_user": float(
            any(_is_privileged_username(session.get("Username")) for session in sessions)
        ),
        "processes.count_norm": _count_norm(len(processes)),
        "processes.unique_user_count_norm": _count_norm(len(process_users)),
        "processes.has_privileged_user": float(
            any(_is_privileged_username(process.get("Username")) for process in processes)
        ),
        "processes.new_count_norm": _count_norm(len(new_processes)),
        "activity.is_scan": float(activity_by_host.get(hostname) == ACTIVITY_SCAN),
        "activity.is_exploit": float(activity_by_host.get(hostname) == ACTIVITY_EXPLOIT),
        "compromised.is_unknown": float(compromised_by_host.get(hostname) == COMPROMISED_UNKNOWN),
        "compromised.is_user": float(compromised_by_host.get(hostname) == COMPROMISED_USER),
        "compromised.is_privileged": float(
            compromised_by_host.get(hostname) == COMPROMISED_PRIVILEGED
        ),
        "user_info.count_norm": _count_norm(len(user_info)),
        "user_info.has_default_password": float(_has_default_password(user_info)),
        "user_info.has_privileged_user": float(
            any(_is_privileged_username(user.get("Username")) for user in user_info)
        ),
        "user_info.has_green_agent": float(
            any(normalize_string(user.get("Username")) == "GreenAgent" for user in user_info)
        ),
        "user_info.has_password_exposed": float(
            any("Password" in user or "Password Hash" in user for user in user_info)
        ),
    }

    expected: dict[str, float] = {}
    for feature_name in feature_names:
        if feature_name in exact_features:
            expected[feature_name] = exact_features[feature_name]
        elif feature_name.startswith("system.") and "=" in feature_name:
            prefix, category = feature_name.split("=", maxsplit=1)
            field = prefix.removeprefix("system.")
            expected[feature_name] = _one_hot_expected(
                system_info.get(field),
                category,
                category_groups[prefix],
            )
        elif feature_name.startswith("interface.Subnet="):
            prefix, category = feature_name.split("=", maxsplit=1)
            expected[feature_name] = _one_hot_expected(
                interface.get("Subnet"),
                category,
                category_groups[prefix],
            )
        else:
            raise KeyError(f"Audit does not know how to verify feature: {feature_name}")

    return expected


def _expected_adjacency(hostnames: tuple[str, ...], latest_hosts: dict[str, dict[str, Any]]) -> np.ndarray:
    host_subnets = [
        normalize_string(_first_dict(latest_hosts.get(hostname, {}).get("Interface")).get("Subnet"))
        for hostname in hostnames
    ]
    size = len(hostnames)
    adjacency = np.zeros((size, size), dtype=np.float32)
    for src in range(size):
        for dst in range(size):
            if src == dst:
                continue
            if host_subnets[src] != UNKNOWN and host_subnets[src] == host_subnets[dst]:
                adjacency[src, dst] = 1.0
    return adjacency


def _nonzero_features(feature_names: tuple[str, ...], row: np.ndarray) -> list[tuple[int, str, float]]:
    return [
        (index, feature_name, float(value))
        for index, (feature_name, value) in enumerate(zip(feature_names, row))
        if value != 0
    ]


def _raw_host_summary(host: dict[str, Any]) -> dict[str, Any]:
    system_info = host.get("System info", {}) if isinstance(host.get("System info"), dict) else {}
    interface = _first_dict(host.get("Interface"))
    sessions = [item for item in _as_list(host.get("Sessions")) if isinstance(item, dict)]
    processes = [item for item in _as_list(host.get("Processes")) if isinstance(item, dict)]
    user_info = [item for item in _as_list(host.get("User Info")) if isinstance(item, dict)]
    return {
        "system": {
            field: normalize_string(system_info.get(field))
            for field in ("Architecture", "OSType", "OSDistribution", "OSVersion")
        },
        "subnet": normalize_string(interface.get("Subnet")),
        "sessions": len(sessions),
        "blue_sessions": sum(normalize_string(session.get("Agent")) == "Blue" for session in sessions),
        "red_sessions": sum(normalize_string(session.get("Agent")) == "Red" for session in sessions),
        "processes": len(processes),
        "process_users": sorted({normalize_string(process.get("Username")) for process in processes}),
        "user_info": len(user_info),
        "users": [normalize_string(user.get("Username")) for user in user_info],
    }


def _format_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def run_audit(steps: int, max_hosts: int, output_path: Path) -> tuple[bool, list[str]]:
    cyborg = make_cc2_env()
    blue_agent = BlueReactRemoveAgent()

    result = cyborg.reset("Blue")
    observation = result.observation
    encoder = build_graph_encoder(observation)
    graph_obs = encode_graph_observation(encoder, observation)

    for _ in range(steps):
        action = blue_agent.get_action(observation, cyborg.get_action_space("Blue"))
        result = cyborg.step("Blue", action)
        observation = result.observation
        graph_obs = encode_graph_observation(encoder, observation)

    flat_obs = encode_flat_observation(graph_obs)
    _, current_step_hosts = split_blue_observation(observation)

    report: list[str] = [
        "# Observation Encoder Audit",
        "",
        f"- Steps before audit: `{steps}`",
        f"- Host count: `{len(graph_obs.hostnames)}`",
        f"- Feature count per host: `{len(graph_obs.feature_names)}`",
        f"- Graph node feature shape: `{list(graph_obs.node_features.shape)}`",
        f"- Adjacency shape: `{list(graph_obs.adjacency.shape)}`",
        f"- Edge index shape: `{list(graph_obs.edge_index.shape)}`",
        f"- Flat observation shape: `{list(flat_obs.shape)}`",
        f"- Hosts present in latest sparse step obs: `{list(current_step_hosts)}`",
        "",
        "## Meaning",
        "",
        "`known` means the host has cached data in `latest_hosts`; it does not mean the host appeared in the latest sparse step observation.",
        "",
    ]

    category_groups = _feature_category_groups(graph_obs.feature_names)
    mismatches: list[str] = []

    for host_index, hostname in enumerate(graph_obs.hostnames):
        host = encoder.latest_hosts.get(hostname, {})
        expected = _expected_host_features(
            hostname=hostname,
            host=host,
            feature_names=graph_obs.feature_names,
            category_groups=category_groups,
            baseline_processes=encoder.baseline_processes,
            activity_by_host=encoder.activity_by_host,
            compromised_by_host=encoder.compromised_by_host,
        )
        row = graph_obs.node_features[host_index]
        for feature_index, feature_name in enumerate(graph_obs.feature_names):
            actual_value = float(row[feature_index])
            expected_value = expected[feature_name]
            if not np.isclose(actual_value, expected_value, atol=1e-6):
                mismatches.append(
                    f"{hostname} feature {feature_index:02d} {feature_name}: "
                    f"actual={actual_value} expected={expected_value}"
                )

    expected_adjacency = _expected_adjacency(graph_obs.hostnames, encoder.latest_hosts)
    if not np.array_equal(graph_obs.adjacency, expected_adjacency):
        mismatches.append("adjacency does not match same-subnet rule from latest_hosts")

    expected_edges = set(zip(*np.nonzero(expected_adjacency)))
    actual_edges = set(zip(graph_obs.edge_index[0], graph_obs.edge_index[1]))
    if actual_edges != expected_edges:
        mismatches.append("edge_index does not match nonzero positions in adjacency")

    num_hosts, feature_dim = graph_obs.node_features.shape
    node_part_len = num_hosts * feature_dim
    flat_node_part = flat_obs[:node_part_len].reshape(num_hosts, feature_dim)
    flat_adjacency_part = flat_obs[node_part_len:].reshape(num_hosts, num_hosts)
    if not np.array_equal(flat_node_part, graph_obs.node_features):
        mismatches.append("flat node feature segment does not match graph node_features")
    if not np.array_equal(flat_adjacency_part, graph_obs.adjacency):
        mismatches.append("flat adjacency segment does not match graph adjacency")

    if mismatches:
        report.extend(["## Status", "", "FAILED", "", "## Mismatches", ""])
        report.extend(f"- {mismatch}" for mismatch in mismatches)
    else:
        report.extend(["## Status", "", "PASSED", ""])

    report.extend(
        [
            "## Feature Names",
            "",
        ]
    )
    for index, feature_name in enumerate(graph_obs.feature_names):
        report.append(f"- `{index:02d}` `{feature_name}`")

    report.extend(["", "## Host Samples", ""])
    for host_index, hostname in enumerate(graph_obs.hostnames[:max_hosts]):
        host = encoder.latest_hosts.get(hostname, {})
        summary = _raw_host_summary(host)
        report.extend(
            [
                f"### `{host_index}` `{hostname}`",
                "",
                f"- Raw system: `{summary['system']}`",
                f"- Raw subnet: `{summary['subnet']}`",
                f"- Raw sessions: `{summary['sessions']}` total, `{summary['blue_sessions']}` Blue, `{summary['red_sessions']}` Red",
                f"- Raw processes: `{summary['processes']}` total, users `{summary['process_users']}`",
                f"- Raw user info: `{summary['user_info']}` total, users `{summary['users']}`",
                "",
                "Non-zero encoded features:",
                "",
            ]
        )
        for feature_index, feature_name, value in _nonzero_features(
            graph_obs.feature_names,
            graph_obs.node_features[host_index],
        ):
            report.append(f"- `{feature_index:02d}` `{feature_name}` = `{_format_float(value)}`")
        report.append("")

    report.extend(
        [
            "## Known Limitations To Track",
            "",
            "- Same-subnet edges are not full router/gateway/firewall topology.",
            "- Process features may represent sparse process/connection observations rather than a complete process table.",
            "- Subnet one-hot categories are seed-specific IP ranges; this should be reviewed before multi-seed benchmark claims.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return not mismatches, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw Blue host encoder outputs against cached raw obs.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-hosts", type=int, default=13)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / "observation_encoder_audit" / "report.md",
    )
    parser.add_argument("--no-fail", action="store_true", help="Write the report but do not exit nonzero on mismatch.")
    args = parser.parse_args()

    passed, report = run_audit(
        steps=args.steps,
        max_hosts=args.max_hosts,
        output_path=args.output,
    )
    print("\n".join(report[:25]))
    print(f"\nreport={args.output}")
    if not passed and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
