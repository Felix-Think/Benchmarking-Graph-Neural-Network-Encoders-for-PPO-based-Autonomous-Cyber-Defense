from enum import Enum
from ipaddress import IPv4Address, IPv4Network
from pprint import pformat
from typing import Any

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from acd_benchmark.cyber.env_factory import make_cc2_env  # noqa: E402


def normalize(value: Any) -> Any:
    """Convert CybORG objects to readable Python values for inspection."""
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (IPv4Address, IPv4Network)):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, dict):
        return {normalize(k): normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize(item) for item in value)
    return value


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def print_key_value_block(data: dict[str, Any], indent: str = "  ") -> None:
    for key, value in data.items():
        print(f"{indent}{key}: {pformat(normalize(value), sort_dicts=False)}")


def print_table(rows: list[dict[str, Any]], indent: str = "  ") -> None:
    if not rows:
        print(f"{indent}- none")
        return

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    normalized_rows = [{column: normalize(row.get(column, "")) for column in columns} for row in rows]
    rendered_rows = [
        {column: pformat(row[column], width=80, sort_dicts=False) for column in columns}
        for row in normalized_rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rendered_rows))
        for column in columns
    }

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    print(f"{indent}{header}")
    print(f"{indent}{divider}")
    for row in rendered_rows:
        print(f"{indent}" + " | ".join(row[column].ljust(widths[column]) for column in columns))


def split_observation(observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metadata = {key: value for key, value in observation.items() if not isinstance(value, dict)}
    hosts = {key: value for key, value in observation.items() if isinstance(value, dict)}
    return metadata, hosts


def print_action_space_summary(action_space: dict[str, Any]) -> None:
    print_section("BLUE ACTION SPACE")
    for category, values in action_space.items():
        print(f"\n{category}:")
        if isinstance(values, dict):
            allowed = [normalize(key) for key, enabled in values.items() if enabled]
            blocked = [normalize(key) for key, enabled in values.items() if not enabled]
            print(f"  allowed ({len(allowed)}): {pformat(allowed, sort_dicts=False)}")
            print(f"  blocked ({len(blocked)}): {pformat(blocked, sort_dicts=False)}")
        else:
            print(f"  {pformat(normalize(values), sort_dicts=False)}")


def print_observation_summary(observation: dict[str, Any], all_hosts: list[str]) -> None:
    metadata, observed_hosts = split_observation(observation)
    unobserved_hosts = [host for host in all_hosts if host not in observed_hosts]
    field_names = sorted(
        {
            field
            for host_observation in observed_hosts.values()
            for field in host_observation
        }
    )

    print_section("BLUE INITIAL OBSERVATION SUMMARY")
    print(f"observation_type: {type(observation).__name__}")
    print(f"top_level_entries: {len(observation)}")
    print(f"metadata_entries: {list(metadata)}")
    print(f"observed_hosts ({len(observed_hosts)}): {list(observed_hosts)}")
    print(f"unobserved_hosts ({len(unobserved_hosts)}): {unobserved_hosts}")
    print(f"host_field_names: {field_names}")
    if metadata:
        print("\nmetadata:")
        print_key_value_block(metadata)


def print_host_observation(hostname: str, host_observation: dict[str, Any]) -> None:
    print(f"\n--- {hostname} ---")

    system_info = host_observation.get("System info")
    if isinstance(system_info, dict):
        print("  System info:")
        print_key_value_block(system_info, indent="    ")

    for field_name in ("Interface", "Sessions", "Processes", "User Info"):
        value = host_observation.get(field_name)
        print(f"  {field_name}:")
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            print_table(value, indent="    ")
        elif value is None:
            print("    - none")
        else:
            print(f"    {pformat(normalize(value), sort_dicts=False)}")

    known_fields = {"System info", "Interface", "Sessions", "Processes", "User Info"}
    extra_fields = {key: value for key, value in host_observation.items() if key not in known_fields}
    if extra_fields:
        print("  Extra fields:")
        print_key_value_block(extra_fields, indent="    ")


def print_full_blue_observation(observation: dict[str, Any]) -> None:
    _, observed_hosts = split_observation(observation)
    print_section("BLUE INITIAL OBSERVATION BY HOST")
    for hostname, host_observation in observed_hosts.items():
        print_host_observation(hostname, host_observation)


def print_normalized_raw_observation(observation: dict[str, Any]) -> None:
    print_section("NORMALIZED RAW BLUE OBSERVATION")
    print(pformat(normalize(observation), width=120, sort_dicts=False))


def main() -> None:
    cyborg = make_cc2_env()
    results = cyborg.reset("Blue")
    action_space = cyborg.get_action_space("Blue")
    all_hosts = list(cyborg.get_ip_map())

    print_section("CC2 HOSTS")
    print(f"host_count: {len(all_hosts)}")
    for index, hostname in enumerate(all_hosts, start=1):
        print(f"  {index:02d}. {hostname}")

    print_action_space_summary(action_space)
    print_observation_summary(results.observation, all_hosts)
    print_full_blue_observation(results.observation)
    print_normalized_raw_observation(results.observation)


if __name__ == "__main__":
    main()
