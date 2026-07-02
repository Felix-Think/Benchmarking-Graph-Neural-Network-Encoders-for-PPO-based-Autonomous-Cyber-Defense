from pprint import pprint

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from acd_benchmark.cyber.env_factory import make_cc2_env


def main() -> None:
    cyborg = make_cc2_env()
    results = cyborg.reset("Blue")
    action_space = cyborg.get_action_space("Blue")

    print("hosts:")
    pprint(list(cyborg.get_ip_map().keys()))
    print("\naction_space:")
    pprint(action_space)
    print("\ninitial_observation_keys:")
    pprint(list(results.observation.keys()))
    print(f"\ninitial_observation_entry_count={len(results.observation)}")


if __name__ == "__main__":
    main()
