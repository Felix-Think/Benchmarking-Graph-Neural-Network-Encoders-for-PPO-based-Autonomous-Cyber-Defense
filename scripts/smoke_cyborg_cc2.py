import argparse

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from acd_benchmark.cyber.env_factory import make_cc2_env
from CybORG.Agents import BlueReactRemoveAgent


def run_episode(steps: int) -> None:
    cyborg = make_cc2_env()
    blue_agent = BlueReactRemoveAgent()

    results = cyborg.reset("Blue")
    observation = results.observation
    print(f"hosts={len(cyborg.get_ip_map())}")

    total_reward = 0.0
    for step in range(steps):
        action_space = cyborg.get_action_space("Blue")
        action = blue_agent.get_action(observation, action_space)
        results = cyborg.step("Blue", action)
        observation = results.observation
        total_reward += float(results.reward)

        print(
            f"step={step} "
            f"blue={cyborg.get_last_action('Blue')} "
            f"red={cyborg.get_last_action('Red')} "
            f"reward={results.reward}"
        )

        if results.done:
            print(f"done=True at step={step}")
            break

    print(f"total_reward={total_reward}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless CybORG CC2 smoke test.")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    run_episode(args.steps)


if __name__ == "__main__":
    main()
