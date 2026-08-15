from CybORG.Agents import BlueReactRemoveAgent

from acd_benchmark.cyber.cc2_env import (
    get_default_scenario_path,
    make_cc2_env,
)


def test_cc2_env_steps_once():
    cyborg = make_cc2_env(get_default_scenario_path())
    agent = BlueReactRemoveAgent()

    results = cyborg.reset("Blue")
    action = agent.get_action(results.observation, cyborg.get_action_space("Blue"))
    next_results = cyborg.step("Blue", action)

    assert len(cyborg.get_ip_map()) == 16
    assert next_results.reward is not None
