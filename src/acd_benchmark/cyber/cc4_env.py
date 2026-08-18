from CybORG.env import CybORG
from CybORG.Agents import (
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
    SleepAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


BLUE_AGENTS = tuple(f"blue_agent_{index}" for index in range(5))


def make_cc4_env(
    episode_steps: int = 200,
    seed: int | None = None,
) -> CybORG:
    scenario_generator = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=episode_steps,
    )

    return CybORG(
        scenario_generator=scenario_generator,
        seed=seed,
    )
if __name__ == "__main__":
    env = make_cc4_env(episode_steps =200, seed=123)
    assert type(env).__name__ == "CybORG"   
    print(f"CybORG interface: {type(env)}")
    print(f"Controller: {type(env.environment_controller)}")
    print(f"Active agents: {env.active_agents}")
    print(BLUE_AGENTS)