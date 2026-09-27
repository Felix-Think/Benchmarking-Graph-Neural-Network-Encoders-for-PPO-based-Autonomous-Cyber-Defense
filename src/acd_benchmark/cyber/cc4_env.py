from CybORG.env import CybORG
from CybORG.Agents import (
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
    SleepAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from acd_benchmark.cyber.cc4_distribution import (
    VALID_ENVIRONMENT_DISTRIBUTIONS,
    RandomPivotEnterpriseScenarioGenerator,
)


BLUE_AGENTS = tuple(f"blue_agent_{index}" for index in range(5))


def make_cc4_env(
    episode_steps: int = 200,
    seed: int | None = None,
    environment_distribution: str = "random_host",
) -> CybORG:
    if environment_distribution not in VALID_ENVIRONMENT_DISTRIBUTIONS:
        raise ValueError(
            "environment_distribution must be one of "
            f"{VALID_ENVIRONMENT_DISTRIBUTIONS}"
        )
    generator_class = (
        RandomPivotEnterpriseScenarioGenerator
        if environment_distribution == "random_host"
        else EnterpriseScenarioGenerator
    )
    scenario_generator = generator_class(
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
