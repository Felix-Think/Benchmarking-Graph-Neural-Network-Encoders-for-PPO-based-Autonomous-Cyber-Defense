from pathlib import Path

from acd_benchmark.cyber.import_path import ensure_cyborg_on_path

ensure_cyborg_on_path()

from CybORG import CybORG  # noqa: E402
from CybORG.Agents import RedMeanderAgent  # noqa: E402
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator  # noqa: E402

from acd_benchmark.cyber.cyborg_cc2 import get_cc2_scenario_path  # noqa: E402
from acd_benchmark.cyber.scalable_scenario import (  # noqa: E402
    ScalableCAGE2ScenarioGenerator,
)


def make_cc2_env(red_agent=None) -> CybORG:
    """Create the official 13-host CAGE-2 environment used by inspections."""
    scenario_generator = FileReaderScenarioGenerator(str(get_cc2_scenario_path()))
    red_agent = RedMeanderAgent() if red_agent is None else red_agent
    return CybORG(
        scenario_generator,
        "sim",
        agents={"Red": red_agent},
    )


def make_scalable_cage2_env(scenario_path: Path, red_agent=None) -> CybORG:
    """Create a generated CAGE-2-style environment from a frozen YAML file."""
    scenario_generator = ScalableCAGE2ScenarioGenerator(str(scenario_path))
    red_agent = RedMeanderAgent() if red_agent is None else red_agent
    return CybORG(
        scenario_generator,
        "sim",
        agents={"Red": red_agent},
    )
