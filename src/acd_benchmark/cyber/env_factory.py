from acd_benchmark.cyber.import_path import ensure_cyborg_on_path

ensure_cyborg_on_path()

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator

from acd_benchmark.cyber.cyborg_cc2 import get_cc2_scenario_path


def make_cc2_env(red_agent=None) -> CybORG:
    """Create the current CC2 smoke-test environment.

    This is intentionally small and explicit. Larger topology generators and
    benchmark wrappers should be added outside this factory after the pilot
    observation/action interfaces are validated.
    """
    scenario_generator = FileReaderScenarioGenerator(str(get_cc2_scenario_path()))
    red_agent = RedMeanderAgent() if red_agent is None else red_agent
    return CybORG(scenario_generator, "sim", agents={"Red": red_agent})
