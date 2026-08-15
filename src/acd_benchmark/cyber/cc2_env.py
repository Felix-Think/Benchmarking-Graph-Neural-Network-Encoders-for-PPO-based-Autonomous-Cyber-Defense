import inspect
from pathlib import Path

from CybORG import CybORG


def get_cyborg_path() -> Path:
    cyborg_file = Path(inspect.getfile(CybORG))
    return cyborg_file.resolve().parent 

def get_default_scenario_path() -> Path:
    cyborg_file = get_cyborg_path() 
    scenario_path = cyborg_file / "Simulator" / "Scenarios" /"scenario_files"/ "Scenario2.yaml"
    assert scenario_path.is_file(), (f"Không tìm thấy scenario mặc định: {scenario_path}")
    return scenario_path


def get_scenario_sematic_file (filename: str ) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    scenario_path = repo_root / "configs" / "scenarios" / filename
    assert scenario_path.is_file(), (f"Không tìm thấy scenario mặc định: {scenario_path}")
    return scenario_path

def make_cc2_env(scenario_path : Path, red_agent = None) -> CybORG:
    """Create the official 13-host CAGE-2 environment used by inspections."""
    from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator
    from CybORG.Agents import RedMeanderAgent

    scenario_generator = FileReaderScenarioGenerator(str(scenario_path))
    red_agent = RedMeanderAgent() if red_agent is None else red_agent
    return CybORG(
        scenario_generator,
        "sim",
        agents={"Red": red_agent},
    )

if __name__ == "__main__":
    env = make_cc2_env(get_default_scenario_path())

    assert type(env).__name__ == "CybORG"
    assert len(env.get_ip_map()) == 16
    
    print(f"CybORG interface: {type(env)}")
    print(f"Controller: {type(env.environment_controller)}")
    print(f"Active agents: {env.active_agents}")