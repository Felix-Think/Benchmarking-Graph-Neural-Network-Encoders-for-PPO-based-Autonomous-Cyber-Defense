import inspect
from pathlib import Path

from acd_benchmark.cyber.import_path import ensure_cyborg_on_path

ensure_cyborg_on_path()

from CybORG import CybORG


def get_cyborg_package_dir() -> Path:
    """Return the local CybORG package directory."""
    cyborg_file = inspect.getfile(CybORG)
    return Path(cyborg_file).resolve().parent


def get_cc2_scenario_path() -> Path:
    """Return the vendored CAGE Challenge 2 scenario path."""
    return get_cyborg_package_dir() / "Simulator" / "Scenarios" / "scenario_files" / "Scenario2.yaml"
