"""Regression coverage for CybORG++ fixes and the local Restore invariant."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from acd_benchmark.cyber.cc2_env import (
    get_default_scenario_path,
    make_cc2_env,
)

from CybORG.Agents.Wrappers.BlueTableWrapper import BlueTableWrapper
from CybORG.Shared import Observation
from CybORG.Shared.Enums import OperatingSystemType, TrinaryEnum
from CybORG.Shared.RedRewardCalculator import PwnRewardCalculator
from CybORG.Simulator.Actions.AbstractActions.Remove import Remove
from CybORG.Simulator.Actions.AbstractActions.Restore import Restore
from CybORG.Simulator.Actions.ConcreteActions.StopProcess import (
    StopProcess,
)
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions.DecoyVsftpd import (
    vsftpd_decoy_factory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = (
    PROJECT_ROOT
    / "CybORG"
    / "CybORG"
    / "Simulator"
    / "Scenarios"
    / "scenario_files"
    / "images"
)


def _load_test_host_image(filename: str) -> dict:
    image = yaml.safe_load((IMAGE_ROOT / filename).read_text(encoding="utf-8"))
    return image["Test_Host"]


def test_patchset_identity_is_frozen() -> None:
    patch_document = (
        PROJECT_ROOT / "patches" / "cyborg_plus_plus" / "README.md"
    ).read_text(encoding="utf-8")

    assert "cyborg_plus_plus_bugfixes_v1_restore_service_invariant_v1" in (
        patch_document
    )
    assert "c343476cb4490ba3e850561e0d11e7036ce50822" in patch_document


def test_remove_reports_failure_when_no_process_is_removed() -> None:
    """Having a Blue session on a host is not itself a successful removal."""
    parent_session = SimpleNamespace(hostname="Defender", sus_pids={})
    target_session = SimpleNamespace(hostname="Enterprise0", ident=1)
    state = SimpleNamespace(
        sessions={"Blue": {0: parent_session, 1: target_session}},
        np_random=SimpleNamespace(choice=lambda sessions: sessions[0]),
    )

    observation = Remove(
        session=0,
        agent="Blue",
        hostname="Enterprise0",
    ).execute(state)

    assert observation.data["success"] == TrinaryEnum.FALSE


def test_remove_propagates_stop_process_result() -> None:
    """Remove must return the result produced by the attempted StopProcess."""
    parent_session = SimpleNamespace(
        hostname="Defender",
        sus_pids={"Enterprise0": [123]},
    )
    target_session = SimpleNamespace(hostname="Enterprise0", ident=1)
    state = SimpleNamespace(
        sessions={"Blue": {0: parent_session, 1: target_session}},
        np_random=SimpleNamespace(choice=lambda sessions: sessions[0]),
    )

    with patch.object(
        StopProcess,
        "execute",
        return_value=Observation(True),
    ) as execute:
        observation = Remove(
            session=0,
            agent="Blue",
            hostname="Enterprise0",
        ).execute(state)

    assert observation.data["success"] == TrinaryEnum.TRUE
    execute.assert_called_once_with(state)


def test_vsftpd_decoy_uses_ftp_port() -> None:
    decoy = vsftpd_decoy_factory.make_decoy(None)

    assert decoy.open_ports == [
        {"local_port": 21, "local_address": "0.0.0.0"}
    ]


def test_internal_image_marks_smb_process_type() -> None:
    processes = _load_test_host_image("Internal_image.yaml")["Processes"]
    smb_process = next(process for process in processes if process["PID"] == 4)

    assert smb_process["Process Type"] == "smb"


def test_linux_user_mysql_uses_sql_injection_port() -> None:
    processes = _load_test_host_image("linux_user_host_image1.yaml")["Processes"]
    mysql_process = next(
        process for process in processes if process["Process Name"] == "mysql"
    )

    assert mysql_process["Connections"][0]["local_port"] == 3390


@pytest.mark.parametrize(
    ("username", "os_type"),
    [
        ("NetworkService", OperatingSystemType.WINDOWS),
        ("vagrant", OperatingSystemType.WINDOWS),
        ("root", OperatingSystemType.LINUX),
        ("SYSTEM", OperatingSystemType.WINDOWS),
        ("pi", OperatingSystemType.LINUX),
        ("www-data", OperatingSystemType.LINUX),
    ],
)
def test_pwn_reward_counts_published_compromise_usernames(
    username: str,
    os_type: OperatingSystemType,
) -> None:
    host = SimpleNamespace(get_confidentiality_value=lambda default: "Medium")
    scenario = SimpleNamespace(get_host=lambda hostname: host)
    calculator = PwnRewardCalculator("Red", scenario)
    current_state = {
        "Target": {
            "System info": {"OSType": os_type},
            "Sessions": [{"Agent": "Red", "Username": username}],
        }
    }

    reward = calculator.calculate_reward(current_state, {}, {}, False)

    assert reward == 1.0
    assert calculator.compromised_hosts == {"Target": 1.0}


def test_exploit_marker_takes_priority_over_scan_shape() -> None:
    activity = [
        {"Connections": [{"local_port": port, "remote_port": remote_port}]}
        for port, remote_port in ((21, 4444), (22, 1001), (80, 1002))
    ]

    result = BlueTableWrapper._interpret_connections(object(), activity)

    assert result == "Exploit"


def test_service_observation_is_not_misclassified_as_scan() -> None:
    activity = [
        {
            "Service Name": "apache",
            "Connections": [{"local_port": 80}],
        }
    ]

    result = BlueTableWrapper._interpret_connections(object(), activity)

    assert result == "None"


def test_restore_preserves_critical_service_process_mapping() -> None:
    """Restore must not orphan OTService from its reconstructed process."""
    cyborg = make_cc2_env(get_default_scenario_path())
    cyborg.reset("Blue", seed=1701)
    cyborg.step(
        "Blue",
        Restore(session=0, agent="Blue", hostname="Op_Server0"),
    )

    host = cyborg.environment_controller.state.hosts["Op_Server0"]
    service = host.services["OTService"]
    process_ids = {process.pid for process in host.processes}
    assert service["active"] is True
    assert service["process"] in process_ids
