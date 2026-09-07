from types import SimpleNamespace

from CybORG.Simulator.SimulationController import _find_observed_session


def test_session_reassignment_can_scan_past_host_observation_without_sessions():
    host_ip = "10.0.0.10"
    observations = [
        SimpleNamespace(data={host_ip: {"Processes": []}}),
        SimpleNamespace(
            data={
                host_ip: {
                    "Sessions": [
                        {
                            "agent": "red_agent_0",
                            "session_id": 7,
                        }
                    ]
                }
            }
        ),
    ]

    matching_observation, matching_session = _find_observed_session(
        observations,
        host_ip,
        "red_agent_0",
        7,
    )

    assert matching_observation is observations[1]
    assert matching_session is observations[1].data[host_ip]["Sessions"][0]
