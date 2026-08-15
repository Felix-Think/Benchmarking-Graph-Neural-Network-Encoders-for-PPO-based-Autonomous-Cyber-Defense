from ipaddress import IPv4Address
from types import SimpleNamespace

from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.SSHBruteForce import (
    SSHBruteForce,
)


def test_ssh_bruteforce_records_exploit_decoy_event():
    action = SSHBruteForce(
        ip_address=IPv4Address("10.0.0.1"),
        agent="Red",
        session=0,
    )
    target_host = SimpleNamespace(events={"NetworkConnections": []})

    action._create_exploit_decoy_event(local_port=4444, target_host=target_host)

    assert target_host.events["NetworkConnections"] == [
        {
            "local_address": IPv4Address("10.0.0.1"),
            "local_port": 4444,
        }
    ]
