"""Regression test for service-session reload in CybORG StopProcess."""

from types import SimpleNamespace
from unittest.mock import Mock

from CybORG.Simulator.Actions.ConcreteActions.StopProcess import StopProcess
from CybORG.Simulator.Session import Session


def test_stop_process_reloads_service_from_session_object() -> None:
    """A session ID must be resolved before copying reload attributes."""
    process = SimpleNamespace(pid=123, user="service-user")
    session = Session(
        ident=4,
        hostname="Enterprise0",
        username="service-user",
        pid=123,
        session_type="shell",
        agent="Red",
        parent=None,
        timeout=7,
    )
    host = SimpleNamespace(
        hostname="Enterprise0",
        processes=[process],
        services={"service": {"process": 123}},
        sessions={"Red": [4]},
        add_process=Mock(),
    )
    state = SimpleNamespace(
        sessions={"Red": {4: session}},
        get_session_from_pid=Mock(return_value=("Red", 4)),
        add_session=Mock(),
    )

    action = StopProcess(session=0, agent="Blue", target_session=1, pid=123)
    action.kill_process(state, host, process)

    assert host.sessions["Red"] == []
    assert state.sessions["Red"] == {}
    state.add_session.assert_called_once_with(
        host="Enterprise0",
        user="service-user",
        session_type=session.session_type,
        agent="Red",
        parent=None,
        timeout=7,
    )
