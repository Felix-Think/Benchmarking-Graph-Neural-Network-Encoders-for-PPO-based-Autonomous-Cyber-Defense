from CybORG.env import CybORG 
from CybORG.Shared.Enums import ProcessName
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import GreenLocalWork, GreenAccessService, Impact

from acd_benchmark.metrics.cyber import CyberMetrics

class CyberMetricsCollector:
    def __init__(self, cyborg: CybORG):
        self.cyborg = cyborg 
        self.metrics = CyberMetrics() 

    def reset(self) -> None:
        controller = self.cyborg.environment_controller
        state = controller.state 

        protected_subnet = set() 

        for agent in controller.team["Blue"]: 
            interface = controller.agent_interfaces[agent]
            protected_subnet.update(interface.allowed_subnets)

        host_metadata = {}
        for hostname, host in state.hosts.items():
            subnet = state.hostname_subnet_map[hostname].value 
            is_router = hostname.endswith('_router')

            host_metadata[hostname] = {
                'subnet': subnet,
                'has_ot': ProcessName.OTSERVICE in host.services,
                'protected': (
                    subnet in protected_subnet and not is_router 
                )
            }
        self.metrics.reset(host_metadata)

    def _collect_action_outcomes(self) -> None:
        controller = self.cyborg.environment_controller
        state = controller.state 
        phase = state.mission_phase 

        agents = controller.team["Green"] + controller.team["Red"]

        for agent in agents:
            for obs in controller.observation[agent].observations:
                action = obs.data.get("action")
                result = obs.data.get("success")

                if result not in (TernaryEnum.TRUE, TernaryEnum.FALSE):
                    continue 
                success = result == TernaryEnum.TRUE
                if isinstance(action, (GreenLocalWork, GreenAccessService)):
                    hostname = state.ip_addresses[action.ip_address]
                    subnet = state.hostname_subnet_map[hostname].value 
                    self.metrics.update_green(phase, subnet, type(action).__name__, success)

                if isinstance(action, Impact):
                    subnet = state.hostname_subnet_map[action.hostname].value
                    self.metrics.update_impact(phase, subnet, success)

    def collect_step(self) -> None:
        controller = self.cyborg.environment_controller 
        state = controller.state 
        phase = state.mission_phase 

        active_ot_host = set()
        


        for hostname, metadata in self.metrics.host_data.items():
            if not metadata['has_ot']:
                continue 

            host = state.hosts[hostname]
            service = host.services.get(ProcessName.OTSERVICE)
            if service is not None and service.active:
                active_ot_host.add(hostname)
        active_red_host = set()
        privileged_red_host= set()
        for agent in controller.team["Red"]:
            for session in state.sessions[agent].values():

                if not session.active: 
                    continue 
                active_red_host.add(session.hostname)
                if session.has_privileged_access():
                    privileged_red_host.add(session.hostname)

        self.metrics.update(phase, active_ot_host, privileged_red_host)
        self.metrics.update_compromised_host(active_red_host)
        self.metrics.update_ot_recovery(active_ot_host)
        self._collect_action_outcomes()

if __name__ == "__main__":
    from pprint import pprint
    from acd_benchmark.cyber.cc4_env import make_cc4_env 

    cyborg = make_cc4_env(episode_steps = 10, seed = 123)
    cyborg.reset(seed = 123)

    collector = CyberMetricsCollector(cyborg)
    collector.reset()

    for _ in range(10):
        cyborg.step()
        collector.collect_step()

        if cyborg.environment_controller.done:
            break
    result = collector.metrics.summary()
    metadata = collector.metrics.host_data 
    assert result["ticks"] == cyborg.environment_controller.step_count
    assert result["ot_total_service_ticks"] == (
        result["ticks"] * sum(item["has_ot"] for item in metadata.values())
    )
    assert result["green_task_success"], "Chua thu duoc ket qua Green"

    pprint(result)