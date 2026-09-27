

class CyberMetrics:
    def reset(self, host_metadata: dict[str, dict]) -> None:
        self.host_data = {
            host: metadata.copy() for host, metadata in host_metadata.items()
        }
        self.ticks = 0

        self.ot_active_service_ticks = 0
        self.ot_total_service_ticks = 0
        self.ot_by_phase_subnet = {}

        self.compromised_host_ticks = 0
        self.protected_host_ticks = 0
        self.compromised_by_phase_subnet = {}
        self.current_compromised_host = set()
        self.ever_compromised_host = set()

        self.impact_by_phase_subnet = {}
        self.phase_ticks = {}
        self.impact_success = 0 
        self.impact_attempt = 0

        self.green_by_phase_subnet_action = {}

        self.ot_down_since = {}
        self.ot_recovery_events = []

    def update(self, phase:int, active_ot_host: set[str], priveileged_red_host: set[str]) -> None:
        self.ticks += 1
        self.phase_ticks[phase] = self.phase_ticks.get(phase, 0) + 1

        # duyyệt qua từng host subnet ở trong self.ot_host_subnet để lấy count
        for host, metadata in self.host_data.items():
            subnet = metadata["subnet"]
            key = (phase, subnet)

            if metadata["has_ot"]:
                is_active = int(host in active_ot_host)
                self.ot_active_service_ticks += is_active
                self.ot_total_service_ticks += 1
                if key not in self.ot_by_phase_subnet:
                    self.ot_by_phase_subnet[key] = {"total": 0, "active": 0}
                self.ot_by_phase_subnet[key]["active"] += is_active
                self.ot_by_phase_subnet[key]["total"] += 1

            if metadata["protected"]:
                is_compromised = int(host in priveileged_red_host)
                self.compromised_host_ticks += is_compromised
                self.protected_host_ticks += 1
                if key not in self.compromised_by_phase_subnet:
                    self.compromised_by_phase_subnet[key] = {"compromised": 0, "protected": 0}
                self.compromised_by_phase_subnet[key]['compromised'] += is_compromised
                self.compromised_by_phase_subnet[key]['protected'] += 1

            

    def update_green(self, phase:int, subnet: str, action_name:str, success:bool) -> None:
            if action_name not in ("GreenLocalWork", "GreenAccessService"):
                raise ValueError(f"Unsupported Green action: {action_name}")
            key = (phase, subnet, action_name)
            if key not in self.green_by_phase_subnet_action:
                self.green_by_phase_subnet_action[key] = {"attempt": 0, "success": 0} # total attempt to connect, and connect successed 
            self.green_by_phase_subnet_action[key]["attempt"] += 1
            self.green_by_phase_subnet_action[key]["success"] += int(success)

    def update_impact(self, phase:int, subnet:str, success: bool) -> None:
        key = (phase, subnet)

        self.impact_attempt += 1
        self.impact_success += int(success)
        if key not in self.impact_by_phase_subnet:
            self.impact_by_phase_subnet[key] = {"attempt": 0, "success" : 0}
        self.impact_by_phase_subnet[key]["attempt"] += 1
        self.impact_by_phase_subnet[key]["success"] += int(success)

    def update_compromised_host(self, active_red_host: set[str]) -> None:
        self.current_compromised_host = {hostname for hostname, metadata in self.host_data.items() if metadata["protected"] and hostname in active_red_host}
        self.ever_compromised_host.update(self.current_compromised_host)

    def update_ot_recovery(self, active_ot_hosts: set[str]) -> None:
        for hostname, metadata in self.host_data.items():
            if not metadata['has_ot']:
                continue 

            if hostname not in active_ot_hosts:
                self.ot_down_since.setdefault(hostname, self.ticks) # khong thay doi value de tinh last_value - first_value

            elif hostname in self.ot_down_since:
                start_tick = self.ot_down_since.pop(hostname)
                self.ot_recovery_events.append({
                    "hostname": hostname, 
                    "start_tick": start_tick,
                    "recovery_tick": self.ticks,
                    "ttr_ticks": self.ticks - start_tick,
                })


    def summary(self) -> dict:
        by_phase_subnet: list[dict] = []
        group_keys = (self.ot_by_phase_subnet.keys() | self.compromised_by_phase_subnet.keys() | self.impact_by_phase_subnet.keys())

        for phase, subnet in sorted(group_keys):
            key = (phase, subnet)

            ot_counts = self.ot_by_phase_subnet.get(key, {})
            compromised_counts = self.compromised_by_phase_subnet.get(key, {})
            impact_counts = self.impact_by_phase_subnet.get(key, {})
            
            ot_total = ot_counts.get('total', 0)
            ot_active = ot_counts.get('active', 0)

            protected_total = compromised_counts.get('protected', 0)
            compromised_total = compromised_counts.get('compromised', 0) 

            
            attempts = impact_counts.get("attempt", 0)
            successes = impact_counts.get("success", 0)

            by_phase_subnet.append({
                "phase": phase, 
                "subnet": subnet,
                "total_service_ticks": ot_total,
                "active_service_ticks": ot_active,
                "phase_availability": ot_active / ot_total if ot_total > 0 else None,
                "protected_host_ticks": protected_total,
                "compromised_host_ticks": compromised_total,
                "phase_compromise_burden": compromised_total / protected_total if protected_total > 0 else None,
                "impact_attempts": attempts,
                "impact_successes": successes, 
                "phase_ticks": self.phase_ticks[phase],
                "successful_impacts_per_tick": successes / self.phase_ticks[phase] if self.phase_ticks[phase] > 0 else None, 

            })
        # tinhs cho green access
        green_results: list[dict] = []
        for (phase, subnet, action_name), counts in self.green_by_phase_subnet_action.items():
            attempt = counts['attempt']
            success = counts['success']

            success_rate = success / attempt if attempt >0 else None
            green_results.append({
                "phase": phase,
                "subnet": subnet,
                "action_name": action_name,
                "attempts": attempt,
                "success": success,
                "success_rate": success_rate,
            })
        
        total_service_ticks = self.ot_total_service_ticks
        active_service_ticks = self.ot_active_service_ticks

        total_compromised_ticks = self.compromised_host_ticks
        total_protected_ticks = self.protected_host_ticks

        total_ticks = self.ticks
        total_successful_impact = self.impact_success
        total_impact_attempt = self.impact_attempt
        return {
            "ticks": total_ticks,
            "ot_total_service_ticks": total_service_ticks ,
            "ot_active_service_ticks": active_service_ticks,
            "availability": active_service_ticks / total_service_ticks if total_service_ticks > 0 else None,
            "ot_recovery_events": [event.copy() for event in self.ot_recovery_events],
            "ot_unrecovery_events": [{
                "hostname": hostname,
                "start_tick": start_tick,
                "observed_tick": total_ticks - start_tick,
                "ttr_ticks": None,
            } for hostname, start_tick in self.ot_down_since.items()],
            "compromised_host_count": len(self.current_compromised_host),
            "unique_compromised_host_count": len(self.ever_compromised_host),
            "compromised_host_ticks": total_compromised_ticks,
            "protected_host_ticks" : total_protected_ticks,
            "compromise_burden": total_compromised_ticks / total_protected_ticks if total_protected_ticks > 0 else None,
            
            "impact_attempts": total_impact_attempt,
            "impact_successes": total_successful_impact,
            "successful_impacts_per_tick": total_successful_impact / total_ticks if total_ticks > 0 else None,
            "by_phase_subnet": by_phase_subnet,
            "green_task_success": green_results,
        }

if __name__ == '__main__':
    metrics = CyberMetrics()
    data = {
        "A": {"subnet": "zone_a", "has_ot": True, "protected": True},
        "B": {"subnet": "zone_b", "has_ot": True, "protected": True},
    }
    metrics.reset(data)

    metrics.update(0, {"A", "B"}, {"A"})
    metrics.update_impact(0, "zone_a", True)

    metrics.update(1, {"A"}, set())
    metrics.update_impact(1, "zone_a", False)

    print(metrics.summary())
