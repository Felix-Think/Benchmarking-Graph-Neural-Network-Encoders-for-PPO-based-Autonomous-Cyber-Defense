from acd_benchmark.metrics.cc4_collector import CyberMetricsCollector


class CyberMetricsEnv:
    def __init__(self, env, cyborg):
        self.env = env
        self.collector = CyberMetricsCollector(cyborg)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None):
        result = self.env.reset(seed=seed)
        self.collector.reset()
        return result

    def step(self, actions):
        result = self.env.step(actions)
        self.collector.collect_step()
        return result

    def cyber_summary(self):
        return self.collector.metrics.summary()