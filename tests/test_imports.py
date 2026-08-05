import importlib


def test_project_imports():
    import acd_benchmark

    assert acd_benchmark.__version__ == "0.1.0"


def test_reorganized_wrapper_and_script_modules_import():
    modules = (
        "acd_benchmark.wrappers.shared.host_obs",
        "acd_benchmark.wrappers.shared.graph_obs",
        "acd_benchmark.wrappers.shared.flat_obs",
        "acd_benchmark.wrappers.shared.partial_obs",
        "acd_benchmark.wrappers.stage1.observation",
        "acd_benchmark.wrappers.stage1.env",
        "acd_benchmark.rl.policies.flat",
        "acd_benchmark.rl.policies.gcn",
        "acd_benchmark.rl.policies.graphsage",
        "acd_benchmark.rl.rollout.collection",
        "acd_benchmark.rl.ppo.advantages",
        "acd_benchmark.rl.ppo.config",
        "acd_benchmark.rl.ppo.optimizer",
        "acd_benchmark.rl.ppo.training",
        "acd_benchmark.rl.ppo.update",
        "scripts.inspection.audit_observation_encoder",
        "scripts.stage1.train_ppo",
        "scripts.stage1.run_matrix",
    )

    for module in modules:
        assert importlib.import_module(module) is not None
