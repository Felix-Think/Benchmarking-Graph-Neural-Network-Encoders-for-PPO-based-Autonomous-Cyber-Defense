from acd_benchmark.cyber.dynamic_host_observation_mask import FixedBudgetDynamicHostMask
from acd_benchmark.rl.ppo.evaluation import (
    evaluate_policy_with_preserved_rng,
    preserved_evaluation_rng,
)


def evaluate_retention_curve(
    env_factory, policy, *, episode_steps, retentions, episodes, seed, mask_seed,
    modes=("stochastic",), global_step=0, logger=None, namespace="validation_retention",
    reference_results=None,
):
    """Fixed-q paired evaluation, never a random mixture of evaluation severities."""
    results = {}
    with preserved_evaluation_rng():
        for q in retentions:
            by_mode = {}
            for mode in modes:
                if q == 1.0 and reference_results is not None and mode in reference_results:
                    result = reference_results[mode]
                else:
                    env = FixedBudgetDynamicHostMask(
                        env_factory(episode_steps=episode_steps, seed=seed),
                        retention=q, mask_seed=mask_seed, mask_process="persistent",
                    )
                    result = evaluate_policy_with_preserved_rng(
                        env, policy, episodes=episodes, seed=seed,
                        deterministic=mode == "deterministic", action_seed=seed,
                    )
                by_mode[mode] = result.to_dict()
                if logger is not None:
                    logger.log_evaluation(
                        result, global_step=global_step,
                        namespace=f"{namespace}/q_{q:g}/{mode}",
                    )
                print(
                    f"retention_evaluation step={global_step} q={q:g} mode={mode} "
                    f"mean_return={result.mean_return:.2f} std_return={result.std_return:.2f}",
                    flush=True,
                )
            results[f"{q:g}"] = by_mode
    return {
        "global_step": global_step, "episodes": episodes, "environment_seed": seed,
        "action_seed": seed, "mask_seed": mask_seed, "mask_process": "persistent",
        "results": results,
    }
