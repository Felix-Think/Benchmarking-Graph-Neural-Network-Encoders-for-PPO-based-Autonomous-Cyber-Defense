import argparse
import json
from pathlib import Path

import numpy as np

from acd_benchmark.cyber.cc4_canonical_env import CC4CanonicalFlatEnv

from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    AllowTrafficZone,
    BlockTrafficZone,
)


CONDITIONS = {
    "v1_mask_off": ("v1", False),
    "v2_mask_off": ("v2", False),
    "v1_mask_on": ("v1", True),
    "v2_mask_on": ("v2", True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that canonical V2 and the semantic traffic mask are "
            "independent interventions."
        )
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episode-steps", type=int, default=20)
    parser.add_argument("--passive-memory-ttl", type=int, default=20)
    parser.add_argument("--analysis-memory-ttl", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/canonical_v2_intervention_validation.json"),
    )
    return parser


def _reset_condition(args, schema: str, semantic_mask: bool):
    env = CC4CanonicalFlatEnv(
        episode_steps=args.episode_steps,
        seed=args.seed,
        canonical_schema=schema,
        passive_memory_ttl=args.passive_memory_ttl,
        analysis_memory_ttl=args.analysis_memory_ttl,
        semantic_action_mask=semantic_mask,
    )
    observations, infos = env.reset(seed=args.seed)
    return env, observations, infos


def _mask_summary(env, infos: dict) -> dict:
    summary = {}
    for agent in env.possible_agents:
        mask = np.asarray(infos[agent]["action_mask"], dtype=np.bool_)
        actions = env.env.actions(agent)
        allow_indices = np.asarray(
            [
                index
                for index, action in enumerate(actions)
                if isinstance(action, AllowTrafficZone)
            ],
            dtype=np.int64,
        )
        block_indices = np.asarray(
            [
                index
                for index, action in enumerate(actions)
                if isinstance(action, BlockTrafficZone)
            ],
            dtype=np.int64,
        )
        summary[agent] = {
            "action_count": int(mask.size),
            "valid_action_count": int(mask.sum()),
            "valid_allow_count": int(mask[allow_indices].sum()),
            "valid_block_count": int(mask[block_indices].sum()),
        }
    return summary


def validate(args) -> dict:
    states = {
        name: _reset_condition(args, schema, semantic_mask)
        for name, (schema, semantic_mask) in CONDITIONS.items()
    }

    for name, (env, observations, infos) in states.items():
        for agent in env.possible_agents:
            observation = np.asarray(observations[agent])
            action_mask = np.asarray(infos[agent]["action_mask"])
            assert observation.shape == env.observation_space(agent).shape
            assert np.isfinite(observation).all()
            assert (observation >= 0.0).all()
            assert (observation <= 1.0).all()
            assert action_mask.shape == (env.action_space(agent).n,)
            assert action_mask.any(), f"{name} removed every action for {agent}"

    comparisons = {}
    for schema in ("v1", "v2"):
        off = states[f"{schema}_mask_off"]
        on = states[f"{schema}_mask_on"]
        comparisons[f"{schema}_observation_unchanged_by_mask"] = all(
            np.array_equal(off[1][agent], on[1][agent])
            for agent in off[0].possible_agents
        )

    for mask_name in ("off", "on"):
        v1 = states[f"v1_mask_{mask_name}"]
        v2 = states[f"v2_mask_{mask_name}"]
        comparisons[f"mask_{mask_name}_unchanged_by_schema"] = all(
            np.array_equal(
                v1[2][agent]["action_mask"],
                v2[2][agent]["action_mask"],
            )
            for agent in v1[0].possible_agents
        )

    comparisons["semantic_mask_is_strict_initial_subset"] = all(
        np.all(
            np.asarray(states["v1_mask_on"][2][agent]["action_mask"])
            <= np.asarray(states["v1_mask_off"][2][agent]["action_mask"])
        )
        and np.any(
            np.asarray(states["v1_mask_on"][2][agent]["action_mask"])
            != np.asarray(states["v1_mask_off"][2][agent]["action_mask"])
        )
        for agent in states["v1_mask_off"][0].possible_agents
    )
    assert all(comparisons.values()), comparisons

    return {
        "seed": args.seed,
        "episode_steps": args.episode_steps,
        "passive_memory_ttl_simulator_ticks": args.passive_memory_ttl,
        "analysis_memory_ttl_simulator_ticks": args.analysis_memory_ttl,
        "conditions": {
            name: {
                "canonical_schema": schema,
                "semantic_action_mask": semantic_mask,
                "observation_dim": env.observation_space(
                    env.possible_agents[0]
                ).shape[0],
                "node_feature_count": len(env.node_feature_names),
                "mask_summary": _mask_summary(env, infos),
            }
            for name, (schema, semantic_mask) in CONDITIONS.items()
            for env, _, infos in (states[name],)
        },
        "independence_checks": comparisons,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
