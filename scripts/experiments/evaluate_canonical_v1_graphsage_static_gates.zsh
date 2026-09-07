#!/usr/bin/env zsh

set -euo pipefail

evaluate_gate() {
  local label="$1"
  local run_dir="$2"
  local seed="$3"
  local seed_dir="${run_dir}/seed_${seed}"
  local output="${seed_dir}/observation_gate_static_only_stochastic.json"
  if [[ -f "${output}" ]]; then
    print "SKIP gate ${label} seed=${seed}"
    return 0
  fi
  PYTHONPATH="${PWD}/CybORG:${PWD}/src:${PYTHONPATH:-}" \
    uv run python -m scripts.evaluation.evaluate_canonical_observation_gate \
    --run-dir "${run_dir}" \
    --seed "${seed}" \
    --policy-type graphsage \
    --checkpoint-path "${seed_dir}/checkpoint.pt" \
    --transform static_only \
    --mode stochastic \
    --eval-episodes 30 \
    --eval-seed 10000 \
    --stochastic-action-seed 10000
}

for seed in 42 43 44 45 46; do
  evaluate_gate \
    S-O runs/canonical_v1_graphsage_random_host_official_mask_100k "${seed}"
  evaluate_gate \
    S-S runs/canonical_v1_graphsage_random_host_semantic_mask_100k "${seed}"
done

print "GRAPHSAGE STATIC GATES COMPLETE"
