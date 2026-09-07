#!/usr/bin/env zsh

set -euo pipefail

audit_run() {
  local label="$1"
  local run_dir="$2"
  local seed="$3"
  local prefix="results/audit_v1_matrix_${label}_seed_${seed}"
  if [[ -f "${prefix}_summary.json" ]]; then
    print "SKIP audit ${label} seed=${seed}"
    return 0
  fi
  PYTHONPATH="${PWD}/CybORG:${PWD}/src:${PYTHONPATH:-}" \
    uv run python -m scripts.analysis.audit_dynamic_evidence_action_reward \
    --run-dir "${run_dir}" \
    --seed "${seed}" \
    --episodes 3 \
    --eval-seed 10000 \
    --action-seed 10000 \
    --output-prefix "${prefix}"
}

for seed in 42 43 44 45 46; do
  audit_run \
    S-O runs/canonical_v1_graphsage_random_host_official_mask_100k "${seed}"
  audit_run \
    S-S runs/canonical_v1_graphsage_random_host_semantic_mask_100k "${seed}"
done

print "GRAPHSAGE MATRIX AUDIT COMPLETE"
