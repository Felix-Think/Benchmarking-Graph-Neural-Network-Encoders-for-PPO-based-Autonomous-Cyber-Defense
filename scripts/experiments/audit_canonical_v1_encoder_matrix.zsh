#!/usr/bin/env zsh

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  print -u2 "uv is not available on PATH."
  exit 127
fi

readonly LOG_DIR="runs/canonical_v1_encoder_matrix_logs"
mkdir -p "${LOG_DIR}"

audit_run() {
  local label="$1"
  local run_dir="$2"
  local seed="$3"
  local prefix="results/audit_v1_matrix_${label}_seed_${seed}"
  local summary="${prefix}_summary.json"
  local log="${LOG_DIR}/audit_${label}_seed_${seed}.log"

  if [[ -f "${summary}" ]]; then
    print "SKIP audit ${label} seed=${seed}"
    return
  fi

  print "START audit ${label} seed=${seed}"
  OMP_NUM_THREADS=5 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PWD}/CybORG:${PWD}/src:${PYTHONPATH:-}" \
    uv run python -m scripts.analysis.audit_dynamic_evidence_action_reward \
    --run-dir "${run_dir}" \
    --seed "${seed}" \
    --episodes 3 \
    --eval-seed 10000 \
    --action-seed 10000 \
    --output-prefix "${prefix}" \
    >"${log}" 2>&1
  print "DONE audit ${label} seed=${seed}"
}

for seed in 42 43 44 45 46; do
  audit_run F-O runs/canonical_flat_random_pivot_100k "${seed}" &
  flat_pid=$!
  audit_run G-O runs/canonical_gcn_random_pivot_100k "${seed}" &
  gcn_pid=$!
  wait "${flat_pid}"
  wait "${gcn_pid}"
done

for seed in 42 43 44 45 46; do
  audit_run F-S runs/canonical_v1_flat_random_host_semantic_mask_100k "${seed}" &
  flat_pid=$!
  audit_run G-S runs/canonical_v1_gcn_random_host_semantic_mask_100k "${seed}" &
  gcn_pid=$!
  wait "${flat_pid}"
  wait "${gcn_pid}"
done

print "MATRIX AUDIT COMPLETE"
