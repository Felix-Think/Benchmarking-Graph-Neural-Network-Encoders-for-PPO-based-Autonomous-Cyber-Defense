#!/usr/bin/env zsh

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  print -u2 "uv is not available on PATH."
  exit 127
fi

readonly LOG_DIR="runs/canonical_v1_encoder_matrix_logs"
mkdir -p "${LOG_DIR}"

evaluate_gate() {
  local label="$1"
  local run_dir="$2"
  local policy_type="$3"
  local seed="$4"
  local seed_dir="${run_dir}/seed_${seed}"
  local output="${seed_dir}/observation_gate_static_only_stochastic.json"
  local log="${LOG_DIR}/gate_${label}_seed_${seed}.log"

  if [[ -f "${output}" ]]; then
    print "SKIP gate ${label} seed=${seed}"
    return
  fi

  print "START gate ${label} seed=${seed}"
  OMP_NUM_THREADS=5 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PWD}/CybORG:${PWD}/src:${PYTHONPATH:-}" \
    uv run \
    python -m scripts.evaluation.evaluate_canonical_observation_gate \
    --run-dir "${run_dir}" \
    --seed "${seed}" \
    --eval-episodes 30 \
    --eval-seed 10000 \
    --stochastic-action-seed 10000 \
    --policy-type "${policy_type}" \
    --checkpoint-path "${seed_dir}/checkpoint.pt" \
    --transform static_only \
    --mode stochastic \
    >"${log}" 2>&1
  print "DONE gate ${label} seed=${seed}"
}

for seed in 42 43 44 45 46; do
  evaluate_gate F-O runs/canonical_flat_random_pivot_100k flat "${seed}" &
  flat_pid=$!
  evaluate_gate G-O runs/canonical_gcn_random_pivot_100k gcn "${seed}" &
  gcn_pid=$!
  wait "${flat_pid}"
  wait "${gcn_pid}"
done

for seed in 42 43 44 45 46; do
  evaluate_gate F-S runs/canonical_v1_flat_random_host_semantic_mask_100k flat "${seed}" &
  flat_pid=$!
  evaluate_gate G-S runs/canonical_v1_gcn_random_host_semantic_mask_100k gcn "${seed}" &
  gcn_pid=$!
  wait "${flat_pid}"
  wait "${gcn_pid}"
done

print "STATIC GATES COMPLETE"
