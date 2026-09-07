#!/usr/bin/env zsh

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  print -u2 "uv is not available on PATH."
  exit 127
fi

readonly LOG_DIR="runs/canonical_v1_persistent_degradation_curve_logs"
readonly EVAL_EPISODES="${EVAL_EPISODES:-30}"
mkdir -p "${LOG_DIR}"

evaluation_complete() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  .venv/bin/python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if int(d["protocol"]["episodes"]) == int(sys.argv[2]) and d["protocol"]["mask_process"] == "persistent_fixed_budget_dynamic_host" else 1)' \
    "${path}" "${EVAL_EPISODES}"
}

evaluate_run() {
  local label="$1"
  local run_dir="$2"
  local policy_type="$3"
  local seed="$4"
  local seed_dir="${run_dir}/seed_${seed}"
  local log_path="${LOG_DIR}/${label}_seed_${seed}.log"

  if evaluation_complete "${seed_dir}/persistent_dynamic_host_retention_q075_stochastic.json" \
     && evaluation_complete "${seed_dir}/persistent_dynamic_host_retention_q050_stochastic.json" \
     && evaluation_complete "${seed_dir}/persistent_dynamic_host_retention_q025_stochastic.json"; then
    print "SKIP complete ${label} seed=${seed}"
    return
  fi

  print "START ${label} seed=${seed} log=${log_path}"
  OMP_NUM_THREADS=5 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PWD}/CybORG:${PWD}/src:${PYTHONPATH:-}" \
    uv run python -m scripts.evaluation.evaluate_dynamic_host_retention \
    --run-dir "${run_dir}" \
    --seed "${seed}" \
    --policy-type "${policy_type}" \
    --checkpoint-path "${seed_dir}/checkpoint.pt" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-seed 10000 \
    --stochastic-action-seed 10000 \
    --mask-seed 30000 \
    --mask-process persistent \
    --retention-levels 0.75 0.50 0.25 \
    --overwrite \
    >"${log_path}" 2>&1
  print "DONE ${label} seed=${seed}"
}

# One policy evaluation process at a time to control CPU temperature.
for seed in 42 43 44 45 46; do
  evaluate_run F-O runs/canonical_flat_random_pivot_100k flat "${seed}"
  evaluate_run G-O runs/canonical_gcn_random_pivot_100k gcn "${seed}"
  evaluate_run S-O runs/canonical_v1_graphsage_random_host_official_mask_100k graphsage "${seed}"
done

print "PERSISTENT DEGRADATION CURVE COMPLETE episodes=${EVAL_EPISODES}"
