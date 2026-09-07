#!/usr/bin/env zsh

set -euo pipefail

readonly LOG_DIR="runs/canonical_v1_counterfactual_gate_logs"
readonly EVAL_EPISODES="${EVAL_EPISODES:-30}"
mkdir -p "${LOG_DIR}"

complete() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  .venv/bin/python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if int(d["protocol"]["episodes"]) == int(sys.argv[2]) else 1)' \
    "${path}" "${EVAL_EPISODES}"
}

evaluate_run() {
  local label="$1"
  local run_dir="$2"
  local policy_type="$3"
  local seed="$4"
  local seed_dir="${run_dir}/seed_${seed}"
  local shuffle_output="${seed_dir}/counterfactual_shuffle_hosts_stochastic.json"
  local shift_output="${seed_dir}/counterfactual_time_shift_lag5_stochastic.json"
  local zero_output="${seed_dir}/counterfactual_zero_evidence_stochastic.json"
  local log="${LOG_DIR}/${label}_seed_${seed}.log"

  if complete "${shuffle_output}" \
     && complete "${shift_output}" \
     && complete "${zero_output}"; then
    print "SKIP complete ${label} seed=${seed}"
    return
  fi

  print "START ${label} seed=${seed}"
  : >"${log}"
  for transform in shuffle_hosts time_shift zero_evidence; do
    OMP_NUM_THREADS=5 \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH="${PWD}/CybORG:${PWD}/src:${PYTHONPATH:-}" \
      uv run python -m scripts.evaluation.evaluate_dynamic_evidence_counterfactual \
      --run-dir "${run_dir}" \
      --seed "${seed}" \
      --policy-type "${policy_type}" \
      --checkpoint-path "${seed_dir}/checkpoint.pt" \
      --transform "${transform}" \
      --time-shift-lag 5 \
      --intervention-seed 40000 \
      --eval-episodes "${EVAL_EPISODES}" \
      --eval-seed 10000 \
      --stochastic-action-seed 10000 \
      --overwrite \
      >>"${log}" 2>&1
  done
  print "DONE ${label} seed=${seed}"
}

for seed in 42 43 44 45 46; do
  evaluate_run F-O runs/canonical_flat_random_pivot_100k flat "${seed}"
  evaluate_run G-O runs/canonical_gcn_random_pivot_100k gcn "${seed}"
  evaluate_run S-O runs/canonical_v1_graphsage_random_host_official_mask_100k graphsage "${seed}"
done

print "COUNTERFACTUAL GATES COMPLETE episodes=${EVAL_EPISODES}"
