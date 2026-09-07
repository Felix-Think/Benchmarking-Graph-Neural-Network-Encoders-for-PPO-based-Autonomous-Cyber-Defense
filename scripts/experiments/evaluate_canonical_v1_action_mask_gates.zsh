#!/usr/bin/env zsh

set -euo pipefail

readonly LOG_DIR="runs/canonical_v1_action_mask_gate_logs"
readonly EVAL_EPISODES="${EVAL_EPISODES:-30}"
mkdir -p "${LOG_DIR}"

complete() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  .venv/bin/python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if len(d["episode_returns"]) == int(sys.argv[2]) else 1)' \
    "${path}" "${EVAL_EPISODES}"
}

evaluate_run() {
  local label="$1"
  local run_dir="$2"
  local policy_type="$3"
  local seed="$4"
  local seed_dir="${run_dir}/seed_${seed}"
  local log="${LOG_DIR}/${label}_seed_${seed}.log"
  local zero_mask="${seed_dir}/observation_gate_zero_all_stochastic.json"
  local zero_no_mask="${seed_dir}/observation_gate_zero_all_no_action_mask_stochastic.json"
  local full_no_mask="${seed_dir}/observation_gate_full_no_action_mask_stochastic.json"

  if complete "${zero_mask}" \
     && complete "${zero_no_mask}" \
     && complete "${full_no_mask}"; then
    print "SKIP complete ${label} seed=${seed}"
    return
  fi

  print "START ${label} seed=${seed}"
  : >"${log}"
  for transform in zero_all zero_all_no_action_mask full_no_action_mask; do
    OMP_NUM_THREADS=5 \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH="${PWD}/CybORG:${PWD}/src:${PYTHONPATH:-}" \
      uv run python -m scripts.evaluation.evaluate_canonical_observation_gate \
      --run-dir "${run_dir}" \
      --seed "${seed}" \
      --eval-episodes "${EVAL_EPISODES}" \
      --eval-seed 10000 \
      --stochastic-action-seed 10000 \
      --policy-type "${policy_type}" \
      --checkpoint-path "${seed_dir}/checkpoint.pt" \
      --transform "${transform}" \
      --mode stochastic \
      >>"${log}" 2>&1
  done
  print "DONE ${label} seed=${seed}"
}

for seed in 42 43 44 45 46; do
  evaluate_run F-O runs/canonical_flat_random_pivot_100k flat "${seed}"
  evaluate_run G-O runs/canonical_gcn_random_pivot_100k gcn "${seed}"
  evaluate_run S-O runs/canonical_v1_graphsage_random_host_official_mask_100k graphsage "${seed}"
done

print "ACTION MASK GATES COMPLETE episodes=${EVAL_EPISODES}"
