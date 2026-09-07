#!/usr/bin/env zsh

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  print -u2 "uv is not available on PATH."
  exit 127
fi

readonly LOG_DIR="runs/canonical_v1_encoder_matrix_logs"
readonly OFFICIAL_DIR="runs/canonical_v1_graphsage_random_host_official_mask_100k"
readonly SEMANTIC_DIR="runs/canonical_v1_graphsage_random_host_semantic_mask_100k"
mkdir -p "${LOG_DIR}"

typeset -a COMMON_ARGS
COMMON_ARGS=(
  --canonical-schema v1
  --canonical-variant action_aware
  --evidence-age-horizon 50
  --environment-distribution random_host
  --total-simulator-steps 100000
  --episode-steps 400
  --sage-hidden-dim 28
  --sage-layers 2
  --latent-dim 128
  --head-hidden-dim 128
  --reward-scale 0.01
  --learning-rate 3e-4
  --gamma 0.99
  --gae-lambda 0.95
  --update-epochs 4
  --minibatch-size 64
  --clip-coef 0.2
  --value-coef 0.5
  --entropy-coef 0.001
  --max-grad-norm 0.5
  --target-kl 0.015
  --device cpu
  --eval-every-updates 0
  --eval-episodes 30
  --eval-seed 10000
  --print-every-updates 50
)

run_complete() {
  local run_dir="$1"
  local seed="$2"
  local seed_dir="${run_dir}/seed_${seed}"
  [[ -f "${seed_dir}/checkpoint.pt" ]] \
    && [[ -f "${seed_dir}/training_summary.json" ]] \
    && [[ -f "${seed_dir}/evaluation_stochastic.json" ]] \
    && [[ -f "${seed_dir}/evaluation_deterministic.json" ]]
}

launch_run() {
  local label="$1"
  local run_dir="$2"
  local seed="$3"
  local semantic_mask="$4"
  local log_path="${LOG_DIR}/${label}_seed_${seed}.log"
  REPLY=""

  if run_complete "${run_dir}" "${seed}"; then
    print "SKIP complete ${label} seed=${seed}"
    return 0
  fi

  typeset -a mask_args
  mask_args=()
  if [[ "${semantic_mask}" == "on" ]]; then
    mask_args+=(--semantic-action-mask)
  fi

  print "START ${label} seed=${seed} log=${log_path}"
  (
    OMP_NUM_THREADS=5 \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH="${PWD}/CybORG:${PWD}/src:${PYTHONPATH:-}" \
      uv run python -m scripts.training.train_canonical_graphsage \
      "${COMMON_ARGS[@]}" \
      "${mask_args[@]}" \
      --seed "${seed}" \
      --run-dir "${run_dir}" \
      >"${log_path}" 2>&1
  ) &
  REPLY=$!
}

for seed in 42 43 44 45 46; do
  launch_run sage_official "${OFFICIAL_DIR}" "${seed}" off
  official_pid="${REPLY}"
  if [[ -n "${official_pid}" ]]; then
    if wait "${official_pid}"; then
      print "DONE sage_official seed=${seed}"
    else
      print -u2 "FAILED sage_official seed=${seed}"
      exit 1
    fi
  fi

  # Run only one policy process at a time to limit sustained CPU load and heat.
  launch_run sage_semantic "${SEMANTIC_DIR}" "${seed}" on
  semantic_pid="${REPLY}"
  if [[ -n "${semantic_pid}" ]]; then
    if wait "${semantic_pid}"; then
      print "DONE sage_semantic seed=${seed}"
    else
      print -u2 "FAILED sage_semantic seed=${seed}"
      exit 1
    fi
  fi
done

print "GRAPHSAGE MATRIX COMPLETE"
