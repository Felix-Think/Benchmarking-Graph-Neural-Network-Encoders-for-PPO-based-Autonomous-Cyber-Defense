#!/usr/bin/env zsh

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  print -u2 "uv is not available on PATH."
  exit 127
fi

readonly LOG_DIR="runs/canonical_v1_encoder_matrix_logs"
mkdir -p "${LOG_DIR}"

typeset -a TRAINING_SEEDS
TRAINING_SEEDS=(42 43 44 45 46)

typeset -a COMMON_ARGS
COMMON_ARGS=(
  --canonical-schema v1
  --canonical-variant action_aware
  --evidence-age-horizon 50
  --environment-distribution random_host
  --total-simulator-steps 100000
  --episode-steps 400
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
  local module="$2"
  local run_dir="$3"
  local seed="$4"
  local encoder_flag="$5"
  local encoder_value="$6"
  local semantic_mask="$7"
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
      uv run python -m "${module}" \
      "${COMMON_ARGS[@]}" \
      "${encoder_flag}" "${encoder_value}" \
      "${mask_args[@]}" \
      --seed "${seed}" \
      --run-dir "${run_dir}" \
      >"${log_path}" 2>&1
  ) &
  REPLY=$!
}

wait_pair() {
  local first_pid="$1"
  local first_label="$2"
  local second_pid="$3"
  local second_label="$4"
  local failed=0

  if [[ -n "${first_pid}" ]]; then
    if ! wait "${first_pid}"; then
      print -u2 "FAILED ${first_label}"
      failed=1
    else
      print "DONE ${first_label}"
    fi
  fi
  if [[ -n "${second_pid}" ]]; then
    if ! wait "${second_pid}"; then
      print -u2 "FAILED ${second_label}"
      failed=1
    else
      print "DONE ${second_label}"
    fi
  fi
  return "${failed}"
}

run_pair() {
  local mask="$1"
  local seed="$2"
  local flat_dir="$3"
  local gcn_dir="$4"
  local suffix="$5"

  launch_run \
    "flat_${suffix}" \
    scripts.training.train_canonical_flat \
    "${flat_dir}" \
    "${seed}" \
    --encoder-hidden-dim \
    128 \
    "${mask}"
  local flat_pid="${REPLY}"

  launch_run \
    "gcn_${suffix}" \
    scripts.training.train_canonical_gcn \
    "${gcn_dir}" \
    "${seed}" \
    --gcn-hidden-dim \
    28 \
    "${mask}"
  local gcn_pid="${REPLY}"

  wait_pair \
    "${flat_pid}" "flat_${suffix} seed=${seed}" \
    "${gcn_pid}" "gcn_${suffix} seed=${seed}"
}

# Native primary. Completed runs are detected and skipped.
for seed in "${TRAINING_SEEDS[@]}"; do
  run_pair \
    off \
    "${seed}" \
    runs/canonical_flat_random_pivot_100k \
    runs/canonical_gcn_random_pivot_100k \
    official
done

# Semantic action-mask ablation. Completed runs are detected and skipped.
for seed in "${TRAINING_SEEDS[@]}"; do
  run_pair \
    on \
    "${seed}" \
    runs/canonical_v1_flat_random_host_semantic_mask_100k \
    runs/canonical_v1_gcn_random_host_semantic_mask_100k \
    semantic
done

print "MATRIX COMPLETE"
