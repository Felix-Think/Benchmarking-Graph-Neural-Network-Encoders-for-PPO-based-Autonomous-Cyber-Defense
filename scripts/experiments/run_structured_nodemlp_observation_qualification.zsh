#!/usr/bin/env zsh
set -euo pipefail

function uvr() {
  chflags -R nohidden .venv
  uv run "$@"
}

RUN_ROOT=${RUN_ROOT:-runs/structured_nodemlp_observation_qualification_50k}
TOTAL_SIMULATOR_STEPS=${TOTAL_SIMULATOR_STEPS:-50000}
EPISODES_PER_UPDATE=${EPISODES_PER_UPDATE:-4}
ENVIRONMENT_DISTRIBUTION=${ENVIRONMENT_DISTRIBUTION:-official}
SEMANTIC_ACTION_MASK=${SEMANTIC_ACTION_MASK:-0}
PASSIVE_MEMORY_TTL=${PASSIVE_MEMORY_TTL:-20}
ANALYSIS_MEMORY_TTL=${ANALYSIS_MEMORY_TTL:-20}
RUN_UNTRAINED=${RUN_UNTRAINED:-1}
RUN_ANALYZER=${RUN_ANALYZER:-1}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-0}
EVAL_EVERY_UPDATES=${EVAL_EVERY_UPDATES:-25}
PERIODIC_EVAL_EPISODES=${PERIODIC_EVAL_EPISODES:-10}
PERIODIC_EVAL_SEED=${PERIODIC_EVAL_SEED:-20000}
EVAL_EPISODES=${EVAL_EPISODES:-30}
EVAL_SEED=${EVAL_SEED:-10000}
PRINT_EVERY_UPDATES=${PRINT_EVERY_UPDATES:-10}
TORCH_CPU_THREADS=${TORCH_CPU_THREADS:-1}
OUTPUT_JSON=${OUTPUT_JSON:-results/structured_nodemlp_observation_qualification.json}
OUTPUT_REPORT=${OUTPUT_REPORT:-results/STRUCTURED_NODEMLP_OBSERVATION_QUALIFICATION.md}
SEED_LIST=${SEEDS:-"42"}
CONDITION_LIST=${CONDITIONS:-"full no_evidence"}
seeds=(${=SEED_LIST})
conditions=(${=CONDITION_LIST})
semantic_mask_args=()
if (( SEMANTIC_ACTION_MASK )); then
  semantic_mask_args=(--semantic-action-mask)
fi

for seed in ${seeds[@]}; do
  export PYTHONHASHSEED="${seed}"
  export OMP_NUM_THREADS="${TORCH_CPU_THREADS}"
  export MKL_NUM_THREADS="${TORCH_CPU_THREADS}"
  for condition in ${conditions[@]}; do
    echo "condition=${condition} seed=${seed}"
    uvr python -m scripts.training.train_canonical_structured_nodemlp \
      --canonical-schema v2 \
      --canonical-variant action_aware \
      --passive-memory-ttl "${PASSIVE_MEMORY_TTL}" \
      --analysis-memory-ttl "${ANALYSIS_MEMORY_TTL}" \
      ${semantic_mask_args[@]} \
      --observation-condition "${condition}" \
      --environment-distribution "${ENVIRONMENT_DISTRIBUTION}" \
      --total-simulator-steps "${TOTAL_SIMULATOR_STEPS}" \
      --episode-steps 500 \
      --episodes-per-update "${EPISODES_PER_UPDATE}" \
      --reward-scale 0.01 \
      --learning-rate 3e-4 \
      --gamma 0.99 \
      --gae-lambda 0.95 \
      --update-epochs 4 \
      --minibatch-size 64 \
      --clip-coef 0.2 \
      --value-coef 0.5 \
      --entropy-coef 0.001 \
      --max-grad-norm 0.5 \
      --target-kl 0.015 \
      --seed "${seed}" \
      --device cpu \
      --eval-every-updates "${EVAL_EVERY_UPDATES}" \
      --periodic-eval-episodes "${PERIODIC_EVAL_EPISODES}" \
      --periodic-eval-seed "${PERIODIC_EVAL_SEED}" \
      --periodic-eval-mode stochastic \
      --eval-episodes "${EVAL_EPISODES}" \
      --eval-seed "${EVAL_SEED}" \
      --print-every-updates "${PRINT_EVERY_UPDATES}" \
      --run-dir "${RUN_ROOT}/nodemlp_${condition}"

    if [[ "${condition}" == "full" ]] && (( RUN_UNTRAINED )); then
      uvr python -m scripts.evaluation.evaluate_untrained_canonical_flat \
        --run-dir "${RUN_ROOT}/nodemlp_full" \
        --seed "${seed}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --eval-seed "${EVAL_SEED}" \
        --episode-steps 500 \
        --canonical-schema v2 \
        --canonical-variant action_aware \
        --passive-memory-ttl "${PASSIVE_MEMORY_TTL}" \
        --analysis-memory-ttl "${ANALYSIS_MEMORY_TTL}" \
        ${semantic_mask_args[@]} \
        --environment-distribution "${ENVIRONMENT_DISTRIBUTION}" \
        --observation-condition full \
        --policy-type structured_nodemlp \
        --mode stochastic \
        --output-prefix untrained_full
    fi

    if (( COOLDOWN_SECONDS > 0 )); then
      echo "cooldown_seconds=${COOLDOWN_SECONDS}"
      sleep "${COOLDOWN_SECONDS}"
    fi
  done
done

if (( RUN_ANALYZER )); then
  uvr python -m scripts.analysis.analyze_observation_qualification \
    --run-root "${RUN_ROOT}" \
    --arm-prefix nodemlp \
    --seeds ${seeds[@]} \
    --output "${OUTPUT_JSON}" \
    --report "${OUTPUT_REPORT}"
fi
