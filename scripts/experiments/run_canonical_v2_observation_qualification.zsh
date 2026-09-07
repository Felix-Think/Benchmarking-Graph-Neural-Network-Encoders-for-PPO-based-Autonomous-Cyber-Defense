#!/usr/bin/env zsh
set -euo pipefail

# Match the interactive `uvr` alias while remaining executable as a script.
function uvr() {
  chflags -R nohidden .venv
  uv run "$@"
}

RUN_ROOT=${RUN_ROOT:-runs/canonical_v2_observation_qualification_300k}
TOTAL_SIMULATOR_STEPS=${TOTAL_SIMULATOR_STEPS:-300000}
EPISODES_PER_UPDATE=${EPISODES_PER_UPDATE:-4}
OUTPUT_JSON=${OUTPUT_JSON:-results/canonical_v2_observation_qualification.json}
OUTPUT_REPORT=${OUTPUT_REPORT:-results/CANONICAL_V2_OBSERVATION_QUALIFICATION.md}
SEED_LIST=${SEEDS:-"42 43 44 45 46"}
CONDITION_LIST=${CONDITIONS:-"full no_evidence"}
seeds=(${=SEED_LIST})
conditions=(${=CONDITION_LIST})

for condition in ${conditions[@]}; do
  for seed in ${seeds[@]}; do
    echo "condition=${condition} seed=${seed}"
    uvr python -m scripts.training.train_canonical_flat \
      --canonical-schema v2 \
      --canonical-variant action_aware \
      --passive-memory-ttl 20 \
      --analysis-memory-ttl 20 \
      --observation-condition "${condition}" \
      --environment-distribution official \
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
      --eval-every-updates 25 \
      --periodic-eval-episodes 10 \
      --periodic-eval-seed 20000 \
      --periodic-eval-mode stochastic \
      --eval-episodes 30 \
      --eval-seed 10000 \
      --print-every-updates 10 \
      --run-dir "${RUN_ROOT}/flat_${condition}"

    if [[ "${condition}" == "full" ]]; then
      uvr python -m scripts.evaluation.evaluate_untrained_canonical_flat \
        --run-dir "${RUN_ROOT}/flat_full" \
        --seed "${seed}" \
        --eval-episodes 30 \
        --eval-seed 10000 \
        --episode-steps 500 \
        --canonical-schema v2 \
        --canonical-variant action_aware \
        --passive-memory-ttl 20 \
        --analysis-memory-ttl 20 \
        --environment-distribution official \
        --observation-condition full \
        --mode stochastic \
        --output-prefix untrained_full
    fi
  done
done

uvr python -m scripts.analysis.analyze_observation_qualification \
  --run-root "${RUN_ROOT}" \
  --seeds ${seeds[@]} \
  --output "${OUTPUT_JSON}" \
  --report "${OUTPUT_REPORT}"
