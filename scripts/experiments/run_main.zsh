#!/bin/zsh
# One policy at a time. Default: print the schedule only.
set -euo pipefail
cd "${0:A:h:h:h}"
mode="${1:---plan}"
if [[ "$mode" != "--plan" && "$mode" != "--execute" ]]; then
    print -u2 "Usage: zsh scripts/experiments/run_main.zsh [--plan|--execute]"
    exit 2
fi
root="${RUN_ROOT:-runs/main_fixed_budget}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for encoder in nodemlp gcn; do
    for seed in 42 43 44 45 46; do
        export PYTHONHASHSEED="$seed"
        args=(
            --encoder "$encoder" --seed "$seed"
            --total-simulator-steps 300000 --episode-steps 500
            --episodes-per-update 4 --environment-distribution random_host
            --observation-condition full
            --node-hidden-dim 128 --gcn-hidden-dim 128 --gcn-layers 2
            --node-embedding-dim 64 --context-dim 128 --head-hidden-dim 128
            --passive-memory-ttl 5 --analysis-memory-ttl 20
            --learning-rate 3e-4 --gamma 0.99 --gae-lambda 0.95
            --reward-scale 0.01 --entropy-coef 0.001 --target-kl 0.015
            --update-epochs 4 --minibatch-size 64 --clip-coef 0.2
            --value-coef 0.5 --max-grad-norm 0.5
            --device cpu --torch-threads 1
            --eval-every-updates 25 --periodic-eval-mode stochastic
            --periodic-eval-episodes 10 --periodic-eval-seed 20000
            --eval-episodes 30 --eval-seed 10000
            --print-every-updates 25 --run-dir "$root/$encoder"
        )
        print "encoder=$encoder seed=$seed mode=$mode output=$root/$encoder/seed_$seed"
        if [[ "$mode" == "--execute" ]]; then
            zsh -lic 'uvr python -m scripts.training.train "$@"' main-run "${args[@]}"
        fi
    done
done
