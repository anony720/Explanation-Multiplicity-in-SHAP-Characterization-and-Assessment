#!/bin/bash
# run_sampling.sh
# Runs experiment_sampling.py for all datasets, models.
# Order: diabetes → german → acs (so smaller datasets finish first)
# Each call trains model once and runs all 5 methods × 5 seeds.

echo "=============================="
echo "Sampling run started: $(date)"
echo "=============================="

cd "$(dirname "$0")"

for DATASET in diabetes german acs; do
    for MODEL in dt rf xgb mlp ftt; do
        echo ""
        echo "--- sampling: $DATASET $MODEL ---"
        python experiment_sampling.py --dataset $DATASET --model $MODEL --split_seed 0
    done
done

echo ""
echo "=============================="
echo "All done: $(date)"
echo "=============================="
