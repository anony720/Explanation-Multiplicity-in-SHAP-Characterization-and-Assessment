#!/bin/bash
# =============================================================
# Run all SHAP sensitivity experiments
# Usage: bash run.sh
# =============================================================

set -e

# Environment setup
eval "$(conda shell.bash hook)"
conda activate shap

export PYTORCH_ALLOC_CONF=max_split_size_mb:128

# Create log directory
mkdir -p log results

# Experiment parameters
DATASETS="acs german diabetes"
MODELS="dt rf xgb ftt mlp tabpfn"
TOTAL_CHUNKS=15

echo "============================================"
echo "SHAP Sensitivity Analysis - Full Experiment"
echo "Datasets: $DATASETS"
echo "Models:   $MODELS"
echo "Seeds:    5 split x 5 model x 5 explainer"
echo "Chunks:   $TOTAL_CHUNKS per combination"
echo "============================================"

for DATASET in $DATASETS; do
  for MODEL in $MODELS; do
    for SPLIT_SEED in 0 1 2 3 4; do
      for MODEL_SEED_IDX in 0 1 2 3 4; do
        for EXPLAINER_SEED_IDX in 0 1 2 3 4; do
          for CHUNK_IDX in $(seq 0 $((TOTAL_CHUNKS - 1))); do

            echo "[RUN] $DATASET | $MODEL | split=$SPLIT_SEED | model_seed=$MODEL_SEED_IDX | exp_seed=$EXPLAINER_SEED_IDX | chunk=$CHUNK_IDX/$TOTAL_CHUNKS"

            python -u experiment.py \
              --dataset "$DATASET" \
              --model "$MODEL" \
              --split_seed "$SPLIT_SEED" \
              --model_seed_idx "$MODEL_SEED_IDX" \
              --explainer_seed_idx "$EXPLAINER_SEED_IDX" \
              --chunk_idx "$CHUNK_IDX" \
              --total_chunks "$TOTAL_CHUNKS"

          done
        done
      done
    done
  done
done

echo "============================================"
echo "All experiments completed."
echo "============================================"
