"""
experiment_proba.py
===================
Save model prediction probabilities for the full test set.

Model probability is independent of background sampling method and BG seed,
so it only needs to be computed once per (dataset, model, model_seed).

Saves: results_tree/{DATASET}_{MODEL}_{SPLIT_SEED}_{MODEL_SEED}_proba.pkl
"""

import sys
import os
os.environ["SEGMENT_DISABLE"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import set_global_seed, clean_memory, load_data
from experiment_sampling import train_clf_ohe, make_predict_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str, required=True,
                        choices=['german', 'diabetes', 'acs', 'gmsc'])
    parser.add_argument('--model',          type=str, required=True,
                        choices=['dt', 'rf', 'xgb', 'mlp', 'ftt'])
    parser.add_argument('--split_seed',     type=int, default=0)
    parser.add_argument('--model_seed_idx', type=int, default=0,
                        help='Index into model_seeds list [0,21,42,63,84]')
    parser.add_argument('--outdir',         type=str, default='.',
                        help='Base output directory')
    args = parser.parse_args()

    DATASET    = args.dataset
    MODEL      = args.model
    SPLIT_SEED = args.split_seed
    model_seeds = [0, 21, 42, 63, 84]
    MODEL_SEED  = model_seeds[args.model_seed_idx]
    OUTDIR      = args.outdir

    save_path = os.path.join(OUTDIR, 'proba',
                             f'{DATASET}_{MODEL}_{SPLIT_SEED}_{MODEL_SEED}_proba.pkl')

    if os.path.exists(save_path):
        print(f'[SKIP] Already exists: {save_path}')
        return

    os.makedirs(os.path.join(OUTDIR, 'proba'), exist_ok=True)

    print(f'=== experiment_proba.py ===')
    print(f'Dataset: {DATASET} | Model: {MODEL} | Split: {SPLIT_SEED} | ModelSeed: {MODEL_SEED}')

    # 1. Load data
    X, Y = load_data(DATASET)
    print(f'Data loaded: {X.shape}')

    # 2. OHE + sanitize
    X_ohe = pd.get_dummies(X, drop_first=False).astype(float)
    X_ohe.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X_ohe.columns]
    ohe_cols = X_ohe.columns.tolist()
    print(f'OHE features: {len(ohe_cols)}')

    # 3. Split
    set_global_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        if i == SPLIT_SEED:
            X_train = X_ohe.iloc[train_idx].reset_index(drop=True)
            X_test  = X_ohe.iloc[test_idx].reset_index(drop=True)
            Y_train = Y.iloc[train_idx].reset_index(drop=True)
            break
    print(f'Split {SPLIT_SEED}: train={len(X_train)}, test={len(X_test)}')

    # 4. Train model
    print(f'Training {MODEL} (seed={MODEL_SEED})...')
    clf = train_clf_ohe(X_train, Y_train, MODEL, MODEL_SEED, ohe_cols)

    # 5. Predict on full test set
    f = make_predict_fn(clf, ohe_cols, MODEL)
    proba = f(X_test.values)
    print(f'Proba shape: {proba.shape}')

    with open(save_path, 'wb') as fout:
        pickle.dump(proba, fout, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved → {save_path}')

    del clf, proba
    clean_memory()


if __name__ == '__main__':
    main()
