"""
experiment_true.py
==================
KernelSHAP with full X_train as background — "true" SHAP ground truth.

- Raw pipeline (train_model from models.py; explainers see raw feature space)
- OrdinalEncoder → all-numerical background for KernelSHAP
- Background: full X_train (no capping)
- Budget: 256 for ACS/Diabetes (exact 2^8), 1024 for GMSC/German
- Chunked test set for large datasets (ACS/GMSC: N_CHUNKS=100)
- Saves: results_true/{ds}_{model}_{split}_0_fulltrain[_c{idx}]_sv.pkl
"""

import sys
import os
os.environ["SEGMENT_DISABLE"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import argparse
import time
import pickle
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from utils import set_global_seed, clean_memory, load_data
from models import train_model


BUDGET_MAP     = {'german': 1024, 'diabetes': 256, 'acs': 256, 'gmsc': 1024}
MODEL_SEED     = 0
EXPLAINER_SEED = 42


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def make_predict_fn_enc(clf, raw_cols, num_features, cat_features, ord_enc):
    """Returns f(X) -> (N,) class-1 probabilities for KernelSHAP.
    Accepts ordinal-encoded numpy array; inverse_transform restores original
    strings before passing to the pipeline.
    """
    def f(X_arr):
        X_df = pd.DataFrame(X_arr, columns=raw_cols)
        if cat_features:
            X_df[cat_features] = ord_enc.inverse_transform(
                np.round(X_df[cat_features].values).astype(int))
        X_df[num_features] = X_df[num_features].astype(float)
        return clf.predict_proba(X_df)[:, 1]
    return f


def main():
    parser = argparse.ArgumentParser(
        description="KernelSHAP with full X_train background (raw pipeline)"
    )
    parser.add_argument('--dataset',    type=str, required=True,
                        choices=['german', 'diabetes', 'acs', 'gmsc'])
    parser.add_argument('--model',      type=str, required=True,
                        choices=['dt', 'rf', 'xgb', 'mlp', 'ftt'])
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--n_chunks',   type=int, default=1,
                        help='Number of test-set chunks for parallelism')
    parser.add_argument('--chunk_idx',  type=int, default=0,
                        help='Which chunk to process (0-indexed)')
    parser.add_argument('--outdir',     type=str, default='.',
                        help='Base output directory')
    args = parser.parse_args()

    DATASET    = args.dataset
    MODEL      = args.model
    SPLIT_SEED = args.split_seed
    N_CHUNKS   = args.n_chunks
    CHUNK_IDX  = args.chunk_idx
    OUTDIR     = args.outdir
    BUDGET     = BUDGET_MAP[DATASET]

    CHUNK_SUFFIX = f'_c{CHUNK_IDX}' if N_CHUNKS > 1 else ''

    os.makedirs(os.path.join(OUTDIR, 'log_true'), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, 'results_true'), exist_ok=True)

    save_path = os.path.join(OUTDIR, 'results_true',
                             f'{DATASET}_{MODEL}_{SPLIT_SEED}_0_fulltrain{CHUNK_SUFFIX}_sv.pkl')
    if os.path.exists(save_path):
        print(f"[SKIP] Already exists: {save_path}")
        return

    log_filename = os.path.join(OUTDIR, 'log_true',
                                f'{DATASET}_{MODEL}_{SPLIT_SEED}_0_true.log')
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout
    print(f"=== experiment_true.py (KernelSHAP + raw pipeline + full BG) ===")
    print(f"Dataset: {DATASET} | Model: {MODEL} | Split: {SPLIT_SEED} | ModelSeed: {MODEL_SEED}")
    print(f"Budget: {BUDGET} | N_CHUNKS: {N_CHUNKS} | Chunk: {CHUNK_IDX}")

    t_total = time.time()

    # 1. Load data
    X, Y = load_data(DATASET)
    print(f"Data loaded: {X.shape}")

    # 2. Detect feature types
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = [c for c in X.columns if c not in cat_features]
    raw_cols     = X.columns.tolist()
    print(f"Raw features: {len(raw_cols)} (num={len(num_features)}, cat={len(cat_features)})")

    # 3. Split (on original X before any encoding)
    set_global_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        if i == SPLIT_SEED:
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test  = X.iloc[test_idx].reset_index(drop=True)
            Y_train = Y.iloc[train_idx].reset_index(drop=True)
            break
    print(f"Split {SPLIT_SEED}: train={len(X_train)}, test={len(X_test)}")

    # 4. OrdinalEncoder — KernelSHAP needs all-numerical background
    #    (shap does numpy arithmetic on background rows; string cats would fail)
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cat_features:
        ord_enc.fit(X_train[cat_features])
        X_train_enc = X_train.copy()
        X_train_enc[cat_features] = ord_enc.transform(X_train[cat_features]).astype(int)
        X_test_enc = X_test.copy()
        X_test_enc[cat_features] = ord_enc.transform(X_test[cat_features]).astype(int)
    else:
        X_train_enc = X_train.copy()
        X_test_enc  = X_test.copy()

    # 5. Chunk the test set
    n_test_total = len(X_test_enc)
    chunk_size   = (n_test_total + N_CHUNKS - 1) // N_CHUNKS
    chunk_start  = CHUNK_IDX * chunk_size
    chunk_end    = min(chunk_start + chunk_size, n_test_total)
    X_test_chunk = X_test_enc.iloc[chunk_start:chunk_end].reset_index(drop=True)
    print(f"Explaining {len(X_test_chunk)} test instances "
          f"(chunk {CHUNK_IDX}/{N_CHUNKS}: [{chunk_start}:{chunk_end}])")

    if len(X_test_chunk) == 0:
        print(f"[SKIP] Empty chunk {CHUNK_IDX}")
        return

    # 6. Train model
    print(f"\nTraining {MODEL} (seed={MODEL_SEED})...")
    t_train = time.time()
    clf = train_model(X_train, Y_train, MODEL, MODEL_SEED, num_features, cat_features)
    print(f"Training done in {(time.time()-t_train)/60:.2f} min")

    # 7. Predict function: ordinal-encoded numpy → inverse_transform → pipeline
    f = make_predict_fn_enc(clf, raw_cols, num_features, cat_features, ord_enc)

    # 8. Background: full X_train_enc (no capping)
    bg_arr = X_train_enc.values.astype(float)
    print(f"\nBackground: full X_train_enc ({len(bg_arr)} instances, {bg_arr.shape[1]} features)")
    print(f"Coalitions per test instance: {BUDGET} × {len(bg_arr)} = {BUDGET * len(bg_arr):,} evals")

    # 9. KernelSHAP
    print(f"Running KernelSHAP on {len(X_test_chunk)} test instances...")
    t_shap = time.time()
    explainer = shap.KernelExplainer(f, bg_arr, link='identity', seed=EXPLAINER_SEED)
    sv_raw = explainer.shap_values(X_test_chunk.values, nsamples=BUDGET, gc_collect=True)
    print(f"KernelSHAP done in {(time.time()-t_shap)/60:.2f} min")

    if isinstance(sv_raw, list):
        sv = sv_raw[1]
        ev = explainer.expected_value
        base_val = float(ev[1] if hasattr(ev, '__len__') else ev)
    else:
        sv = sv_raw
        base_val = float(explainer.expected_value)

    err = np.abs(sv.sum(axis=1) + base_val - f(X_test_chunk.values)).mean()
    print(f"Efficiency axiom mean error: {err:.6f}")
    print(f"sv shape: {sv.shape} | base_val: {base_val:.4f} | chunk_start={chunk_start}")

    # 10. Save
    shap_exp = shap.Explanation(
        values=sv, base_values=base_val,
        data=X_test_chunk.values, feature_names=raw_cols
    )
    with open(save_path, 'wb') as fout:
        pickle.dump(shap_exp, fout, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved → {save_path}")

    print(f"\nTotal time: {(time.time()-t_total)/60:.2f} min")
    del clf, explainer, sv_raw, shap_exp
    clean_memory()


if __name__ == "__main__":
    main()
