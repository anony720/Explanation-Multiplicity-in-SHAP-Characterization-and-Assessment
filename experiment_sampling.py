"""
experiment_sampling.py
======================
KernelSHAP and PermutationSHAP with background sampling methods × 5 BG seeds.

- Pipeline-based preprocessing (OneHotEncoder inside pipeline; explainers see raw features)
- Methods: random, kmeans, pred_stratified, cte
- BG seeds: [0, 42, 84, 126, 168]
- Fixed explainer seed: 42
- Trains model ONCE per (dataset, model), then runs all method × seed combos
- Saves: results_tree/{ds}_{model}_{split}_{m_seed}_{method}_s{s_idx}_sv.pkl
          (PermutationSHAP: perm_{method})
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
from shap.utils._legacy import DenseData as ShapDenseData
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from goodpoints.compress import compress_kt

from utils import set_global_seed, clean_memory, load_data
from models import train_model


BG_SEEDS   = [0, 42, 84, 126, 168]
BG_SIZE    = 100          # default; overridden by --bg_size
NSAMPLES   = 1024
N_STRATA   = 5
METHODS    = ['random', 'kmeans', 'pred_stratified', 'cte']
MAX_TEST   = {'german': None, 'diabetes': None, 'acs': None, 'gmsc': None}
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


# ── Background selectors ──────────────────────────────────────────────────────

def bg_random(Xtr, Ytr, f, seed):
    return np.random.default_rng(seed).choice(len(Xtr), BG_SIZE, replace=False)


def bg_pred_stratified(Xtr, Ytr, f, seed):
    # Xtr: raw DataFrame (string cats ok); f handles numpy→DataFrame conversion internally
    preds = f(Xtr.values)
    bins  = np.percentile(preds, np.linspace(0, 100, N_STRATA + 1)); bins[-1] += 1e-9
    rng = np.random.default_rng(seed)
    idx = []
    per = BG_SIZE // N_STRATA
    for i in range(N_STRATA):
        s = np.where((preds >= bins[i]) & (preds < bins[i + 1]))[0]
        idx.extend(rng.choice(s, min(per, len(s)), replace=False).tolist())
    rem = BG_SIZE - len(idx)
    if rem > 0:
        pool = [j for j in range(len(Xtr)) if j not in set(idx)]
        idx.extend(rng.choice(pool, min(rem, len(pool)), replace=False).tolist())
    return np.array(idx[:BG_SIZE])


def bg_kmeans_shap(Xtr_enc, Ytr, f, seed):
    """shap.kmeans equivalent with custom random_state.
    Xtr_enc: ordinal-encoded DataFrame (all numerical).
    Returns shap DenseData (weighted centroids) — same format as shap.kmeans().
    Each centroid dimension is rounded to the nearest observed value in the training data
    (replicates shap.kmeans round_values=True default behavior).
    """
    X = Xtr_enc.values.astype(np.float64)
    km = KMeans(n_clusters=BG_SIZE, random_state=seed, n_init=10).fit(X)
    centers = km.cluster_centers_.copy()
    for i in range(BG_SIZE):
        for j in range(X.shape[1]):
            col = X[:, j]
            centers[i, j] = col[np.argmin(np.abs(col - centers[i, j]))]
    weights = 1.0 * np.bincount(km.labels_) / X.shape[0]
    return ShapDenseData(centers,
                         [str(i) for i in range(X.shape[1])],
                         None, weights)


def bg_cte(Xtr, Ytr, f, seed):
    # Xtr: ordinal-encoded DataFrame (all numerical); StandardScaler requires numerical input
    Xs = StandardScaler().fit_transform(Xtr.values.astype(np.float64)).astype(np.float64)
    d  = Xs.shape[1]
    kp = np.array([2.0 * d], dtype=np.float64)
    for g in range(8):
        idx = compress_kt(Xs, kernel_type=b'gaussian', k_params=kp, g=g, seed=seed)
        if len(idx) >= BG_SIZE:
            break
    return idx[:BG_SIZE]


BG_FNS = {
    'random':          bg_random,
    'kmeans':          bg_kmeans_shap,   # uses X_train_enc; returns shap DenseData
    'pred_stratified': bg_pred_stratified,
    'cte':             bg_cte,
}


# ── Predict functions ─────────────────────────────────────────────────────────

def make_predict_fn(clf, raw_cols, num_features, cat_features):
    """Returns f(X) -> (N,) class-1 probabilities for KernelSHAP and PermutationSHAP.
    Accepts raw DataFrame or numpy array; pipeline handles OHE internally.
    """
    def f(X_arr):
        if isinstance(X_arr, np.ndarray):
            X_df = pd.DataFrame(X_arr, columns=raw_cols)
            for c in cat_features:
                X_df[c] = X_df[c].astype(str)
            for c in num_features:
                X_df[c] = X_df[c].astype(float)
        else:
            X_df = X_arr
        return clf.predict_proba(X_df)[:, 1]
    return f



def make_predict_fn_enc(clf, raw_cols, num_features, cat_features, ord_enc):
    """Returns f(X) -> (N,) class-1 probabilities for PermutationSHAP.
    shap.maskers.Independent calls np.isclose() internally, which fails on string
    categoricals — so PermutationSHAP must also receive ordinal-encoded data.
    Accepts ordinal-encoded numpy array; inverse_transform restores original strings
    before passing to the pipeline.
    """
    def f(X_arr):
        X_df = pd.DataFrame(X_arr, columns=raw_cols)
        if cat_features:
            # np.round before astype(int) handles float centroids from shap.kmeans
            X_df[cat_features] = ord_enc.inverse_transform(
                np.round(X_df[cat_features].values).astype(int))
        X_df[num_features] = X_df[num_features].astype(float)
        return clf.predict_proba(X_df)[:, 1]
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    type=str, required=True,
                        choices=['german', 'diabetes', 'acs', 'gmsc'])
    parser.add_argument('--model',      type=str, required=True,
                        choices=['dt', 'rf', 'xgb', 'mlp', 'ftt'])
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--bg_size',    type=int, default=None,
                        help='Background size (overrides default BG_SIZE)')
    parser.add_argument('--n_chunks',   type=int, default=1,
                        help='Number of test-set chunks (for parallelism)')
    parser.add_argument('--chunk_idx',  type=int, default=0,
                        help='Which chunk to process (0-indexed)')
    parser.add_argument('--outdir',     type=str, default='.',
                        help='Base output directory (e.g. /scratch/hh3884/framework)')
    parser.add_argument('--model_seed_idx', type=int, default=0,
                        help='Index into model_seeds list [0,21,42,63,84]')
    parser.add_argument('--explainer',      type=str, default='kernel_shap',
                        choices=['kernel_shap', 'permutation_shap'],
                        help='Explainer to use: kernel_shap or permutation_shap')
    parser.add_argument('--only_method',   type=str, default=None,
                        choices=['random', 'kmeans', 'pred_stratified', 'cte'],
                        help='If set, run only this background method (overrides ACTIVE_METHODS)')

    args = parser.parse_args()

    DATASET    = args.dataset
    MODEL      = args.model
    SPLIT_SEED = args.split_seed
    MODEL_SEED_IDX = args.model_seed_idx
    model_seeds    = [0, 21, 42, 63, 84]
    MODEL_SEED     = model_seeds[MODEL_SEED_IDX]

    # ACS and Diabetes have d=8, so 2^8=256 exhausts all coalitions exactly
    NSAMPLES = 256 if DATASET in ('acs', 'diabetes') else 1024

    EXPLAINER_TYPE = args.explainer

    # Sub-experiment routing:
    #   model_seed_idx=0 → Sampling method experiment: all 3 methods, all 5 BG seeds, fixed model seed
    #   model_seed_idx>0 → Model sensitivity experiment: random only, first BG seed
    if MODEL_SEED_IDX == 0:
        ACTIVE_METHODS  = ['random', 'kmeans', 'pred_stratified', 'cte']
        ACTIVE_BG_SEEDS = BG_SEEDS
        ACTIVE_S_IDXS   = list(range(len(BG_SEEDS)))
    else:
        ACTIVE_METHODS  = ['random', 'kmeans']
        ACTIVE_BG_SEEDS = [BG_SEEDS[0]]
        ACTIVE_S_IDXS   = [0]

    if args.only_method is not None:
        ACTIVE_METHODS = [args.only_method]

    MT        = MAX_TEST.get(DATASET)
    N_CHUNKS  = args.n_chunks
    CHUNK_IDX = args.chunk_idx
    OUTDIR    = args.outdir

    # Override BG_SIZE if specified
    global BG_SIZE
    if args.bg_size is not None:
        BG_SIZE = args.bg_size

    # File suffix: chunk info when chunked, else empty
    CHUNK_SUFFIX = f'_c{CHUNK_IDX}' if N_CHUNKS > 1 else ''
    BG_SUFFIX    = f'_bg{BG_SIZE}' if BG_SIZE != 100 else ''

    RES_SUBDIR = 'background_results' if BG_SIZE != 100 else 'results_tree'

    def result_path(method, s_idx):
        prefix = 'perm_' if EXPLAINER_TYPE == 'permutation_shap' else ''
        return os.path.join(OUTDIR, RES_SUBDIR,
                            f'{DATASET}_{MODEL}_{SPLIT_SEED}_{MODEL_SEED}'
                            f'_{prefix}{method}{BG_SUFFIX}_s{s_idx}{CHUNK_SUFFIX}_sv.pkl')

    # Check if all results already exist → skip entirely
    all_exist = all(
        os.path.exists(result_path(method, s_idx))
        for method in ACTIVE_METHODS for s_idx in ACTIVE_S_IDXS
    )
    if all_exist:
        print(f"[SKIP] All results exist for {DATASET}/{MODEL}/{EXPLAINER_TYPE}")
        return

    os.makedirs(os.path.join(OUTDIR, 'log_sampling'), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, RES_SUBDIR), exist_ok=True)

    log_suffix = '_perm' if EXPLAINER_TYPE == 'permutation_shap' else ''
    log_filename = os.path.join(OUTDIR, 'log_sampling',
                                f'{DATASET}_{MODEL}_{SPLIT_SEED}_{MODEL_SEED}{log_suffix}.log')
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout
    print(f"=== experiment_sampling.py ===")
    print(f"Dataset: {DATASET} | Model: {MODEL} | Split: {SPLIT_SEED} | ModelSeed: {MODEL_SEED} | Explainer: {EXPLAINER_TYPE}")
    print(f"Methods: {METHODS}")
    print(f"BG seeds: {BG_SEEDS} | Explainer seed: {EXPLAINER_SEED} (fixed)")
    print(f"BG_SIZE: {BG_SIZE} | NSAMPLES: {NSAMPLES} | MAX_TEST: {MT}")
    print(f"Chunks: {N_CHUNKS} | Chunk idx: {CHUNK_IDX}")

    t_total = time.time()

    # 1. Load data
    X, Y = load_data(DATASET)
    print(f"\nData loaded: {X.shape}")

    # 2. Detect feature types — pipeline handles OHE internally; explainers see raw feature space
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

    # OrdinalEncoder for PermutationSHAP and kmeans/cte background selectors:
    #   shap.maskers.Independent and KMeans/StandardScaler require all-numerical data;
    #   f_perm inverse-transforms ordinal ints back to original strings before calling the pipeline.
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cat_features:
        ord_enc.fit(X_train[cat_features])
        X_train_enc = X_train.copy()
        X_train_enc[cat_features] = ord_enc.transform(X_train[cat_features]).astype(int)
        X_test_enc = X_test.copy()
        X_test_enc[cat_features] = ord_enc.transform(X_test[cat_features]).astype(int)
    else:
        X_train_enc = X_train   # no-op for all-numerical datasets (Diabetes, GMSC)
        X_test_enc  = X_test

    # Cap test set for large datasets
    if MT is not None and len(X_test) > MT:
        X_test     = X_test.iloc[:MT]
        X_test_enc = X_test_enc.iloc[:MT]

    # Chunk the test set for parallel execution
    n_test_total = len(X_test)
    chunk_size   = (n_test_total + N_CHUNKS - 1) // N_CHUNKS
    chunk_start  = CHUNK_IDX * chunk_size
    chunk_end    = min(chunk_start + chunk_size, n_test_total)
    X_test     = X_test.iloc[chunk_start:chunk_end].reset_index(drop=True)
    X_test_enc = X_test_enc.iloc[chunk_start:chunk_end].reset_index(drop=True)
    print(f"Explaining {len(X_test)} test instances (chunk {CHUNK_IDX}/{N_CHUNKS}: [{chunk_start}:{chunk_end}])")

    if len(X_test) == 0:
        print(f"[SKIP] Empty chunk {CHUNK_IDX} (chunk_start={chunk_start} >= n_test_total={n_test_total})")
        return

    # 4. Train model ONCE for all method × seed combos (pipeline handles OHE internally)
    print(f"\nTraining {MODEL} (seed={MODEL_SEED})...")
    t_train = time.time()
    clf = train_model(X_train, Y_train, MODEL, MODEL_SEED, num_features, cat_features)
    print(f"Training done in {(time.time()-t_train)/60:.2f} min")

    f      = make_predict_fn(clf, raw_cols, num_features, cat_features)
    f_perm = make_predict_fn_enc(clf, raw_cols, num_features, cat_features, ord_enc)

    # PermutationSHAP budget: max_evals=NSAMPLES matches KernelSHAP nsamples=NSAMPLES
    # (each model evaluation counts equally; SHAP converts to permutations internally)
    # For reference: approx n_perm = NSAMPLES // (2 * (d + 1))
    _d_raw = len(raw_cols)
    _n_perm_ref = max(1, round(NSAMPLES / (2 * (_d_raw + 1))))
    if EXPLAINER_TYPE == 'permutation_shap':
        print(f"PermutationSHAP: d={_d_raw}, max_evals={NSAMPLES} (~{_n_perm_ref} permutations)")

    # 5. Loop: method × BG seed
    for method in ACTIVE_METHODS:
        print(f"\n{'='*50}")
        print(f"Method: {method}")

        for s_idx, bg_seed in zip(ACTIVE_S_IDXS, ACTIVE_BG_SEEDS):
            save_path = result_path(method, s_idx)
            if os.path.exists(save_path):
                print(f"  [SKIP] s{s_idx} (bg_seed={bg_seed}): {save_path}")
                continue

            print(f"  Selecting background s{s_idx} (bg_seed={bg_seed})...")
            # kmeans and cte both need ordinal-encoded data (StandardScaler / KMeans require numerical);
            # random and pred_stratified use raw DataFrame (f handles type conversion internally).
            # kmeans returns shap DenseData (weighted centroids); others return index arrays.
            if method in ('cte', 'kmeans'):
                bg_raw = BG_FNS[method](X_train_enc, Y_train, f, bg_seed)
            else:
                bg_raw = BG_FNS[method](X_train, Y_train, f, bg_seed)

            if method == 'kmeans':
                bg_shap    = bg_raw                      # shap DenseData (weighted centroids)
                bg_arr_enc = bg_shap.data                # (BG_SIZE, d) float, ordinal-encoded scale
                print(f"  Background selected: {BG_SIZE} kmeans centroids")
            else:
                bg_idx     = bg_raw                      # integer indices into training set
                bg_df      = X_train.iloc[bg_idx]        # raw DataFrame for KernelSHAP
                bg_arr_enc = X_train_enc.values[bg_idx]  # ordinal-encoded array for PermSHAP
                print(f"  Background selected: {len(bg_idx)} instances")

            t_shap = time.time()

            if EXPLAINER_TYPE == 'kernel_shap':
                print(f"  Running KernelSHAP (explainer_seed={EXPLAINER_SEED})...")
                if method == 'kmeans':
                    # bg_shap: weighted DenseData; f_perm handles float ordinal centroids via
                    # np.round → inverse_transform → pipeline
                    explainer = shap.KernelExplainer(f_perm, bg_shap, link='identity', seed=EXPLAINER_SEED)
                    sv_raw = explainer.shap_values(X_test_enc.values, nsamples=NSAMPLES, gc_collect=True)
                else:
                    explainer = shap.KernelExplainer(f, bg_df, link='identity', seed=EXPLAINER_SEED)
                    sv_raw = explainer.shap_values(X_test.values, nsamples=NSAMPLES, gc_collect=True)
                elapsed = (time.time() - t_shap) / 60
                print(f"  Done in {elapsed:.2f} min")

                if isinstance(sv_raw, list):
                    sv = sv_raw[1]
                    ev = explainer.expected_value
                    base_val = float(ev[1] if hasattr(ev, '__len__') else ev)
                else:
                    sv = sv_raw
                    base_val = float(explainer.expected_value)

                # efficiency error only for non-kmeans (kmeans uses f_perm / X_test_enc)
                if method != 'kmeans':
                    err = np.abs(sv.sum(axis=1) + base_val - f(X_test.values)).mean()
                    print(f"  Efficiency error: {err:.6f} | shape: {sv.shape} | chunk_start={chunk_start}")
                else:
                    print(f"  shape: {sv.shape} | chunk_start={chunk_start}")
                del explainer, sv_raw

            elif EXPLAINER_TYPE == 'permutation_shap':
                print(f"  Running PermutationSHAP (explainer_seed={EXPLAINER_SEED}, max_evals={NSAMPLES})...")
                # All methods use ordinal-encoded data + f_perm:
                # shap.maskers.Independent calls np.isclose() internally which fails on strings.
                masker   = shap.maskers.Independent(bg_arr_enc, max_samples=BG_SIZE)
                perm_exp = shap.PermutationExplainer(f_perm, masker, seed=EXPLAINER_SEED)
                explanation = perm_exp(X_test_enc.values, max_evals=NSAMPLES)
                sv       = explanation.values
                base_val = float(np.mean(explanation.base_values))
                elapsed  = (time.time() - t_shap) / 60
                print(f"  Done in {elapsed:.2f} min | shape: {sv.shape} | chunk_start={chunk_start}")
                del perm_exp, explanation

            shap_exp = shap.Explanation(
                values=sv, base_values=base_val,
                data=X_test.values, feature_names=raw_cols
            )
            with open(save_path, 'wb') as fout:
                pickle.dump(shap_exp, fout, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Saved → {save_path}")

            del sv, shap_exp
            clean_memory()

    print(f"\nTotal time: {(time.time()-t_total)/60:.2f} min")
    del clf
    clean_memory()


if __name__ == "__main__":
    main()
