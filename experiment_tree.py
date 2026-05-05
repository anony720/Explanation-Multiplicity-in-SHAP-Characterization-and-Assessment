"""
experiment_tree.py
==================
TreeSHAP (path_dependent, model_output='raw') ground truth.

- One-hot encoding (deterministic, no fitted statistics)
- dt / rf only (raw output = leaf class fraction = probability space)
- Fully deterministic: no random seeds, no background
- Saves: results_tree/{DATASET}_{MODEL}_{SPLIT_SEED}_0_tree_sv.pkl
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import set_global_seed, clean_memory, load_data


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


def train_clf_ohe(X_train_ohe, Y_train, model, model_seed):
    X_arr = X_train_ohe.values if hasattr(X_train_ohe, 'values') else X_train_ohe
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    if model == 'dt':
        base = DecisionTreeClassifier(random_state=model_seed)
        grid = {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 10]}
    elif model == 'rf':
        base = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=model_seed)
        grid = {"max_depth": [None, 7, 15], "min_samples_leaf": [1, 5]}
    else:
        raise ValueError(f"Unsupported model: {model}")
    gs = GridSearchCV(base, grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_arr, Y_train)
    print(f"  Best: {gs.best_params_} | AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_


def main():
    parser = argparse.ArgumentParser(
        description="TreeSHAP (path_dependent) for dt/rf on OHE data — deterministic"
    )
    parser.add_argument('--dataset',    type=str, default='german',
                        help='Dataset: german, diabetes, acs')
    parser.add_argument('--model',      type=str, default='dt',
                        choices=['dt', 'rf'])
    parser.add_argument('--split_seed', type=int, default=0,
                        help='Fold index for StratifiedKFold (0-4)')
    args = parser.parse_args()

    DATASET    = args.dataset
    MODEL      = args.model
    SPLIT_SEED = args.split_seed
    MODEL_SEED = 0   # fixed

    os.makedirs('log', exist_ok=True)
    os.makedirs('results_tree', exist_ok=True)

    save_path = f'./results_tree/{DATASET}_{MODEL}_{SPLIT_SEED}_0_tree_sv.pkl'
    if os.path.exists(save_path):
        print(f"[SKIP] Already exists: {save_path}")
        return

    log_filename = f"log/{DATASET}_{MODEL}_{SPLIT_SEED}_0_tree.log"
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout
    print(f"Logging to: {log_filename}")
    print(f"=== experiment_tree.py (TreeSHAP, OHE) ===")
    print(f"Dataset: {DATASET} | Model: {MODEL} | Split: {SPLIT_SEED} | ModelSeed: {MODEL_SEED}")

    # 1. Load data
    X, Y = load_data(DATASET)
    print(f"Data loaded: {X.shape}")

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = [c for c in X.columns if c not in cat_features]
    print(f"Numeric: {len(num_features)}, Categorical: {len(cat_features)}")

    # 2. OHE (deterministic)
    X_ohe = pd.get_dummies(X, drop_first=False).astype(float)
    ohe_cols = X_ohe.columns.tolist()
    print(f"OHE feature count: {len(ohe_cols)}")

    # 3. Split
    set_global_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        if i == SPLIT_SEED:
            X_train = X_ohe.iloc[train_idx].reset_index(drop=True)
            X_test  = X_ohe.iloc[test_idx].reset_index(drop=True)
            Y_train = Y.iloc[train_idx].reset_index(drop=True)
            break
    print(f"Split {SPLIT_SEED}: train={len(X_train)}, test={len(X_test)}")

    # 4. Train
    t_start = time.time()
    print(f"\nTraining {MODEL} (seed={MODEL_SEED})...")
    clf = train_clf_ohe(X_train, Y_train, MODEL, MODEL_SEED)
    print(f"Training done in {(time.time()-t_start)/60:.2f} min")

    # 5. TreeSHAP — path_dependent, model_output='raw'
    #    For dt/rf: raw = leaf class fraction = probability space
    #    Fully deterministic (no background, uses tree structure internally)
    print(f"\nComputing TreeSHAP (path_dependent, model_output=raw)...")
    t_tree = time.time()
    tree_explainer = shap.TreeExplainer(
        clf,
        feature_perturbation='tree_path_dependent',
        model_output='raw'
    )
    sv_raw = tree_explainer.shap_values(X_test.values)
    print(f"TreeSHAP done in {(time.time()-t_tree)/60:.2f} min")

    ev = tree_explainer.expected_value
    if isinstance(sv_raw, list):
        sv = sv_raw[1]
        base_val = float(ev[1] if hasattr(ev, '__len__') else ev)
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        sv = sv_raw[:, :, 1]
        base_val = float(ev[1] if hasattr(ev, '__len__') else ev)
    else:
        sv = sv_raw
        base_val = float(ev[1] if hasattr(ev, '__len__') else ev)

    # Efficiency check
    proba = clf.predict_proba(X_test.values)[:, 1]
    err = np.abs(sv.sum(axis=1) + base_val - proba).mean()
    print(f"Efficiency axiom mean error: {err:.6f}")
    print(f"sv shape: {sv.shape} | base_val: {base_val:.4f}")

    # 6. Save
    shap_exp = shap.Explanation(
        values=sv,
        base_values=base_val,
        data=X_test.values,
        feature_names=ohe_cols
    )
    with open(save_path, 'wb') as fout:
        pickle.dump(shap_exp, fout, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved → {save_path}")

    print(f"\nTotal time: {(time.time()-t_start)/60:.2f} min")
    del tree_explainer, sv_raw, shap_exp
    clean_memory()


if __name__ == "__main__":
    main()
