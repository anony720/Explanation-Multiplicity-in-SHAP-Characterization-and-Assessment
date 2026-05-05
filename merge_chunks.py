"""
merge_chunks.py
===============
Merge chunked SHAP result files into a single file.

Usage:
    python merge_chunks.py --dataset acs --model dt --n_chunks 10
    python merge_chunks.py --dataset gmsc --n_chunks 10   # all models
"""

import os
import pickle
import argparse
import numpy as np
import shap

METHODS_DEFAULT = ['pred_stratified', 'dist_matching', 'cte']
METHODS_GMSC    = ['random', 'pred_stratified', 'dist_matching', 'cte']
MODELS          = ['dt', 'rf', 'xgb', 'mlp', 'ftt']
RES_DIR         = './results_tree'


def merge_one(dataset, model, method, s_idx, n_chunks, split=0, m_seed=0):
    chunks = []
    for c in range(n_chunks):
        path = (f'{RES_DIR}/{dataset}_{model}_{split}_{m_seed}'
                f'_{method}_s{s_idx}_c{c}_sv.pkl')
        if not os.path.exists(path):
            print(f'  MISSING chunk {c}: {path}')
            return False
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        chunks.append(obj)

    # Concatenate values and data along axis 0
    merged_values = np.concatenate([c.values for c in chunks], axis=0)
    merged_data   = np.concatenate([c.data   for c in chunks], axis=0)
    base_val      = chunks[0].base_values  # same for all chunks
    feat_names    = chunks[0].feature_names

    merged = shap.Explanation(
        values=merged_values,
        base_values=base_val,
        data=merged_data,
        feature_names=feat_names,
    )

    out_path = (f'{RES_DIR}/{dataset}_{model}_{split}_{m_seed}'
                f'_{method}_s{s_idx}_sv.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Merged ({merged_values.shape[0]} instances) → {out_path}')

    # Remove chunk files after successful merge
    for c in range(n_chunks):
        os.remove(f'{RES_DIR}/{dataset}_{model}_{split}_{m_seed}'
                  f'_{method}_s{s_idx}_c{c}_sv.pkl')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  type=str, required=True)
    parser.add_argument('--model',    type=str, default=None,
                        help='Specific model (default: all)')
    parser.add_argument('--n_chunks', type=int, required=True)
    args = parser.parse_args()

    methods = METHODS_GMSC if args.dataset == 'gmsc' else METHODS_DEFAULT
    models  = [args.model] if args.model else MODELS

    for model in models:
        print(f'\n=== {args.dataset}/{model} ===')
        for method in methods:
            for s_idx in range(5):
                ok = merge_one(args.dataset, model, method, s_idx, args.n_chunks)
                if not ok:
                    print(f'  SKIP: {method} s{s_idx} — incomplete chunks')

    print('\nDone.')


if __name__ == '__main__':
    main()
