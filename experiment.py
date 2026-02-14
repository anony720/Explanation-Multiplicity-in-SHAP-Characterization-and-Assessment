import sys
import os
# Telemetry 비활성화 (가장 먼저 실행)
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

# 로컬 모듈 Import
from utils import set_global_seed, clean_memory, load_data
from models import train_model

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")  # "a"는 append(이어쓰기), "w"는 덮어쓰기

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 바로바로 파일에 쓰도록 강제

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    parser = argparse.ArgumentParser(description="Run SHAP Experiments")
    
    # Arguments
    parser.add_argument('--dataset', type=str, default='acs', help='Dataset name (e.g., acs, diabetes)')
    parser.add_argument('--model', type=str, default='ftt', choices=['ftt', 'tabpfn', 'lr', 'dt', 'rf', 'xgb'], help='Model name')
    parser.add_argument('--split_seed', type=int, default=0, help='Seed for StratifiedKFold split (0-4)')
    parser.add_argument('--model_seed_idx', type=int, default=0, help='Index for model random seed list')
    parser.add_argument('--explainer_seed_idx', type=int, default=0, help='Index for explainer random seed list')
    parser.add_argument('--chunk_idx', type=int, default=0, help='현재 조각 번호 (0부터 시작)')
    parser.add_argument('--total_chunks', type=int, default=1, help='전체 조각 개수')
    args = parser.parse_args()

    # Parameters
    DATASET = args.dataset
    MODEL = args.model
    SPLIT_SEED = args.split_seed
    MODEL_SEED_IDX = args.model_seed_idx
    EXPLAINER_SEED_IDX = args.explainer_seed_idx
    CHUNK_IDX = args.chunk_idx
    TOTAL_CHUNKS = args.total_chunks

    # experiment.py 내부
    save_path = f'results/{DATASET}_{MODEL}_{SPLIT_SEED}_{MODEL_SEED_IDX}_{EXPLAINER_SEED_IDX}_chunk{CHUNK_IDX}_sv.pkl'

    # ★ 이 3줄이 핵심입니다. 꼭 넣으세요!
    if os.path.exists(save_path):
        print(f"[SKIP] Found existing result: {save_path}")
        return

    if not os.path.exists('log'):
        os.makedirs('log')

    # 2. 로그 파일명 생성 (결과 파일명 규칙과 동일하게)
    log_filename = f"log/{DATASET}_{MODEL}_{SPLIT_SEED}_{MODEL_SEED_IDX}_{EXPLAINER_SEED_IDX}.log"
    
    # 3. 시스템 출력(stdout)을 가로채서 파일과 화면 양쪽에 쏨
    sys.stdout = Logger(log_filename)
    
    # Error log
    sys.stderr = sys.stdout 
    
    print(f"Logging started. Saving to: {log_filename}")
    
    FOLDS = 5
    model_seeds = [0, 21, 42, 63, 84]
    MODEL_SEED = model_seeds[MODEL_SEED_IDX]
    explainer_seeds = [25, 50, 75, 100, 125]
    EXPLAINER_SEED = explainer_seeds[EXPLAINER_SEED_IDX]
    
    print(f"=== Experiment Start: {DATASET} | {MODEL} | Split:{SPLIT_SEED} | ModelSeed:{MODEL_SEED} | ExplainerSeed:{EXPLAINER_SEED} ===")
    
    # 1. Load Data
    try:
        X, Y = load_data(DATASET)
        print(f"Data Loaded. X shape: {X.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Feature Setup
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = [c for c in X.columns if c not in cat_features]
    
    # Output Directory
    os.makedirs('./results', exist_ok=True)

    # 3. Main Loop
    set_global_seed(42)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        # 지정된 Split Seed(Fold)만 실행
        if i == SPLIT_SEED:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            
            print(f'[{MODEL}] Train start (Fold {i})')
            t_start = time.time()
            
            # Train Model
            best_pipe = train_model(
                X_train, Y_train, MODEL, MODEL_SEED, num_features, cat_features
            )
            
            print('Train done')
            print(f'Training took {(time.time()-t_start)/60:.2f} minutes')
            
            # --- SHAP Preparation ---
            def f(X_raw):
                if isinstance(X_raw, np.ndarray):
                    X_df = pd.DataFrame(X_raw, columns=X.columns)
                    for c in cat_features:
                        X_df[c] = X_df[c].astype(str)
                    for c in num_features:
                        X_df[c] = X_df[c].astype(float)
                else:
                    X_df = X_raw
                
                # Probability Space Output
                return best_pipe.predict_proba(X_df)[:, 1]

            # Save Probabilities
            proba = f(X_test)
            with open(f'./results/{DATASET}_{MODEL}_{SPLIT_SEED}_{MODEL_SEED_IDX}_proba.pkl', 'wb') as file:
                pickle.dump(proba, file, protocol=pickle.HIGHEST_PROTOCOL)  
            del proba

            # --- KernelSHAP ---
            print('SHAP calculation start')
            
            bg_raw = shap.utils.sample(X_test, 100, random_state=EXPLAINER_SEED)

            # TabPFN Noise Injection
            if MODEL == 'tabpfn':
                bg_raw_noisy = bg_raw.copy() 
                bg_raw_noisy[num_features] += np.random.normal(0, 1e-7, size=bg_raw_noisy[num_features].shape)
                bg_raw = bg_raw_noisy


            # KernelExplainer (Identity Link for Probability)
            explainer = shap.KernelExplainer(f, bg_raw, link="identity",seed=42)
            
            # Explain Entire Test Set (or subset if too large)
            # nsamples=1024 for paper quality
            print(f"Explaining {len(X_test)} samples with nsamples=1024...")
            X_test_chunks = np.array_split(X_test, TOTAL_CHUNKS)
            if CHUNK_IDX >= len(X_test_chunks):
                print(f"Error: Chunk index {CHUNK_IDX} out of bounds.")
                return

            X_explain = X_test_chunks[CHUNK_IDX]
            
            print(f"=== Chunk Processing [{CHUNK_IDX+1}/{TOTAL_CHUNKS}] ===")
            print(f"Total Test Data: {len(X_test)}")
            print(f"My Chunk Size: {len(X_explain)}")
            print("========================================")
            shap_values_list = explainer.shap_values(X_explain, nsamples=1024,gc_collect=True)
            
            print('SHAP calculation done')
            print(f'Total pipeline took {(time.time()-t_start)/60:.2f} minutes')

            # Create Explanation Object
            if isinstance(shap_values_list, list):
                sv = shap_values_list[1] # Class 1
                base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
            else:
                sv = shap_values_list
                base_val = explainer.expected_value

            shap_explanation = shap.Explanation(
                values=sv,
                base_values=base_val,
                data=X_test.values,
                feature_names=X.columns.tolist()
            )

            # Save Results
            if TOTAL_CHUNKS > 1:
                save_path = f'./results/{DATASET}_{MODEL}_{SPLIT_SEED}_{MODEL_SEED_IDX}_{EXPLAINER_SEED_IDX}_{CHUNK_IDX}_sv.pkl'
            else:
                save_path = f'./results/{DATASET}_{MODEL}_{SPLIT_SEED}_{MODEL_SEED_IDX}_{EXPLAINER_SEED_IDX}_sv.pkl'
            with open(save_path, 'wb') as file:
                pickle.dump(shap_explanation, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved SHAP values to {save_path}")

            # Optional Plot (Headless server might not show this)
            # shap.plots.beeswarm(shap_explanation, show=False)

            # Cleanup
            del best_pipe, explainer, shap_values_list, shap_explanation, bg_raw
            clean_memory()

if __name__ == "__main__":
    main()