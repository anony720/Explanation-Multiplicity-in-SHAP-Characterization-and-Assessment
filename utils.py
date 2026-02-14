import os
import gc
import random
import pickle
import numpy as np
import pandas as pd
import torch

def set_global_seed(seed: int):
    """
    Python, Numpy, PyTorch의 랜덤 시드를 고정합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 재현성을 위해 결정론적 연산 강제 (속도는 조금 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_memory():
    """
    CPU RAM과 GPU VRAM을 정리합니다.
    """
    # 1. Python Garbage Collection
    gc.collect()

    # 2. PyTorch CUDA Cache Clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_data(dataset_name: str, data_dir: str = 'dataset'):
    """
    dataset 폴더에서 X, Y pickle 파일을 로드합니다.
    """
    x_path = os.path.join(data_dir, f'{dataset_name}_X.pkl')
    y_path = os.path.join(data_dir, f'{dataset_name}_Y.pkl')

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found at {x_path} or {y_path}")

    with open(x_path, 'rb') as f:
        X = pickle.load(f)
    with open(y_path, 'rb') as f:
        Y = pickle.load(f)
    
    # Y가 DataFrame이면 Series로 변환 (인덱싱 편의성)
    if isinstance(Y, pd.DataFrame):
        Y = Y.iloc[:, 0]
        
    return X, Y