import os
import gc
import random
import pickle
import numpy as np
import pandas as pd
import torch

def set_global_seed(seed: int):
    """
    Fix random seeds for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic operations for reproducibility (may reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_memory():
    """
    Clean up CPU RAM and GPU VRAM.
    """
    # 1. Python Garbage Collection
    gc.collect()

    # 2. PyTorch CUDA Cache Clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_data(dataset_name: str, data_dir: str = 'dataset'):
    """
    Load X, Y pickle files from the dataset directory.
    """
    x_path = os.path.join(data_dir, f'{dataset_name}_X.pkl')
    y_path = os.path.join(data_dir, f'{dataset_name}_Y.pkl')

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found at {x_path} or {y_path}")

    with open(x_path, 'rb') as f:
        X = pickle.load(f)
    with open(y_path, 'rb') as f:
        Y = pickle.load(f)
    
    # Convert DataFrame to Series for indexing convenience
    if isinstance(Y, pd.DataFrame):
        Y = Y.iloc[:, 0]
        
    return X, Y