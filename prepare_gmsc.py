"""
prepare_gmsc.py
===============
Download Give Me Some Credit dataset from OpenML and save as pkl.
- 150K rows → stratified subsample to ~45K (comparable to ACS Income)
- 10 numeric features
- Target: SeriousDlqin2yrs (binary)
- NaN: median imputation
- Save: dataset/gmsc_X.pkl, dataset/gmsc_Y.pkl
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SAMPLE_SIZE = 45000
RANDOM_STATE = 42

print("Downloading Give Me Some Credit from OpenML...")
d = fetch_openml('give-me-some-credit', version=1, as_frame=True, parser='auto')

X = d.data.copy()
Y = d.target.copy()

print(f"Raw shape: {X.shape}")
print(f"NaN per column:\n{X.isnull().sum()}")

# Median imputation for NaN
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

# Convert target to int (0/1)
Y = Y.astype(int)

print(f"\nAfter imputation — X: {X.shape}, Y: {Y.shape}")
print(f"Class balance (full): {Y.value_counts().to_dict()}")

# Stratified subsample to ~45K rows (comparable to ACS Income: 45,960 rows)
X_sub, _, Y_sub, _ = train_test_split(
    X, Y,
    train_size=SAMPLE_SIZE,
    stratify=Y,
    random_state=RANDOM_STATE
)
X, Y = X_sub.reset_index(drop=True), Y_sub.reset_index(drop=True)

print(f"After stratified sampling ({SAMPLE_SIZE}) — X: {X.shape}, Y: {Y.shape}")
print(f"Class balance (sampled): {Y.value_counts().to_dict()}")
print(f"Features: {X.columns.tolist()}")

import os
os.makedirs('dataset', exist_ok=True)

with open('dataset/gmsc_X.pkl', 'wb') as f:
    pickle.dump(X, f, protocol=4)
with open('dataset/gmsc_Y.pkl', 'wb') as f:
    pickle.dump(Y, f, protocol=4)

print("\nSaved: dataset/gmsc_X.pkl, dataset/gmsc_Y.pkl")
