# ========================================================
# SHAP Sensitivity Analysis - Refactored
# ========================================================

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import StratifiedKFold

# Use science plots for publication-quality figures
try:
    plt.style.use(['science', 'no-latex'])
except:
    print("Warning: scienceplots not installed. Using default matplotlib style.")
    print("Install with: pip install SciencePlots")

# ========================================================
# Configuration
# ========================================================

class Config:
    """Global configuration for the analysis"""
    DATASET = 'acs'  # Default dataset - can be overridden in notebook
    RESULT_DIR = 'results'
    FIGURE_DIR = 'figures'  # Output directory for saved figures
    MODELS = ['dt', 'rf', 'xgb', 'ftt', 'mlp', 'tabpfn']

    N_SPLITS = 5
    N_MODEL_SEEDS = 5
    N_EXPLAINER_SEEDS = 5

    # Top-K features to display
    TOP_K_FEATURES = 6  # Adjustable: 5-6 for 8 features, 10 for 16 features

    # Metric parameters
    JACCARD_K = 3  # Top-K for Jaccard distance
    # RBO_P is now computed dynamically as p = 1 - (1/n_features)
    # This makes p dataset-dependent: diabetes/acs (8 features) → p=0.875, german (16 features) → p=0.9375

    # Y-axis limits for cross-dataset comparison (set to None for auto)
    # These values should be set to accommodate the maximum across all 3 datasets
    L2_YLIM = (0, 0.35)      # Unified y-limit for L2 distance plots (1-a, 1-b, 2-a, 3-a)
    JACCARD_YLIM = (0, 1.05) # Unified y-limit for Jaccard distance plots
    RBO_YLIM = (0, 1.05)     # Unified y-limit for RBO distance plots
    FEATURE_L2_YLIM = (0, 0.2)  # Unified x-limit for feature-wise overall plots (1-d)
    FEATURE_L2_YLIM_CERTAINTY = (0, 0.03)  # Unified x-limit for certainty feature plots (2-b)
    FEATURE_L2_YLIM_PREDICTION = (0, 0.03)  # Unified x-limit for prediction feature plots (3-b)

    # Model colors (consistent across all plots)
    MODEL_COLORS = {
        'dt': '#ff7f0e',      # Orange
        'rf': '#2ca02c',      # Green
        'xgb': '#d62728',     # Red
        'ftt': '#9467bd',     # Purple
        'mlp': '#1f77b4',     # Blue
        'tabpfn': '#8c564b'   # Brown
    }

    # Other plot colors
    COLORS = {
        'overall': 'tab:blue',
        'explainer': 'tab:orange',
        'model': 'tab:green',
        'Certain': 'tab:blue',      # Certainty subgroup colors
        'Uncertain': 'tab:red',
        'TP': 'tab:green',          # Prediction subgroup colors
        'TN': 'tab:blue',
        'FP': 'tab:red',
        'FN': 'tab:orange',
        # Legacy lowercase keys for backward compatibility
        'certain': 'tab:blue',
        'uncertain': 'tab:red',
        'tp': 'tab:green',
        'tn': 'tab:blue',
        'fp': 'tab:red',
        'fn': 'tab:orange'
    }

# ========================================================
# Distance Metrics
# ========================================================

def compute_l2_distance(v1, v2):
    """
    Compute L2 (Euclidean) distance between two SHAP value vectors.

    Args:
        v1, v2: Arrays of shape (N, F) where N=samples, F=features

    Returns:
        Array of shape (N,) with L2 distance for each sample
    """
    return np.sqrt(np.sum((v1 - v2) ** 2, axis=1))


# ========================================================
# Baseline Computations (from PDF)
# ========================================================

def compute_jaccard_baseline(d, r, q_values=None):
    """
    Compute expected Jaccard distance using Monte Carlo with Mallows model.

    Sweeps over multiple q values and returns min/max range for shaded baseline region.

    Args:
        d: Number of features
        r: Top-k size (k for Jaccard@r)
        q_values: List of Mallows dispersion parameters. If None, uses [0.3, 0.4, 0.5]

    Returns:
        dict: {'min': min baseline distance, 'max': max baseline distance}
    """
    if q_values is None:
        q_values = [0.3, 0.4, 0.5]

    # Import baseline computation from baseline.py
    import baseline

    baseline_distances = []
    for q in q_values:
        result = baseline.compute_jaccard_baseline_simplified(
            d=d, r=r, q=q, n_trials=20000
        )
        # Result['mean'] is already distance
        baseline_distances.append(result['mean'])

    return {'min': min(baseline_distances), 'max': max(baseline_distances)}


def compute_rbo_baseline(n_features, p=None, q_values=None):
    """
    Compute expected RBO distance for random rankings using Monte Carlo with Mallows model.

    Sweeps over multiple q values and returns min/max range for shaded baseline region.

    Args:
        n_features: Total number of features (required)
        p: Persistence parameter. If None, uses p = 1 - (1/n_features)
        q_values: List of Mallows dispersion parameters. If None, uses [0.3, 0.4, 0.5]

    Returns:
        dict: {'min': min baseline distance, 'max': max baseline distance}
    """
    if p is None:
        p = 1.0 - (1.0 / n_features)

    if q_values is None:
        q_values = [0.3, 0.4, 0.5]

    # Import baseline computation from baseline.py
    import baseline

    baseline_distances = []
    for q in q_values:
        result = baseline.compute_rbo_baseline_simplified(
            d=n_features, r=n_features, p=p, q=q, n_trials=20000
        )
        # Result['mean'] is already distance
        baseline_distances.append(result['mean'])

    return {'min': min(baseline_distances), 'max': max(baseline_distances)}


def compute_l2_baseline(d, r, rho_values=None, kappa_values=None, T=0.4):
    """
    Compute expected squared L2 distance under heavy-tailed Dirichlet prior.

    Sweeps over multiple rho and kappa values and returns min/max range for shaded baseline region.

    From Proposition 6 in the PDF:
    E‖X - Y‖²₂ = [2T²/(κ+1)] × [1 - ρ²/r - (1-ρ)²/(d-r)]

    Args:
        d: Number of features
        r: Number of top features (concentrated mass)
        rho_values: List of mass proportions on top-r features. If None, uses [0.6, 0.7, 0.8]
        kappa_values: List of concentration parameters. If None, uses [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        T: Total SHAP mass (default 0.4)

    Returns:
        dict: {'min': min baseline distance, 'max': max baseline distance}
    """
    if rho_values is None:
        rho_values = [0.6, 0.7, 0.8]

    if kappa_values is None:
        kappa_values = list(range(5, 16))  # [5, 6, 7, ..., 15]

    baseline_distances = []
    for rho in rho_values:
        for kappa in kappa_values:
            term1 = (rho ** 2) / r
            term2 = ((1 - rho) ** 2) / (d - r)
            exp_sq_l2 = (2 * T ** 2) / (kappa + 1) * (1 - term1 - term2)
            # Return L2 distance (not squared)
            baseline_distances.append(np.sqrt(exp_sq_l2))

    return {'min': min(baseline_distances), 'max': max(baseline_distances)}


def compute_jaccard_distance(v1, v2, k=3):
    """
    Compute Jaccard distance based on top-K feature agreement.

    Args:
        v1, v2: Arrays of shape (N, F)
        k: Number of top features to consider

    Returns:
        Array of shape (N,) with Jaccard distance (1 - Jaccard similarity)
    """
    n_samples = v1.shape[0]
    distances = np.zeros(n_samples)

    # Get top-K feature indices based on absolute SHAP values
    top_k_v1 = np.argsort(np.abs(v1), axis=1)[:, -k:]
    top_k_v2 = np.argsort(np.abs(v2), axis=1)[:, -k:]

    for i in range(n_samples):
        set1 = set(top_k_v1[i])
        set2 = set(top_k_v2[i])

        if not set1 and not set2:
            distances[i] = 0.0
            continue

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        # Jaccard distance = 1 - Jaccard similarity
        distances[i] = 1.0 - (intersection / union) if union > 0 else 1.0

    return distances


def compute_rbo_similarity(v1, v2, p=None, depth=None):
    """
    Compute Rank-Biased Overlap (RBO) similarity between SHAP value vectors.

    RBO measures agreement between two rankings with higher weight on top-ranked items.
    Formula: RBO(S,T,p) = (1-p) × Σ_{d=1}^depth [p^(d-1) × |S:d ∩ T:d| / d]

    Args:
        v1, v2: Arrays of shape (N, F) - SHAP values for N samples and F features
        p: Persistence parameter (0 to 1). If None, uses p = 1 - (1/n_features).
           Higher p = more weight on top ranks.
        depth: Maximum depth to compute. If None or > n_features, uses n_features.

    Returns:
        Array of shape (N,) with RBO similarity values in [0, 1]
        1 = perfect agreement, 0 = no agreement
    """
    n_samples, n_features = v1.shape

    # Set p dynamically if not provided
    if p is None:
        p = 1.0 - (1.0 / n_features)

    # Set depth to n_features (can't go deeper than available features)
    if depth is None or depth > n_features:
        depth = n_features

    similarities = np.zeros(n_samples)

    # Precompute weights: (1-p) × p^(d-1) for d=1 to depth
    weights = (1 - p) * (p ** np.arange(depth))

    for i in range(n_samples):
        # Rank features by absolute SHAP values (descending order)
        rank1 = np.argsort(np.abs(v1[i]))[::-1]
        rank2 = np.argsort(np.abs(v2[i]))[::-1]

        rbo_sum = 0.0
        for d in range(1, depth + 1):
            # Get top-d features from each ranking
            top_d_rank1 = set(rank1[:d])
            top_d_rank2 = set(rank2[:d])

            # Compute overlap at depth d
            overlap = len(top_d_rank1.intersection(top_d_rank2))

            # Add weighted contribution
            rbo_sum += weights[d-1] * (overlap / d)

        similarities[i] = rbo_sum

    return similarities


def compute_rbo_distance(v1, v2, p=None):
    """
    Compute RBO distance (1 - RBO similarity).

    Returns distance metric where 0 = perfect agreement (consistent with L2 and Jaccard).

    Args:
        v1, v2: Arrays of shape (N, F)
        p: Persistence parameter. If None, uses p = 1 - (1/n_features)

    Returns:
        Array of shape (N,) with RBO distance in [0, 1]
    """
    return 1.0 - compute_rbo_similarity(v1, v2, p=p)


# ========================================================
# Analysis Engine
# ========================================================

class SensitivityAnalyzer:
    """
    Main analysis engine for SHAP sensitivity analysis.

    Structure:
        - shap_5d: (S, M, E, N, F) where:
            S = splits (folds)
            M = model seeds
            E = explainer seeds
            N = samples
            F = features
        - proba_4d: (S, M, N, C) where C = classes
    """

    def __init__(self, shap_5d, proba_4d=None, y_test_list=None, feature_names=None):
        self.shap_5d = shap_5d
        self.proba_4d = proba_4d
        self.y_test_list = y_test_list
        self.feature_names = feature_names

        self.S, self.M, self.E, self.N_max, self.F = shap_5d.shape

    # ========================================================
    # 1-a: Overall Sensitivity (All 25 seeds pooled)
    # ========================================================

    def compute_overall_pooled(self):
        """
        Compute overall sensitivity by comparing all 25 seed combinations.

        Returns:
            dict: {'l2': [...], 'jaccard': [...], 'rbo': [...]} with per-sample distances
        """
        l2_all = []
        jac_all = []
        rbo_all = []

        for s in range(self.S):
            # Collect all 25 vectors (5 model seeds × 5 explainer seeds)
            vectors = []
            for m in range(self.M):
                for e in range(self.E):
                    vectors.append(self.shap_5d[s, m, e])

            n_vectors = len(vectors)
            if n_vectors < 2:
                continue

            n_samples = vectors[0].shape[0]

            # Accumulate distances from all pairs
            sum_l2 = np.zeros(n_samples)
            sum_jac = np.zeros(n_samples)
            sum_rbo = np.zeros(n_samples)
            pair_counts = np.zeros(n_samples)

            # Compare all pairs (300 comparisons for 25 vectors)
            for i in range(n_vectors):
                for j in range(i + 1, n_vectors):
                    v1 = vectors[i]
                    v2 = vectors[j]

                    # Valid data mask (exclude NaN padding)
                    valid = ~np.isnan(v1[:, 0]) & ~np.isnan(v2[:, 0])

                    if not np.any(valid):
                        continue

                    # Compute distances
                    d_l2 = compute_l2_distance(v1[valid], v2[valid])
                    d_jac = compute_jaccard_distance(v1[valid], v2[valid], k=Config.JACCARD_K)
                    d_rbo = compute_rbo_distance(v1[valid], v2[valid], p=None)

                    # Accumulate
                    sum_l2[valid] += d_l2
                    sum_jac[valid] += d_jac
                    sum_rbo[valid] += d_rbo
                    pair_counts[valid] += 1

            # Average over all pairs
            valid_instances = pair_counts > 0
            if np.any(valid_instances):
                avg_l2 = sum_l2[valid_instances] / pair_counts[valid_instances]
                avg_jac = sum_jac[valid_instances] / pair_counts[valid_instances]
                avg_rbo = sum_rbo[valid_instances] / pair_counts[valid_instances]

                l2_all.extend(avg_l2)
                jac_all.extend(avg_jac)
                rbo_all.extend(avg_rbo)

        return {'l2': l2_all, 'jaccard': jac_all, 'rbo': rbo_all}

    # ========================================================
    # 1-b: Separated Sensitivity (Explainer vs Model)
    # ========================================================

    def compute_separated_sensitivity(self):
        """
        Compute sensitivity separately for:
        - Explainer: Fix model seed, vary explainer seed
        - Model: Fix explainer seed, vary model seed

        For each data point:
        - Explainer: Compute pairwise average for each model seed, then average across model seeds
        - Model: Compute pairwise average for each explainer seed, then average across explainer seeds

        Returns:
            dict: {
                'explainer': {'l2': [...], 'jaccard': [...], 'rbo': [...]},
                'model': {'l2': [...], 'jaccard': [...], 'rbo': [...]}
            }
        """
        results = {'explainer': {}, 'model': {}}

        # === Explainer Sensitivity ===
        # For each split, accumulate per-data-point averages
        for s in range(self.S):
            # Get reference valid mask (same across all seeds for a split)
            ref_vec = self.shap_5d[s, 0, 0]
            valid = ~np.isnan(ref_vec[:, 0])
            n_valid = np.sum(valid)

            if n_valid == 0:
                continue

            # Accumulate distances for each model seed
            all_l2_per_model = []
            all_jac_per_model = []
            all_rbo_per_model = []

            for m in range(self.M):  # Fix model seed
                # Get all explainer seed vectors
                vectors = []
                for e in range(self.E):
                    v = self.shap_5d[s, m, e]
                    if not np.isnan(v[:, 0]).all():
                        vectors.append(v[valid])

                if len(vectors) < 2:
                    continue

                # Pairwise comparisons (10 pairs for 5 vectors)
                sum_l2 = np.zeros(n_valid)
                sum_jac = np.zeros(n_valid)
                sum_rbo = np.zeros(n_valid)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i], vectors[j])
                        d_jac = compute_jaccard_distance(vectors[i], vectors[j], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i], vectors[j], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    all_l2_per_model.append(sum_l2 / count)
                    all_jac_per_model.append(sum_jac / count)
                    all_rbo_per_model.append(sum_rbo / count)

            # Average across model seeds for each data point
            if all_l2_per_model:
                avg_l2 = np.mean(all_l2_per_model, axis=0)
                avg_jac = np.mean(all_jac_per_model, axis=0)
                avg_rbo = np.mean(all_rbo_per_model, axis=0)

                if 'l2' not in results['explainer']:
                    results['explainer']['l2'] = []
                    results['explainer']['jaccard'] = []
                    results['explainer']['rbo'] = []

                results['explainer']['l2'].extend(avg_l2)
                results['explainer']['jaccard'].extend(avg_jac)
                results['explainer']['rbo'].extend(avg_rbo)

        # === Model Sensitivity ===
        for s in range(self.S):
            # Get reference valid mask
            ref_vec = self.shap_5d[s, 0, 0]
            valid = ~np.isnan(ref_vec[:, 0])
            n_valid = np.sum(valid)

            if n_valid == 0:
                continue

            # Accumulate distances for each explainer seed
            all_l2_per_explainer = []
            all_jac_per_explainer = []
            all_rbo_per_explainer = []

            for e in range(self.E):  # Fix explainer seed
                # Get all model seed vectors
                vectors = []
                for m in range(self.M):
                    v = self.shap_5d[s, m, e]
                    if not np.isnan(v[:, 0]).all():
                        vectors.append(v[valid])

                if len(vectors) < 2:
                    continue

                sum_l2 = np.zeros(n_valid)
                sum_jac = np.zeros(n_valid)
                sum_rbo = np.zeros(n_valid)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i], vectors[j])
                        d_jac = compute_jaccard_distance(vectors[i], vectors[j], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i], vectors[j], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    all_l2_per_explainer.append(sum_l2 / count)
                    all_jac_per_explainer.append(sum_jac / count)
                    all_rbo_per_explainer.append(sum_rbo / count)

            # Average across explainer seeds for each data point
            if all_l2_per_explainer:
                avg_l2 = np.mean(all_l2_per_explainer, axis=0)
                avg_jac = np.mean(all_jac_per_explainer, axis=0)
                avg_rbo = np.mean(all_rbo_per_explainer, axis=0)

                if 'l2' not in results['model']:
                    results['model']['l2'] = []
                    results['model']['jaccard'] = []
                    results['model']['rbo'] = []

                results['model']['l2'].extend(avg_l2)
                results['model']['jaccard'].extend(avg_jac)
                results['model']['rbo'].extend(avg_rbo)

        return results

    # ========================================================
    # 1-c: Seed-wise Decomposition
    # ========================================================

    def compute_seedwise_explainer(self):
        """
        Compute explainer sensitivity for each model seed separately.

        Returns:
            dict: {model_seed_idx: {'l2': [...], 'jaccard': [...], 'rbo': [...]}}
        """
        results = {m: {'l2': [], 'jaccard': [], 'rbo': []} for m in range(self.M)}

        for s in range(self.S):
            for m in range(self.M):
                vectors = []
                for e in range(self.E):
                    v = self.shap_5d[s, m, e]
                    if not np.isnan(v[:, 0]).all():
                        vectors.append(v)

                if len(vectors) < 2:
                    continue

                ref_vec = vectors[0]
                valid = ~np.isnan(ref_vec[:, 0])
                n_valid = np.sum(valid)

                if n_valid == 0:
                    continue

                sum_l2 = np.zeros(n_valid)
                sum_jac = np.zeros(n_valid)
                sum_rbo = np.zeros(n_valid)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i][valid], vectors[j][valid])
                        d_jac = compute_jaccard_distance(vectors[i][valid], vectors[j][valid], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i][valid], vectors[j][valid], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    results[m]['l2'].extend(sum_l2 / count)
                    results[m]['jaccard'].extend(sum_jac / count)
                    results[m]['rbo'].extend(sum_rbo / count)

        return results

    def compute_seedwise_model(self):
        """
        Compute model sensitivity for each explainer seed separately.

        Returns:
            dict: {explainer_seed_idx: {'l2': [...], 'jaccard': [...], 'rbo': [...]}}
        """
        results = {e: {'l2': [], 'jaccard': [], 'rbo': []} for e in range(self.E)}

        for s in range(self.S):
            for e in range(self.E):
                vectors = []
                for m in range(self.M):
                    v = self.shap_5d[s, m, e]
                    if not np.isnan(v[:, 0]).all():
                        vectors.append(v)

                if len(vectors) < 2:
                    continue

                ref_vec = vectors[0]
                valid = ~np.isnan(ref_vec[:, 0])
                n_valid = np.sum(valid)

                if n_valid == 0:
                    continue

                sum_l2 = np.zeros(n_valid)
                sum_jac = np.zeros(n_valid)
                sum_rbo = np.zeros(n_valid)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i][valid], vectors[j][valid])
                        d_jac = compute_jaccard_distance(vectors[i][valid], vectors[j][valid], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i][valid], vectors[j][valid], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    results[e]['l2'].extend(sum_l2 / count)
                    results[e]['jaccard'].extend(sum_jac / count)
                    results[e]['rbo'].extend(sum_rbo / count)

        return results

    # ========================================================
    # 1-d & 2-b & 3-b: Feature-wise Sensitivity
    # ========================================================

    def compute_feature_sensitivity(self, mode='explainer', subgroup_mask=None):
        """
        Compute feature-wise instability.

        For each data point:
        - Explainer mode: Compute pairwise average for each model seed, then average across model seeds
        - Model mode: Compute pairwise average for each explainer seed, then average across explainer seeds

        Args:
            mode: 'explainer' or 'model'
            subgroup_mask: Optional dict {split: {model_seed: mask_array}}
                          for subgroup analysis

        Returns:
            Array of shape (Total_instances, F) with abs differences
        """
        instability_vectors = []

        if mode == 'explainer':
            for s in range(self.S):
                # Get mask for this split if subgroup analysis
                if subgroup_mask is not None:
                    if s not in subgroup_mask:
                        continue
                    # Use first model seed's mask (should be consistent)
                    if 0 not in subgroup_mask[s]:
                        continue
                    mask = subgroup_mask[s][0]
                else:
                    mask = None

                # Accumulate feature diffs for each model seed
                all_diffs_per_model = []

                for m in range(self.M):
                    # Collect vectors
                    vectors = []
                    for e in range(self.E):
                        v = self.shap_5d[s, m, e]
                        if np.isnan(v[:, 0]).all():
                            continue

                        # Apply subgroup mask if provided
                        if mask is not None:
                            # Get valid indices
                            valid_idx = ~np.isnan(v[:, 0])
                            if np.sum(valid_idx) != len(mask):
                                continue
                            v = v[valid_idx][mask]

                        vectors.append(v)

                    if len(vectors) < 2:
                        continue

                    # Pairwise absolute differences
                    pair_diffs = []
                    for i in range(len(vectors)):
                        for j in range(i + 1, len(vectors)):
                            diff = np.abs(vectors[i] - vectors[j])
                            pair_diffs.append(diff)

                    if pair_diffs:
                        mean_diff = np.mean(pair_diffs, axis=0)
                        all_diffs_per_model.append(mean_diff)

                # Average across model seeds for each data point
                if all_diffs_per_model:
                    avg_diff = np.mean(all_diffs_per_model, axis=0)
                    instability_vectors.append(avg_diff)

        elif mode == 'model':
            for s in range(self.S):
                # Accumulate feature diffs for each explainer seed
                all_diffs_per_explainer = []

                for e in range(self.E):
                    vectors = []
                    for m in range(self.M):
                        v = self.shap_5d[s, m, e]
                        if not np.isnan(v[:, 0]).all():
                            vectors.append(v)

                    if len(vectors) < 2:
                        continue

                    pair_diffs = []
                    for i in range(len(vectors)):
                        for j in range(i + 1, len(vectors)):
                            diff = np.abs(vectors[i] - vectors[j])
                            pair_diffs.append(diff)

                    if pair_diffs:
                        mean_diff = np.mean(pair_diffs, axis=0)
                        all_diffs_per_explainer.append(mean_diff)

                # Average across explainer seeds for each data point
                if all_diffs_per_explainer:
                    avg_diff = np.mean(all_diffs_per_explainer, axis=0)
                    instability_vectors.append(avg_diff)

        if not instability_vectors:
            return np.array([])

        return np.vstack(instability_vectors)

    # ========================================================
    # 2-a & 3-a: Subgroup Analysis (Certainty & Prediction)
    # ========================================================

    def compute_certainty_subgroups(self):
        """
        Compute explainer sensitivity for certainty-based subgroups.

        For each data point:
        - Compute pairwise average for each model seed, then average across model seeds

        Returns:
            dict: {
                'Certain': {'l2': [...], 'jaccard': [...], 'rbo': [...]},
                'Uncertain': {'l2': [...], 'jaccard': [...], 'rbo': [...]}
            }
        """
        if self.proba_4d is None:
            return {}

        results = {'Certain': {'l2': [], 'jaccard': [], 'rbo': []},
                  'Uncertain': {'l2': [], 'jaccard': [], 'rbo': []}}

        for s in range(self.S):
            # Get reference valid mask
            ref_vec = self.shap_5d[s, 0, 0]
            valid = ~np.isnan(ref_vec[:, 0])

            if not np.any(valid):
                continue

            # Get probabilities from first model seed (masks should be consistent)
            probs = self.proba_4d[s, 0][valid]
            if probs.ndim == 2:
                p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
            else:
                p = probs

            # Define subgroups
            masks = {
                'Certain': (p >= 0.9) | (p <= 0.1),
                'Uncertain': (p >= 0.4) & (p <= 0.6)
            }

            n_samples = len(p)

            # Accumulate distances for each model seed
            all_l2_per_model = []
            all_jac_per_model = []
            all_rbo_per_model = []

            for m in range(self.M):
                # Collect vectors for this (split, model_seed)
                vectors = []
                for e in range(self.E):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Compute pairwise distances
                sum_l2 = np.zeros(n_samples)
                sum_jac = np.zeros(n_samples)
                sum_rbo = np.zeros(n_samples)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i], vectors[j])
                        d_jac = compute_jaccard_distance(vectors[i], vectors[j], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i], vectors[j], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    all_l2_per_model.append(sum_l2 / count)
                    all_jac_per_model.append(sum_jac / count)
                    all_rbo_per_model.append(sum_rbo / count)

            # Average across model seeds for each data point
            if all_l2_per_model:
                avg_l2 = np.mean(all_l2_per_model, axis=0)
                avg_jac = np.mean(all_jac_per_model, axis=0)
                avg_rbo = np.mean(all_rbo_per_model, axis=0)

                # Store by subgroup
                for grp_name, mask in masks.items():
                    if np.any(mask):
                        results[grp_name]['l2'].extend(avg_l2[mask])
                        results[grp_name]['jaccard'].extend(avg_jac[mask])
                        results[grp_name]['rbo'].extend(avg_rbo[mask])

        return results

    def compute_prediction_subgroups(self):
        """
        Compute explainer sensitivity for prediction-based subgroups.

        For each data point:
        - Compute pairwise average for each model seed, then average across model seeds

        Returns:
            dict: {
                'TP': {'l2': [...], 'jaccard': [...], 'rbo': [...]},
                'TN': {...},
                'FP': {...},
                'FN': {...}
            }
        """
        if self.proba_4d is None or self.y_test_list is None:
            return {}

        results = {grp: {'l2': [], 'jaccard': [], 'rbo': []}
                  for grp in ['TP', 'TN', 'FP', 'FN']}

        for s in range(self.S):
            y_true = self.y_test_list[s]

            # Get reference valid mask
            ref_vec = self.shap_5d[s, 0, 0]
            valid = ~np.isnan(ref_vec[:, 0])

            if not np.any(valid):
                continue

            # Get probabilities from first model seed (masks should be consistent)
            probs = self.proba_4d[s, 0][valid]
            if probs.ndim == 2:
                p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
            else:
                p = probs

            # Get corresponding y_true
            n_valid = np.sum(valid)
            y_subset = y_true[:n_valid]

            # Predictions
            y_pred = (p >= 0.5).astype(int)
            y_gt = y_subset.astype(int)

            # Define subgroups
            masks = {
                'TP': (y_gt == 1) & (y_pred == 1),
                'TN': (y_gt == 0) & (y_pred == 0),
                'FP': (y_gt == 0) & (y_pred == 1),
                'FN': (y_gt == 1) & (y_pred == 0)
            }

            n_samples = len(p)

            # Accumulate distances for each model seed
            all_l2_per_model = []
            all_jac_per_model = []
            all_rbo_per_model = []

            for m in range(self.M):
                # Collect vectors
                vectors = []
                for e in range(self.E):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Compute pairwise distances
                sum_l2 = np.zeros(n_samples)
                sum_jac = np.zeros(n_samples)
                sum_rbo = np.zeros(n_samples)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i], vectors[j])
                        d_jac = compute_jaccard_distance(vectors[i], vectors[j], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i], vectors[j], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    all_l2_per_model.append(sum_l2 / count)
                    all_jac_per_model.append(sum_jac / count)
                    all_rbo_per_model.append(sum_rbo / count)

            # Average across model seeds for each data point
            if all_l2_per_model:
                avg_l2 = np.mean(all_l2_per_model, axis=0)
                avg_jac = np.mean(all_jac_per_model, axis=0)
                avg_rbo = np.mean(all_rbo_per_model, axis=0)

                for grp_name, mask in masks.items():
                    if np.any(mask):
                        results[grp_name]['l2'].extend(avg_l2[mask])
                        results[grp_name]['jaccard'].extend(avg_jac[mask])
                        results[grp_name]['rbo'].extend(avg_rbo[mask])

        return results

    # ========================================================
    # Helper: Get Subgroup Masks for Feature Analysis
    # ========================================================

    def get_certainty_masks(self):
        """
        Get masks for certainty-based subgroups.

        Returns:
            dict: {
                'Certain': {split: {model_seed: boolean_array}},
                'Uncertain': {...}
            }
        """
        if self.proba_4d is None:
            return {}

        masks = {'Certain': {}, 'Uncertain': {}}

        for s in range(self.S):
            masks['Certain'][s] = {}
            masks['Uncertain'][s] = {}

            for m in range(self.M):
                ref_vec = self.shap_5d[s, m, 0]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, m][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                masks['Certain'][s][m] = (p >= 0.9) | (p <= 0.1)
                masks['Uncertain'][s][m] = (p >= 0.4) & (p <= 0.6)

        return masks

    def get_prediction_masks(self):
        """
        Get masks for prediction-based subgroups.

        Returns:
            dict: {
                'TP': {split: {model_seed: boolean_array}},
                ...
            }
        """
        if self.proba_4d is None or self.y_test_list is None:
            return {}

        masks = {grp: {} for grp in ['TP', 'TN', 'FP', 'FN']}

        for grp in masks:
            for s in range(self.S):
                masks[grp][s] = {}

        for s in range(self.S):
            y_true = self.y_test_list[s]

            for m in range(self.M):
                ref_vec = self.shap_5d[s, m, 0]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, m][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                n_valid = np.sum(valid)
                y_subset = y_true[:n_valid]

                y_pred = (p >= 0.5).astype(int)
                y_gt = y_subset.astype(int)

                masks['TP'][s][m] = (y_gt == 1) & (y_pred == 1)
                masks['TN'][s][m] = (y_gt == 0) & (y_pred == 0)
                masks['FP'][s][m] = (y_gt == 0) & (y_pred == 1)
                masks['FN'][s][m] = (y_gt == 1) & (y_pred == 0)

        return masks

    # ========================================================
    # 2-b & 2-c: Separated Certainty Subgroup Sensitivity
    # ========================================================

    def compute_certainty_separated_sensitivity(self):
        """
        Compute certainty sensitivity separately for explainer vs model seeds.

        Returns:
            dict: {
                'explainer': {
                    'Certain': {'l2': [...], 'jaccard': [...], 'rbo': [...]},
                    'Uncertain': {...}
                },
                'model': {
                    'Certain': {'l2': [...], 'jaccard': [...], 'rbo': [...]},
                    'Uncertain': {...}
                }
            }
        """
        if self.proba_4d is None:
            return {}

        results = {
            'explainer': {'Certain': {'l2': [], 'jaccard': [], 'rbo': []},
                         'Uncertain': {'l2': [], 'jaccard': [], 'rbo': []}},
            'model': {'Certain': {'l2': [], 'jaccard': [], 'rbo': []},
                     'Uncertain': {'l2': [], 'jaccard': [], 'rbo': []}}
        }

        # === Explainer Sensitivity (fix model seed) ===
        for s in range(self.S):
            for m in range(self.M):
                # Get probabilities for this (split, model_seed)
                ref_vec = self.shap_5d[s, m, 0]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, m][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                # Define masks
                certain_mask = (p >= 0.9) | (p <= 0.1)
                uncertain_mask = (p >= 0.4) & (p <= 0.6)

                # Collect explainer seed vectors
                vectors = []
                for e in range(self.E):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Compute pairwise distances
                n_samples = len(p)
                sum_l2 = np.zeros(n_samples)
                sum_jac = np.zeros(n_samples)
                sum_rbo = np.zeros(n_samples)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i], vectors[j])
                        d_jac = compute_jaccard_distance(vectors[i], vectors[j], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i], vectors[j], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    avg_l2 = sum_l2 / count
                    avg_jac = sum_jac / count
                    avg_rbo = sum_rbo / count

                    # Store by subgroup
                    if np.any(certain_mask):
                        results['explainer']['Certain']['l2'].extend(avg_l2[certain_mask])
                        results['explainer']['Certain']['jaccard'].extend(avg_jac[certain_mask])
                        results['explainer']['Certain']['rbo'].extend(avg_rbo[certain_mask])

                    if np.any(uncertain_mask):
                        results['explainer']['Uncertain']['l2'].extend(avg_l2[uncertain_mask])
                        results['explainer']['Uncertain']['jaccard'].extend(avg_jac[uncertain_mask])
                        results['explainer']['Uncertain']['rbo'].extend(avg_rbo[uncertain_mask])

        # === Model Sensitivity (fix explainer seed) ===
        for s in range(self.S):
            for e in range(self.E):
                # Get probabilities (use first model seed's proba as reference)
                ref_vec = self.shap_5d[s, 0, e]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, 0][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                # Define masks
                certain_mask = (p >= 0.9) | (p <= 0.1)
                uncertain_mask = (p >= 0.4) & (p <= 0.6)

                # Collect model seed vectors
                vectors = []
                for m in range(self.M):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Compute pairwise distances
                n_samples = len(p)
                sum_l2 = np.zeros(n_samples)
                sum_jac = np.zeros(n_samples)
                sum_rbo = np.zeros(n_samples)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i], vectors[j])
                        d_jac = compute_jaccard_distance(vectors[i], vectors[j], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i], vectors[j], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    avg_l2 = sum_l2 / count
                    avg_jac = sum_jac / count
                    avg_rbo = sum_rbo / count

                    # Store by subgroup
                    if np.any(certain_mask):
                        results['model']['Certain']['l2'].extend(avg_l2[certain_mask])
                        results['model']['Certain']['jaccard'].extend(avg_jac[certain_mask])
                        results['model']['Certain']['rbo'].extend(avg_rbo[certain_mask])

                    if np.any(uncertain_mask):
                        results['model']['Uncertain']['l2'].extend(avg_l2[uncertain_mask])
                        results['model']['Uncertain']['jaccard'].extend(avg_jac[uncertain_mask])
                        results['model']['Uncertain']['rbo'].extend(avg_rbo[uncertain_mask])

        return results

    # ========================================================
    # 3-b & 3-c: Separated Prediction Subgroup Sensitivity
    # ========================================================

    def compute_prediction_separated_sensitivity(self):
        """
        Compute prediction sensitivity separately for explainer vs model seeds.

        Returns:
            dict: {
                'explainer': {
                    'TP': {'l2': [...], 'jaccard': [...], 'rbo': [...]},
                    'TN': {...}, 'FP': {...}, 'FN': {...}
                },
                'model': {
                    'TP': {'l2': [...], 'jaccard': [...], 'rbo': [...]},
                    'TN': {...}, 'FP': {...}, 'FN': {...}
                }
            }
        """
        if self.proba_4d is None or self.y_test_list is None:
            return {}

        results = {
            'explainer': {grp: {'l2': [], 'jaccard': [], 'rbo': []} for grp in ['TP', 'TN', 'FP', 'FN']},
            'model': {grp: {'l2': [], 'jaccard': [], 'rbo': []} for grp in ['TP', 'TN', 'FP', 'FN']}
        }

        # === Explainer Sensitivity (fix model seed) ===
        for s in range(self.S):
            y_true = self.y_test_list[s]

            for m in range(self.M):
                ref_vec = self.shap_5d[s, m, 0]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, m][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                # Get corresponding y_true
                n_valid = np.sum(valid)
                y_subset = y_true[:n_valid]

                # Predictions
                y_pred = (p >= 0.5).astype(int)
                y_gt = y_subset.astype(int)

                # Define masks
                masks = {
                    'TP': (y_gt == 1) & (y_pred == 1),
                    'TN': (y_gt == 0) & (y_pred == 0),
                    'FP': (y_gt == 0) & (y_pred == 1),
                    'FN': (y_gt == 1) & (y_pred == 0)
                }

                # Collect explainer seed vectors
                vectors = []
                for e in range(self.E):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Compute pairwise distances
                n_samples = len(p)
                sum_l2 = np.zeros(n_samples)
                sum_jac = np.zeros(n_samples)
                sum_rbo = np.zeros(n_samples)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i], vectors[j])
                        d_jac = compute_jaccard_distance(vectors[i], vectors[j], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i], vectors[j], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    avg_l2 = sum_l2 / count
                    avg_jac = sum_jac / count
                    avg_rbo = sum_rbo / count

                    # Store by subgroup
                    for grp_name, mask in masks.items():
                        if np.any(mask):
                            results['explainer'][grp_name]['l2'].extend(avg_l2[mask])
                            results['explainer'][grp_name]['jaccard'].extend(avg_jac[mask])
                            results['explainer'][grp_name]['rbo'].extend(avg_rbo[mask])

        # === Model Sensitivity (fix explainer seed) ===
        for s in range(self.S):
            y_true = self.y_test_list[s]

            for e in range(self.E):
                ref_vec = self.shap_5d[s, 0, e]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, 0][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                # Get corresponding y_true
                n_valid = np.sum(valid)
                y_subset = y_true[:n_valid]

                # Predictions
                y_pred = (p >= 0.5).astype(int)
                y_gt = y_subset.astype(int)

                # Define masks
                masks = {
                    'TP': (y_gt == 1) & (y_pred == 1),
                    'TN': (y_gt == 0) & (y_pred == 0),
                    'FP': (y_gt == 0) & (y_pred == 1),
                    'FN': (y_gt == 1) & (y_pred == 0)
                }

                # Collect model seed vectors
                vectors = []
                for m in range(self.M):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Compute pairwise distances
                n_samples = len(p)
                sum_l2 = np.zeros(n_samples)
                sum_jac = np.zeros(n_samples)
                sum_rbo = np.zeros(n_samples)
                count = 0

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        d_l2 = compute_l2_distance(vectors[i], vectors[j])
                        d_jac = compute_jaccard_distance(vectors[i], vectors[j], k=Config.JACCARD_K)
                        d_rbo = compute_rbo_distance(vectors[i], vectors[j], p=None)

                        sum_l2 += d_l2
                        sum_jac += d_jac
                        sum_rbo += d_rbo
                        count += 1

                if count > 0:
                    avg_l2 = sum_l2 / count
                    avg_jac = sum_jac / count
                    avg_rbo = sum_rbo / count

                    # Store by subgroup
                    for grp_name, mask in masks.items():
                        if np.any(mask):
                            results['model'][grp_name]['l2'].extend(avg_l2[mask])
                            results['model'][grp_name]['jaccard'].extend(avg_jac[mask])
                            results['model'][grp_name]['rbo'].extend(avg_rbo[mask])

        return results

    # ========================================================
    # 2-e & 2-f: Separated Feature-wise Certainty Sensitivity
    # ========================================================

    def compute_certainty_feature_separated(self):
        """
        Compute feature-wise certainty sensitivity separately for explainer vs model seeds.

        Returns:
            dict: {
                'explainer': {
                    'Certain': array(N_instances, N_features),
                    'Uncertain': array(...)
                },
                'model': {
                    'Certain': array(...),
                    'Uncertain': array(...)
                }
            }
        """
        if self.proba_4d is None:
            return {}

        results = {
            'explainer': {'Certain': [], 'Uncertain': []},
            'model': {'Certain': [], 'Uncertain': []}
        }

        # === Explainer Sensitivity (fix model seed) ===
        for s in range(self.S):
            for m in range(self.M):
                # Get probabilities
                ref_vec = self.shap_5d[s, m, 0]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, m][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                # Define masks
                certain_mask = (p >= 0.9) | (p <= 0.1)
                uncertain_mask = (p >= 0.4) & (p <= 0.6)

                # Collect explainer seed vectors
                vectors = []
                for e in range(self.E):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Pairwise absolute differences
                pair_diffs = []
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        diff = np.abs(vectors[i] - vectors[j])
                        pair_diffs.append(diff)

                if pair_diffs:
                    mean_diff = np.mean(pair_diffs, axis=0)  # Shape: (N_samples, N_features)

                    # Filter by subgroup
                    if np.any(certain_mask):
                        results['explainer']['Certain'].append(mean_diff[certain_mask])

                    if np.any(uncertain_mask):
                        results['explainer']['Uncertain'].append(mean_diff[uncertain_mask])

        # === Model Sensitivity (fix explainer seed) ===
        for s in range(self.S):
            for e in range(self.E):
                # Get probabilities
                ref_vec = self.shap_5d[s, 0, e]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, 0][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                # Define masks
                certain_mask = (p >= 0.9) | (p <= 0.1)
                uncertain_mask = (p >= 0.4) & (p <= 0.6)

                # Collect model seed vectors
                vectors = []
                for m in range(self.M):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Pairwise absolute differences
                pair_diffs = []
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        diff = np.abs(vectors[i] - vectors[j])
                        pair_diffs.append(diff)

                if pair_diffs:
                    mean_diff = np.mean(pair_diffs, axis=0)  # Shape: (N_samples, N_features)

                    # Filter by subgroup
                    if np.any(certain_mask):
                        results['model']['Certain'].append(mean_diff[certain_mask])

                    if np.any(uncertain_mask):
                        results['model']['Uncertain'].append(mean_diff[uncertain_mask])

        # Convert lists to arrays
        for sensitivity_type in ['explainer', 'model']:
            for subgroup in ['Certain', 'Uncertain']:
                if results[sensitivity_type][subgroup]:
                    results[sensitivity_type][subgroup] = np.vstack(results[sensitivity_type][subgroup])
                else:
                    results[sensitivity_type][subgroup] = np.array([])

        return results

    # ========================================================
    # 3-e & 3-f: Separated Feature-wise Prediction Sensitivity
    # ========================================================

    def compute_prediction_feature_separated(self):
        """
        Compute feature-wise prediction sensitivity separately for explainer vs model seeds.

        Returns:
            dict: {
                'explainer': {
                    'TP': array(N_instances, N_features),
                    'TN': array(...), 'FP': array(...), 'FN': array(...)
                },
                'model': {
                    'TP': array(...), 'TN': array(...), 'FP': array(...), 'FN': array(...)
                }
            }
        """
        if self.proba_4d is None or self.y_test_list is None:
            return {}

        results = {
            'explainer': {grp: [] for grp in ['TP', 'TN', 'FP', 'FN']},
            'model': {grp: [] for grp in ['TP', 'TN', 'FP', 'FN']}
        }

        # === Explainer Sensitivity (fix model seed) ===
        for s in range(self.S):
            y_true = self.y_test_list[s]

            for m in range(self.M):
                ref_vec = self.shap_5d[s, m, 0]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, m][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                # Get corresponding y_true
                n_valid = np.sum(valid)
                y_subset = y_true[:n_valid]

                # Predictions
                y_pred = (p >= 0.5).astype(int)
                y_gt = y_subset.astype(int)

                # Define masks
                masks = {
                    'TP': (y_gt == 1) & (y_pred == 1),
                    'TN': (y_gt == 0) & (y_pred == 0),
                    'FP': (y_gt == 0) & (y_pred == 1),
                    'FN': (y_gt == 1) & (y_pred == 0)
                }

                # Collect explainer seed vectors
                vectors = []
                for e in range(self.E):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Pairwise absolute differences
                pair_diffs = []
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        diff = np.abs(vectors[i] - vectors[j])
                        pair_diffs.append(diff)

                if pair_diffs:
                    mean_diff = np.mean(pair_diffs, axis=0)  # Shape: (N_samples, N_features)

                    # Filter by subgroup
                    for grp_name, mask in masks.items():
                        if np.any(mask):
                            results['explainer'][grp_name].append(mean_diff[mask])

        # === Model Sensitivity (fix explainer seed) ===
        for s in range(self.S):
            y_true = self.y_test_list[s]

            for e in range(self.E):
                ref_vec = self.shap_5d[s, 0, e]
                valid = ~np.isnan(ref_vec[:, 0])

                if not np.any(valid):
                    continue

                probs = self.proba_4d[s, 0][valid]
                if probs.ndim == 2:
                    p = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    p = probs

                # Get corresponding y_true
                n_valid = np.sum(valid)
                y_subset = y_true[:n_valid]

                # Predictions
                y_pred = (p >= 0.5).astype(int)
                y_gt = y_subset.astype(int)

                # Define masks
                masks = {
                    'TP': (y_gt == 1) & (y_pred == 1),
                    'TN': (y_gt == 0) & (y_pred == 0),
                    'FP': (y_gt == 0) & (y_pred == 1),
                    'FN': (y_gt == 1) & (y_pred == 0)
                }

                # Collect model seed vectors
                vectors = []
                for m in range(self.M):
                    v = self.shap_5d[s, m, e][valid]
                    vectors.append(v)

                if len(vectors) < 2:
                    continue

                # Pairwise absolute differences
                pair_diffs = []
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        diff = np.abs(vectors[i] - vectors[j])
                        pair_diffs.append(diff)

                if pair_diffs:
                    mean_diff = np.mean(pair_diffs, axis=0)  # Shape: (N_samples, N_features)

                    # Filter by subgroup
                    for grp_name, mask in masks.items():
                        if np.any(mask):
                            results['model'][grp_name].append(mean_diff[mask])

        # Convert lists to arrays
        for sensitivity_type in ['explainer', 'model']:
            for subgroup in ['TP', 'TN', 'FP', 'FN']:
                if results[sensitivity_type][subgroup]:
                    results[sensitivity_type][subgroup] = np.vstack(results[sensitivity_type][subgroup])
                else:
                    results[sensitivity_type][subgroup] = np.array([])

        return results

    # ========================================================
    # Mean Absolute SHAP Value Computation
    # ========================================================

    def compute_mean_abs_shap(self, subgroup_mask=None):
        """
        Compute mean absolute SHAP values for each feature.

        Process:
        1. Take absolute value of SHAP values
        2. Average over explainer seeds (5)
        3. Average over model seeds (5)
        4. Average over all instances (optionally filtered by subgroup_mask)

        Args:
            subgroup_mask: Optional dict {split: {model_seed: boolean_array}} or list of masks per split

        Returns:
            Array of shape (F,) with mean absolute SHAP value per feature
        """
        all_mean_shap = []

        for s in range(self.S):
            # Get valid instances for this split
            ref_vec = self.shap_5d[s, 0, 0]
            valid = ~np.isnan(ref_vec[:, 0])

            if not np.any(valid):
                continue

            # Apply subgroup mask if provided
            # Handle both dict and list formats
            if subgroup_mask is not None:
                if isinstance(subgroup_mask, dict):
                    # Dict format: {split: {model_seed: mask}}
                    # We need to aggregate masks across all model seeds for this split
                    if s in subgroup_mask:
                        # Combine masks from all model seeds (use OR to include all masked instances)
                        split_masks = []
                        for m in range(self.M):
                            if m in subgroup_mask[s]:
                                # Create full mask with False for invalid instances
                                full_mask = np.zeros(len(valid), dtype=bool)
                                full_mask[valid] = subgroup_mask[s][m]
                                split_masks.append(full_mask)

                        if split_masks:
                            # Combine all model seed masks (OR operation)
                            combined_mask = np.logical_or.reduce(split_masks)
                            mask = valid & combined_mask
                        else:
                            mask = valid
                    else:
                        mask = valid
                elif isinstance(subgroup_mask, list) and s < len(subgroup_mask):
                    # List format: [mask_split0, mask_split1, ...]
                    mask = valid & subgroup_mask[s]
                else:
                    mask = valid
            else:
                mask = valid

            if not np.any(mask):
                continue

            # Collect SHAP values: shape (M, E, N_masked, F)
            shap_values = []
            for m in range(self.M):
                explainer_means = []
                for e in range(self.E):
                    v = self.shap_5d[s, m, e][mask]  # Shape: (N_masked, F)
                    explainer_means.append(np.abs(v))

                # Average over explainer seeds: shape (N_masked, F)
                mean_over_e = np.mean(explainer_means, axis=0)
                shap_values.append(mean_over_e)

            # Average over model seeds: shape (N_masked, F)
            mean_over_m = np.mean(shap_values, axis=0)

            # Collect all instances from this split
            all_mean_shap.append(mean_over_m)

        if not all_mean_shap:
            return np.array([])

        # Concatenate all splits and average over instances: shape (F,)
        all_instances = np.vstack(all_mean_shap)
        mean_per_feature = np.mean(all_instances, axis=0)

        return mean_per_feature


# ========================================================
# Plotting Functions
# ========================================================

class PlotHelper:
    """Helper class for creating publication-quality plots"""

    @staticmethod
    def setup_axis(ax, ylabel, xlabel='Model', ylim=None, fontsize_scale=1.0):
        """Configure axis with consistent styling

        Args:
            fontsize_scale: Multiplier for font sizes (1.0 for single plots, larger for subplots)
        """
        base_label_size = 26
        base_tick_size = 26

        ax.set_ylabel(ylabel, fontsize=int(base_label_size * fontsize_scale))
        ax.set_xlabel(xlabel, fontsize=int(base_label_size * fontsize_scale))
        ax.grid(axis='y', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='both', labelsize=int(base_tick_size * fontsize_scale))

        if ylim is not None:
            ax.set_ylim(ylim)

        # Prevent label overlap
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    @staticmethod
    def add_legend(ax, loc='center left', ncol=1, bbox=(1.04, 1), fontsize_scale=1.0):
        """Add legend above the plot area
        Args:
            loc: Legend location
            bbox: bbox_to_anchor position
            fontsize_scale: Multiplier for font sizes (1.0 for single plots, larger for subplots)
        """
        base_legend_size = 26

        # legend = ax.legend(loc=loc, bbox_to_anchor=bbox, ncol=ncol,
        #                   frameon=True, fontsize=int(base_legend_size * fontsize_scale),
        #                   columnspacing=1.0,mode='expand')
        
        legend = ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=ncol, fontsize=26,columnspacing=1.0)
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(0.5)

    @staticmethod
    def save_fig(fig, filename, dpi=300):
        """Save figure with high quality to figures directory as PDF"""
        # Create figures directory if it doesn't exist
        os.makedirs(Config.FIGURE_DIR, exist_ok=True)

        # Add dataset prefix and change extension to .pdf
        base_name = filename.replace('.png', '')
        pdf_filename = f'{Config.DATASET}_{base_name}.pdf'

        # Full path
        filepath = os.path.join(Config.FIGURE_DIR, pdf_filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format='pdf')
        print(f"  💾 Saved: {filepath}")


def plot_1a_overall_pooled(results_dict, metric='l2', show_baseline=True,
                           baseline_params=None):
    """
    Plot 1-a: Overall Sensitivity (All 25 seeds pooled)

    Shows the overall instability when comparing all 25 seed combinations
    without distinguishing between model and explainer seeds.

    Args:
        show_baseline: If True, add baseline expectation line
        baseline_params: Dict with 'd', 'r', 'rho', 'kappa', 'T' (optional)
    """
    print(f"\n{'='*60}")
    print(f"Plot 1-a: Overall Sensitivity (All 25 pooled) - {metric.upper()}")
    print(f"{'='*60}")

    # Prepare data
    data = []
    n_features = None
    for model_name, model_results in results_dict.items():
        values = model_results['overall_pooled'][metric]

        # Get number of features
        if n_features is None and 'feature_names' in model_results:
            n_features = len(model_results['feature_names'])

        for v in values:
            data.append({'Model': model_name, 'Value': v})

    df = pd.DataFrame(data)

    if df.empty:
        print("⚠️  No data available")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Violin plot with model-specific colors
    model_order = Config.MODELS
    palette = [Config.MODEL_COLORS[m] for m in model_order]

    sns.violinplot(data=df, x='Model', y='Value', order=model_order,
                   palette=palette, ax=ax, linewidth=1.2, cut=0, bw_adjust=1.0, scale='width')

    # Add baseline line
    if show_baseline and n_features is not None:
        if baseline_params is None:
            baseline_params = {}

        d = baseline_params.get('d', n_features)
        r = baseline_params.get('r', Config.JACCARD_K)

        if metric == 'jaccard':
            baseline = compute_jaccard_baseline(d, r, q_values=[0.3, 0.4, 0.5])
            baseline_min = baseline['min']
            baseline_max = baseline['max']

            # Draw baseline shaded region
            ax.axhspan(ymin=baseline_min, ymax=baseline_max,
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

            print(f"\n📊 Jaccard Baseline (d={d}, r={r}, q=[0.3,0.4,0.5]):")
            print(f"   Range: [{baseline_min:.4f}, {baseline_max:.4f}]")

        elif metric == 'rbo':
            p = 1.0 - (1.0 / d)  # Compute p dynamically
            baseline = compute_rbo_baseline(n_features=d, p=p, q_values=[0.3, 0.4, 0.5])
            baseline_min = baseline['min']
            baseline_max = baseline['max']

            # Draw baseline shaded region
            ax.axhspan(ymin=baseline_min, ymax=baseline_max,
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

            print(f"\n📊 RBO Baseline (p={p:.4f}, d={d}, q=[0.3,0.4,0.5]):")
            print(f"   Range: [{baseline_min:.4f}, {baseline_max:.4f}]")

        elif metric == 'l2':
            baseline = compute_l2_baseline(d, r, rho_values=[0.6, 0.7, 0.8])
            baseline_min = baseline['min']
            baseline_max = baseline['max']

            # Draw baseline shaded region
            ax.axhspan(ymin=baseline_min, ymax=baseline_max,
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

            print(f"\n📊 L2 Baseline (d={d}, r={r}, ρ=[0.5,0.6,0.7]):")
            print(f"   Range: [{baseline_min:.4f}, {baseline_max:.4f}]")

    # Styling
    if metric == 'l2':
        ylabel = 'L2 Distance'
        ylim = Config.L2_YLIM
    elif metric == 'jaccard':
        ylabel = 'Jaccard Distance'
        ylim = Config.JACCARD_YLIM
    elif metric == 'rbo':
        ylabel = 'RBO Distance'
        ylim = Config.RBO_YLIM
    else:
        ylabel = f'{metric.upper()} Distance'
        ylim = None
    PlotHelper.setup_axis(ax, ylabel=ylabel, ylim=ylim, fontsize_scale=1.5)

    # Add legend (same style as Plot 3-a)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=Config.MODEL_COLORS[m], label=m)
                      for m in model_order]

    if show_baseline and n_features is not None:
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=39,columnspacing=1.0)
    else:
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=39,columnspacing=1.0)

    plt.tight_layout()
    plt.show()

    # Save figure
    PlotHelper.save_fig(fig, f'plot_1a_overall_{metric}_violin.png')

    # Print statistics
    print(f"\nStatistics (mean ± std):")
    stats = df.groupby('Model')['Value'].agg(['mean', 'std', 'count'])
    print(stats)


def plot_1b_separated(results_dict, metric='l2', show_baseline=True,
                     baseline_params=None):
    """
    Plot 1-b: Separated Sensitivity (Explainer vs Model)

    Compares explainer sensitivity (fix model, vary explainer)
    vs model sensitivity (fix explainer, vary model).

    Args:
        show_baseline: If True, add baseline expectation line
        baseline_params: Dict with 'd', 'r', 'rho', 'kappa', 'T' (optional)
    """
    print(f"\n{'='*60}")
    print(f"Plot 1-b: Separated Sensitivity - {metric.upper()}")
    print(f"{'='*60}")

    # Prepare data
    data = []
    n_features = None
    for model_name, model_results in results_dict.items():
        sep = model_results['separated']

        # Get number of features
        if n_features is None and 'feature_names' in model_results:
            n_features = len(model_results['feature_names'])

        for v in sep['explainer'][metric]:
            data.append({'Model': model_name, 'Type': 'Explainer', 'Value': v})

        for v in sep['model'][metric]:
            data.append({'Model': model_name, 'Type': 'Model', 'Value': v})

    df = pd.DataFrame(data)

    if df.empty:
        print("⚠️  No data available")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Split violin plot with hue
    model_order = Config.MODELS
    sns.violinplot(data=df, x='Model', y='Value', hue='Type', order=model_order,
                   palette=[Config.COLORS['explainer'], Config.COLORS['model']],
                   ax=ax, split=True, linewidth=1.2, cut=0, bw_adjust=1.0, scale='width')

    # Add baseline line
    if show_baseline and n_features is not None:
        if baseline_params is None:
            baseline_params = {}

        d = baseline_params.get('d', n_features)
        r = baseline_params.get('r', Config.JACCARD_K)

        if metric == 'jaccard':
            baseline = compute_jaccard_baseline(d, r, q_values=[0.3, 0.4, 0.5])
            ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

        elif metric == 'rbo':
            p = 1.0 - (1.0 / d)
            baseline = compute_rbo_baseline(n_features=d, p=p, q_values=[0.3, 0.4, 0.5])
            ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

        elif metric == 'l2':
            baseline = compute_l2_baseline(d, r, rho_values=[0.6, 0.7, 0.8])
            ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

    # Styling
    if metric == 'l2':
        ylabel = 'L2 Distance'
        ylim = Config.L2_YLIM
    elif metric == 'jaccard':
        ylabel = 'Jaccard Distance'
        ylim = Config.JACCARD_YLIM
    elif metric == 'rbo':
        ylabel = 'RBO Distance'
        ylim = Config.RBO_YLIM
    else:
        ylabel = f'{metric.upper()} Distance'
        ylim = None
    PlotHelper.setup_axis(ax, ylabel=ylabel, ylim=ylim, fontsize_scale=1.5)
    # PlotHelper.add_legend(ax, loc='upper center', ncol=3, bbox=(1.04, 1))
    PlotHelper.add_legend(ax, loc="lower left", ncol=3, bbox=(0, 1.02, 1, 0.2), fontsize_scale=1.5)

    plt.tight_layout()
    plt.show()

    # Save figure
    PlotHelper.save_fig(fig, f'plot_1b_separated_{metric}_violin.png')

    # Print statistics
    print(f"\nStatistics (mean ± std):")
    stats = df.groupby(['Model', 'Type'])['Value'].agg(['mean', 'std', 'count'])
    print(stats)


def plot_1c_seedwise(results_dict, model_name, metric='l2', sensitivity_type='explainer',
                     show_baseline=True, baseline_params=None):
    """
    Plot 1-c: Seed-wise Decomposition

    Shows how sensitivity varies across different outer seeds.
    For explainer sensitivity: x-axis = model seed index
    For model sensitivity: x-axis = explainer seed index
    """
    print(f"\n{'='*60}")
    print(f"Plot 1-c: Seed-wise [{model_name}] ({sensitivity_type}) - {metric.upper()}")
    print(f"{'='*60}")

    # Get seed-wise data
    if sensitivity_type == 'explainer':
        seedwise = results_dict[model_name]['seedwise_explainer']
        xlabel = 'Model Seed'
        palette = 'Blues'
    else:
        seedwise = results_dict[model_name]['seedwise_model']
        xlabel = 'Explainer Seed'
        palette = 'Greens'

    # Prepare data
    data = []
    for seed_idx, values in seedwise.items():
        for v in values[metric]:
            data.append({'Seed': seed_idx, 'Value': v})

    df = pd.DataFrame(data)

    if df.empty:
        print("⚠️  No data available")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Violin plot
    sns.violinplot(data=df, x='Seed', y='Value', palette=palette,
                   ax=ax, linewidth=1.2, cut=0, bw_adjust=1.0, scale='width')

    # Add baseline for all metrics
    if show_baseline:
        if baseline_params is None:
            baseline_params = {}

        # Get number of features
        feature_names = results_dict[model_name].get('feature_names', [])
        n_features = len(feature_names) if feature_names else None

        if n_features is not None:
            if metric == 'jaccard':
                d = baseline_params.get('d', n_features)
                r = baseline_params.get('r', Config.JACCARD_K)
                baseline = compute_jaccard_baseline(d, r, q_values=[0.3, 0.4, 0.5])
                ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                          color='red', alpha=0.25, label='Baseline Range', zorder=0)
                ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=26,columnspacing=1.0)

            elif metric == 'rbo':
                d = baseline_params.get('d', n_features)
                p = 1.0 - (1.0 / d)
                baseline = compute_rbo_baseline(n_features=d, p=p, q_values=[0.3, 0.4, 0.5])
                ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                          color='red', alpha=0.25, label='Baseline Range', zorder=0)
                ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=26,columnspacing=1.0)

            elif metric == 'l2':
                d = baseline_params.get('d', n_features)
                r = baseline_params.get('r', Config.JACCARD_K)
                baseline = compute_l2_baseline(d, r, rho_values=[0.6, 0.7, 0.8])
                ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                          color='red', alpha=0.25, label='Baseline Range', zorder=0)
                ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=26,columnspacing=1.0)

    # Styling
    if metric == 'l2':
        ylabel = 'L2 Distance'
        ylim = Config.L2_YLIM
    elif metric == 'jaccard':
        ylabel = 'Jaccard Distance'
        ylim = Config.JACCARD_YLIM
    elif metric == 'rbo':
        ylabel = 'RBO Distance'
        ylim = Config.RBO_YLIM
    else:
        ylabel = f'{metric.upper()} Distance'
        ylim = None
    PlotHelper.setup_axis(ax, ylabel=ylabel, xlabel=xlabel, ylim=ylim)

    plt.tight_layout()
    plt.show()

    # Save figure
    PlotHelper.save_fig(fig, f'plot_1c_seedwise_{model_name}_{sensitivity_type}_{metric}_violin.png')


def plot_1d_feature_overall(results_dict, model_name, mode='explainer', use_boxplot=True):
    """
    Plot 1-d: Feature-wise Overall Sensitivity

    Shows which features are most unstable globally for a specific mode.
    If use_boxplot=True, shows distribution; otherwise shows mean only.
    Y-axis labels show: feature_name=mean_abs_shap_value

    Args:
        mode: 'explainer' or 'model' - which seed type to analyze
    """
    print(f"\n{'='*60}")
    print(f"Plot 1-d: Feature-wise Overall ({mode.capitalize()}) [{model_name}]")
    print(f"{'='*60}")

    feature_data = results_dict[model_name]['feature_overall']
    feature_names = results_dict[model_name]['feature_names']

    # Get mean absolute SHAP values for y-axis labels
    mean_abs_shap = results_dict[model_name].get('mean_abs_shap', None)

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 8))

    inst_matrix = feature_data[mode]  # Shape: (N_instances, F)

    if inst_matrix.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(f'{mode.capitalize()} Sensitivity')
        plt.tight_layout()
        plt.show()
        return

    if use_boxplot:
        # Show distribution for top-K features
        # Sort by mean absolute SHAP value (highest first, so highest appears at top of barh)
        if mean_abs_shap is not None and len(mean_abs_shap) == inst_matrix.shape[1]:
            # Use mean_abs_shap for sorting (descending order - largest at top)
            top_k_idx = np.argsort(-np.array(mean_abs_shap))[:Config.TOP_K_FEATURES]
        else:
            # Fallback: sort by instability (descending order - largest at top)
            mean_vals = np.mean(inst_matrix, axis=0)
            top_k_idx = np.argsort(-mean_vals)[:Config.TOP_K_FEATURES]

        # Create feature labels with mean SHAP values
        feature_labels = []
        for feat_idx in top_k_idx:
            fname = feature_names[feat_idx] if feature_names else f'F{feat_idx}'
            # Shorten long feature names
            if fname == 'DiabetesPedigreeFunction':
                fname = 'PDF'
            elif fname == 'installment_commitment':
                fname = 'installment'
            elif 'checking_status' in fname:
                fname = 'checking_status'
            elif 'credit_history' in fname:
                fname = 'credit_history'
            elif fname.startswith('Race_'):
                fname = 'Race'
            elif fname.startswith('Occupation_'):
                fname = 'Occupation'
            if mean_abs_shap is not None and feat_idx < len(mean_abs_shap):
                label = f'{fname}\n={mean_abs_shap[feat_idx]:.3f}'
            else:
                label = fname
            feature_labels.append(label)

        # Prepare data
        plot_data = []
        for feat_idx, label in zip(top_k_idx, feature_labels):
            for val in inst_matrix[:, feat_idx]:
                plot_data.append({'Feature': label, 'Instability': val})

        df = pd.DataFrame(plot_data)

        # Preserve order by using categorical
        df['Feature'] = pd.Categorical(df['Feature'], categories=feature_labels, ordered=True)

        # Boxplot with increased spacing
        sns.boxplot(data=df, y='Feature', x='Instability',
                   palette='viridis', ax=ax, fliersize=1.5, linewidth=1.0,
                   order=feature_labels, width=0.6)

        ax.set_xlabel('Mean Absolute Difference', fontsize=26)
        ax.set_ylabel('Feature', fontsize=26)

        # Increase y-spacing to prevent label overlap
        ax.tick_params(axis='y', pad=10)
    else:
        # Bar plot of means
        # Sort by mean absolute SHAP value (highest first, so highest appears at top of barh)
        if mean_abs_shap is not None and len(mean_abs_shap) == inst_matrix.shape[1]:
            # Use mean_abs_shap for sorting (descending order - largest at top)
            top_k_idx = np.argsort(-np.array(mean_abs_shap))[:Config.TOP_K_FEATURES]
        else:
            # Fallback: sort by instability (descending order - largest at top)
            mean_vals = np.mean(inst_matrix, axis=0)
            top_k_idx = np.argsort(-mean_vals)[:Config.TOP_K_FEATURES]

        # Create feature labels with mean SHAP values
        top_names = []
        for i in top_k_idx:
            fname = feature_names[i] if feature_names else f'F{i}'
            # Shorten long feature names
            if fname == 'DiabetesPedigreeFunction':
                fname = 'PDF'
            elif fname == 'installment_commitment':
                fname = 'installment'
            if mean_abs_shap is not None and i < len(mean_abs_shap):
                top_names.append(f'{fname}\n={mean_abs_shap[i]:.3f}')
            else:
                top_names.append(fname)

        mean_vals = np.mean(inst_matrix, axis=0)
        top_vals = mean_vals[top_k_idx]

        ax.barh(top_names, top_vals, height=0.6, color='steelblue')
        ax.set_xlabel('Mean Absolute Difference', fontsize=26)
        ax.set_ylabel('Feature', fontsize=26)

    ax.set_title(f'{mode.capitalize()}', fontsize=26, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=26)
    ax.tick_params(axis='y', pad=10)

    # Set x-axis limit for cross-dataset comparison
    if Config.FEATURE_L2_YLIM is not None:
        ax.set_xlim(Config.FEATURE_L2_YLIM)

    plt.tight_layout()
    plt.show()

    # Save figure
    PlotHelper.save_fig(fig, f'plot_1d_feature_overall_{model_name}_{mode}_violin.png')

def plot_2a_certainty(results_dict, metric='l2', show_baseline=True, baseline_params=None):
    """
    Plot 2-a: Certainty-based Sensitivity (Explainer Seeds Only)

    Shows explainer sensitivity across certainty subgroups (Certain vs Uncertain).
    Only explainer seed variation is used because model seed changes affect certainty masks.

    Args:
        metric: 'l2' or 'jaccard'
        show_baseline: Whether to show random baseline (only for Jaccard)
        baseline_params: Dict with 'd' and 'r' for baseline computation
    """
    print(f"\n{'='*60}")
    print(f"Plot 2-a: Certainty Sensitivity - {metric.upper()}")
    print(f"{'='*60}")

    # Prepare data
    data = []
    n_features = None
    for model_name, model_results in results_dict.items():
        if 'certainty_separated' not in model_results:
            continue

        # Get number of features
        if n_features is None and 'feature_names' in model_results:
            n_features = len(model_results['feature_names'])

        for grp in ['Certain', 'Uncertain']:
            values = model_results['certainty_separated']['explainer'][grp][metric]
            for v in values:
                data.append({'Model': model_name, 'Group': grp, 'Value': v})

    df = pd.DataFrame(data)

    if df.empty:
        print("⚠️  No data available")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Violin plot
    group_order = ['Certain', 'Uncertain']
    model_order = Config.MODELS
    sns.violinplot(data=df, x='Group', y='Value', hue='Model', order=group_order,
                   hue_order=model_order, ax=ax, linewidth=1.0, cut=0, bw_adjust=1.0, scale='width')

    # Add baseline line
    if show_baseline and n_features is not None:
        if baseline_params is None:
            baseline_params = {}

        d = baseline_params.get('d', n_features)
        r = baseline_params.get('r', Config.JACCARD_K)

        if metric == 'jaccard':
            baseline = compute_jaccard_baseline(d, r, q_values=[0.3, 0.4, 0.5])
            ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

        elif metric == 'rbo':
            p = 1.0 - (1.0 / d)
            baseline = compute_rbo_baseline(n_features=d, p=p, q_values=[0.3, 0.4, 0.5])
            ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

        elif metric == 'l2':
            baseline = compute_l2_baseline(d, r, rho_values=[0.6, 0.7, 0.8])
            ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

    # Styling
    if metric == 'l2':
        ylabel = 'L2 Distance'
        ylim = Config.L2_YLIM
    elif metric == 'jaccard':
        ylabel = 'Jaccard Distance'
        ylim = Config.JACCARD_YLIM
    elif metric == 'rbo':
        ylabel = 'RBO Distance'
        ylim = Config.RBO_YLIM
    else:
        ylabel = f'{metric.upper()} Distance'
        ylim = None

    PlotHelper.setup_axis(ax, ylabel=ylabel, xlabel='Certainty', ylim=ylim, fontsize_scale=1.5)
    PlotHelper.add_legend(ax, loc="lower left", ncol=3, bbox=(0, 1.02, 1, 0.2), fontsize_scale=1.5)

    plt.tight_layout()
    plt.show()

    # Save figure
    PlotHelper.save_fig(fig, f'plot_2a_certainty_{metric}_violin.png')

    # Print statistics
    print(f"\nStatistics (mean ± std):")
    stats = df.groupby(['Group', 'Model'])['Value'].agg(['mean', 'std', 'count'])
    print(stats)


def plot_2b_certainty_features(results_dict, model_name, subgroup='Certain'):
    """
    Plot 2-b: Certainty Feature Sensitivity (Explainer Only)

    Shows which features are most unstable for a specific certainty subgroup (explainer seeds only).
    Y-axis labels show: feature_name=mean_abs_shap_value

    Args:
        subgroup: 'Certain' or 'Uncertain'

    Note: Model seed column removed because model seed changes affect certainty masks.
    """
    print(f"\n{'='*60}")
    print(f"Plot 2-b: Certainty Feature Sensitivity (Explainer) - {subgroup} [{model_name}]")
    print(f"{'='*60}")

    if 'certainty_feature_separated' not in results_dict[model_name]:
        print("⚠️  No separated feature data available")
        return

    feature_names = results_dict[model_name]['feature_names']

    # Get mean absolute SHAP values for this certainty group
    if subgroup == 'Certain':
        mean_abs_shap = results_dict[model_name].get('mean_abs_shap_certain', None)
    else:
        mean_abs_shap = results_dict[model_name].get('mean_abs_shap_uncertain', None)

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Only use explainer seed type
    inst_matrix = results_dict[model_name]['certainty_feature_separated']['explainer'][subgroup]

    if inst_matrix.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(f'Explainer - {subgroup}', fontsize=26, fontweight='bold')
        plt.tight_layout()
        plt.show()
        return

    # Sort by mean absolute SHAP value (highest first, so highest appears at bottom of barh)
    if mean_abs_shap is not None and len(mean_abs_shap) == inst_matrix.shape[1]:
        # Use mean_abs_shap for sorting (descending order - largest at top)
        top_k_idx = np.argsort(-np.array(mean_abs_shap))[:Config.TOP_K_FEATURES]
    else:
        # Fallback: sort by instability (descending order - largest at top)
        mean_vals = np.mean(inst_matrix, axis=0)
        top_k_idx = np.argsort(-mean_vals)[:Config.TOP_K_FEATURES]

    # Create feature labels with mean SHAP values
    top_names = []
    for i in top_k_idx:
        fname = feature_names[i] if feature_names else f'F{i}'
        # Shorten long feature names
        if fname == 'DiabetesPedigreeFunction':
            fname = 'PDF'
        elif fname == 'installment_commitment':
            fname = 'installment'
        elif 'checking_status' in fname:
            fname = 'checking_status'
        elif 'credit_history' in fname:
            fname = 'credit_history'
        elif fname.startswith('Race_'):
            fname = 'Race'
        elif fname.startswith('Occupation_'):
            fname = 'Occupation'
        if mean_abs_shap is not None and i < len(mean_abs_shap):
            top_names.append(f'{fname}\n={mean_abs_shap[i]:.3f}')
        else:
            top_names.append(fname)

    mean_vals = np.mean(inst_matrix, axis=0)
    top_vals = mean_vals[top_k_idx]

    # Reverse for barh to show descending from top to bottom
    top_names = top_names[::-1]
    top_vals = top_vals[::-1]

    # Bar plot
    ax.barh(top_names, top_vals, height=0.6, color=Config.COLORS[subgroup])
    ax.set_xlabel('Mean Absolute Difference', fontsize=26)
    ax.set_ylabel('Feature', fontsize=26)
    ax.set_title(f'Explainer - {subgroup}', fontsize=26, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=26)
    ax.tick_params(axis='y', pad=10)

    # Set x-axis limit
    if Config.FEATURE_L2_YLIM_CERTAINTY is not None:
        ax.set_xlim(Config.FEATURE_L2_YLIM_CERTAINTY)

    plt.tight_layout()
    plt.show()

    PlotHelper.save_fig(fig, f'plot_2b_certainty_features_{model_name}_{subgroup.lower()}_violin.png')

def plot_3a_prediction(results_dict, metric='l2', show_baseline=True, baseline_params=None):
    """
    Plot 3-a: Prediction-based Sensitivity (Explainer Seeds Only)

    Shows explainer sensitivity across prediction types (TP, TN, FP, FN).
    Only explainer seed variation is used because model seed changes affect prediction masks.

    Args:
        metric: 'l2' or 'jaccard'
        show_baseline: Whether to show random baseline (only for Jaccard)
        baseline_params: Dict with 'd' and 'r' for baseline computation
    """
    print(f"\n{'='*60}")
    print(f"Plot 3-a: Prediction Sensitivity - {metric.upper()}")
    print(f"{'='*60}")

    # Prepare data
    data = []
    n_features = None
    for model_name, model_results in results_dict.items():
        if 'prediction_separated' not in model_results:
            continue

        # Get number of features
        if n_features is None and 'feature_names' in model_results:
            n_features = len(model_results['feature_names'])

        for grp in ['TP', 'TN', 'FP', 'FN']:
            values = model_results['prediction_separated']['explainer'][grp][metric]
            for v in values:
                data.append({'Model': model_name, 'Group': grp, 'Value': v})

    df = pd.DataFrame(data)

    if df.empty:
        print("⚠️  No data available")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Violin plot
    group_order = ['TP', 'TN', 'FP', 'FN']
    model_order = Config.MODELS
    sns.violinplot(data=df, x='Group', y='Value', hue='Model', order=group_order,
                   hue_order=model_order, ax=ax, linewidth=1.0, cut=0, bw_adjust=1.0, scale='width')

    # Add baseline line
    if show_baseline and n_features is not None:
        if baseline_params is None:
            baseline_params = {}

        d = baseline_params.get('d', n_features)
        r = baseline_params.get('r', Config.JACCARD_K)

        if metric == 'jaccard':
            baseline = compute_jaccard_baseline(d, r, q_values=[0.3, 0.4, 0.5])
            ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

        elif metric == 'rbo':
            p = 1.0 - (1.0 / d)
            baseline = compute_rbo_baseline(n_features=d, p=p, q_values=[0.3, 0.4, 0.5])
            ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

        elif metric == 'l2':
            baseline = compute_l2_baseline(d, r, rho_values=[0.6, 0.7, 0.8])
            ax.axhspan(ymin=baseline['min'], ymax=baseline['max'],
                      color='red', alpha=0.25, label='Baseline Range', zorder=0)

    # Styling
    if metric == 'l2':
        ylabel = 'L2 Distance'
        ylim = Config.L2_YLIM
    elif metric == 'jaccard':
        ylabel = 'Jaccard Distance'
        ylim = Config.JACCARD_YLIM
    elif metric == 'rbo':
        ylabel = 'RBO Distance'
        ylim = Config.RBO_YLIM
    else:
        ylabel = f'{metric.upper()} Distance'
        ylim = None

    PlotHelper.setup_axis(ax, ylabel=ylabel, xlabel='Prediction Type', ylim=ylim, fontsize_scale=1.5)
    PlotHelper.add_legend(ax, loc="lower left", ncol=4, bbox=(0, 1.02, 1, 0.2), fontsize_scale=1.5)

    plt.tight_layout()
    plt.show()

    # Save figure
    PlotHelper.save_fig(fig, f'plot_3a_prediction_{metric}_violin.png')

    # Print statistics
    print(f"\nStatistics (mean ± std):")
    stats = df.groupby(['Group', 'Model'])['Value'].agg(['mean', 'std', 'count'])
    print(stats)


def plot_3b_prediction_features(results_dict, model_name, subgroup='TP'):
    """
    Plot 3-b: Prediction Feature Sensitivity (Explainer Only)

    Shows which features are most unstable for a specific prediction subgroup (explainer seeds only).
    Y-axis labels show: feature_name=mean_abs_shap_value

    Args:
        subgroup: 'TP', 'TN', 'FP', or 'FN'

    Note: Model seed column removed because model seed changes affect prediction masks.
    """
    print(f"\n{'='*60}")
    print(f"Plot 3-b: Prediction Feature Sensitivity (Explainer) - {subgroup} [{model_name}]")
    print(f"{'='*60}")

    if 'prediction_feature_separated' not in results_dict[model_name]:
        print("⚠️  No separated feature data available")
        return

    feature_names = results_dict[model_name]['feature_names']

    # Get mean absolute SHAP values for this prediction group
    if subgroup == 'TP':
        mean_abs_shap = results_dict[model_name].get('mean_abs_shap_tp', None)
    elif subgroup == 'TN':
        mean_abs_shap = results_dict[model_name].get('mean_abs_shap_tn', None)
    elif subgroup == 'FP':
        mean_abs_shap = results_dict[model_name].get('mean_abs_shap_fp', None)
    else:  # FN
        mean_abs_shap = results_dict[model_name].get('mean_abs_shap_fn', None)

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Only use explainer seed type
    inst_matrix = results_dict[model_name]['prediction_feature_separated']['explainer'][subgroup]

    if inst_matrix.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(f'Explainer - {subgroup}', fontsize=26, fontweight='bold')
        plt.tight_layout()
        plt.show()
        return

    # Sort by mean absolute SHAP value (highest first, so highest appears at bottom of barh)
    if mean_abs_shap is not None and len(mean_abs_shap) == inst_matrix.shape[1]:
        # Use mean_abs_shap for sorting (descending order - largest at top)
        top_k_idx = np.argsort(-np.array(mean_abs_shap))[:Config.TOP_K_FEATURES]
    else:
        # Fallback: sort by instability (descending order - largest at top)
        mean_vals = np.mean(inst_matrix, axis=0)
        top_k_idx = np.argsort(-mean_vals)[:Config.TOP_K_FEATURES]

    # Create feature labels with mean SHAP values
    top_names = []
    for i in top_k_idx:
        fname = feature_names[i] if feature_names else f'F{i}'
        # Shorten long feature names
        if fname == 'DiabetesPedigreeFunction':
            fname = 'PDF'
        elif fname == 'installment_commitment':
            fname = 'installment'
        elif 'checking_status' in fname:
            fname = 'checking_status'
        elif 'credit_history' in fname:
            fname = 'credit_history'
        elif fname.startswith('Race_'):
            fname = 'Race'
        elif fname.startswith('Occupation_'):
            fname = 'Occupation'
        if mean_abs_shap is not None and i < len(mean_abs_shap):
            top_names.append(f'{fname}\n={mean_abs_shap[i]:.3f}')
        else:
            top_names.append(fname)

    mean_vals = np.mean(inst_matrix, axis=0)
    top_vals = mean_vals[top_k_idx]

    # Reverse for barh to show descending from top to bottom
    top_names = top_names[::-1]
    top_vals = top_vals[::-1]

    # Bar plot
    ax.barh(top_names, top_vals, height=0.6, color=Config.COLORS[subgroup])
    ax.set_xlabel('Mean Absolute Difference', fontsize=26)
    ax.set_ylabel('Feature', fontsize=26)
    ax.set_title(f'Explainer - {subgroup}', fontsize=26, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=26)
    ax.tick_params(axis='y', pad=10)

    # Set x-axis limit
    if Config.FEATURE_L2_YLIM_PREDICTION is not None:
        ax.set_xlim(Config.FEATURE_L2_YLIM_PREDICTION)

    plt.tight_layout()
    plt.show()

    PlotHelper.save_fig(fig, f'plot_3b_prediction_features_{model_name}_{subgroup.lower()}_violin.png')

def plot_4_probability_distribution(results_dict, model_name):
    """
    Plot 4: Probability Distribution by Model Seeds

    Shows how prediction probabilities vary across different model seeds.
    - X-axis: Probability (0 to 1)
    - Y-axis: Model Seed (0 to 4)
    - Background shading:
      - High certainty: [0, 0.1] and [0.9, 1.0]
      - Low certainty: [0.4, 0.6]
    """
    print(f"\n{'='*60}")
    print(f"Plot 4: Probability Distribution [{model_name}]")
    print(f"{'='*60}")

    # Get probability data from raw_results (need to pass it separately)
    # Since we don't have raw_results here, we'll need to get proba from results_dict
    # This requires the notebook to pass raw_data
    if 'proba_4d' not in results_dict[model_name]:
        print("⚠️  No probability data available")
        return

    proba_4d = results_dict[model_name]['proba_4d']

    # Flatten across splits and instances
    # proba_4d shape: (N_splits=5, N_model_seeds=5, N_instances, N_classes)
    data = []
    for model_seed in range(Config.N_MODEL_SEEDS):
        probas = []
        for split in range(Config.N_SPLITS):
            # Get probabilities for this split and model seed
            p = proba_4d[split, model_seed]

            # Handle binary classification (take class 1 probability)
            if p.ndim == 2 and p.shape[1] == 2:
                p = p[:, 1]
            elif p.ndim == 2:
                p = p[:, 0]

            # Filter out NaN values
            valid = ~np.isnan(p)
            probas.extend(p[valid])

        # Add to dataframe
        for prob in probas:
            data.append({'Model Seed': model_seed, 'Probability': prob})

    df = pd.DataFrame(data)

    if df.empty:
        print("⚠️  No valid probability data")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Add background shading for certainty regions
    # High certainty regions (light blue)
    ax.axvspan(0.0, 0.1, alpha=0.15, color='blue', label='High Certainty')
    ax.axvspan(0.9, 1.0, alpha=0.15, color='blue')

    # Low certainty region (light red)
    ax.axvspan(0.4, 0.6, alpha=0.15, color='red', label='Low Certainty')

    # Violin plot
    sns.violinplot(data=df, x='Probability', y='Model Seed',
                   orient='h', ax=ax, palette='viridis', linewidth=1.0, scale='width')

    # Styling
    PlotHelper.setup_axis(ax, xlabel='Probability', ylabel='Model Seed')
    ax.set_xlim(0, 1)
    ax.set_yticks(range(Config.N_MODEL_SEEDS))
    PlotHelper.add_legend(ax, loc="lower left", ncol=2, bbox=(0, 1.02, 1, 0.2))

    plt.tight_layout()
    plt.show()

    PlotHelper.save_fig(fig, f'plot_4_probability_distribution_{model_name}_violin.png')

    # Print statistics
    print(f"\nProbability Statistics by Model Seed:")
    stats = df.groupby('Model Seed')['Probability'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(stats)


print("✅ Analysis and plotting functions loaded successfully!")
