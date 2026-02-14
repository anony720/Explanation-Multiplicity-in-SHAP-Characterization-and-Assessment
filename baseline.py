import numpy as np
import random
import math


def compute_l2_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2, axis=1))


def compute_l2_baseline(d, r, rho=0.7, kappa=11, T=0.4):
    term1 = (rho ** 2) / r
    term2 = ((1 - rho) ** 2) / (d - r)
    exp_sq_l2 = (2 * T ** 2) / (kappa + 1) * (1 - term1 - term2)
    return np.sqrt(exp_sq_l2)


def compute_jaccard_distance(v1, v2, r=3):
    n_samples = v1.shape[0]
    distances = np.zeros(n_samples)

    top_r_v1 = np.argsort(np.abs(v1), axis=1)[:, -r:]
    top_r_v2 = np.argsort(np.abs(v2), axis=1)[:, -r:]

    for i in range(n_samples):
        set1 = set(top_r_v1[i])
        set2 = set(top_r_v2[i])

        if not set1 and not set2:
            distances[i] = 0.0
            continue

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        distances[i] = 1.0 - (intersection / union) if union > 0 else 1.0

    return distances


def compute_rbo_similarity(v1, v2, p=0.9, r=100):
    """
    Compute Rank-Biased Overlap (RBO) similarity between two SHAP value matrices.

    Parameters:
    -----------
    v1, v2 : np.ndarray
        SHAP value matrices of shape (n_samples, n_features)
    p : float, default=0.9
        Persistence parameter (0 <= p < 1). Controls how much weight is given to top-ranked features.
        - Higher p (e.g., 0.9): more weight on deeper ranks, considers more features
        - Lower p (e.g., 2/3): more weight on top ranks, focuses on most important features
        - Common values: 0.9 (original paper), 2/3 (balanced), 1-1/d (adaptive)
    r : int, default=100
        Evaluation depth. Maximum rank position to consider in the comparison.
        - Typical values: 3-100 depending on total number of features
        - Set to n_features to evaluate full ranking

    Returns:
    --------
    similarities : np.ndarray of shape (n_samples,)
        RBO similarity score for each sample (0 = completely different, 1 = identical rankings)
    """
    n_samples, n_features = v1.shape
    r = min(r, n_features)

    similarities = np.zeros(n_samples)
    weights = (1 - p) * (p ** np.arange(r))

    for i in range(n_samples):
        rank1 = np.argsort(np.abs(v1[i]))[::-1]
        rank2 = np.argsort(np.abs(v2[i]))[::-1]

        rbo_sum = 0.0
        for d in range(1, r + 1):
            top_d_rank1 = set(rank1[:d])
            top_d_rank2 = set(rank2[:d])
            overlap = len(top_d_rank1.intersection(top_d_rank2))
            rbo_sum += weights[d - 1] * (overlap / d)

        similarities[i] = rbo_sum

    return similarities


def compute_rbo_distance(v1, v2, r=100, p=0.9):
    return 1.0 - compute_rbo_similarity(v1, v2, p=p, r=r)


def mallows_sample_insertion(d, q, rng):
    if not (0.0 <= q < 1.0):
        raise ValueError("q must be in [0,1).")
    perm = []
    for i in range(d):
        weights = [q ** (i - j) for j in range(i + 1)]
        total = sum(weights)
        u = rng.random() * total
        c = 0.0
        pos = 0
        for j, w in enumerate(weights):
            c += w
            if u <= c:
                pos = j
                break
        perm.insert(pos, i)
    return perm


def jaccard_similarity_top_r(pi, pi2, r):
    a = set(pi[:r])
    b = set(pi2[:r])
    u = len(a.union(b))
    if u == 0:
        return 1.0
    return len(a.intersection(b)) / u


def rbo_similarity_at_r(rank_a, rank_b, r=3, p=2/3):
    """
    Compute RBO similarity between two rankings at evaluation depth r.

    Parameters:
    -----------
    rank_a, rank_b : list or array
        Two rankings (permutations) to compare
    r : int, default=3
        Evaluation depth. Maximum rank position to consider.
    p : float, default=2/3
        Persistence parameter (0 <= p < 1). See compute_rbo_similarity for details.

    Returns:
    --------
    similarity : float
        RBO similarity score (0 = completely different, 1 = identical rankings)
    """
    if r <= 0:
        raise ValueError("r must be positive.")
    if not (0.0 <= p < 1.0):
        raise ValueError("p must be in [0,1).")
    r = min(r, len(rank_a), len(rank_b))

    weights = (1 - p) * (p ** np.arange(r))
    s = 0.0
    for d in range(1, r + 1):
        a = set(rank_a[:d])
        b = set(rank_b[:d])
        s += weights[d - 1] * (len(a.intersection(b)) / d)
    return s


def compute_jaccard_baseline(d, r, q, n_trials=20000, seed=0, return_std=True):
    d = int(d); r = int(r)
    if d <= 0: raise ValueError("d must be positive.")
    if r <= 0: raise ValueError("r must be positive.")
    if not (0.0 <= q < 1.0): raise ValueError("q must be in [0,1).")
    r = min(r, d)

    rng = random.Random(seed)
    s1 = 0.0
    s2 = 0.0

    for _ in range(int(n_trials)):
        pi  = mallows_sample_insertion(d, q, rng)
        pi2 = mallows_sample_insertion(d, q, rng)

        sim = jaccard_similarity_top_r(pi, pi2, r=r)
        dist = 1.0 - sim

        s1 += dist
        if return_std:
            s2 += dist * dist

    mean_dist = s1 / float(n_trials)
    out = {
        "mean": mean_dist,
        "E_sim": 1.0 - mean_dist,
        "d": d, "r": r, "q": q,
        "n_trials": int(n_trials),
        "seed": seed,
        "null": "mallows_identity_center_simplified",
    }
    if return_std:
        var = max(0.0, (s2 / float(n_trials)) - mean_dist * mean_dist)
        out["std"] = math.sqrt(var)
        out["std_sim"] = out["std"]
    return out


def compute_rbo_baseline_simplified(d, r, p, q, n_trials=20000, seed=0, return_std=False):
    """
    Compute baseline RBO distance using Mallows model.

    Parameters:
    -----------
    d : int
        Number of features (dimensionality)
    r : int
        Evaluation depth for RBO computation
    p : float
        RBO persistence parameter (0 <= p < 1). See compute_rbo_similarity for details.
    q : float
        Mallows dispersion parameter (0 <= q < 1). Controls how much rankings deviate from identity.
        - q=0: all rankings are identity permutation (no randomness)
        - q→1: rankings are more dispersed/random
    n_trials : int, default=20000
        Number of Monte Carlo samples for baseline estimation
    seed : int, default=0
        Random seed for reproducibility
    return_std : bool, default=False
        If True, also compute and return standard deviation

    Returns:
    --------
    out : dict
        Dictionary containing baseline statistics (mean distance, E_sim, etc.)
    """
    d = int(d); r = int(r)
    if d <= 0: raise ValueError("d must be positive.")
    if r <= 0: raise ValueError("r must be positive.")
    if not (0.0 <= p < 1.0): raise ValueError("p must be in [0,1).")
    if not (0.0 <= q < 1.0): raise ValueError("q must be in [0,1).")
    r = min(r, d)

    rng = random.Random(seed)
    s1 = 0.0
    s2 = 0.0

    for _ in range(int(n_trials)):
        pi  = mallows_sample_insertion(d, q, rng)
        pi2 = mallows_sample_insertion(d, q, rng)

        sim = rbo_similarity_at_r(pi, pi2, r=r, p=p)
        dist = 1.0 - sim

        s1 += dist
        if return_std:
            s2 += dist * dist

    mean_dist = s1 / float(n_trials)
    out = {
        "mean": mean_dist,
        "E_sim": 1.0 - mean_dist,
        "d": d, "r": r, "p": p, "q": q,
        "n_trials": int(n_trials),
        "seed": seed,
        "null": "mallows_identity_center_simplified",
    }
    if return_std:
        var = max(0.0, (s2 / float(n_trials)) - mean_dist * mean_dist)
        out["std"] = math.sqrt(var)
    return out
    
# Alias for compatibility with plot_refactored.py
compute_jaccard_baseline_simplified = compute_jaccard_baseline

if __name__ == "__main__":
    print(compute_rbo_baseline_simplified(8, 3, 2/3, 0.5))

