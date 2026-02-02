"""Baseline generation for null comparisons (word-salad, permutation)."""

import re
from typing import Any, Dict, List

import numpy as np


def generate_word_salad_texts(
    theme_texts: List[str], n_samples: int = 100, seed: int = 42
) -> List[List[str]]:
    """Generate word-salad versions of themes for null baseline.

    Takes all words from themes, shuffles them, and chunks into strings
    with the same length distribution as originals. This destroys semantic
    coherence while preserving vocabulary and length properties.

    Args:
        theme_texts: Original theme strings (e.g., "name: description")
        n_samples: Number of word-salad sets to generate
        seed: Random seed for reproducibility (enables embedding cache hits)

    Returns:
        List of n_samples lists, each containing len(theme_texts) word-salad strings
    """
    # set seed for reproducibility - same inputs will produce same word salad,
    # enabling embedding cache hits on subsequent runs
    rng = np.random.default_rng(seed)

    # tokenize all themes (simple word split, lowercase)
    all_words = []
    theme_lengths = []
    for text in theme_texts:
        words = re.findall(r"\b\w+\b", text.lower())
        all_words.extend(words)
        theme_lengths.append(max(1, len(words)))

    if not all_words:
        # fallback: return original texts if no words found
        return [[t for t in theme_texts] for _ in range(n_samples)]

    # generate N word salad sets
    results = []
    for _ in range(n_samples):
        shuffled = rng.permutation(all_words).tolist()

        # chunk into same length distribution as originals
        salad_themes = []
        idx = 0
        for length in theme_lengths:
            chunk_words = []
            for _ in range(length):
                chunk_words.append(shuffled[idx % len(shuffled)])
                idx += 1
            salad_themes.append(" ".join(chunk_words))

        results.append(salad_themes)

    return results


def compute_permutation_baseline(
    cost_matrix: np.ndarray,
    k_values: List[float],
    n_permutations: int = 50,
    mode: str = "unbalanced",
    reg: float = 0.01,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute OT shared mass and splits/joins under row-permuted cost matrices.

    Tests whether similarity geometry encodes pair-specific structure.
    If B_perm(K) ~ B_true(K), the transport plan is underdetermined --
    meaning OT can find equally good alignments with scrambled pairings.

    Also tracks splits/joins under permutation. Even if shared mass is similar,
    permuted alignments typically have MORE splits/joins (fragmented transport),
    indicating worse alignment quality.

    This is a necessary condition for meaningful OT alignment: if the
    permutation baseline doesn't collapse, shared mass reflects regularisation
    strength rather than true semantic correspondence.

    Args:
        cost_matrix: Original cost matrix (n_A x n_B)
        k_values: Mass penalty (reg_m) values to test
        n_permutations: Number of random permutations (default 50)
        mode: OT mode ("balanced" or "unbalanced")
        reg: Entropic regularisation (default 0.01)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping K values to permutation statistics:
        - perm_shared_mass_mean: Mean shared mass under permutation
        - perm_shared_mass_std: Std dev across permutations
        - perm_shared_mass_95ci: [2.5%, 97.5%] confidence interval
        - perm_shared_mass_distribution: Full distribution for plotting
        - perm_splits_joins_mean: Mean (splits + joins) / 2 under permutation
        - perm_splits_joins_std: Std dev of splits/joins
        - perm_splits_joins_distribution: Full distribution for comparison
    """
    import ot as pot

    rng = np.random.default_rng(seed)
    n_A, n_B = cost_matrix.shape

    if n_A == 0 or n_B == 0:
        return {}

    # uniform mass distributions
    a = np.ones(n_A) / n_A
    b = np.ones(n_B) / n_B

    # ensure cost matrix is non-negative
    M = np.clip(cost_matrix, 0, None)

    # ensure minimum regularisation
    reg = max(reg, 1e-6)

    def run_ot_for_k(cost, k_val):
        """Run OT with given cost matrix and K value, return plan and mass."""
        k_val = max(k_val, 1e-6)
        if mode == "balanced":
            P = pot.emd(a, b, cost)
        else:
            P = pot.unbalanced.sinkhorn_unbalanced(
                a, b, cost, reg=reg, reg_m=k_val, numItermax=1000, stopThr=1e-9
            )
        return P, float(P.sum())

    def compute_splits_joins_quick(P, threshold_ratio=0.01):
        """Compute average splits+joins for a transport plan (fast version)."""
        threshold = threshold_ratio * P.max() if P.max() > 0 else 0
        # splits: number of B themes each A connects to
        splits_per_a = np.sum(P > threshold, axis=1)
        # joins: number of A themes each B receives from
        joins_per_b = np.sum(P > threshold, axis=0)
        # return average of (mean splits + mean joins) / 2
        mean_splits = float(np.mean(splits_per_a))
        mean_joins = float(np.mean(joins_per_b))
        return (mean_splits + mean_joins) / 2

    results_by_k = {}

    for k_val in k_values:
        perm_masses = []
        perm_splits_joins = []

        for _ in range(n_permutations):
            # permute rows of cost matrix (scrambles A-to-B mapping)
            perm_idx = rng.permutation(n_A)
            M_perm = M[perm_idx, :]
            P_perm, shared_mass = run_ot_for_k(M_perm, k_val)
            perm_masses.append(shared_mass)
            perm_splits_joins.append(compute_splits_joins_quick(P_perm))

        perm_mass_arr = np.array(perm_masses)
        perm_sj_arr = np.array(perm_splits_joins)

        results_by_k[k_val] = {
            # shared mass stats
            "perm_shared_mass_mean": float(perm_mass_arr.mean()),
            "perm_shared_mass_std": float(perm_mass_arr.std()),
            "perm_shared_mass_95ci": [
                float(np.percentile(perm_mass_arr, 2.5)),
                float(np.percentile(perm_mass_arr, 97.5)),
            ],
            "perm_shared_mass_distribution": perm_mass_arr.tolist(),
            # splits/joins stats
            "perm_splits_joins_mean": float(perm_sj_arr.mean()),
            "perm_splits_joins_std": float(perm_sj_arr.std()),
            "perm_splits_joins_95ci": [
                float(np.percentile(perm_sj_arr, 2.5)),
                float(np.percentile(perm_sj_arr, 97.5)),
            ],
            "perm_splits_joins_distribution": perm_sj_arr.tolist(),
        }

    return results_by_k
