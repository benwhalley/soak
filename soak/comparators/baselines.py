"""Baseline generation for null comparisons (word-salad) and cost matrix structure analysis."""

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


def compute_cost_matrix_structure(cost_matrix: np.ndarray) -> Dict[str, Any]:
    """Analyse cost matrix structure to understand alignment potential.

    Computes metrics that indicate whether themes have clear preferred partners
    (enabling meaningful OT alignment) or are interchangeable (diffuse overlap).

    Args:
        cost_matrix: Cost matrix (n_A x n_B), where lower = more similar

    Returns:
        Dict with structure metrics:
        - row_preference_strength: How much better each A's best match is vs average
        - col_preference_strength: How much better each B's best match is vs average
        - relative_preference: Preference strength as % of mean cost (normalised)
        - spikiness: Std/range of cost matrix (higher = more varied similarities)
        - mean_cost: Average cost across all pairs
        - interpretation: Qualitative description of the structure
    """
    M = np.clip(cost_matrix, 0, None)
    n_A, n_B = M.shape

    if n_A == 0 or n_B == 0:
        return {"interpretation": "Empty cost matrix"}

    # For each A theme: how much better is best B match vs average B match?
    row_mins = M.min(axis=1)
    row_means = M.mean(axis=1)
    row_preference = float((row_means - row_mins).mean())

    # For each B theme: how much better is best A match vs average A match?
    col_mins = M.min(axis=0)
    col_means = M.mean(axis=0)
    col_preference = float((col_means - col_mins).mean())

    # Average preference strength
    avg_preference = (row_preference + col_preference) / 2

    # Normalise by mean cost to get relative preference (%)
    mean_cost = float(M.mean())
    relative_preference = avg_preference / mean_cost if mean_cost > 0 else 0

    # Spikiness: how varied is the cost matrix?
    cost_range = M.max() - M.min()
    spikiness = float(M.std() / (cost_range + 1e-9))

    # Qualitative interpretation (no hard thresholds, just guidance)
    if relative_preference > 0.15:
        interpretation = "Strong preferences: themes have clear preferred partners"
    elif relative_preference > 0.08:
        interpretation = "Moderate preferences: some themes have preferred partners"
    else:
        interpretation = "Weak preferences: themes are largely interchangeable"

    return {
        "row_preference_strength": row_preference,
        "col_preference_strength": col_preference,
        "avg_preference_strength": avg_preference,
        "relative_preference": relative_preference,
        "spikiness": spikiness,
        "mean_cost": mean_cost,
        "cost_std": float(M.std()),
        "cost_range": float(cost_range),
        "interpretation": interpretation,
    }
