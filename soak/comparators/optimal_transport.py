"""Optimal transport computation and related functions."""

import hashlib
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from soak.models.base import memory

logger = logging.getLogger(__name__)


def _hash_array(arr: np.ndarray) -> str:
    """Create a stable hash of a numpy array for cache keys."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _hash_array_list(arrays: List[np.ndarray]) -> str:
    """Create a stable hash of a list of numpy arrays."""
    combined = "".join(_hash_array(np.asarray(a)) for a in arrays)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


@memory.cache
def _compute_ot_single(
    cost_matrix: np.ndarray,
    mode: str,
    reg: float,
    reg_m: float,
) -> Dict[str, Any]:
    """Cached OT for a single cost matrix. Tiny args = fast joblib persistence."""
    import ot

    n_A, n_B = cost_matrix.shape

    if n_A == 0 or n_B == 0:
        return {
            "shared_mass": 0.0,
            "avg_cost": float("nan"),
            "unmatched_mass": 1.0,
            "transport_plan": [],
            "coverage_a": [],
            "coverage_b": [],
        }

    # uniform mass distribution
    a = np.ones(n_A) / n_A
    b = np.ones(n_B) / n_B

    # ensure cost matrix is non-negative
    M = np.clip(cost_matrix, 0, None)

    # ensure minimum regularisation for numerical stability
    reg = max(reg, 1e-6)
    reg_m = max(reg_m, 1e-6)

    if mode == "balanced":
        P = ot.emd(a, b, M)
    else:
        P = ot.unbalanced.sinkhorn_unbalanced(
            a, b, M, reg=reg, reg_m=reg_m, numItermax=1000, stopThr=1e-9
        )

    shared_mass = float(P.sum())
    avg_cost = (
        float((P * M).sum() / shared_mass) if shared_mass > 1e-9 else float("nan")
    )
    unmatched_mass = 1.0 - shared_mass

    coverage_a = P.sum(axis=1).tolist()
    coverage_b = P.sum(axis=0).tolist()

    return {
        "shared_mass": shared_mass,
        "avg_cost": avg_cost,
        "unmatched_mass": unmatched_mass,
        "transport_plan": P.tolist(),
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
    }


def compute_ot(
    cost_matrix,
    null_cost_matrices: Optional[List] = None,
    mode: str = "unbalanced",
    reg: float = 0.01,
    reg_m: float = 0.2,
):
    """Compute optimal transport metrics for theme similarity.

    Unbalanced OT allows unmatched mass, representing genuinely novel or missing
    themes rather than forcing all themes to align. The reg_m parameter (K)
    controls when themes are treated as unmatched rather than forced to align.

    Each individual OT computation (main + each null matrix) is cached separately
    via _compute_ot_single, so joblib only needs to hash one small matrix per call.

    Args:
        cost_matrix: (n_A x n_B) numpy array of costs (1 - similarity)
        null_cost_matrices: Pre-computed null cost matrices (e.g., from word-salad).
                           If provided, used for null baseline instead of permutation.
        mode: "unbalanced" (default) or "balanced" for comparison
        reg: Entropic regularisation for numerical stability (default: 0.01)
        reg_m: Mass penalty K (default: 0.2). Fixed value for cross-analysis
               comparability. Lower K = more selective matching.

    Returns:
        Dictionary with OT metrics including shared_mass, avg_cost, unmatched_mass
    """
    cost_arr = np.asarray(cost_matrix, dtype=np.float64)

    # main OT -- cached, tiny args
    main = _compute_ot_single(cost_arr, mode, reg, reg_m)

    result = dict(main)
    result["reg"] = max(reg, 1e-6)
    result["reg_m"] = max(reg_m, 1e-6)
    result["mode"] = mode

    # convert transport_plan back to numpy array
    if result["transport_plan"]:
        result["transport_plan"] = np.array(result["transport_plan"])
    else:
        result["transport_plan"] = np.zeros((0, 0))

    # null baseline from pre-computed null cost matrices (word-salad)
    if null_cost_matrices is not None and len(null_cost_matrices) > 0:
        null_arrays = [np.asarray(m, dtype=np.float64) for m in null_cost_matrices]

        # each null matrix cached individually -- tiny args per call
        null_results = [_compute_ot_single(m, mode, reg, reg_m) for m in null_arrays]

        null_shared_arr = np.array([r["shared_mass"] for r in null_results])
        null_avg_costs = [r["avg_cost"] for r in null_results]
        null_avg_arr = np.array([x for x in null_avg_costs if not np.isnan(x)])

        shared_mass = result["shared_mass"]
        avg_cost = result["avg_cost"]

        result["null_shared_mass_mean"] = float(null_shared_arr.mean())
        result["null_shared_mass_95pct"] = float(np.percentile(null_shared_arr, 95))
        result["null_shared_mass_distribution"] = null_shared_arr.tolist()

        if len(null_avg_arr) > 0:
            result["null_avg_cost_mean"] = float(null_avg_arr.mean())
            result["null_avg_cost_5pct"] = float(np.percentile(null_avg_arr, 5))
            result["null_avg_cost_distribution"] = null_avg_arr.tolist()
        else:
            result["null_avg_cost_mean"] = float("nan")
            result["null_avg_cost_5pct"] = float("nan")
            result["null_avg_cost_distribution"] = []

        # === INTERPRETABLE RELATIVE METRICS ===

        # shared_mass_excess: raw difference above null
        null_mean = result["null_shared_mass_mean"]
        result["shared_mass_excess"] = float(shared_mass - null_mean)

        # shared_mass_relative: 0 = same as random, 1 = perfect transport
        if null_mean < 1.0:
            result["shared_mass_relative"] = float(
                (shared_mass - null_mean) / (1.0 - null_mean)
            )
        else:
            result["shared_mass_relative"] = 0.0

        # shared_mass_effect: robust effect size using MAD
        null_median = np.median(null_shared_arr)
        null_mad = np.median(np.abs(null_shared_arr - null_median))
        result["shared_mass_effect"] = float(
            (shared_mass - null_mean) / (null_mad + 1e-9)
        )
        result["null_shared_mass_mad"] = float(null_mad)

        # avg_cost metrics (lower is better)
        if len(null_avg_arr) > 0 and not np.isnan(avg_cost):
            null_cost_mean = result["null_avg_cost_mean"]
            result["avg_cost_improvement"] = float(null_cost_mean - avg_cost)
            if null_cost_mean > 0:
                result["avg_cost_relative"] = float(
                    (null_cost_mean - avg_cost) / null_cost_mean
                )
            else:
                result["avg_cost_relative"] = 0.0
            null_cost_median = np.median(null_avg_arr)
            null_cost_mad = np.median(np.abs(null_avg_arr - null_cost_median))
            result["avg_cost_effect"] = float(
                (null_cost_mean - avg_cost) / (null_cost_mad + 1e-9)
            )
            result["null_avg_cost_mad"] = float(null_cost_mad)
        else:
            result["avg_cost_improvement"] = 0.0
            result["avg_cost_relative"] = 0.0
            result["avg_cost_effect"] = 0.0
            result["null_avg_cost_mad"] = 0.0

    return result


def compute_split_join_stats(
    transport_plan,
    threshold_ratio: float = 0.01,
) -> Dict[str, Any]:
    """Compute statistics about splits and joins in a transport plan.

    A "split" occurs when mass from one theme in A flows to multiple themes in B.
    A "join" occurs when mass from multiple themes in A flows to one theme in B.

    Args:
        transport_plan: (n_A x n_B) transport coupling matrix P
        threshold_ratio: Links below this fraction of max flow are ignored

    Returns:
        Dictionary with split/join statistics including counts, mean, median, mode, max
    """
    P = np.asarray(transport_plan)
    n_A, n_B = P.shape

    if n_A == 0 or n_B == 0:
        return {
            "splits_from_a": {
                "counts": {},
                "mean": 0.0,
                "median": 0.0,
                "mode": 0,
                "max": 0,
                "total": 0,
            },
            "joins_to_b": {
                "counts": {},
                "mean": 0.0,
                "median": 0.0,
                "mode": 0,
                "max": 0,
                "total": 0,
            },
        }

    threshold = threshold_ratio * P.max() if P.max() > 0 else 0

    # count outgoing connections for each A theme (splits)
    splits_per_a = []
    for i in range(n_A):
        n_targets = np.sum(P[i, :] > threshold)
        splits_per_a.append(int(n_targets))

    # count incoming connections for each B theme (joins)
    joins_per_b = []
    for j in range(n_B):
        n_sources = np.sum(P[:, j] > threshold)
        joins_per_b.append(int(n_sources))

    def compute_stats(values: List[int]) -> Dict[str, Any]:
        if not values:
            return {
                "counts": {},
                "mean": 0.0,
                "median": 0.0,
                "mode": 0,
                "max": 0,
                "total": 0,
            }

        counts = Counter(values)
        values_arr = np.array(values)

        # mode is the most common value
        mode_val = counts.most_common(1)[0][0] if counts else 0

        # count themes with >1 connection (actual splits/joins)
        n_multiple = sum(1 for v in values if v > 1)

        return {
            "counts": dict(sorted(counts.items())),
            "mean": float(np.mean(values_arr)),
            "median": float(np.median(values_arr)),
            "mode": mode_val,
            "max": int(np.max(values_arr)),
            "total": len(values),
            "n_multiple": n_multiple,
            "pct_multiple": float(n_multiple / len(values)) if values else 0.0,
            "distribution": values,
        }

    return {
        "splits_from_a": compute_stats(splits_per_a),
        "joins_to_b": compute_stats(joins_per_b),
    }


def filter_transport_plan(
    transport_plan: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Filter weak edges from transport plan to reduce splits/joins.

    Removes edges where mass is below threshold fraction of row or column total.
    An edge (i,j) is kept only if BOTH:
    - P[i,j] >= threshold * sum(P[i,:])  (significant for row i)
    - P[i,j] >= threshold * sum(P[:,j])  (significant for column j)

    This removes "minor" splits and joins that contribute little mass but
    clutter the alignment graph.

    Args:
        transport_plan: (n_A x n_B) transport matrix from OT
        threshold: Minimum mass fraction to keep edge (default 0.05 = 5%)
                   Set to 0 to disable filtering.

    Returns:
        filtered_plan: Filtered transport matrix (same shape, weak edges zeroed)
        stats: Dictionary with filtering statistics
    """
    P = np.asarray(transport_plan).copy()
    original_mass = float(P.sum())
    n_A, n_B = P.shape

    if threshold <= 0 or original_mass == 0:
        # no filtering
        edges_count = int((P > 0).sum())
        edges_max_possible = n_A * n_B
        return P, {
            "original_mass": original_mass,
            "filtered_mass": original_mass,
            "mass_retained_pct": 100.0,
            "edges_original": edges_count,
            "edges_filtered": edges_count,
            "edges_removed": 0,
            "edges_max_possible": edges_max_possible,
            "edges_retained_pct": 100.0,
            "edges_original_pct": (
                float(edges_count / edges_max_possible * 100)
                if edges_max_possible > 0
                else 0.0
            ),
            "edges_filtered_pct": (
                float(edges_count / edges_max_possible * 100)
                if edges_max_possible > 0
                else 0.0
            ),
            "threshold": threshold,
            "filtering_enabled": False,
        }

    # Row-wise criterion: P[i,j] must be >= threshold * row_sum
    row_sums = P.sum(axis=1, keepdims=True)
    # avoid division by zero for empty rows
    row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
    row_mask = P >= threshold * row_sums_safe

    # Column-wise criterion: P[i,j] must be >= threshold * col_sum
    col_sums = P.sum(axis=0, keepdims=True)
    col_sums_safe = np.where(col_sums > 0, col_sums, 1.0)
    col_mask = P >= threshold * col_sums_safe

    # Keep edge only if it passes both criteria
    mask = row_mask & col_mask
    P_filtered = np.where(mask, P, 0.0)

    filtered_mass = float(P_filtered.sum())
    edges_original = int((P > 0).sum())
    edges_filtered = int((P_filtered > 0).sum())
    edges_max_possible = P.shape[0] * P.shape[1]

    stats = {
        "original_mass": original_mass,
        "filtered_mass": filtered_mass,
        "mass_retained_pct": (
            float(filtered_mass / original_mass * 100) if original_mass > 0 else 0.0
        ),
        "edges_original": edges_original,
        "edges_filtered": edges_filtered,
        "edges_removed": edges_original - edges_filtered,
        "edges_max_possible": edges_max_possible,
        "edges_retained_pct": (
            float(edges_filtered / edges_original * 100) if edges_original > 0 else 0.0
        ),
        "edges_original_pct": (
            float(edges_original / edges_max_possible * 100)
            if edges_max_possible > 0
            else 0.0
        ),
        "edges_filtered_pct": (
            float(edges_filtered / edges_max_possible * 100)
            if edges_max_possible > 0
            else 0.0
        ),
        "threshold": threshold,
        "filtering_enabled": True,
    }

    return P_filtered, stats


def compute_best_matches_for_k(
    transport_plan: np.ndarray,
    similarity_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    n_a: int,
    n_b: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Compute best matches for a given K's transport plan.

    Args:
        transport_plan: (n_A x n_B) transport coupling matrix P for this K
        similarity_matrix: (n_A x n_B) similarity matrix (K-independent)
        cost_matrix: (n_A x n_B) cost matrix (K-independent)
        n_a: Number of themes in set A
        n_b: Number of themes in set B

    Returns:
        Tuple of (best_matches_a_to_b, best_matches_b_to_a) lists
    """
    P = np.asarray(transport_plan)

    def round_match(d):
        """Round numeric values in a match dict."""
        decimals = {
            "similarity": 3,
            "mass_transferred": 4,
            "mass_total": 4,
            "mass_pct": 1,
            "cost": 3,
        }
        return {
            k: round(v, decimals.get(k, 3)) if isinstance(v, float) else v
            for k, v in d.items()
        }

    best_matches_a_to_b = []
    if n_a > 0 and n_b > 0:
        for i in range(n_a):
            best_b_idx = int(similarity_matrix[i, :].argmax())
            mass_total_out = P[i, :].sum()
            best_matches_a_to_b.append(
                round_match(
                    {
                        "theme_a_index": i,
                        "theme_b_index": best_b_idx,
                        "similarity": float(similarity_matrix[i, best_b_idx]),
                        "mass_transferred": float(P[i, best_b_idx]),
                        "mass_total": float(mass_total_out),
                        "mass_pct": (
                            float(P[i, best_b_idx] / mass_total_out * 100)
                            if mass_total_out > 0
                            else 0.0
                        ),
                        "cost": float(cost_matrix[i, best_b_idx]),
                    }
                )
            )
        best_matches_a_to_b.sort(key=lambda x: x["similarity"], reverse=True)

    best_matches_b_to_a = []
    if n_a > 0 and n_b > 0:
        for j in range(n_b):
            best_a_idx = int(similarity_matrix[:, j].argmax())
            mass_total_in = P[:, j].sum()
            best_matches_b_to_a.append(
                round_match(
                    {
                        "theme_b_index": j,
                        "theme_a_index": best_a_idx,
                        "similarity": float(similarity_matrix[best_a_idx, j]),
                        "mass_transferred": float(P[best_a_idx, j]),
                        "mass_total": float(mass_total_in),
                        "mass_pct": (
                            float(P[best_a_idx, j] / mass_total_in * 100)
                            if mass_total_in > 0
                            else 0.0
                        ),
                        "cost": float(cost_matrix[best_a_idx, j]),
                    }
                )
            )
        best_matches_b_to_a.sort(key=lambda x: x["similarity"], reverse=True)

    return best_matches_a_to_b, best_matches_b_to_a


def hungarian_matching(
    similarity_matrix,
    threshold: float = 0.6,
):
    """Compute optimal 1-to-1 theme matching using Hungarian algorithm.

    Args:
        similarity_matrix: (n_A x n_B) numpy array of similarities
        threshold: Similarity threshold for considering a match valid

    Returns:
        Dictionary with:
        - matched_pairs: List of (i, j, similarity) for matched pairs above threshold
        - all_pairs: List of all optimal pairs regardless of threshold
        - thresholded_metrics: Dict with coverage_a, coverage_b, true_jaccard (and legacy
          precision/recall/f1 for backward compatibility)
        - soft_metrics: Dict with soft_precision (mean assignment similarity),
          soft_recall (normalised total similarity), soft_f1
        - distribution: Dict with min, q1, median, q3, max of matched similarities
    """
    from scipy.optimize import linear_sum_assignment

    n_A, n_B = similarity_matrix.shape

    # handle empty sets
    if n_A == 0 or n_B == 0:
        return {
            "matched_pairs": [],
            "all_pairs": [],
            "thresholded_metrics": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "true_jaccard": 0.0,
                "coverage_a": 0.0,
                "coverage_b": 0.0,
            },
            "soft_metrics": {
                "soft_precision": 0.0,
                "soft_recall": 0.0,
                "soft_f1": 0.0,
            },
            "distribution": {
                "min": 0.0,
                "q1": 0.0,
                "median": 0.0,
                "q3": 0.0,
                "max": 0.0,
                "n_pairs": 0,
            },
        }

    # pad to square matrix for Hungarian algorithm
    size = max(n_A, n_B)
    sim_padded = np.zeros((size, size))
    sim_padded[:n_A, :n_B] = similarity_matrix

    # Hungarian algorithm minimizes cost, so convert similarity to cost
    cost = 1 - sim_padded
    row_ind, col_ind = linear_sum_assignment(cost)

    # filter out padding and extract real pairs
    all_pairs = [
        (int(i), int(j), float(similarity_matrix[i, j]))
        for i, j in zip(row_ind, col_ind)
        if i < n_A and j < n_B
    ]

    # filter pairs above threshold for thresholded metrics
    matched_pairs = [(i, j, sim) for i, j, sim in all_pairs if sim >= threshold]

    # extract similarities for all optimal pairs (for soft metrics)
    all_sims = np.array([sim for _, _, sim in all_pairs])

    # extract similarities for matched pairs (for distribution)
    matched_sims = (
        np.array([sim for _, _, sim in matched_pairs])
        if matched_pairs
        else np.array([])
    )

    # === THRESHOLDED METRICS (COVERAGE) ===
    n_matched = len(matched_pairs)  # pairs above threshold

    # Coverage: proportion of each set that has a good match (above threshold)
    coverage_a = n_matched / n_A if n_A > 0 else 0.0
    coverage_b = n_matched / n_B if n_B > 0 else 0.0

    # True Jaccard: intersection / union (matched pairs / total unique themes)
    # Useful for comparison with Raza et al.
    true_jaccard = n_matched / (n_A + n_B - n_matched + 1e-9)

    # Legacy metrics (kept for backward compatibility, but not recommended for thematic analysis)
    TP = n_matched
    FP = len(all_pairs) - TP  # optimal pairs below threshold
    FN = max(n_A, n_B) - TP  # unmatched items from larger set
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # === FIDELITY METRICS (using Shepard similarities - valid to average) ===
    # Mean assignment similarity: average quality of all optimal pairs
    soft_precision = float(all_sims.mean()) if len(all_sims) > 0 else 0.0
    # Normalised total similarity: sum of similarities / larger set size
    soft_recall = float(all_sims.sum()) / max(n_A, n_B)
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall + 1e-9)

    # === DIRECTIONAL BEST-MATCH METRICS ===
    # Forward: for each A, find best match in B
    forward_similarity = (
        float(np.mean(np.max(similarity_matrix, axis=1))) if n_A > 0 else 0.0
    )
    # Backward: for each B, find best match in A
    backward_similarity = (
        float(np.mean(np.max(similarity_matrix, axis=0))) if n_B > 0 else 0.0
    )
    # Bidirectional: harmonic mean of forward and backward
    bidirectional_similarity = (
        2
        * forward_similarity
        * backward_similarity
        / (forward_similarity + backward_similarity + 1e-9)
    )
    # Hungarian: mean similarity of optimal 1:1 pairs
    hungarian_similarity = soft_precision  # same as soft_precision, clearer name

    # === DISTRIBUTION STATS (for all optimal pairs) ===
    if len(all_sims) > 0:
        distribution = {
            "min": float(all_sims.min()),
            "q1": float(np.percentile(all_sims, 25)),
            "median": float(np.median(all_sims)),
            "q3": float(np.percentile(all_sims, 75)),
            "max": float(all_sims.max()),
            "n_pairs": len(all_sims),
        }
    else:
        distribution = {
            "min": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "max": 0.0,
            "n_pairs": 0,
        }

    return {
        "matched_pairs": matched_pairs,
        "all_pairs": all_pairs,
        "thresholded_metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_jaccard": float(true_jaccard),
            "coverage_a": float(coverage_a),
            "coverage_b": float(coverage_b),
        },
        "soft_metrics": {
            "soft_precision": soft_precision,
            "soft_recall": soft_recall,
            "soft_f1": soft_f1,
        },
        "directional_metrics": {
            "forward_similarity": forward_similarity,
            "backward_similarity": backward_similarity,
            "bidirectional_similarity": bidirectional_similarity,
            "hungarian_similarity": hungarian_similarity,
        },
        "distribution": distribution,
    }
