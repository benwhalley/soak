"""Theme and code similarity comparison using embeddings."""

import base64
import itertools
import logging
import textwrap
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from soak.models import QualitativeAnalysis, QualitativeAnalysisComparison
from soak.models.base import get_embedding, memory

logger = logging.getLogger(__name__)


def format_similarity_matrix(
    matrix,
    row_names: List[str],
    col_names: List[str],
    round_dp: int = 2,
    set_a_name: str = "A",
    set_b_name: str = "B",
    show_legend: bool = False,
) -> str:
    """Format similarity matrix for console output with numbered indices and legend.

    Args:
        matrix: Similarity matrix to format (numpy array)
        row_names: Names for rows (theme names from set A)
        col_names: Names for columns (theme names from set B)
        round_dp: Number of decimal places for rounding
        set_a_name: Name of set A (for legend)
        set_b_name: Name of set B (for legend)
        show_legend: Whether to include the legend mapping indices to theme names

    Returns:
        Formatted string with optional legend and numbered matrix
    """
    import numpy as np
    import pandas as pd

    output_parts = []

    # create legend if requested
    if show_legend:
        legend_lines = []
        legend_lines.append(f"\n{set_a_name} Themes (rows):")
        for i, name in enumerate(row_names):
            legend_lines.append(f"  {i}: {name}")

        legend_lines.append(f"\n{set_b_name} Themes (columns):")
        for i, name in enumerate(col_names):
            legend_lines.append(f"  {i}: {name}")

        output_parts.append("\n".join(legend_lines))

    # create numbered matrix
    df = pd.DataFrame(
        np.round(matrix, round_dp),
        index=[str(i) for i in range(len(row_names))],
        columns=[str(i) for i in range(len(col_names))]
    )

    output_parts.append(str(df))

    return "\n\n".join(output_parts) if show_legend else output_parts[0]


def compute_ot(
    shepard_similarity,
    compute_baselines: bool = True,
    n_bootstrap: int = 100,
):
    """Compute optimal transport metrics for theme similarity.

    Args:
        shepard_similarity: (n_A x n_B) numpy array of Shepard similarities
        compute_baselines: Whether to compute null baseline
        n_bootstrap: Number of bootstrap iterations for null distribution

    Returns:
        Dictionary with OT metrics and null baseline
    """
    import numpy as np
    import ot

    n_A, n_B = shepard_similarity.shape

    if n_A == 0 or n_B == 0:
        return {
            "similarity": 0.0,
            "cost": 1.0,
            "concentration": 0.0,
            "coverage_a": [],
            "coverage_b": [],
            "null_mean": 0.0,
            "null_95pct": 0.0,
            "null_relative": 0.0,
        }

    # uniform mass distribution
    a = np.ones(n_A) / n_A
    b = np.ones(n_B) / n_B

    # cost matrix (lower = better)
    cost = 1 - shepard_similarity

    # compute optimal transport coupling
    P = ot.emd(a, b, cost)
    ot_cost = float(np.sum(P * cost))
    ot_sim = 1 - ot_cost

    # mass concentration -- how focused is the alignment?
    flat = np.sort(P.flatten())[::-1]  # descending
    concentration_k5 = float(flat[:5].sum() / (flat.sum() + 1e-9))

    # coverage -- how much of each theme's mass maps to the other set
    coverage_a = P.sum(axis=1).tolist()  # how much of each A maps to B
    coverage_b = P.sum(axis=0).tolist()  # how much of each B maps to A

    result = {
        "similarity": float(ot_sim),
        "cost": float(ot_cost),
        "concentration": float(concentration_k5),
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
    }

    if compute_baselines:
        # null distribution via bootstrap
        logger.debug(f"Computing null distribution with {n_bootstrap} bootstrap iterations")
        null_sims = []
        for _ in range(n_bootstrap):
            # permute columns (shuffle B themes)
            perm = np.random.permutation(n_B)
            shuf = shepard_similarity[:, perm]
            cost_shuf = 1 - shuf
            P_shuf = ot.emd(a, b, cost_shuf)
            null_sim = float(1 - np.sum(P_shuf * cost_shuf))
            null_sims.append(null_sim)

        null_dist = np.array(null_sims)
        null_mean = float(null_dist.mean())
        null_95 = float(np.percentile(null_dist, 95))

        # null-relative: scale from random (0) to perfect (1)
        null_relative = (ot_sim - null_mean) / (1.0 - null_mean + 1e-9)

        result["null_mean"] = null_mean
        result["null_95pct"] = null_95
        result["null_distribution"] = null_dist.tolist()
        result["null_relative"] = float(null_relative)

    return result


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
        - thresholded_metrics: Dict with precision, recall, f1, true_jaccard
        - soft_metrics: Dict with soft_precision, soft_recall, soft_f1
        - distribution: Dict with min, q1, median, q3, max of matched similarities
    """
    import numpy as np
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
    matched_pairs = [
        (i, j, sim) for i, j, sim in all_pairs if sim >= threshold
    ]

    # extract similarities for all optimal pairs (for soft metrics)
    all_sims = np.array([sim for _, _, sim in all_pairs])

    # extract similarities for matched pairs (for distribution)
    matched_sims = np.array([sim for _, _, sim in matched_pairs]) if matched_pairs else np.array([])

    # === THRESHOLDED METRICS ===
    TP = len(matched_pairs)  # true positives
    FP = len(all_pairs) - TP  # false positives (matched but below threshold)
    FN = max(n_A, n_B) - TP  # false negatives (unmatched from larger set)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # true Jaccard: intersection / union
    true_jaccard = TP / (n_A + n_B - TP + 1e-9)

    # coverage metrics
    coverage_a = TP / n_A if n_A > 0 else 0.0
    coverage_b = TP / n_B if n_B > 0 else 0.0

    # === SOFT METRICS (using Shepard similarities - valid to average) ===
    soft_precision = float(all_sims.mean()) if len(all_sims) > 0 else 0.0
    soft_recall = float(all_sims.sum()) / max(n_A, n_B)
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall + 1e-9)

    # === DISTRIBUTION STATS (for matched pairs above threshold) ===
    if len(matched_sims) > 0:
        distribution = {
            "min": float(matched_sims.min()),
            "q1": float(np.percentile(matched_sims, 25)),
            "median": float(np.median(matched_sims)),
            "q3": float(np.percentile(matched_sims, 75)),
            "max": float(matched_sims.max()),
            "n_pairs": len(matched_sims),
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
        "distribution": distribution,
    }


class Base64ImageFile:
    """Simple wrapper for BytesIO that provides base64 encoding."""

    def __init__(self, buffer, name=None):
        self.buffer = buffer
        self.name = name

    @property
    def base64(self):
        self.buffer.seek(0)
        return base64.b64encode(self.buffer.read()).decode("utf-8")


@memory.cache
def compare_result_similarity(
    A: QualitativeAnalysis,
    B: QualitativeAnalysis,
    threshold: float = 0.6,
    embedding_template: str = "{name}",
    embedding_backend: str = "local",
    embedding_model: str = "all-MiniLM-L6-v2",
    k: float = 1.0,
) -> Dict[str, Any]:
    """
    Compare two sets of theme embeddings.

    Allows many-to-one matches: each theme may match multiple from the other set.

    Args:
        A: First QualitativeAnalysis to compare
        B: Second QualitativeAnalysis to compare
        threshold: Similarity threshold for matching (default: 0.6)
        embedding_template: Python format string for generating embeddings from themes.
                          Available fields: {name}, {description}
                          Default: "{name}"
        k: Shepard similarity decay parameter (default: 1.0)

    Returns:
        Dictionary with similarity metrics including:
        - precision: % of B themes with at least one A match
        - recall: % of A themes with at least one B match
        - f1: harmonic mean of precision and recall
        - jaccard: proportion of theme pairs with similarity > threshold
        - match_matrix: binary matrix [n_A x n_B], 1 = similarity above threshold
        - similarity_matrix: raw cosine similarity values
        - angle_similarity_matrix: angular distance normalized to [0,1]
        - shepard_similarity_matrix: Shepard similarity with specified k
        - percentile_normalized_shepard: Shepard normalized by within-set percentiles
        - z_score_normalized_shepard: Shepard normalized by within-set z-scores
    """

    # extract theme names and analysis names before reassigning A and B
    theme_names_A = [theme.name for theme in A.themes]
    theme_names_B = [theme.name for theme in B.themes]
    analysis_name_A = A.name
    analysis_name_B = B.name

    A = [
        embedding_template.format(name=i.name, description=i.description)
        for i in A.themes
    ]
    B = [
        embedding_template.format(name=i.name, description=i.description)
        for i in B.themes
    ]

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    logger.debug("Getting embeddings for A and B")
    emb_A = get_embedding(
        list(map(lambda x: x.strip(), A)),
        backend=embedding_backend,
        model_name=embedding_model,
    )
    emb_B = get_embedding(
        list(map(lambda x: x.strip(), B)),
        backend=embedding_backend,
        model_name=embedding_model,
    )
    logger.debug("Got embeddings for A and B")
    assert len(emb_A) == len(A), f"Mismatch in emb_A: {len(emb_A)} != {len(A)}"
    assert len(emb_B) == len(B), f"Mismatch in emb_B: {len(emb_B)} != {len(B)}"

    # Handle empty theme sets
    if len(emb_A) == 0 or len(emb_B) == 0:
        n_A = len(emb_A)
        n_B = len(emb_B)
        return {
            "error": "No themes found in any results. Cannot perform similarity comparison.",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "match_matrix": np.zeros((n_A, n_B), dtype=int),
            "similarity_matrix": np.zeros((n_A, n_B)),
            "a_b_most_similar": 0.0,
            "b_a_most_similar": 0.0,
            "similarity_f1": 0.0,
        }

    sim_matrix = cosine_similarity(emb_A, emb_B)

    # angular distance similarity (proper metric, normalized to [0,1])
    angle_matrix = np.degrees(np.arccos(np.clip(sim_matrix, -1.0, 1.0)))
    angle_sim = 1 - angle_matrix / 180.0

    # Shepard similarity with configurable k parameter
    theta = np.arccos(np.clip(sim_matrix, -1.0, 1.0))
    shepard_sim = (np.exp(-k * theta) - np.exp(-k * np.pi)) / (1 - np.exp(-k * np.pi))

    # within-set Shepard similarities for normalization
    def pairwise_shepard(emb, k_val):
        cos = cosine_similarity(emb, emb)
        theta_inner = np.arccos(np.clip(cos, -1, 1))
        S = (np.exp(-k_val * theta_inner) - np.exp(-k_val * np.pi)) / (1 - np.exp(-k_val * np.pi))
        # drop diagonal and duplicates
        n = S.shape[0]
        iu = np.triu_indices(n, k=1)
        return S[iu]

    S_within_A = pairwise_shepard(emb_A, k)
    S_within_B = pairwise_shepard(emb_B, k)
    S_within = np.concatenate([S_within_A, S_within_B])

    # percentile normalization
    S_within_sorted = np.sort(S_within)

    def percentile_scale(x):
        # proportion of within-set pairs with similarity <= x
        return np.searchsorted(S_within_sorted, x, side="right") / len(S_within_sorted)

    shepard_percentile = np.vectorize(percentile_scale)(shepard_sim)

    # z-score normalization
    mu = S_within.mean()
    sigma = S_within.std()
    shepard_z = (shepard_sim - mu) / (sigma + 1e-9)

    # === HUNGARIAN MATCHING (optimal 1-to-1 assignment) ===
    # use Shepard similarity for soft F1 (valid to average, unlike cosine)
    hungarian_results = hungarian_matching(shepard_sim, threshold=threshold)

    # log Hungarian results
    logger.info(f"\n=== Hungarian Matching (1-to-1, Shepard k={k}) ===")
    logger.info(f"Optimal assignment: {hungarian_results['distribution']['n_pairs']}/{min(len(emb_A), len(emb_B))} pairs above threshold")
    logger.info(f"Coverage: {hungarian_results['thresholded_metrics']['coverage_a']:.1%} of A, {hungarian_results['thresholded_metrics']['coverage_b']:.1%} of B")
    logger.info(f"Thresholded -- Precision: {hungarian_results['thresholded_metrics']['precision']:.3f}, Recall: {hungarian_results['thresholded_metrics']['recall']:.3f}, F1: {hungarian_results['thresholded_metrics']['f1']:.3f}")
    logger.info(f"Soft metrics -- Precision: {hungarian_results['soft_metrics']['soft_precision']:.3f}, Recall: {hungarian_results['soft_metrics']['soft_recall']:.3f}, F1: {hungarian_results['soft_metrics']['soft_f1']:.3f}")
    logger.info(f"True Jaccard: {hungarian_results['thresholded_metrics']['true_jaccard']:.3f}")

    dist = hungarian_results['distribution']
    if dist['n_pairs'] > 0:
        logger.info(f"Similarity distribution: median={dist['median']:.3f} (Q1={dist['q1']:.3f}, Q3={dist['q3']:.3f}, range: {dist['min']:.3f}-{dist['max']:.3f})")

    # === OPTIMAL TRANSPORT (many-to-many alignment) ===
    logger.info("\n=== Computing Optimal Transport Metrics ===")
    ot_results = compute_ot(shepard_sim, compute_baselines=True, n_bootstrap=100)

    # log OT results
    ot_sim = ot_results["similarity"]
    logger.info(f"OT Similarity: {ot_sim:.3f}")
    logger.info(f"OT Cost: {ot_results['cost']:.3f}")
    logger.info(f"OT Concentration (top-5): {ot_results['concentration']:.3f}")
    logger.info(f"Null distribution: mean={ot_results['null_mean']:.3f}, 95th percentile={ot_results['null_95pct']:.3f}")
    logger.info(f"Null-relative similarity: {ot_results['null_relative']:.3f} (0=random, 1=perfect)")

    # log all matrices (show legend only once at the start)
    logger.info("\n=== Theme Index Legend ===")
    logger.info(f"\n{analysis_name_A} Themes (rows):")
    for i, name in enumerate(theme_names_A):
        logger.info(f"  {i}: {name}")
    logger.info(f"\n{analysis_name_B} Themes (columns):")
    for i, name in enumerate(theme_names_B):
        logger.info(f"  {i}: {name}")

    logger.info("\n=== Cosine Similarity ===\n" + format_similarity_matrix(
        sim_matrix, theme_names_A, theme_names_B, set_a_name=analysis_name_A, set_b_name=analysis_name_B, show_legend=False
    ))

    logger.info("\n=== Angular Similarity (normalized) ===\n" + format_similarity_matrix(
        angle_sim, theme_names_A, theme_names_B, set_a_name=analysis_name_A, set_b_name=analysis_name_B, show_legend=False
    ))

    logger.info(f"\n=== Shepard Similarity (k={k}) ===\n" + format_similarity_matrix(
        shepard_sim, theme_names_A, theme_names_B, set_a_name=analysis_name_A, set_b_name=analysis_name_B, show_legend=False
    ))

    logger.info("\n=== Percentile-Normalized Shepard ===\n" + format_similarity_matrix(
        shepard_percentile, theme_names_A, theme_names_B, set_a_name=analysis_name_A, set_b_name=analysis_name_B, show_legend=False
    ))

    logger.info("\n=== Z-Score Normalized Shepard ===\n" + format_similarity_matrix(
        shepard_z, theme_names_A, theme_names_B, set_a_name=analysis_name_A, set_b_name=analysis_name_B, show_legend=False
    ))
    
    match_matrix = sim_matrix >= threshold

    # Recall: % of A themes with any match
    recall_hits = match_matrix.any(axis=1).sum()
    recall = recall_hits / len(emb_A) if len(emb_A) > 0 else 0

    # Precision: % of B themes with any match
    precision_hits = match_matrix.any(axis=0).sum()
    precision = precision_hits / len(emb_B) if len(emb_B) > 0 else 0

    f1 = (
        0
        if (precision + recall) == 0
        else 2 * (precision * recall) / (precision + recall)
    )

    # Jaccard: intersection / union across all pairwise theme comparisons
    intersection = match_matrix.sum()
    union = match_matrix.size  # total possible pairs = n_A * n_B
    jaccard = intersection / union if union > 0 else 0

    # best match of A themes with any B score
    a_b_most_similar = sim_matrix.max(axis=1).mean().round(3) if len(emb_A) > 0 else 0

    # best match of B themes with any A score
    b_a_most_similar = sim_matrix.max(axis=0).mean().round(3) if len(emb_B) > 0 else 0

    similarity_f1 = (
        2
        * (a_b_most_similar * b_a_most_similar)
        / (a_b_most_similar + b_a_most_similar)
        if (a_b_most_similar + b_a_most_similar) > 0
        else 0
    )

    # For each theme in A, find the best matching theme in B
    best_matches_a_to_b = []
    if len(emb_A) > 0 and len(emb_B) > 0:
        for i in range(len(emb_A)):
            best_b_idx = sim_matrix[i, :].argmax()
            best_similarity = sim_matrix[i, best_b_idx]
            best_matches_a_to_b.append(
                {
                    "theme_a_index": i,
                    "theme_b_index": int(best_b_idx),
                    "similarity": float(np.round(best_similarity, 3)),
                }
            )

    # For each theme in B, find the best matching theme in A
    best_matches_b_to_a = []
    if len(emb_A) > 0 and len(emb_B) > 0:
        for j in range(len(emb_B)):
            best_a_idx = sim_matrix[:, j].argmax()
            best_similarity = sim_matrix[best_a_idx, j]
            best_matches_b_to_a.append(
                {
                    "theme_b_index": j,
                    "theme_a_index": int(best_a_idx),
                    "similarity": float(np.round(best_similarity, 3)),
                }
            )

    return {
        # binary threshold metrics
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "match_matrix": match_matrix.astype(int),
        # continuous similarity metrics
        "similarity_matrix": np.round(sim_matrix, 3),
        "angle_similarity_matrix": np.round(angle_sim, 3),
        "shepard_similarity_matrix": np.round(shepard_sim, 3),
        "shepard_k_value": k,
        # normalized metrics
        "percentile_normalized_shepard": np.round(shepard_percentile, 3),
        "z_score_normalized_shepard": np.round(shepard_z, 3),
        # within-set statistics used for normalization
        "within_set_stats": {
            "mean": float(mu),
            "std": float(sigma),
            "n_pairs": len(S_within),
        },
        # Hungarian matching (optimal 1-to-1 assignment using Shepard similarity)
        "hungarian": hungarian_results,
        # Optimal Transport (many-to-many alignment using Shepard similarity)
        "ot": ot_results,
        # aggregated metrics (legacy - averaging cosine is problematic)
        "a_b_most_similar": a_b_most_similar,
        "b_a_most_similar": b_a_most_similar,
        "similarity_f1": similarity_f1,
        # best matches
        "best_matches_a_to_b": best_matches_a_to_b,
        "best_matches_b_to_a": best_matches_b_to_a,
    }


@memory.cache
def network_similarity_plot(
    pipeline_results: List[QualitativeAnalysis],
    method="umap",
    n_neighbors=5,
    min_dist=0.01,
    threshold=0.6,
    exclude_within_set_edges=True,
    embedding_template="{name}",
    embedding_backend: str = "local",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> str:
    """Create similarity plot using embedding visualization.

    Args:
        pipeline_results: List of QualitativeAnalysis objects to compare
        method: Dimensionality reduction method ("umap", "mds", or "pca")
        n_neighbors: UMAP parameter for number of neighbors
        min_dist: UMAP parameter for minimum distance
        threshold: Similarity threshold for drawing edges
        exclude_within_set_edges: If True, don't draw edges between themes from the same set
        embedding_template: Python format string for embeddings. Available: {name}, {description}

    Note:
        Theme labels must be set before calling this function (via theme.set_label())
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_similarity
    from umap import UMAP

    # Extract themes using embedding_template for similarity calculation
    theme_sets_for_embedding_ = [
        [
            embedding_template.format(name=j.name, description=j.description)
            for j in i.themes
        ]
        for i in pipeline_results
    ]
    theme_sets_for_embedding = [i for i in theme_sets_for_embedding_ if i]

    # Extract theme labels for display
    theme_sets_for_labels_ = [
        [theme.label for theme in result.themes]
        for result in pipeline_results
    ]
    theme_sets_for_labels = [i for i in theme_sets_for_labels_ if i]

    pipeline_names = [i.name for i in pipeline_results]

    # Get embeddings for all sets using embedding_template
    embeddings = [
        get_embedding(set_str, backend=embedding_backend, model_name=embedding_model)
        for set_str in theme_sets_for_embedding
    ]
    all_emb = np.vstack(embeddings)

    # Calculate similarity matrix
    sim_matrix = cosine_similarity(all_emb)

    # Create graph
    G = nx.Graph()
    start_index = 0
    colors = [plt.cm.Set1(i) for i in range(len(embeddings))]
    valid_indices = list(range(len(embeddings)))

    # Track which set each node belongs to
    node_to_set = {}

    for plot_idx, (emb, original_idx) in enumerate(zip(embeddings, valid_indices)):
        set_str = theme_sets_for_labels[original_idx]
        lines = set_str
        for i, phrase in enumerate(lines, start=start_index):
            if not phrase.strip():
                continue
            G.add_node(i, label=phrase, set=chr(65 + plot_idx))
            node_to_set[i] = plot_idx
        start_index += len(emb)

    # Create 2D embedding for visualization
    if method == "umap":
        # Adjust n_neighbors if it's too large for the dataset
        effective_n_neighbors = min(n_neighbors, len(all_emb) - 1)
        effective_n_neighbors = max(2, effective_n_neighbors)

        reducer = UMAP(
            n_components=2,
            n_neighbors=effective_n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        pos_2d = reducer.fit_transform(all_emb)
    elif method == "mds":
        # Classical MDS expects a distance matrix, so convert similarity
        dist_matrix = 1 - sim_matrix
        reducer = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        pos_2d = reducer.fit_transform(dist_matrix)
    else:
        reducer = PCA(n_components=2)
        pos_2d = reducer.fit_transform(all_emb)

    pos = {i: pos_2d[i] for i in range(len(all_emb))}

    # Add edges based on threshold
    for i in range(len(all_emb)):
        for j in range(i + 1, len(all_emb)):
            # Skip edges within the same set if requested
            if exclude_within_set_edges:
                if i in node_to_set and j in node_to_set:
                    if node_to_set[i] == node_to_set[j]:
                        continue

            if sim_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=sim_matrix[i, j])

    # Create plot
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 10))
    node_colors = [colors[ord(G.nodes[n].get("set", "?")) - 65] for n in G.nodes]

    # Draw nodes with colors
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, alpha=0.8, node_size=200, ax=ax
    )

    # Add legend for sets
    legend_labels = [pipeline_names[idx] for idx in valid_indices]
    for i, label in enumerate(legend_labels):
        ax.scatter([], [], color=colors[i], label=label)
    ax.legend(title="Pipeline Results", loc="upper right")

    # Draw edges with alpha proportional to similarity weight
    edges = G.edges(data=True)
    for u, v, d in edges:
        weight = d["weight"]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            alpha=np.clip(weight, 0.1, 1.0),
            width=2,
            ax=ax,
        )

    labels = nx.get_node_attributes(G, "label")

    wrapped_labels = {k: textwrap.fill(label, width=20) for k, label in labels.items()}
    label_pos = {k: (v[0] + 0.05, v[1]) for k, v in pos.items()}
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=wrapped_labels,
        font_size=7,
        verticalalignment="top",
        horizontalalignment="left",
        ax=ax,
    )

    # Update title to indicate if within-set edges are excluded
    edge_info = " (cross-set edges only)" if exclude_within_set_edges else ""

    ax.text(
        1.0,
        -0.15,
        f"Theme similarity network with {method} layout (threshold={threshold}){edge_info}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
    )
    ax.axis("off")
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Save plot
    buffer = BytesIO()
    fig.savefig(buffer, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)
    return Base64ImageFile(buffer, name="similarity_plot.png")


@memory.cache
def create_pairwise_heatmap(
    a: QualitativeAnalysis,
    b: QualitativeAnalysis,
    threshold=0.6,
    use_threshold=True,
    embedding_template="{name}",
    sort_by_average=False,
    embedding_backend: str = "local",
    embedding_model: str = "all-MiniLM-L6-v2",
    metric_type: str = "cosine",
    k: float = 1.0,
) -> str:
    """Create a heatmap visualization for a single pair of pipeline results.

    Args:
        a: First QualitativeAnalysis
        b: Second QualitativeAnalysis
        threshold: Similarity threshold for matching
        use_threshold: Whether to use threshold-based binary heatmap
        embedding_template: Python format string for embeddings. Available: {name}, {description}
        sort_by_average: If True, sort rows and columns by average similarity (highest first)
        metric_type: Type of similarity metric ("cosine", "angle", "shepard", "percentile", "z_score")
        k: Shepard similarity decay parameter (default: 1.0)

    Note:
        Theme labels must be set before calling this function (via theme.set_label())
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-GUI backend for headless use (saves to file only)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    def truncate_theme(theme: str, max_len: int = 25) -> str:  # Reduced from 40
        if len(theme) <= max_len:
            return theme
        return theme[: max_len - 3] + "..."

    themes_a = [theme.label for theme in a.themes]
    themes_b = [theme.label for theme in b.themes]
    themes_a_display = [truncate_theme(t) for t in themes_a]
    themes_b_display = [truncate_theme(t) for t in themes_b]

    # Better figure sizing accounting for label length
    avg_label_len_b = np.mean([len(label) for label in themes_b_display])
    fig_height = max(8, len(themes_a) * 0.4)
    fig_width = max(
        12, len(themes_b) * 0.5 + avg_label_len_b * 0.1
    )  # Account for label width

    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    comparison = compare_result_similarity(
        a,
        b,
        threshold=threshold or 0.5,  # ensure not None
        embedding_template=embedding_template,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        k=k,
    )

    # select matrix based on metric type
    metric_labels = {
        "cosine": ("Cosine Similarity", "similarity_matrix"),
        "angle": ("Angular Similarity", "angle_similarity_matrix"),
        "shepard": (f"Shepard Similarity (k={k})", "shepard_similarity_matrix"),
        "percentile": ("Percentile-Normalized Shepard", "percentile_normalized_shepard"),
        "z_score": ("Z-Score Normalized Shepard", "z_score_normalized_shepard"),
    }

    metric_label, matrix_key = metric_labels.get(metric_type, ("Cosine Similarity", "similarity_matrix"))
    similarity_matrix = comparison[matrix_key]

    # Sort by average similarity if requested
    if sort_by_average:
        # Calculate average similarity for each row (A themes) and column (B themes)
        row_averages = similarity_matrix.mean(axis=1)  # average across B themes
        col_averages = similarity_matrix.mean(axis=0)  # average across A themes

        # Get sorting indices (descending order -- highest similarity first)
        row_order = np.argsort(row_averages)[::-1]
        col_order = np.argsort(col_averages)[::-1]

        # Reorder matrix and labels together
        similarity_matrix = similarity_matrix[row_order, :][:, col_order]
        themes_a_display = [themes_a_display[i] for i in row_order]
        themes_b_display = [themes_b_display[i] for i in col_order]

    df_sim = pd.DataFrame(
        similarity_matrix, index=themes_a_display, columns=themes_b_display
    )

    assert similarity_matrix.shape == (
        len(themes_a_display),
        len(themes_b_display),
    ), f"Shape mismatch: {similarity_matrix.shape} vs {len(themes_a_display)} x {len(themes_b_display)}"

    if use_threshold:
        df_binary = (df_sim >= threshold).astype(int)
        cmap = LinearSegmentedColormap.from_list(
            "threshold_cmap", ["white", "green"], N=2
        )

        data = df_binary
        annot = False
        vmin = 0  # Explicitly set minimum
        vmax = 1  # Explicitly set maximum
    else:
        data = df_sim
        # Use perceptually uniform colormap (viridis is colorblind-safe)
        cmap = "viridis"
        annot = True
        vmin = None  # Fixed scale from 0
        vmax = None  # Cap at 0.8 - anything higher shows as max color (yellow)

    # Create heatmap with better spacing
    sns.heatmap(
        data,
        annot=annot,
        fmt=".2f" if annot else None,
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={"label": "Match" if use_threshold else metric_label},
        ax=ax,
        square=False,  # Don't force square aspect ratio
        vmin=vmin,  # Add explicit scale limits
        vmax=vmax,  # Add explicit scale limits
    )

    threshold_str = f". Threshold: {threshold}" if use_threshold else ""
    ax.set_title(
        f"{metric_label}\n{a.name} vs {b.name}{threshold_str}"
    )
    ax.set_xlabel(b.name)
    ax.set_ylabel(a.name)

    # Better tick label handling
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    ax.set_aspect("equal")

    # Ensure labels are properly positioned
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    # Use tight_layout with padding
    fig.tight_layout(pad=2.0)

    # Additional spacing adjustment if needed
    plt.subplots_adjust(bottom=0.2)  # Add extra space at bottom for rotated labels

    threshold_suffix = f"_threshold={threshold}" if use_threshold else ""
    metric_suffix = f"_{metric_type}" if metric_type != "cosine" else ""
    plot_name = f"heatmap_{a.name}_{b.name}{metric_suffix}{threshold_suffix}.png"
    buffer = BytesIO()
    fig.savefig(buffer, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)

    return Base64ImageFile(buffer, name=plot_name)


class SimilarityComparator:
    """Comparator calculates similarity statistics and makes plot/heatmaps."""

    def compare(self, pipeline_results: List[QualitativeAnalysis], config={}):
        threshold = config.get("threshold", 0.6)
        n_neighbors = config.get("n_neighbors", 5)
        min_dist = config.get("min_dist", 0.01)
        method = config.get("method", "umap")
        label_template = config.get("label_template", "{name}")
        embedding_template = config.get("embedding_template", "{name}")
        sort_by_average = config.get("sort_by_average", False)
        embedding_backend = config.get("embedding_backend", "local")
        embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        k = config.get("k", 1.0)

        # Set labels on all themes once at the beginning
        for result in pipeline_results:
            for i, theme in enumerate(result.themes, start=1):
                theme.set_label(label_template, i)

        result_combinations = list(itertools.combinations(pipeline_results, 2))

        # Build embedded strings mapping for each result
        embedded_strings_map = {}
        for result in pipeline_results:
            embedded_strings_map[result.name] = [
                {
                    "theme_name": theme.name,
                    "theme_description": theme.description,
                    "label": theme.label,
                    "embedded_string": embedding_template.format(
                        name=theme.name, description=theme.description
                    ),
                }
                for theme in result.themes
            ]

        # run synchronously
        similarity_results = [
            compare_result_similarity(
                i,
                j,
                threshold=threshold,
                embedding_template=embedding_template,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                k=k,
            )
            for i, j in result_combinations
        ]

        # generate heatmaps for all metric types
        metric_types = ["cosine", "angle", "shepard", "percentile", "z_score"]

        heatmaps_by_metric = {}
        for metric_type in metric_types:
            heatmaps_by_metric[metric_type] = [
                create_pairwise_heatmap(
                    a,
                    b,
                    threshold=threshold,
                    use_threshold=False,
                    embedding_template=embedding_template,
                    sort_by_average=sort_by_average,
                    embedding_backend=embedding_backend,
                    embedding_model=embedding_model,
                    metric_type=metric_type,
                    k=k,
                )
                for a, b in result_combinations
            ]

        # keep legacy cosine heatmaps for backward compatibility
        heatmaps = heatmaps_by_metric["cosine"]

        # thresholded heatmaps (only meaningful for cosine similarity)
        thresholded_heatmaps = [
            create_pairwise_heatmap(
                a,
                b,
                threshold=threshold,
                use_threshold=True,
                embedding_template=embedding_template,
                sort_by_average=sort_by_average,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                metric_type="cosine",
                k=k,
            )
            for a, b in result_combinations
        ]

        network_plot = network_similarity_plot(
            [i for i in pipeline_results],
            method=method,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            threshold=threshold,
            embedding_template=embedding_template,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )

        result_combinations_dict = OrderedDict(
            {i.name + "_" + j.name: (i, j) for i, j in result_combinations}
        )

        stats_dict = {
            k: v for k, v in zip(result_combinations_dict.keys(), similarity_results)
        }

        # create dictionaries for all metric heatmaps
        heatmap_dicts_by_metric = {
            metric_type: {
                k: v for k, v in zip(result_combinations_dict.keys(), heatmaps_list)
            }
            for metric_type, heatmaps_list in heatmaps_by_metric.items()
        }

        # legacy compatibility
        heatmap_dict = heatmap_dicts_by_metric["cosine"]

        thresh_heatmap_dict = {
            k: v for k, v in zip(result_combinations_dict.keys(), thresholded_heatmaps)
        }

        return QualitativeAnalysisComparison(
            results=pipeline_results,
            combinations=result_combinations_dict,
            statistics=stats_dict,
            comparison_plots={
                "heatmaps": heatmap_dict,
                "thresholded_heatmaps": thresh_heatmap_dict,
                # all metric-specific heatmaps
                "heatmaps_cosine": heatmap_dicts_by_metric["cosine"],
                "heatmaps_angle": heatmap_dicts_by_metric["angle"],
                "heatmaps_shepard": heatmap_dicts_by_metric["shepard"],
                "heatmaps_percentile": heatmap_dicts_by_metric["percentile"],
                "heatmaps_z_score": heatmap_dicts_by_metric["z_score"],
            },
            additional_plots={
                "network_plot": network_plot,
            },
            config=config,
            embedded_strings=embedded_strings_map,
        )


if False:
    from wellspring.models import Analysis

    pipeline_results = [
        QualitativeAnalysis(**j)
        for j in [
            i.result_json for i in Analysis.objects.filter(result_json__isnull=False)
        ][-6:]
        if isinstance(j, dict)
    ]
    pipeline_results[0].name

    x = list(reversed(pipeline_results))
    comp = SimilarityComparator().compare(pipeline_results)
