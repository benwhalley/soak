"""Theme and code similarity comparison using embeddings.

This module provides functions and classes for comparing thematic analyses
using embedding similarity and optimal transport.

The module is organized into submodules:
- rescaling: Similarity matrix rescaling methods
- optimal_transport: Optimal transport computation
- baselines: Baseline generation (word-salad, permutation)
- paraphrasing: LLM-based paraphrase generation
- visualizations: Visualization functions (Sankey, heatmaps, etc.)
- utils: Utility functions
"""

import asyncio
import itertools
import logging
import os
import sys
import textwrap
from collections import OrderedDict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from soak.models import QualitativeAnalysis, QualitativeAnalysisComparison
from soak.models.base import get_embedding, memory

# import from submodules
from .rescaling import RescaleMethod, rescale_similarity
from .optimal_transport import (
    _hash_array,
    _hash_array_list,
    _compute_ot_cached,
    compute_ot,
    compute_split_join_stats,
    filter_transport_plan,
    compute_best_matches_for_k,
    hungarian_matching,
)
from .baselines import generate_word_salad_texts, compute_permutation_baseline
from .paraphrasing import (
    generate_paraphrase_texts,
    generate_short_labels,
    prepare_paraphrase_cost_matrix,
    compute_paraphrase_ot_at_k,
    compute_paraphrase_baseline,
)
from .visualizations import (
    SankeyHTML,
    Base64ImageFile,
    create_transport_sankey,
    create_transport_heatmap,
    find_elbow_points,
    _cost_to_color,
)
from .utils import create_embeddings_csv_base64, format_similarity_matrix

logger = logging.getLogger(__name__)

# re-export for backward compatibility
__all__ = [
    # Types
    "RescaleMethod",
    # Rescaling
    "rescale_similarity",
    # Optimal Transport
    "compute_ot",
    "compute_split_join_stats",
    "filter_transport_plan",
    "compute_best_matches_for_k",
    "hungarian_matching",
    # Baselines
    "generate_word_salad_texts",
    "compute_permutation_baseline",
    # Paraphrasing
    "generate_paraphrase_texts",
    "generate_short_labels",
    "prepare_paraphrase_cost_matrix",
    "compute_paraphrase_ot_at_k",
    "compute_paraphrase_baseline",
    # Visualizations
    "SankeyHTML",
    "Base64ImageFile",
    "create_transport_sankey",
    "create_transport_heatmap",
    "find_elbow_points",
    # Utils
    "create_embeddings_csv_base64",
    "format_similarity_matrix",
    # Main functions
    "create_shared_mass_scree_plot",
    "create_alignment_scree_plot",
    "create_splits_joins_scree_plot",
    "compare_result_similarity",
    "network_similarity_plot",
    "create_pairwise_heatmap",
    # Main class
    "SimilarityComparator",
]
def create_shared_mass_scree_plot(
    ot_by_k: Dict[float, Dict],
    k_values: List[float],
    analysis_name_a: str = "A",
    analysis_name_b: str = "B",
    elbow_k_values: List[float] = None,
) -> Dict[str, Any]:
    """Create scree plot showing shared mass across different K values.

    Shows both maximum curvature elbow and diminishing returns points for K selection.
    Baseline curves (paraphrase ceiling and word-salad floor) are extracted from
    ot_by_k for each K value and plotted as reference envelopes.

    Args:
        ot_by_k: Dictionary mapping K values to OT results
        k_values: List of K values to display in plot
        analysis_name_a: Name of analysis A
        analysis_name_b: Name of analysis B
        elbow_k_values: K values for elbow detection (may include anchor points not displayed)

    Returns:
        Dictionary with:
        - image: Base64ImageFile containing the scree plot
        - chord_k: K value at maximum curvature elbow
        - diminishing_k: K value at diminishing returns point
        - plateau_reached: Whether the curve has clearly asymptoted
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # use elbow_k_values if provided, otherwise use k_values
    if elbow_k_values is None:
        elbow_k_values = k_values

    # extract shared mass for display K values
    shared_masses = [ot_by_k[k]["ot"]["shared_mass"] * 100 for k in k_values]

    # extract shared mass for elbow detection (may include anchor K)
    elbow_shared_masses = [ot_by_k[k]["ot"]["shared_mass"] * 100 for k in elbow_k_values]

    # extract baseline values for each K (if available)
    paraphrase_ceilings = []
    word_salad_floors = []
    permutation_baselines = []
    permutation_95ci_lo = []
    permutation_95ci_hi = []
    for k in k_values:
        ot_data = ot_by_k[k]["ot"]
        ceiling = ot_data.get("paraphrase_upper_bound")
        floor = ot_data.get("null_shared_mass_mean")
        perm_mean = ot_data.get("perm_shared_mass_mean")
        perm_95ci = ot_data.get("perm_shared_mass_95ci")
        paraphrase_ceilings.append(ceiling * 100 if ceiling is not None else None)
        word_salad_floors.append(floor * 100 if floor is not None else None)
        permutation_baselines.append(perm_mean * 100 if perm_mean is not None else None)
        if perm_95ci is not None:
            permutation_95ci_lo.append(perm_95ci[0] * 100)
            permutation_95ci_hi.append(perm_95ci[1] * 100)
        else:
            permutation_95ci_lo.append(None)
            permutation_95ci_hi.append(None)

    # check if we have baseline data
    has_ceiling = any(c is not None for c in paraphrase_ceilings)
    has_floor = any(f is not None for f in word_salad_floors)
    has_permutation = any(p is not None for p in permutation_baselines)

    # find both elbow points using full K range (including anchor)
    elbow_points = find_elbow_points(elbow_k_values, elbow_shared_masses)
    chord_k = elbow_points["chord_k"]
    diminishing_k = elbow_points["diminishing_k"]
    plateau_reached = elbow_points["plateau_reached"]

    # get shared mass values at elbow points (from elbow_k_values data)
    chord_shared = elbow_shared_masses[elbow_points["chord_idx"]]
    diminishing_shared = elbow_shared_masses[elbow_points["diminishing_idx"]]

    # find indices in display k_values (for annotation skipping)
    chord_idx = k_values.index(chord_k) if chord_k in k_values else None
    diminishing_idx = k_values.index(diminishing_k) if diminishing_k in k_values else None

    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # plot shared mass curve (display k_values only)
    ax.plot(k_values, shared_masses, "o-", color="#27ae60", linewidth=2, markersize=6, label="Observed")

    # add spline fit overlay for sanity checking (same as used for curvature detection)
    from scipy.interpolate import UnivariateSpline

    min_k_for_elbow = 0.1
    max_k_for_elbow = 2.0
    elbow_k_arr = np.array(elbow_k_values)
    elbow_mask = (elbow_k_arr >= min_k_for_elbow) & (elbow_k_arr <= max_k_for_elbow)
    K_elbow_fit = elbow_k_arr[elbow_mask]
    s_elbow_fit = np.array(elbow_shared_masses)[elbow_mask]
    if len(K_elbow_fit) >= 4:
        K_fit_uniform = np.linspace(K_elbow_fit.min(), K_elbow_fit.max(), 100)
        s_fit_uniform = np.interp(K_fit_uniform, K_elbow_fit, s_elbow_fit)
        x_norm = (K_fit_uniform - K_fit_uniform.min()) / (K_fit_uniform.max() - K_fit_uniform.min())
        y_norm = (s_fit_uniform - s_fit_uniform.min()) / (s_fit_uniform.max() - s_fit_uniform.min() + 1e-12)
        try:
            spline = UnivariateSpline(x_norm, y_norm, s=0.01, k=4)
            y_spline = spline(x_norm)
            # convert back to original scale
            s_spline = y_spline * (s_fit_uniform.max() - s_fit_uniform.min()) + s_fit_uniform.min()
            ax.plot(K_fit_uniform, s_spline, "--", color="#888888", linewidth=1.5, alpha=0.6,
                    label=f"Spline fit ({len(spline.get_knots())} knots)")
        except Exception:
            pass  # skip spline overlay if fitting fails

    ax.set_xlabel("K (Mass Penalty)", fontsize=11)
    ax.set_ylabel("Shared Mass (%)", fontsize=11)
    title = f"Shared Mass vs K\n{analysis_name_a} ↔ {analysis_name_b}"
    if not plateau_reached:
        title += " (curve may not have plateaued)"
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 0.7)  # fixed scale for comparability across analyses
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)

    # highlight maximum curvature elbow (filled purple diamond, lower layer)
    ax.scatter(
        [chord_k],
        [chord_shared],
        color="#9b59b6",
        s=150,
        zorder=5,
        marker="D",
        edgecolors="white",
        linewidths=2,
        label=f"Max curvature (K={chord_k})",
    )
    ax.axvline(x=chord_k, color="#9b59b6", linestyle=":", alpha=0.7)

    # highlight diminishing returns point (open orange triangle, on top)
    # always show, even when overlapping - open marker makes overlap visible
    ax.scatter(
        [diminishing_k],
        [diminishing_shared],
        facecolors="none",
        s=200,
        zorder=6,
        marker="^",
        edgecolors="#e67e22",
        linewidths=3,
        label=f"Dim. returns (K={diminishing_k})",
    )
    if diminishing_k != chord_k:
        ax.axvline(x=diminishing_k, color="#e67e22", linestyle=":", alpha=0.7)

    # add baseline reference curves (varying with K)
    if has_ceiling:
        valid_k_ceiling = [k for k, c in zip(k_values, paraphrase_ceilings) if c is not None]
        valid_ceiling = [c for c in paraphrase_ceilings if c is not None]
        if valid_ceiling:
            ax.plot(
                valid_k_ceiling,
                valid_ceiling,
                "--",
                color="#2ecc71",
                linewidth=2,
                alpha=0.8,
                label="Paraphrase ceiling",
            )
            ax.fill_between(valid_k_ceiling, valid_ceiling, 100, alpha=0.1, color="#999999")
    if has_floor:
        valid_k_floor = [k for k, f in zip(k_values, word_salad_floors) if f is not None]
        valid_floor = [f for f in word_salad_floors if f is not None]
        if valid_floor:
            ax.plot(
                valid_k_floor,
                valid_floor,
                "--",
                color="#e74c3c",
                linewidth=2,
                alpha=0.8,
                label="Word-salad floor",
            )
            ax.fill_between(valid_k_floor, 0, valid_floor, alpha=0.1, color="#999999")

    # permutation baseline (alignment identifiability diagnostic)
    if has_permutation:
        valid_k_perm = [k for k, p in zip(k_values, permutation_baselines) if p is not None]
        valid_perm = [p for p in permutation_baselines if p is not None]
        valid_lo = [lo for lo, p in zip(permutation_95ci_lo, permutation_baselines) if p is not None]
        valid_hi = [hi for hi, p in zip(permutation_95ci_hi, permutation_baselines) if p is not None]
        if valid_perm:
            ax.plot(
                valid_k_perm,
                valid_perm,
                "-.",
                color="#8e44ad",
                linewidth=2,
                alpha=0.8,
                label="Permutation baseline",
            )
            # add 95% CI shading
            if valid_lo and valid_hi and all(x is not None for x in valid_lo + valid_hi):
                ax.fill_between(valid_k_perm, valid_lo, valid_hi, alpha=0.15, color="#8e44ad")

    ax.legend(loc="lower right", fontsize=9)

    # annotate selected points to avoid clutter
    skip_indices = set()
    if chord_idx is not None:
        skip_indices.add(chord_idx)
    if diminishing_idx is not None:
        skip_indices.add(diminishing_idx)
    for i, (k, sm) in enumerate(zip(k_values, shared_masses)):
        if i in skip_indices:
            continue
        if i == 0 or i == len(k_values) - 1 or i % 4 == 0:
            ax.annotate(
                f"{sm:.1f}%",
                (k, sm),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8,
            )

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)

    return {
        "image": Base64ImageFile(buffer, name="shared_mass_scree.png"),
        "chord_k": chord_k,
        "chord_idx": chord_idx,
        "diminishing_k": diminishing_k,
        "diminishing_idx": diminishing_idx,
        "plateau_reached": plateau_reached,
    }


def create_alignment_scree_plot(
    ot_by_k: Dict[float, Dict],
    k_values: List[float],
    analysis_name_a: str = "A",
    analysis_name_b: str = "B",
) -> Dict[str, Any]:
    """Create scree plot showing alignment (1 - cost) across different K values.

    Baseline curves (paraphrase ceiling and word-salad floor) are extracted from
    ot_by_k for each K value and plotted as reference envelopes.

    Args:
        ot_by_k: Dictionary mapping K values to OT results
        k_values: List of K values used
        analysis_name_a: Name of analysis A
        analysis_name_b: Name of analysis B

    Returns:
        Dictionary with:
        - image: Base64ImageFile containing the alignment scree plot
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # extract alignment (1 - cost) for each K
    alignments = [(1 - ot_by_k[k]["ot"]["avg_cost"]) for k in k_values]

    # extract baseline alignment values for each K (if available)
    paraphrase_ceilings = []
    word_salad_floors = []
    for k in k_values:
        ot_data = ot_by_k[k]["ot"]
        ceiling = ot_data.get("alignment_paraphrase_ceiling")
        floor = ot_data.get("alignment_null_floor")
        paraphrase_ceilings.append(ceiling if ceiling is not None else None)
        word_salad_floors.append(floor if floor is not None else None)

    has_ceiling = any(c is not None for c in paraphrase_ceilings)
    has_floor = any(f is not None for f in word_salad_floors)

    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # plot alignment curve
    ax.plot(k_values, alignments, "o-", color="#3498db", linewidth=2, markersize=6)
    ax.fill_between(k_values, alignments, alpha=0.2, color="#3498db")

    ax.set_xlabel("K (Mass Penalty)", fontsize=11)
    ax.set_ylabel("Alignment (1 - cost)", fontsize=11)
    ax.set_title(
        f"Semantic Alignment vs K\n{analysis_name_a} ↔ {analysis_name_b}", fontsize=12
    )
    ax.set_xlim(0, 0.7)  # fixed scale for comparability across analyses
    # set y-axis to floor - 2% to ceiling + 2% (max 100) for better visual scaling
    y_floor = min([f for f in word_salad_floors if f is not None]) if has_floor else min(alignments)
    y_ceiling = max([c for c in paraphrase_ceilings if c is not None]) if has_ceiling else max(alignments)
    y_min = max(0, y_floor - 0.02)
    y_max = min(1, y_ceiling + 0.02)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    # add baseline reference curves (varying with K)
    if has_ceiling:
        valid_k_ceiling = [k for k, c in zip(k_values, paraphrase_ceilings) if c is not None]
        valid_ceiling = [c for c in paraphrase_ceilings if c is not None]
        if valid_ceiling:
            ax.plot(
                valid_k_ceiling,
                valid_ceiling,
                "--",
                color="#2ecc71",
                linewidth=2,
                alpha=0.8,
                label="Paraphrase ceiling",
            )
            ax.fill_between(valid_k_ceiling, valid_ceiling, 1.0, alpha=0.1, color="#2ecc71")
    if has_floor:
        valid_k_floor = [k for k, f in zip(k_values, word_salad_floors) if f is not None]
        valid_floor = [f for f in word_salad_floors if f is not None]
        if valid_floor:
            ax.plot(
                valid_k_floor,
                valid_floor,
                "--",
                color="#e74c3c",
                linewidth=2,
                alpha=0.8,
                label="Word-salad floor",
            )
            ax.fill_between(valid_k_floor, 0, valid_floor, alpha=0.1, color="#e74c3c")

    ax.legend(loc="lower right", fontsize=9)

    # annotate some points
    for i, (k, align) in enumerate(zip(k_values, alignments)):
        if i == 0 or i == len(k_values) - 1 or i % 4 == 0:
            ax.annotate(
                f"{align:.2f}",
                (k, align),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8,
            )

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)

    return {
        "image": Base64ImageFile(buffer, name="alignment_scree.png"),
    }


def create_splits_joins_scree_plot(
    ot_by_k: Dict[float, Dict],
    k_values: List[float],
    analysis_name_a: str = "A",
    analysis_name_b: str = "B",
) -> Dict[str, Any]:
    """Create scree plot showing average splits/joins across different K values.

    Args:
        ot_by_k: Dictionary mapping K values to OT results (must include split_join_stats)
        k_values: List of K values used
        analysis_name_a: Name of analysis A
        analysis_name_b: Name of analysis B

    Returns:
        Dictionary with:
        - image: Base64ImageFile containing the splits/joins scree plot
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # extract average splits/joins for each K (average of splits_from_a and joins_to_b means)
    splits_joins = []
    for k in k_values:
        stats = ot_by_k[k].get("split_join_stats", {})
        splits_mean = stats.get("splits_from_a", {}).get("mean", 1.0)
        joins_mean = stats.get("joins_to_b", {}).get("mean", 1.0)
        splits_joins.append((splits_mean + joins_mean) / 2)

    # extract baseline splits/joins for each K (if available)
    paraphrase_ceilings = []
    word_salad_floors = []
    for k in k_values:
        ot_data = ot_by_k[k].get("ot", ot_by_k[k])
        ceiling = ot_data.get("paraphrase_splits_joins_mean")
        floor = ot_data.get("null_splits_joins_mean")
        paraphrase_ceilings.append(ceiling if ceiling is not None else None)
        word_salad_floors.append(floor if floor is not None else None)

    has_ceiling = any(c is not None for c in paraphrase_ceilings)
    has_floor = any(f is not None for f in word_salad_floors)

    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # plot splits/joins curve
    ax.plot(k_values, splits_joins, "o-", color="#9b59b6", linewidth=2, markersize=6)

    ax.set_xlabel("K (Mass Penalty)", fontsize=11)
    ax.set_ylabel("Avg Splits/Joins per Theme", fontsize=11)
    ax.set_title(
        f"Splits/Joins vs K\n{analysis_name_a} ↔ {analysis_name_b}", fontsize=12
    )
    ax.set_xlim(0, 0.7)  # fixed scale for comparability across analyses

    # set y-axis with some padding
    y_min_data = min(splits_joins)
    y_max_data = max(splits_joins)
    if has_floor:
        floor_vals = [f for f in word_salad_floors if f is not None]
        if floor_vals:
            y_max_data = max(y_max_data, max(floor_vals))
    if has_ceiling:
        ceiling_vals = [c for c in paraphrase_ceilings if c is not None]
        if ceiling_vals:
            y_min_data = min(y_min_data, min(ceiling_vals))

    y_range = y_max_data - y_min_data
    ax.set_ylim(max(0.9, y_min_data - y_range * 0.1), y_max_data + y_range * 0.1)
    ax.grid(True, alpha=0.3)

    # add baseline reference curves (varying with K) if available
    if has_ceiling:
        valid_k_ceiling = [k for k, c in zip(k_values, paraphrase_ceilings) if c is not None]
        valid_ceiling = [c for c in paraphrase_ceilings if c is not None]
        if valid_ceiling:
            ax.plot(
                valid_k_ceiling,
                valid_ceiling,
                "--",
                color="#2ecc71",
                linewidth=2,
                alpha=0.8,
                label="Paraphrase ceiling",
            )
            ax.fill_between(valid_k_ceiling, 0.9, valid_ceiling, alpha=0.1, color="#999999")
    if has_floor:
        valid_k_floor = [k for k, f in zip(k_values, word_salad_floors) if f is not None]
        valid_floor = [f for f in word_salad_floors if f is not None]
        if valid_floor:
            ax.plot(
                valid_k_floor,
                valid_floor,
                "--",
                color="#e74c3c",
                linewidth=2,
                alpha=0.8,
                label="Word-salad floor",
            )
            # shade above floor (worse = more splits/joins for random)
            ax.fill_between(valid_k_floor, valid_floor, y_max_data + y_range * 0.1, alpha=0.1, color="#999999")

    # add reference line at 1.0 (perfect 1:1 matching)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect 1:1")

    ax.legend(loc="upper right", fontsize=9)

    # annotate some points
    for i, (k, sj) in enumerate(zip(k_values, splits_joins)):
        if i == 0 or i == len(k_values) - 1 or i % 4 == 0:
            ax.annotate(
                f"{sj:.2f}",
                (k, sj),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8,
            )

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)

    return {
        "image": Base64ImageFile(buffer, name="splits_joins_scree.png"),
    }


def compare_result_similarity(
    A: QualitativeAnalysis,
    B: QualitativeAnalysis,
    threshold: float = 0.6,
    embedding_template: str = "{name}: {description}",
    embedding_model: str = "text-embedding-3-large",
    k: float = 1.0,
    reg_m: float = 0.2,
    n_null_samples: int = 100,
    distance: str = "angular",
    compute_paraphrase_bound: bool = True,
    n_paraphrases: int = 7,
    paraphrase_model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    # rescaling parameters
    rescale_method: RescaleMethod = "clip",
    rescale_min: Optional[float] = 0.5,
    rescale_max: Optional[float] = 0.9,
    rescale_lower_percentile: float = 5.0,
    rescale_upper_percentile: float = 95.0,
    rescale_sigmoid_center: float = 0.7,
    rescale_sigmoid_steepness: float = 10.0,
    rescale_temperature: float = 0.5,
    # transport plan filtering
    filter_threshold: float = 0.05,
    # color scale (K-relative)
    color_green: float = 0.2,
    color_red: float = 1.1,
    short_labels_a: Optional[List[str]] = None,
    short_labels_b: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare two sets of theme embeddings.

    Allows many-to-one matches: each theme may match multiple from the other set.

    Null Baseline Method (Symmetric Word-Salad):
        To test whether observed alignment exceeds chance, we construct a symmetric
        null baseline by scrambling both theme sets:
        1. A vs word-salad-B: Tests if B's themes have real semantic content
        2. word-salad-A vs B: Tests if A's themes have real semantic content

        Word-salad is generated by pooling all words from a theme set, shuffling them
        randomly, and chunking back into strings matching the original length
        distribution. This destroys semantic coherence while preserving vocabulary
        and length properties.

        The null distribution averages both directions (N/2 samples each), giving
        a robust, symmetric baseline that doesn't favour either analysis.

    Args:
        A: First QualitativeAnalysis to compare
        B: Second QualitativeAnalysis to compare
        threshold: Similarity threshold for matching (default: 0.6)
        embedding_template: Python format string for generating embeddings from themes.
                          Available fields: {name}, {description}
                          Default: "{name}"
        k: Shepard similarity decay parameter (default: 1.0)
        reg_m: OT mass penalty K (default: 0.2). Fixed value for cross-analysis
               comparability. Lower = more selective matching.
        n_null_samples: Number of word-salad samples for null baseline (default: 100).
                       Split evenly between both directions (50 each by default).
        distance: Distance metric to use (default: "angular"). Options:
                 - "angular": Angular similarity (1 - arccos(cos)/pi). Preferred as it
                   satisfies the triangle inequality and avoids high-similarity compression.
                 - "cosine": Raw cosine similarity. Not a proper metric.
                 - "shepard": Shepard similarity with exponential decay controlled by k.
        compute_paraphrase_bound: Generate LLM paraphrases for realistic upper bound (default: True)
        n_paraphrases: Number of paraphrases per theme (default: 7)
        paraphrase_model: LLM model for paraphrase generation (default: gpt-4.1-mini)
        api_key: API key for LLM (uses LLM_API_KEY env var if not provided)
        base_url: API base URL (uses LLM_API_BASE env var if not provided)

        Rescaling parameters (control how similarity matrix is transformed before OT):
        rescale_method: Method for rescaling similarity matrix. Options:
            - "off": No rescaling
            - "clip": Hard clip to [rescale_min, rescale_max] then stretch to [0, 1]
            - "adaptive": Use percentile-based bounds from the data
            - "sigmoid": Soft clipping with sigmoid (preserves tails)
            - "temperature": Power transformation (preserves ordering)
            - "rank": Convert to percentile ranks (most robust)
        rescale_min: Floor for clip method (default: 0.5)
        rescale_max: Ceiling for clip method (default: 0.9)
        rescale_lower_percentile: Lower percentile for adaptive method (default: 5.0)
        rescale_upper_percentile: Upper percentile for adaptive method (default: 95.0)
        rescale_sigmoid_center: Center for sigmoid method (default: 0.7)
        rescale_sigmoid_steepness: Steepness for sigmoid method (default: 10.0)
        rescale_temperature: Temperature for power method (default: 0.5, < 1 sharpens)

    Returns:
        Dictionary with similarity metrics including:
        Coverage metrics (hit rates):
        - hit_rate_a: % of A themes with at least one B match above threshold
        - hit_rate_b: % of B themes with at least one A match above threshold
        Fidelity metrics (mean best-match similarity):
        - mean_max_sim_a_to_b: average of each A theme's best match similarity in B
        - mean_max_sim_b_to_a: average of each B theme's best match similarity in A
        - fidelity: harmonic mean of the two directional fidelity scores
        Other metrics:
        - jaccard: proportion of theme pairs with similarity > threshold
        - match_matrix: binary matrix [n_A x n_B], 1 = similarity above threshold
        - similarity_matrix: raw cosine similarity values
        - angle_similarity_matrix: angular distance normalized to [0,1]
        - shepard_similarity_matrix: Shepard similarity with specified k
    """

    # extract theme names/labels and analysis names before reassigning A and B
    # use label if set (from --llm-labels), otherwise fall back to name
    theme_names_A = [theme.label if theme.label else theme.name for theme in A.themes]
    theme_names_B = [theme.label if theme.label else theme.name for theme in B.themes]
    analysis_name_A = A.name
    analysis_name_B = B.name

    # use short labels for plots if provided (these already include set prefix like "A1: Label")
    # otherwise fall back to truncated theme names with index
    if short_labels_a:
        plot_labels_A = list(short_labels_a)
    else:
        plot_labels_A = [f"{i+1}. {name[:20]}" for i, name in enumerate(theme_names_A)]
    if short_labels_b:
        plot_labels_B = list(short_labels_b)
    else:
        plot_labels_B = [f"{i+1}. {name[:20]}" for i, name in enumerate(theme_names_B)]

    A_texts = [
        embedding_template.format(name=i.name, description=i.description)
        for i in A.themes
    ]
    B_texts = [
        embedding_template.format(name=i.name, description=i.description)
        for i in B.themes
    ]

    # compute embedding length metadata (word count as proxy for tokens)
    def count_words(text: str) -> int:
        return len(text.split())

    mean_embedding_len_A = (
        sum(count_words(t) for t in A_texts) / len(A_texts) if A_texts else 0
    )
    mean_embedding_len_B = (
        sum(count_words(t) for t in B_texts) / len(B_texts) if B_texts else 0
    )

    # keep A and B as the text lists for backward compatibility
    A = A_texts
    B = B_texts

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    logger.debug("Getting embeddings for A and B")
    emb_A = get_embedding(
        list(map(lambda x: x.strip(), A)),
        model=embedding_model,
    )
    emb_B = get_embedding(
        list(map(lambda x: x.strip(), B)),
        model=embedding_model,
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
            "hit_rate_a": 0.0,
            "hit_rate_b": 0.0,
            "jaccard": 0.0,
            "match_matrix": np.zeros((n_A, n_B), dtype=int),
            "similarity_matrix": np.zeros((n_A, n_B)),
            "mean_max_sim_a_to_b": 0.0,
            "mean_max_sim_b_to_a": 0.0,
            "fidelity": 0.0,
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
        S = (np.exp(-k_val * theta_inner) - np.exp(-k_val * np.pi)) / (
            1 - np.exp(-k_val * np.pi)
        )
        # drop diagonal and duplicates
        n = S.shape[0]
        iu = np.triu_indices(n, k=1)
        return S[iu]

    # === HUNGARIAN MATCHING (optimal 1-to-1 assignment) ===
    # Select similarity matrix based on distance metric
    distance_matrices = {
        "angular": angle_sim,
        "cosine": sim_matrix,
        "shepard": shepard_sim,
    }
    selected_sim = distance_matrices.get(distance, angle_sim)

    # helper to apply rescaling with current parameters
    def apply_rescaling(sim):
        return rescale_similarity(
            sim,
            method=rescale_method,
            rescale_min=rescale_min,
            rescale_max=rescale_max,
            lower_percentile=rescale_lower_percentile,
            upper_percentile=rescale_upper_percentile,
            sigmoid_center=rescale_sigmoid_center,
            sigmoid_steepness=rescale_sigmoid_steepness,
            temperature=rescale_temperature,
        )

    # Apply rescaling if enabled
    selected_sim_raw = selected_sim.copy()  # keep original for reference
    if rescale_method != "off":
        selected_sim = apply_rescaling(selected_sim)
        logger.info(f"Rescaled similarity using method='{rescale_method}'")
        logger.info(f"  Original similarity: mean={selected_sim_raw.mean():.3f}, range=[{selected_sim_raw.min():.3f}, {selected_sim_raw.max():.3f}]")
        logger.info(f"  Rescaled similarity: mean={selected_sim.mean():.3f}, range=[{selected_sim.min():.3f}, {selected_sim.max():.3f}]")
        if rescale_method == "clip":
            logger.info(f"  Clip bounds: [{rescale_min}, {rescale_max}]")
        elif rescale_method == "adaptive":
            logger.info(f"  Percentile bounds: [{rescale_lower_percentile}%, {rescale_upper_percentile}%]")
        elif rescale_method == "sigmoid":
            logger.info(f"  Sigmoid: center={rescale_sigmoid_center}, steepness={rescale_sigmoid_steepness}")
        elif rescale_method == "temperature":
            logger.info(f"  Temperature: {rescale_temperature}")

    # Prepare rescaled similarity matrices for BOTH angular and shepard (for OT toggle)
    angle_sim_raw = angle_sim.copy()
    shepard_sim_raw = shepard_sim.copy()
    if rescale_method != "off":
        angle_sim_rescaled = apply_rescaling(angle_sim)
        shepard_sim_rescaled = apply_rescaling(shepard_sim)
    else:
        angle_sim_rescaled = angle_sim.copy()
        shepard_sim_rescaled = shepard_sim.copy()

    # Cost matrices for both metrics (used in OT computation)
    cost_matrix_angular = 1 - angle_sim_rescaled
    cost_matrix_shepard = 1 - shepard_sim_rescaled

    hungarian_results = hungarian_matching(selected_sim, threshold=threshold)

    # log Hungarian results
    logger.info(f"\n=== Hungarian Matching (1-to-1, {distance} similarity) ===")
    logger.info(
        f"Optimal assignment: {hungarian_results['distribution']['n_pairs']}/{min(len(emb_A), len(emb_B))} pairs above threshold"
    )
    logger.info(
        f"Coverage: {hungarian_results['thresholded_metrics']['coverage_a']:.1%} of A, {hungarian_results['thresholded_metrics']['coverage_b']:.1%} of B"
    )
    logger.info(
        f"True Jaccard: {hungarian_results['thresholded_metrics']['true_jaccard']:.3f}"
    )
    logger.info(
        f"Fidelity -- Mean assignment similarity: {hungarian_results['soft_metrics']['soft_precision']:.3f}, Normalised total: {hungarian_results['soft_metrics']['soft_recall']:.3f}"
    )

    dist = hungarian_results["distribution"]
    if dist["n_pairs"] > 0:
        logger.info(
            f"Similarity distribution: median={dist['median']:.3f} (Q1={dist['q1']:.3f}, Q3={dist['q3']:.3f}, range: {dist['min']:.3f}-{dist['max']:.3f})"
        )

    # === UNBALANCED OPTIMAL TRANSPORT (many-to-many alignment) ===
    # use selected distance metric for cost matrix (for backwards compatibility)
    cost_matrix = 1 - selected_sim  # rescaled similarity used for OT and coloring
    cost_matrix_original = 1 - selected_sim_raw  # original similarity for hover display
    logger.info("\n=== Computing Unbalanced Optimal Transport Metrics ===")
    logger.info(f"Cost matrix (rescaled): mean={cost_matrix.mean():.3f}")
    logger.info(f"Cost matrix (original): mean={cost_matrix_original.mean():.3f}")

    # === SYMMETRIC WORD-SALAD NULL BASELINE ===
    # To avoid asymmetry, we scramble BOTH sets and average the null distributions:
    #   1. A vs word-salad-B: tests if B's themes have real semantic content
    #   2. word-salad-A vs B: tests if A's themes have real semantic content
    # Averaging both directions gives a robust, symmetric null baseline.
    #
    # OPTIMIZATION: Null baseline is only computed for the default K value.
    # For other K values, we skip null comparison since we only need absolute
    # metrics (shared_mass, unmatched_mass) for the scree plot comparison.

    from tqdm import tqdm

    n_samples_per_direction = n_null_samples // 2  # split samples between directions

    logger.info(
        f"Generating symmetric word-salad null ({n_samples_per_direction} samples each direction)..."
    )

    # Direction 1: A vs word-salad-B
    salad_B_list = generate_word_salad_texts(B_texts, n_samples=n_samples_per_direction)
    # Direction 2: word-salad-A vs B
    salad_A_list = generate_word_salad_texts(A_texts, n_samples=n_samples_per_direction)

    # batch all word salad texts into a single embedding call for performance
    all_salad_texts = [
        t.strip() for salad_texts in salad_B_list + salad_A_list for t in salad_texts
    ]

    logger.info(f"Embedding {len(all_salad_texts)} word salad texts in single batch...")
    all_salad_embeddings = np.asarray(
        get_embedding(all_salad_texts, model=embedding_model)
    )

    # reshape and split: (n_samples, n_themes, embedding_dim)
    n_B_total = n_samples_per_direction * len(B_texts)
    emb_B_salads = all_salad_embeddings[:n_B_total].reshape(
        n_samples_per_direction, len(B_texts), -1
    )
    emb_A_salads = all_salad_embeddings[n_B_total:].reshape(
        n_samples_per_direction, len(A_texts), -1
    )

    # helper to compute similarity using the selected distance metric (with rescaling)
    def compute_similarity(emb_a, emb_b, metric, k_val, do_rescale=True):
        cos_sim = cosine_similarity(emb_a, emb_b)
        if metric == "cosine":
            sim = cos_sim
        elif metric == "angular":
            angle_mat = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
            sim = 1 - angle_mat / 180.0
        elif metric == "shepard":
            theta = np.arccos(np.clip(cos_sim, -1.0, 1.0))
            sim = (np.exp(-k_val * theta) - np.exp(-k_val * np.pi)) / (
                1 - np.exp(-k_val * np.pi)
            )
        else:
            sim = cos_sim
        # apply rescaling if enabled
        if do_rescale and rescale_method != "off":
            sim = apply_rescaling(sim)
        return sim

    null_cost_matrices_B = [
        1 - compute_similarity(emb_A, emb, distance, k) for emb in emb_B_salads
    ]
    null_cost_matrices_A = [
        1 - compute_similarity(emb, emb_B, distance, k) for emb in emb_A_salads
    ]

    # Combine both directions for symmetric null
    null_cost_matrices = null_cost_matrices_B + null_cost_matrices_A

    # store word salad samples for display (showing B direction as example)
    word_salad_samples = salad_B_list

    logger.info(
        f"Generated {len(null_cost_matrices)} null cost matrices ({len(null_cost_matrices_B)} A vs B_salad + {len(null_cost_matrices_A)} A_salad vs B)"
    )

    # === WORD-SALAD SELF-SIMILARITY ===
    # Estimate the similarity between independent word-salad samples.
    # This gives a baseline for "what does random similarity look like?"
    # We compare different word-salad samples (different scramblings) to each other.
    word_salad_self_similarity = None
    word_salad_self_similarity_raw = None
    if n_samples_per_direction >= 2:
        salad_self_sims = []
        salad_self_sims_raw = []
        # compare pairs of word-salad samples (different scramblings)
        # use both A and B salad sets for robustness
        for emb_salads in [emb_B_salads, emb_A_salads]:
            n_samples = emb_salads.shape[0]
            for i in range(min(n_samples, 3)):  # cap at 3 pairs per set for efficiency
                for j in range(i + 1, min(n_samples, 3)):
                    # compare salad[i] vs salad[j] -- two different scramblings
                    self_sim_rescaled = compute_similarity(
                        emb_salads[i], emb_salads[j], distance, k, do_rescale=True
                    )
                    self_sim_raw = compute_similarity(
                        emb_salads[i], emb_salads[j], distance, k, do_rescale=False
                    )
                    # mean off-diagonal similarity (diagonal would be same theme scrambled differently)
                    salad_self_sims.append(float(np.mean(self_sim_rescaled)))
                    salad_self_sims_raw.append(float(np.mean(self_sim_raw)))

        if salad_self_sims:
            word_salad_self_similarity = float(np.mean(salad_self_sims))
            word_salad_self_similarity_raw = float(np.mean(salad_self_sims_raw))
            logger.info(
                f"Word-salad self-similarity: {word_salad_self_similarity_raw:.3f} (rescaled: {word_salad_self_similarity:.3f})"
            )

    # === PARAPHRASE UPPER BOUND BASELINE ===
    # Generate LLM paraphrases to establish a realistic upper bound for relative metrics.
    # This represents what similarity we'd expect if two analyses captured identical
    # concepts but expressed them with different wording.
    paraphrase_baseline = None
    paraphrase_upper_bound = None
    paraphrase_cost_lower_bound = None
    paraphrase_cost_matrix_A = None
    paraphrase_cost_matrix_B = None

    if compute_paraphrase_bound:
        import asyncio

        logger.info(f"Generating paraphrase upper bound baseline ({n_paraphrases} paraphrases per theme)...")

        try:
            # generate paraphrases for both sets (symmetric like word-salad)
            paraphrases_A, meta_A = asyncio.run(
                generate_paraphrase_texts(
                    A_texts,
                    n_paraphrases=n_paraphrases,
                    model_name=paraphrase_model,
                    api_key=api_key,
                    base_url=base_url,
                )
            )
            paraphrases_B, meta_B = asyncio.run(
                generate_paraphrase_texts(
                    B_texts,
                    n_paraphrases=n_paraphrases,
                    model_name=paraphrase_model,
                    api_key=api_key,
                    base_url=base_url,
                )
            )

            # compute paraphrase cost matrices (embed once, use for all K values)
            baseline_A = compute_paraphrase_baseline(
                A_texts, emb_A, paraphrases_A,
                embedding_model=embedding_model,
                distance=distance,
                shepard_k=k,
                reg_m=0.3,  # initial K for logging, will recompute per-K
                rescale_method=rescale_method,
                rescale_min=rescale_min,
                rescale_max=rescale_max,
                rescale_lower_percentile=rescale_lower_percentile,
                rescale_upper_percentile=rescale_upper_percentile,
                rescale_sigmoid_center=rescale_sigmoid_center,
                rescale_sigmoid_steepness=rescale_sigmoid_steepness,
                rescale_temperature=rescale_temperature,
            )
            baseline_B = compute_paraphrase_baseline(
                B_texts, emb_B, paraphrases_B,
                embedding_model=embedding_model,
                distance=distance,
                shepard_k=k,
                reg_m=0.3,
                rescale_method=rescale_method,
                rescale_min=rescale_min,
                rescale_max=rescale_max,
                rescale_lower_percentile=rescale_lower_percentile,
                rescale_upper_percentile=rescale_upper_percentile,
                rescale_sigmoid_center=rescale_sigmoid_center,
                rescale_sigmoid_steepness=rescale_sigmoid_steepness,
                rescale_temperature=rescale_temperature,
            )

            # store cost matrices for K-specific baseline computation
            paraphrase_cost_matrix_A = baseline_A.get("cost_matrix")
            paraphrase_cost_matrix_B = baseline_B.get("cost_matrix")

            # store baseline info for display (samples, metadata)
            # OT metrics will be computed per-K in the loop
            paraphrase_baseline = {
                "paraphrase_similarity_mean": None,  # computed per-K
                "paraphrase_similarity_std": (
                    baseline_A["paraphrase_similarity_std"] +
                    baseline_B["paraphrase_similarity_std"]
                ) / 2,
                "paraphrase_cost_mean": None,  # computed per-K
                "samples_a": baseline_A["samples"],
                "samples_b": baseline_B["samples"],
                "metadata": {
                    "model": paraphrase_model or "gpt-4.1-mini",
                    "n_paraphrases": n_paraphrases,
                },
            }

            logger.info("Paraphrase cost matrices prepared for K-specific OT computation")

        except Exception as e:
            logger.warning(f"Paraphrase baseline generation failed: {e}")
            logger.warning("Continuing without paraphrase upper bound")

    # === PERMUTATION BASELINE (alignment identifiability test) ===
    # Tests whether the similarity geometry encodes pair-specific structure.
    # If permuted shared mass ~ true shared mass, alignments are underdetermined.
    # This is a necessary condition for meaningful OT interpretation.
    logger.info("\n=== Computing permutation baseline (alignment identifiability) ===")

    # K values for permutation test (subset for efficiency)
    PERM_K_VALUES = [0.1, 0.2, 0.3, 0.5, 1.0]
    permutation_baseline = compute_permutation_baseline(
        cost_matrix,
        k_values=PERM_K_VALUES,
        n_permutations=50,
        mode="unbalanced",
        reg=0.01,
        seed=42,
    )
    logger.info(f"Permutation baseline computed for K values: {PERM_K_VALUES}")

    # === COMPUTE OT FOR MULTIPLE K VALUES ===
    # K (reg_m) controls when themes are left unmatched vs forced to align
    # Higher K = stronger penalty for unmatching = more mass forced to transport
    # Lower K = weaker penalty = more mass can remain unmatched
    # K values: fine gradations in critical range 0.05-0.5, coarser above
    # 0.05-0.5: every 0.025 | 0.5-1.0: every 0.1 | above 1.0: coarser
    K_VALUES = [
        # Fine gradations in critical range (0.025 steps)
        0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275,
        0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5,
        # Coarser above 0.5 (0.1 steps)
        0.6, 0.7, 0.8, 0.9, 1.0,
        # Even coarser for high K
        1.5, 2.0, 3.0, 4.0,
    ]
    EXTENDED_K_VALUES = [6.0, 8.0, 10.0]
    PLATEAU_ANCHOR_K = 5.0  # always compute for reliable plateau detection, but hide from plots
    MIN_K_BEFORE_STOP = 0.7  # minimum K to compute before allowing early stopping

    # PHASE 1: Compute OT for all K values in parallel (up to MIN_K_BEFORE_STOP)
    # Then continue sequentially with early stopping for higher K values
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def compute_ot_for_k(k_val):
        """Compute OT and split/join stats for a single K value."""
        ot_result = compute_ot(
            cost_matrix,
            null_cost_matrices=null_cost_matrices,
            mode="unbalanced",
            reg_m=k_val,
        )
        split_join_stats = compute_split_join_stats(ot_result["transport_plan"])
        return k_val, ot_result, split_join_stats

    ot_by_k = {}
    computed_k_values = []

    # split K values into parallel batch (up to MIN_K_BEFORE_STOP) and sequential batch
    parallel_k_values = [k for k in K_VALUES if k <= MIN_K_BEFORE_STOP]
    sequential_k_values = [k for k in K_VALUES if k > MIN_K_BEFORE_STOP]

    # compute parallel batch using thread pool
    n_cores = os.cpu_count() or 4
    n_workers = min(n_cores, len(parallel_k_values))
    logger.info(f"Computing OT for {len(parallel_k_values)} K values in parallel ({n_workers} workers)...")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_ot_for_k, k): k for k in parallel_k_values}

        # collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing OT (parallel)", file=sys.stderr):
            k_val, ot_result, split_join_stats = future.result()
            ot_by_k[k_val] = {
                "ot_result": ot_result,
                "split_join_stats": split_join_stats,
            }
            logger.debug(
                f"K={k_val:.3f}: shared_mass={ot_result['shared_mass']:.1%}, avg_cost={ot_result['avg_cost']:.2f}"
            )

    # sort computed K values
    computed_k_values = sorted(ot_by_k.keys())

    # continue sequentially for higher K values with early stopping
    if sequential_k_values:
        prev_shared_mass = ot_by_k[computed_k_values[-1]]["ot_result"]["shared_mass"]

        for k_val in tqdm(sequential_k_values, desc="Computing OT (sequential)", file=sys.stderr):
            k_val, ot_result, split_join_stats = compute_ot_for_k(k_val)

            ot_by_k[k_val] = {
                "ot_result": ot_result,
                "split_join_stats": split_join_stats,
            }
            computed_k_values.append(k_val)

            logger.debug(
                f"K={k_val:.3f}: shared_mass={ot_result['shared_mass']:.1%}, avg_cost={ot_result['avg_cost']:.2f}"
            )

            # stop early if improvement < 2.5% (curve has plateaued)
            current_shared_mass = ot_result["shared_mass"]
            improvement = (current_shared_mass - prev_shared_mass) / (
                prev_shared_mass + 1e-9
            )
            prev_shared_mass = current_shared_mass
            if improvement < 0.025:
                logger.info(
                    f"Shared mass improvement {improvement:.1%} < 2.5% at K={k_val}, stopping early"
                )
                break

    computed_k_values = sorted(ot_by_k.keys())

    # adaptive extension: if last improvement was still >= 2.5%, add extra K values
    all_k_values = list(computed_k_values)
    last_shared_mass = ot_by_k[computed_k_values[-1]]["ot_result"]["shared_mass"]
    if len(computed_k_values) >= 2:
        second_last_shared = ot_by_k[computed_k_values[-2]]["ot_result"]["shared_mass"]
        last_improvement = (last_shared_mass - second_last_shared) / (
            second_last_shared + 1e-9
        )
    else:
        last_improvement = 1.0  # assume we need extension if only 1 K value
    if last_improvement >= 0.025:
        logger.info(
            f"Last improvement ({last_improvement:.1%}) >= 2.5%, extending K values"
        )
        for k_val in tqdm(
            EXTENDED_K_VALUES, desc="Computing OT for extended K", file=sys.stderr
        ):
            logger.debug(f"\n--- Computing OT with K={k_val} (extended) ---")

            ot_result = compute_ot(
                cost_matrix,
                null_cost_matrices=null_cost_matrices,
                mode="unbalanced",
                reg_m=k_val,
            )

            split_join_stats = compute_split_join_stats(ot_result["transport_plan"])

            ot_by_k[k_val] = {
                "ot_result": ot_result,
                "split_join_stats": split_join_stats,
            }
            all_k_values.append(k_val)

            logger.debug(
                f"K={k_val:.1f}: shared_mass={ot_result['shared_mass']:.3f}, avg_cost={ot_result['avg_cost']:.3f}"
            )

            # stop early if improvement < 2.5%
            current_shared_mass = ot_result["shared_mass"]
            improvement = (current_shared_mass - prev_shared_mass) / (
                prev_shared_mass + 1e-9
            )
            prev_shared_mass = current_shared_mass
            if improvement < 0.025:
                logger.info(
                    f"Shared mass improvement {improvement:.1%} < 2.5% at K={k_val}, stopping extension"
                )
                break

    # always compute plateau anchor K for reliable elbow detection (but don't show in plots)
    if PLATEAU_ANCHOR_K not in ot_by_k:
        logger.debug(f"\n--- Computing OT with K={PLATEAU_ANCHOR_K} (plateau anchor) ---")
        ot_result = compute_ot(
            cost_matrix,
            null_cost_matrices=null_cost_matrices,
            mode="unbalanced",
            reg_m=PLATEAU_ANCHOR_K,
        )
        split_join_stats = compute_split_join_stats(ot_result["transport_plan"])
        ot_by_k[PLATEAU_ANCHOR_K] = {
            "ot_result": ot_result,
            "ot": ot_result,  # also store as 'ot' for scree plot compatibility
            "split_join_stats": split_join_stats,
        }
        logger.debug(
            f"K={PLATEAU_ANCHOR_K:.1f}: shared_mass={ot_result['shared_mass']:.3f} (anchor for plateau detection)"
        )

    # k values for elbow detection includes anchor; visualization k values excludes it
    elbow_k_values = sorted(ot_by_k.keys())

    # PHASE 2: Set color scale from baselines (data-driven, smooth gradient)
    # Color anchors:
    #   - Red anchor: word-salad self-similarity + 20% (random noise baseline with margin)
    #   - Green anchor: paraphrase self-similarity - 20% (paraphrase ceiling with margin)
    # Smooth gradient: red → amber → green across the full range

    # Use word-salad SELF-similarity (similarity between different word-salad samples)
    # This represents true random baseline -- what similarity looks like for noise
    if word_salad_self_similarity is not None:
        base_floor = word_salad_self_similarity
        logger.info(f"Word-salad self-similarity (rescaled): {base_floor:.3f}")
    elif null_cost_matrices:
        # fallback: use mean of null cost matrices (A vs word-salad-B)
        null_similarities = [1.0 - np.mean(m) for m in null_cost_matrices]
        base_floor = float(np.mean(null_similarities))
        logger.info(f"Word-salad baseline similarity (fallback): {base_floor:.3f}")
    else:
        base_floor = 0.3  # fallback if no null baseline
        logger.info(f"No word-salad baseline, using default floor: {base_floor:.3f}")

    # Use paraphrase self-similarity (theme to its paraphrases)
    if paraphrase_cost_matrix_A is not None and paraphrase_cost_matrix_B is not None:
        # Diagonal of cost matrix = cost of theme to its best paraphrase
        # Similarity = 1 - cost
        diag_sim_A = 1.0 - np.diag(paraphrase_cost_matrix_A)
        diag_sim_B = 1.0 - np.diag(paraphrase_cost_matrix_B)
        base_ceiling = float((np.mean(diag_sim_A) + np.mean(diag_sim_B)) / 2)
        logger.info(f"Paraphrase self-similarity (rescaled): {base_ceiling:.3f}")
    else:
        base_ceiling = 0.9  # fallback if no paraphrase baseline
        logger.info(f"No paraphrase baseline, using default ceiling: {base_ceiling:.3f}")

    # Apply 20% margins to create the color anchors
    # Red anchor: word-salad + 20% of range (give benefit of doubt to low values)
    # Green anchor: paraphrase - 20% of range (reserve green for truly good matches)
    baseline_range = base_ceiling - base_floor
    margin = baseline_range * 0.20
    color_red_anchor = base_floor + margin      # similarity below this → red
    color_green_anchor = base_ceiling - margin  # similarity above this → green

    # Store for output documentation (K-relative color scale)
    color_scale_info = {
        "word_salad_self_similarity": base_floor,
        "paraphrase_self_similarity": base_ceiling,
        "scaling_method": "k_relative",
        "color_green_mult": color_green,
        "color_red_mult": color_red,
        "description": f"K-relative colour scale: green = cost < {color_green}*K, red = cost > {color_red}*K. "
                       "Red links are close to being dropped at the current K threshold."
    }

    logger.info(f"Color scale info (K-relative: green < {color_green}*K, red > {color_red}*K):")
    logger.info(f"  Word-salad baseline: {base_floor:.3f}")
    logger.info(f"  Paraphrase baseline: {base_ceiling:.3f}")

    # Create visualisations for all K values with K-relative color scale
    # Green = cost 0 (perfect match), Red = cost approaching K (marginal, about to be dropped)
    for k_val in tqdm(all_k_values, desc="Creating visualisations", file=sys.stderr):
        ot_result = ot_by_k[k_val]["ot_result"]
        split_join_stats = ot_by_k[k_val]["split_join_stats"]

        # Apply transport plan filtering to reduce weak edges (splits/joins)
        # This affects visualizations only; shared_mass is computed from unfiltered plan
        transport_plan_raw = ot_result["transport_plan"]
        filtered_plan, filter_stats = filter_transport_plan(transport_plan_raw, threshold=filter_threshold)

        # Compute split/join stats on filtered plan for display
        filtered_split_join_stats = compute_split_join_stats(filtered_plan)

        # K-relative color scale: green at cost<green*K, red at cost>red*K
        # Green = strong match, Red = marginal (approaching threshold)
        color_cost_min = k_val * color_green  # green anchor
        color_cost_max = k_val * color_red  # red anchor

        transport_sankey_k = create_transport_sankey(
            filtered_plan,  # use filtered plan for visualization
            plot_labels_A,
            plot_labels_B,
            cost_matrix=cost_matrix,  # rescaled, for coloring
            cost_matrix_original=cost_matrix_original,  # original, for hover display
            analysis_name_a=analysis_name_A,
            analysis_name_b=analysis_name_B,
            cost_min=color_cost_min,
            cost_max=color_cost_max,
        )
        transport_heatmap_k = create_transport_heatmap(
            filtered_plan,  # use filtered plan for visualization
            plot_labels_A,
            plot_labels_B,
            analysis_name_a=analysis_name_A,
            analysis_name_b=analysis_name_B,
        )

        # prepare OT results for serialisation (remove numpy array)
        ot_serialisable_k = {
            key: v for key, v in ot_result.items() if key != "transport_plan"
        }
        ot_serialisable_k["transport_plan"] = np.round(
            ot_result["transport_plan"], 4
        ).tolist()
        # split/join stats from unfiltered plan (original OT result)
        ot_serialisable_k["split_join_stats"] = split_join_stats

        # filtering stats (for interpretability)
        ot_serialisable_k["filter_stats"] = filter_stats
        ot_serialisable_k["filtered_transport_plan"] = np.round(filtered_plan, 4).tolist()
        ot_serialisable_k["filtered_split_join_stats"] = filtered_split_join_stats

        # compute K-specific paraphrase baselines
        paraphrase_upper_bound_k = None
        paraphrase_cost_lower_bound_k = None
        if paraphrase_cost_matrix_A is not None and paraphrase_cost_matrix_B is not None:
            # run OT on paraphrase cost matrices at this K
            para_ot_A = compute_paraphrase_ot_at_k(paraphrase_cost_matrix_A, reg_m=k_val)
            para_ot_B = compute_paraphrase_ot_at_k(paraphrase_cost_matrix_B, reg_m=k_val)
            # average for symmetric baseline
            paraphrase_upper_bound_k = (para_ot_A["shared_mass"] + para_ot_B["shared_mass"]) / 2
            paraphrase_cost_lower_bound_k = (para_ot_A["avg_cost"] + para_ot_B["avg_cost"]) / 2

        # add paraphrase-scaled metrics if paraphrase baseline available
        if paraphrase_upper_bound_k is not None:
            null_mean = ot_serialisable_k.get("null_shared_mass_mean", 0.0)
            shared_mass = ot_serialisable_k.get("shared_mass", 0.0)

            # shared_mass_pct_of_ceiling: observed / paraphrase (absolute % of best case)
            if paraphrase_upper_bound_k > 0:
                ot_serialisable_k["shared_mass_pct_of_ceiling"] = float(
                    shared_mass / paraphrase_upper_bound_k
                )
            else:
                ot_serialisable_k["shared_mass_pct_of_ceiling"] = 0.0

            # shared_mass_improvement_vs_null: (observed - null) / (paraphrase - null)
            # how much of the possible improvement from word-salad to paraphrase was achieved
            if paraphrase_upper_bound_k > null_mean:
                ot_serialisable_k["shared_mass_improvement_vs_null"] = float(
                    (shared_mass - null_mean) / (paraphrase_upper_bound_k - null_mean)
                )
            else:
                ot_serialisable_k["shared_mass_improvement_vs_null"] = 0.0

            # keep old name for backward compatibility
            ot_serialisable_k["shared_mass_relative_paraphrase"] = ot_serialisable_k["shared_mass_improvement_vs_null"]

            # convert cost to alignment (1 - cost) so higher is always better
            # this makes interpretation consistent with shared mass
            if paraphrase_cost_lower_bound_k is not None:
                null_cost_mean = ot_serialisable_k.get("null_avg_cost_mean", 1.0)
                avg_cost = ot_serialisable_k.get("avg_cost", 1.0)

                # convert to alignment (similarity) - higher is better
                observed_alignment = 1.0 - avg_cost
                paraphrase_alignment = 1.0 - paraphrase_cost_lower_bound_k  # ceiling (best)
                null_alignment = 1.0 - null_cost_mean  # floor (worst)

                ot_serialisable_k["alignment_observed"] = observed_alignment
                ot_serialisable_k["alignment_paraphrase_ceiling"] = paraphrase_alignment
                ot_serialisable_k["alignment_null_floor"] = null_alignment

                # alignment_pct_of_ceiling: observed / paraphrase (absolute % of best case)
                if paraphrase_alignment > 0:
                    ot_serialisable_k["alignment_pct_of_ceiling"] = float(
                        observed_alignment / paraphrase_alignment
                    )
                else:
                    ot_serialisable_k["alignment_pct_of_ceiling"] = 0.0

                # alignment_improvement_vs_null: (observed - null) / (paraphrase - null)
                # how much of the possible improvement from word-salad to paraphrase was achieved
                if paraphrase_alignment > null_alignment:
                    ot_serialisable_k["alignment_improvement_vs_null"] = float(
                        (observed_alignment - null_alignment) / (paraphrase_alignment - null_alignment)
                    )
                else:
                    ot_serialisable_k["alignment_improvement_vs_null"] = 0.0

                # alignment_effect: effect size in MADs (same as avg_cost_effect since alignment = 1 - cost)
                ot_serialisable_k["alignment_effect"] = ot_serialisable_k.get("avg_cost_effect", 0.0)

                # keep old names for backward compatibility
                ot_serialisable_k["avg_cost_pct_of_floor"] = ot_serialisable_k["alignment_pct_of_ceiling"]
                ot_serialisable_k["avg_cost_improvement_vs_null"] = ot_serialisable_k["alignment_improvement_vs_null"]
                ot_serialisable_k["avg_cost_relative_paraphrase"] = ot_serialisable_k["alignment_improvement_vs_null"]

            # store the K-specific bounds for display
            ot_serialisable_k["paraphrase_upper_bound"] = paraphrase_upper_bound_k
            ot_serialisable_k["paraphrase_cost_lower_bound"] = paraphrase_cost_lower_bound_k

        # === PERMUTATION BASELINE METRICS ===
        # Add permutation-based identifiability metrics for this K value
        shared_mass = ot_serialisable_k.get("shared_mass", 0.0)
        if k_val in permutation_baseline:
            perm_data = permutation_baseline[k_val]
            perm_mean = perm_data["perm_shared_mass_mean"]
            perm_std = perm_data["perm_shared_mass_std"]
            perm_95ci = perm_data["perm_shared_mass_95ci"]

            ot_serialisable_k["perm_shared_mass_mean"] = perm_mean
            ot_serialisable_k["perm_shared_mass_std"] = perm_std
            ot_serialisable_k["perm_shared_mass_95ci"] = perm_95ci
            ot_serialisable_k["perm_shared_mass_distribution"] = perm_data["perm_shared_mass_distribution"]

            # alignment_gap: how much better is true alignment than random permutation
            ot_serialisable_k["alignment_gap"] = float(shared_mass - perm_mean)

            # identifiability_ratio: true / permuted (should be >> 1 for meaningful alignment)
            if perm_mean > 0:
                ot_serialisable_k["identifiability_ratio"] = float(shared_mass / perm_mean)
            else:
                ot_serialisable_k["identifiability_ratio"] = float("inf") if shared_mass > 0 else 1.0

            # identifiability_pvalue: fraction of permutations >= true shared mass
            perm_dist = np.array(perm_data["perm_shared_mass_distribution"])
            ot_serialisable_k["identifiability_pvalue"] = float((perm_dist >= shared_mass).mean())

            # === SPLITS/JOINS IDENTIFIABILITY ===
            # Even if shared mass is similar, permuted alignments should have MORE splits/joins
            # (fragmented transport) if the true alignment has meaningful structure
            true_splits_joins = (
                split_join_stats.get("splits_from_a", {}).get("mean", 1.0) +
                split_join_stats.get("joins_to_b", {}).get("mean", 1.0)
            ) / 2

            perm_sj_mean = perm_data.get("perm_splits_joins_mean", true_splits_joins)
            perm_sj_std = perm_data.get("perm_splits_joins_std", 0.0)
            perm_sj_95ci = perm_data.get("perm_splits_joins_95ci", [perm_sj_mean, perm_sj_mean])

            ot_serialisable_k["true_splits_joins"] = float(true_splits_joins)
            ot_serialisable_k["perm_splits_joins_mean"] = float(perm_sj_mean)
            ot_serialisable_k["perm_splits_joins_std"] = float(perm_sj_std)
            ot_serialisable_k["perm_splits_joins_95ci"] = perm_sj_95ci

            # splits_joins_gap: permuted - true (should be positive: permuted has more fragmentation)
            ot_serialisable_k["splits_joins_gap"] = float(perm_sj_mean - true_splits_joins)

            # splits_joins_ratio: permuted / true (should be > 1 if true alignment is cleaner)
            if true_splits_joins > 0:
                ot_serialisable_k["splits_joins_ratio"] = float(perm_sj_mean / true_splits_joins)
            else:
                ot_serialisable_k["splits_joins_ratio"] = 1.0

            # p-value: fraction of permutations with FEWER splits/joins than true
            # (lower is better for true alignment, so we want few permutations to beat us)
            perm_sj_dist = np.array(perm_data.get("perm_splits_joins_distribution", [perm_sj_mean]))
            ot_serialisable_k["splits_joins_pvalue"] = float((perm_sj_dist <= true_splits_joins).mean())

            # identifiability_warning: flag if alignment is not significantly better than permuted
            # Consider BOTH mass ratio AND splits/joins ratio
            # warn if: mass ratio < 1.2 AND splits/joins ratio < 1.2
            # (either metric being good is sufficient)
            mass_ratio = ot_serialisable_k["identifiability_ratio"]
            sj_ratio = ot_serialisable_k["splits_joins_ratio"]
            mass_pval = ot_serialisable_k["identifiability_pvalue"]
            sj_pval = ot_serialisable_k["splits_joins_pvalue"]

            # warning if BOTH metrics fail (neither shows pair-specific structure)
            mass_ok = (mass_ratio >= 1.1) or (mass_pval < 0.1)
            sj_ok = (sj_ratio >= 1.1) or (sj_pval < 0.1)
            ot_serialisable_k["identifiability_warning"] = not (mass_ok or sj_ok)

        # compute K-specific best matches
        best_matches_a_to_b_k, best_matches_b_to_a_k = compute_best_matches_for_k(
            ot_result["transport_plan"],
            selected_sim,
            cost_matrix,
            len(emb_A),
            len(emb_B),
        )

        ot_by_k[k_val] = {
            "ot": ot_serialisable_k,
            "transport_sankey": transport_sankey_k,
            "transport_heatmap": transport_heatmap_k,
            "split_join_stats": split_join_stats,
            "best_matches_a_to_b": best_matches_a_to_b_k,
            "best_matches_b_to_a": best_matches_b_to_a_k,
        }

    # compute elbow points first to determine chord_k for reference results
    elbow_shared_masses = [ot_by_k[k]["ot"]["shared_mass"] * 100 for k in elbow_k_values]
    elbow_points = find_elbow_points(elbow_k_values, elbow_shared_masses)
    chord_k = elbow_points["chord_k"]
    diminishing_k = elbow_points["diminishing_k"]
    plateau_reached = elbow_points["plateau_reached"]

    logger.info(f"\n=== Elbow Detection ===")
    logger.info(f"Max curvature elbow: K={chord_k} (maximum curvature point)")
    logger.info(f"Diminishing returns: K={diminishing_k} (slope < 20% of initial)")
    if not plateau_reached:
        logger.warning(
            "Curve may not have plateaued -- elbow estimates may be less reliable"
        )

    # use chord_k for reference results (the automatically selected K)
    ot_results = ot_by_k[chord_k]["ot"]
    ot_results["transport_plan"] = np.array(
        ot_results["transport_plan"]
    )  # convert back for later use
    transport_sankey = ot_by_k[chord_k]["transport_sankey"]
    transport_heatmap = ot_by_k[chord_k]["transport_heatmap"]

    # populate paraphrase_baseline with chord_k-specific values
    if paraphrase_baseline is not None:
        paraphrase_baseline["paraphrase_similarity_mean"] = ot_results.get("paraphrase_upper_bound")
        paraphrase_baseline["paraphrase_cost_mean"] = ot_results.get("paraphrase_cost_lower_bound")

    # filter displayed K values: show up to chord_k + 0.5, max 1.5
    k_display_max = min(chord_k + 0.5, 1.5)
    display_k_values = [k for k in all_k_values if k <= k_display_max]
    logger.info(f"Displaying K values up to {k_display_max:.2f} (chord={chord_k} + 0.5, max 1.5)")

    # generate scree plots (baselines extracted from ot_by_k for each K)
    # use elbow_k_values (includes anchor) for elbow detection, display_k_values for display
    scree_result = create_shared_mass_scree_plot(
        ot_by_k,
        display_k_values,
        analysis_name_a=analysis_name_A,
        analysis_name_b=analysis_name_B,
        elbow_k_values=elbow_k_values,
    )
    shared_mass_scree = scree_result["image"]

    alignment_scree_result = create_alignment_scree_plot(
        ot_by_k,
        display_k_values,
        analysis_name_a=analysis_name_A,
        analysis_name_b=analysis_name_B,
    )
    alignment_scree = alignment_scree_result["image"]

    splits_joins_scree_result = create_splits_joins_scree_plot(
        ot_by_k,
        display_k_values,
        analysis_name_a=analysis_name_A,
        analysis_name_b=analysis_name_B,
    )
    splits_joins_scree = splits_joins_scree_result["image"]

    # log chord_k results
    logger.info(f"\n=== Chord K={chord_k} Results ===")
    logger.info(
        f"Shared Mass: {ot_results['shared_mass']:.1%} (proportion of mass transported)"
    )
    logger.info(
        f"Unmatched Mass: {ot_results['unmatched_mass']:.1%} (novel/missing themes)"
    )
    logger.info(
        f"Average Cost: {ot_results['avg_cost']:.2f} (lower = better alignment)"
    )
    logger.info(
        f"Regularisation: reg={ot_results['reg']:.4f}, reg_m (K)={ot_results['reg_m']:.4f}"
    )

    # null comparison with interpretable relative metrics
    if "null_shared_mass_mean" in ot_results:
        logger.info(f"--- Null baseline comparison ---")
        logger.info(
            f"Null shared_mass: mean={ot_results['null_shared_mass_mean']:.1%}, 95pct={ot_results['null_shared_mass_95pct']:.1%}"
        )
        logger.info(
            f"Shared mass excess: +{ot_results['shared_mass_excess']:.1%} (raw improvement over null)"
        )
        logger.info(
            f"Shared mass relative: {ot_results['shared_mass_relative']:.1%} (0=random, 1=perfect)"
        )
        logger.info(
            f"Shared mass effect: {ot_results['shared_mass_effect']:.2f} MADs above null"
        )
        logger.info(f"--- Cost metrics ---")
        logger.info(
            f"Null avg_cost: mean={ot_results['null_avg_cost_mean']:.2f}, 5pct={ot_results['null_avg_cost_5pct']:.2f}"
        )
        logger.info(
            f"Avg cost improvement: {ot_results['avg_cost_improvement']:.2f} (positive = better than null)"
        )
        logger.info(
            f"Avg cost relative: {ot_results['avg_cost_relative']:.2f} (0=random, 1=perfect)"
        )

    # permutation baseline diagnostic
    if "perm_shared_mass_mean" in ot_results:
        logger.info(f"--- Permutation baseline (alignment identifiability) ---")
        logger.info(
            f"Permutation shared_mass: mean={ot_results['perm_shared_mass_mean']:.1%}, 95% CI=[{ot_results['perm_shared_mass_95ci'][0]:.1%}, {ot_results['perm_shared_mass_95ci'][1]:.1%}]"
        )
        logger.info(
            f"Alignment gap: {ot_results['alignment_gap']:.1%} (observed - permuted)"
        )
        logger.info(
            f"Mass identifiability ratio: {ot_results['identifiability_ratio']:.2f}x (should be >> 1.0)"
        )
        logger.info(
            f"Mass identifiability p-value: {ot_results['identifiability_pvalue']:.3f}"
        )

        # splits/joins identifiability (complementary metric)
        if "perm_splits_joins_mean" in ot_results:
            logger.info(
                f"True splits/joins: {ot_results['true_splits_joins']:.2f}, Permuted: {ot_results['perm_splits_joins_mean']:.2f}"
            )
            logger.info(
                f"Splits/joins ratio: {ot_results['splits_joins_ratio']:.2f}x (permuted/true, should be > 1.0)"
            )
            logger.info(
                f"Splits/joins gap: +{ot_results['splits_joins_gap']:.2f} (permuted has more fragmentation)"
            )

        if ot_results.get("identifiability_warning"):
            logger.warning(
                "⚠ Low alignment identifiability: alignment may not be pair-specific (neither mass nor splits/joins shows significant difference from permuted)"
            )

    # filtering stats
    if "filter_stats" in ot_results:
        fs = ot_results["filter_stats"]
        if fs.get("filtering_enabled"):
            logger.info(
                f"Transport filtered at {fs['threshold']:.0%} threshold: "
                f"{fs['mass_retained_pct']:.1f}% mass retained, "
                f"{fs['edges_filtered']}/{fs['edges_original']} edges ({fs['edges_retained_pct']:.0f}% retained) "
                f"vs {fs['edges_original']}/{fs['edges_max_possible']} ({fs['edges_original_pct']:.0f}%) unfiltered"
            )
            # filtered split/join stats
            fss = ot_results.get("filtered_split_join_stats", {})
            splits_pct = fss.get("splits_from_a", {}).get("pct_multiple", 0) * 100
            joins_pct = fss.get("joins_to_b", {}).get("pct_multiple", 0) * 100
            logger.info(
                f"Filtered splits: {splits_pct:.0f}% of rows have >1 connection"
            )
            logger.info(
                f"Filtered joins: {joins_pct:.0f}% of cols have >1 connection"
            )

    # log all matrices (show legend only once at the start)
    logger.info("\n=== Theme Index Legend ===")
    logger.info(f"\n{analysis_name_A} Themes (rows):")
    for i, name in enumerate(theme_names_A):
        logger.info(f"  {i}: {name}")
    logger.info(f"\n{analysis_name_B} Themes (columns):")
    for i, name in enumerate(theme_names_B):
        logger.info(f"  {i}: {name}")

    logger.info(
        "\n=== Cosine Similarity ===\n"
        + format_similarity_matrix(
            sim_matrix,
            theme_names_A,
            theme_names_B,
            set_a_name=analysis_name_A,
            set_b_name=analysis_name_B,
            show_legend=False,
        )
    )

    logger.info(
        "\n=== Angular Similarity (normalized) ===\n"
        + format_similarity_matrix(
            angle_sim,
            theme_names_A,
            theme_names_B,
            set_a_name=analysis_name_A,
            set_b_name=analysis_name_B,
            show_legend=False,
        )
    )

    logger.info(
        f"\n=== Shepard Similarity (k={k}) ===\n"
        + format_similarity_matrix(
            shepard_sim,
            theme_names_A,
            theme_names_B,
            set_a_name=analysis_name_A,
            set_b_name=analysis_name_B,
            show_legend=False,
        )
    )

    # === COVERAGE AND FIDELITY use selected_sim for consistency ===
    # All metrics now use the same similarity metric (angular by default)
    match_matrix = selected_sim >= threshold

    # === COVERAGE METRICS (hit rates) ===
    # Hit Rate A: % of A themes with at least one match in B above threshold
    hit_rate_a_count = match_matrix.any(axis=1).sum()
    hit_rate_a = hit_rate_a_count / len(emb_A) if len(emb_A) > 0 else 0

    # Hit Rate B: % of B themes with at least one match in A above threshold
    hit_rate_b_count = match_matrix.any(axis=0).sum()
    hit_rate_b = hit_rate_b_count / len(emb_B) if len(emb_B) > 0 else 0

    # Jaccard: intersection / union across all pairwise theme comparisons
    intersection = match_matrix.sum()
    union = match_matrix.size  # total possible pairs = n_A * n_B
    jaccard = intersection / union if union > 0 else 0

    # === FIDELITY METRICS (mean best-match similarity) ===
    # Mean max similarity A→B: for each A theme, find best match in B, then average
    mean_max_sim_a_to_b = (
        selected_sim.max(axis=1).mean().round(3) if len(emb_A) > 0 else 0
    )

    # Mean max similarity B→A: for each B theme, find best match in A, then average
    mean_max_sim_b_to_a = (
        selected_sim.max(axis=0).mean().round(3) if len(emb_B) > 0 else 0
    )

    # Fidelity: harmonic mean of the two directional fidelity scores
    fidelity = (
        2
        * (mean_max_sim_a_to_b * mean_max_sim_b_to_a)
        / (mean_max_sim_a_to_b + mean_max_sim_b_to_a)
        if (mean_max_sim_a_to_b + mean_max_sim_b_to_a) > 0
        else 0
    )

    # For each theme in A, find the best matching theme in B
    # Include OT statistics: mass transferred and cost
    P = ot_results["transport_plan"]

    def round_match(
        d,
        decimals={
            "similarity": 3,
            "mass_transferred": 4,
            "mass_total": 4,
            "mass_pct": 1,
            "cost": 3,
        },
    ):
        """Round numeric values in a match dict."""
        return {
            k: round(v, decimals.get(k, 3)) if isinstance(v, float) else v
            for k, v in d.items()
        }

    best_matches_a_to_b = []
    if len(emb_A) > 0 and len(emb_B) > 0:
        for i in range(len(emb_A)):
            best_b_idx = int(selected_sim[i, :].argmax())
            mass_total_out = P[i, :].sum()
            best_matches_a_to_b.append(
                round_match(
                    {
                        "theme_a_index": i,
                        "theme_b_index": best_b_idx,
                        "similarity": float(selected_sim[i, best_b_idx]),
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

    # For each theme in B, find the best matching theme in A
    best_matches_b_to_a = []
    if len(emb_A) > 0 and len(emb_B) > 0:
        for j in range(len(emb_B)):
            best_a_idx = int(selected_sim[:, j].argmax())
            mass_total_in = P[:, j].sum()
            best_matches_b_to_a.append(
                round_match(
                    {
                        "theme_b_index": j,
                        "theme_a_index": best_a_idx,
                        "similarity": float(selected_sim[best_a_idx, j]),
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

    # log best matches with OT statistics
    logger.info(
        f"\n=== Best Matches (many:many) with OT Statistics (K={chord_k}) ==="
    )
    logger.info(f"\n{analysis_name_A} → {analysis_name_B}:")
    logger.info(
        f"{'Theme A':<30} {'Best Match B':<30} {'Sim':>6} {'Mass':>8} {'%':>6} {'Cov':>6}"
    )
    logger.info("-" * 90)
    for m in best_matches_a_to_b[:10]:  # top 10
        name_a = theme_names_A[m["theme_a_index"]][:28]
        name_b = theme_names_B[m["theme_b_index"]][:28]
        logger.info(
            f"{name_a:<30} {name_b:<30} {m['similarity']:>6.2f} {m['mass_transferred']:>8.4f} {m['mass_pct']:>5.0f}% {m['mass_total']*100:>5.1f}%"
        )
    if len(best_matches_a_to_b) > 10:
        logger.info(f"... and {len(best_matches_a_to_b) - 10} more")

    logger.info(f"\n{analysis_name_B} → {analysis_name_A}:")
    logger.info(
        f"{'Theme B':<30} {'Best Match A':<30} {'Sim':>6} {'Mass':>8} {'%':>6} {'Cov':>6}"
    )
    logger.info("-" * 90)
    for m in best_matches_b_to_a[:10]:  # top 10
        name_b = theme_names_B[m["theme_b_index"]][:28]
        name_a = theme_names_A[m["theme_a_index"]][:28]
        logger.info(
            f"{name_b:<30} {name_a:<30} {m['similarity']:>6.2f} {m['mass_transferred']:>8.4f} {m['mass_pct']:>5.0f}% {m['mass_total']*100:>5.1f}%"
        )
    if len(best_matches_b_to_a) > 10:
        logger.info(f"... and {len(best_matches_b_to_a) - 10} more")

    # prepare OT results for serialisation (remove numpy array)
    ot_serialisable = {
        key: v for key, v in ot_results.items() if key != "transport_plan"
    }
    ot_serialisable["transport_plan"] = np.round(
        ot_results["transport_plan"], 4
    ).tolist()

    return {
        # similarity metric used for coverage, fidelity, OT (for display purposes)
        "similarity_metric": distance,
        # rescaling configuration
        "rescale_method": rescale_method,
        "rescale_min": rescale_min,
        "rescale_max": rescale_max,
        "rescale_lower_percentile": rescale_lower_percentile,
        "rescale_upper_percentile": rescale_upper_percentile,
        "rescale_sigmoid_center": rescale_sigmoid_center,
        "rescale_sigmoid_steepness": rescale_sigmoid_steepness,
        "rescale_temperature": rescale_temperature,
        "selected_similarity_matrix": np.round(selected_sim, 3),
        "selected_similarity_matrix_raw": np.round(selected_sim_raw, 3),
        # coverage metrics (hit rates)
        "hit_rate_a": hit_rate_a,
        "hit_rate_b": hit_rate_b,
        "jaccard": jaccard,
        "match_matrix": match_matrix.astype(int),
        # fidelity metrics (mean best-match similarity)
        "mean_max_sim_a_to_b": mean_max_sim_a_to_b,
        "mean_max_sim_b_to_a": mean_max_sim_b_to_a,
        "fidelity": fidelity,
        # continuous similarity metrics (all three always computed for reference)
        "similarity_matrix": np.round(sim_matrix, 3),
        "angle_similarity_matrix": np.round(angle_sim, 3),
        "shepard_similarity_matrix": np.round(shepard_sim, 3),
        "shepard_k_value": k,
        # Hungarian matching (optimal 1-to-1 assignment using Shepard similarity)
        "hungarian": hungarian_results,
        # Unbalanced Optimal Transport (many-to-many alignment with unmatched mass)
        "ot": ot_serialisable,
        # Transport visualisations (for default K)
        "transport_sankey": transport_sankey,
        "transport_heatmap": transport_heatmap,
        # OT results for displayed K values (for tabbed display)
        "ot_by_k": ot_by_k,
        "k_values": display_k_values,
        "default_k": chord_k,  # default is now chord elbow
        "chord_k": chord_k,
        "diminishing_k": diminishing_k,
        "plateau_reached": plateau_reached,
        # color scale info (anchored to baselines, smooth gradient)
        "color_scale_info": color_scale_info,
        "color_sim_min": round(1 - color_cost_max, 3),  # red anchor (similarity)
        "color_sim_max": round(1 - color_cost_min, 3),  # green anchor (similarity)
        # Scree plots of metrics vs K
        "shared_mass_scree": shared_mass_scree,
        "alignment_scree": alignment_scree,
        "splits_joins_scree": splits_joins_scree,
        # best matches
        "best_matches_a_to_b": best_matches_a_to_b,
        "best_matches_b_to_a": best_matches_b_to_a,
        # embedding metadata (for interpreting effect sizes)
        "mean_embedding_words_a": mean_embedding_len_A,
        "mean_embedding_words_b": mean_embedding_len_B,
        # all word salad samples used in null baseline
        "word_salad_samples": word_salad_samples,
        # word-salad self-similarity: baseline for random matching
        "word_salad_self_similarity": word_salad_self_similarity,  # rescaled
        "word_salad_self_similarity_raw": word_salad_self_similarity_raw,  # original
        # paraphrase baseline for upper bound scaling
        "paraphrase_baseline": paraphrase_baseline,
        # permutation baseline for alignment identifiability
        "permutation_baseline": permutation_baseline,
        # raw embeddings for export (with labels)
        "embeddings_a": {
            "labels": theme_names_A,
            "texts": A_texts,
            "vectors": emb_A.tolist() if hasattr(emb_A, "tolist") else list(emb_A),
        },
        "embeddings_b": {
            "labels": theme_names_B,
            "texts": B_texts,
            "vectors": emb_B.tolist() if hasattr(emb_B, "tolist") else list(emb_B),
        },
    }


@memory.cache
def network_similarity_plot(
    pipeline_results: List[QualitativeAnalysis],
    method="umap",
    n_neighbors=5,
    min_dist=0.01,
    threshold=0.6,
    exclude_within_set_edges=True,
    embedding_template="{name}: {description}",
    embedding_model: str = "text-embedding-3-large",
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
        [theme.label for theme in result.themes] for result in pipeline_results
    ]
    theme_sets_for_labels = [i for i in theme_sets_for_labels_ if i]

    pipeline_names = [i.name for i in pipeline_results]

    # Get embeddings for all sets using embedding_template
    embeddings = [
        get_embedding(set_str, model=embedding_model)
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
        )
        pos_2d = reducer.fit_transform(all_emb)
    elif method == "mds":
        # Classical MDS expects a distance matrix, so convert similarity
        dist_matrix = 1 - sim_matrix
        reducer = MDS(n_components=2, dissimilarity="precomputed")
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
    embedding_template="{name}: {description}",
    embedding_model: str = "text-embedding-3-large",
    metric_type: str = "cosine",
    k: float = 1.0,
    comparison_result: Optional[Dict[str, Any]] = None,
    short_labels_a: Optional[Tuple[str, ...]] = None,
    short_labels_b: Optional[Tuple[str, ...]] = None,
) -> str:
    """Create a heatmap visualization for a single pair of pipeline results.

    Themes are sorted alphanumerically for consistency across runs.

    Args:
        a: First QualitativeAnalysis
        b: Second QualitativeAnalysis
        threshold: Similarity threshold for matching
        use_threshold: Whether to use threshold-based binary heatmap
        embedding_template: Python format string for embeddings. Available: {name}, {description}
        metric_type: Type of similarity metric ("cosine", "angle", "shepard")
        k: Shepard similarity decay parameter (default: 1.0)
        comparison_result: Pre-computed result from compare_result_similarity. If provided,
            skips recomputation (avoids redundant OT calculations).

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

    # use short_labels if provided (these already include set prefix like "A1: Label")
    # otherwise fall back to theme.label with index
    if short_labels_a:
        themes_a = list(short_labels_a)
    else:
        themes_a = [f"{i+1}. {theme.label}" for i, theme in enumerate(a.themes)]
    if short_labels_b:
        themes_b = list(short_labels_b)
    else:
        themes_b = [f"{i+1}. {theme.label}" for i, theme in enumerate(b.themes)]
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

    # use pre-computed comparison if provided, otherwise compute
    if comparison_result is not None:
        comparison = comparison_result
    else:
        comparison = compare_result_similarity(
            a,
            b,
            threshold=threshold or 0.5,  # ensure not None
            embedding_template=embedding_template,
            embedding_model=embedding_model,
            k=k,
        )

    # select matrix based on metric type
    metric_labels = {
        "cosine": ("Cosine Similarity", "similarity_matrix"),
        "angle": ("Angular Similarity", "angle_similarity_matrix"),
        "shepard": (f"Shepard Similarity (k={k})", "shepard_similarity_matrix"),
    }

    metric_label, matrix_key = metric_labels.get(
        metric_type, ("Cosine Similarity", "similarity_matrix")
    )
    similarity_matrix = comparison[matrix_key]

    # sort alphanumerically for consistency across runs
    row_order = np.argsort([t.lower() for t in themes_a_display])
    col_order = np.argsort([t.lower() for t in themes_b_display])
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
    ax.set_title(f"{metric_label}\n{a.name} vs {b.name}{threshold_str}")
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
        embedding_template = config.get("embedding_template", "{name}: {description}")
        embedding_model = config.get("embedding_model", "text-embedding-3-large")
        k = config.get("k", 1.0)
        reg_m = config.get("reg_m", 0.2)
        distance = config.get("distance", "angular")
        # paraphrase baseline parameters
        compute_paraphrase_bound = config.get("compute_paraphrase_bound", True)
        n_paraphrases = config.get("n_paraphrases", 3)
        paraphrase_model = config.get("paraphrase_model", None)
        api_key = config.get("api_key", None)
        base_url = config.get("base_url", None)
        # rescaling parameters
        rescale_method = config.get("rescale_method", "clip")
        rescale_min = config.get("rescale_min", 0.5)
        rescale_max = config.get("rescale_max", 0.9)
        rescale_lower_pct = config.get("rescale_lower_pct", 5.0)
        rescale_upper_pct = config.get("rescale_upper_pct", 95.0)
        rescale_sigmoid_center = config.get("rescale_sigmoid_center", 0.7)
        rescale_sigmoid_steepness = config.get("rescale_sigmoid_steepness", 10.0)
        rescale_temp = config.get("rescale_temp", 0.5)
        # transport plan filtering
        filter_threshold = config.get("filter_threshold", 0.05)
        # color scale (K-relative)
        color_green = config.get("color_green", 0.2)
        color_red = config.get("color_red", 1.1)
        # short label generation for plots
        compute_short_labels = config.get("compute_short_labels", True)
        short_label_model = config.get("short_label_model", None)

        # Set labels on all themes once at the beginning (only if not already set)
        for result in pipeline_results:
            for i, theme in enumerate(result.themes, start=1):
                if not theme.label:
                    theme.set_label(label_template, i)

        result_combinations = list(itertools.combinations(pipeline_results, 2))

        # Create A, B, C... labels for each analysis set (used throughout report)
        set_letters = {result.name: chr(65 + i) for i, result in enumerate(pipeline_results)}

        # Build embedded strings mapping for each result
        embedded_strings_map = {}
        for result in pipeline_results:
            set_letter = set_letters[result.name]
            embedded_strings_map[result.name] = [
                {
                    "theme_name": theme.name,
                    "theme_description": theme.description,
                    "label": theme.label,
                    "embedded_string": embedding_template.format(
                        name=theme.name, description=theme.description
                    ),
                    "short_label": None,  # populated below if compute_short_labels
                    "set_letter": set_letter,
                    "theme_index": str(i + 1),  # 1-based index as string
                }
                for i, theme in enumerate(result.themes)
            ]

        # Generate short labels for all results if enabled
        if compute_short_labels:
            import asyncio

            logger.info("Generating short labels for themes...")
            for result in pipeline_results:
                # use raw theme text without embedding template for short labels
                # this ensures consistent labels across different embedding models
                theme_texts = [
                    f"{t.name}: {t.description}" for t in result.themes
                ]
                try:
                    short_labels = asyncio.run(
                        generate_short_labels(
                            theme_texts,
                            model_name=short_label_model,
                            api_key=api_key,
                            base_url=base_url,
                        )
                    )
                    # add short labels to embedded_strings_map
                    for i, short_label in enumerate(short_labels):
                        if i < len(embedded_strings_map[result.name]):
                            embedded_strings_map[result.name][i]["short_label"] = short_label
                except Exception as e:
                    logger.warning(f"Short label generation failed for {result.name}: {e}")
                    # fallback: use numbered theme identifiers
                    for i, entry in enumerate(embedded_strings_map[result.name]):
                        entry["short_label"] = f"Theme {i + 1}"

        # helper to get short labels for an analysis (with set prefix like "A1: Label")
        def get_short_labels(analysis_name: str) -> Optional[List[str]]:
            if analysis_name in embedded_strings_map:
                entries = embedded_strings_map[analysis_name]
                labels = []
                for e in entries:
                    short = e.get("short_label")
                    if short:
                        # format: "A1: Short Label"
                        labels.append(f"{e['set_letter']}{e['theme_index']}: {short}")
                    else:
                        return None  # not all labels are set
                return labels
            return None

        # run synchronously
        similarity_results = [
            compare_result_similarity(
                i,
                j,
                threshold=threshold,
                embedding_template=embedding_template,
                embedding_model=embedding_model,
                k=k,
                reg_m=reg_m,
                distance=distance,
                compute_paraphrase_bound=compute_paraphrase_bound,
                n_paraphrases=n_paraphrases,
                paraphrase_model=paraphrase_model,
                api_key=api_key,
                base_url=base_url,
                rescale_method=rescale_method,
                rescale_min=rescale_min,
                rescale_max=rescale_max,
                rescale_lower_percentile=rescale_lower_pct,
                rescale_upper_percentile=rescale_upper_pct,
                rescale_sigmoid_center=rescale_sigmoid_center,
                rescale_sigmoid_steepness=rescale_sigmoid_steepness,
                rescale_temperature=rescale_temp,
                filter_threshold=filter_threshold,
                color_green=color_green,
                color_red=color_red,
                short_labels_a=get_short_labels(i.name),
                short_labels_b=get_short_labels(j.name),
            )
            for i, j in result_combinations
        ]

        # generate heatmaps for all metric types, reusing pre-computed similarity results
        metric_types = ["cosine", "angle", "shepard"]

        # helper to get short labels as tuple (for caching)
        def get_short_labels_tuple(analysis_name: str) -> Optional[Tuple[str, ...]]:
            labels = get_short_labels(analysis_name)
            return tuple(labels) if labels else None

        heatmaps_by_metric = {}
        for metric_type in metric_types:
            heatmaps_by_metric[metric_type] = [
                create_pairwise_heatmap(
                    a,
                    b,
                    threshold=threshold,
                    use_threshold=False,
                    embedding_template=embedding_template,
                    embedding_model=embedding_model,
                    metric_type=metric_type,
                    k=k,
                    comparison_result=sim_result,
                    short_labels_a=get_short_labels_tuple(a.name),
                    short_labels_b=get_short_labels_tuple(b.name),
                )
                for (a, b), sim_result in zip(result_combinations, similarity_results)
            ]

        # use the selected distance metric for primary heatmaps
        # map distance names to metric_type names (angular -> angle)
        distance_to_metric = {
            "angular": "angle",
            "cosine": "cosine",
            "shepard": "shepard",
        }
        primary_metric = distance_to_metric.get(distance, "angle")
        heatmaps = heatmaps_by_metric[primary_metric]

        # thresholded heatmaps (only meaningful for cosine similarity)
        thresholded_heatmaps = [
            create_pairwise_heatmap(
                a,
                b,
                threshold=threshold,
                use_threshold=True,
                embedding_template=embedding_template,
                embedding_model=embedding_model,
                metric_type="cosine",
                k=k,
                comparison_result=sim_result,
                short_labels_a=get_short_labels_tuple(a.name),
                short_labels_b=get_short_labels_tuple(b.name),
            )
            for (a, b), sim_result in zip(result_combinations, similarity_results)
        ]

        network_plot = network_similarity_plot(
            [i for i in pipeline_results],
            method=method,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            threshold=threshold,
            embedding_template=embedding_template,
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

        # extract transport plots from similarity results
        transport_sankey_dict = {
            k: v["transport_sankey"]
            for k, v in zip(result_combinations_dict.keys(), similarity_results)
        }
        transport_heatmap_dict = {
            k: v["transport_heatmap"]
            for k, v in zip(result_combinations_dict.keys(), similarity_results)
        }

        # create embeddings CSV for each comparison pair
        embeddings_csv_dict = {}
        for (a, b), sim_result in zip(result_combinations, similarity_results):
            key = f"{a.name}_{b.name}"
            if "embeddings_a" in sim_result and "embeddings_b" in sim_result:
                embeddings_csv_dict[key] = create_embeddings_csv_base64(
                    sim_result["embeddings_a"],
                    sim_result["embeddings_b"],
                    a.name,
                    b.name,
                )

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
                # transport visualisations from unbalanced OT
                "transport_sankey": transport_sankey_dict,
                "transport_heatmap": transport_heatmap_dict,
                # embeddings CSV for download
                "embeddings_csv": embeddings_csv_dict,
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