"""Theme and code similarity comparison using embeddings."""

import base64
import itertools
import logging
import sys
import textwrap
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    import re

    import numpy as np

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
    import numpy as np
    from collections import Counter

    P = np.asarray(transport_plan)
    n_A, n_B = P.shape

    if n_A == 0 or n_B == 0:
        return {
            "splits_from_a": {"counts": {}, "mean": 0.0, "median": 0.0, "mode": 0, "max": 0, "total": 0},
            "joins_to_b": {"counts": {}, "mean": 0.0, "median": 0.0, "mode": 0, "max": 0, "total": 0},
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
            return {"counts": {}, "mean": 0.0, "median": 0.0, "mode": 0, "max": 0, "total": 0}

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
    import numpy as np
    import ot

    n_A, n_B = cost_matrix.shape

    if n_A == 0 or n_B == 0:
        return {
            "shared_mass": 0.0,
            "avg_cost": float("nan"),
            "unmatched_mass": 1.0,
            "transport_plan": np.zeros((0, 0)),
            "coverage_a": [],
            "coverage_b": [],
            "null_shared_mass_mean": 0.0,
            "null_shared_mass_95pct": 0.0,
            "null_avg_cost_mean": 0.0,
            "null_avg_cost_5pct": 0.0,
            "mode": mode,
        }

    # uniform mass distribution
    a = np.ones(n_A) / n_A
    b = np.ones(n_B) / n_B

    # ensure cost matrix is float64 and non-negative
    M = np.asarray(cost_matrix, dtype=np.float64)
    M = np.clip(M, 0, None)

    # ensure minimum regularisation for numerical stability
    reg = max(reg, 1e-6)
    reg_m = max(reg_m, 1e-6)

    def run_ot(cost, a_dist, b_dist, mode_inner):
        """Run OT with given cost matrix and mode."""
        if mode_inner == "balanced":
            P = ot.emd(a_dist, b_dist, cost)
        else:
            P = ot.unbalanced.sinkhorn_unbalanced(
                a_dist, b_dist, cost,
                reg=reg,
                reg_m=reg_m,
                numItermax=1000,
                stopThr=1e-9,
            )
        return P

    # compute optimal transport coupling
    P = run_ot(M, a, b, mode)

    # interpretable quantities
    shared_mass = float(P.sum())
    if shared_mass > 1e-9:
        avg_cost = float((P * M).sum() / shared_mass)
    else:
        avg_cost = float("nan")
    unmatched_mass = 1.0 - shared_mass

    # coverage -- how much of each theme's mass maps to the other set
    coverage_a = P.sum(axis=1).tolist()
    coverage_b = P.sum(axis=0).tolist()

    result = {
        "shared_mass": shared_mass,
        "avg_cost": avg_cost,
        "unmatched_mass": unmatched_mass,
        "transport_plan": P,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "reg": reg,
        "reg_m": reg_m,
        "mode": mode,
    }

    # compute null baseline from pre-computed null cost matrices (word-salad)
    if null_cost_matrices is not None and len(null_cost_matrices) > 0:
        logger.debug(f"Computing null distribution from {len(null_cost_matrices)} word-salad samples")
        null_shared_masses = []
        null_avg_costs = []

        for M_null in null_cost_matrices:
            M_null = np.asarray(M_null, dtype=np.float64)
            M_null = np.clip(M_null, 0, None)

            # null may have different n_B, so recompute b distribution
            n_B_null = M_null.shape[1]
            b_null = np.ones(n_B_null) / n_B_null

            P_null = run_ot(M_null, a, b_null, mode)

            null_shared = float(P_null.sum())
            null_shared_masses.append(null_shared)

            if null_shared > 1e-9:
                null_avg = float((P_null * M_null).sum() / null_shared)
            else:
                null_avg = float("nan")
            null_avg_costs.append(null_avg)

        null_shared_arr = np.array(null_shared_masses)
        null_avg_arr = np.array([x for x in null_avg_costs if not np.isnan(x)])

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

        # shared_mass_excess: raw difference above null (how much MORE mass transported)
        null_mean = result["null_shared_mass_mean"]
        result["shared_mass_excess"] = float(shared_mass - null_mean)

        # shared_mass_relative: 0 = same as random, 1 = perfect transport
        # formula: (observed - null) / (1 - null)
        # answers: "of the improvement possible beyond random, what fraction did we achieve?"
        if null_mean < 1.0:
            result["shared_mass_relative"] = float(
                (shared_mass - null_mean) / (1.0 - null_mean)
            )
        else:
            result["shared_mass_relative"] = 0.0

        # shared_mass_effect: robust effect size using MAD (median absolute deviation)
        # MAD is less sensitive to variance collapse from longer embeddings
        null_median = np.median(null_shared_arr)
        null_mad = np.median(np.abs(null_shared_arr - null_median))
        result["shared_mass_effect"] = float(
            (shared_mass - null_mean) / (null_mad + 1e-9)
        )
        result["null_shared_mass_mad"] = float(null_mad)

        # avg_cost metrics (lower is better, so we measure improvement)
        if len(null_avg_arr) > 0 and not np.isnan(avg_cost):
            null_cost_mean = result["null_avg_cost_mean"]
            # avg_cost_improvement: positive = better than null
            result["avg_cost_improvement"] = float(null_cost_mean - avg_cost)
            # avg_cost_relative: 0 = same as random, 1 = perfect (zero cost)
            if null_cost_mean > 0:
                result["avg_cost_relative"] = float(
                    (null_cost_mean - avg_cost) / null_cost_mean
                )
            else:
                result["avg_cost_relative"] = 0.0
            # avg_cost_effect: robust effect size using MAD
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


def _cost_to_color(norm_cost: float, opacity: float = 0.6) -> str:
    """Convert normalised cost [0,1] to RGBA colour string using green-amber-red gradient.

    Args:
        norm_cost: Normalised cost value (0 = best match/green, 1 = worst match/red)
        opacity: Alpha value for the colour (0-1)

    Returns:
        RGBA colour string for Plotly
    """
    # Three-colour gradient: green → amber → red
    # Green (good match): RGB(39, 174, 96) - #27ae60
    # Amber (medium match): RGB(243, 156, 18) - #f39c12
    # Red (poor match): RGB(231, 76, 60) - #e74c3c
    green = (39, 174, 96)
    amber = (243, 156, 18)
    red = (231, 76, 60)

    if norm_cost <= 0.5:
        # Interpolate green → amber (0 to 0.5)
        t = norm_cost * 2  # scale to 0-1
        r = int(green[0] + (amber[0] - green[0]) * t)
        g = int(green[1] + (amber[1] - green[1]) * t)
        b = int(green[2] + (amber[2] - green[2]) * t)
    else:
        # Interpolate amber → red (0.5 to 1)
        t = (norm_cost - 0.5) * 2  # scale to 0-1
        r = int(amber[0] + (red[0] - amber[0]) * t)
        g = int(amber[1] + (red[1] - amber[1]) * t)
        b = int(amber[2] + (red[2] - amber[2]) * t)

    return f"rgba({r}, {g}, {b}, {opacity})"


# CSS to force opaque hover labels (Plotly Sankey inherits link opacity by default)
_SANKEY_HOVER_CSS = """
<style>
.hoverlayer .hovertext path {
    fill: white !important;
    fill-opacity: 1 !important;
    stroke: #ccc !important;
    stroke-opacity: 1 !important;
}
.hoverlayer .hovertext text {
    fill: black !important;
    fill-opacity: 1 !important;
}
</style>
"""

# Plotly config to hide modebar and logo
_SANKEY_PLOTLY_CONFIG = {
    'displayModeBar': False,
    'displaylogo': False,
    'staticPlot': False,
}


class SankeyHTML:
    """Wrapper for Sankey diagram that provides both HTML and base64 PNG."""

    def __init__(self, html_content: str, png_buffer: BytesIO, name: str = "transport_sankey"):
        self.html_content = html_content
        self.png_buffer = png_buffer
        self.name = name

    @property
    def html(self) -> str:
        """Return full HTML with CSS for embedding."""
        return self.html_content

    @property
    def base64(self) -> str:
        """Return base64-encoded PNG for static display."""
        self.png_buffer.seek(0)
        return base64.b64encode(self.png_buffer.read()).decode("utf-8")


def create_transport_sankey(
    transport_plan,
    theme_names_a: List[str],
    theme_names_b: List[str],
    cost_matrix=None,
    analysis_name_a: str = "A",
    analysis_name_b: str = "B",
    threshold_ratio: float = 0.01,
    link_opacity: float = 0.6,
) -> "SankeyHTML":
    """Create interactive Sankey diagram visualising optimal transport flow.

    Features:
    - PiYG colour scale for alignment quality (green = good, pink = poor)
    - Labels outside plot area with text wrapping and hyphenation
    - A nodes in fixed alphabetical order, B nodes positioned to minimise crossings
    - Hover text showing mass proportions, cost contribution, and unit cost
    - Opaque hover labels

    Args:
        transport_plan: (n_A x n_B) transport coupling matrix P
        theme_names_a: Theme names for set A (left side)
        theme_names_b: Theme names for set B (right side)
        cost_matrix: Optional cost matrix for colouring links (1 - similarity)
        analysis_name_a: Name of analysis A
        analysis_name_b: Name of analysis B
        threshold_ratio: Drop links below this fraction of max flow
        link_opacity: Opacity of link colours (0-1)

    Returns:
        SankeyHTML object with .html and .base64 properties
    """
    import numpy as np
    import plotly.graph_objects as go
    import pyphen

    P = np.asarray(transport_plan)
    n_A, n_B = P.shape

    # handle empty case
    if n_A == 0 or n_B == 0:
        empty_html = "<div>No themes to display</div>"
        empty_buffer = BytesIO()
        return SankeyHTML(empty_html, empty_buffer, name="transport_sankey")

    threshold = threshold_ratio * P.max()

    # sort nodes alphanumerically for consistency
    a_order = np.argsort([n.lower() for n in theme_names_a])
    b_order = np.argsort([n.lower() for n in theme_names_b])

    P_sorted = P[a_order, :][:, b_order]
    names_a_sorted = [theme_names_a[i] for i in a_order]
    names_b_sorted = [theme_names_b[i] for i in b_order]

    if cost_matrix is not None:
        M_sorted = np.asarray(cost_matrix)[a_order, :][:, b_order]
    else:
        M_sorted = None

    # text wrapping with hyphenation and widow control
    dic = pyphen.Pyphen(lang='en_GB')

    def hyphenate_word(word, max_len=10):
        if len(word) <= max_len:
            return [word]
        pairs = dic.pairs(word)
        if not pairs:
            return [word]
        mid = len(word) // 2
        best_pair = min(pairs, key=lambda p: abs(len(p[0]) - mid))
        return [best_pair[0] + '-', best_pair[1]]

    def wrap_with_hyphenation(text, width):
        words = text.split()
        lines = []
        current_line = []
        current_len = 0

        for word in words:
            word_len = len(word)
            space_needed = 1 if current_line else 0

            if current_len + space_needed + word_len <= width:
                current_line.append(word)
                current_len += space_needed + word_len
            elif word_len > width:
                parts = hyphenate_word(word)
                for part in parts:
                    if current_len + (1 if current_line else 0) + len(part) <= width:
                        current_line.append(part)
                        current_len += (1 if current_line else 0) + len(part)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [part]
                        current_len = len(part)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_len = word_len

        if current_line:
            lines.append(' '.join(current_line))
        return lines

    def wrap_text(text, max_width=35, min_last_line_ratio=0.6):
        if len(text) <= max_width:
            return text

        best_lines = None
        best_score = float('inf')

        for width in range(max(20, max_width - 8), max_width + 1):
            lines = wrap_with_hyphenation(text, width)
            if not lines:
                continue

            lengths = [len(line) for line in lines]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

            last_len = lengths[-1]
            max_len = max(lengths)
            widow_ratio = last_len / max_len if max_len > 0 else 1
            widow_penalty = 50 * (1 - widow_ratio) ** 2 if widow_ratio < min_last_line_ratio else 0

            score = len(lines) * 5 + variance * 0.5 + widow_penalty

            if score < best_score:
                best_score = score
                best_lines = lines

        return "<br>".join(best_lines) if best_lines else text

    # node hover texts (show full analysis name in hover)
    hover_texts = (
        [f"A{i+1} ({analysis_name_a}): {names_a_sorted[i]}" for i in range(n_A)] +
        [f"B{j+1} ({analysis_name_b}): {names_b_sorted[j]}" for j in range(n_B)]
    )

    # collect links and unit costs
    sources, targets, values, link_costs = [], [], [], []
    for i in range(n_A):
        for j in range(n_B):
            flow = P_sorted[i, j]
            if flow > threshold:
                sources.append(i)
                targets.append(n_A + j)
                values.append(float(flow))
                if M_sorted is not None:
                    link_costs.append(M_sorted[i, j])

    # normalise unit costs for colour scale
    if link_costs:
        cost_min, cost_max = min(link_costs), max(link_costs)
        cost_range = cost_max - cost_min
        colors = []
        for cost in link_costs:
            norm_cost = (cost - cost_min) / cost_range if cost_range > 0 else 0.5
            colors.append(_cost_to_color(norm_cost, link_opacity))
    else:
        colors = [f"rgba(100, 150, 200, {link_opacity})"] * len(sources)
        link_costs = [0.5] * len(sources)  # default for hover text

    # A positions: fixed top to bottom
    y_a = np.linspace(0.02, 0.98, n_A).tolist() if n_A > 1 else [0.5]

    # B positions: flow-weighted to minimise crossings
    b_weighted_y = []
    for j in range(n_B):
        total_flow = 0
        weighted_sum = 0
        for i in range(n_A):
            flow = P_sorted[i, j]
            if flow > 0:
                weighted_sum += flow * y_a[i]
                total_flow += flow
        b_weighted_y.append(weighted_sum / total_flow if total_flow > 0 else 0.5)

    b_order_by_y = np.argsort(b_weighted_y)
    y_b = [0.0] * n_B
    y_positions = np.linspace(0.02, 0.98, n_B) if n_B > 1 else [0.5]
    for rank, b_idx in enumerate(b_order_by_y):
        y_b[b_idx] = y_positions[rank]

    node_x = [0.05] * n_A + [0.95] * n_B
    node_y = y_a + y_b

    # build link hover text
    total_cost = sum(values[i] * link_costs[i] for i in range(len(values))) if link_costs else 1
    total_mass = sum(values) if values else 1
    a_mass_totals = P_sorted.sum(axis=1)
    b_mass_totals = P_sorted.sum(axis=0)

    link_hovers = []
    for idx in range(len(sources)):
        src_idx = sources[idx]
        tgt_idx = targets[idx] - n_A
        flow = values[idx]
        cost = link_costs[idx]

        mass_prop_a = flow / a_mass_totals[src_idx] if a_mass_totals[src_idx] > 0 else 0
        mass_prop_b = flow / b_mass_totals[tgt_idx] if b_mass_totals[tgt_idx] > 0 else 0

        link_cost_contribution = flow * cost
        cost_prop = link_cost_contribution / total_cost if total_cost > 0 else 0

        link_hovers.append(
            f"A{src_idx+1} → B{tgt_idx+1}<br>"
            f"<b>Mass:</b> {mass_prop_a:.0%} of A{src_idx+1}, "
            f"{mass_prop_b:.0%} of B{tgt_idx+1}<br>"
            f"<b>Cost contribution:</b> {cost_prop:.1%} of total<br>"
            f"<b>Unit cost:</b> {cost:.2f} (colour scale)"
        )

    # empty node labels (full labels in annotations)
    node_labels = [""] * (n_A + n_B)

    hoverlabel_style = dict(
        bgcolor="white",
        bordercolor="#ccc",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=12,
            color="black",
        ),
    )

    fig = go.Figure(data=[go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=20,
            thickness=2,
            line=dict(color="#666666", width=0.5),
            label=node_labels,
            color=["#666666"] * (n_A + n_B),
            x=node_x,
            y=node_y,
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
            hoverlabel=hoverlabel_style,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            customdata=link_hovers,
            hovertemplate="%{customdata}<extra></extra>",
            hoverlabel=hoverlabel_style,
        )
    )])

    # annotations for labels outside plot area
    annotations = []

    for i, name in enumerate(names_a_sorted):
        annotations.append(dict(
            x=-0.01,
            y=1 - y_a[i],
            xref="paper",
            yref="paper",
            text=f"{wrap_text(name, 35)} (A{i+1})",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(size=13),
            align="right",
        ))

    for j, name in enumerate(names_b_sorted):
        annotations.append(dict(
            x=1.01,
            y=1 - y_b[j],
            xref="paper",
            yref="paper",
            text=f"{wrap_text(name, 35)} (B{j+1})",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=13),
            align="left",
        ))

    padding = 48  # 3em
    fig.update_layout(
        title_text="",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif",
            size=9,
        ),
        width=950,
        height=max(900, 90 * max(n_A, n_B)),
        margin=dict(l=305 + padding, r=305 + padding, t=padding, b=padding),
        annotations=annotations,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 1)",
            bordercolor="rgba(200, 200, 200, 1)",
            font=dict(
                family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                size=12,
                color="rgba(0, 0, 0, 1)",
            ),
            namelength=-1,
        ),
    )

    # generate HTML with CSS injection (use CDN to reduce size)
    html_str = fig.to_html(config=_SANKEY_PLOTLY_CONFIG, include_plotlyjs='cdn', full_html=True)
    html_str = html_str.replace("<head>", f"<head>{_SANKEY_HOVER_CSS}")

    # generate PNG for fallback
    try:
        img_bytes = fig.to_image(format="png", scale=2)
        png_buffer = BytesIO(img_bytes)
    except Exception as e:
        logger.debug(f"PNG export failed: {e}")
        png_buffer = BytesIO()

    return SankeyHTML(html_str, png_buffer, name="transport_sankey")


def create_transport_heatmap(
    transport_plan,
    theme_names_a: List[str],
    theme_names_b: List[str],
    analysis_name_a: str = "A",
    analysis_name_b: str = "B",
) -> "Base64ImageFile":
    """Create heatmap of transport plan P, sorted alphanumerically.

    Args:
        transport_plan: (n_A x n_B) transport coupling matrix P
        theme_names_a: Theme names for set A (rows)
        theme_names_b: Theme names for set B (columns)
        analysis_name_a: Name of analysis A
        analysis_name_b: Name of analysis B

    Returns:
        Base64ImageFile containing the heatmap
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    P = np.asarray(transport_plan)
    n_A, n_B = P.shape

    if n_A == 0 or n_B == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No themes to display", ha="center", va="center")
        ax.axis("off")
        buffer = BytesIO()
        fig.savefig(buffer, dpi=150, bbox_inches="tight", format="png")
        plt.close(fig)
        buffer.seek(0)
        return Base64ImageFile(buffer, name="transport_heatmap.png")

    names_a = list(theme_names_a)
    names_b = list(theme_names_b)

    # sort alphanumerically for consistency across runs
    a_order = np.argsort([n.lower() for n in names_a])
    b_order = np.argsort([n.lower() for n in names_b])
    P = P[a_order, :][:, b_order]
    names_a = [names_a[i] for i in a_order]
    names_b = [names_b[i] for i in b_order]

    # truncate names
    def truncate(s, max_len=30):
        return s if len(s) <= max_len else s[:max_len - 3] + "..."

    names_a_display = [truncate(n) for n in names_a]
    names_b_display = [truncate(n) for n in names_b]

    # normalize by shared_mass so values sum to 100% of transported mass
    shared_mass = P.sum()
    if shared_mass > 1e-9:
        P_pct = (P / shared_mass) * 100
    else:
        P_pct = P * 100
    logger.info(f"Transport heatmap: max={P_pct.max():.1f}%, sum={P_pct.sum():.1f}% (shared_mass={shared_mass:.3f})")

    fig_height = max(6, n_A * 0.4)
    fig_width = max(10, n_B * 0.5)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    df = pd.DataFrame(P_pct, index=names_a_display, columns=names_b_display)

    sns.heatmap(
        df,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "% of Transported Mass"},
        ax=ax,
        vmin=0,
    )

    ax.set_title(f"Transport Plan: {analysis_name_a} → {analysis_name_b}")
    ax.set_xlabel(analysis_name_b)
    ax.set_ylabel(analysis_name_a)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)
    return Base64ImageFile(buffer, name="transport_heatmap.png")


class Base64ImageFile:
    """Simple wrapper for BytesIO that provides base64 encoding."""

    def __init__(self, buffer, name=None):
        self.buffer = buffer
        self.name = name

    @property
    def base64(self):
        self.buffer.seek(0)
        return base64.b64encode(self.buffer.read()).decode("utf-8")


def create_unmatched_mass_scree_plot(
    ot_by_k: Dict[float, Dict],
    k_values: List[float],
    analysis_name_a: str = "A",
    analysis_name_b: str = "B",
) -> "Base64ImageFile":
    """Create scree plot showing unmatched mass across different K values.

    Args:
        ot_by_k: Dictionary mapping K values to OT results
        k_values: List of K values used
        analysis_name_a: Name of analysis A
        analysis_name_b: Name of analysis B

    Returns:
        Base64ImageFile containing the scree plot
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # extract unmatched mass for each K
    unmatched_masses = [ot_by_k[k]["ot"]["unmatched_mass"] * 100 for k in k_values]
    shared_masses = [ot_by_k[k]["ot"]["shared_mass"] * 100 for k in k_values]

    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plot 1: unmatched mass
    ax1.plot(k_values, unmatched_masses, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax1.fill_between(k_values, unmatched_masses, alpha=0.2, color='#e74c3c')
    ax1.set_xlabel('K (Mass Penalty)', fontsize=11)
    ax1.set_ylabel('Unmatched Mass (%)', fontsize=11)
    ax1.set_title(f'Unmatched Mass vs K\n{analysis_name_a} ↔ {analysis_name_b}', fontsize=12)
    ax1.set_xlim(-0.05, max(k_values) + 0.05)
    ax1.set_ylim(0, max(unmatched_masses) * 1.1 + 1)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # annotate selected points to avoid clutter
    for i, (k, um) in enumerate(zip(k_values, unmatched_masses)):
        # only annotate every few points to avoid overlap
        if i == 0 or i == len(k_values) - 1 or i % 3 == 0:
            ax1.annotate(f'{um:.1f}%', (k, um), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8)

    # plot 2: shared mass (inverse perspective)
    ax2.plot(k_values, shared_masses, 'o-', color='#27ae60', linewidth=2, markersize=6)
    ax2.fill_between(k_values, shared_masses, alpha=0.2, color='#27ae60')
    ax2.set_xlabel('K (Mass Penalty)', fontsize=11)
    ax2.set_ylabel('Shared Mass (%)', fontsize=11)
    ax2.set_title(f'Shared Mass vs K\n{analysis_name_a} ↔ {analysis_name_b}', fontsize=12)
    ax2.set_xlim(-0.05, max(k_values) + 0.05)
    ax2.set_ylim(min(shared_masses) * 0.9, 100)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    # annotate selected points to avoid clutter
    for i, (k, sm) in enumerate(zip(k_values, shared_masses)):
        # only annotate every few points to avoid overlap
        if i == 0 or i == len(k_values) - 1 or i % 3 == 0:
            ax2.annotate(f'{sm:.1f}%', (k, sm), textcoords="offset points",
                         xytext=(0, -15), ha='center', fontsize=8)

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)

    return Base64ImageFile(buffer, name="unmatched_mass_scree.png")


def compare_result_similarity(
    A: QualitativeAnalysis,
    B: QualitativeAnalysis,
    threshold: float = 0.6,
    embedding_template: str = "{name}",
    embedding_backend: str = "local",
    embedding_model: str = "all-MiniLM-L6-v2",
    k: float = 1.0,
    reg_m: float = 0.2,
    n_null_samples: int = 100,
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
        - percentile_normalized_shepard: Shepard normalized by within-set percentiles
        - z_score_normalized_shepard: Shepard normalized by within-set z-scores
    """

    # extract theme names and analysis names before reassigning A and B
    theme_names_A = [theme.name for theme in A.themes]
    theme_names_B = [theme.name for theme in B.themes]
    analysis_name_A = A.name
    analysis_name_B = B.name

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

    mean_embedding_len_A = sum(count_words(t) for t in A_texts) / len(A_texts) if A_texts else 0
    mean_embedding_len_B = sum(count_words(t) for t in B_texts) / len(B_texts) if B_texts else 0

    # keep A and B as the text lists for backward compatibility
    A = A_texts
    B = B_texts

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
    # Use Angular similarity: proper metric (satisfies triangle inequality), valid to average,
    # and scale is similar to cosine (0-1) so threshold 0.6 is sensible
    hungarian_results = hungarian_matching(angle_sim, threshold=threshold)

    # log Hungarian results
    logger.info(f"\n=== Hungarian Matching (1-to-1, Angular similarity) ===")
    logger.info(f"Optimal assignment: {hungarian_results['distribution']['n_pairs']}/{min(len(emb_A), len(emb_B))} pairs above threshold")
    logger.info(f"Coverage: {hungarian_results['thresholded_metrics']['coverage_a']:.1%} of A, {hungarian_results['thresholded_metrics']['coverage_b']:.1%} of B")
    logger.info(f"True Jaccard: {hungarian_results['thresholded_metrics']['true_jaccard']:.3f}")
    logger.info(f"Fidelity -- Mean assignment similarity: {hungarian_results['soft_metrics']['soft_precision']:.3f}, Normalised total: {hungarian_results['soft_metrics']['soft_recall']:.3f}")

    dist = hungarian_results['distribution']
    if dist['n_pairs'] > 0:
        logger.info(f"Similarity distribution: median={dist['median']:.3f} (Q1={dist['q1']:.3f}, Q3={dist['q3']:.3f}, range: {dist['min']:.3f}-{dist['max']:.3f})")

    # === UNBALANCED OPTIMAL TRANSPORT (many-to-many alignment) ===
    # use cosine distance as cost matrix
    cost_matrix = 1 - sim_matrix
    logger.info("\n=== Computing Unbalanced Optimal Transport Metrics ===")

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

    logger.info(f"Generating symmetric word-salad null ({n_samples_per_direction} samples each direction)...")

    # Direction 1: A vs word-salad-B
    salad_B_list = generate_word_salad_texts(B_texts, n_samples=n_samples_per_direction)
    null_cost_matrices_B = []
    for salad_texts in tqdm(salad_B_list, desc="Null embeddings (A vs B_salad)", file=sys.stderr, leave=False):
        emb_B_salad = get_embedding(
            [t.strip() for t in salad_texts],
            backend=embedding_backend,
            model_name=embedding_model,
        )
        sim_salad = cosine_similarity(emb_A, emb_B_salad)
        null_cost_matrices_B.append(1 - sim_salad)

    # Direction 2: word-salad-A vs B
    salad_A_list = generate_word_salad_texts(A_texts, n_samples=n_samples_per_direction)
    null_cost_matrices_A = []
    for salad_texts in tqdm(salad_A_list, desc="Null embeddings (A_salad vs B)", file=sys.stderr, leave=False):
        emb_A_salad = get_embedding(
            [t.strip() for t in salad_texts],
            backend=embedding_backend,
            model_name=embedding_model,
        )
        sim_salad = cosine_similarity(emb_A_salad, emb_B)
        null_cost_matrices_A.append(1 - sim_salad)

    # Combine both directions for symmetric null
    null_cost_matrices = null_cost_matrices_B + null_cost_matrices_A

    # store word salad samples for display (showing B direction as example)
    word_salad_samples = salad_B_list

    logger.info(f"Generated {len(null_cost_matrices)} null cost matrices ({len(null_cost_matrices_B)} A vs B_salad + {len(null_cost_matrices_A)} A_salad vs B)")

    # === COMPUTE OT FOR MULTIPLE K VALUES ===
    # K (reg_m) controls when themes are left unmatched vs forced to align
    # Higher K = stronger penalty for unmatching = more mass forced to transport
    # Lower K = weaker penalty = more mass can remain unmatched
    # K values: finer granularity at low end where behavior changes most
    # 0.01-0.5: steps of 0.05 | 0.5-1.0: steps of 0.1 | 1.0-2.0: steps of 0.25
    K_VALUES = [
        0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.6, 0.7, 0.8, 0.9, 1.0,
        1.25, 1.5, 1.75, 2.0
    ]
    DEFAULT_K = 0.2

    ot_by_k = {}
    for k_val in tqdm(K_VALUES, desc="Computing OT for K values", file=sys.stderr):
        logger.debug(f"\n--- Computing OT with K={k_val} ---")

        # OPTIMIZATION: Only compute null baseline for default K
        # Other K values only need absolute metrics for scree comparison
        use_null = null_cost_matrices if k_val == DEFAULT_K else None

        ot_result = compute_ot(
            cost_matrix,
            null_cost_matrices=use_null,
            mode="unbalanced",
            reg_m=k_val,
        )

        # generate transport visualisations for this K
        transport_sankey_k = create_transport_sankey(
            ot_result["transport_plan"],
            theme_names_A,
            theme_names_B,
            cost_matrix=cost_matrix,
            analysis_name_a=analysis_name_A,
            analysis_name_b=analysis_name_B,
        )
        transport_heatmap_k = create_transport_heatmap(
            ot_result["transport_plan"],
            theme_names_A,
            theme_names_B,
            analysis_name_a=analysis_name_A,
            analysis_name_b=analysis_name_B,
        )

        # compute split/join statistics for this transport plan
        split_join_stats = compute_split_join_stats(ot_result["transport_plan"])

        # prepare OT results for serialisation (remove numpy array)
        ot_serialisable_k = {key: v for key, v in ot_result.items() if key != "transport_plan"}
        ot_serialisable_k["transport_plan"] = np.round(ot_result["transport_plan"], 4).tolist()
        ot_serialisable_k["split_join_stats"] = split_join_stats

        ot_by_k[k_val] = {
            "ot": ot_serialisable_k,
            "transport_sankey": transport_sankey_k,
            "transport_heatmap": transport_heatmap_k,
            "split_join_stats": split_join_stats,
        }

        logger.debug(f"K={k_val:.1f}: shared_mass={ot_result['shared_mass']:.3f}, avg_cost={ot_result['avg_cost']:.3f}")
        logger.debug(f"  Splits from A: mean={split_join_stats['splits_from_a']['mean']:.2f}, max={split_join_stats['splits_from_a']['max']}")
        logger.debug(f"  Joins to B: mean={split_join_stats['joins_to_b']['mean']:.2f}, max={split_join_stats['joins_to_b']['max']}")

    # use default K for backward compatibility
    ot_results = ot_by_k[DEFAULT_K]["ot"]
    ot_results["transport_plan"] = np.array(ot_results["transport_plan"])  # convert back for later use
    transport_sankey = ot_by_k[DEFAULT_K]["transport_sankey"]
    transport_heatmap = ot_by_k[DEFAULT_K]["transport_heatmap"]

    # generate scree plot showing unmatched mass vs K
    unmatched_mass_scree = create_unmatched_mass_scree_plot(
        ot_by_k,
        K_VALUES,
        analysis_name_a=analysis_name_A,
        analysis_name_b=analysis_name_B,
    )

    # log default K results
    logger.info(f"\n=== Default K={DEFAULT_K} Results ===")
    logger.info(f"Shared Mass: {ot_results['shared_mass']:.3f} (proportion of mass transported)")
    logger.info(f"Unmatched Mass: {ot_results['unmatched_mass']:.3f} (novel/missing themes)")
    logger.info(f"Average Cost: {ot_results['avg_cost']:.3f} (lower = better alignment)")
    logger.info(f"Regularisation: reg={ot_results['reg']:.4f}, reg_m (K)={ot_results['reg_m']:.4f}")

    # null comparison with interpretable relative metrics
    if "null_shared_mass_mean" in ot_results:
        logger.info(f"--- Null baseline comparison ---")
        logger.info(f"Null shared_mass: mean={ot_results['null_shared_mass_mean']:.3f}, 95pct={ot_results['null_shared_mass_95pct']:.3f}")
        logger.info(f"Shared mass excess: +{ot_results['shared_mass_excess']:.3f} (raw improvement over null)")
        logger.info(f"Shared mass relative: {ot_results['shared_mass_relative']:.3f} (0=random, 1=perfect)")
        logger.info(f"Shared mass effect: {ot_results['shared_mass_effect']:.2f} MADs above null")
        logger.info(f"--- Cost metrics ---")
        logger.info(f"Null avg_cost: mean={ot_results['null_avg_cost_mean']:.3f}, 5pct={ot_results['null_avg_cost_5pct']:.3f}")
        logger.info(f"Avg cost improvement: {ot_results['avg_cost_improvement']:.3f} (positive = better than null)")
        logger.info(f"Avg cost relative: {ot_results['avg_cost_relative']:.3f} (0=random, 1=perfect)")

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
    mean_max_sim_a_to_b = sim_matrix.max(axis=1).mean().round(3) if len(emb_A) > 0 else 0

    # Mean max similarity B→A: for each B theme, find best match in A, then average
    mean_max_sim_b_to_a = sim_matrix.max(axis=0).mean().round(3) if len(emb_B) > 0 else 0

    # Fidelity: harmonic mean of the two directional fidelity scores
    fidelity = (
        2
        * (mean_max_sim_a_to_b * mean_max_sim_b_to_a)
        / (mean_max_sim_a_to_b + mean_max_sim_b_to_a)
        if (mean_max_sim_a_to_b + mean_max_sim_b_to_a) > 0
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
        # sort by similarity descending
        best_matches_a_to_b.sort(key=lambda x: x["similarity"], reverse=True)

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
        # sort by similarity descending
        best_matches_b_to_a.sort(key=lambda x: x["similarity"], reverse=True)

    # prepare OT results for serialisation (remove numpy array)
    ot_serialisable = {key: v for key, v in ot_results.items() if key != "transport_plan"}
    ot_serialisable["transport_plan"] = np.round(ot_results["transport_plan"], 4).tolist()

    return {
        # coverage metrics (hit rates)
        "hit_rate_a": hit_rate_a,
        "hit_rate_b": hit_rate_b,
        "jaccard": jaccard,
        "match_matrix": match_matrix.astype(int),
        # fidelity metrics (mean best-match similarity)
        "mean_max_sim_a_to_b": mean_max_sim_a_to_b,
        "mean_max_sim_b_to_a": mean_max_sim_b_to_a,
        "fidelity": fidelity,
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
        # Unbalanced Optimal Transport (many-to-many alignment with unmatched mass)
        "ot": ot_serialisable,
        # Transport visualisations (for default K)
        "transport_sankey": transport_sankey,
        "transport_heatmap": transport_heatmap,
        # OT results for all K values (for tabbed display)
        "ot_by_k": ot_by_k,
        "k_values": K_VALUES,
        "default_k": DEFAULT_K,
        # Scree plot of unmatched mass vs K
        "unmatched_mass_scree": unmatched_mass_scree,
        # best matches
        "best_matches_a_to_b": best_matches_a_to_b,
        "best_matches_b_to_a": best_matches_b_to_a,
        # embedding metadata (for interpreting effect sizes)
        "mean_embedding_words_a": mean_embedding_len_A,
        "mean_embedding_words_b": mean_embedding_len_B,
        # all word salad samples used in null baseline
        "word_salad_samples": word_salad_samples,
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
    embedding_template="{name}",
    embedding_backend: str = "local",
    embedding_model: str = "all-MiniLM-L6-v2",
    metric_type: str = "cosine",
    k: float = 1.0,
) -> str:
    """Create a heatmap visualization for a single pair of pipeline results.

    Themes are sorted alphanumerically for consistency across runs.

    Args:
        a: First QualitativeAnalysis
        b: Second QualitativeAnalysis
        threshold: Similarity threshold for matching
        use_threshold: Whether to use threshold-based binary heatmap
        embedding_template: Python format string for embeddings. Available: {name}, {description}
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
        embedding_backend = config.get("embedding_backend", "local")
        embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        k = config.get("k", 1.0)
        reg_m = config.get("reg_m", 0.2)

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
                reg_m=reg_m,
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

        # extract transport plots from similarity results
        transport_sankey_dict = {
            k: v["transport_sankey"]
            for k, v in zip(result_combinations_dict.keys(), similarity_results)
        }
        transport_heatmap_dict = {
            k: v["transport_heatmap"]
            for k, v in zip(result_combinations_dict.keys(), similarity_results)
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
                # transport visualisations from unbalanced OT
                "transport_sankey": transport_sankey_dict,
                "transport_heatmap": transport_heatmap_dict,
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
