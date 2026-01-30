"""Theme and code similarity comparison using embeddings."""

import base64
import csv
import hashlib
import io
import itertools
import logging
import sys
import textwrap
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from soak.models import QualitativeAnalysis, QualitativeAnalysisComparison
from soak.models.base import get_embedding, memory

logger = logging.getLogger(__name__)


def rescale_similarity(
    sim_matrix: np.ndarray,
    rescale_min: float = 0.5,
    rescale_max: float = 0.9,
) -> np.ndarray:
    """Rescale similarity matrix to focus on meaningful range.

    Empirical testing shows angular similarity for text embeddings has:
    - Close paraphrases: ~0.83
    - Unrelated texts: ~0.55

    This function truncates values outside [rescale_min, rescale_max] and
    rescales to [0, 1], amplifying differences in the meaningful range.

    Args:
        sim_matrix: Similarity matrix with values nominally in [0, 1]
        rescale_min: Floor value - similarities below this become 0
        rescale_max: Ceiling value - similarities above this become 1

    Returns:
        Rescaled similarity matrix with values in [0, 1]
    """
    # clip to range
    clipped = np.clip(sim_matrix, rescale_min, rescale_max)
    # rescale to 0-1
    rescaled = (clipped - rescale_min) / (rescale_max - rescale_min)
    return rescaled


def _hash_array(arr: np.ndarray) -> str:
    """Create a stable hash of a numpy array for cache keys."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _hash_array_list(arrays: List[np.ndarray]) -> str:
    """Create a stable hash of a list of numpy arrays."""
    combined = "".join(_hash_array(np.asarray(a)) for a in arrays)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


@memory.cache
def _compute_ot_cached(
    cost_matrix_hash: str,
    cost_matrix_tuple: Tuple[Tuple[float, ...], ...],
    null_hashes: Optional[str],
    null_matrices_tuple: Optional[Tuple[Tuple[Tuple[float, ...], ...], ...]],
    mode: str,
    reg: float,
    reg_m: float,
) -> Dict[str, Any]:
    """Cached OT computation. Inputs are hashable tuples.

    This is the cached inner function. The wrapper compute_ot() handles
    numpy array conversion.
    """
    import ot

    # convert tuples back to numpy arrays
    cost_matrix = np.array(cost_matrix_tuple, dtype=np.float64)
    null_cost_matrices = None
    if null_matrices_tuple is not None:
        null_cost_matrices = [
            np.array(m, dtype=np.float64) for m in null_matrices_tuple
        ]

    n_A, n_B = cost_matrix.shape

    if n_A == 0 or n_B == 0:
        return {
            "shared_mass": 0.0,
            "avg_cost": float("nan"),
            "unmatched_mass": 1.0,
            "transport_plan": [],
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

    # ensure cost matrix is non-negative
    M = np.clip(cost_matrix, 0, None)

    # ensure minimum regularisation for numerical stability
    reg = max(reg, 1e-6)
    reg_m = max(reg_m, 1e-6)

    def run_ot(cost, a_dist, b_dist, mode_inner):
        """Run OT with given cost matrix and mode."""
        if mode_inner == "balanced":
            P = ot.emd(a_dist, b_dist, cost)
        else:
            P = ot.unbalanced.sinkhorn_unbalanced(
                a_dist,
                b_dist,
                cost,
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
        "transport_plan": P.tolist(),  # convert to list for caching
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "reg": reg,
        "reg_m": reg_m,
        "mode": mode,
    }

    # compute null baseline from pre-computed null cost matrices (word-salad)
    if null_cost_matrices is not None and len(null_cost_matrices) > 0:
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


def create_embeddings_csv_base64(
    embeddings_a: dict, embeddings_b: dict, name_a: str, name_b: str
) -> str:
    """Create a base64-encoded CSV of embeddings for download.

    Args:
        embeddings_a: Dict with 'labels', 'texts', 'vectors' for set A
        embeddings_b: Dict with 'labels', 'texts', 'vectors' for set B
        name_a: Name of analysis A
        name_b: Name of analysis B

    Returns:
        Base64-encoded CSV string
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # determine embedding dimension
    dim = len(embeddings_a["vectors"][0]) if embeddings_a["vectors"] else 0

    # header row
    header = ["analysis", "label", "text"] + [f"dim_{i}" for i in range(dim)]
    writer.writerow(header)

    # write A embeddings
    for label, text, vec in zip(
        embeddings_a["labels"], embeddings_a["texts"], embeddings_a["vectors"]
    ):
        writer.writerow([name_a, label, text] + vec)

    # write B embeddings
    for label, text, vec in zip(
        embeddings_b["labels"], embeddings_b["texts"], embeddings_b["vectors"]
    ):
        writer.writerow([name_b, label, text] + vec)

    csv_content = output.getvalue()
    return base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")


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
        columns=[str(i) for i in range(len(col_names))],
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


async def _generate_paraphrases_for_theme(
    theme_text: str,
    n_paraphrases: int,
    model_name: str,
    credentials: "LLMCredentials",
) -> List[str]:
    """Generate paraphrases for a single theme using LLM.

    Args:
        theme_text: The theme text to paraphrase
        n_paraphrases: Number of paraphrases to generate
        model_name: LLM model name
        credentials: LLM credentials

    Returns:
        List of n_paraphrases alternative phrasings
    """
    from jinja2 import StrictUndefined, Template
    from struckdown import LLM, chatter_async

    # load prompt template from .sd file
    prompt_path = Path(__file__).parent.parent / "pipelines" / "paraphrase_theme.sd"
    prompt_template = prompt_path.read_text()

    # render template with context
    template = Template(prompt_template, undefined=StrictUndefined)
    prompt = template.render(theme_text=theme_text, n_paraphrases=n_paraphrases)

    llm = LLM(model_name=model_name)

    try:
        result = await chatter_async(
            multipart_prompt=prompt,
            model=llm,
            credentials=credentials,
        )

        # extract paraphrases from result
        if hasattr(result, "outputs") and "alternative_phrasing" in result.outputs:
            paraphrases = result.outputs["alternative_phrasing"]
            if isinstance(paraphrases, list):
                return paraphrases
            elif hasattr(paraphrases, "alternative_phrasing"):
                return paraphrases.alternative_phrasing

        logger.warning(f"Paraphrase generation returned unexpected format for theme: {theme_text[:50]}...")
        return [theme_text] * n_paraphrases  # fallback to original

    except Exception as e:
        logger.warning(f"Paraphrase generation failed for theme: {theme_text[:50]}... Error: {e}")
        return [theme_text] * n_paraphrases  # fallback to original


async def generate_paraphrase_texts(
    theme_texts: List[str],
    n_paraphrases: int = 7,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    """Generate LLM paraphrases of themes for realistic upper bound baseline.

    Each theme gets n_paraphrases alternative phrasings that preserve meaning
    but vary wording. This establishes what similarity we'd expect if two
    analyses captured identical concepts but expressed them differently.

    Args:
        theme_texts: Original theme strings (e.g., "name: description")
        n_paraphrases: Number of paraphrases per theme (default: 7)
        model_name: LLM model for paraphrase generation (default: gpt-4.1-mini)
        api_key: API key (uses LLM_API_KEY env var if not provided)
        base_url: API base URL (uses LLM_API_BASE env var if not provided)

    Returns:
        Tuple of:
        - List of n_themes lists, each containing n_paraphrases strings
        - Metadata dict with model_name, n_paraphrases, etc.
    """
    import asyncio
    from struckdown import LLMCredentials

    if model_name is None:
        model_name = "gpt-4.1-mini"

    # create credentials - use env vars if api_key not explicitly provided
    if api_key is not None:
        credentials = LLMCredentials(api_key=api_key, base_url=base_url)
    else:
        credentials = LLMCredentials()  # uses LLM_API_KEY and LLM_API_BASE env vars

    # generate paraphrases for all themes concurrently
    tasks = [
        _generate_paraphrases_for_theme(text, n_paraphrases, model_name, credentials)
        for text in theme_texts
    ]
    results = await asyncio.gather(*tasks)

    metadata = {
        "model_name": model_name,
        "n_paraphrases": n_paraphrases,
        "n_themes": len(theme_texts),
    }

    return list(results), metadata


def prepare_paraphrase_cost_matrix(
    theme_texts: List[str],
    theme_embeddings: np.ndarray,
    paraphrases: List[List[str]],
    embedding_model: str = "text-embedding-3-large",
    distance: str = "angular",
    shepard_k: float = 1.0,
    rescale_min: Optional[float] = None,
    rescale_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Prepare paraphrase cost matrix for OT computation (without running OT).

    Embeds paraphrases, selects the best paraphrase per theme (highest similarity,
    excluding identical strings), and computes the cost matrix.
    The cost matrix can then be used to run OT at different K values.

    Args:
        theme_texts: Original theme strings
        theme_embeddings: Pre-computed embeddings for theme_texts (n_themes x dim)
        paraphrases: List of paraphrase lists from generate_paraphrase_texts
        embedding_model: Model for embedding paraphrases
        distance: Distance metric (angular, cosine, shepard)
        shepard_k: Shepard k parameter if distance="shepard"

    Returns:
        Dictionary with:
        - cost_matrix: n×n cost matrix for OT
        - sim_matrix: n×n similarity matrix
        - per_theme_similarities: best paraphrase similarity per theme (max, excluding sim >= 1)
        - samples: sample themes with paraphrases for display
    """
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

    if not paraphrases or not theme_texts:
        return None

    # ensure theme_embeddings is a numpy array
    theme_embeddings = np.asarray(theme_embeddings)
    n_themes = len(theme_texts)

    # flatten all paraphrases for batch embedding
    all_paraphrase_texts = []
    for para_list in paraphrases:
        for para in para_list:
            all_paraphrase_texts.append(para)

    # embed all paraphrases in single batch
    logger.info(f"Embedding {len(all_paraphrase_texts)} paraphrase texts...")
    all_paraphrase_embeddings = np.asarray(
        get_embedding(all_paraphrase_texts, model=embedding_model)
    )

    # for each theme, select the best paraphrase (highest similarity, excluding sim >= 1.0)
    # this aligns with how comparisons work: we find the best match for each theme
    n_paraphrases_per_theme = len(paraphrases[0]) if paraphrases else 0
    best_paraphrase_embeddings = np.zeros_like(theme_embeddings)

    for theme_idx in range(n_themes):
        start_idx = theme_idx * n_paraphrases_per_theme
        end_idx = start_idx + n_paraphrases_per_theme
        para_embs = all_paraphrase_embeddings[start_idx:end_idx]

        # compute similarities to each paraphrase
        theme_emb = theme_embeddings[theme_idx].reshape(1, -1)
        para_sims = sklearn_cosine_similarity(theme_emb, para_embs)[0]

        # find best paraphrase (excluding identical strings with sim >= 1.0)
        valid_indices = [j for j, s in enumerate(para_sims) if s < 0.9999]
        if valid_indices:
            best_idx = max(valid_indices, key=lambda j: para_sims[j])
            best_paraphrase_embeddings[theme_idx] = para_embs[best_idx]
        else:
            # all paraphrases were identical, use average as fallback
            avg_emb = para_embs.mean(axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            best_paraphrase_embeddings[theme_idx] = avg_emb

    # compute similarity matrix between original and best paraphrase embeddings
    cos_sim_matrix = sklearn_cosine_similarity(theme_embeddings, best_paraphrase_embeddings)

    # convert to selected distance metric
    if distance == "cosine":
        sim_matrix = cos_sim_matrix
    elif distance == "angular":
        angles = np.degrees(np.arccos(np.clip(cos_sim_matrix, -1.0, 1.0)))
        sim_matrix = 1 - angles / 180.0
    elif distance == "shepard":
        thetas = np.arccos(np.clip(cos_sim_matrix, -1.0, 1.0))
        sim_matrix = (np.exp(-shepard_k * thetas) - np.exp(-shepard_k * np.pi)) / (
            1 - np.exp(-shepard_k * np.pi)
        )
    else:
        sim_matrix = cos_sim_matrix

    # apply rescaling if enabled
    if rescale_min is not None and rescale_max is not None:
        sim_matrix = rescale_similarity(sim_matrix, rescale_min, rescale_max)

    # compute cost matrix for OT
    cost_matrix = 1.0 - sim_matrix

    # compute per-theme similarities using max of individual paraphrases (excluding sim >= 1.0)
    # this aligns with how comparison works: for each theme, find best match
    per_theme_similarities = []
    samples = []

    for i in range(n_themes):
        start_idx = i * n_paraphrases_per_theme
        end_idx = start_idx + n_paraphrases_per_theme
        para_embs = all_paraphrase_embeddings[start_idx:end_idx]

        theme_emb = theme_embeddings[i].reshape(1, -1)
        para_sims = sklearn_cosine_similarity(theme_emb, para_embs)[0]
        if distance == "angular":
            angles = np.degrees(np.arccos(np.clip(para_sims, -1.0, 1.0)))
            para_sims = 1 - angles / 180.0

        # filter out identical strings (sim >= 1.0) and take max of remaining
        valid_sims = [s for s in para_sims if s < 0.9999]
        if valid_sims:
            best_sim = max(valid_sims)
        else:
            # all paraphrases were identical to original, use the averaged embedding similarity
            best_sim = float(sim_matrix[i, i])

        per_theme_similarities.append(best_sim)

        # create samples for display (first few themes)
        if i < 5:
            samples.append({
                "original": theme_texts[i],
                "paraphrases": paraphrases[i],
                "similarity": best_sim,
            })

    return {
        "cost_matrix": cost_matrix,
        "sim_matrix": sim_matrix,
        "per_theme_similarities": per_theme_similarities,
        "samples": samples,
    }


def compute_paraphrase_ot_at_k(
    cost_matrix: np.ndarray,
    reg_m: float,
) -> Dict[str, float]:
    """Run OT on paraphrase cost matrix at a specific K value.

    Args:
        cost_matrix: Pre-computed cost matrix from prepare_paraphrase_cost_matrix
        reg_m: OT mass penalty K

    Returns:
        Dictionary with shared_mass and avg_cost for this K
    """
    ot_result = compute_ot(
        cost_matrix=cost_matrix,
        null_cost_matrices=None,
        mode="unbalanced",
        reg=0.01,
        reg_m=reg_m,
    )

    return {
        "shared_mass": ot_result.get("shared_mass", 1.0),
        "avg_cost": ot_result.get("avg_cost", 0.0),
    }


def compute_paraphrase_baseline(
    theme_texts: List[str],
    theme_embeddings: np.ndarray,
    paraphrases: List[List[str]],
    embedding_model: str = "text-embedding-3-large",
    distance: str = "angular",
    shepard_k: float = 1.0,
    reg_m: float = 0.4,
    rescale_min: Optional[float] = None,
    rescale_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute paraphrase-based upper bound using OT between themes and paraphrases.

    This is a convenience wrapper that prepares the cost matrix and runs OT at a single K.
    For K-specific baselines, use prepare_paraphrase_cost_matrix + compute_paraphrase_ot_at_k.

    Args:
        theme_texts: Original theme strings
        theme_embeddings: Pre-computed embeddings for theme_texts (n_themes x dim)
        paraphrases: List of paraphrase lists from generate_paraphrase_texts
        embedding_model: Model for embedding paraphrases
        distance: Distance metric (angular, cosine, shepard)
        shepard_k: Shepard k parameter if distance="shepard"
        reg_m: OT mass penalty K

    Returns:
        Dictionary with OT-based metrics and samples for display
    """
    prep = prepare_paraphrase_cost_matrix(
        theme_texts, theme_embeddings, paraphrases,
        embedding_model, distance, shepard_k,
        rescale_min, rescale_max
    )

    if prep is None:
        return {
            "paraphrase_similarity_mean": 1.0,
            "paraphrase_similarity_std": 0.0,
            "paraphrase_similarity_per_theme": [],
            "paraphrase_cost_mean": 0.0,
            "samples": [],
        }

    ot_metrics = compute_paraphrase_ot_at_k(prep["cost_matrix"], reg_m)

    std_similarity = float(np.std(prep["per_theme_similarities"]))

    logger.info(
        f"Paraphrase OT baseline (K={reg_m}): shared_mass={ot_metrics['shared_mass']:.1%}, "
        f"avg_cost={ot_metrics['avg_cost']:.2f}"
    )

    return {
        "paraphrase_similarity_mean": ot_metrics["shared_mass"],
        "paraphrase_similarity_std": std_similarity,
        "paraphrase_similarity_per_theme": prep["per_theme_similarities"],
        "paraphrase_cost_mean": ot_metrics["avg_cost"],
        "samples": prep["samples"],
        "cost_matrix": prep["cost_matrix"],  # include for K-specific computation
    }


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
    from collections import Counter

    import numpy as np

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

    Results are cached based on input hashes for performance.

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
    # convert to numpy and ensure float64
    cost_arr = np.asarray(cost_matrix, dtype=np.float64)

    # create hashable tuples for caching
    cost_hash = _hash_array(cost_arr)
    cost_tuple = tuple(tuple(row) for row in cost_arr)

    null_hashes = None
    null_tuple = None
    if null_cost_matrices is not None and len(null_cost_matrices) > 0:
        null_arrays = [np.asarray(m, dtype=np.float64) for m in null_cost_matrices]
        null_hashes = _hash_array_list(null_arrays)
        null_tuple = tuple(tuple(tuple(row) for row in m) for m in null_arrays)

    # call cached function
    result = _compute_ot_cached(
        cost_hash, cost_tuple, null_hashes, null_tuple, mode, reg, reg_m
    )

    # convert transport_plan back to numpy array
    result = dict(result)  # make a copy since cached result shouldn't be modified
    if result["transport_plan"]:
        result["transport_plan"] = np.array(result["transport_plan"])
    else:
        result["transport_plan"] = np.zeros((0, 0))

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

# Plotly config with export buttons
_SANKEY_PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "staticPlot": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    ],
    "modeBarButtonsToAdd": [],
    "toImageButtonOptions": {
        "format": "svg",  # default to SVG for print quality
        "filename": "sankey_diagram",
        "height": None,
        "width": None,
        "scale": 2,
    },
}

# System UI font stack (consistent with struckdown online editor)
_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif"


class SankeyHTML:
    """Wrapper for Sankey diagram that provides both HTML and base64 PNG."""

    def __init__(
        self, html_content: str, png_buffer: BytesIO, name: str = "transport_sankey"
    ):
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
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
) -> "SankeyHTML":
    """Create interactive Sankey diagram visualising optimal transport flow.

    Features:
    - Green-amber-red colour scale for alignment quality (green = good, red = poor)
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
        cost_min: Minimum cost for color scale (green). If None, computed from links.
        cost_max: Maximum cost for color scale (red). If None, computed from links.

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
    dic = pyphen.Pyphen(lang="en_GB")

    def hyphenate_word(word, max_len=10):
        if len(word) <= max_len:
            return [word]
        pairs = dic.pairs(word)
        if not pairs:
            return [word]
        mid = len(word) // 2
        best_pair = min(pairs, key=lambda p: abs(len(p[0]) - mid))
        return [best_pair[0] + "-", best_pair[1]]

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
                            lines.append(" ".join(current_line))
                        current_line = [part]
                        current_len = len(part)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_len = word_len

        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def wrap_text(text, max_width=35, min_last_line_ratio=0.6):
        if len(text) <= max_width:
            return text

        best_lines = None
        best_score = float("inf")

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
            widow_penalty = (
                50 * (1 - widow_ratio) ** 2 if widow_ratio < min_last_line_ratio else 0
            )

            score = len(lines) * 5 + variance * 0.5 + widow_penalty

            if score < best_score:
                best_score = score
                best_lines = lines

        return "<br>".join(best_lines) if best_lines else text

    # node hover texts (show full analysis name in hover)
    hover_texts = [
        f"A{i+1} ({analysis_name_a}): {names_a_sorted[i]}" for i in range(n_A)
    ] + [f"B{j+1} ({analysis_name_b}): {names_b_sorted[j]}" for j in range(n_B)]

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

    # map costs to colours using the provided or computed min/max range
    # this ensures colors are comparable across different K values when using shared scale
    if link_costs:
        # determine color scale range
        if cost_min is None:
            cost_min = min(link_costs)
        if cost_max is None:
            cost_max = max(link_costs)
        cost_range = cost_max - cost_min
        if cost_range < 1e-9:
            cost_range = 1.0  # avoid division by zero if all costs identical

        colors = []
        for cost in link_costs:
            # normalise cost to [0, 1] within the min/max range
            # cost_min → 0 (green), cost_max → 1 (red)
            norm_cost = (cost - cost_min) / cost_range
            norm_cost = max(0.0, min(1.0, norm_cost))  # clamp to [0, 1]
            colors.append(_cost_to_color(norm_cost, link_opacity))
    else:
        colors = [f"rgba(100, 150, 200, {link_opacity})"] * len(sources)
        link_costs = [0.5] * len(sources)  # default for hover text
        cost_min, cost_max = 0.0, 1.0  # defaults for colorbar

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
    total_cost = (
        sum(values[i] * link_costs[i] for i in range(len(values))) if link_costs else 1
    )
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

        similarity = 1 - cost  # cost is angular distance, so similarity = 1 - cost
        link_hovers.append(
            f"A{src_idx+1} → B{tgt_idx+1}<br>"
            f"<b>Similarity:</b> {similarity:.3f}<br>"
            f"<b>Mass:</b> {mass_prop_a:.0%} of A{src_idx+1}, "
            f"{mass_prop_b:.0%} of B{tgt_idx+1}<br>"
            f"<b>Cost contribution:</b> {cost_prop:.1%} of total"
        )

    # empty node labels (full labels in annotations)
    node_labels = [""] * (n_A + n_B)

    hoverlabel_style = dict(
        bgcolor="white",
        bordercolor="#ccc",
        font=dict(
            family=_FONT_STACK,
            size=12,
            color="black",
        ),
    )

    fig = go.Figure(
        data=[
            go.Sankey(
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
                ),
            )
        ]
    )

    # annotations for labels outside plot area
    annotations = []

    for i, name in enumerate(names_a_sorted):
        annotations.append(
            dict(
                x=-0.01,
                y=1 - y_a[i],
                xref="paper",
                yref="paper",
                text=f"{wrap_text(name, 50)} (A{i+1})",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(size=13),
                align="right",
            )
        )

    for j, name in enumerate(names_b_sorted):
        annotations.append(
            dict(
                x=1.01,
                y=1 - y_b[j],
                xref="paper",
                yref="paper",
                text=f"{wrap_text(name, 50)} (B{j+1})",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=13),
                align="left",
            )
        )

    padding = 48  # 3em

    # add continuous colorbar showing similarity scale based on actual cost range
    # cost_min → green (high similarity), cost_max → red (low similarity)
    if link_costs:
        # convert cost range to similarity range
        sim_max = 1 - cost_min  # green end (highest similarity in this analysis)
        sim_min = 1 - cost_max  # red end (lowest similarity in this analysis)

        # Similarity colorscale: low similarity (red) → high similarity (green)
        colorscale = [
            [0.0, "#e74c3c"],  # red (low similarity)
            [0.5, "#f39c12"],  # amber (medium similarity)
            [1.0, "#27ae60"],  # green (high similarity)
        ]

        # generate tick values spread across the similarity range
        sim_range = sim_max - sim_min
        tick_vals = [sim_min + sim_range * i / 4 for i in range(5)]

        # add invisible scatter trace just for the colorbar
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale=colorscale,
                    cmin=sim_min,
                    cmax=sim_max,
                    color=[(sim_min + sim_max) / 2],
                    colorbar=dict(
                        title=dict(
                            text="Similarity",
                            side="top",
                            font=dict(size=13),
                        ),
                        orientation="h",
                        x=0.5,
                        y=-0.08,
                        xanchor="center",
                        yanchor="top",
                        len=0.4,
                        thickness=15,
                        tickfont=dict(size=11),
                        tickformat=".2f",
                        tickvals=tick_vals,
                    ),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title_text="",
        font=dict(
            family=_FONT_STACK,
            size=9,
        ),
        width=1100,
        height=max(600, 60 * max(n_A, n_B)),
        margin=dict(l=380 + padding, r=380 + padding, t=padding, b=padding + 80),
        annotations=annotations,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 1)",
            bordercolor="rgba(200, 200, 200, 1)",
            font=dict(
                family=_FONT_STACK,
                size=12,
                color="rgba(0, 0, 0, 1)",
            ),
            namelength=-1,
        ),
    )

    # generate HTML with CSS injection (use CDN to reduce size)
    html_str = fig.to_html(
        config=_SANKEY_PLOTLY_CONFIG, include_plotlyjs="cdn", full_html=True
    )
    html_str = html_str.replace("<head>", f"<head>{_SANKEY_HOVER_CSS}")

    # add download buttons below the chart
    download_buttons_html = """
<div style="text-align: center; margin: 20px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;">
    <span style="margin-right: 10px; color: #666; font-size: 13px;">Download:</span>
    <button onclick="downloadSankey('svg')" style="
        padding: 8px 16px; margin: 0 5px; cursor: pointer;
        background: #3498db; color: white; border: none; border-radius: 4px;
        font-size: 13px; font-family: inherit;
    ">SVG (vector)</button>
    <button onclick="downloadSankey('png')" style="
        padding: 8px 16px; margin: 0 5px; cursor: pointer;
        background: #27ae60; color: white; border: none; border-radius: 4px;
        font-size: 13px; font-family: inherit;
    ">PNG (2x)</button>
    <button onclick="downloadSankey('pdf')" style="
        padding: 8px 16px; margin: 0 5px; cursor: pointer;
        background: #9b59b6; color: white; border: none; border-radius: 4px;
        font-size: 13px; font-family: inherit;
    ">PDF</button>
</div>
<script>
function downloadSankey(format) {
    var gd = document.querySelector('.plotly-graph-div');
    var filename = 'sankey_diagram';
    if (format === 'pdf') {
        // For PDF, download as SVG and note that user can convert
        Plotly.downloadImage(gd, {format: 'svg', filename: filename + '_for_pdf', scale: 2});
        alert('SVG downloaded. Open in Inkscape, Illustrator, or use an online converter to save as PDF.');
    } else {
        Plotly.downloadImage(gd, {format: format, filename: filename, scale: 2});
    }
}
</script>
"""
    html_str = html_str.replace("</body>", download_buttons_html + "</body>")

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
        return s if len(s) <= max_len else s[: max_len - 3] + "..."

    names_a_display = [truncate(n) for n in names_a]
    names_b_display = [truncate(n) for n in names_b]

    # normalize by shared_mass so values sum to 100% of transported mass
    shared_mass = P.sum()
    if shared_mass > 1e-9:
        P_pct = (P / shared_mass) * 100
    else:
        P_pct = P * 100
    logger.info(
        f"Transport heatmap: max={P_pct.max():.1f}%, sum={P_pct.sum():.1f}% (shared_mass={shared_mass:.1%})"
    )

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


def find_elbow_points(
    k_values: List[float],
    shared_mass: List[float],
    *,
    n_interp: int = 100,
    eps: float = 1e-12,
    plateau_threshold: float = 0.20,
    min_k_for_elbow: float = 0.1,
    max_k_for_elbow: float = 2.0,
) -> dict:
    """Find elbow points using maximum curvature and diminishing returns methods.

    Uses maximum curvature (κ = |y''| / (1 + y'²)^(3/2)) to find the elbow point
    where the curve bends most sharply, indicating the optimal K value.

    Args:
        k_values: K parameter values (should be positive)
        shared_mass: Corresponding shared mass values (should increase with K)
        n_interp: Number of points for uniform grid (default: 100)
        eps: Small constant for numerical stability (default: 1e-12)
        plateau_threshold: Threshold for diminishing returns (default: 0.20)
        min_k_for_elbow: Minimum K to consider for elbow detection (default: 0.1)
            Small K values often have noisy behaviour at the boundary.
        max_k_for_elbow: Maximum K to consider for elbow detection (default: 2.0)
            Anchor points beyond this are excluded from curvature calculation.

    Returns:
        Dictionary with:
        - elbow_idx, elbow_k: Index and K value for maximum curvature elbow
        - diminishing_idx, diminishing_k: Index and K value for diminishing returns point
        - plateau_reached: Whether curve has clearly asymptoted
    """
    import numpy as np
    from scipy.interpolate import UnivariateSpline

    K = np.asarray(k_values, float)
    s = np.asarray(shared_mass, float)

    # for elbow detection, use K in [min_k_for_elbow, max_k_for_elbow]
    elbow_mask = (K >= min_k_for_elbow) & (K <= max_k_for_elbow)
    K_elbow = K[elbow_mask]
    s_elbow = s[elbow_mask]

    # interpolate to uniform grid in linear K space (for elbow detection)
    K_uniform = np.linspace(K_elbow.min(), K_elbow.max(), n_interp)
    s_uniform = np.interp(K_uniform, K_elbow, s_elbow)

    # light smoothing with window=3 (simple moving average)
    if len(s_uniform) >= 3:
        kernel = np.ones(3) / 3
        s_padded = np.pad(s_uniform, (1, 1), mode="edge")
        s_uniform = np.convolve(s_padded, kernel, mode="valid")

    # for plateau detection, use all K values (including anchor)
    K_all_uniform = np.linspace(K.min(), K.max(), n_interp)
    s_all_uniform = np.interp(K_all_uniform, K, s)
    if len(s_all_uniform) >= 3:
        kernel = np.ones(3) / 3
        s_all_padded = np.pad(s_all_uniform, (1, 1), mode="edge")
        s_all_uniform = np.convolve(s_all_padded, kernel, mode="valid")

    # compute slopes for plateau detection (using all K values)
    slope = np.diff(s_all_uniform)
    n_window = min(5, len(slope) // 4) if len(slope) > 4 else 1
    initial_slope = np.mean(np.abs(slope[:n_window])) if len(slope) >= n_window else 0
    final_slope = np.mean(np.abs(slope[-n_window:])) if len(slope) >= n_window else 0

    # === PLATEAU DETECTION ===
    # criterion 1: relative slope -- final slope < 25% of initial
    relative_plateau = (initial_slope <= 0) or (
        final_slope / (initial_slope + 1e-12) < 0.25
    )
    # criterion 2: absolute change in last 20% of curve < 4 points
    n_tail = max(1, len(s_all_uniform) // 5)
    tail_range = s_all_uniform[-1] - s_all_uniform[-n_tail]
    absolute_plateau = abs(tail_range) < 4.0
    # criterion 3: high value (already near maximum)
    high_value_plateau = s_all_uniform[-1] > 85.0
    # criterion 4: consistent deceleration -- curve is flattening even if not flat
    if len(slope) >= 4:
        mid = len(slope) // 2
        first_half_slope = np.mean(np.abs(slope[:mid]))
        second_half_slope = np.mean(np.abs(slope[mid:]))
        decelerating = second_half_slope < 0.5 * first_half_slope
    else:
        decelerating = False
    plateau_reached = (
        relative_plateau or absolute_plateau or high_value_plateau or decelerating
    )

    # === MAXIMUM CURVATURE ELBOW ===
    # normalise both axes to [0, 1] so curvature is scale-invariant
    x_min, x_max = K_uniform.min(), K_uniform.max()
    y_min, y_max = s_uniform.min(), s_uniform.max()
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    x_norm = (K_uniform - x_min) / x_range
    y_norm = (s_uniform - y_min) / y_range

    # fit a smoothing spline to get derivatives
    # use smoothing factor s=0.01 for light smoothing
    try:
        spline = UnivariateSpline(x_norm, y_norm, s=0.01, k=4)
        # first derivative
        dy = spline.derivative(1)(x_norm)
        # second derivative
        d2y = spline.derivative(2)(x_norm)
    except Exception:
        # fallback to finite differences if spline fails
        dy = np.gradient(y_norm, x_norm)
        d2y = np.gradient(dy, x_norm)

    # curvature: κ = |y''| / (1 + y'^2)^(3/2)
    curvature = np.abs(d2y) / (1 + dy**2) ** 1.5

    # find maximum curvature (excluding both endpoints which can have artifacts)
    # exclude first 5% and last 10% (endpoints have unstable spline derivatives)
    start_margin = max(1, n_interp // 20)
    end_margin = max(1, n_interp // 10)
    inner_curvature = curvature[start_margin:-end_margin]
    max_curv_idx = start_margin + int(np.argmax(inner_curvature))

    # map back to original K values (from full K array, not just elbow subset)
    elbow_K = K_uniform[max_curv_idx]
    elbow_orig_idx = int(np.argmin(np.abs(K - elbow_K)))

    # === DIMINISHING RETURNS (slope < 20% of initial) ===
    # Compute in ORIGINAL K space (not log space) for intuitive results
    # Use the raw k_values and shared_mass, not the interpolated log-space data
    k_arr = np.asarray(k_values, float)
    sm_arr = np.asarray(shared_mass, float)

    # compute actual slopes: change in shared_mass per unit change in K
    dk = np.diff(k_arr)
    dsm = np.diff(sm_arr)
    slopes_original = dsm / (dk + 1e-12)  # shared_mass change per K unit

    # initial slope is average of first few points
    n_init = min(3, len(slopes_original))
    initial_slope_orig = np.mean(slopes_original[:n_init]) if n_init > 0 else 0

    # find where slope drops below 20% of initial
    thr_orig = plateau_threshold * initial_slope_orig if initial_slope_orig > 0 else 0
    diminishing_orig_idx = None

    # look for 2 consecutive points below threshold
    run = 0
    for i, slope_val in enumerate(slopes_original):
        if slope_val < thr_orig:
            run += 1
            if run >= 2:
                diminishing_orig_idx = i - 1  # index of first point below threshold
                break
        else:
            run = 0

    if diminishing_orig_idx is None:
        # fallback: find last point where slope >= threshold
        above_thr = np.where(slopes_original >= thr_orig)[0]
        if len(above_thr) > 0:
            diminishing_orig_idx = above_thr[-1] + 1
        else:
            diminishing_orig_idx = len(k_arr) - 1

    diminishing_orig_idx = min(diminishing_orig_idx, len(k_arr) - 1)
    diminishing_K = k_arr[diminishing_orig_idx]

    return {
        # keep chord_idx/chord_k for backward compatibility
        "chord_idx": elbow_orig_idx,
        "chord_k": K[elbow_orig_idx],
        "elbow_idx": elbow_orig_idx,
        "elbow_k": K[elbow_orig_idx],
        "diminishing_idx": diminishing_orig_idx,
        "diminishing_k": K[diminishing_orig_idx],
        "plateau_reached": plateau_reached,
    }


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
    for k in k_values:
        ot_data = ot_by_k[k]["ot"]
        ceiling = ot_data.get("paraphrase_upper_bound")
        floor = ot_data.get("null_shared_mass_mean")
        paraphrase_ceilings.append(ceiling * 100 if ceiling is not None else None)
        word_salad_floors.append(floor * 100 if floor is not None else None)

    # check if we have baseline data
    has_ceiling = any(c is not None for c in paraphrase_ceilings)
    has_floor = any(f is not None for f in word_salad_floors)

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
    ax.set_xlim(min(k_values), max(k_values))  # show only displayed K values
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
    ax.set_xlim(min(k_values), max(k_values))
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
    ax.set_xlim(min(k_values), max(k_values))

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
    embedding_template: str = "{name}",
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
    green_above: float = 0.8,
    red_below: float = 0.65,
    rescale_min: Optional[float] = 0.5,
    rescale_max: Optional[float] = 0.9,
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
        rescale_min: Floor for rescaling similarity (default: 0.5). Values below become 0.
                    Set to None to disable rescaling.
        rescale_max: Ceiling for rescaling similarity (default: 0.9). Values above become 1.
                    Empirically, close paraphrases score ~0.83 and unrelated ~0.55.

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

    # extract theme names/labels and analysis names before reassigning A and B
    # use label if set (from --llm-labels), otherwise fall back to name
    theme_names_A = [theme.label if theme.label else theme.name for theme in A.themes]
    theme_names_B = [theme.label if theme.label else theme.name for theme in B.themes]
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
    # Select similarity matrix based on distance metric
    distance_matrices = {
        "angular": angle_sim,
        "cosine": sim_matrix,
        "shepard": shepard_sim,
    }
    selected_sim = distance_matrices.get(distance, angle_sim)

    # Apply rescaling if enabled (rescale_min and rescale_max both set)
    selected_sim_raw = selected_sim.copy()  # keep original for reference
    if rescale_min is not None and rescale_max is not None:
        selected_sim = rescale_similarity(selected_sim, rescale_min, rescale_max)
        logger.info(
            f"Rescaled similarity from [{rescale_min}, {rescale_max}] to [0, 1]"
        )

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
    # use selected distance metric for cost matrix
    cost_matrix = 1 - selected_sim
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
    def compute_similarity(emb_a, emb_b, metric, k_val, apply_rescale=True):
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
        if apply_rescale and rescale_min is not None and rescale_max is not None:
            sim = rescale_similarity(sim, rescale_min, rescale_max)
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
                rescale_min=rescale_min,
                rescale_max=rescale_max,
            )
            baseline_B = compute_paraphrase_baseline(
                B_texts, emb_B, paraphrases_B,
                embedding_model=embedding_model,
                distance=distance,
                shepard_k=k,
                reg_m=0.3,
                rescale_min=rescale_min,
                rescale_max=rescale_max,
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

    # === COMPUTE OT FOR MULTIPLE K VALUES ===
    # K (reg_m) controls when themes are left unmatched vs forced to align
    # Higher K = stronger penalty for unmatching = more mass forced to transport
    # Lower K = weaker penalty = more mass can remain unmatched
    # K values: fine at low end for elbow detection, coarser at high end
    # 0.025-0.05: very low K | 0.1-1.0: every 0.1 | 1.0-2.0: every 0.5 | 2.0-4.0: every 1.0
    K_VALUES = [
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
    ]
    EXTENDED_K_VALUES = [6.0, 8.0, 10.0]
    PLATEAU_ANCHOR_K = 5.0  # always compute for reliable plateau detection, but hide from plots
    MIN_K_BEFORE_STOP = 0.3  # minimum K to compute before allowing early stopping

    # PHASE 1: Compute OT for all K values (without visualisations yet)
    ot_by_k = {}
    computed_k_values = []
    prev_shared_mass = 0.0
    for k_val in tqdm(K_VALUES, desc="Computing OT for K values", file=sys.stderr):
        logger.debug(f"\n--- Computing OT with K={k_val} ---")

        ot_result = compute_ot(
            cost_matrix,
            null_cost_matrices=null_cost_matrices,
            mode="unbalanced",
            reg_m=k_val,
        )

        split_join_stats = compute_split_join_stats(ot_result["transport_plan"])

        # store transport plan as numpy array for later visualisation
        ot_by_k[k_val] = {
            "ot_result": ot_result,
            "split_join_stats": split_join_stats,
        }
        computed_k_values.append(k_val)

        logger.debug(
            f"K={k_val:.1f}: shared_mass={ot_result['shared_mass']:.1%}, avg_cost={ot_result['avg_cost']:.2f}"
        )
        logger.debug(
            f"  Splits from A: mean={split_join_stats['splits_from_a']['mean']:.2f}, max={split_join_stats['splits_from_a']['max']}"
        )
        logger.debug(
            f"  Joins to B: mean={split_join_stats['joins_to_b']['mean']:.2f}, max={split_join_stats['joins_to_b']['max']}"
        )

        # stop early if improvement < 2.5% (curve has plateaued)
        current_shared_mass = ot_result["shared_mass"]
        improvement = (current_shared_mass - prev_shared_mass) / (
            prev_shared_mass + 1e-9
        )
        prev_shared_mass = current_shared_mass
        if improvement < 0.025 and k_val >= MIN_K_BEFORE_STOP:
            logger.info(
                f"Shared mass improvement {improvement:.1%} < 2.5% at K={k_val}, stopping early"
            )
            break

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

    # PHASE 2: Set color scale from user thresholds (or defaults)
    # If rescaling is enabled, convert thresholds from original to rescaled space
    effective_green_above = green_above
    effective_red_below = red_below
    if rescale_min is not None and rescale_max is not None:
        # convert user's original-space thresholds to rescaled space
        scale_range = rescale_max - rescale_min
        effective_green_above = np.clip((green_above - rescale_min) / scale_range, 0.0, 1.0)
        effective_red_below = np.clip((red_below - rescale_min) / scale_range, 0.0, 1.0)
        logger.info(
            f"Color scale rescaled: original [{red_below:.2f}, {green_above:.2f}] → "
            f"rescaled [{effective_red_below:.2f}, {effective_green_above:.2f}]"
        )

    # green_above (similarity) → cost_min = 1 - green_above
    # red_below (similarity) → cost_max = 1 - red_below
    color_cost_min = 1.0 - effective_green_above  # e.g., 0.8 similarity → 0.2 cost (green)
    color_cost_max = 1.0 - effective_red_below    # e.g., 0.65 similarity → 0.35 cost (red)
    logger.info(
        f"Color scale: similarity [{effective_red_below:.2f}, {effective_green_above:.2f}] → cost [{color_cost_min:.3f}, {color_cost_max:.3f}]"
    )

    # Create visualisations for all K values with shared color scale
    for k_val in tqdm(all_k_values, desc="Creating visualisations", file=sys.stderr):
        ot_result = ot_by_k[k_val]["ot_result"]
        split_join_stats = ot_by_k[k_val]["split_join_stats"]

        transport_sankey_k = create_transport_sankey(
            ot_result["transport_plan"],
            theme_names_A,
            theme_names_B,
            cost_matrix=cost_matrix,
            analysis_name_a=analysis_name_A,
            analysis_name_b=analysis_name_B,
            cost_min=color_cost_min,
            cost_max=color_cost_max,
        )
        transport_heatmap_k = create_transport_heatmap(
            ot_result["transport_plan"],
            theme_names_A,
            theme_names_B,
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
        ot_serialisable_k["split_join_stats"] = split_join_stats

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

        ot_by_k[k_val] = {
            "ot": ot_serialisable_k,
            "transport_sankey": transport_sankey_k,
            "transport_heatmap": transport_heatmap_k,
            "split_join_stats": split_join_stats,
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

    logger.info(
        "\n=== Percentile-Normalized Shepard ===\n"
        + format_similarity_matrix(
            shepard_percentile,
            theme_names_A,
            theme_names_B,
            set_a_name=analysis_name_A,
            set_b_name=analysis_name_B,
            show_legend=False,
        )
    )

    logger.info(
        "\n=== Z-Score Normalized Shepard ===\n"
        + format_similarity_matrix(
            shepard_z,
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
        "rescale_min": rescale_min,
        "rescale_max": rescale_max,
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
        # OT results for displayed K values (for tabbed display)
        "ot_by_k": ot_by_k,
        "k_values": display_k_values,
        "default_k": chord_k,  # default is now chord elbow
        "chord_k": chord_k,
        "diminishing_k": diminishing_k,
        "plateau_reached": plateau_reached,
        # color scale range (from default K, used across all K plots)
        "color_sim_min": round(1 - color_cost_max, 3),
        "color_sim_max": round(1 - color_cost_min, 3),
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
        # paraphrase baseline for upper bound scaling
        "paraphrase_baseline": paraphrase_baseline,
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
    embedding_template="{name}",
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
    embedding_template="{name}",
    embedding_model: str = "text-embedding-3-large",
    metric_type: str = "cosine",
    k: float = 1.0,
    comparison_result: Optional[Dict[str, Any]] = None,
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
        "percentile": (
            "Percentile-Normalized Shepard",
            "percentile_normalized_shepard",
        ),
        "z_score": ("Z-Score Normalized Shepard", "z_score_normalized_shepard"),
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
        embedding_template = config.get("embedding_template", "{name}")
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
        # color scale thresholds for Sankey plots
        green_above = config.get("green_above", 0.8)
        red_below = config.get("red_below", 0.65)
        # rescaling parameters (set to None to disable)
        rescale_min = config.get("rescale_min", 0.5)
        rescale_max = config.get("rescale_max", 0.9)

        # Set labels on all themes once at the beginning (only if not already set)
        for result in pipeline_results:
            for i, theme in enumerate(result.themes, start=1):
                if not theme.label:
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
                embedding_model=embedding_model,
                k=k,
                reg_m=reg_m,
                distance=distance,
                compute_paraphrase_bound=compute_paraphrase_bound,
                n_paraphrases=n_paraphrases,
                paraphrase_model=paraphrase_model,
                api_key=api_key,
                base_url=base_url,
                green_above=green_above,
                red_below=red_below,
                rescale_min=rescale_min,
                rescale_max=rescale_max,
            )
            for i, j in result_combinations
        ]

        # generate heatmaps for all metric types, reusing pre-computed similarity results
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
                    embedding_model=embedding_model,
                    metric_type=metric_type,
                    k=k,
                    comparison_result=sim_result,
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
                "heatmaps_percentile": heatmap_dicts_by_metric["percentile"],
                "heatmaps_z_score": heatmap_dicts_by_metric["z_score"],
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
