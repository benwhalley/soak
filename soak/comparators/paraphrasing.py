"""LLM-based paraphrase generation and paraphrase baseline computation."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from soak.models.base import get_embedding

from .optimal_transport import compute_ot
from .rescaling import RescaleMethod, rescale_similarity

logger = logging.getLogger(__name__)


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


async def generate_short_labels(
    theme_texts: List[str],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[str]:
    """Generate 3-4 word short labels for themes using LLM.

    Each theme gets a concise, unique identifier suitable for use in
    space-constrained contexts like plot labels and matrix headers.

    Args:
        theme_texts: Original theme strings (e.g., "name: description")
        model_name: LLM model for label generation (default: gpt-4.1-mini)
        api_key: API key (uses LLM_API_KEY env var if not provided)
        base_url: API base URL (uses LLM_API_BASE env var if not provided)

    Returns:
        List of short labels (3-4 words each), same length as theme_texts
    """
    from struckdown import LLM, LLMCredentials, chatter_async

    if model_name is None:
        model_name = "gpt-4.1-mini"

    # create credentials - use env vars if api_key not explicitly provided
    if api_key is not None:
        credentials = LLMCredentials(api_key=api_key, base_url=base_url)
    else:
        credentials = LLMCredentials()

    llm = LLM(model_name=model_name)

    single_theme_prompt = """<system>
Generate a unique 3-4 word short label for this theme that captures its essence.
The label should be a concise identifier that captures the core concept.

Guidelines:
- Exactly 3-4 words
- No punctuation or special characters
- Use title case (e.g., "Data Privacy Concerns")
</system>

Theme: {theme}

Generate a 3-4 word short label for this theme.

[[respond:short_label]]
"""

    short_labels = []
    pbar = tqdm(theme_texts, desc="Short labels", file=sys.stderr)
    for i, theme in enumerate(pbar):
        try:
            prompt = single_theme_prompt.format(theme=theme)
            result = await chatter_async(
                multipart_prompt=prompt,
                model=llm,
                credentials=credentials,
            )

            if hasattr(result, "outputs") and "short_label" in result.outputs:
                label = result.outputs["short_label"]
                if isinstance(label, str) and label.strip():
                    short_labels.append(label.strip())
                    continue

            logger.warning(f"Short label generation returned unexpected format for theme {i + 1}")
            short_labels.append(f"Theme {i + 1}")

        except Exception as e:
            logger.warning(f"Short label generation failed for theme {i + 1}: {e}")
            short_labels.append(f"Theme {i + 1}")

    pbar.close()
    return short_labels


def prepare_paraphrase_cost_matrix(
    theme_texts: List[str],
    theme_embeddings: np.ndarray,
    paraphrases: List[List[str]],
    embedding_model: str = "text-embedding-3-large",
    distance: str = "angular",
    shepard_k: float = 1.0,
    # rescaling parameters
    rescale_method: RescaleMethod = "clip",
    rescale_min: float = 0.5,
    rescale_max: float = 0.9,
    rescale_lower_percentile: float = 5.0,
    rescale_upper_percentile: float = 95.0,
    rescale_sigmoid_center: float = 0.7,
    rescale_sigmoid_steepness: float = 10.0,
    rescale_temperature: float = 0.5,
) -> Optional[Dict[str, Any]]:
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
        - cost_matrix: n*n cost matrix for OT
        - sim_matrix: n*n similarity matrix
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
    if rescale_method != "off":
        sim_matrix = rescale_similarity(
            sim_matrix,
            method=rescale_method,
            rescale_min=rescale_min,
            rescale_max=rescale_max,
            lower_percentile=rescale_lower_percentile,
            upper_percentile=rescale_upper_percentile,
            sigmoid_center=rescale_sigmoid_center,
            sigmoid_steepness=rescale_sigmoid_steepness,
            temperature=rescale_temperature,
        )

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
    # rescaling parameters
    rescale_method: RescaleMethod = "clip",
    rescale_min: float = 0.5,
    rescale_max: float = 0.9,
    rescale_lower_percentile: float = 5.0,
    rescale_upper_percentile: float = 95.0,
    rescale_sigmoid_center: float = 0.7,
    rescale_sigmoid_steepness: float = 10.0,
    rescale_temperature: float = 0.5,
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
        rescale_method=rescale_method,
        rescale_min=rescale_min,
        rescale_max=rescale_max,
        rescale_lower_percentile=rescale_lower_percentile,
        rescale_upper_percentile=rescale_upper_percentile,
        rescale_sigmoid_center=rescale_sigmoid_center,
        rescale_sigmoid_steepness=rescale_sigmoid_steepness,
        rescale_temperature=rescale_temperature,
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
