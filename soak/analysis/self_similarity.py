"""Self-similarity analysis for qualitative analysis themes.

Computes theme-to-theme similarity matrices using embeddings.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# default embedding model for similarity computation
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_theme_embedding_texts(
    themes: List[Dict[str, Any]],
    mode: str = "theme_text",
) -> List[Tuple[str, List[str]]]:
    """Get embedding texts for each theme based on mode.

    Args:
        themes: List of theme dicts with 'name', 'description', 'codes' keys
        mode: Embedding mode:
            - "theme_text": Theme name + description
            - "theme_and_codes": Theme + code descriptions

    Returns:
        List of (theme_name, [embedding_texts]) tuples.
        Each theme may have multiple texts to embed (for averaging).
    """
    result = []

    for t in themes:
        name = t.get("name", "")
        description = t.get("description", "")

        if mode == "theme_text":
            embedding_text = f"{name}: {description}" if description else name
            result.append((name, [embedding_text]))

        elif mode == "theme_and_codes":
            texts = []
            # add theme text
            theme_text = f"{name}: {description}" if description else name
            texts.append(theme_text)
            # add code descriptions
            codes = t.get("codes", [])
            for code in codes:
                if isinstance(code, dict):
                    code_name = code.get("name", "")
                    code_desc = code.get("description", "")
                    if code_desc:
                        texts.append(f"{code_name}: {code_desc}")
                    elif code_name:
                        texts.append(code_name)
                elif isinstance(code, str):
                    texts.append(code)
            result.append((name, texts))

        else:
            # default to theme_text
            embedding_text = f"{name}: {description}" if description else name
            result.append((name, [embedding_text]))

    return result


def _get_embeddings(texts: List[str], model: str) -> List[List[float]]:
    """Get embeddings for texts using struckdown."""
    from struckdown import get_embedding

    embeddings = get_embedding(texts, model=model)
    return [list(e) for e in embeddings]


def _cosine_to_angular(cosine_sim: float) -> float:
    """Convert cosine similarity to angular similarity.

    Angular similarity maps cosine to 0-1 scale satisfying triangle inequality.
    """
    angle = np.degrees(np.arccos(np.clip(cosine_sim, -1.0, 1.0)))
    return 1 - angle / 180.0


def _calibrate_similarities(
    cosine_sims: List[float],
    embedding_model: str,
) -> List[float]:
    """Calibrate cosine similarity scores using bundled calibration.

    Converts cosine similarity to angular similarity, then applies
    calibration model if available.

    Args:
        cosine_sims: List of raw cosine similarities (-1 to 1)
        embedding_model: Embedding model name for calibration lookup

    Returns:
        List of calibrated scores (0-1)
    """
    if not cosine_sims:
        return []

    try:
        from ..calibration import calibrate, get_bundled_calibration, load_calibration

        cal_path = get_bundled_calibration(embedding_model)
        if cal_path is None:
            logger.debug(f"No bundled calibration for {embedding_model}, using clamped cosine")
            return [max(0, min(1, s)) for s in cosine_sims]

        model, metadata = load_calibration(cal_path, embedding_model)
        method = metadata.get("method", "scam")

        # convert all to angular similarity
        angular_sims = np.array([_cosine_to_angular(s) for s in cosine_sims])

        # apply calibration
        calibrated = calibrate(angular_sims, model, method=method)
        return [float(c) for c in calibrated]

    except ImportError:
        logger.debug("Calibration module not available, using clamped cosine")
        return [max(0, min(1, s)) for s in cosine_sims]
    except Exception as e:
        logger.warning(f"Calibration failed: {e}, using clamped cosine")
        return [max(0, min(1, s)) for s in cosine_sims]


def compute_self_similarity_matrix(
    themes: List[Dict[str, Any]],
    embedding_mode: str = "theme_text",
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute theme self-similarity matrix (theme A vs theme B).

    Embeds all themes and computes pairwise cosine similarity between them.
    The result is a symmetric matrix showing how similar themes are to each other.

    Args:
        themes: List of theme dicts with 'name', 'description', 'codes' keys
        embedding_mode: How to create theme embeddings:
            - "theme_text": Theme name + description (default)
            - "theme_and_codes": Theme + code descriptions
        embedding_model: Model to use for embeddings (default: text-embedding-3-small)

    Returns:
        Dictionary with:
        - themes: List of theme names
        - short_labels: Short labels (T1, T2, etc.)
        - raw_matrix: [n_themes x n_themes] raw cosine similarity matrix
        - calibrated_matrix: [n_themes x n_themes] calibrated similarity matrix
        - embedding_model: Model used for embeddings
        - embedding_mode: Mode used for embeddings
        - n_themes: Number of themes
    """
    if not themes:
        return _empty_self_similarity_result()

    if embedding_model is None:
        embedding_model = DEFAULT_EMBEDDING_MODEL

    theme_names = [t.get("name", "") for t in themes]
    short_labels = [f"T{i+1}" for i in range(len(themes))]

    # get theme embedding texts based on mode
    theme_data = get_theme_embedding_texts(themes, mode=embedding_mode)
    if not theme_data:
        return _empty_self_similarity_result()

    # build theme embeddings (average if multiple texts)
    theme_embeddings = []
    for name, texts in theme_data:
        if len(texts) == 1:
            emb = np.array(_get_embeddings(texts, model=embedding_model)[0])
        else:
            # embed all texts and average
            embs = np.array(_get_embeddings(texts, model=embedding_model))
            emb = embs.mean(axis=0)
            # normalise the averaged embedding
            emb = emb / np.linalg.norm(emb)
        theme_embeddings.append(emb)

    theme_embeddings = np.array(theme_embeddings)
    n_themes = len(theme_names)

    # compute pairwise cosine similarity
    raw_matrix = cosine_similarity(theme_embeddings, theme_embeddings)

    # calibrate the raw similarities
    raw_flat = raw_matrix.flatten().tolist()
    calibrated_flat = _calibrate_similarities(raw_flat, embedding_model)
    calibrated_matrix = np.array(calibrated_flat).reshape(n_themes, n_themes)

    return {
        "themes": theme_names,
        "short_labels": short_labels,
        "raw_matrix": raw_matrix.tolist(),
        "calibrated_matrix": calibrated_matrix.tolist(),
        "embedding_model": embedding_model,
        "embedding_mode": embedding_mode,
        "n_themes": int(n_themes),
    }


def _empty_self_similarity_result() -> Dict[str, Any]:
    """Return empty self-similarity result."""
    return {
        "themes": [],
        "short_labels": [],
        "raw_matrix": [],
        "calibrated_matrix": [],
        "embedding_model": "",
        "embedding_mode": "theme_text",
        "n_themes": 0,
    }


def compute_self_similarity_from_analysis(
    analysis: Union["QualitativeAnalysis", Dict[str, Any]],
    embedding_mode: str = "theme_text",
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute self-similarity matrix from a QualitativeAnalysis object or dict.

    Convenience wrapper around compute_self_similarity_matrix that accepts
    QualitativeAnalysis objects or dicts from JSON deserialization.

    Args:
        analysis: QualitativeAnalysis object or dict with 'themes' key
        embedding_mode: How to create theme embeddings
        embedding_model: Model to use for embeddings

    Returns:
        Self-similarity result dict
    """
    if hasattr(analysis, "themes"):
        # QualitativeAnalysis or similar object
        themes = analysis.themes
        if hasattr(themes, "model_dump"):
            themes = [t.model_dump() for t in themes]
        elif not isinstance(themes, list):
            themes = list(themes)
    elif isinstance(analysis, dict):
        themes = analysis.get("themes", [])
    else:
        themes = []

    return compute_self_similarity_matrix(
        themes=themes,
        embedding_mode=embedding_mode,
        embedding_model=embedding_model,
    )
