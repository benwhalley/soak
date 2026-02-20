"""HyDE and paraphrase-based embeddings for theme coverage analysis.

Generates synthetic quotes or paraphrases to improve semantic matching with corpus chunks.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Paths to prompt templates
HYDE_TEMPLATE_PATH = Path(__file__).parent / "templates" / "hyde_quotes.sd"
PARAPHRASE_TEMPLATE_PATH = Path(__file__).parent / "templates" / "paraphrase_quote.sd"


async def _generate_quotes_for_theme(
    theme_text: str,
    n_quotes: int,
    model_name: str,
    credentials: "LLMCredentials",
) -> List[str]:
    """Generate synthetic quotes for a single theme using LLM.

    Args:
        theme_text: The theme text (name: description format)
        n_quotes: Number of quotes to generate
        model_name: LLM model name
        credentials: LLM credentials

    Returns:
        List of n_quotes synthetic participant quotes
    """
    from jinja2 import StrictUndefined, Template
    from struckdown import LLM, chatter_async

    prompt_template = HYDE_TEMPLATE_PATH.read_text()
    template = Template(prompt_template, undefined=StrictUndefined)
    prompt = template.render(theme_text=theme_text, n_quotes=n_quotes)

    llm = LLM(model_name=model_name)

    try:
        result = await chatter_async(
            multipart_prompt=prompt,
            model=llm,
            credentials=credentials,
        )

        if hasattr(result, "outputs") and "quotes" in result.outputs:
            quotes = result.outputs["quotes"]
            if isinstance(quotes, list):
                return quotes[:n_quotes]

        logger.warning(
            f"HyDE generation returned unexpected format for theme: {theme_text[:50]}..."
        )
        return [theme_text]  # fallback to original theme text

    except Exception as e:
        logger.warning(
            f"HyDE generation failed for theme: {theme_text[:50]}... Error: {e}"
        )
        return [theme_text]  # fallback to original theme text


async def generate_hyde_quotes(
    theme_texts: List[str],
    n_quotes: int = 5,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    """Generate HyDE quotes for multiple themes.

    For each theme, generates n_quotes synthetic participant quotes that
    represent what someone might say in an interview about that theme.

    Args:
        theme_texts: List of theme strings (e.g., "name: description")
        n_quotes: Number of quotes per theme (default: 5)
        model_name: LLM model name (default: gpt-4.1-mini)
        api_key: API key (uses LLM_API_KEY env var if not provided)
        base_url: API base URL (uses LLM_API_BASE env var if not provided)

    Returns:
        Tuple of:
        - List of n_themes lists, each containing n_quotes strings
        - Metadata dict with model_name, n_quotes, etc.
    """
    from struckdown import LLMCredentials

    if model_name is None:
        model_name = "gpt-4.1-mini"

    if api_key is not None:
        credentials = LLMCredentials(api_key=api_key, base_url=base_url)
    else:
        credentials = LLMCredentials()

    tasks = [
        _generate_quotes_for_theme(text, n_quotes, model_name, credentials)
        for text in theme_texts
    ]
    results = await asyncio.gather(*tasks)

    metadata = {
        "model_name": model_name,
        "n_quotes": n_quotes,
        "n_themes": len(theme_texts),
    }

    return list(results), metadata


def compute_hyde_similarity_matrix(
    hyde_quotes: List[List[str]],
    chunk_embeddings: np.ndarray,
    chunk_doc_indices: List[int],
    n_docs: int,
    embedding_model: str = "text-embedding-3-large",
) -> np.ndarray:
    """Compute similarity matrix using HyDE quotes.

    For each theme:
    1. Embed all HyDE quotes for that theme
    2. Compute similarity between each HyDE quote and all chunks
    3. For each chunk, take max similarity across all HyDE quotes
    4. For each document, aggregate chunk similarities (max)

    Args:
        hyde_quotes: List of quote lists, one per theme
        chunk_embeddings: Pre-computed chunk embeddings (n_chunks x dim)
        chunk_doc_indices: Document index for each chunk
        n_docs: Total number of documents
        embedding_model: Model for embedding HyDE quotes

    Returns:
        Similarity matrix of shape (n_themes, n_docs)
    """
    from soak.models.base import get_embedding

    n_themes = len(hyde_quotes)
    result_matrix = np.zeros((n_themes, n_docs))

    for theme_idx, quotes in enumerate(hyde_quotes):
        if not quotes:
            continue

        # Embed all quotes for this theme
        quote_embeddings = np.array(get_embedding(quotes, model=embedding_model))

        # Compute similarity between each quote and all chunks
        # Shape: (n_quotes, n_chunks)
        quote_chunk_sims = cosine_similarity(quote_embeddings, chunk_embeddings)

        # For each chunk, take max similarity across all quotes
        # Shape: (n_chunks,)
        max_quote_sims = np.max(quote_chunk_sims, axis=0)

        # Aggregate by document (max similarity per document)
        for chunk_idx, doc_idx in enumerate(chunk_doc_indices):
            current_max = result_matrix[theme_idx, doc_idx]
            result_matrix[theme_idx, doc_idx] = max(current_max, max_quote_sims[chunk_idx])

    return result_matrix


def get_theme_embedding_texts(
    themes: List[Dict[str, Any]],
    mode: str = "theme_text",
    hyde_quotes: Optional[List[List[str]]] = None,
) -> List[List[str]]:
    """Get embedding texts for themes based on mode.

    Args:
        themes: List of theme dicts with 'name' and 'description' keys
        mode: Embedding mode:
            - "theme_text": Just theme name + description
            - "theme_and_codes": Theme text + associated code descriptions
            - "hyde": Return None (HyDE uses separate quote embeddings)
        hyde_quotes: Pre-generated HyDE quotes (only used if mode="hyde")

    Returns:
        List of lists of strings to embed per theme.
        For theme_text/theme_and_codes: single-element lists
        For hyde: the pre-generated quotes
    """
    result = []

    for i, theme in enumerate(themes):
        name = theme.get("name", "")
        description = theme.get("description", "")

        if mode == "theme_text":
            text = f"{name}: {description}" if description else name
            result.append([text])

        elif mode == "theme_and_codes":
            parts = [f"{name}: {description}" if description else name]
            codes = theme.get("codes", [])
            for code in codes:
                if isinstance(code, dict):
                    code_name = code.get("name", "")
                    code_desc = code.get("description", "")
                    if code_desc:
                        parts.append(f"{code_name}: {code_desc}")
                    elif code_name:
                        parts.append(code_name)
                elif isinstance(code, str):
                    parts.append(code)
            # Join all parts into one text for embedding
            result.append([" | ".join(parts)])

        elif mode == "hyde":
            if hyde_quotes and i < len(hyde_quotes):
                result.append(hyde_quotes[i])
            else:
                # Fallback to theme text
                text = f"{name}: {description}" if description else name
                result.append([text])

        else:
            # Default to theme_text
            text = f"{name}: {description}" if description else name
            result.append([text])

    return result


# --- Paraphrased Quotes Functions ---


async def _paraphrase_quote(
    quote_text: str,
    n_paraphrases: int,
    model_name: str,
    credentials: "LLMCredentials",
) -> List[str]:
    """Generate paraphrases for a single quote using LLM.

    Args:
        quote_text: The quote to paraphrase
        n_paraphrases: Number of paraphrases to generate
        model_name: LLM model name
        credentials: LLM credentials

    Returns:
        List of n_paraphrases alternative phrasings
    """
    from jinja2 import StrictUndefined, Template
    from struckdown import LLM, chatter_async

    prompt_template = PARAPHRASE_TEMPLATE_PATH.read_text()
    template = Template(prompt_template, undefined=StrictUndefined)
    prompt = template.render(quote_text=quote_text, n_paraphrases=n_paraphrases)

    llm = LLM(model_name=model_name)

    try:
        result = await chatter_async(
            multipart_prompt=prompt,
            model=llm,
            credentials=credentials,
        )

        if hasattr(result, "outputs") and "paraphrases" in result.outputs:
            paraphrases = result.outputs["paraphrases"]
            if isinstance(paraphrases, list):
                return paraphrases[:n_paraphrases]

        logger.warning(
            f"Paraphrase generation returned unexpected format for quote: {quote_text[:50]}..."
        )
        return [quote_text]  # fallback to original

    except Exception as e:
        logger.warning(
            f"Paraphrase generation failed for quote: {quote_text[:50]}... Error: {e}"
        )
        return [quote_text]  # fallback to original


def extract_quotes_from_themes(
    themes: List[Dict[str, Any]],
    max_quotes_per_theme: int = 10,
) -> List[List[str]]:
    """Extract quotes from codes attached to each theme.

    Args:
        themes: List of theme dicts with 'codes_with_quotes' or 'codes' containing quote lists
        max_quotes_per_theme: Maximum quotes to extract per theme (default: 10)

    Returns:
        List of quote lists, one per theme (limited to max_quotes_per_theme each)
    """
    result = []

    for theme in themes:
        theme_quotes = []
        # Use codes_with_quotes if available (Django model property), otherwise fall back to codes
        codes = theme.get("codes_with_quotes", theme.get("codes", []))

        for code in codes:
            if isinstance(code, dict):
                # Extract quotes from code
                quotes = code.get("quotes", [])
                for quote in quotes:
                    if isinstance(quote, dict):
                        # Quote might be a dict with 'text' key
                        quote_text = quote.get("text", quote.get("quote", ""))
                    else:
                        quote_text = str(quote)

                    if quote_text and quote_text.strip():
                        theme_quotes.append(quote_text.strip())
                        # Stop if we've reached the limit for this theme
                        if len(theme_quotes) >= max_quotes_per_theme:
                            break
            if len(theme_quotes) >= max_quotes_per_theme:
                break

        result.append(theme_quotes)

    return result


async def generate_paraphrased_quotes(
    themes: List[Dict[str, Any]],
    n_paraphrases: int = 2,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    """Generate paraphrases of quotes from theme codes.

    For each theme:
    1. Extract all quotes from attached codes
    2. Generate n_paraphrases for each quote
    3. Return flattened list of all paraphrases (not originals)

    Args:
        themes: List of theme dicts with 'codes' containing quotes
        n_paraphrases: Number of paraphrases per quote (default: 3)
        model_name: LLM model name (default: gpt-4.1-mini)
        api_key: API key (uses LLM_API_KEY env var if not provided)
        base_url: API base URL (uses LLM_API_BASE env var if not provided)

    Returns:
        Tuple of:
        - List of paraphrase lists, one per theme (all paraphrases flattened)
        - Metadata dict with counts and details
    """
    from struckdown import LLMCredentials

    if model_name is None:
        model_name = "gpt-4.1-mini"

    if api_key is not None:
        credentials = LLMCredentials(api_key=api_key, base_url=base_url)
    else:
        credentials = LLMCredentials()

    # Extract quotes from themes
    theme_quotes = extract_quotes_from_themes(themes)

    # Build flat list of all tasks with theme indices for reassembly
    all_tasks = []
    task_theme_indices = []  # Track which theme each task belongs to

    for theme_idx, quotes in enumerate(theme_quotes):
        for quote in quotes:
            all_tasks.append(
                _paraphrase_quote(quote, n_paraphrases, model_name, credentials)
            )
            task_theme_indices.append(theme_idx)

    # Run ALL paraphrase generation concurrently
    if all_tasks:
        all_results = await asyncio.gather(*all_tasks)
    else:
        all_results = []

    # Reassemble results by theme
    all_paraphrases = [[] for _ in themes]
    for task_idx, paraphrases in enumerate(all_results):
        theme_idx = task_theme_indices[task_idx]
        all_paraphrases[theme_idx].extend(paraphrases)

    total_quotes = sum(len(q) for q in theme_quotes)
    total_paraphrases = sum(len(p) for p in all_paraphrases)

    metadata = {
        "model_name": model_name,
        "n_paraphrases": n_paraphrases,
        "n_themes": len(themes),
        "total_quotes": total_quotes,
        "total_paraphrases": total_paraphrases,
        "quotes_per_theme": [len(q) for q in theme_quotes],
    }

    logger.info(
        f"Generated {total_paraphrases} paraphrases from {total_quotes} quotes "
        f"across {len(themes)} themes (parallel)"
    )

    return all_paraphrases, metadata
