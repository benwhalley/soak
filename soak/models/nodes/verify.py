"""VerifyQuotes node for validating quotes against source documents."""

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple,
                    Union)

if TYPE_CHECKING:
    from soak.models.progress import ProgressManager

import anyio
import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from struckdown import ChatterResult, LLMError, chatter_async
from struckdown.parsing import parse_syntax
from tqdm import tqdm

from soak.error_handlers import (ErrorBehavior, get_error_behavior,
                                 log_error_to_stderr, managed_llm_call,
                                 should_continue_pipeline)
from soak.models.alignment import trim_span_to_quote
from soak.models.base import (TrackedItem, get_action_lookup,
                              get_embedding_async, memory, safe_json_dump,
                              semaphore)
from soak.models.text_utils import (ELLIPSIS_RE, create_document_boundaries,
                                    extract_context_window,
                                    find_source_document, is_match_truncated,
                                    make_windows, snap_to_boundaries)

from .base import CompletionDAGNode

logger = logging.getLogger(__name__)

# fast regex tokenizer for BM25 (replaces slow nltk.word_tokenize)
_WORD_RE = re.compile(r"\w+")


def _fast_tokenize(text: str) -> List[str]:
    r"""Fast regex tokenizer for BM25.

    Uses \w+ pattern which:
    - Strips all punctuation (appropriate for BM25)
    - Splits contractions: "don't" -> ["don", "t"]
    - Splits hyphenated: "well-known" -> ["well", "known"]
    """
    return _WORD_RE.findall(text.lower())


def _hash_bm25_inputs(window_texts: List[str], k1: float, b: float) -> str:
    """Create a stable hash for BM25 cache key."""
    content = "\n".join(window_texts) + f"|k1={k1}|b={b}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@memory.cache
def _cached_build_bm25_index(
    corpus_hash: str,
    window_texts: List[str],
    bm25_k1: float,
    bm25_b: float,
) -> Tuple[List[List[str]], Dict[str, set]]:
    """Cached BM25 index building - tokenizes windows and builds inverted index.

    Returns tokenized windows and token index. BM25Okapi is recreated from
    tokenized since it's not picklable.
    """
    logger.info(f"BM25 cache miss - building index (hash={corpus_hash})")
    tokenized = [_fast_tokenize(w) for w in window_texts]
    token_index = _build_token_index(tokenized)
    return tokenized, token_index


def _build_token_index(tokenized: List[List[str]]) -> Dict[str, set]:
    """Build inverted index: token -> set of window indices.

    Args:
        tokenized: List of tokenized windows (each window is a list of tokens)

    Returns:
        Dict mapping each token to set of window indices containing it
    """
    from collections import defaultdict

    index: Dict[str, set] = defaultdict(set)
    for idx, tokens in enumerate(tokenized):
        for token in set(tokens):  # dedupe tokens per window
            index[token].add(idx)
    return dict(index)


def _find_candidate_windows(
    query_tokens: List[str],
    token_index: Dict[str, set],
    num_windows: int,
) -> List[int]:
    """Find windows containing any query token (union).

    Args:
        query_tokens: Tokenized query
        token_index: Inverted index from _build_token_index
        num_windows: Total number of windows (fallback if no matches)

    Returns:
        Sorted list of candidate window indices
    """
    if not query_tokens:
        return list(range(num_windows))  # fallback to all

    candidates: set = set()
    for token in query_tokens:
        if token in token_index:
            candidates.update(token_index[token])

    return sorted(candidates) if candidates else list(range(num_windows))


# verification result models


@dataclass
class LLMVerdict:
    """Result from LLM-based existence check."""

    is_contained: Optional[bool]
    explanation: str
    chatter_result: Optional[Any] = None


@dataclass
class LLMFairnessVerdict:
    """Result from LLM-based fairness check."""

    is_fair: Optional[bool]
    explanation: str
    chatter_result: Optional[Any] = None


@dataclass
class QuoteVerificationResult:
    """Comprehensive verification result for a single quote.

    Includes BM25 matching, embedding similarity, optional LLM checks,
    and source attribution.
    """

    # original quote
    quote: str
    quote_hash: str

    # source attribution
    source_doc: str
    source_doc_id: str  # reference to shared storage (not full content!)

    # match location
    span_text: str
    global_start: int
    global_end: int

    # matching metrics
    bm25_score: float
    bm25_ratio: float  # top1/top2 ratio (uniqueness)
    cosine_similarity: float
    match_ratio: Optional[float] = None  # fuzzy match confidence

    # optional code/theme associations
    code_slug: Optional[str] = None
    theme_slug: Optional[str] = None
    code_name: Optional[str] = None
    code_description: Optional[str] = None
    theme_name: Optional[str] = None
    theme_description: Optional[str] = None

    # context snippets (short extracts, not full document)
    context_before: str = ""
    context_after: str = ""

    # source line numbers
    source_line_start: Optional[int] = None
    source_line_end: Optional[int] = None

    # LLM verdicts
    llm_existence: Optional[LLMVerdict] = None
    llm_fairness: Optional[LLMFairnessVerdict] = None

    # metadata
    alignment_method: str = "hybrid"
    matched_window_indices: List[int] = field(default_factory=list)
    verification_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backwards compatibility and export."""
        result = {
            "quote": self.quote,
            "quote_hash": self.quote_hash,
            "source_doc": self.source_doc,
            "source_doc_id": self.source_doc_id,
            "span_text": self.span_text,
            "global_start": self.global_start,
            "global_end": self.global_end,
            "bm25_score": self.bm25_score,
            "bm25_ratio": self.bm25_ratio,
            "cosine_similarity": self.cosine_similarity,
            "match_ratio": self.match_ratio,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "alignment_method": self.alignment_method,
        }

        # add optional fields
        if self.code_slug:
            result["code_slug"] = self.code_slug
        if self.theme_slug:
            result["theme_slug"] = self.theme_slug
        if self.code_name:
            result["code_name"] = self.code_name
        if self.code_description:
            result["code_description"] = self.code_description
        if self.theme_name:
            result["theme_name"] = self.theme_name
        if self.theme_description:
            result["theme_description"] = self.theme_description
        if self.source_line_start is not None:
            result["source_line_start"] = self.source_line_start
        if self.source_line_end is not None:
            result["source_line_end"] = self.source_line_end

        # add LLM verdicts
        if self.llm_existence:
            # fill missing values with True because we don't check all of them if we definitly found the quote using bM25
            result["llm_is_contained"] = self.llm_existence.is_contained.fillna(True)
            result["llm_explanation"] = self.llm_existence.explanation
        if self.llm_fairness:
            result["llm_is_fair"] = self.llm_fairness.is_fair
            result["llm_fairness_explanation"] = self.llm_fairness.explanation

        return result


@dataclass
class VerificationSummary:
    """Aggregate verification statistics for a Code or Theme."""

    total_quotes: int
    mean_bm25_score: float
    mean_cosine_similarity: float
    mean_match_ratio: float
    failed_quotes: int  # quotes below threshold
    llm_existence_checks: int = 0
    llm_existence_failures: int = 0
    llm_fairness_checks: int = 0
    llm_fairness_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_quotes": self.total_quotes,
            "mean_bm25_score": self.mean_bm25_score,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "mean_match_ratio": self.mean_match_ratio,
            "failed_quotes": self.failed_quotes,
            "llm_existence_checks": self.llm_existence_checks,
            "llm_existence_failures": self.llm_existence_failures,
            "llm_fairness_checks": self.llm_fairness_checks,
            "llm_fairness_failures": self.llm_fairness_failures,
        }


# core verification functions


def _calculate_bm25_ratio(scores: np.ndarray) -> Tuple[float, float, float]:
    """Calculate BM25 top1, top2, and ratio from score array.

    Args:
        scores: Array of BM25 scores

    Returns:
        Tuple of (top1, top2, ratio) where ratio = top1 / (top2 + epsilon)
    """
    sorted_scores = np.sort(scores)[::-1]
    top1 = float(sorted_scores[0]) if len(sorted_scores) > 0 else 0.0
    top2 = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    ratio = top1 / (top2 + 1e-6)
    return top1, top2, ratio


def _expand_window(
    best_idx: int,
    original_windows: List[Tuple[str, int, int]],
    original_text: str,
    expand_neighbors: int,
) -> Tuple[str, int]:
    """Expand window to include neighbors and return expanded span.

    Args:
        best_idx: Index of best matching window
        original_windows: List of (window_text, start_pos, end_pos) tuples
        original_text: Full concatenated text
        expand_neighbors: Number of neighbors to include on each side

    Returns:
        Tuple of (expanded_span, start_pos)
    """
    if expand_neighbors > 0:
        start_idx = max(0, best_idx - expand_neighbors)
        end_idx = min(len(original_windows), best_idx + expand_neighbors + 1)
        expanded_start_pos = original_windows[start_idx][1]
        expanded_end_pos = original_windows[end_idx - 1][2]
        return original_text[expanded_start_pos:expanded_end_pos], expanded_start_pos
    else:
        span, start_pos, _ = original_windows[best_idx]
        return span, start_pos


def _trim_and_position(
    quote: str,
    span_text: str,
    window_start_pos: int,
    trim_spans: bool,
    trim_method: str,
    min_fuzzy_ratio: float,
) -> Tuple[str, int, int, Optional[float]]:
    """Apply span trimming if enabled and return positioning info.

    Args:
        quote: Quote text to match
        span_text: Span text to trim
        window_start_pos: Global start position of window
        trim_spans: Whether to apply trimming
        trim_method: Method to use for trimming
        min_fuzzy_ratio: Minimum fuzzy match ratio

    Returns:
        Tuple of (span_text, global_start, global_end, match_ratio)
    """
    if trim_spans:
        match_result = trim_span_to_quote(
            quote, span_text, trim_method, min_fuzzy_ratio
        )
        return (
            match_result["matched_text"],
            window_start_pos + match_result["start_char"],
            window_start_pos + match_result["end_char"],
            match_result["match_ratio"],
        )
    else:
        return (span_text, window_start_pos, window_start_pos + len(span_text), None)


def _extract_llm_field(result, field_name: str, default=None):
    """Extract a field from ChatterResult with safe attribute access.

    Args:
        result: ChatterResult object
        field_name: Name of field to extract
        default: Default value if field not found

    Returns:
        Field output value or default
    """
    field_obj = result.results.get(field_name, {})
    return field_obj.output if hasattr(field_obj, "output") else default


async def verify_quotes_bm25_first(
    extracted_quotes: List["Quote"],
    original_windows: List[Tuple[str, int, int]],
    original_text: str,
    doc_boundaries: List[Tuple[str, int, int]],
    doc_content_map: Dict[str, str],
    get_embedding_async,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.4,
    ellipsis_max_gap: Optional[int] = 300,
    trim_spans: bool = True,
    trim_method: Literal["fuzzy", "sliding_bm25", "hybrid"] = "hybrid",
    min_fuzzy_ratio: float = 0.6,
    expand_window_neighbors: int = 1,
    show_progress: bool = True,
    progress_manager: Optional["ProgressManager"] = None,
) -> List[Dict[str, Any]]:
    """Verify quotes using BM25-first matching with ellipsis support.

    For each quote:
    - If no ellipsis: find best BM25 window
    - If ellipsis: match head/tail separately, reconstruct span
    - Always compute embedding similarity on combined span
    - Return BM25 score, ratio (top1/top2, indicating whether there is a single clear match for the quote), cosine similarity, hash, and full source document content

    Args:
        extracted_quotes: List of Quote objects to verify
        original_windows: List of (window_text, start_pos, end_pos) tuples
        original_text: Full concatenated source text
        doc_boundaries: List of (doc_name, start_pos, end_pos) for document tracking
        doc_content_map: Dict mapping doc_name to full document content
        show_progress: Whether to show progress bars
    """

    if not extracted_quotes:
        return []

    # Extract window texts and build BM25 index with inverted index for fast pre-filtering
    window_texts = [w[0] for w in original_windows]

    # Use cached tokenization and token index
    corpus_hash = _hash_bm25_inputs(window_texts, bm25_k1, bm25_b)
    logger.info(
        f"Building BM25 index from {len(original_windows)} windows (hash={corpus_hash})..."
    )
    tokenized, token_index = _cached_build_bm25_index(
        corpus_hash, window_texts, bm25_k1, bm25_b
    )

    # BM25Okapi isn't picklable, so recreate from cached tokenized data
    bm25 = BM25Okapi(tokenized, k1=bm25_k1, b=bm25_b)
    logger.info(
        f"Built BM25 index and inverted index ({len(token_index)} unique tokens)"
    )

    # Build mapping: doc_name -> list of window indices for per-document filtering
    from collections import defaultdict

    doc_to_windows: Dict[str, List[int]] = defaultdict(list)
    for idx, (_, start, end) in enumerate(original_windows):
        # Find which document this window belongs to
        for doc_name, doc_start, doc_end in doc_boundaries:
            if doc_start <= start < doc_end:
                doc_to_windows[doc_name].append(idx)
                break
    logger.info(f"Built per-document window index: {len(doc_to_windows)} docs")
    logger.info(f"Sample doc names in index: {list(doc_to_windows.keys())[:5]}")

    # Build base ID -> doc name mapping (handles any extension)
    # e.g., "tSR_rftWR44" -> "tSR_rftWR44.txt"
    base_id_to_doc: Dict[str, str] = {}
    for doc_name in doc_to_windows:
        # Extract base (filename without extension)
        base = doc_name.rsplit(".", 1)[0] if "." in doc_name else doc_name
        base_id_to_doc[base] = doc_name

    # Phase 1: BM25 matching (no embeddings yet)
    # Collect intermediate results with texts that need embedding
    logger.debug(f"Phase 1: BM25 matching for {len(extracted_quotes)} quotes...")
    if extracted_quotes:
        sample_sources = [q.source for q in extracted_quotes[:5] if q.source]
        logger.debug(f"Sample quote sources: {sample_sources}")
    intermediate_results = []

    # Create progress bar for BM25 matching
    if show_progress:
        if progress_manager:
            pbar_bm25 = progress_manager.create_progress_bar(
                total=len(extracted_quotes),
                desc="VerifyQuotes: BM25 matching",
                unit="quote",
            )
        else:
            desc = "VerifyQuotes: BM25 matching".ljust(35)
            pbar_bm25 = tqdm(
                total=len(extracted_quotes),
                desc=desc,
                unit="quote",
                file=sys.stderr,
                ncols=120,
                leave=True,
                mininterval=0.1,
            )
    else:
        pbar_bm25 = None

    for i, quote_obj in enumerate(extracted_quotes):
        if pbar_bm25:
            pbar_bm25.update(1)
        quote = quote_obj.text
        has_ellipsis = bool(ELLIPSIS_RE.search(quote))

        # Get windows for this quote's source document only (major performance win)
        quote_source = quote_obj.source
        quote_source_base = None
        if quote_source:
            # Strip __chunks__N suffix to get base ID
            base = re.sub(r"__chunks__\d+$", "", quote_source)
            # Look up actual doc name (handles extensions)
            quote_source_base = base_id_to_doc.get(base)

        if quote_source_base and quote_source_base in doc_to_windows:
            doc_window_indices = doc_to_windows[quote_source_base]
            doc_window_set = set(doc_window_indices)
        else:
            # Fallback: search all windows if source unknown
            doc_window_indices = list(range(len(original_windows)))
            doc_window_set = set(doc_window_indices)
            if quote_source:
                base = re.sub(r"__chunks__\d+$", "", quote_source)
                logger.warning(
                    f"Quote source '{quote_source}' (base: '{base}') not in doc_to_windows, searching all"
                )

        if not has_ellipsis:
            # Simple case: single BM25 match with candidate pre-filtering
            query_tokens = _fast_tokenize(quote)
            candidate_indices = _find_candidate_windows(
                query_tokens, token_index, len(original_windows)
            )

            # Intersect token candidates with document windows
            doc_candidates = [i for i in candidate_indices if i in doc_window_set]
            if not doc_candidates:
                doc_candidates = doc_window_indices  # Fallback to all doc windows

            # Score only candidate windows (major performance win)
            if len(doc_candidates) < len(original_windows):
                candidate_scores = bm25.get_batch_scores(query_tokens, doc_candidates)
                best_local_idx = int(np.argmax(candidate_scores))
                best_idx = doc_candidates[best_local_idx]
                # Build full scores array for ratio calculation
                scores = np.zeros(len(original_windows))
                for i, idx in enumerate(doc_candidates):
                    scores[idx] = candidate_scores[i]
            else:
                scores = bm25.get_scores(query_tokens)
                best_idx = int(np.argmax(scores))

            top1, top2, ratio = _calculate_bm25_ratio(scores)
            bm25_score = top1

            # Extract span with optional expansion
            span_text, window_start_pos = _expand_window(
                best_idx, original_windows, original_text, expand_window_neighbors
            )
            trim_target = quote

        else:
            # Ellipsis case: match head and tail with candidate pre-filtering
            parts = [p.strip() for p in ELLIPSIS_RE.split(quote) if p.strip()]
            if len(parts) < 2:
                parts = [quote.replace("...", "").replace("…", "").strip()]

            head = parts[0]
            tail = parts[-1]

            # BM25 for head with candidate pre-filtering
            head_tokens = _fast_tokenize(head)
            head_candidates = _find_candidate_windows(
                head_tokens, token_index, len(original_windows)
            )
            # Intersect token candidates with document windows
            head_doc_candidates = [i for i in head_candidates if i in doc_window_set]
            if not head_doc_candidates:
                head_doc_candidates = doc_window_indices  # Fallback to all doc windows

            if len(head_doc_candidates) < len(original_windows):
                head_candidate_scores = bm25.get_batch_scores(
                    head_tokens, head_doc_candidates
                )
                head_local_idx = int(np.argmax(head_candidate_scores))
                head_idx = head_doc_candidates[head_local_idx]
                head_scores = np.zeros(len(original_windows))
                for i, idx in enumerate(head_doc_candidates):
                    head_scores[idx] = head_candidate_scores[i]
            else:
                head_scores = bm25.get_scores(head_tokens)
                head_idx = int(np.argmax(head_scores))
            head_score = float(head_scores[head_idx])

            # BM25 for tail with candidate pre-filtering
            tail_tokens = _fast_tokenize(tail)
            tail_candidates = _find_candidate_windows(
                tail_tokens, token_index, len(original_windows)
            )
            # Intersect token candidates with document windows
            tail_doc_candidates = [i for i in tail_candidates if i in doc_window_set]
            if not tail_doc_candidates:
                tail_doc_candidates = doc_window_indices  # Fallback to all doc windows

            if len(tail_doc_candidates) < len(original_windows):
                tail_candidate_scores = bm25.get_batch_scores(
                    tail_tokens, tail_doc_candidates
                )
                tail_local_idx = int(np.argmax(tail_candidate_scores))
                tail_idx = tail_doc_candidates[tail_local_idx]
                tail_scores = np.zeros(len(original_windows))
                for i, idx in enumerate(tail_doc_candidates):
                    tail_scores[idx] = tail_candidate_scores[i]
            else:
                tail_scores = bm25.get_scores(tail_tokens)
                tail_idx = int(np.argmax(tail_scores))
            tail_score = float(tail_scores[tail_idx])

            # Ensure head comes before tail
            if tail_idx < head_idx:
                head_idx, tail_idx = tail_idx, head_idx
                head_score, tail_score = tail_score, head_score

            # Check gap constraint
            gap_exceeded = (
                ellipsis_max_gap is not None
                and (tail_idx - head_idx) > ellipsis_max_gap
            )

            if gap_exceeded:
                # Use best single window
                best_idx = head_idx if head_score >= tail_score else tail_idx
                bm25_score = max(head_score, tail_score)

                # Calculate ratio for chosen part
                scores = head_scores if head_score >= tail_score else tail_scores
                top1, top2, ratio = _calculate_bm25_ratio(scores)

                # Extract span with expansion
                span_text, window_start_pos = _expand_window(
                    best_idx, original_windows, original_text, expand_window_neighbors
                )

                # Trim to chosen part
                trim_target = head if head_score >= tail_score else tail
            else:
                # Use head-to-tail span
                head_start_pos = original_windows[head_idx][1]
                tail_end_pos = original_windows[tail_idx][2]
                span_text = original_text[head_start_pos:tail_end_pos]
                window_start_pos = head_start_pos

                # Average scores
                bm25_score = (head_score + tail_score) / 2.0

                # Average ratios
                head_top1, head_top2, head_ratio = _calculate_bm25_ratio(head_scores)
                tail_top1, tail_top2, tail_ratio = _calculate_bm25_ratio(tail_scores)
                ratio = (head_ratio + tail_ratio) / 2.0

                # Trim to head part
                trim_target = head

        # Trim span (unified for all cases)
        span_text, global_start, global_end, match_ratio = _trim_and_position(
            trim_target,
            span_text,
            window_start_pos,
            trim_spans,
            trim_method,
            min_fuzzy_ratio,
        )

        # Find source document
        source_doc, source_doc_content = find_source_document(
            global_start, doc_boundaries, doc_content_map
        )

        # Extract context window around match (2000 chars centered on match)
        # Find document start position to calculate relative position
        doc_start = 0
        for doc_name, start, end in doc_boundaries:
            if doc_name == source_doc:
                doc_start = start
                break
        # Position within the source document
        local_start = global_start - doc_start
        local_end = global_end - doc_start
        # Create window centered on match
        context_size = 2000
        match_center = (local_start + local_end) // 2
        ctx_start = max(0, match_center - context_size // 2)
        ctx_end = min(len(source_doc_content), match_center + context_size // 2)
        context_around_match = source_doc_content[ctx_start:ctx_end]
        # Add ellipsis if truncated
        if ctx_start > 0:
            context_around_match = "..." + context_around_match
        if ctx_end < len(source_doc_content):
            context_around_match = context_around_match + "..."

        # Compute truncated texts for embedding (but don't embed yet)
        max_embed_chars = min(max(len(quote) * 2, 500), 4000)
        quote_truncated = quote[:max_embed_chars]
        span_truncated = span_text[:max_embed_chars]

        intermediate_results.append(
            {
                "quote": quote,
                "quote_hash": quote_obj.hash(),
                "source_doc": source_doc,
                "span_text": span_text,
                "source_doc_content": source_doc_content,  # kept for LLM verification
                "context_around_match": context_around_match,  # truncated for export
                "bm25_score": float(bm25_score),
                "bm25_ratio": float(ratio),
                "global_start": int(global_start),
                "global_end": int(global_end),
                "match_ratio": float(match_ratio) if match_ratio is not None else None,
                # texts for embedding (will be removed after embedding)
                "_quote_truncated": quote_truncated,
                "_span_truncated": span_truncated,
            }
        )

    if pbar_bm25:
        pbar_bm25.close()
    logger.debug(f"BM25 matching complete. {len(intermediate_results)} matches found.")

    # Phase 2: Batch embed all unique texts
    # Collect all unique texts needing embeddings
    texts_to_embed = set()
    for r in intermediate_results:
        texts_to_embed.add(r["_quote_truncated"])
        texts_to_embed.add(r["_span_truncated"])

    unique_texts = list(texts_to_embed)
    logger.info(
        f"Phase 2: Computing embeddings for {len(unique_texts)} unique texts..."
    )

    # Batch embed with progress callback
    if show_progress:
        if progress_manager:
            pbar = progress_manager.create_progress_bar(
                total=len(unique_texts),
                desc="VerifyQuotes: embeddings",
                unit="text",
            )
        else:
            desc = "VerifyQuotes: embeddings".ljust(35)
            pbar = tqdm(
                total=len(unique_texts),
                desc=desc,
                unit="text",
                file=sys.stderr,
                ncols=120,
                leave=True,
                mininterval=0.1,
            )
        try:
            embeddings_list = await get_embedding_async(
                unique_texts,
                progress_callback=lambda n: pbar.update(n),
            )
        finally:
            pbar.close()
    else:
        embeddings_list = await get_embedding_async(unique_texts)

    # Build lookup from text -> embedding
    text_to_embedding = {text: emb for text, emb in zip(unique_texts, embeddings_list)}
    logger.info("Embeddings computed.")

    # Phase 3: Compute cosine similarities using cached embeddings
    logger.info("Phase 3: Computing cosine similarities...")
    results = []
    for r in intermediate_results:
        quote_emb = np.array([text_to_embedding[r["_quote_truncated"]]])
        span_emb = np.array([text_to_embedding[r["_span_truncated"]]])
        cosine_sim = float(cosine_similarity(quote_emb, span_emb)[0][0])

        # Remove internal fields and add cosine similarity
        result = {k: v for k, v in r.items() if not k.startswith("_")}
        result["cosine_similarity"] = cosine_sim
        results.append(result)

    logger.info(f"Verification complete: {len(results)} quotes verified.")
    return results


class VerifyQuotes(CompletionDAGNode):
    """Quote verification using BM25-first matching with ellipsis support.

    Uses BM25 (lexical search) to identify candidate spans, including handling
    quotes with ellipses by matching head/tail separately and reconstructing
    the full span. Embeddings are computed on identified spans for verification.

    Configurable windowing (default: 1.1 × longest quote, capped at 500 chars,
    30% overlap). Returns BM25 scores, ratios (top1/top2), and cosine similarity.

    Automatic neighbor expansion (±1 window by default) catches quotes that span
    window boundaries.
    """

    type: Literal["VerifyQuotes"] = "VerifyQuotes"

    # Input configuration
    quotes_from: Optional[str] = (
        None  # Node to extract quotes from (replaces inputs[0])
    )
    search_in: Optional[str] = None  # Node to search in (None = documents)
    check_fairness: bool = False  # Enable fairness checking for themes
    context_window_size: int = 1000  # Context window for fairness check

    # BM25/embedding parameters
    window_size: Optional[int] = 300
    overlap: Optional[int] = None
    bm25_k1: float = 1.5
    bm25_b: float = 0.4  # Lower value reduces long-doc penalty
    ellipsis_max_gap: Optional[int] = 3  # Max windows between head/tail
    trim_spans: bool = True
    trim_method: Literal["fuzzy", "sliding_bm25", "hybrid"] = "fuzzy"
    min_fuzzy_ratio: float = 0.6
    expand_window_neighbors: int = 1  # Search ±N windows around BM25 best match

    # LLM judge templates
    template: Optional[str] = None  # Custom LLM-as-judge prompt template
    fairness_template: Optional[str] = None  # Custom fairness verification template

    # Results
    stats: Optional[Dict[str, Any]] = None
    original_sentences: Optional[List[str]] = None
    extracted_sentences: Optional[List[str]] = None
    sentence_matches: Optional[List[Dict[str, Union[str, Any]]]] = None
    verification_type: Optional[str] = None  # "code" or "theme"

    # Private attribute for organizing LLM results by type (for export)
    _llm_results_by_type: Dict[str, Dict[str, ChatterResult]] = PrivateAttr(
        default_factory=dict
    )

    def validate_template(self):
        """Validate template if provided."""
        if self.template:
            try:
                parse_syntax(self.template)
                return True
            except Exception as e:
                logger.error(f"Judge template syntax error: {e}")
                return False
        return True

    def _normalize_to_outputs_list(
        self, input_data
    ) -> List[Union["Themes", "CodeList"]]:
        """Normalize any input format to list of Themes/CodeList objects.

        Handles:
        - List of ChatterResults (from Map nodes)
        - Single ChatterResult (from Transform/Reduce nodes)
        - Direct Themes or CodeList objects
        - List of Code objects (wrapped into CodeList)
        - List of Theme objects (wrapped into Themes)

        Returns:
            List of Themes or CodeList objects ready for quote extraction
        """
        from soak.models.base import Code, CodeList, Theme, Themes

        outputs = []

        # convert to list if single item
        items = input_data if isinstance(input_data, list) else [input_data]

        # check if items is a list of Code or Theme objects directly
        if items and all(isinstance(item, Code) for item in items):
            # wrap plain list of Code objects into CodeList
            outputs.append(CodeList(codes=items))
            return outputs
        if items and all(isinstance(item, Theme) for item in items):
            # wrap plain list of Theme objects into Themes
            outputs.append(Themes(themes=items))
            return outputs

        for item in items:
            if isinstance(item, (Themes, CodeList)):
                # direct model -- add to outputs
                outputs.append(item)
            elif hasattr(item, "outputs"):
                # ChatterResult -- extract outputs dict
                for output_val in item.outputs.values():
                    if isinstance(output_val, (Themes, CodeList)):
                        outputs.append(output_val)

        return outputs

    def extract_quotes_and_context(self, input_data) -> List[Dict[str, Any]]:
        """Extract quotes with theme/code context if available.

        Returns list of dicts with:
        - quote: Quote object
        - type: "code" or "theme"
        - code: Code object (if available)
        - theme: Theme object (if type=="theme")
        """
        from soak.models.base import CodeList, Themes

        quotes_with_context = []

        # normalize all input formats to list of Themes/CodeList
        outputs = self._normalize_to_outputs_list(input_data)

        # extract quotes from normalized outputs
        for output_val in outputs:
            if isinstance(output_val, Themes):
                # extract from themes
                for theme in output_val.themes:
                    for code in theme.resolved_codes:
                        for quote in code.all_quotes:
                            quotes_with_context.append(
                                {
                                    "quote": quote,
                                    "type": "theme",
                                    "theme": theme,
                                    "code": code,
                                }
                            )
            elif isinstance(output_val, CodeList):
                # extract from codes
                for code in output_val.codes:
                    for quote in code.all_quotes:
                        quotes_with_context.append(
                            {
                                "quote": quote,
                                "type": "code",
                                "code": code,
                            }
                        )

        if not quotes_with_context:
            raise ValueError(
                "No quotes found in input. Check that input contains Code or Theme objects with quotes."
            )

        # set verification type based on first item
        self.verification_type = quotes_with_context[0]["type"]
        return quotes_with_context

    def get_search_corpus(self) -> str:
        """Get text corpus to search for quotes.

        Returns concatenated text from either documents or specified node output.
        """
        if self.search_in is None:
            # Default: search in documents
            return "\n\n".join(doc.content for doc in self.dag.config.documents)
        else:
            # Search in specified node output
            node_output = self.context.get(self.search_in)
            if node_output is None:
                raise ValueError(f"Node '{self.search_in}' not found in context")

            # Extract text from various output types
            if isinstance(node_output, str):
                return node_output
            elif isinstance(node_output, list):
                # List of TrackedItems or strings
                texts = []
                for item in node_output:
                    if isinstance(item, TrackedItem):
                        texts.append(item.content)
                    elif isinstance(item, str):
                        texts.append(item)
                    elif hasattr(item, "response"):
                        # ChatterResult - use response text
                        texts.append(str(item.response))
                return "\n\n".join(texts)
            elif isinstance(node_output, TrackedItem):
                return node_output.content
            elif hasattr(node_output, "response"):
                # Single ChatterResult
                return str(node_output.response)
            else:
                raise ValueError(
                    f"Cannot extract text from node '{self.search_in}' output type: {type(node_output)}"
                )

    async def llm_as_judge(self, quote: str, source: str) -> Dict[str, Any]:
        """Use LLM to verify if a quote is truly contained in the source text.

        This is a last-resort if the lexical and semantic matches are low. We use the LLM to verify if the quote is truly contained in the source text.

        Args:
            quote: The extracted quote to verify
            source: The source text where the quote should appear

        Returns:
            Dict with 'explanation' and 'is_contained' keys
        """
        # Use custom template if provided, otherwise load default from file
        if self.template:
            prompt = self.template
        else:
            # Templates are in soak/templates, not soak/models/templates
            template_path = (
                Path(__file__).parent.parent.parent
                / "templates"
                / "does_quote_exist.sd"
            )
            prompt = template_path.read_text()

        # get LLM kwargs including seed
        extra_kwargs = self.get_llm_kwargs()

        result = await managed_llm_call(
            node_name=self.name,
            config=self.dag.config,
            llm_func=chatter_async,
            item_index=None,
            multipart_prompt=prompt,
            context={"source": source, "quote": quote},
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            extra_kwargs=extra_kwargs,
        )

        # If managed_llm_call returned None (error was skipped), return error indicator
        if result is None:
            return {"explanation": "LLM Error occurred (skipped)", "is_contained": None}

        # accumulate cost from LLM-as-judge call
        self._accumulate_costs(result)

        # Extract the parsed results
        explanation = _extract_llm_field(result, "explanation", "")
        is_contained = _extract_llm_field(result, "is_contained", None)

        return {
            "explanation": explanation,
            "is_contained": is_contained,
            "chatter_result": result,
        }

    async def check_quote_fairness(
        self,
        theme_name: str,
        theme_description: str,
        code_name: str,
        code_description: str,
        quote: str,
        original_text: str,
    ) -> Dict[str, Any]:
        """Use LLM to verify if a quote is used fairly to support a theme.

        Args:
            theme_name: Name of the theme
            theme_description: Description of the theme
            code_name: Name of the code
            code_description: Description of the code
            quote: The quote being verified
            original_text: Context window around the quote from source document

        Returns:
            Dict with 'explanation', 'is_fair', and 'chatter_result' keys
        """
        # Use custom template if provided, otherwise load default
        if self.fairness_template:
            prompt = self.fairness_template
        else:
            template_path = (
                Path(__file__).parent.parent.parent
                / "templates"
                / "verify_theme_quotes.sd"
            )
            prompt = template_path.read_text()

        # Get LLM kwargs including seed
        extra_kwargs = self.get_llm_kwargs()

        # Build context for template
        context = {
            "theme_name": theme_name,
            "theme_description": theme_description,
            "code_name": code_name,
            "code_description": code_description,
            "quote": quote,
            "original_text": original_text,
        }

        result = await managed_llm_call(
            node_name=self.name,
            config=self.dag.config,
            llm_func=chatter_async,
            item_index=None,
            multipart_prompt=prompt,
            context=context,
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            extra_kwargs=extra_kwargs,
        )

        # If managed_llm_call returned None (error was skipped), return error indicator
        if result is None:
            return {
                "explanation": "LLM Error occurred (skipped)",
                "is_fair": None,
                "chatter_result": None,
            }

        # Accumulate cost from LLM call
        self._accumulate_costs(result)

        # Extract the parsed results
        explanation = _extract_llm_field(result, "explanation", "")
        is_fair = _extract_llm_field(result, "is_fair", None)

        return {
            "explanation": explanation,
            "is_fair": is_fair,
            "chatter_result": result,
        }

    def extract_context_window(
        self,
        quote_text: str,
        source_doc_content: str,
        global_start: int,
        global_end: int,
    ) -> str:
        """Extract context window around a quote from the source document.

        Args:
            quote_text: The quote text
            source_doc_content: Full source document content
            global_start: Start position of quote in concatenated text
            global_end: End position of quote in concatenated text

        Returns:
            Context window string (or full document if shorter than window size)
        """
        # If document is shorter than window, return full document
        if len(source_doc_content) <= self.context_window_size:
            return source_doc_content

        # Calculate center of quote
        quote_center = (global_start + global_end) // 2

        # Calculate window bounds
        half_window = self.context_window_size // 2
        window_start = max(0, quote_center - half_window)
        window_end = min(len(source_doc_content), quote_center + half_window)

        # Adjust if we hit document boundaries
        if window_start == 0:
            window_end = min(len(source_doc_content), self.context_window_size)
        elif window_end == len(source_doc_content):
            window_start = max(0, len(source_doc_content) - self.context_window_size)

        return source_doc_content[window_start:window_end]

    async def run(self) -> List[Any]:
        await super().run()

        # Get search corpus (documents or custom node output)
        logger.info(f"VerifyQuotes '{self.name}': Loading search corpus...")
        search_corpus = self.get_search_corpus()
        logger.info(f"Search corpus loaded: {len(search_corpus)} chars")

        # Create document boundaries for tracking source documents
        logger.info("Creating document boundaries...")
        doc_boundaries, doc_content_map = create_document_boundaries(
            self.dag.config.documents
        )
        logger.info(f"Document boundaries created: {len(doc_boundaries)} documents")

        # Store boundaries for output (web UI needs this for position mapping)
        self._doc_boundaries = doc_boundaries

        # Backward compatibility: support both old inputs[0] and new quotes_from
        source_node = self.quotes_from or (self.inputs[0] if self.inputs else None)
        if not source_node:
            raise ValueError("VerifyQuotes requires quotes_from parameter or inputs[0]")

        input_data = self.context.get(source_node)
        if not input_data:
            raise ValueError(f"No output from node '{source_node}'")

        # Extract quotes with context (handles Code and Themes)
        logger.info(f"Extracting quotes from '{source_node}'...")
        quotes_with_context = self.extract_quotes_and_context(input_data)

        logger.info(
            f"Verifying {len(quotes_with_context)} quotes (type: {self.verification_type})"
        )
        logger.info(
            f"Fairness checking: {'enabled' if self.check_fairness else 'disabled'}"
        )

        # Extract Quote objects for verification
        quotes_to_verify = [item["quote"] for item in quotes_with_context]

        # Stage 1: Verify quote existence using BM25 + embeddings
        quote_texts = [q.text for q in quotes_to_verify]
        logger.info(f"Creating search windows (window_size={self.window_size})...")
        windows = make_windows(
            search_corpus,
            window_size=self.window_size,
            overlap=self.overlap,
            extracted_sentences=quote_texts,
        )
        logger.info(f"Created {len(windows)} search windows")
        matches = await verify_quotes_bm25_first(
            quotes_to_verify,
            windows,
            search_corpus,
            doc_boundaries,
            doc_content_map,
            get_embedding_async,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            ellipsis_max_gap=self.ellipsis_max_gap,
            trim_spans=self.trim_spans,
            trim_method=self.trim_method,
            min_fuzzy_ratio=self.min_fuzzy_ratio,
            expand_window_neighbors=self.expand_window_neighbors,
            show_progress=self.dag.config.show_progress,
            progress_manager=self.dag.progress_manager,
        )

        if not matches:
            raise ValueError("No quotes found during existence verification")

        logger.info(f"BM25+embedding verification complete: {len(matches)} matches")

        # Store windows and extracted sentences for compatibility
        self.original_sentences = [w[0] for w in windows]
        self.extracted_sentences = quote_texts

        # Create lookup from quote hash to match results
        quote_hash_to_match = {r["quote_hash"]: r for r in matches}

        # Initialize storage for LLM ChatterResults (for export organization)
        self._llm_results_by_type = {
            "existence": {},  # quote_hash -> ChatterResult
            "fairness": {},  # quote_hash -> ChatterResult  (for themes only)
        }

        # --- Convert to dataframe and compute stats ---
        df = pd.DataFrame(matches)

        # Stage 1.5: LLM-based existence verification for poor matches
        # ratio < 2 means second match is >50% as 'good' (ambiguous)
        logger.info("Identifying poor matches for LLM verification...")
        poor_match_mask = ((df["bm25_score"] < 30) & (df["bm25_ratio"] < 2)) | (
            (df["bm25_score"] < 20) & (df["cosine_similarity"] < 0.7)
        )
        poor_matches = df[poor_match_mask]
        logger.info(
            f"Found {len(poor_matches)}/{len(df)} poor matches needing LLM verification"
        )

        if len(poor_matches) > 0:
            logger.info(
                f"Running LLM existence verification on {len(poor_matches)} poor matches"
            )

            # Initialize columns for all rows
            df["llm_explanation"] = None
            df["llm_is_contained"] = None

            # Run LLM judge on poor matches in parallel
            # Use progress manager if available for coordinated display
            if self.dag.progress_manager:
                if self.dag.cost_tracker:
                    pbar = self.dag.progress_manager.create_cost_progress_bar(
                        total=len(poor_matches),
                        node_name=f"{self.type}: {self.name} (LLM existence)",
                        unit="item",
                    )
                else:
                    pbar = self.dag.progress_manager.create_progress_bar(
                        total=len(poor_matches),
                        desc=f"{self.type}: {self.name} (LLM existence)",
                        unit="item",
                    )
            elif self.dag.cost_tracker:
                from soak.models.progress import CostProgressBar

                pbar = CostProgressBar(
                    tracker=self.dag.cost_tracker,
                    node_name=f"{self.type}: {self.name} (LLM existence)",
                    total=len(poor_matches),
                    unit="item",
                )
            else:
                desc = f"{self.type}: {self.name} (LLM existence)".ljust(35)
                pbar = tqdm(
                    total=len(poor_matches),
                    desc=desc,
                    file=sys.stderr,
                    unit="item",
                    ncols=120,
                    leave=True,
                    mininterval=0.1,
                )

            async with anyio.create_task_group() as tg:

                async def check_match(idx, quote_hash, quote, span_text):
                    try:
                        async with semaphore:
                            result = await self.llm_as_judge(quote, span_text)
                            df.at[idx, "llm_explanation"] = result.get("explanation")
                            df.at[idx, "llm_is_contained"] = result.get("is_contained")
                            # Store ChatterResult for export and cache statistics
                            if result.get("chatter_result"):
                                self._llm_results_by_type["existence"][quote_hash] = (
                                    result["chatter_result"]
                                )
                                self._llm_results.append(result["chatter_result"])

                                # Update progress bar with per-node cost if using CostProgressBar
                                from soak.models.progress import \
                                    CostProgressBar

                                if isinstance(pbar, CostProgressBar):
                                    chatter = result["chatter_result"]
                                    pbar.update_cost(
                                        chatter.fresh_cost,
                                        chatter.prompt_tokens
                                        + chatter.completion_tokens,
                                    )
                    finally:
                        pbar.update(1)

                for idx, row in poor_matches.iterrows():
                    tg.start_soon(
                        check_match,
                        idx,
                        row["quote_hash"],
                        row["quote"],
                        row["source_doc_content"],
                    )

            pbar.close()
            logger.info(
                f"LLM existence verification complete for {len(poor_matches)} quotes"
            )

        # Add code_name to all matches by linking via quote_hash
        # This is done regardless of check_fairness so verification results always have code info
        quote_hash_to_context = {
            item["quote"].hash(): item for item in quotes_with_context
        }
        df["code_name"] = df["quote_hash"].apply(
            lambda h: quote_hash_to_context.get(h, {}).get("code", {})
        ).apply(lambda c: getattr(c, "name", None) if c else None)

        # Stage 2: Fairness verification for themes (optional)
        if self.check_fairness and self.verification_type == "theme":
            logger.info(
                f"Running fairness verification on {len(quotes_with_context)} theme-quote usages"
            )

            # Initialize fairness columns
            df["theme"] = None
            df["theme_description"] = None
            df["code_name"] = None
            df["code_description"] = None
            df["llm_fairness_explanation"] = None
            df["llm_is_fair"] = None

            logger.debug(
                f"Initialized fairness columns. DataFrame has {len(df.columns)} columns: {list(df.columns)}"
            )

            # Run fairness checks in parallel
            # Use progress manager if available for coordinated display
            if self.dag.progress_manager:
                if self.dag.cost_tracker:
                    pbar_fairness = self.dag.progress_manager.create_cost_progress_bar(
                        total=len(df),
                        node_name=f"{self.type}: {self.name} (LLM fairness)",
                        unit="item",
                    )
                else:
                    pbar_fairness = self.dag.progress_manager.create_progress_bar(
                        total=len(df),
                        desc=f"{self.type}: {self.name} (LLM fairness)",
                        unit="item",
                    )
            elif self.dag.cost_tracker:
                from soak.models.progress import CostProgressBar

                pbar_fairness = CostProgressBar(
                    tracker=self.dag.cost_tracker,
                    node_name=f"{self.type}: {self.name} (LLM fairness)",
                    total=len(df),
                    unit="item",
                )
            else:
                desc_fairness = f"{self.type}: {self.name} (LLM fairness)".ljust(35)
                pbar_fairness = tqdm(
                    total=len(df),
                    desc=desc_fairness,
                    file=sys.stderr,
                    unit="item",
                    ncols=120,
                    leave=True,
                    mininterval=0.1,
                )

            async with anyio.create_task_group() as tg:

                async def check_fairness(idx, item, match_result):
                    try:
                        async with semaphore:
                            # Use full source document for consistency with is_contained check
                            source_doc_content = match_result.get(
                                "source_doc_content", ""
                            )

                            # If no source document found, log warning and skip fairness check
                            if not source_doc_content:
                                logger.warning(
                                    f"Cannot verify fairness for quote '{item['quote'].text[:60]}...' - "
                                    f"source document not found. Skipping fairness check."
                                )
                                return  # Skip this quote's fairness check

                            # Run fairness LLM check with full document
                            fairness_result = await self.check_quote_fairness(
                                theme_name=item["theme"].name,
                                theme_description=item["theme"].description,
                                code_name=item["code"].name,
                                code_description=item["code"].description,
                                quote=item["quote"].text,
                                original_text=source_doc_content,
                            )

                            # Update dataframe
                            df.at[idx, "theme"] = item["theme"].name
                            df.at[idx, "theme_description"] = item["theme"].description
                            df.at[idx, "code_name"] = item["code"].name
                            df.at[idx, "code_description"] = item["code"].description
                            df.at[idx, "llm_fairness_explanation"] = fairness_result[
                                "explanation"
                            ]
                            df.at[idx, "llm_is_fair"] = fairness_result["is_fair"]

                            # Store ChatterResult for export and cache statistics
                            if fairness_result["chatter_result"]:
                                quote_hash = item["quote"].hash()
                                self._llm_results_by_type["fairness"][quote_hash] = (
                                    fairness_result["chatter_result"]
                                )
                                self._llm_results.append(
                                    fairness_result["chatter_result"]
                                )

                                # Update progress bar with per-node cost if using CostProgressBar
                                from soak.models.progress import \
                                    CostProgressBar

                                if isinstance(pbar_fairness, CostProgressBar):
                                    chatter = fairness_result["chatter_result"]
                                    pbar_fairness.update_cost(
                                        chatter.fresh_cost,
                                        chatter.prompt_tokens
                                        + chatter.completion_tokens,
                                    )
                    finally:
                        pbar_fairness.update(1)

                # build lookup dict once (O(m))
                quote_hash_to_context = {
                    item["quote"].hash(): item for item in quotes_with_context
                }

                # match quotes_with_context to df rows by quote hash (O(n))
                for idx, row in df.iterrows():
                    quote_hash = row["quote_hash"]
                    context_item = quote_hash_to_context.get(quote_hash)

                    if context_item:
                        match_result = quote_hash_to_match[quote_hash]
                        tg.start_soon(check_fairness, idx, context_item, match_result)

            pbar_fairness.close()
            logger.info(f"Fairness verification complete for {len(df)} quotes")
        else:
            logger.info(
                f"Skipping fairness verification (check_fairness={self.check_fairness}, verification_type={self.verification_type})"
            )

        self.sentence_matches = df.to_dict(orient="records")

        # Compute statistics
        logger.info("Computing verification statistics...")
        n_quotes = len(df)
        n_with_ellipses = df["quote"].apply(lambda q: bool(ELLIPSIS_RE.search(q))).sum()

        self.stats = {
            "n_quotes": int(n_quotes),
            "n_with_ellipses": int(n_with_ellipses),
            "mean_bm25_score": float(df["bm25_score"].mean()),
            "median_bm25_score": float(df["bm25_score"].median()),
            "mean_bm25_ratio": float(df["bm25_ratio"].mean()),
            "median_bm25_ratio": float(df["bm25_ratio"].median()),
            "mean_cosine": float(df["cosine_similarity"].mean()),
            "median_cosine": float(df["cosine_similarity"].median()),
            "min_cosine": float(df["cosine_similarity"].min()),
            "max_cosine": float(df["cosine_similarity"].max()),
        }

        # Add match_ratio stats if available
        if "match_ratio" in df.columns and df["match_ratio"].notna().any():
            valid_match = df["match_ratio"].dropna()
            self.stats.update(
                {
                    "mean_match_ratio": (
                        float(valid_match.mean()) if len(valid_match) > 0 else None
                    ),
                    "median_match_ratio": (
                        float(valid_match.median()) if len(valid_match) > 0 else None
                    ),
                    "min_match_ratio": (
                        float(valid_match.min()) if len(valid_match) > 0 else None
                    ),
                    "n_low_match_confidence": (
                        int((valid_match < self.min_fuzzy_ratio).sum())
                        if len(valid_match) > 0
                        else 0
                    ),
                    "mean_span_length": float(
                        (df["global_end"] - df["global_start"]).mean()
                    ),
                }
            )

        # add fairness stats if applicable
        if "llm_is_fair" in df.columns and df["llm_is_fair"].notna().any():
            # use unique quote hashes to avoid double counting
            n_quotes = df["quote_hash"].nunique()

            valid_fair = df["llm_is_fair"].dropna()
            n_fair = int(valid_fair.sum()) if len(valid_fair) > 0 else 0
            self.stats.update(
                {
                    "n_fair": n_fair,
                    "n_unfair": len(valid_fair) - n_fair,
                    "n_not_checked": n_quotes - len(valid_fair),
                    "pct_fair": (
                        float(n_fair / n_quotes * 100) if n_quotes > 0 else None
                    ),
                }
            )

        # Store output for downstream nodes to access via context
        # Include document boundaries for position mapping in web UI
        self.output = {
            "matches": self.sentence_matches,
            "doc_boundaries": [
                {"name": name, "start": start, "end": end}
                for name, start, end in self._doc_boundaries
            ],
            "stats": self.stats,
        }

        logger.info(
            f"VerifyQuotes '{self.name}' complete: {n_quotes} quotes, "
            f"mean_bm25={self.stats['mean_bm25_score']:.1f}, "
            f"mean_cosine={self.stats['mean_cosine']:.3f}"
        )
        return matches

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, DataFrame of quote matches and statistics."""
        # Get base metadata from parent
        result = super().result()

        df = pd.DataFrame(self.sentence_matches)

        # Reorder columns for readability
        if df.empty:
            raise Exception("No matches found when verifying quotes.")

        # Sort by confidence
        if "bm25_ratio" in df.columns:
            df = df.sort_values(
                ["bm25_score", "bm25_ratio", "cosine_similarity"],
                ascending=[True, True, True],
            )

        # Add VerifyQuotes-specific data
        result["matches_df"] = df
        result["stats"] = self.stats
        result["metadata"]["num_quotes"] = len(df)
        result["metadata"]["min_fuzzy_ratio"] = self.min_fuzzy_ratio

        return result

    def export(self, folder: Path, unique_id: str = ""):
        super().export(folder, unique_id=unique_id)
        (folder / "info.txt").write_text(VerifyQuotes.__doc__)
        pd.DataFrame(self.stats, index=[0]).melt().to_csv(
            folder / "stats.csv", index=False
        )

        # Export quote verification as Excel with formatting
        uid_suffix = f"_{unique_id}" if unique_id else ""
        excel_path = folder / f"quote_verification{uid_suffix}.xlsx"
        df = pd.DataFrame(self.sentence_matches)

        # Drop full document content (too large for Excel, kept internally for LLM verification)
        if "source_doc_content" in df.columns:
            df = df.drop(columns=["source_doc_content"])

        # Rename columns for clarity
        df = df.rename(
            columns={
                "quote": "extracted_quote",
                "span_text": "found_in_original",
                "context_around_match": "source_context",
            }
        )

        # Reorder columns: text columns first, then theme info (if themes), then LLM verifications, then metrics
        priority_cols = [
            "extracted_quote",
            "source_doc",
        ]

        if "llm_is_fair" in df.columns:
            priority_cols.extend(["llm_is_fair", "llm_fairness_explanation"])

        # Add theme columns if present (for theme verification)
        if "theme" in df.columns:
            priority_cols.extend(
                ["theme", "theme_description", "code_name", "code_description"]
            )

        # All other columns at the end
        other_cols = [col for col in df.columns if col not in priority_cols]
        df = df[priority_cols + other_cols]

        # Sort by fairness (False first), then LLM existence verification (False first), then BM25 metrics
        # This puts the most 'problematic' quotes at top
        sort_cols = []
        sort_ascending = []

        if "llm_is_fair" in df.columns and df["llm_is_fair"].notna().any():
            sort_cols.append("llm_is_fair")
            sort_ascending.append(True)  # False sorts before True

        if "llm_is_contained" in df.columns and df["llm_is_contained"].notna().any():
            sort_cols.append("llm_is_contained")
            sort_ascending.append(True)  # False sorts before True

        sort_cols.extend(["bm25_score", "bm25_ratio", "cosine_similarity"])
        sort_ascending.extend([True, True, True])

        df = df.sort_values(sort_cols, ascending=sort_ascending)

        # Use xlsxwriter engine - more permissive with special characters
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Quote Verification", index=False)

            workbook = writer.book
            worksheet = writer.sheets["Quote Verification"]

            # Define formats
            wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})
            default_format = workbook.add_format({"font_size": 11})

            # Set column widths based on header names
            text_cols = [
                "extracted_quote",
                "found_in_original",
                "source_context",
                "llm_explanation",
                "llm_fairness_explanation",
                "theme_description",
                "code_description",
            ]
            bool_cols = ["llm_is_contained", "llm_is_fair"]
            name_cols = ["theme", "code_name", "source_doc"]
            num_cols = [
                "bm25_score",
                "bm25_ratio",
                "cosine_similarity",
                "global_start",
                "global_end",
                "match_ratio",
            ]

            for col_idx, col_name in enumerate(df.columns):
                if col_name in text_cols:
                    worksheet.set_column(col_idx, col_idx, 60, wrap_format)
                elif col_name in bool_cols:
                    worksheet.set_column(col_idx, col_idx, 18, default_format)
                elif col_name in name_cols:
                    worksheet.set_column(col_idx, col_idx, 25, default_format)
                elif col_name in num_cols:
                    worksheet.set_column(col_idx, col_idx, 15, default_format)
                else:
                    worksheet.set_column(col_idx, col_idx, 20, default_format)

        # Export LLM prompts and responses
        if self._llm_results_by_type:
            from ..utils import export_chatter_result

            # Export existence check LLM calls
            if self._llm_results_by_type.get("existence"):
                existence_folder = folder / "llm_existence_checks"
                existence_folder.mkdir(exist_ok=True)

                for idx, (quote_hash, result) in enumerate(
                    sorted(self._llm_results_by_type["existence"].items())
                ):
                    file_prefix = f"{idx:04d}_{quote_hash}"
                    export_chatter_result(result, existence_folder, file_prefix)

            # Export fairness check LLM calls (themes only)
            if self._llm_results_by_type.get("fairness"):
                fairness_folder = folder / "llm_fairness_checks"
                fairness_folder.mkdir(exist_ok=True)

                for idx, (quote_hash, result) in enumerate(
                    sorted(self._llm_results_by_type["fairness"].items())
                ):
                    file_prefix = f"{idx:04d}_{quote_hash}"
                    export_chatter_result(result, fairness_folder, file_prefix)
