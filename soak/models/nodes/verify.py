"""VerifyQuotes node for validating quotes against source documents."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import anyio
import nltk
import numpy as np
import pandas as pd
from pydantic import Field
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from struckdown import chatter_async
from struckdown.parsing import parse_syntax

from ..base import (TrackedItem, get_action_lookup, get_embedding,
                    safe_json_dump, semaphore)
from .base import CompletionDAGNode

logger = logging.getLogger(__name__)


def make_windows(
    text: str,
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    extracted_sentences: Optional[List[str]] = None,
) -> List[Tuple[str, int, int]]:
    """Create overlapping windows of text.

    Returns list of tuples: (window_text, start_pos, end_pos)

    Defaults:
    - overlap: 30% of window_size (helps catch quotes spanning window boundaries)
    """

    if not overlap:
        overlap = int(window_size * 0.3)  # 30% overlap for better boundary coverage

    windows = []
    i = 0
    while i < len(text):
        start = i
        end = min(i + window_size, len(text))
        windows.append((text[start:end], start, end))
        i += window_size - overlap
    return windows


ELLIPSIS_RE = re.compile(r"\.{3,}|…")


def create_document_boundaries(
    documents: List["TrackedItem"],
) -> Tuple[List[Tuple[str, int, int]], Dict[str, str]]:
    """Create a list of (doc_name, start_pos, end_pos) for each document in concatenated text.

    Assumes documents are joined with "\n\n" separator.

    Returns:
        Tuple of (boundaries, doc_content_map) where:
        - boundaries: List of (doc_name, start_pos, end_pos)
        - doc_content_map: Dict mapping doc_name to full document content
    """
    boundaries = []
    doc_content_map = {}
    current_pos = 0

    for doc in documents:
        content_len = len(doc.content)

        doc_name = None
        doc_name = (
            doc.metadata.get("filename")
            if hasattr(doc, "metadata") and doc.metadata
            else (
                doc.source_id
                if hasattr(doc, "source_id")
                else getattr(doc, "path", "unknown")
            )
        )
        # Fall back to source_id or path attribute
        if not doc_name:
            doc_name = (
                doc.source_id
                if hasattr(doc, "source_id")
                else getattr(doc, "path", "unknown")
            )

        doc_name_str = str(doc_name)
        boundaries.append((doc_name_str, current_pos, current_pos + content_len))
        doc_content_map[doc_name_str] = doc.content
        current_pos += content_len + 2  # +2 for "\n\n" separator

    return boundaries, doc_content_map


def find_source_document(
    position: int,
    doc_boundaries: List[Tuple[str, int, int]],
    doc_content_map: Dict[str, str],
) -> Tuple[str, str]:
    """Find which document a character position belongs to.

    Returns:
        Tuple of (doc_name, doc_content)
    """
    for doc_name, start, end in doc_boundaries:
        if start <= position < end:
            return doc_name, doc_content_map.get(doc_name, "")
    return "unknown", ""


def find_alignment_fuzzy(
    quote: str, span: str, min_ratio: float = 0.6, context_pad: int = 30
) -> Dict[str, Any]:
    """Find best character offset in span where quote aligns using fuzzy matching.

    Uses difflib.SequenceMatcher.get_matching_blocks() to find exact positions.

    Returns dict with: start_char, end_char, match_ratio, matched_text
    """
    from difflib import SequenceMatcher

    # Normalize for matching
    clean = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    quote_clean = clean(quote)
    span_clean = clean(span)

    if not quote_clean or not span_clean:
        return {"start_char": 0, "end_char": 0, "match_ratio": 0.0, "matched_text": ""}

    # Fast path: exact substring
    if quote_clean in span_clean:
        offset = span_clean.index(quote_clean)
        start = offset
        end = offset + len(quote_clean)

        # Snap to word boundaries
        start, end = snap_to_boundaries(span, start, end, snap_to="word")

        return {
            "start_char": start,
            "end_char": end,
            "match_ratio": 1.0,
            "matched_text": span[start:end],
        }

    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, span_clean, quote_clean)

    # Get all matching blocks above minimum length
    blocks = [b for b in matcher.get_matching_blocks() if b[2] > 5]

    if not blocks:
        # Fallback: no good match
        return {
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": 0.0,
            "matched_text": span,
        }

    # Find the best block (longest contiguous match)
    best_block = max(blocks, key=lambda b: b[2])
    i1, i2, n = best_block
    match_ratio = matcher.ratio()

    # Add context padding
    start = max(i1 - context_pad, 0)
    end = min(i1 + n + context_pad, len(span))

    # Snap to word boundaries to avoid mid-word cuts
    start, end = snap_to_boundaries(span, start, end, snap_to="word")

    return {
        "start_char": start,
        "end_char": end,
        "match_ratio": float(match_ratio),
        "matched_text": span[start:end],
    }


def find_alignment_sliding_bm25(quote: str, span: str) -> Tuple[int, float]:
    """Find best alignment using word-level sliding BM25.

    Returns (start_char_offset, bm25_score)
    """
    quote_tokens = nltk.word_tokenize(quote.lower())
    span_tokens = nltk.word_tokenize(span.lower())

    if len(quote_tokens) > len(span_tokens):
        return 0, 0.0

    # Build windows
    window_size = max(len(quote_tokens), int(len(quote_tokens) * 1.2))
    windows = []
    window_starts = []  # token positions

    for i in range(len(span_tokens) - window_size + 1):
        windows.append(span_tokens[i : i + window_size])
        window_starts.append(i)

    if not windows:
        return 0, 0.0

    # Score with BM25
    bm25 = BM25Okapi(windows)
    scores = bm25.get_scores(quote_tokens)
    best_idx = int(np.argmax(scores))
    best_token_offset = window_starts[best_idx] if best_idx < len(window_starts) else 0

    # Convert token offset to character offset (approximate)
    # Rebuild span up to that token
    if best_token_offset > 0:
        char_offset = len(" ".join(span_tokens[:best_token_offset])) + 1  # +1 for space
    else:
        char_offset = 0

    return char_offset, float(scores[best_idx])


def trim_span_to_quote(
    quote: str,
    span: str,
    method: Literal["fuzzy", "sliding_bm25", "hybrid"] = "hybrid",
    min_fuzzy_ratio: float = 0.6,
    min_span_length_multiplier: float = 1.2,
    context_pad: int = 30,
) -> Dict[str, Any]:
    """Trim span to align with quote start using matching blocks.

    Returns dict with: matched_text, start_char, end_char, match_ratio
    """
    if not span or not quote:
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span) if span else 0,
            "match_ratio": 0.0,
        }

    # Don't trim very short quotes (unreliable)
    if len(quote) < 20:
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": 0.0,
        }

    if method == "fuzzy":
        result = find_alignment_fuzzy(quote, span, min_fuzzy_ratio, context_pad)

    elif method == "sliding_bm25":
        offset, confidence = find_alignment_sliding_bm25(quote, span)
        start, end = offset, len(span)
        # Snap to boundaries
        start, end = snap_to_boundaries(span, start, end, snap_to="word")
        # Convert to dict format
        result = {
            "start_char": start,
            "end_char": end,
            "match_ratio": confidence,
            "matched_text": span[start:end],
        }

    elif method == "hybrid":
        # Try fuzzy first
        result = find_alignment_fuzzy(quote, span, min_fuzzy_ratio, context_pad)
        if result["match_ratio"] < min_fuzzy_ratio:
            # Fall back to BM25
            offset_bm25, confidence_bm25 = find_alignment_sliding_bm25(quote, span)
            # Use BM25 result if it seems reasonable
            if offset_bm25 < len(span) * 0.5:  # heuristic: not too far into span
                start, end = offset_bm25, len(span)
                # Snap to boundaries
                start, end = snap_to_boundaries(span, start, end, snap_to="word")
                result = {
                    "start_char": start,
                    "end_char": end,
                    "match_ratio": confidence_bm25,
                    "matched_text": span[start:end],
                }
    else:
        result = {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": 0.0,
        }

    # Safety: don't trim if result would be too short (preserve some context)
    min_result_length = int(len(quote) * min_span_length_multiplier)
    matched_len = result["end_char"] - result["start_char"]
    if matched_len < min_result_length and matched_len < len(span):
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": result["match_ratio"],
        }

    return result


def snap_to_boundaries(
    text: str, start: int, end: int, snap_to: Literal["word", "sentence"] = "word"
) -> Tuple[int, int]:
    """Expand start/end to nearest word or sentence boundary.

    Prevents ugly mid-word cuts by snapping outward to natural boundaries.

    Args:
        text: Full text
        start: Start index (inclusive)
        end: End index (exclusive, as in text[start:end])
    """
    if snap_to == "word":
        # Define word boundary characters (whitespace and punctuation)
        boundaries = {
            " ",
            "\n",
            "\t",
            "\r",
            ".",
            "!",
            "?",
            ",",
            ";",
            ":",
            "-",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            '"',
            "'",
            "/",
            "\\",
        }

        # Expand left until we hit a boundary (or reach start of text)
        while start > 0 and text[start - 1] not in boundaries:
            start -= 1

        # Expand right until we hit a boundary (or reach end of text)
        # Note: end is exclusive (text[start:end]), so we check text[end] if it exists
        while end < len(text) and text[end] not in boundaries:
            end += 1

        # Trim leading whitespace
        while start < end and text[start] in (" ", "\n", "\t", "\r"):
            start += 1

        # Trim trailing whitespace
        while end > start and text[end - 1] in (" ", "\n", "\t", "\r"):
            end -= 1

    elif snap_to == "sentence":
        # Find sentence boundaries (periods, exclamation, question marks followed by space/newline)
        sentence_pattern = r"[.!?][\s\n]+"
        sentence_ends = [m.end() for m in re.finditer(sentence_pattern, text)]

        # Expand left to previous sentence end (or beginning)
        prev_end = 0
        for pos in sentence_ends:
            if pos < start:
                prev_end = pos
            else:
                break
        start = prev_end

        # Expand right to next sentence end (or end of text)
        next_end = len(text)
        for pos in sentence_ends:
            if pos > end:
                next_end = pos
                break
        end = next_end

    return start, end


def is_match_truncated(
    match_result: Dict[str, Any], span_text: str, boundary_threshold: int = 30
) -> bool:
    """Detect if a match looks truncated and might benefit from window expansion.

    Only checks boundary positions (not match_ratio) because low ratio can mean
    either truncation OR too much context (we can't distinguish).

    Returns True if:
    - Matched text starts very close to span beginning (might extend left)
    - Matched text ends very close to span end (might extend right)
    """
    start_char = match_result.get("start_char", 0)
    end_char = match_result.get("end_char", len(span_text))

    # Match starts at/near beginning → might be left-truncated
    starts_at_boundary = start_char < boundary_threshold

    # Match ends at/near end → might be right-truncated
    ends_at_boundary = end_char > len(span_text) - boundary_threshold

    return starts_at_boundary or ends_at_boundary


def verify_quotes_bm25_first(
    extracted_sentences: List[str],
    original_windows: List[Tuple[str, int, int]],
    original_text: str,
    doc_boundaries: List[Tuple[str, int, int]],
    doc_content_map: Dict[str, str],
    get_embedding,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.4,
    ellipsis_max_gap: Optional[int] = 300,
    trim_spans: bool = True,
    trim_method: Literal["fuzzy", "sliding_bm25", "hybrid"] = "hybrid",
    min_fuzzy_ratio: float = 0.6,
    expand_window_neighbors: int = 1,
) -> List[Dict[str, Any]]:
    """Verify quotes using BM25-first matching with ellipsis support.

    For each quote:
    - If no ellipsis: find best BM25 window
    - If ellipsis: match head/tail separately, reconstruct span
    - Always compute embedding similarity on combined span
    - Return BM25 score, ratio (top1/top2), cosine similarity, and full source document content

    Args:
        original_windows: List of (window_text, start_pos, end_pos) tuples
        original_text: Full concatenated source text
        doc_boundaries: List of (doc_name, start_pos, end_pos) for document tracking
        doc_content_map: Dict mapping doc_name to full document content
    """

    if not extracted_sentences:
        return []

    # Extract window texts and build BM25 index
    window_texts = [w[0] for w in original_windows]
    tokenized = [nltk.word_tokenize(w.lower()) for w in window_texts]
    bm25 = BM25Okapi(tokenized, k1=bm25_k1, b=bm25_b)

    results = []

    for quote in extracted_sentences:
        has_ellipsis = bool(ELLIPSIS_RE.search(quote))

        if not has_ellipsis:
            # Simple case: single BM25 match
            query_tokens = nltk.word_tokenize(quote.lower())
            scores = bm25.get_scores(query_tokens)
            sorted_scores = np.sort(scores)[::-1]

            top1 = float(sorted_scores[0]) if len(sorted_scores) > 0 else 0.0
            top2 = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            ratio = top1 / (top2 + 1e-6)

            best_idx = int(np.argmax(scores))
            bm25_score = top1

            # Get window with position info
            single_window_text, window_start_pos, window_end_pos = original_windows[
                best_idx
            ]

            if trim_spans:
                # Try matching in single window
                match_result = trim_span_to_quote(
                    quote, single_window_text, trim_method, min_fuzzy_ratio
                )

                # Check if match looks truncated
                if expand_window_neighbors > 0 and is_match_truncated(
                    match_result, single_window_text
                ):
                    # Expand to neighbors and retry using original text
                    start_idx = max(0, best_idx - expand_window_neighbors)
                    end_idx = min(
                        len(original_windows), best_idx + expand_window_neighbors + 1
                    )

                    # Get global positions for expanded range
                    expanded_start_pos = original_windows[start_idx][1]
                    expanded_end_pos = original_windows[end_idx - 1][2]
                    expanded_span = original_text[expanded_start_pos:expanded_end_pos]

                    # Retry matching in expanded span
                    expanded_result = trim_span_to_quote(
                        quote, expanded_span, trim_method, min_fuzzy_ratio
                    )

                    # Use expanded result if it's better
                    if expanded_result.get("match_ratio", 0) > match_result.get(
                        "match_ratio", 0
                    ):
                        match_result = expanded_result
                        window_start_pos = expanded_start_pos  # Update base position

                span_text = match_result["matched_text"]
                window_relative_start = match_result["start_char"]
                window_relative_end = match_result["end_char"]
                match_ratio = match_result["match_ratio"]

                # Convert to global positions
                global_start = window_start_pos + window_relative_start
                global_end = window_start_pos + window_relative_end
            else:
                span_text = single_window_text
                global_start = window_start_pos
                global_end = window_end_pos
                match_ratio = None

            # Find source document and get full document content
            source_doc, source_doc_content = find_source_document(
                global_start, doc_boundaries, doc_content_map
            )

        else:
            # Ellipsis case: match head and tail
            parts = [p.strip() for p in ELLIPSIS_RE.split(quote) if p.strip()]

            if len(parts) < 2:
                # Degenerate case: ellipsis but only one part
                parts = [quote.replace("...", "").replace("…", "").strip()]

            head = parts[0]
            tail = parts[-1]

            # BM25 for head
            head_tokens = nltk.word_tokenize(head.lower())
            head_scores = bm25.get_scores(head_tokens)
            head_idx = int(np.argmax(head_scores))
            head_score = float(head_scores[head_idx])

            # BM25 for tail
            tail_tokens = nltk.word_tokenize(tail.lower())
            tail_scores = bm25.get_scores(tail_tokens)
            tail_idx = int(np.argmax(tail_scores))
            tail_score = float(tail_scores[tail_idx])

            # Ensure head comes before tail
            if tail_idx < head_idx:
                head_idx, tail_idx = tail_idx, head_idx
                head_score, tail_score = tail_score, head_score

            # Check gap constraint
            if (
                ellipsis_max_gap is not None
                and (tail_idx - head_idx) > ellipsis_max_gap
            ):
                # Fall back to single best window
                best_idx = head_idx if head_score >= tail_score else tail_idx

                # Expand search space to include neighbor windows using original text
                if expand_window_neighbors > 0:
                    start_idx = max(0, best_idx - expand_window_neighbors)
                    end_idx = min(
                        len(original_windows), best_idx + expand_window_neighbors + 1
                    )

                    expanded_start_pos = original_windows[start_idx][1]
                    expanded_end_pos = original_windows[end_idx - 1][2]
                    expanded_span = original_text[expanded_start_pos:expanded_end_pos]
                    window_start_pos = expanded_start_pos
                else:
                    expanded_span, window_start_pos, _ = original_windows[best_idx]

                bm25_score = max(head_score, tail_score)

                # Calculate ratio for the chosen part
                scores = head_scores if head_score >= tail_score else tail_scores
                sorted_scores = np.sort(scores)[::-1]
                top1 = float(sorted_scores[0]) if len(sorted_scores) > 0 else 0.0
                top2 = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
                ratio = top1 / (top2 + 1e-6)

                # Trim span to align with the chosen part (using expanded search space)
                if trim_spans:
                    part_to_match = head if head_score >= tail_score else tail
                    match_result = trim_span_to_quote(
                        part_to_match, expanded_span, trim_method, min_fuzzy_ratio
                    )
                    span_text = match_result["matched_text"]
                    window_relative_start = match_result["start_char"]
                    window_relative_end = match_result["end_char"]
                    match_ratio = match_result["match_ratio"]

                    # Convert to global positions
                    global_start = window_start_pos + window_relative_start
                    global_end = window_start_pos + window_relative_end
                else:
                    span_text = expanded_span
                    global_start = window_start_pos
                    global_end = window_start_pos + len(expanded_span)
                    match_ratio = None

                # Find source document and get full document content
                source_doc, source_doc_content = find_source_document(
                    global_start, doc_boundaries, doc_content_map
                )
            else:
                # Reconstruct span from head to tail using global positions
                head_start_pos = original_windows[head_idx][1]
                tail_end_pos = original_windows[tail_idx][2]
                span_text = original_text[head_start_pos:tail_end_pos]

                bm25_score = (head_score + tail_score) / 2.0

                # Calculate ratio as average of head and tail ratios
                head_sorted = np.sort(head_scores)[::-1]
                tail_sorted = np.sort(tail_scores)[::-1]

                head_top1 = float(head_sorted[0]) if len(head_sorted) > 0 else 0.0
                head_top2 = float(head_sorted[1]) if len(head_sorted) > 1 else 0.0
                tail_top1 = float(tail_sorted[0]) if len(tail_sorted) > 0 else 0.0
                tail_top2 = float(tail_sorted[1]) if len(tail_sorted) > 1 else 0.0

                head_ratio = head_top1 / (head_top2 + 1e-6)
                tail_ratio = tail_top1 / (tail_top2 + 1e-6)
                ratio = (head_ratio + tail_ratio) / 2.0

                # Trim span to align with head part
                if trim_spans:
                    head_part = parts[0]
                    match_result = trim_span_to_quote(
                        head_part, span_text, trim_method, min_fuzzy_ratio
                    )
                    span_text = match_result["matched_text"]
                    window_relative_start = match_result["start_char"]
                    window_relative_end = match_result["end_char"]
                    match_ratio = match_result["match_ratio"]

                    # Convert to global positions
                    global_start = head_start_pos + window_relative_start
                    global_end = head_start_pos + window_relative_end
                else:
                    global_start = head_start_pos
                    global_end = tail_end_pos
                    match_ratio = None

                # Find source document and get full document content
                source_doc, source_doc_content = find_source_document(
                    global_start, doc_boundaries, doc_content_map
                )

        # Compute embedding similarity between original quote and identified span
        # For long spans (e.g., ellipsis quotes spanning multiple windows), we need to
        # avoid exceeding the embedding model's context window (8192 tokens).
        # Strategy: embed comparable-length excerpts - use 2x quote length (bounded by 2000 chars)
        # to allow for context while staying within limits
        max_embed_chars = min(max(len(quote) * 2, 500), 4000)
        quote_truncated = quote[:max_embed_chars]

        # For very long spans, take start portion (where quote likely appears)
        # rather than truncating the quote comparison unfairly
        span_truncated = span_text[:max_embed_chars]

        quote_emb = np.array(get_embedding([quote_truncated]))
        span_emb = np.array(get_embedding([span_truncated]))
        cosine_sim = float(cosine_similarity(quote_emb, span_emb)[0][0])

        results.append(
            {
                "quote": quote,
                "source_doc": source_doc,
                "span_text": span_text,
                "source_doc_content": source_doc_content,
                "bm25_score": float(bm25_score),
                "bm25_ratio": float(ratio),
                "global_start": int(global_start),
                "global_end": int(global_end),
                "match_ratio": float(match_ratio) if match_ratio is not None else None,
                "cosine_similarity": float(cosine_sim),
            }
        )

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
    window_size: Optional[int] = 300
    overlap: Optional[int] = None
    bm25_k1: float = 1.5
    bm25_b: float = 0.4  # Lower value reduces long-doc penalty
    ellipsis_max_gap: Optional[int] = 3  # Max windows between head/tail
    trim_spans: bool = True
    trim_method: Literal["fuzzy", "sliding_bm25", "hybrid"] = "fuzzy"
    min_fuzzy_ratio: float = 0.6
    expand_window_neighbors: int = 1  # Search ±N windows around BM25 best match
    template_text: Optional[str] = None  # Custom LLM-as-judge prompt template

    stats: Optional[Dict[str, Any]] = None
    original_sentences: Optional[List[str]] = None
    extracted_sentences: Optional[List[str]] = None
    sentence_matches: Optional[List[Dict[str, Union[str, Any]]]] = None

    def validate_template(self):
        """Validate template_text if provided."""
        if self.template_text:
            try:
                parse_syntax(self.template_text)
                return True
            except Exception as e:
                logger.error(f"Judge template syntax error: {e}")
                return False
        return True

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
        if self.template_text:
            prompt = self.template_text
        else:
            # Templates are in soak/templates, not soak/models/templates
            template_path = (
                Path(__file__).parent.parent.parent
                / "templates"
                / "nodes"
                / "llm_as_judge.md"
            )
            prompt = template_path.read_text()

        try:
            result = await chatter_async(
                multipart_prompt=prompt,
                context={"text1": quote, "text2": source},
                model=self.get_model(),
                credentials=self.dag.config.llm_credentials,
                action_lookup=get_action_lookup(),
            )

            # Extract the parsed results
            explanation = (
                result.results.get("explanation", {}).output
                if hasattr(result.results.get("explanation", {}), "output")
                else ""
            )
            is_contained = (
                result.results.get("is_contained", {}).output
                if hasattr(result.results.get("is_contained", {}), "output")
                else None
            )

            return {"explanation": explanation, "is_contained": is_contained}
        except Exception as e:
            logger.error(f"Error in llm_as_judge: {e}")
            return {"explanation": f"Error: {str(e)}", "is_contained": None}

    async def run(self) -> List[Any]:
        await super().run()

        alldocs = "\n\n".join(doc.content for doc in self.dag.config.documents)

        # Create document boundaries for tracking source documents
        doc_boundaries, doc_content_map = create_document_boundaries(
            self.dag.config.documents
        )

        codes = self.context.get("codes")
        if not codes:
            raise Exception("VerifyQuotes must be run after node called `codes`")

        # collect extracted quotes
        self.extracted_sentences = []
        # Wrap in list if it's a single result
        codes_list = [codes] if not isinstance(codes, list) else codes
        for result in codes_list:
            try:
                # Try different ways to extract codes from the result
                cset = None
                if hasattr(result, "outputs"):
                    if hasattr(result.outputs, "codes"):
                        cset = result.outputs.codes
                    elif isinstance(result.outputs, dict) and "codes" in result.outputs:
                        cset = result.outputs["codes"]
                elif hasattr(result, "results") and result.results.get("codes"):
                    output = result.results.get("codes").output
                    if hasattr(output, "codes"):
                        cset = output.codes
                    elif isinstance(output, dict) and "codes" in output:
                        cset = output["codes"]

                if not cset:
                    logger.warning(
                        f"Could not extract codes from result: {type(result)}"
                    )
                    continue

                # Handle CodeList wrapper
                codes_to_process = cset.codes if hasattr(cset, "codes") else cset
                for code in codes_to_process:
                    if hasattr(code, "quotes"):
                        self.extracted_sentences.extend(code.quotes)
                    else:
                        logger.warning(
                            f"Code object missing 'quotes' attribute: {type(code)}"
                        )
            except Exception as e:
                logger.error(f"Error extracting quotes from result: {e}")
                raise

        # Check if we found any quotes - fail early if not
        if not self.extracted_sentences:
            logger.error("No quotes found in codes to verify")
            import pdb

            pdb.set_trace()
            raise ValueError(
                "No quotes found in codes to verify. Check that the 'codes' node produces Code objects with quotes."
            )

        # --- Create windowed slices through haystack ---
        windows_with_positions = make_windows(
            alldocs,
            window_size=self.window_size,
            overlap=self.overlap,
            extracted_sentences=self.extracted_sentences,
        )

        # Store just the text for serialization (not the position tuples)
        self.original_sentences = [w[0] for w in windows_with_positions]

        # --- Run quote matching with BM25-first approach ---
        matches = verify_quotes_bm25_first(
            self.extracted_sentences,
            windows_with_positions,
            alldocs,
            doc_boundaries,
            doc_content_map,
            get_embedding,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            ellipsis_max_gap=self.ellipsis_max_gap,
            trim_spans=self.trim_spans,
            trim_method=self.trim_method,
            min_fuzzy_ratio=self.min_fuzzy_ratio,
            expand_window_neighbors=self.expand_window_neighbors,
        )

        # --- Convert to dataframe and compute stats ---
        df = pd.DataFrame(matches)

        # --- LLM-based verification for poor matches ---
        # Identify poor matches based on multiple criteria
        # TODO formalise why I picked this heuristic
        poor_match_mask = ((df["bm25_score"] < 30) & (df["bm25_ratio"] < 2)) | (
            (df["bm25_score"] < 20) & (df["cosine_similarity"] < 0.7)
        )
        poor_matches = df[poor_match_mask]

        if len(poor_matches) > 0:
            logger.info(f"Running LLM verification on {len(poor_matches)} poor matches")

            # Initialize columns for all rows
            df["llm_explanation"] = None
            df["llm_is_contained"] = None

            # Run LLM judge on poor matches in parallel
            async with anyio.create_task_group() as tg:

                async def check_match(idx, quote, span_text):
                    async with semaphore:
                        result = await self.llm_as_judge(quote, span_text)
                        df.at[idx, "llm_explanation"] = result["explanation"]
                        df.at[idx, "llm_is_contained"] = result["is_contained"]

                for idx, row in poor_matches.iterrows():
                    tg.start_soon(
                        check_match, idx, row["quote"], row["source_doc_content"]
                    )

        self.sentence_matches = df.to_dict(orient="records")

        # Compute statistics
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

        # Rename columns for clarity
        df = df.rename(
            columns={
                "quote": "extracted_quote",
                "span_text": "found_in_original",
                "source_doc_content": "full_original_text",
            }
        )

        # Reorder columns: text columns first, then source_doc, then LLM judge, then metrics
        priority_cols = [
            "extracted_quote",
            "found_in_original",
            "source_doc",
            "full_original_text",
        ]

        # Add LLM columns after full_original_text if they exist
        llm_cols = []
        if "llm_is_contained" in df.columns:
            llm_cols.append("llm_is_contained")
        if "llm_explanation" in df.columns:
            llm_cols.append("llm_explanation")

        priority_cols.extend(llm_cols)
        other_cols = [col for col in df.columns if col not in priority_cols]
        df = df[priority_cols + other_cols]

        # Sort by LLM verification (False first) then BM25 metrics, so most problematic quotes appear at top
        sort_cols = []
        sort_ascending = []

        if "llm_is_contained" in df.columns:
            sort_cols.append("llm_is_contained")
            sort_ascending.append(
                True
            )  # False sorts before True, putting failed verifications first

        sort_cols.extend(["bm25_score", "bm25_ratio", "cosine_similarity"])
        sort_ascending.extend([True, True, True])

        df = df.sort_values(sort_cols, ascending=sort_ascending)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Quote Verification", index=False)

            # Access the worksheet to apply formatting
            worksheet = writer.sheets["Quote Verification"]

            # Apply text wrapping and set column widths
            from openpyxl.styles import Alignment, Font

            # Default font size increased by 20% (11pt -> 13pt)
            default_font = Font(size=13)

            for column in worksheet.columns:
                column_letter = column[0].column_letter
                header_value = column[0].value

                # Set width and wrapping for text columns
                if header_value in [
                    "extracted_quote",
                    "found_in_original",
                    "full_original_text",
                ]:
                    worksheet.column_dimensions[column_letter].width = 80
                    for cell in column:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                        cell.font = default_font
                elif header_value == "llm_explanation":
                    # LLM explanation - wide with wrapping
                    worksheet.column_dimensions[column_letter].width = 60
                    for cell in column:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                        cell.font = default_font
                elif header_value == "llm_is_contained":
                    # Boolean column - narrow
                    worksheet.column_dimensions[column_letter].width = 18
                    for cell in column:
                        cell.font = default_font
                elif header_value in [
                    "bm25_score",
                    "bm25_ratio",
                    "cosine_similarity",
                    "global_start",
                    "global_end",
                    "match_ratio",
                ]:
                    # Number columns - narrower
                    worksheet.column_dimensions[column_letter].width = 15
                    for cell in column:
                        cell.font = default_font
                elif header_value == "source_doc":
                    # Source document column - medium width
                    worksheet.column_dimensions[column_letter].width = 25
                    for cell in column:
                        cell.font = default_font
                else:
                    # Auto-width for other columns
                    max_length = 0
                    for cell in column:
                        if cell.value:
                            max_length = max(max_length, len(str(cell.value)))
                        cell.font = default_font
                    worksheet.column_dimensions[column_letter].width = min(
                        max_length + 2, 30
                    )
