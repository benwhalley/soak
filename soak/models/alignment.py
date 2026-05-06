"""Alignment strategies for matching quotes to source text spans.

Provides multiple algorithms for finding where a quote aligns within a larger text span.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Literal, Tuple

import nltk
import numpy as np
from rank_bm25 import BM25Okapi

from .text_utils import snap_to_boundaries


@dataclass
class AlignmentResult:
    """Result of quote-to-span alignment.

    Attributes:
        start_char: Character offset where match starts in span
        end_char: Character offset where match ends in span (exclusive)
        match_ratio: Confidence score (0.0-1.0 for fuzzy, BM25 score for BM25)
        matched_text: The aligned text from the span
        method: Which alignment method was used
    """

    start_char: int
    end_char: int
    match_ratio: float
    matched_text: str
    method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backwards compatibility."""
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "match_ratio": self.match_ratio,
            "matched_text": self.matched_text,
        }


class AlignmentStrategy(ABC):
    """Abstract base class for quote alignment strategies."""

    @abstractmethod
    def align(self, quote: str, span: str, **kwargs) -> AlignmentResult:
        """Find best alignment of quote within span.

        Args:
            quote: The quote text to find
            span: The larger text span to search in
            **kwargs: Strategy-specific parameters

        Returns:
            AlignmentResult with match details
        """
        pass


class FuzzyAlignment(AlignmentStrategy):
    """Fuzzy string matching using difflib.SequenceMatcher."""

    def __init__(self, min_ratio: float = 0.6, context_pad: int = 30):
        """Initialize fuzzy alignment strategy.

        Args:
            min_ratio: Minimum match ratio to consider valid (0.0-1.0)
            context_pad: Characters of context to include around match
        """
        self.min_ratio = min_ratio
        self.context_pad = context_pad

    def align(self, quote: str, span: str, **kwargs) -> AlignmentResult:
        """Find alignment using fuzzy string matching.

        Uses difflib.SequenceMatcher to find best matching blocks.
        Fast path for exact substring matches.
        """
        # normalize for matching
        clean = lambda s: re.sub(r"\s+", " ", s.strip().lower())
        quote_clean = clean(quote)
        span_clean = clean(span)

        if not quote_clean or not span_clean:
            return AlignmentResult(
                start_char=0,
                end_char=0,
                match_ratio=0.0,
                matched_text="",
                method="fuzzy",
            )

        # fast path: exact substring
        if quote_clean in span_clean:
            offset = span_clean.index(quote_clean)
            start = offset
            end = offset + len(quote_clean)

            # snap to word boundaries
            start, end = snap_to_boundaries(span, start, end, snap_to="word")

            return AlignmentResult(
                start_char=start,
                end_char=end,
                match_ratio=1.0,
                matched_text=span[start:end],
                method="fuzzy",
            )

        # use SequenceMatcher to find matching blocks
        matcher = SequenceMatcher(None, span_clean, quote_clean)

        # get all matching blocks above minimum length
        blocks = [b for b in matcher.get_matching_blocks() if b[2] > 5]

        if not blocks:
            # fallback: no good match
            return AlignmentResult(
                start_char=0,
                end_char=len(span),
                match_ratio=0.0,
                matched_text=span,
                method="fuzzy",
            )

        # find the best block (longest contiguous match)
        best_block = max(blocks, key=lambda b: b[2])
        i1, i2, n = best_block
        match_ratio = matcher.ratio()

        # add context padding
        start = max(i1 - self.context_pad, 0)
        end = min(i1 + n + self.context_pad, len(span))

        # snap to word boundaries to avoid mid-word cuts
        start, end = snap_to_boundaries(span, start, end, snap_to="word")

        return AlignmentResult(
            start_char=start,
            end_char=end,
            match_ratio=float(match_ratio),
            matched_text=span[start:end],
            method="fuzzy",
        )


class BM25SlidingAlignment(AlignmentStrategy):
    """Word-level sliding window BM25 alignment."""

    def align(self, quote: str, span: str, **kwargs) -> AlignmentResult:
        """Find alignment using sliding BM25 over word tokens.

        Tokenizes both quote and span, then uses BM25 to find best
        matching window. More robust to paraphrasing than exact matching.
        """
        quote_tokens = nltk.word_tokenize(quote.lower())
        span_tokens = nltk.word_tokenize(span.lower())

        if len(quote_tokens) > len(span_tokens):
            return AlignmentResult(
                start_char=0,
                end_char=0,
                match_ratio=0.0,
                matched_text="",
                method="bm25",
            )

        # build windows
        window_size = max(len(quote_tokens), int(len(quote_tokens) * 1.2))
        windows = []
        window_starts = []  # token positions

        for i in range(len(span_tokens) - window_size + 1):
            windows.append(span_tokens[i : i + window_size])
            window_starts.append(i)

        if not windows:
            return AlignmentResult(
                start_char=0,
                end_char=0,
                match_ratio=0.0,
                matched_text="",
                method="bm25",
            )

        # score with BM25
        bm25 = BM25Okapi(windows)
        scores = bm25.get_scores(quote_tokens)
        best_idx = int(np.argmax(scores))
        best_token_offset = (
            window_starts[best_idx] if best_idx < len(window_starts) else 0
        )

        # convert token offset to character offset (approximate)
        if best_token_offset > 0:
            char_offset = (
                len(" ".join(span_tokens[:best_token_offset])) + 1
            )  # +1 for space
        else:
            char_offset = 0

        # snap to word boundaries
        start, end = char_offset, len(span)
        start, end = snap_to_boundaries(span, start, end, snap_to="word")

        return AlignmentResult(
            start_char=start,
            end_char=end,
            match_ratio=float(scores[best_idx]),
            matched_text=span[start:end],
            method="bm25",
        )


class HybridAlignment(AlignmentStrategy):
    """Hybrid strategy: try fuzzy first, fall back to BM25 if needed."""

    def __init__(self, min_ratio: float = 0.6, context_pad: int = 30):
        """Initialize hybrid alignment strategy.

        Args:
            min_ratio: Minimum fuzzy ratio before falling back to BM25
            context_pad: Characters of context for fuzzy matching
        """
        self.fuzzy = FuzzyAlignment(min_ratio=min_ratio, context_pad=context_pad)
        self.bm25 = BM25SlidingAlignment()
        self.min_ratio = min_ratio

    def align(self, quote: str, span: str, **kwargs) -> AlignmentResult:
        """Try fuzzy matching first, fall back to BM25 if confidence is low."""
        # try fuzzy first
        result = self.fuzzy.align(quote, span)

        if result.match_ratio < self.min_ratio:
            # fall back to BM25
            bm25_result = self.bm25.align(quote, span)

            # use BM25 result if it seems reasonable
            # heuristic: don't use if match starts too far into span
            if bm25_result.start_char < len(span) * 0.5:
                bm25_result.method = "hybrid(bm25)"
                return bm25_result

            # keep fuzzy result but mark as low confidence
            result.method = "hybrid(fuzzy-low)"
            return result

        result.method = "hybrid(fuzzy)"
        return result


def trim_span_to_quote(
    quote: str,
    span: str,
    method: Literal["fuzzy", "sliding_bm25", "hybrid"] = "hybrid",
    min_fuzzy_ratio: float = 0.6,
    min_span_length_multiplier: float = 1.2,
    context_pad: int = 30,
) -> Dict[str, Any]:
    """Trim span to align with quote start using specified matching method.

    Args:
        quote: The quote text to find
        span: The larger text span to search in
        method: Which alignment method to use
        min_fuzzy_ratio: Minimum ratio for fuzzy matching
        min_span_length_multiplier: Safety multiplier for minimum result length
        context_pad: Context padding for fuzzy matching

    Returns:
        Dict with: matched_text, start_char, end_char, match_ratio
    """
    if not span or not quote:
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span) if span else 0,
            "match_ratio": 0.0,
        }

    # don't trim very short quotes (unreliable)
    if len(quote) < 20:
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": 0.0,
        }

    # get appropriate strategy
    if method == "fuzzy":
        strategy = FuzzyAlignment(min_ratio=min_fuzzy_ratio, context_pad=context_pad)
    elif method == "sliding_bm25":
        strategy = BM25SlidingAlignment()
    elif method == "hybrid":
        strategy = HybridAlignment(min_ratio=min_fuzzy_ratio, context_pad=context_pad)
    else:
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": 0.0,
        }

    result = strategy.align(quote, span)

    # safety: don't trim if result would be too short (preserve some context)
    min_result_length = int(len(quote) * min_span_length_multiplier)
    matched_len = result.end_char - result.start_char
    if matched_len < min_result_length and matched_len < len(span):
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": result.match_ratio,
        }

    return result.to_dict()


