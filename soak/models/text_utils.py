"""Text processing utilities for windowing, boundary detection, and document tracking.

Shared utilities used across nodes (primarily VerifyQuotes, but reusable elsewhere).
"""

import re
from typing import Any, Dict, List, Literal, Optional, Tuple


ELLIPSIS_RE = re.compile(r"\.{2,4}|…")


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
            else (doc.id if hasattr(doc, "id") else getattr(doc, "path", "unknown"))
        )
        # fall back to id or path attribute
        if not doc_name:
            doc_name = doc.id if hasattr(doc, "id") else getattr(doc, "path", "unknown")

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
        # define word boundary characters (whitespace and punctuation)
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

        # expand left until we hit a boundary (or reach start of text)
        while start > 0 and text[start - 1] not in boundaries:
            start -= 1

        # expand right until we hit a boundary (or reach end of text)
        # note: end is exclusive (text[start:end]), so we check text[end] if it exists
        while end < len(text) and text[end] not in boundaries:
            end += 1

        # trim leading whitespace
        while start < end and text[start] in (" ", "\n", "\t", "\r"):
            start += 1

        # trim trailing whitespace
        while end > start and text[end - 1] in (" ", "\n", "\t", "\r"):
            end -= 1

    elif snap_to == "sentence":
        # find sentence boundaries (periods, exclamation, question marks followed by space/newline)
        sentence_pattern = r"[.!?][\s\n]+"
        sentence_ends = [m.end() for m in re.finditer(sentence_pattern, text)]

        # expand left to previous sentence end (or beginning)
        prev_end = 0
        for pos in sentence_ends:
            if pos < start:
                prev_end = pos
            else:
                break
        start = prev_end

        # expand right to next sentence end (or end of text)
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

    # match starts at/near beginning → might be left-truncated
    starts_at_boundary = start_char < boundary_threshold

    # match ends at/near end → might be right-truncated
    ends_at_boundary = end_char > len(span_text) - boundary_threshold

    return starts_at_boundary or ends_at_boundary


def extract_context_window(
    quote_text: str,
    source_doc_content: str,
    global_start: int,
    global_end: int,
    context_window_size: int = 1000,
) -> str:
    """Extract a fixed-size context window centered on the quote.

    Used primarily for LLM-based fairness checks.

    Args:
        quote_text: The quote text (for reference, not used in calculation)
        source_doc_content: Full source document content
        global_start: Character position where quote starts in concatenated corpus
        global_end: Character position where quote ends in concatenated corpus
        context_window_size: Total size of context window (default 1000 chars)

    Returns:
        Context string centered on the quote (or full document if shorter than window)
    """
    # if document is shorter than window, return full document
    if len(source_doc_content) <= context_window_size:
        return source_doc_content

    # calculate center of quote
    quote_center = (global_start + global_end) // 2

    # calculate window bounds
    half_window = context_window_size // 2
    window_start = max(0, quote_center - half_window)
    window_end = min(len(source_doc_content), quote_center + half_window)

    # adjust if we hit document boundaries
    if window_start == 0:
        window_end = min(len(source_doc_content), context_window_size)
    elif window_end == len(source_doc_content):
        window_start = max(0, len(source_doc_content) - context_window_size)

    return source_doc_content[window_start:window_end]
