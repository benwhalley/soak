"""Text processing utilities for windowing, boundary detection, and document tracking.

Shared utilities used across nodes (primarily VerifyQuotes, but reusable elsewhere).
"""

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

ELLIPSIS_RE = re.compile(r"\.{2,4}|…")


# Note: Manual escaping of struckdown syntax has been removed.
# Escaping is now handled automatically by Jinja2's finalize function in struckdown package.
# See struckdown.struckdown_finalize() and soak.models.dag.render_strict_template()


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
    text: str,
    start: int,
    end: int,
    snap_to: Literal["word", "sentence"] = "word",
    max_expansion: Optional[int] = None,
) -> Tuple[int, int]:
    """Expand start/end to nearest word or sentence boundary.

    Prevents ugly mid-word cuts by snapping outward to natural boundaries.
    Expansion is capped by max_expansion to avoid runaway growth on boundary-free text.

    Args:
        text: Full text
        start: Start index (inclusive)
        end: End index (exclusive, as in text[start:end])
        snap_to: Snap to "word" or "sentence" boundaries
        max_expansion: Maximum chars to expand in each direction.
            Defaults to 20 for word, 200 for sentence.

    Examples:

        Word boundary snapping expands to complete words:

        >>> snap_to_boundaries("the quick brown fox jumps", 5, 14)
        (4, 15)
        >>> "the quick brown fox jumps"[4:15]
        'quick brown'

        Mid-word positions snap outward to complete the word:

        >>> snap_to_boundaries("foo bar baz", 5, 6)
        (4, 7)
        >>> "foo bar baz"[4:7]
        'bar'

        Already on a boundary -- no change:

        >>> snap_to_boundaries("one two three", 0, 3)
        (0, 3)

        Sentence boundary snapping expands to full sentences:

        >>> snap_to_boundaries("First sentence. Second one. Third.", 17, 22, snap_to="sentence")
        (16, 28)
        >>> "First sentence. Second one. Third."[16:28]
        'Second one. '

        max_expansion limits how far we look for a boundary:

        >>> snap_to_boundaries("abcdefghij klmno", 2, 5, max_expansion=3)
        (0, 5)

        Text start counts as a boundary, but the space at index 10 is too far right:

        >>> snap_to_boundaries("abcdefghij klmno", 8, 9, max_expansion=1)
        (8, 9)
    """
    if max_expansion is None:
        max_expansion = 20 if snap_to == "word" else 200

    # NOTE: word boundary chars are ASCII-centric; adequate for English research text.

    if snap_to == "word":
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

        orig_start, orig_end = start, end

        # expand left until we hit a boundary char or text start
        new_start = start
        found_left = False
        while new_start > 0 and (orig_start - new_start) < max_expansion:
            if text[new_start - 1] in boundaries:
                found_left = True
                break
            new_start -= 1
        # accept if we found a boundary char, or reached text start within the limit
        if found_left or (new_start == 0 and (orig_start - new_start) <= max_expansion):
            start = new_start

        # expand right until we hit a boundary char or text end
        new_end = end
        found_right = False
        while new_end < len(text) and (new_end - orig_end) < max_expansion:
            if text[new_end] in boundaries:
                found_right = True
                break
            new_end += 1
        # accept if we found a boundary char, or reached text end within the limit
        if found_right or (new_end == len(text) and (new_end - orig_end) <= max_expansion):
            end = new_end

        # trim leading whitespace
        while start < end and text[start] in (" ", "\n", "\t", "\r"):
            start += 1

        # trim trailing whitespace
        while end > start and text[end - 1] in (" ", "\n", "\t", "\r"):
            end -= 1

    elif snap_to == "sentence":
        from pysbd import Segmenter

        seg = Segmenter(language="en", clean=False, char_span=True)
        spans = seg.segment(text)

        # find the sentence boundary edges from pysbd spans
        sentence_ends = [s.end for s in spans]

        # expand left to previous sentence end (or beginning), within max_expansion
        prev_end = max(0, start - max_expansion)
        for pos in sentence_ends:
            if pos <= start and (start - pos) <= max_expansion:
                prev_end = pos
            elif pos > start:
                break
        start = prev_end

        # expand right to next sentence end (or end of text), within max_expansion
        next_end = min(len(text), end + max_expansion)
        for pos in sentence_ends:
            if pos >= end:
                if (pos - end) <= max_expansion:
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
    """Extract a fixed-size context window centered on a quote from a text.

    Used primarily for LLM-based fairness checks.

    Args:
        quote_text: The quote text (for reference, not used in calculation)
        source_doc_content: Full source document content
        global_start: Character position where quote starts in concatenated corpus
        global_end: Character position where quote ends in concatenated corpus
        context_window_size: Total size of context window (default 1000 chars)

    Returns:
        Context string centered on the quote (or full document if shorter than window)

    Examples:

        Short document returned in full:

        >>> extract_context_window("hello", "short text", 0, 5, context_window_size=100)
        'short text'

        Window centred on the quote position:

        >>> doc = "A" * 100 + "QUOTE" + "B" * 100
        >>> result = extract_context_window("QUOTE", doc, 100, 105, context_window_size=20)
        >>> len(result)
        20
        >>> "QUOTE" in result
        True

        Quote near start -- window anchored at beginning:

        >>> doc = "XY" + "Z" * 50
        >>> extract_context_window("XY", doc, 0, 2, context_window_size=10)
        'XYZZZZZZZZ'
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
