"""Utility functions for node operations.

Common helpers used across multiple node types to reduce code duplication.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from soak.models.base import Code, CodeList, Quote, QuoteReference, safe_json_dump

logger = logging.getLogger(__name__)


def extract_output_dict(
    output_item, serialize_iterables: bool = True
) -> Dict[str, Any]:
    """Extract output dictionary from a ChatterResult or similar output item.

    Handles both output objects with .outputs attribute and plain dicts,
    normalizing them to a consistent dict format for DataFrame export.

    Args:
        output_item: ChatterResult, dict, or other output object
        serialize_iterables: If True, convert lists/tuples to strings for CSV export

    Returns:
        Dict with output key-value pairs
    """
    # extract outputs based on type
    if hasattr(output_item, "outputs"):
        output_dict = output_item.outputs
    elif isinstance(output_item, dict):
        # filter out private/internal keys
        output_dict = {k: v for k, v in output_item.items() if not k.startswith("__")}
    else:
        return {}

    # serialize iterables if requested (for CSV/DataFrame compatibility)
    if serialize_iterables:
        result = {}
        for k, v in output_dict.items():
            if isinstance(v, (list, tuple)) or (
                hasattr(v, "__iter__") and not isinstance(v, str)
            ):
                result[k] = str(v)
            else:
                result[k] = v
        return result

    return output_dict


def export_chatter_result(result, folder: Path, file_prefix: str) -> None:
    """Export ChatterResult prompt, response, and JSON to files.

    Standardizes the export pattern used across multiple node types.
    Safely handles missing attributes and logs warnings on failure.

    Additionally exports structured responses (Code/Theme/etc.) to CSV if detected.

    Args:
        result: ChatterResult object to export
        folder: Directory to write files to
        file_prefix: Prefix for output filenames (e.g., "0001_item")
    """
    try:
        # export prompt if available
        if hasattr(result, "results") and result.results:
            first_seg = next(iter(result.results.values()), None)
            if first_seg and hasattr(first_seg, "prompt"):
                prompt_file = folder / f"{file_prefix}_prompt.md"
                prompt_file.write_text(first_seg.prompt)

        # export response text if available
        if hasattr(result, "response"):
            response_text = str(result.response)
            response_file = folder / f"{file_prefix}_response.txt"
            response_file.write_text(response_text)

            # export structured responses to CSV
            _export_structured_response_to_csv(result.response, folder, file_prefix)

        # always export JSON
        json_file = folder / f"{file_prefix}_response.json"
        json_file.write_text(safe_json_dump(result))

    except Exception as e:
        logger.warning(f"Failed to export ChatterResult for {file_prefix}: {e}")


def export_slots_as_text_files(result, folder: Path) -> None:
    """Export each response slot as a separate text file.

    For Transform nodes where we have one ChatterResult with multiple slots,
    this creates one .txt file per slot instead of a combined response.txt.

    Args:
        result: ChatterResult object to export
        folder: Directory to write files to
    """
    try:
        if not hasattr(result, "results") or not result.results:
            logger.warning("No output slots found in ChatterResult")
            return

        # export each slot as a separate text file
        for slot_name, segment_result in result.results.items():
            slot_file = folder / f"{slot_name}.txt"
            slot_file.write_text(str(segment_result))

    except Exception as e:
        logger.warning(f"Failed to export slots as text files: {e}")


def _export_structured_response_to_csv(
    response, folder: Path, file_prefix: str
) -> None:
    """Export structured responses (Code/Theme/etc.) to CSV.

    Detects if response is Code, CodeList, Theme, or Themes and exports to CSV.

    Args:
        response: The response object (may be Code, CodeList, Theme, Themes, or other)
        folder: Directory to write files to
        file_prefix: Prefix for output filenames
    """
    import pandas as pd

    from soak.models.base import Code, CodeList, Theme, Themes

    try:
        # detect type and export accordingly
        if isinstance(response, CodeList):
            _export_codes_to_csv(response.codes, folder, file_prefix)
        elif isinstance(response, Code):
            _export_codes_to_csv([response], folder, file_prefix)
        elif isinstance(response, Themes):
            _export_themes_to_csv(response.themes, folder, file_prefix)
        elif isinstance(response, Theme):
            _export_themes_to_csv([response], folder, file_prefix)
    except Exception as e:
        logger.debug(f"Could not export structured response to CSV: {e}")


def _export_codes_to_csv(codes: List, folder: Path, file_prefix: str) -> None:
    """Export list of Code objects to CSV.

    Args:
        codes: List of Code objects
        folder: Directory to write files to
        file_prefix: Prefix for output filename
    """
    import pandas as pd

    from soak.export_utils import export_to_csv

    rows = []
    for code in codes:
        # extract quotes as text list
        quotes_text = []
        if hasattr(code, "all_quotes"):
            for q in code.all_quotes:
                quotes_text.append(q.text if hasattr(q, "text") else str(q))
        elif hasattr(code, "quotes"):
            for q in code.quotes:
                if hasattr(q, "text"):
                    quotes_text.append(q.text)
                else:
                    quotes_text.append(str(q))

        rows.append(
            {
                "slug": code.slug if hasattr(code, "slug") else "",
                "name": code.name if hasattr(code, "name") else "",
                "description": code.description if hasattr(code, "description") else "",
                "quotes": " | ".join(quotes_text),  # pipe-separated for CSV
                "num_quotes": len(quotes_text),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        csv_file = folder / f"{file_prefix}_codes.csv"
        export_to_csv(df, csv_file)


def _export_themes_to_csv(themes: List, folder: Path, file_prefix: str) -> None:
    """Export list of Theme objects to CSV.

    Args:
        themes: List of Theme objects
        folder: Directory to write files to
        file_prefix: Prefix for output filename
    """
    import pandas as pd

    from soak.export_utils import export_to_csv

    rows = []
    for theme in themes:
        # extract code slugs
        code_slugs = []
        if hasattr(theme, "code_slugs"):
            code_slugs = theme.code_slugs

        rows.append(
            {
                "name": theme.name if hasattr(theme, "name") else "",
                "description": (
                    theme.description if hasattr(theme, "description") else ""
                ),
                "code_slugs": ", ".join(code_slugs),  # comma-separated
                "num_codes": len(code_slugs),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        csv_file = folder / f"{file_prefix}_themes.csv"
        export_to_csv(df, csv_file)


# Quote provenance post-processing functions


def post_process_code_quotes(code: Code, context: Dict[str, Any]):
    """Post-process quotes for a Code object.

    Mode 1 (Primary Extraction): Capture source from context for Quote objects
    Mode 2 (Rationalization): Resolve QuoteReference objects to Quote objects

    Raises:
        QuoteProvenanceError: If quotes are extracted after Reduce (multiple sources)
    """
    from soak.models.base import QuoteProvenanceError, TrackedItem

    resolved = []

    # Check if we have existing codes in context
    input_codes = collect_input_codes(context)

    if not input_codes:
        # Mode 1: Primary extraction (no existing codes in context)
        # Determine source from context
        source = None

        # Try to extract source from TrackedItem in context
        for value in context.values():
            if isinstance(value, TrackedItem):
                # Check provenance - quotes require single source
                if len(value.sources) > 1:
                    raise QuoteProvenanceError(
                        f"Cannot extract quotes with multiple sources ({len(value.sources)} sources). "
                        f"This Code was extracted after a Reduce node, which combines multiple sources. "
                        f"Use QuoteReference (with hash) instead of Quote in Map nodes after Reduce. "
                        f"Sources: {value.sources[:5]}{'...' if len(value.sources) > 5 else ''}"
                    )
                # Use the item's ID as source (single source context)
                source = value.id
                break

        # Fallback: check for source_id key (backward compatibility)
        if not source:
            source = context.get("source_id", "")

        for q in code.quotes:
            if isinstance(q, Quote):
                # Update source from context
                q.source = source
                resolved.append(q.model_dump())
            elif isinstance(q, QuoteReference):
                # Shouldn't happen in primary extraction, but handle gracefully
                logger.warning(
                    f"Unexpected QuoteReference in primary extraction: {q.hash}"
                )
                resolved.append(
                    Quote(
                        text=f"[unresolved: {q.hash}]", source=source or ""
                    ).model_dump()
                )
    else:
        # Mode 2: Rationalization (codes exist in context)
        # Build hash -> Quote lookup
        quote_by_hash = build_quote_lookup(input_codes)

        for q in code.quotes:
            if isinstance(q, Quote):
                # Quote object in rationalization mode - capture source if missing
                # This shouldn't normally happen (quotes should come via QuoteReference)
                if not q.source:
                    logger.warning(
                        f"Quote without source in rationalization mode: {q.text[:50]}"
                    )
                    q.source = ""
                resolved.append(q.model_dump())
            elif isinstance(q, QuoteReference):
                # Resolve reference to Quote
                matched_quote = resolve_quote_reference(q, quote_by_hash)
                resolved.append(matched_quote.model_dump())

    code.resolved_quotes = resolved


def build_quote_lookup(input_codes: List[Code]) -> Dict[str, Quote]:
    """Build hash -> Quote mapping from input codes."""
    import os

    quote_by_hash = {}
    collisions = []

    for ic in input_codes:
        for q in ic.all_quotes:
            q_hash = q.hash()

            if q_hash in quote_by_hash:
                # Check if this is a true duplicate or a hash collision
                existing = quote_by_hash[q_hash]

                # True duplicate: same text AND same source
                if existing.text == q.text and existing.source == q.source:
                    # This is the same quote referenced by multiple codes - skip gracefully
                    logger.debug(
                        f"Skipping duplicate quote: hash={q_hash}, text='{q.text[:50]}...'"
                    )
                    continue  # Don't add to lookup again, don't count as collision

                # Hash collision: different text or source but same hash
                collisions.append(
                    {
                        "hash": q_hash,
                        "existing_text": existing.text[:80],
                        "new_text": q.text[:80],
                        "existing_source": existing.source,
                        "new_source": q.source,
                    }
                )
                logger.error(f"Hash collision detected: {q_hash}")
                logger.error(
                    f"  Existing: '{existing.text[:60]}...' (source: {existing.source})"
                )
                logger.error(f"  New: '{q.text[:60]}...' (source: {q.source})")
                # Continue to overwrite (keeps last occurrence)

            quote_by_hash[q_hash] = q
            logger.debug(
                f"  Added quote to lookup: hash={q_hash}, text='{q.text[:50]}...', source={q.source}"
            )

    if collisions:
        error_msg = (
            f"Hash collision detected: {len(collisions)} quote(s) have duplicate hashes. "
            f"Increase QUOTE_HASH_LENGTH (currently {os.getenv('QUOTE_HASH_LENGTH', '6')}) "
            f"by setting environment variable: export QUOTE_HASH_LENGTH=8"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"build_quote_lookup: Built lookup with {len(quote_by_hash)} quotes")
    return quote_by_hash


def resolve_quote_reference(
    ref: QuoteReference, quote_by_hash: Dict[str, Quote], threshold: float = 0.85
) -> Quote:
    """Resolve QuoteReference to Quote by hash.

    Falls back to fuzzy matching if exact hash not found.
    """
    # Try exact hash match
    if ref.hash in quote_by_hash:
        return quote_by_hash[ref.hash]

    # Fallback: fuzzy match (in case of minor hash differences)
    # This shouldn't normally happen, but provides robustness
    logger.warning(f"Quote hash not found: {ref.hash}, trying fuzzy match")

    from difflib import SequenceMatcher

    best_match = None
    best_ratio = threshold

    for h, quote in quote_by_hash.items():
        ratio = SequenceMatcher(None, ref.hash, h).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = quote

    if best_match:
        logger.info(f"Fuzzy matched {ref.hash} to {best_match.hash()}")
        return best_match

    # No match found - create placeholder
    logger.error(f"Could not resolve quote reference: {ref.hash}")
    return Quote(text=f"[unresolved reference: {ref.hash}]")


def post_process_theme_code_refs(theme, context: Dict[str, Any]):
    """Post-process Theme.code_slugs to match actual Code objects.

    Fuzzy matches slugs to Codes in context and stores resolved references.
    """
    # Collect all codes from context
    input_codes = collect_input_codes(context)
    if not input_codes:
        return

    # Match each slug
    matched_codes = []
    for slug in theme.code_slugs:
        matched_code = fuzzy_match_code_slug(slug, input_codes)
        if matched_code:
            # Store complete Code data including quotes and resolved_quotes
            code_data = matched_code.model_dump()
            matched_codes.append(code_data)

    theme.resolved_code_refs = matched_codes


def fuzzy_match_code_slug(
    target_slug: str, candidates: List[Code], threshold: float = 0.85
) -> Optional[Code]:
    """Find best matching Code by slug using edit distance."""
    from difflib import SequenceMatcher

    target_clean = target_slug.strip().lower()
    best_match = None
    best_ratio = threshold

    for candidate in candidates:
        candidate_clean = candidate.slug.strip().lower()
        ratio = SequenceMatcher(None, target_clean, candidate_clean).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    return best_match


def collect_input_codes(context: Dict[str, Any]) -> List[Code]:
    """Collect all Code objects from context."""
    codes = []

    logger.debug(f"collect_input_codes: Searching context with {len(context)} keys")

    for key, value in context.items():
        if isinstance(value, CodeList):
            codes.extend(value.codes)
        elif isinstance(value, list):
            # Handle list of ChatterResults (from Map nodes)
            for item in value:
                if hasattr(item, "outputs"):
                    # Find CodeList in outputs
                    for output_val in item.outputs.values():
                        if isinstance(output_val, CodeList):
                            codes.extend(output_val.codes)

    logger.debug(f"collect_input_codes: Found total of {len(codes)} codes")
    return codes
