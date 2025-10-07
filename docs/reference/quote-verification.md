# Quote Verification Algorithm

## Overview

The `VerifyQuotes` node locates extracted quotes in source documents using BM25 lexical search and embedding-based semantic similarity. See [Quote Verification Approach](../explanation/quote-verification-approach.md) for design rationale.

## Algorithm

### Stage 1: Lexical Search (BM25)

1. **Windowing**: Split documents into overlapping windows
   - Size: 1.1× longest quote (capped at 500 chars)
   - Overlap: 30% of window size
   - Each window tracks global character positions

2. **Indexing**: Build BM25 index using NLTK word tokenization (lowercased)

3. **Matching**:
   - **Non-ellipsis quotes**: Find best BM25 window
   - **Ellipsis quotes** (e.g., `"beginning ... end"`):
     - Split on pattern `\.{3,}|…`
     - Match head and tail fragments separately
     - Reconstruct span from head start to tail end
     - Apply gap constraint if configured

4. **Span Refinement** (optional):
   - Trim window to quote boundaries using fuzzy matching
   - Snap to word boundaries
   - Expand to ±N neighbor windows if match appears truncated

### Stage 2: Semantic Verification (Embeddings)

- Compute embeddings for quote and matched span
- Calculate cosine similarity
- Return combined metrics


## Output Schema

```python
{
    "quote": str,              # Extracted quote
    "source_doc": str,         # Source document name
    "span_text": str,          # Matched text
    "context_window": str,     # ±300 chars around match
    "bm25_score": float,       # Lexical score
    "bm25_ratio": float,       # top1/(top2+ε)
    "global_start": int,       # Character position
    "global_end": int,         # End position
    "match_ratio": float,      # Fuzzy match quality (if trimming)
    "cosine_similarity": float # Embedding similarity (0-1)
}
```

## Configuration

```yaml
type: VerifyQuotes
window_size: null           # Auto: 1.1 × max quote, cap 500
overlap: null               # Auto: 30% of window_size
bm25_k1: 1.5                # Term frequency saturation
bm25_b: 0.4                 # Length normalization
ellipsis_max_gap: null      # Max windows between head/tail
trim_spans: true            # Enable span refinement
trim_method: "fuzzy"        # "fuzzy" | "sliding_bm25" | "hybrid"
min_fuzzy_ratio: 0.6        # Minimum match quality
expand_window_neighbors: 1  # Search ±N windows if truncated
```

### Trimming Methods

**Fuzzy** (default):
- Uses `difflib.SequenceMatcher` to find longest matches
- Adds ±30 char context padding
- Snaps to word boundaries

**Sliding BM25**:
- Word-level sliding window BM25 within span
- Finds best alignment offset

**Hybrid**:
- Fuzzy first
- Falls back to BM25 if `match_ratio < min_fuzzy_ratio`

## Statistics

Aggregate metrics computed across all quotes:

```python
{
    "n_quotes": int,
    "n_with_ellipses": int,
    "mean_bm25_score": float,
    "median_bm25_score": float,
    "mean_bm25_ratio": float,
    "median_bm25_ratio": float,
    "mean_cosine": float,
    "median_cosine": float,
    "min_cosine": float,
    "max_cosine": float,
    "mean_match_ratio": float,
    "n_low_match_confidence": int,
    "mean_span_length": float
}
```

## Export Formats

**Excel** (`quote_verification.xlsx`):
- Sorted by confidence (lowest first)
- Text columns: 80 chars wide, wrapped
- Number columns: 15 chars wide

**CSV** (`stats.csv`):
- Melted format (variable, value)

## Helper Functions

### `make_windows(text, window_size, overlap, extracted_sentences)`
Creates overlapping windows with position tracking.

Returns: `List[Tuple[str, int, int]]` (text, start_pos, end_pos)

### `verify_quotes_bm25_first(extracted_sentences, windows, text, doc_boundaries, get_embedding, ...)`
Main verification function.

Returns: `List[Dict[str, Any]]` with metrics for each quote

### `trim_span_to_quote(quote, span, method, min_fuzzy_ratio)`
Refines window boundaries to align with quote.

Returns: `Dict` with matched_text, start_char, end_char, match_ratio

### `find_alignment_fuzzy(quote, span, min_ratio, context_pad)`
Fuzzy matching using `SequenceMatcher.get_matching_blocks()`.

### `find_alignment_sliding_bm25(quote, span)`
Word-level BM25 sliding window alignment.

### `snap_to_boundaries(text, start, end, snap_to="word")`
Expands indices to nearest word/sentence boundaries.

### `is_match_truncated(match_result, span_text, boundary_threshold=30)`
Detects if match touches window boundaries (potential truncation).

### `create_document_boundaries(documents)`
Builds position map for document tracking (assumes `\n\n` separator).

### `find_source_document(position, doc_boundaries)`
Maps character position to source document name.

## Academic Reporting

**Methods Section Template:**

> Quote verification employed a two-stage hybrid approach. Source documents were segmented into overlapping windows (adaptive sizing: 1.1× longest quote, 30% overlap). BM25 (Okapi, k1=1.5, b=0.4) identified lexically similar spans for each extracted quote. Quotes containing ellipses were handled by separately matching head and tail fragments and reconstructing the intervening span. Semantic similarity between quotes and matched spans was computed using [embedding model] with cosine distance. Span boundaries were refined using fuzzy matching (Ratcliff-Obershelp algorithm) and snapped to word boundaries.

**Validation Template:**

> Matches were considered validated if cosine similarity exceeded [threshold] and BM25 ratio exceeded [threshold]. [X]% of quotes met these criteria. Low-confidence matches (n=[Y]) were manually reviewed, with [Z]% confirmed as paraphrases and [W]% excluded as hallucinations.

**Reproducibility Checklist:**
- Report all parameter values
- Include aggregate statistics (mean/median scores)
- Describe manual review process
- Archive Excel exports as supplementary materials

## See Also

- [Quote Verification Approach](../explanation/quote-verification-approach.md) - Design rationale
- [Node Reference](node-reference.md) - VerifyQuotes parameters
- [Provenance Tracking](../explanation/provenance-tracking.md) - Document lineage
