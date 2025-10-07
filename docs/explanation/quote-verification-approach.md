# Quote Verification Approach

## Problem

LLMs may hallucinate, paraphrase, or truncate quotes during qualitative analysis. Quote verification locates extracted quotes in source documents and provides confidence metrics.

## Method

### Two-Stage Hybrid Approach

**Stage 1: BM25 (Lexical)**
- Segments documents into overlapping windows
- Builds BM25 index for fast retrieval
- Finds best matching window for each quote

**Stage 2: Embeddings (Semantic)**
- Computes cosine similarity between quote and matched span
- Validates semantic equivalence

BM25 narrows the search space efficiently; embeddings catch paraphrases.

### Ellipsis Handling

Quotes like `"beginning ... end"` are matched by:
1. Splitting on ellipsis pattern
2. Finding BM25 matches for head and tail fragments separately
3. Reconstructing the span between them
4. Applying gap constraints to prevent false matches

### Windowing

Documents are split into overlapping windows:
- Size: 1.1× longest quote (capped at 500 chars)
- Overlap: 30% of window size
- Position tracking enables source document reconstruction

### Span Refinement

Matched windows are trimmed to quote boundaries using fuzzy matching:
- Finds longest character-level matches
- Snaps to word boundaries
- Expands to neighbor windows if truncated at boundaries

## Output Metrics

| Metric | Meaning |
|--------|---------|
| `bm25_score` | Lexical match strength |
| `bm25_ratio` | Match uniqueness (top1/top2) |
| `cosine_similarity` | Semantic equivalence (0-1) |
| `match_ratio` | Boundary alignment quality |

**Interpretation:**
- High BM25 + High cosine = Verbatim quote
- Low BM25 + High cosine = Paraphrase (review)
- Low BM25 + Low cosine = Hallucination (reject)

## Configuration

**BM25 Parameters:**
- `k1=1.5`: Term frequency saturation
- `b=0.4`: Length normalization (lower than default to reduce penalties for variable-length quotes)

**Trimming:**
- Method: `fuzzy` (default), `sliding_bm25`, or `hybrid`
- `min_fuzzy_ratio=0.6`: Minimum match quality threshold

**Windows:**
- `expand_window_neighbors=1`: Search ±N windows if match appears truncated

## Validation Workflow

1. Run VerifyQuotes node
2. Review aggregate statistics (`mean_cosine`, `n_low_match_confidence`)
3. Inspect low-confidence matches in Excel export (sorted by confidence)
4. Manually verify or exclude problematic quotes

## Limitations

- Quotes exceeding 500 characters may fail
- English-centric tokenization (NLTK)
- Cannot definitively distinguish paraphrases from hallucinations
- Embedding computation scales linearly with quote count

## Academic Reporting

**Algorithm Description:**
> Quote verification used a two-stage hybrid approach. Source documents were segmented into overlapping windows (1.1× longest quote, 30% overlap). BM25 (k1=1.5, b=0.4) identified candidate spans. Quotes with ellipses were matched by locating head and tail fragments separately. Semantic similarity was computed using [embedding model] with cosine distance. Span boundaries were refined using fuzzy matching and snapped to word boundaries.

**Validation:**
> Matches were validated if cosine similarity exceeded [threshold] and BM25 ratio exceeded [threshold]. [X]% of quotes met these criteria. Low-confidence matches (n=[Y]) were manually reviewed.

**Reproducibility:**
- Report all parameter values
- Include aggregate statistics
- Archive Excel exports as supplementary materials

## Philosophy

This system supports human judgment rather than replacing it:
- Efficiently triages thousands of quotes
- Provides multiple confidence signals
- Surfaces edge cases for expert review
- Enables reproducibility through transparent parameters

## See Also

- [Quote Verification Algorithm](../reference/quote-verification.md) - Technical details
- [Node Reference](../reference/node-reference.md) - Parameters
