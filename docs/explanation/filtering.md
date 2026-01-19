# Filtering and Chunk Overlap

This document explains how the Filter node works and how soak handles overlapping chunks to avoid text duplication when reconstructing documents.

## Overview

The Filter node allows you to selectively process document chunks based on relevance criteria. When combined with Split nodes that use overlap, the system ensures that filtered and recombined text does not contain duplicated content from overlapping regions.

## Filter Node Basics

The Filter node evaluates each item against a boolean expression and either keeps the original content or replaces it with placeholder text:

```yaml
- name: filtered_chunks
  type: Filter
  template: filter_template.sd
  expression: "is_relevant == True"
  omitted_text: "[.. text omitted ..]"
  inputs:
    - chunks
```

### Key behaviours

1. **Items are not removed** -- filtered items remain in the list but their content is replaced with `omitted_text`
2. **Provenance is preserved** -- the item's ID, sources, and metadata are retained (with `filtered: True` added)
3. **Position is maintained** -- the document structure stays intact, allowing proper reconstruction

### Filter modes

**LLM mode** (default when template provided):
- Runs the template through an LLM to extract fields
- Evaluates the expression against extracted fields

**Simple mode** (when no template):
- Evaluates the expression directly against item content or metadata

```yaml
# LLM mode - uses template to determine relevance
- name: relevance_filter
  type: Filter
  template: check_relevance.sd
  expression: "is_relevant == True"
  inputs:
    - chunks

# Simple mode - filter on content length directly
- name: length_filter
  type: Filter
  expression: "len(input) > 100"
  inputs:
    - chunks
```

## Handling Overlapping Chunks

When splitting documents for LLM processing, overlap between chunks provides context continuity. However, this creates a challenge: when recombining chunks, overlapping regions could appear twice.

### How Split creates overlap metadata

The Split node tracks which portion of each chunk is "core" content versus overlap:

```yaml
- name: chunks
  type: Split
  split_unit: words
  chunk_size: 100
  overlap: 25
  inputs:
    - documents
```

Each chunk stores a `content_excluding_overlap` tuple indicating the character positions of core content:

- **First chunk**: Full content is core (no preceding overlap)
- **Middle chunks**: Core content starts after the overlap region
- **Last chunk**: Core content starts after the overlap region

### How Reduce handles overlap

The Reduce node has `exclude_overlap: True` by default. When joining chunks:

```yaml
- name: recombined
  type: Reduce
  inputs:
    - filtered_chunks
```

For each chunk, Reduce calls `get_core_content()` which returns only the non-overlapping portion. This prevents duplication.

### Visual example

Consider a document split into 3 chunks with 25-word overlap:

```
Original: "The patient reported significant improvement after starting the new treatment
           protocol. Sleep quality improved first, followed by energy levels. By week
           four, most symptoms had resolved completely."

Chunk 1: "The patient reported significant improvement after starting the new treatment
          protocol. Sleep quality improved"
          [_______________ core content _______________][overlap]

Chunk 2: "protocol. Sleep quality improved first, followed by energy levels. By week four,"
          [overlap][_____________ core content _____________][overlap]

Chunk 3: "energy levels. By week four, most symptoms had resolved completely."
          [overlap][____________ core content ____________]
```

When reduced, only core content from each chunk is joined:

```
Reduced: "The patient reported significant improvement after starting the new treatment
          protocol. Sleep quality improved first, followed by energy levels. By week
          four, most symptoms had resolved completely."
```

## Complete Pipeline Example

A typical pre-filtering pipeline for qualitative analysis:

```yaml
name: filtered_thematic_analysis
default_context:
  research_question: Understanding patient recovery experiences

nodes:
  # Split documents into overlapping chunks for context
  - name: chunks
    type: Split
    split_unit: words
    chunk_size: 100
    min_split: 50
    overlap: 25
    inputs:
      - by_document

  # Filter to relevant content only
  - name: filtered_chunks
    type: Filter
    template: relevance_check.sd
    expression: "is_relevant == True"
    omitted_text: "[.. text omitted ..]"
    inputs:
      - chunks

  # Recombine into documents (overlap excluded automatically)
  - name: relevant_docs
    type: Reduce
    inputs:
      - filtered_chunks

  # Continue with analysis on filtered documents
  - name: codes
    type: Map
    template: coding.sd
    inputs:
      - relevant_docs
```

With a filter template like:

```
---#relevance_check.sd
You are reviewing text for relevance to: {{research_question}}

Text to evaluate:
{{input}}

Is this text relevant to the research question?

[[bool:is_relevant]]
```

## What happens to filtered chunks

When a chunk fails the filter expression:

1. Its content is replaced with `omitted_text` (e.g., `"[.. text omitted ..]"`)
2. The item stays in the output list at its original position
3. Metadata is preserved with `filtered: True` added
4. The `content_excluding_overlap` tuple is not preserved (not needed for placeholder text)

When the Reduce node processes filtered chunks:
- Kept chunks: core content (excluding overlap) is used
- Filtered chunks: the placeholder text is used in full

This results in a reconstructed document with irrelevant sections replaced by placeholders, and no duplication from overlapping regions.

## Configuration options

### Filter node

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expression` | (required) | Boolean expression to evaluate |
| `template` | None | LLM template for field extraction |
| `omitted_text` | `" ... "` | Replacement text for filtered items |
| `mode` | auto | `"llm"` or `"simple"` (auto-detected from template) |

### Reduce node

| Parameter | Default | Description |
|-----------|---------|-------------|
| `template` | `"{{input}} "` | Template for rendering each item |
| `exclude_overlap` | `True` | Use core content only (no overlap) |

## Next steps

- [Node Types](node-types.md) -- Overview of all node types
- [Pre-extract Workflow](../how-to/pre-extract-workflow.md) -- Using filtering in practice
- [Thematic Analysis](../how-to/thematic-analysis.md) -- Complete analysis workflows
