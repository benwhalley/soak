---
layout: default
title: HyDE Coverage
parent: Explanation
nav_order: 10
---

# Advanced Coverage Analysis Methods

## Overview

The coverage analysis tab provides several methods for computing semantic similarity between themes and document content. Each method answers a slightly different question about how well themes represent the corpus.

!!! note "Web UI Only"
    These advanced coverage methods are currently only available through the web interface. CLI support is planned for a future release.


## Embedding Modes

The coverage tab offers four embedding modes:

| Mode | Description | Best For |
|------|-------------|----------|
| **Theme text** | Embeds theme name + description only | Quick analysis, simple themes |
| **Theme + codes** | Embeds theme text plus all associated code descriptions | Themes with rich code structure |
| **Theme quotes** | Embeds actual quotes from theme codes | Direct validation of quote coverage |
| **HyDE quotes** | Generates synthetic interview quotes | Validating theme descriptions against data |


## Theme Quotes

### How It Works

Theme quotes mode uses the **actual quotes** attached to each theme's codes. For each theme:

1. Extract all quotes from the codes assigned to that theme
2. Embed each quote using the corpus embedding model
3. For each corpus chunk, compute similarity to all quote embeddings
4. Take the **maximum** similarity across quotes as the chunk's score

### When to Use

Use theme quotes when you want to see how well the coded quotes literally match other parts of the corpus. This answers: "Where else in the data do we see language similar to what we already coded?"

High coverage suggests good saturation -- the coded quotes represent patterns that appear throughout the data.


## HyDE (Hypothetical Document Embeddings)

### How It Works

HyDE answers a different question: "Given only this theme description, can we imagine realistic participant quotes that match what people actually said?"

With HyDE, an LLM generates synthetic quotes that a participant might say about the theme:

```
Theme: "Privacy concerns about AI"

Generated quotes:
1. "I worry about what happens to my data when I use these AI tools"
2. "It feels like they know too much about me already"
3. "I don't trust that my information stays private"
4. "Who knows where all this personal data ends up?"
5. "The whole thing makes me uncomfortable -- I never agreed to share all this"
```

Each quote is embedded separately, and for each corpus chunk, we compute the maximum similarity across all quotes.

### Why It's Useful

HyDE is useful for **validating theme descriptions**. If an LLM can generate plausible quotes from just the theme text, and those imagined quotes match your real data, it suggests the theme description captures something genuine in the corpus -- not just patterns the analyst projected onto it.

Consider these scenarios:

| HyDE Coverage | Theme Quote Coverage | Interpretation |
|---------------|---------------------|----------------|
| High | High | Theme is well-described and well-evidenced |
| High | Low | Theme description is good but under-coded |
| Low | High | Theme may need a clearer description |
| Low | Low | Theme may not be well-grounded in data |


### Technical Details

- **Model**: gpt-4.1-mini (configurable)
- **Quotes per theme**: 5
- **Prompt guidelines**: Natural interview language, 1-3 sentences each, varied perspectives


## Similarity Computation

For both Theme quotes and HyDE modes:

1. Generate/extract quotes for each theme
2. Embed each quote using the corpus embedding model
3. For each corpus chunk, compute cosine similarity to all quote embeddings
4. Take the **maximum** similarity across quotes as the chunk's score
5. Aggregate chunk scores per document (default: max)
6. Apply calibration to convert raw cosine similarity to interpretable scale


## References

The HyDE technique was introduced for information retrieval in:

> Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels. *arXiv preprint arXiv:2212.10496*.

The original paper uses HyDE for document retrieval, generating hypothetical documents that might answer a query. Our adaptation generates hypothetical quotes that might represent a theme.
