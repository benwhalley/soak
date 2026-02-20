# Bundled Calibration Data

Pre-computed calibration data for embedding models, enabling meaningful
interpretation of similarity scores from `soak compare`.

## Why Calibration?

Raw embedding similarity scores are not directly interpretable -- a score of 0.85
might mean "identical" for one model but only "somewhat related" for another.
Calibration maps these raw scores to a standardised scale:

| Calibrated Score | Meaning |
|------------------|---------|
| 0.9 | Same meaning (paraphrases) |
| 0.75 | Close meaning |
| 0.5 | Diverging (partial overlap) |
| 0.3 | Distant (weak relation) |
| 0.1 | Unrelated |

### Target Value Selection

The target values (0.9, 0.75, 0.5, 0.3, 0.1) were empirically optimised based on
the natural spacing of embedding similarities across semantic categories.

**Key finding**: Target gaps should be roughly proportional to raw embedding gaps.
When targets require uniform stretch across the similarity range, the calibration
spline fits better and classification accuracy improves.

| Targets Tested | Accuracy | Notes |
|----------------|----------|-------|
| 0.9/0.8/0.6/0.4/0.1 (original) | 82.9% | Uneven stretch (2x to 6x) |
| 0.9/0.75/0.5/0.25/0.1 (symmetric) | 83.1% | Better for middle categories |
| **0.9/0.75/0.5/0.3/0.1 (data-driven)** | **83.7%** | **Best overall** |

The data-driven targets achieve more uniform stretch ratios (2.8x to 3.8x) across
all category boundaries, resulting in a smoother calibration curve and balanced
per-category accuracy (77-89% across all categories).

### How It Works

1. **Generate paraphrases**: For each source theme, an LLM generates text at five
   semantic distances: same, close, diverging, distant, unrelated.

2. **Compute similarities**: Each paraphrase pair is embedded and angular similarity
   computed.

3. **Fit calibration model**: A monotonic spline (SCAM) maps raw similarities to
   target values, learned from the known category labels.

4. **Apply to new data**: The fitted model transforms raw similarities from
   `soak compare` into interpretable calibrated scores.

## Recommended Model

**BAAI/bge-base-en-v1.5** -- best accuracy with good balance of speed and robustness.

```bash
uv run soak compare results1.json results2.json \
  --embedding-model local/BAAI/bge-base-en-v1.5
```

Calibration is auto-detected for bundled models.

## Model Comparison

Evaluated on 11,175 paraphrase pairs from 448 qualitative research themes
(generated with GPT-5.2, calibrated with df=7):

| Model | Accuracy | MAE | Notes |
|-------|----------|-----|-------|
| **text-embedding-3-small** | **83.5%** | 0.050 | **Best accuracy** (OpenAI API) |
| text-embedding-3-large | 83.4% | 0.050 | Same accuracy, higher cost |
| BAAI/bge-base-en-v1.5 | 79.6% | 0.057 | **Best local model** |
| intfloat/e5-base | 79.6% | 0.057 | Requires template |
| BAAI/bge-large-en-v1.5 | 79.3% | 0.058 | Larger, no accuracy gain |
| thenlper/gte-base | 78.2% | 0.058 | Compact model |
| all-MiniLM-L6-v2 | 77.8% | 0.062 | Fastest local |
| all-mpnet-base-v2 | 77.8% | 0.059 | Best raw separation |
| BAAI/bge-small-en-v1.5 | 77.5% | 0.062 | Fast |

**Accuracy**: % of test samples classified into correct semantic category.

### Why We Recommend bge-base

1. **Top accuracy** (79.6%, tied with e5-base)
2. **No special template required** (unlike e5-base which needs "passage: {text}")
3. **Faster than bge-large** with equal or better performance
4. **Well-supported** by the BGE team with regular updates

## Usage

### Auto-detection (recommended)

```bash
uv run soak compare results.json --embedding-model local/BAAI/bge-large-en-v1.5
# Bundled calibration automatically applied
```

### Explicit path

```bash
uv run soak compare results.json \
  --calibration soak/calibration-data/models/BAAI-bge-large-en-v1.5/
```

### Programmatic

```python
from soak.calibration import get_bundled_calibration, load_calibration, calibrate

path = get_bundled_calibration("local/BAAI/bge-large-en-v1.5")
model, metadata = load_calibration(path)
calibrated = calibrate(raw_similarities, model, method=metadata["method"])
```

## Comparing Models

```bash
uv run python scripts/compare_calibrations.py
uv run python scripts/compare_calibrations.py --plot comparison.png
```

## Regenerating Calibrations

To recalibrate with different paraphrases or add new models:

```bash
# Generate new paraphrases (uses ~$0.50 for all 448 themes with gpt-4.1-mini)
uv run soak calibrate soak/calibration-data/thematic_analysis_papers.csv \
  --prompt soak/calibration-data/paraphrases.sd \
  --column theme_description \
  --llm-model gpt-4.1-mini \
  -g doi \
  --embedding-model local/BAAI/bge-large-en-v1.5 \
  --output soak/calibration-data/new-calibration/

# Or use existing paraphrases with a new embedding model
uv run soak calibrate \
  --paraphrases soak/calibration-data/paraphrases.csv \
  --embedding-model local/new-model-name \
  -g doi \
  --output soak/calibration-data/models/new-model/
```

See `scripts/regenerate_calibrations.sh` for batch processing.

## Data Sources

- **Source themes**: 448 themes from published qualitative research papers
- **Paraphrases**: 11,200 pairs generated with GPT-5.2
- **Categories**: same, close, diverging, distant, unrelated (5 each per theme)

### Paraphrase Quality

GPT-5.2 produces higher quality paraphrases than GPT-4.1-mini, with cleaner
category boundaries that improve calibration accuracy (+7% on matched samples).
The improvement is especially pronounced for subtle distinctions (same vs close,
diverging vs distant).

---

## Decision Log

This section documents the empirical choices made during calibration development.

### Target Values (0.9, 0.75, 0.5, 0.3, 0.1)

**Problem**: Original targets (0.9, 0.8, 0.6, 0.4, 0.1) required uneven stretch
ratios across the similarity range (2x to 6x), forcing the spline to bend sharply
in some regions.

**Solution**: Data-driven targets where gaps are proportional to natural embedding
gaps. The "distant" category moved from 0.4 to 0.3 to better match the embedding
distribution.

**Result**: +0.8% accuracy improvement, more balanced per-category performance.

### Spline Degrees of Freedom (df=7)

**Problem**: Lower df values (df=5) cause the calibration curve to hit 1.0 abruptly
at high similarities, creating a "cliff" effect. Higher df values smooth the edges
but risk overfitting.

**Testing**:

| df | Accuracy | MAE | Curve Behaviour |
|----|----------|-----|-----------------|
| 5 | 80.3% | 0.060 | Hits 1.0 at ~0.92, steep edge |
| 6 | 79.0% | 0.058 | Reaches ~0.98 at 1.0 |
| 7 | 79.6% | 0.058 | Smooth S-curve, asymptotes at ~0.92 |

**Decision**: df=7 provides:

- Smooth S-curve that asymptotes rather than hitting bounds abruptly
- Better MAE than df=5 (smoother overall fit)
- Better accuracy than df=6
- Preserves rank ordering for scores above 0.9

### LLM for Paraphrase Generation (GPT-5.2)

**Comparison** (100 themes, bge-large):

| Model | Category Accuracy |
|-------|-------------------|
| GPT-5.2 | 82.9% |
| GPT-4.1-mini | 75.7% |

**Decision**: GPT-5.2 produces cleaner category boundaries, especially for subtle
distinctions (same vs close: +14% accuracy). The cost increase (~$9 for all 448
themes) is justified by the quality improvement.

### Embedding Model (BAAI/bge-base-en-v1.5)

**Finding**: bge-base ties with e5-base for top accuracy (79.6%), but:

- bge-base requires no special template
- bge-large shows no accuracy improvement over bge-base
- e5-base requires "passage: {text}" template prefix

**Decision**: Recommend bge-base as default for simplicity and performance.
