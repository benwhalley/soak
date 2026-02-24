---
layout: default
title: Calibration
parent: Explanation
nav_order: 8
---

# Understanding Calibration

Calibration transforms raw embedding similarity scores into interpretable values that correspond to semantic distance categories.

## Why Calibration Is Needed

Raw embedding similarities have two problems:

1. **Model-specific ranges**: A score of 0.85 might mean "identical" for one model but "moderately similar" for another.

2. **Compressed distributions**: Modern embeddings cluster semantically related texts tightly. Even unrelated themes often have similarities above 0.6.

Calibration addresses both by mapping raw scores to a standardised scale.

## The Calibration Scale

| Calibrated Score | Meaning |
|------------------|---------|
| 0.9 | Same meaning (paraphrases) |
| 0.75 | Close meaning |
| 0.5 | Diverging (partial overlap) |
| 0.3 | Distant (weak relation) |
| 0.1 | Unrelated |

After calibration, a score of 0.75 means "close meaning" regardless of which embedding model you use.

## How It Works

### 1. Generate Training Data

We generate paraphrases of real themes at five semantic distances:

- **Same**: Identical meaning, different wording
- **Close**: Core meaning preserved, different framing
- **Diverging**: Partial overlap, different focus
- **Distant**: Same domain, different construct
- **Unrelated**: No meaningful overlap

### 2. Compute Similarities

For each original-paraphrase pair, we compute angular similarity:

```
angular_sim = 1 - arccos(cosine_sim) / π
```

Angular similarity is a proper metric (satisfies triangle inequality) with range [0, 1].

### 3. Fit Monotonic Spline

A shape-constrained additive model (SCAM) learns the mapping from raw similarity to target values. The monotonicity constraint ensures higher raw scores always map to higher calibrated scores.

### 4. Apply to New Data

The fitted model transforms similarities in `soak compare`:

```bash
uv run soak compare results.json --embedding-model local/BAAI/bge-base-en-v1.5
# Bundled calibration automatically applied
```

## Model Recommendations

| Use Case | Model | Accuracy |
|----------|-------|----------|
| Best accuracy (API) | text-embedding-3-small | 83.5% |
| Best local | BAAI/bge-base-en-v1.5 | 79.6% |
| Fastest | all-MiniLM-L6-v2 | 77.8% |

See [calibration-data/README.md](https://github.com/benwhalley/soak/tree/main/soak/calibration-data) for full comparison.

## Technical Details

### Target Value Selection

Targets (0.9, 0.75, 0.5, 0.3, 0.1) were empirically optimised to match natural embedding gaps, achieving uniform stretch ratios across the calibration curve.

### Degrees of Freedom

We use df=7 for the spline, which produces a smooth S-curve that:

- Asymptotes at the edges rather than hitting bounds abruptly
- Preserves rank ordering for scores above 0.9
- Balances fit quality (MAE) with category discrimination

### Validation

Calibrations are validated using grouped holdout (20% of papers held out), ensuring generalisation to unseen themes.

## See Also

- [CLI Reference: soak calibrate](../reference/cli.md#calibrate)
- [Similarity Metrics](similarity-metrics.md)
