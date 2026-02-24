---
layout: default
title: Similarity Metrics
parent: Explanation
nav_order: 7
---

# Theme Similarity Metrics

## Introduction

When comparing two sets of qualitative themes (e.g., human-generated vs LLM-generated, or multiple LLM runs), we need metrics that capture how well the theme sets align. This document explores three approaches to measuring theme similarity:

1. **Paper's approach** (Raza et al. 2025): Jaccard Similarity with n×m denominator
2. **Current implementation**: Precision/Recall/F1 with bidirectional max-similarity
3. **Hungarian matching**: Optimal one-to-one assignment

Each approach makes different assumptions about what constitutes "good alignment" and produces different scores for the same data.

## The Paper's Approach (Raza et al. 2025)

Reference: [Raza et al. 2025 - LLM-TA Pipeline](https://arxiv.org/abs/2502.01620)

### Mathematical Definitions

Let H = {h₁, h₂, ..., hₙ} represent human-generated themes, and L = {l₁, l₂, ..., lₘ} represent LLM-generated themes.

For each pair (hᵢ, lⱼ) in H × L, compute a similarity score s(hᵢ, lⱼ) (typically cosine similarity of embeddings).

Define Sθ = {(hᵢ, lⱼ) ∈ H × L | s(hᵢ, lⱼ) ≥ θ}, where θ is the similarity threshold.

**Jaccard Similarity:**
```
Jaccard = |Sθ| / (n × m)
```

Where:
- |Sθ| = number of theme pairs with similarity ≥ threshold
- n × m = total possible pairs

**Hit Rate:**
```
Hit Rate = |Hₛ| / n
```

Where Hₛ = {h ∈ H | ∃ l ∈ L, s(h, l) ≥ θ} (themes in H that have at least one match in L)

### What This Measures

**Jaccard Similarity** in this formulation measures the **density of semantic overlap** across the entire theme space. It answers: "What fraction of all possible thematic relationships are similar?"

**Hit Rate** measures **coverage**: what proportion of one set finds representation in the other.

### Key Characteristics

- **Many-to-many matching**: One theme can match multiple themes in the other set
- **Density-based**: Numerator counts all pairs above threshold, not unique themes
- **Size-dependent**: Larger theme sets produce lower Jaccard scores even with perfect one-to-one alignment

### Evaluation Methods

Raza et al. use four parallel evaluation methods:

1. **all-MiniLM-L6-v2** (embedding) → cosine similarity
2. **all-mpnet-base-v2** (embedding) → cosine similarity
3. **sentence-T5-xxl** (embedding) → cosine similarity
4. **LLM judge (GPT-based)** → similarity score 0-1

For each method, they:
- Generate pairwise similarity scores for all H × L theme pairs
- Convert to binary using threshold θ
- Calculate Jaccard and Hit Rate separately per method
- **Never aggregate or average across methods**

The LLM judge assigns similarity scores based on conceptual overlap, with penalties for specificity mismatches (one theme very specific, other very general).

## Critical Note: Mathematical Properties of Cosine Similarity

Before discussing implementation choices, it's essential to understand fundamental limitations of cosine similarity as a metric.

### Cosine Similarity is Not a True Metric

**Violates triangle inequality** (Schubert 2021):
Cosine distance (1 − cosine similarity) does not satisfy the triangle inequality property fundamental to distance metrics. Concrete counterexample: d₃ = 0.3562 > d₁ + d₂ = 0.2286, violating d(x,z) ≤ d(x,y) + d(y,z).

**Angular distance is the proper metric**:
The angle θ = arccos(cosine) forms a true metric on the unit sphere, but cosine values themselves do not.

### Can Yield Arbitrary Results

**Embedding instability** (Rendle et al. 2024):
Research shows "cosine similarity of learned embeddings can in fact yield arbitrary results." Different L2 regularizations produce identical prediction models but vastly different cosine similarities -- rendering comparisons potentially meaningless.

**Loss of semantic information** (arXiv:2509.19323):
Cosine similarity discards magnitude information entirely, yet vector magnitude can encode meaningful semantic properties (specificity, importance). Additionally, representation collapse in modern embeddings means "even semantically disparate sentences can exhibit high cosine similarity."

### Nonlinear Relationship to Semantic Distance

Cosine values have a **nonlinear** relationship to actual semantic distance:
- Small changes near 1.0 (cos 0° → cos 10°: 1.0 → 0.985) represent tiny semantic shifts
- Same magnitude changes near 0.5 (cos 60° → cos 70°: 0.5 → 0.342) represent much larger shifts
- The function θ = arccos(cos θ) reveals the nonlinearity

This means: 0.9 → 0.95 is semantically very different from 0.3 → 0.35, despite both being 0.05 increases.

### Why Averaging is Problematic

**Not a proper measurement scale**:

Cosine similarity is a **bounded similarity coefficient**, not a ratio or interval scale:
- **Not ratio scale**: No true zero (0 = orthogonal, not "no similarity"); 0.8 ≠ "twice as similar" as 0.4
- **Not interval scale**: Differences are **nonlinear** and not homogeneous across the scale
  - 0.9→0.8 = 11.1° angular change
  - 0.3→0.2 = 7.2° angular change
  - Same numerical difference (0.1) ≠ same semantic/geometric distance
- **Bounded affine-ish**: Monotonic with angular distance but no equal-interval guarantee

**Implication**: Averaging cosine similarities lacks mathematical foundation. You're averaging nonlinear transformations of angles.

**Practical consequences**:
```python
mean([0.95, 0.95, 0.30]) = 0.73  # Suggests "moderate similarity"
# But reality: 2/3 themes have excellent matches, 1/3 has poor match
# The mean masks the bimodal distribution AND loses geometric meaning
```

**Rigorous alternative**: Average **angular distances** θ = arccos(cosine), which IS a proper metric.

**What Raza et al. do instead**:
1. Threshold to binary (≥ θ → 1, else 0)
2. Count proportions (which ARE ratio scale)
3. Report methods separately (never average across embedding models or judge scores)

This approach converts similarity to a proper ratio scale (proportions/percentages) before aggregation.

## Current Implementation

Our implementation in `soak/comparators/similarity_comparator.py` calculates:

### Metrics Computed

1. **Match Matrix**: Binary matrix where `match_matrix[i,j] = 1` if `similarity[i,j] ≥ threshold`

2. **Recall (equivalent to paper's Hit Rate)**:
   ```python
   recall = (number of A themes with ≥1 match in B) / |A|
   ```

3. **Precision (symmetric Hit Rate)**:
   ```python
   precision = (number of B themes with ≥1 match in A) / |B|
   ```

4. **F1 Score**:
   ```python
   f1 = 2 * (precision * recall) / (precision + recall)
   ```

5. **Jaccard (paper's formulation)**:
   ```python
   jaccard = match_matrix.sum() / (|A| × |B|)
   ```

6. **Bidirectional Max-Similarity**:
   ```python
   a_b_most_similar = mean(max similarity for each A theme across all B themes)
   b_a_most_similar = mean(max similarity for each B theme across all A themes)
   similarity_f1 = harmonic mean of above two
   ```

   ⚠️ **Note**: This metric averages raw cosine similarity scores, which is mathematically questionable (see "Why Averaging is Problematic" above). While pragmatically common in practice, this violates measurement theory since cosine similarity is not a ratio scale. Consider using threshold-based metrics (precision/recall/F1) or reporting full distributions (min, Q1, median, Q3, max) instead.

### Implementation Details

```python
# From compare_result_similarity() function
sim_matrix = cosine_similarity(emb_A, emb_B)  # |A| × |B| matrix
match_matrix = sim_matrix >= threshold         # Binary matrix

# Recall: % of A themes with any match
recall = match_matrix.any(axis=1).sum() / len(emb_A)

# Precision: % of B themes with any match
precision = match_matrix.any(axis=0).sum() / len(emb_B)

# Jaccard: matches / all possible pairs
jaccard = match_matrix.sum() / match_matrix.size
```

## Hungarian Matching Alternative

### Concept

The Hungarian algorithm (also called Kuhn-Munkres) finds the **optimal one-to-one assignment** between two sets that maximizes total similarity.

Unlike the many-to-many approaches above, this enforces that each theme maps to at most one theme in the other set.

### Algorithm

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

# Convert similarity to cost (algorithm minimizes)
cost_matrix = 1 - similarity_matrix

# Find optimal assignment
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# Extract matched pairs above threshold
matched_pairs = [
    (i, j, similarity_matrix[i, j])
    for i, j in zip(row_indices, col_indices)
    if similarity_matrix[i, j] >= threshold
]
```

### Metrics from Hungarian Matching

1. **Mean Matched Similarity**: Average similarity of optimal pairs
2. **Coverage A**: Proportion of A themes in matched pairs above threshold
3. **Coverage B**: Proportion of B themes in matched pairs above threshold
4. **Hungarian F1**: Harmonic mean of Coverage A and Coverage B
5. **True Jaccard** (optional): Using matched themes:
   ```
   intersection = number of matched pairs above threshold
   union = |A| + |B| - intersection
   jaccard = intersection / union
   ```

### What This Measures

Hungarian matching measures **optimal alignment quality** under one-to-one constraint. It answers: "If each theme can match only one counterpart, what's the best possible alignment?"

## Concrete Examples

### Example 1: Perfect 5×5 Alignment

**Setup:**
- Set A: 5 themes [A1, A2, A3, A4, A5]
- Set B: 5 themes [B1, B2, B3, B4, B5]
- Similarity matrix: Perfect diagonal (each Aᵢ matches only Bᵢ with similarity 0.85)
- Threshold: 0.7

**Similarity Matrix:**
```
      B1   B2   B3   B4   B5
A1  0.85 0.30 0.25 0.20 0.15
A2  0.30 0.85 0.30 0.25 0.20
A3  0.25 0.30 0.85 0.30 0.25
A4  0.20 0.25 0.30 0.85 0.30
A5  0.15 0.20 0.25 0.30 0.85
```

**Results:**

| Metric | Paper's Jaccard | Current Recall/Precision | Hungarian |
|--------|----------------|-------------------------|-----------|
| **Primary score** | 5/25 = **0.20** | F1 = **1.0** | Coverage F1 = **1.0** |
| **Interpretation** | Only 20% of pairs similar | Perfect coverage both ways | Perfect optimal alignment |
| **Coverage A** | N/A | 5/5 = 1.0 | 5/5 = 1.0 |
| **Coverage B** | N/A | 5/5 = 1.0 | 5/5 = 1.0 |
| **Mean similarity** | N/A | 0.85 (bidirectional) | 0.85 (optimal pairs) |
| **"True" Jaccard** | 0.20 | N/A | 5/(5+5-5) = **1.0** |

**Analysis:**
- Paper's Jaccard gives 0.20 for perfect alignment because only 5 out of 25 possible pairs are similar
- Current metrics (F1 = 1.0) correctly identify perfect coverage
- Hungarian gives perfect scores and can compute proper Jaccard = 1.0

---

### Example 2: Perfect 10×10 Alignment

**Setup:**
- Same as Example 1 but with 10 themes in each set
- Perfect diagonal alignment

**Results:**

| Metric | Paper's Jaccard | Current F1 | Hungarian |
|--------|----------------|-----------|-----------|
| **Primary score** | 10/100 = **0.10** | **1.0** | Coverage F1 = **1.0** |
| **"True" Jaccard** | 0.10 | N/A | 10/(10+10-10) = **1.0** |

**Analysis:**
- Paper's Jaccard **gets worse** (0.10 vs 0.20) despite identical quality
- Shows size-dependency problem: larger sets → lower scores even with perfect alignment
- Current F1 and Hungarian both correctly give 1.0

**Key Insight:** Paper's Jaccard penalizes having more themes, which is problematic if theme count varies.

---

### Example 3: Asymmetric Sets (5A × 10B)

**Setup:**
- Set A: 5 themes
- Set B: 10 themes
- All 5 A themes match something in B
- Only 5 of 10 B themes got matched
- Threshold: 0.7

**Results:**

| Metric | Paper's Jaccard | Current Metrics | Hungarian |
|--------|----------------|----------------|-----------|
| **Primary score** | 5/50 = **0.10** | Recall=1.0, Precision=0.5, F1=**0.67** | Coverage F1=**0.67** |
| **Coverage A** | N/A | 5/5 = 1.0 | 5/5 = 1.0 |
| **Coverage B** | N/A | 5/10 = 0.5 | 5/10 = 0.5 |
| **"True" Jaccard** | 0.10 | N/A | 5/(5+10-5) = **0.50** |

**Analysis:**
- Paper's Jaccard (0.10) severely underestimates quality
- Current F1 (0.67) balances full A coverage with partial B coverage
- Hungarian F1 (0.67) gives same result with one-to-one constraint
- "True" Jaccard (0.50) reflects that half the themes in the union were matched

**Interpretation:** Set A is fully represented in B, but B has redundant/extra themes. This is captured better by F1 than paper's Jaccard.

---

### Example 4: Many-to-Many Overlapping Themes

**Setup:**
- Set A: 3 themes [A1, A2, A3]
- Set B: 3 themes [B1, B2, B3]
- A1 and A2 both match B1 and B2 (overlapping concepts)
- A3 matches only B3
- Threshold: 0.7

**Similarity Matrix:**
```
      B1   B2   B3
A1  0.80 0.75 0.40
A2  0.75 0.80 0.30
A3  0.20 0.30 0.85
```

**Match Matrix (≥0.7):**
```
      B1   B2   B3
A1    1    1    0
A2    1    1    0
A3    0    0    1
```

**Results:**

| Metric | Paper's Jaccard | Current Metrics | Hungarian |
|--------|----------------|----------------|-----------|
| **Primary score** | 5/9 = **0.56** | F1 = **1.0** | Coverage F1 = **1.0** |
| **Matched pairs** | 5 pairs | All themes covered | 3 optimal pairs |
| **Mean similarity** | N/A | 0.775 (bidirectional) | 0.817 (optimal) |
| **Optimal assignment** | N/A | N/A | A1→B1 (0.80), A2→B2 (0.80), A3→B3 (0.85) |

**Analysis:**
- Paper's Jaccard (0.56) reflects many-to-many matching (5 pairs out of 9 possible)
- Current F1 (1.0) shows perfect coverage -- every theme finds a match
- Hungarian finds globally optimal one-to-one alignment with higher mean similarity (0.817)

**Key Insight:** When themes overlap conceptually (A1 and A2 both match B1, B2), different approaches give different answers:
- Paper's Jaccard: counts all 5 similar pairs
- Current F1: just checks coverage (all themes covered = 1.0)
- Hungarian: picks best one-to-one mapping (A1→B1, A2→B2 rather than A1→B2, A2→B1)

---

### Example 5: Partial Mismatch

**Setup:**
- Set A: 5 themes
- Set B: 5 themes
- Only 3 themes align (diagonal positions 1,2,3)
- Themes 4 and 5 don't match anything
- Threshold: 0.7

**Match Matrix:**
```
      B1   B2   B3   B4   B5
A1    1    0    0    0    0
A2    0    1    0    0    0
A3    0    0    1    0    0
A4    0    0    0    0    0
A5    0    0    0    0    0
```

**Results:**

| Metric | Paper's Jaccard | Current Metrics | Hungarian |
|--------|----------------|----------------|-----------|
| **Primary score** | 3/25 = **0.12** | Recall=0.6, Precision=0.6, F1=**0.60** | Coverage F1=**0.60** |
| **Coverage A** | N/A | 3/5 = 0.6 | 3/5 = 0.6 |
| **Coverage B** | N/A | 3/5 = 0.6 | 3/5 = 0.6 |
| **"True" Jaccard** | 0.12 | N/A | 3/(5+5-3) = **0.43** |

**Analysis:**
- Paper's Jaccard (0.12) very low despite 60% coverage
- Current F1 (0.60) accurately reflects 60% coverage on both sides
- Hungarian F1 (0.60) gives same result
- "True" Jaccard (0.43) is higher than paper's because denominator is 7 not 25

**Key Insight:** All approaches agree on relative quality (partial match), but paper's Jaccard produces much lower absolute scores.

---

## Summary Comparison Table

| Scenario | Paper Jaccard | Current F1 | Hungarian F1 | "True" Jaccard |
|----------|--------------|-----------|--------------|---------------|
| Perfect 5×5 | 0.20 | 1.0 | 1.0 | 1.0 |
| Perfect 10×10 | 0.10 | 1.0 | 1.0 | 1.0 |
| Asymmetric 5×10 | 0.10 | 0.67 | 0.67 | 0.50 |
| Many-to-many 3×3 | 0.56 | 1.0 | 1.0 | 1.0 |
| Partial 5×5 | 0.12 | 0.60 | 0.60 | 0.43 |

**Key Observations:**

1. **Paper's Jaccard is systematically low** and cannot reach 1.0 for one-to-one alignment
2. **Size dependency**: Perfect 10×10 scores lower than perfect 5×5
3. **Current F1 and Hungarian F1 are similar** but Hungarian uses optimal one-to-one matching
4. **"True" Jaccard** (with union = |A|+|B|-intersection) gives interpretable results

---

## When to Use Each Approach

### Use Paper's Jaccard When:

- Replicating published methodology for comparability
- Expecting dense many-to-many theme relationships
- Want to measure "semantic overlap density" across theme space
- Theme redundancy/overlap is important to capture

**Limitations:**
- Cannot reach 1.0 for perfect one-to-one alignment
- Penalizes larger theme sets
- Non-intuitive scale (hard to interpret absolute values)

### Use Current Precision/Recall/F1 When:

- Want to measure **coverage**: what proportion of themes find matches
- Don't care about one-to-one constraint (themes can match multiple counterparts)
- Need interpretable metrics (1.0 = perfect coverage)
- Want bidirectional coverage assessment

**Strengths:**
- Intuitive interpretation
- Handles asymmetric set sizes well
- F1 = 1.0 means all themes (both sets) found matches

**Limitations:**
- Doesn't distinguish between one-to-one and many-to-many matching
- Doesn't consider match quality beyond threshold
- Can give 1.0 even with weak matches (just above threshold)

### Use Hungarian Matching When:

- Conceptual model is "each theme should match one counterpart"
- Want globally optimal one-to-one alignment
- Need match quality metrics (mean similarity of best pairs)
- Comparing analyses where you expect similar theme counts
- Want both coverage and quality assessment

**Strengths:**
- Enforces one-to-one constraint (no double-counting)
- Finds globally optimal assignment
- Provides interpretable quality metrics
- Can compute "true" Jaccard with proper denominator

**Limitations:**
- Forces one-to-one even when many-to-many might be appropriate
- More complex to implement and explain
- Assumes sets should be roughly same size

---

## Recommendations

### For Your Use Case

Given your assumption that "perfect alignment = each theme in A has exactly one match in B":

**Primary recommendation: Add Hungarian matching alongside current metrics**

Implement Hungarian algorithm to provide:

1. **Coverage metrics**: Proportion of themes matched above threshold (primary metric)
2. **Hungarian F1**: Harmonic mean of Coverage A and Coverage B
3. **"True" Jaccard**: `matched / (|A| + |B| - matched)` for proper Jaccard interpretation
4. **Distribution of match quality**: Report min, Q1, median, Q3, max of matched pair similarities (instead of mean)

⚠️ **Avoid averaging raw cosine similarities** due to non-ratio-scale properties. If continuous scores are needed, report full distribution or use percentile-based metrics (e.g., "% of themes with similarity > 0.8").

**Keep existing metrics:**

- **Current F1**: Still useful for quick coverage assessment (threshold-based, no averaging)
- **Bidirectional max-similarity**: Provides quality estimate but note averaging limitations
- **Paper's Jaccard**: Keep for methodological comparison/replication

**Consider adding:**

- **Per-method reporting**: Like Raza et al., report results separately for each embedding model
- **Percentile metrics**: "% themes above 0.7", "% themes above 0.8", etc.
- **Angular distance**: Use arccos(cosine) for a proper metric (if mathematically needed)

### Implementation Strategy

```python
def hungarian_matching(
    similarity_matrix: np.ndarray,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """Compute optimal one-to-one theme matching."""
    from scipy.optimize import linear_sum_assignment

    n_A, n_B = similarity_matrix.shape

    # Pad to square matrix if needed
    size = max(n_A, n_B)
    sim_padded = np.zeros((size, size))
    sim_padded[:n_A, :n_B] = similarity_matrix

    # Find optimal assignment (minimize cost = maximize similarity)
    cost = 1 - sim_padded
    row_ind, col_ind = linear_sum_assignment(cost)

    # Extract real pairs (not padding)
    pairs = [
        (i, j, similarity_matrix[i, j])
        for i, j in zip(row_ind, col_ind)
        if i < n_A and j < n_B
    ]

    # Filter by threshold
    matched_pairs = [p for p in pairs if p[2] >= threshold]

    # Compute metrics
    coverage_A = len(matched_pairs) / n_A if n_A > 0 else 0.0
    coverage_B = len(matched_pairs) / n_B if n_B > 0 else 0.0

    hungarian_f1 = (
        2 * coverage_A * coverage_B / (coverage_A + coverage_B)
        if (coverage_A + coverage_B) > 0 else 0.0
    )

    # "True" Jaccard
    intersection = len(matched_pairs)
    union = n_A + n_B - intersection
    true_jaccard = intersection / union if union > 0 else 0.0

    # Distribution of match quality (instead of mean)
    similarities = [p[2] for p in matched_pairs]
    if similarities:
        similarity_dist = {
            "min": float(np.min(similarities)),
            "q1": float(np.percentile(similarities, 25)),
            "median": float(np.median(similarities)),
            "q3": float(np.percentile(similarities, 75)),
            "max": float(np.max(similarities)),
        }
    else:
        similarity_dist = {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0}

    return {
        "matched_pairs": matched_pairs,
        "similarity_distribution": similarity_dist,
        "coverage_A": coverage_A,
        "coverage_B": coverage_B,
        "hungarian_f1": hungarian_f1,
        "true_jaccard": true_jaccard,
        "n_matched": len(matched_pairs),
        "n_total_pairs": len(pairs),
    }
```

### Reporting Strategy

When comparing theme sets, report:

1. **Hungarian F1** as primary metric (one-to-one alignment quality)
2. **Current F1** for coverage (allows many-to-many)
3. **Similarity distribution** from Hungarian (median and quartiles, not mean)
4. **Coverage metrics** (A and B separately) to detect asymmetry
5. **Paper's Jaccard** in parentheses for methodological comparison

**Example output:**
```
Theme Alignment: Hungarian F1 = 0.85, Coverage F1 = 0.92
  - Optimal one-to-one: 17/20 themes matched
  - Match quality: median=0.78 (Q1=0.71, Q3=0.84, range: 0.65-0.91)
  - Coverage: 17/20 A themes (85%), 17/19 B themes (89%)
  - Paper's Jaccard: 0.23 (for comparison)
```

This gives users:
- **One-to-one quality** (Hungarian) matching your conceptual model
- **Many-to-many coverage** (current F1) for flexibility
- **Distribution not mean** (avoids ratio-scale violation)
- **Methodological comparability** (paper's Jaccard)

### Handling the Current "Jaccard" Metric

Options (in order of recommendation):

1. **Rename to `match_density`**: Keep calculation, change name to reflect what it actually measures
2. **Add `hungarian_jaccard` or `true_jaccard`**: Compute proper Jaccard from Hungarian matching
3. **Keep both**: Report paper's formula as `jaccard_raza2025` and new one as `jaccard`

This maintains backward compatibility while adding clarity.

---

## Ideas for Future Exploration

This section presents conceptual approaches not yet implemented, intended to facilitate expert discussion about alternative metrics for theme similarity.

### The Gap-Based Distinctiveness Problem

**Core question**: Does a theme have a clear unique match, or is it ambiguous between multiple candidates?

**Current approach**: Use absolute threshold (e.g., similarity ≥ 0.7) to define matches. But this doesn't distinguish between:
- **Case A**: Best match = 0.85, second-best = 0.45 (clear winner)
- **Case B**: Best match = 0.85, second-best = 0.82 (ambiguous)

Both pass the threshold, but Case A suggests a unique match while Case B suggests multiple plausible candidates.

**Proposed solution**: Measure the **gap** between best and second-best match.

#### Pragmatic Justification

**Why this approach is useful** (despite limited mathematical rigor):

**Empirical track record**:
- Gap-based methods widely successful in ML and computer vision
- Lowe's ratio test, silhouette coefficient, confidence margins all rely on similar logic
- Empirically correlates with match robustness

**Practical advantages over averaging**:
- Preserves information about distinctiveness (unique vs ambiguous)
- More interpretable: "20% better than runner-up" vs "average similarity 0.73"
- Common engineering heuristic with proven utility

**Mathematical limitations**:
- Cosine similarity is **not an interval scale** -- differences are nonlinear (as shown earlier)
- `sim_best - sim_second` is a **heuristic**, not a mathematically grounded metric difference
- For rigorous treatment: use **angular distance gaps** `θ_best - θ_second` (see below)

**Trade-off**: Raw cosine gaps are simpler and familiar; angular gaps are more principled. For most applications, cosine gaps adequate.

#### Parallels in ML and Computer Vision

**Lowe's Ratio Test** (SIFT feature matching, 2004):
```
ratio = distance_best / distance_second
if ratio < 0.8:  # Best is >25% better
    accept_match()
```

Used extensively in computer vision to reject ambiguous correspondences. A distinctive feature match should have a large gap to the second-best candidate.

**Silhouette Coefficient** (clustering quality):
```
a = cohesion (within-cluster distance)
b = separation (to nearest different cluster)
silhouette = (b - a) / max(a, b)
```

Measures how well-separated clusters are. Analogously, for themes:
```
distinctiveness = (sim_best - sim_second) / max(sim_best, sim_second)
```

**Confidence Margin** (classification, active learning):
```
margin = prob_top_class - prob_second_class
```

Small margin indicates uncertain prediction. In active learning, these are prioritized for labeling. For theme matching, small margins suggest ambiguous matches requiring expert review.

#### Proposed Gap-Based Metrics

1. **Absolute gap**: `gap = sim_best - sim_second`
   - Simple, intuitive
   - Issue: Same gap means different things at different similarity levels

2. **Relative gap**: `gap_rel = (sim_best - sim_second) / sim_best`
   - Percentage improvement over second-best
   - "Best match is 20% better than runner-up"

3. **Normalized gap**: `gap_norm = (sim_best - sim_second) / max(sim_best, sim_second)`
   - Silhouette-style, range [0,1]
   - 1.0 = maximally distinctive

#### Rigorous Alternative: Angular Distance

For mathematical rigor, use **angular distance** instead of raw cosine similarity:

```python
θ = arccos(cosine_similarity)  # Convert to angle in radians or degrees
```

**Properties**:
- **True metric**: Satisfies triangle inequality
- **Proper interval scale**: Equal differences = equal geometric distances on unit sphere
- **Range**: [0°, 180°] for general vectors; [0°, 90°] for non-negative embeddings

**Example** (demonstrating nonlinearity of cosine):
```
sim = 0.9 → θ = 25.8°
sim = 0.8 → θ = 36.9°
gap = 11.1° (geometrically meaningful)

sim = 0.3 → θ = 72.5°
sim = 0.2 → θ = 78.5°
gap = 6.0° (smaller angular change despite same numerical difference!)
```

**Gap-based metrics using angular distance**:
- Absolute gap: `θ_best - θ_second` (true metric difference)
- Relative gap: `(θ_best - θ_second) / θ_best`
- All approaches below can use θ instead of raw cosine

**Trade-offs**:
- **Pros**: Mathematically principled, proper metric, interpretable as geometry
- **Cons**: Slightly more computation (arccos), less familiar to practitioners, need to convert back for display

**Recommendation**: Use angular distance if mathematical rigor matters (academic work, methodological development). Use raw cosine for practical applications where engineering heuristics suffice.

#### Two-Condition Matching

Combine **quality** (absolute similarity) with **distinctiveness** (gap):

```
Quality:         sim_best ≥ 0.7
Distinctiveness: gap ≥ 0.15 (15% improvement)

Results in 4 categories:
- High quality + high gap  → "confident unique match"
- High quality + low gap   → "ambiguous (multiple candidates)"
- Low quality + high gap   → "clear but poor match"
- Low quality + low gap    → "no good match"
```

This provides richer information than binary accept/reject.

#### Aggregate Metrics

- **Unique match rate**: Proportion of themes with distinctive best match
- **Gap distribution**: Median, quartiles of gap values (not mean, per earlier discussion)
- **Match confidence profile**: Distribution across 4 categories above

### The Within-Set Baseline Problem

**Deeper question**: Should match thresholds be **adaptive** based on intrinsic similarity structure?

**Observation**: Themes within set A may have natural conceptual overlap. Example:
```
A1: "Rural healthcare barriers"
A2: "Transportation to medical facilities"
A3: "Telemedicine adoption"

All address healthcare access → naturally ~0.65 similar to each other
```

**Issue**: If we find B1 with similarity 0.70 to A1:
- **Absolute view**: 0.70 passes threshold → match
- **Relative view**: 0.70 vs baseline 0.65 → only 0.05 improvement → weak signal

**Key insight**: The **within-set similarity distribution** provides a **baseline** or **null model** against which to judge cross-set matches.

#### Statistical Framing

This is analogous to:
- **Permutation testing**: Is observed similarity better than expected by chance?
- **Signal-to-noise ratio**: Signal (best match) vs noise (typical within-set similarity)
- **Effect size** (Cohen's d): Mean difference relative to within-group variance
- **Isotropy correction** in embeddings: Adjusting for the "narrow cone" problem

#### Four Approaches to Baseline Calibration

**Approach A: Baseline-Adjusted Similarity**

*Concept*: Subtract theme's typical within-set similarity from cross-set scores

```
adjusted_sim = sim(A_i, B_k) - mean(sim(A_i, A_j) for j≠i)
```

*Interpretation*:
- `adjusted > 0`: B match better than typical A similarity
- `adjusted > 0.2`: Substantially better than baseline

*Parallel*: Background subtraction in signal processing, image analysis

*Pros*: Simple, intuitive
*Cons*: Doesn't account for variance in within-set similarities

*Rigor note*: Can apply to raw cosine (pragmatic) or angular distance (mathematically principled)

---

**Approach B: Z-Score Normalization** ⭐

*Concept*: Normalize by both mean AND standard deviation of within-set similarity

```
z = (sim_cross - μ_within) / σ_within
```

*Interpretation*:
- `z > 2`: Match is >2 standard deviations above baseline (statistically significant)
- `z > 3`: Highly significant match
- `z < 0`: Match is worse than typical within-set similarity

*Parallels*:
- **Cohen's d** (effect size in statistics)
- **Isotropy correction** (Mu & Viswanath 2018; Ethayarajh 2019) for embedding anisotropy
- Standard practice in hypothesis testing

*Pros*:
- Standard statistical interpretation
- Accounts for both location and spread
- Handles heterogeneous within-set structure
- Puts all themes on comparable scale

*Cons*:
- Requires sufficient within-set samples to estimate σ
- Assumes approximately normal within-set distribution

*Why recommended*: Most statistically principled; widely understood by ML practitioners

*Rigor note*: Z-scores of raw cosine are "heuristic normalization" (nonlinear scale). For full rigor, compute z-scores on angular distances θ. Trade-off: cosine z-scores simpler and often adequate.

---

**Approach C: Relative Gap Distinctiveness**

*Concept*: Compare cross-set gap to typical within-set gaps

```
within_gap = max(sim(A_i, A)) - second_max(sim(A_i, A))
cross_gap = max(sim(A_i, B)) - second_max(sim(A_i, B))
relative_gap = cross_gap / within_gap
```

*Interpretation*:
- `relative_gap > 1`: B match is MORE distinctive than A's structure suggests
- `relative_gap < 1`: B match is LESS distinctive than typical within-A patterns

*Parallel*: Gap statistics in clustering (Tibshirani et al. 2001)

*Pros*:
- Captures distinctiveness structure, not just central tendency
- Adaptive to each theme's specific pattern

*Cons*:
- More complex to interpret
- Undefined if within_gap = 0

*Rigor note*: Can apply to raw cosine or angular distance gaps

---

**Approach D: Percentile Ranking**

*Concept*: Rank best B match against distribution of within-A similarities

```
within_sims = [sim(A_i, A_j) for j≠i]
best_B = max(sim(A_i, B_k) for k)
percentile = percentile_rank(within_sims, best_B)
```

*Interpretation*:
- 95th percentile: Better than 95% of within-A similarities
- 50th percentile: Typical A-level similarity (not distinctive)

*Parallel*: Non-parametric hypothesis testing

*Pros*:
- No distributional assumptions
- Intuitive interpretation
- Robust to outliers

*Cons*:
- Requires sufficient within-set samples
- Less sensitive to tail behavior than z-scores

*Rigor note*: Works with raw cosine or angular distance; ranking preserves order either way

#### Comparison of Approaches

| Method | Accounts for Mean | Accounts for Variance | Statistical Interpretation | Computational Cost | Robustness |
|--------|-------------------|----------------------|---------------------------|-------------------|------------|
| Baseline-adjusted | ✓ | ✗ | Simple improvement | Low | High |
| Z-score | ✓ | ✓ | Significance testing | Low | Medium |
| Relative gap | ✓ | Partial | Comparative structure | Medium | Medium |
| Percentile | ✓ | ✓ (implicit) | Non-parametric rank | Medium | High |

#### Broader Statistical Precedents

**Isotropy Correction in Embeddings**:

Modern language models produce anisotropic embeddings -- vectors occupy a narrow cone in embedding space, leading to artificially high cosine similarities even for unrelated items.

Standard correction (Mu & Viswanath 2018):
```
corrected_sim(x, y) = raw_sim(x, y) - mean(raw_sim(x, all_items))
```

The within-set baseline approach is a more sophisticated variant: rather than using global mean, use **context-specific baseline** (within-A distribution for each theme).

**Permutation Testing / Null Models**:

Common in network analysis and link prediction:
- Generate random baseline (permute edges, reshuffle labels)
- Compare observed statistic to null distribution
- Accept if significantly better than chance

Applied here: Within-A similarity acts as "null model" -- what we'd expect for themes in the same semantic space. Cross-set matches should exceed this baseline.

**Signal-to-Noise Ratio**:

Used in signal processing, biosignals (EEG, fMRI), image analysis:
```
SNR = signal_power / noise_power
```

Analogously:
```
SNR_theme = best_match_similarity / typical_within_similarity
```

High SNR indicates match stands out clearly from background.

### Open Questions for Expert Discussion

1. **Symmetry**: Should we also compute within-B baselines and require matches exceed BOTH A and B baselines?

2. **Small set sizes**: How to handle when |A| < 5? Insufficient samples for meaningful σ or percentile estimates.

3. **Integration with Hungarian matching**: Could baseline-adjusted scores replace raw similarities in Hungarian algorithm?

4. **Threshold selection**: Can we replace fixed thresholds (0.7) with adaptive thresholds based on within-set statistics?

5. **Multiple testing correction**: If testing many themes, should we apply Bonferroni or FDR correction to z-scores?

6. **Anisotropy diagnosis**: Can we quantify whether within-set high similarity is due to:
   - Genuine semantic overlap (expected)
   - Embedding space anisotropy (technical artifact)

7. **Stability**: How sensitive are these methods to outliers in within-set distribution?

8. **Interpretability trade-offs**: Z-scores have standard interpretation but may be less intuitive to non-statisticians than simple gaps.

9. **Rigor vs practicality**: Should we use angular distance (mathematically rigorous) or raw cosine (simpler, more familiar)? Does the added mathematical rigor matter for qualitative research applications, or are engineering heuristics adequate?

### Potential Applications

**Adaptive Quality Control**:
- Flag themes that don't rise above their within-set baseline
- "A1 matched B3, but B3 is only 0.05 better than typical A similarities → review"

**Domain-Aware Thresholding**:
- High within-set similarity (specialized domain) → require larger gaps
- Low within-set similarity (diverse themes) → accept smaller gaps
- Automatically adapts to semantic structure

**Confidence-Weighted Metrics**:
- Weight matches by z-score or gap when computing aggregate statistics
- High-confidence matches count more than ambiguous matches

**Handling Embedding Isotropy**:
- Directly addresses the "narrow cone" problem in modern embeddings
- Makes comparisons more robust to technical artifacts

**Methodological Comparison**:
- Compare raw vs baseline-adjusted approaches on same data
- Quantify how much baseline correction changes match decisions

### Relationship to Existing Metrics

**Precision/Recall/F1**: Binary (matched or not)
- Baseline correction could inform **confidence weights**
- E.g., weighted F1 where high-z matches count more

**Paper's Jaccard**: Density of matches
- Could threshold using z-scores instead of raw similarities
- "Jaccard with z > 2" vs "Jaccard with raw > 0.7"

**Hungarian matching**: Optimal one-to-one assignment
- Could operate on baseline-adjusted or z-score matrix instead of raw similarities
- May find different optimal matching

**Bidirectional max-similarity**: Averages raw scores
- **Should be replaced** with gap-based or z-score metrics
- More principled measurement properties

### Summary for Discussion

Two key ideas emerge:

1. **Gap-based distinctiveness**: Measure confidence in match via gap to second-best
   - Mathematically valid (differences for interval scales)
   - Strong precedents (Lowe's test, silhouette, margins)
   - Distinguishes confident from ambiguous matches

2. **Baseline calibration**: Adjust for within-set similarity structure
   - Accounts for intrinsic semantic overlap
   - Analogous to isotropy correction, signal-to-noise, effect size
   - Z-score approach recommended for statistical properties

Both approaches address limitations of current metrics (averaging raw similarities, fixed thresholds) while connecting to established statistical and ML practices.

**Next steps**: Discuss with domain experts to determine:
- Which approach(es) best match conceptual model of theme matching
- Practical threshold values (e.g., z > 2 vs z > 1.5)
- Whether benefits justify added complexity
- Suitability for reporting to qualitative researchers

---

## References

Raza, M. Z., Xu, J., Lim, T., Boddy, L., Mery, C. M., Well, A., & Ding, Y. (2025). "LLM-TA: An LLM-Enhanced Thematic Analysis Pipeline for Transcripts from Parents of Children with Congenital Heart Disease." arXiv:2502.01620. https://arxiv.org/abs/2502.01620

Schubert, E. (2021). "A Triangle Inequality for Cosine Similarity." In Similarity Search and Applications: 14th International Conference, SISAP 2021. arXiv:2107.04071. https://arxiv.org/abs/2107.04071

Rendle, S., Krichene, W., Zhang, L., & Anderson, J. (2024). "Is Cosine-Similarity of Embeddings Really About Similarity?" arXiv:2403.05440. https://arxiv.org/abs/2403.05440

Stevens, S. S. (1946). "On the theory of scales of measurement." Science, 103(2684), 677-680.

Hungarian Algorithm: Kuhn, H. W. (1955). "The Hungarian Method for the assignment problem." Naval Research Logistics Quarterly, 2(1-2), 83-97.

Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints." International Journal of Computer Vision, 60(2), 91-110.

Mu, J., & Viswanath, B. (2018). "All-but-the-Top: Simple and Effective Postprocessing for Word Representations." In Proceedings of NAACL-HLT 2018.

Ethayarajh, K. (2019). "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings." In Proceedings of EMNLP 2019.












