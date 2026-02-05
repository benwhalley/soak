# Django Live Comparison View Architecture

## Overview

Design a Django view that replicates `soak compare` as a live, interactive interface. Uses Celery for expensive pre-computation and HTMX for progressive loading.

## Computation Analysis

Based on exploration of `soak/comparators/similarity_comparator.py`:

### EXPENSIVE (Must Pre-compute via Celery)
| Component | Time | Trigger |
|-----------|------|---------|
| Embeddings | 1-5s per analysis | On analysis upload |
| Word-salad baseline | 5-15s | On comparison request |
| Paraphrase baseline | 30-60s | On comparison request |
| OT for all K values | 5-30s | On comparison request |

### CHEAP (On-demand in Django view)
| Component | Time | Notes |
|-----------|------|-------|
| Similarity matrices | <100ms | From cached embeddings |
| Rescaling | <10ms | Pure numpy |
| Visualizations | 1-2s | Plotly rendering |
| Template rendering | <100ms | Jinja2 |

**Key insight**: OT is pre-computed for 40+ K values, so K-slider is instant.

---

## Caching Strategy: Lazy Computation

Rather than an explicit DAG, use **content-addressed caching** where each computation's cache key is a hash of its inputs. This gives automatic invalidation:

```python
def get_ot_results(analysis_a, analysis_b, embedding_model, rescale_method, rescale_params):
    cache_key = hash(analysis_a.content_hash, analysis_b.content_hash,
                     embedding_model, rescale_method, rescale_params)

    cached = OTCache.objects.filter(key=cache_key).first()
    if cached:
        return cached.results

    # Compute (uses cached similarity matrices internally)
    matrices = get_similarity_matrices(analysis_a, analysis_b, embedding_model)
    ot_results = compute_ot(matrices, rescale_method, rescale_params)

    OTCache.objects.create(key=cache_key, results=ot_results)
    return ot_results
```

**Parameter change effects:**
| Parameter changed | Cache key changes at | Recomputes |
|-------------------|---------------------|------------|
| `embedding_model` | All levels | Everything |
| `rescale_method` | OT level only | OT + viz (reuses embeddings) |
| `threshold` | None | Nothing (filter only) |
| `k` value | None | Nothing (pre-computed for all K) |

---

## Data Models

```python
from pgvector.django import VectorField

class ThemeEmbedding(models.Model):
    """Individual theme embedding - enables pgvector HNSW indexing."""
    analysis = models.ForeignKey('runs.QualitativeResult', on_delete=models.CASCADE)
    theme_index = models.IntegerField()
    theme_text = models.TextField()
    content_hash = models.CharField(max_length=64, db_index=True)
    embedding_model = models.CharField(max_length=100)
    embedding = VectorField(dimensions=3072)  # pgvector, HNSW-indexable
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['analysis', 'theme_index', 'embedding_model']


class ComparisonResult(models.Model):
    """Cached comparison computation results."""
    comparison = models.ForeignKey('Comparison', on_delete=models.CASCADE)
    cache_key = models.CharField(max_length=64, unique=True, db_index=True)

    # Parameters (for display/debugging)
    embedding_model = models.CharField(max_length=100)
    rescale_method = models.CharField(max_length=20)
    rescale_params = models.JSONField(default=dict)

    # Status
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('computing', 'Computing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ])
    progress_pct = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)

    # Cached results
    statistics = models.JSONField(null=True)  # All stats including ot_by_k

    # Timestamps
    started_at = models.DateTimeField(null=True)
    completed_at = models.DateTimeField(null=True)
```

---

## Celery Task

```python
@shared_task(bind=True)
def compute_comparison(self, comparison_id: int, config: dict):
    """Compute full comparison, updating progress as we go."""
    result = ComparisonResult.objects.get(comparison_id=comparison_id, cache_key=config['cache_key'])

    try:
        result.status = 'computing'
        result.progress_pct = 10
        result.save()

        # Load analyses
        analyses = [run.qualitative_result for run in result.comparison.runs.all()]

        # Call existing soak comparator
        from soak.comparators.similarity_comparator import SimilarityComparator
        comparator = SimilarityComparator()
        comparison = comparator.compare(analyses, config=config)

        # Store results
        result.statistics = comparison.to_dict()  # or similar serialization
        result.status = 'completed'
        result.progress_pct = 100
        result.completed_at = timezone.now()

    except Exception as e:
        result.status = 'failed'
        result.error_message = str(e)

    result.save()
```

---

## HTMX Views

```python
def comparison_detail(request, comparison_id):
    """Main view - shows cached result or starts computation."""
    comparison = get_object_or_404(Comparison, id=comparison_id)
    config = build_config_from_request(request)
    cache_key = compute_cache_key(comparison, config)

    result, created = ComparisonResult.objects.get_or_create(
        comparison=comparison,
        cache_key=cache_key,
        defaults={'status': 'pending', **config}
    )

    if created or result.status == 'pending':
        compute_comparison.delay(comparison.id, config)

    return render(request, 'comparisons/detail.html', {
        'comparison': comparison,
        'result': result,
    })


def comparison_progress(request, result_id):
    """HTMX polling endpoint."""
    result = get_object_or_404(ComparisonResult, id=result_id)
    return render(request, 'comparisons/partials/progress.html', {'result': result})


def stats_for_k(request, result_id):
    """HTMX endpoint for K-slider (instant, pre-computed)."""
    result = get_object_or_404(ComparisonResult, id=result_id)
    k = float(request.GET.get('k', 0.25))
    stats = result.statistics['ot_by_k'].get(str(k), {})
    return render(request, 'comparisons/partials/k_stats.html', {'k': k, 'stats': stats})
```

---

## Template Structure

```
templates/comparisons/
├── detail.html              # Main page
├── partials/
│   ├── progress.html        # HTMX-polled progress bar
│   ├── config_form.html     # Parameter controls
│   ├── k_slider.html        # K-value slider (instant updates)
│   ├── k_stats.html         # Stats for current K
│   ├── similarity_matrix.html
│   ├── transport_sankey.html
│   └── summary_stats.html
```

Key HTMX patterns:
- Progress polling: `hx-get` with `hx-trigger="every 2s"` until completed
- K-slider: `hx-get` on `change` event, swaps `#k-stats` div
- Lazy sections: `hx-trigger="revealed"` for below-fold content

---

## Decisions

1. **Embedding storage**: pgvector VectorField for future HNSW indexing
2. **Rescale changes**: Full OT recompute (correct, since OT uses rescaled cost matrix)
3. **Templates**: Rewrite as Django templates (not embed existing HTML)
4. **Caching**: Lazy content-addressed (hash of inputs), not explicit DAG

## Dependencies

```
pgvector>=0.2.0
psycopg[binary]
celery[redis]
django-htmx
```

PostgreSQL: `CREATE EXTENSION IF NOT EXISTS vector;`
