# Task: Add Cost Tracking to Struckdown Embeddings

## Problem

Struckdown's embedding functions (`get_embedding`, `get_embedding_async`) return raw numpy arrays, unlike LLM calls which return `ChatterResult` objects with cost information. This means embedding costs cannot be tracked for budget enforcement or reporting.

## Current Interface

```python
from struckdown import get_embedding, get_embedding_async

# Returns np.ndarray directly - no cost info
embedding = get_embedding("some text", model="text-embedding-3-small")
embeddings = await get_embedding_async(["text1", "text2"], model="text-embedding-3-small")
```

## Desired Behaviour

The return value should:
1. **Work exactly like before** -- callers using it as an array don't need to change
2. **Expose cost information** via `.cost` or similar attribute

```python
# Existing code continues to work unchanged
embedding = get_embedding("some text")
similarity = np.dot(embedding, other_embedding)  # still works

# But now you can also access costs
print(embedding.cost)          # 0.00002
print(embedding.tokens)        # 5
print(embedding.model)         # "text-embedding-3-small"
print(embedding.cached)        # False
```

## Suggested Implementation

Create an `EmbeddingResult` class that wraps the numpy array but delegates array operations:

```python
class EmbeddingResult(np.ndarray):
    """
    Numpy array subclass that carries cost metadata.

    Behaves exactly like np.ndarray for all operations,
    but also exposes .cost, .tokens, .model, .cached attributes.
    """

    def __new__(cls, array, cost=0.0, tokens=0, model="", cached=False):
        obj = np.asarray(array).view(cls)
        obj.cost = cost
        obj.tokens = tokens
        obj.model = model
        obj.cached = cached
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.cost = getattr(obj, 'cost', 0.0)
        self.tokens = getattr(obj, 'tokens', 0)
        self.model = getattr(obj, 'model', "")
        self.cached = getattr(obj, 'cached', False)
```

## For Batch Embeddings

When embedding multiple texts, return a container that:
- Iterates/indexes like a list of arrays
- Has aggregate `.total_cost`, `.total_tokens`
- Each item has individual cost info

```python
results = get_embedding(["text1", "text2", "text3"])

# Iterate as before
for emb in results:
    process(emb)

# Access aggregate costs
print(results.total_cost)    # 0.00006
print(results.total_tokens)  # 15

# Or per-item
print(results[0].cost)       # 0.00002
```

## Files to Modify in Struckdown

1. **struckdown/embeddings.py** (or wherever `get_embedding` lives)
   - Create `EmbeddingResult` class
   - Modify `get_embedding` and `get_embedding_async` to return `EmbeddingResult`
   - Calculate cost from token count and model pricing

2. **struckdown/pricing.py** (or similar)
   - Add embedding model pricing data
   - Helper to calculate cost from tokens + model

## Files to Update in Soak (callers)

After struckdown is updated, these soak files use embeddings and could benefit from cost tracking:

1. `soak/comparators/similarity.py` -- comparison embeddings
2. `soak/nodes/cluster.py` -- clustering embeddings
3. `soak/nodes/verify_quotes.py` -- quote verification embeddings
4. `soak/coverage.py` -- coverage analysis embeddings

For each, optionally accumulate embedding costs into the node's cost tracking:

```python
# Before (still works)
embedding = get_embedding(text)

# After (if we want to track costs)
embedding = get_embedding(text)
if hasattr(self, '_accumulate_embedding_cost'):
    self._accumulate_embedding_cost(embedding.cost, embedding.tokens)
```

## Testing

1. Existing tests should pass unchanged (backwards compatible)
2. New tests for cost attributes
3. Test that array operations preserve/propagate cost metadata appropriately

## Notes

- OpenAI returns `usage.total_tokens` in embedding responses -- use this
- Local models (e.g., sentence-transformers) have zero API cost but could track compute time
- Cached embeddings should have `cached=True` and `cost=0.0`
