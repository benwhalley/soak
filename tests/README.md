# Soak LLM Node Tests

Comprehensive test suite for all node types in the soak pipeline system.

## Test Coverage

### ✅ Passing Tests

1. **test_split_node** - Tests Split node with different split_unit options
   - tokens, words, sentences, paragraphs splitting
   - chunk_size and overlap parameters
   - Provenance tracking (source_id in chunks)

2. **test_map_node** - Tests Map node parallel processing
   - Processes each item independently
   - Output count equals input count
   - All items processed

3. **test_classifier_node_with_agreement** - Tests Classifier with multiple models
   - Uses 2 models (gpt-4o-mini, gpt-4.1-mini)
   - Agreement statistics calculation (gwet_ac1, kripp_alpha, percent_agreement)
   - CSV/JSON export for each model
   - Agreement stats export

4. **test_filter_node** - Tests Filter node inclusion/exclusion logic
   - Some items pass filter
   - Some items excluded
   - Tracks both included and excluded items

5. **test_transform_node_multi_slot** - Tests Transform node
   - Single input (aggregated via Reduce first)
   - ChatterResult structured output
   - Response validation

6. **test_reduce_node** - Tests Reduce node aggregation
   - Multiple inputs → single string output
   - Concatenates all inputs

7. **test_batch_node** - Tests Batch node grouping
   - Groups items into batches
   - Returns BatchList with batches attribute
   - Reduces total item count

8. **test_pipeline_serialization** - Tests JSON serialization
   - Pipelines can serialize to JSON
   - Includes all nodes and configuration

### ⏭️ Skipped Tests

- **test_verify_quotes_node** - Requires complex setup with `codes` node containing quotes
  - Should be tested as part of full integration pipeline
  - Needs specific structure: response.codes with quotes attribute

## Running Tests

Run all tests:
```bash
uv run pytest tests/test_nodes.py -v
```

Run specific test:
```bash
uv run pytest tests/test_nodes.py::test_split_node -v
```

Run with detailed output:
```bash
uv run pytest tests/test_nodes.py -v -s
```

## Test Data

Tests use `data/chainstorecoffee.txt` - an article about coffee pricing and inflation.

## Test Pipelines

All test pipelines are in `tests/pipelines/`:

- `test_split.yaml` - Split node with 4 split_unit options
- `test_map.yaml` - Map node summarization
- `test_classifier.yaml` - Classifier with 2 models and agreement fields
- `test_filter.yaml` - Filter for coffee-related content
- `test_transform.yaml` - Transform with Reduce step
- `test_reduce.yaml` - Reduce aggregation
- `test_batch.yaml` - Batch grouping
- `test_verify_quotes.yaml` - Quote verification (not used - too complex for unit test)

## Test Design Principles

1. **Avoid Brittleness**
   - Test structural properties (counts, types) not exact LLM outputs
   - Use simple, predictable prompts
   - Focus on pipeline mechanics, not content quality

2. **Speed**
   - Use small, fast models (gpt-4o-mini, gpt-4.1-mini)
   - Keep test data small
   - Parallel execution where possible

3. **Coverage**
   - Each node type has dedicated test
   - Tests verify core functionality
   - Tests validate output structure

## Key Findings

### Node Output Types

- **Split**: List[TrackedItem] - chunks with provenance
- **Map**: List[TrackedItem/ChatterResult] - one per input
- **Classifier**: List[TrackedItem] with classifications
- **Filter**: List[TrackedItem] (filtered subset)
- **Transform**: ChatterResult (single structured output)
- **Reduce**: str (concatenated string)
- **Batch**: BatchList (object with .batches attribute containing tuples)

### Important Notes

1. **TrackedItem** - Used throughout for provenance tracking
   - Has `content`, `source_id`, `metadata` attributes
   - Source_id format: `parent__nodename__index`

2. **ChatterResult** - LLM completion results
   - Has `response` attribute (can be str or dict)
   - Structured outputs accessed via response

3. **Agreement Statistics** - Classifier node
   - Requires multiple `model_names`
   - Requires `agreement_fields` to specify which fields to compare
   - Auto-exports CSVs and agreement stats

4. **Batch Node** - Returns special BatchList object
   - Not a plain list - has `.batches` attribute
   - Each batch is a tuple/iterable of items

## Future Enhancements

1. Add integration test for VerifyQuotes with full pipeline
2. Test error handling and edge cases
3. Add performance benchmarks
4. Test with larger documents
5. Test concurrent pipeline execution
6. Add tests for TransformReduce node
