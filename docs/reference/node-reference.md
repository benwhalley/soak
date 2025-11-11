# Node Reference

Complete reference for all node types in soak pipelines.

## Node Types

### Split

Divide documents or text into smaller chunks.

**Type:** `Split`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `"chunks"` | Node name |
| `inputs` | List[str] | `["documents"]` | Input nodes (max 1) |
| `chunk_size` | int | `20000` | Target chunk size |
| `min_split` | int | `500` | Minimum chunk size |
| `overlap` | int | `0` | Overlap between chunks (in units) |
| `split_unit` | str | `"tokens"` | Unit: `"chars"`, `"tokens"`, `"words"`, `"sentences"`, `"paragraphs"` |
| `encoding_name` | str | `"cl100k_base"` | Tokenizer for `split_unit="tokens"` |

**Input:** List of documents or TrackedItems
**Output:** List of text chunks (as TrackedItems with provenance)

**Example:**

```yaml
- name: chunks
  type: Split
  chunk_size: 30000
  overlap: 500
  split_unit: tokens
```

**Export:**

```
01_Split_chunks/
├── inputs/
│   ├── 0000_doc_name.txt
│   └── 0000_doc_name_metadata.json
├── outputs/
│   ├── 0000_doc_name__chunks__0.txt
│   ├── 0000_doc_name__chunks__0_metadata.json
│   └── ...
├── split_summary.txt
└── meta.txt
```

**Provenance:**

Source IDs include node name:
- Input: `doc_A`
- Output: `doc_A__chunks__0`, `doc_A__chunks__1`, ...

### Map

Apply an LLM prompt to each item independently in parallel.

**Type:** `Map`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Node name |
| `inputs` | List[str] | Required | Input nodes |
| `template` | str | Required | Jinja2 + struckdown template |
| `model_name` | str | From config | LLM model |
| `max_tokens` | int | `4096` | Max response tokens |
| `temperature` | float | `0.7` | LLM temperature |

**Input:** List of items
**Output:** List of ChatterResult objects (one per input item)

**Template Access:**

- `{{input}}` - Current item content
- `{{__source_id}}` - Source tracking ID
- `{{__metadata}}` - Item metadata
- Any context variables from pipeline

**Example:**

```yaml
- name: summaries
  type: Map
  max_tokens: 8000
  inputs:
    - chunks

---#summaries
Summarize this text in 2-3 sentences:

{{input}}

[[summary]]
```

**Export:**

```
02_Map_summaries/
├── inputs/
│   ├── 0000_doc__chunks__0.txt
│   └── ...
├── 0000_doc__chunks__0_prompt.md
├── 0000_doc__chunks__0_response.json
└── ...
```

### Classifier

Extract structured data from each item using multiple choice and typed fields.

**Type:** `Classifier`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Node name |
| `inputs` | List[str] | Required | Input nodes |
| `template` | str | Required | Template with structured outputs |
| `model_name` | str | From config | Single model name |
| `model_names` | List[str] | None | Multiple models for agreement analysis |
| `agreement_fields` | List[str] | None | Fields to calculate agreement on |
| `max_tokens` | int | `4096` | Max response tokens |

**Input:** List of items
**Output:** List of dictionaries with extracted fields

**Template Syntax:**

- `[[pick:field|opt1,opt2,opt3]]` - Single choice
- `[[pick*:field|opt1,opt2]]` - Multiple choice
- `[[int:field]]` - Integer
- `[[boolean:field]]` - True/False
- `[[text:field]]` - Free text
- `¡OBLIVIATE` - Clear context between questions

**Example:**

```yaml
- name: classify
  type: Classifier
  model_names:
    - gpt-4o-mini
    - gpt-4.1-mini
  agreement_fields:
    - topic
    - sentiment
  inputs:
    - chunks

---#classify
Classify this text:

{{input}}

What is the topic?
[[pick:topic|health,tech,education,other]]

¡OBLIVIATE

What is the sentiment?
[[pick:sentiment|positive,negative,neutral]]
```

**Multi-model Agreement:**

When `model_names` has 2+ models:
- Each model classifies independently
- Agreement statistics calculated (Gwet's AC1, Krippendorff's Alpha, % agreement)
- Results include per-model classifications and statistics

**Export:**

```
03_Classifier_classify/
├── inputs/
├── classifications.csv          # Main output with source tracking
├── classifications.json
├── summary.txt                  # Field distributions
├── prompt_template.sd.md
├── agreement_stats.json         # If multi-model
├── human_rating_template.txt    # Template for human raters
└── 0000_*_response.json         # Per-item responses
```

**CSV Format:**

```csv
index,source_id,doc_index,original_file,topic,sentiment
0,doc__chunks__0,0,data/doc.txt,health,positive
1,doc__chunks__1,0,data/doc.txt,tech,neutral
```

### Reduce

Concatenate multiple items into single text.

**Type:** `Reduce`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Node name |
| `inputs` | List[str] | Required | Input nodes (max 1) |
| `template` | str | `"{{input}}\n"` | Template for each item |

**Input:** List of items
**Output:** Single concatenated string

**Example:**

```yaml
- name: all_codes
  type: Reduce
  inputs:
    - chunk_codes

---#all_codes
{{input.codes}}
```

Extracts `.codes` field from each ChatterResult and concatenates.

**Export:**

```
04_Reduce_all_codes/
├── inputs/
├── result.txt
└── meta.txt
```

### Transform

Apply LLM prompt to single input item (often the output of Reduce).

**Type:** `Transform`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Node name |
| `inputs` | List[str] | Required | Input nodes |
| `template` | str | Required | Jinja2 + struckdown template |
| `model_name` | str | From config | LLM model |
| `max_tokens` | int | `4096` | Max response tokens |
| `temperature` | float | `0.7` | LLM temperature |

**Input:** Single item (asserts exactly one input)
**Output:** ChatterResult

**Example:**

```yaml
- name: codes
  type: Transform
  max_tokens: 32000
  inputs:
    - all_codes
    - all_themes

---#codes
Consolidate these preliminary codes:

{{all_codes}}

And these themes:

{{all_themes}}

[[codenotes]]

[[codes:codes]]
```

**Multiple Inputs:**

When multiple inputs, all are available in template context:

```yaml
inputs:
  - all_codes
  - all_themes

# Template can access:
{{all_codes}}
{{all_themes}}
```

**Export:**

```
05_Transform_codes/
├── inputs/
├── prompt.md
├── response.json
├── result.json
└── meta.txt
```

### VerifyQuotes

Unified quote verification node that can:
- Extract quotes from **Codes** OR **Themes**
- Search in **documents** OR any **custom node output**
- Verify quote **existence** (BM25 + embeddings + LLM-as-judge)
- Optionally verify **fairness** of quote usage (for themes)

**Type:** `VerifyQuotes`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `"checkquotes"` | Node name |
| `quotes_from` | str | required* | Node containing quotes (Codes or Themes) |
| `search_in` | str | `null` | Node to search in (null = documents) |
| `check_fairness` | bool | `false` | Enable fairness verification (themes only) |
| `context_window_size` | int | `1000` | Context window for fairness check |
| `window_size` | int | `300` | Window size for BM25 search |
| `overlap` | int | `null` | Window overlap (auto: 30% of window_size) |
| `bm25_k1` | float | `1.5` | BM25 term frequency saturation |
| `bm25_b` | float | `0.4` | BM25 length normalization |
| `ellipsis_max_gap` | int | `3` | Max windows between ellipsis head/tail |
| `trim_spans` | bool | `true` | Enable span refinement |
| `trim_method` | str | `"fuzzy"` | Trimming: `"fuzzy"`, `"sliding_bm25"`, `"hybrid"` |
| `min_fuzzy_ratio` | float | `0.6` | Minimum fuzzy match quality threshold |
| `expand_window_neighbors` | int | `1` | Expand search to ±N windows if truncated |
| `template` | str | `null` | Custom LLM existence verification template |
| `fairness_template` | str | `null` | Custom fairness verification template |

\* **Note:** For backward compatibility, `inputs[0]` is used if `quotes_from` is not specified.

**Input:** Codes OR Themes (containing quotes)
**Output:** Verification results with existence metrics and optional fairness verification

**How it Works:**

**Stage 1: Existence Verification (BM25 + Embeddings)**
1. Extracts quotes from Codes or Themes
2. Creates overlapping windows from search corpus
3. Builds BM25 index over windows
4. For each quote:
   - Finds best BM25 window (with ellipsis support)
   - Trims span to align with quote boundaries
   - Expands to neighbor windows if truncated
5. Computes embedding similarity
6. Tracks source document and positions

**Stage 1.5: LLM Existence Check (for poor matches)**
- Runs LLM-as-judge on quotes with low BM25/cosine scores
- Asks: "Is this quote contained in the source text?"
- Returns explanation + boolean verification

**Stage 2: Fairness Verification (optional, themes only)**
- If `check_fairness=True` and input is Themes:
  - Extracts context window around each quote
  - Presents LLM with: Theme + Code + Quote + Context
  - Asks: "Is this quote used fairly to support this theme?"
  - Returns explanation + boolean fairness judgment

**Examples:**

**Verify Code quotes (backward compatible):**
```yaml
- name: checkquotes
  type: VerifyQuotes
  quotes_from: codes  # or use old 'inputs: [codes]'
  window_size: 450
```

**Verify Theme quotes with fairness checking:**
```yaml
- name: verify_themes
  type: VerifyQuotes
  quotes_from: themes
  check_fairness: true
  context_window_size: 1000
```

**Search in custom corpus:**
```yaml
- name: verify_in_summaries
  type: VerifyQuotes
  quotes_from: codes
  search_in: summaries  # Search in 'summaries' node output instead of documents
```

**Export:**

```
07_VerifyQuotes_checkquotes/
├── quote_verification.xlsx           # Formatted Excel (sorted by fairness/confidence)
├── stats.csv                          # Aggregate statistics
├── info.txt                           # Algorithm description
├── meta.txt
├── llm_existence_checks/              # LLM prompts/responses for poor matches
│   ├── 0000_{hash}_prompt.md
│   ├── 0000_{hash}_response.txt
│   └── 0000_{hash}_response.json
└── llm_fairness_checks/               # LLM prompts/responses (themes only)
    ├── 0000_{hash}_prompt.md
    ├── 0000_{hash}_response.txt
    └── 0000_{hash}_response.json
```

**Output Metrics:**

**For all quotes:**
- `bm25_score`: Lexical relevance score
- `bm25_ratio`: Match uniqueness (top1/top2)
- `cosine_similarity`: Embedding similarity (0-1)
- `match_ratio`: Fuzzy alignment quality (if trimming enabled)
- `source_doc`: Source document name
- `global_start`, `global_end`: Character positions
- `span_text`: Matched text from source
- `llm_explanation`: LLM explanation (poor matches only)
- `llm_is_contained`: Boolean existence verification (poor matches only)

**Additionally for themes (if `check_fairness=True`):**
- `theme`: Theme name
- `theme_description`: Theme description
- `code_name`: Code name
- `code_description`: Code description
- `llm_fairness_explanation`: LLM explanation for fairness
- `llm_is_fair`: Boolean fairness judgment

**Interpreting Results:**

**Codes:**
| BM25 Score | BM25 Ratio | Cosine Sim | LLM Contained | Interpretation |
|-----------|-----------|-----------|---------------|----------------|
| High | High | ~1.0 | N/A | ✓ Perfect verbatim match |
| High | High | >0.9 | N/A | ✓ Near-exact (minor edits) |
| Low | Low | >0.85 | True | ⚠ Poor match but LLM confirms |
| Low | Low | <0.7 | False | ✗ Likely hallucination |

**Themes:**
| Existence | Fairness | Interpretation |
|-----------|----------|----------------|
| High BM25/Cosine | True | ✓ Quote exists and supports theme |
| High BM25/Cosine | False | ⚠ Quote exists but taken out of context |
| Low BM25/Cosine | True | ⚠ Poor match but fair usage |
| Low BM25/Cosine | False | ✗ Hallucinated or misused |

**See Also:**
- [Quote Verification Algorithm](quote-verification.md) - Detailed algorithm specification
- [Quote Verification Approach](../explanation/quote-verification-approach.md) - Design rationale

### Batch

Group items into batches for processing.

**Type:** `Batch`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Node name |
| `inputs` | List[str] | Required | Input nodes |
| `batch_size` | int | `10` | Items per batch |

**Input:** List of items
**Output:** BatchList (list of lists)

**Example:**

```yaml
- name: batched_chunks
  type: Batch
  batch_size: 5
  inputs:
    - chunks
```

Used with Reduce to process batches:

```yaml
- name: batch_summaries
  type: Reduce
  inputs:
    - batched_chunks

---#batch_summaries
Summarize these {{input|length}} chunks together:
{% for chunk in input %}
{{chunk}}
{% endfor %}
```

### GroupBy

Group items by one or more field values, creating nested batch structures.

**Type:** `GroupBy`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Node name |
| `inputs` | List[str] | Required | Input nodes |
| `group_by` | List[str] | Required | Field names to group by |

**Input:** List of items (TrackedItem or dict with metadata/outputs)
**Output:** BatchList (nested if multiple group_by fields)

**How It Works:**

- Groups items by values in specified fields
- Fields can be from metadata, outputs, or ChatterResult attributes
- Multiple fields create nested BatchLists
- Each batch contains items sharing same field values

**Single Field Example:**

```yaml
- name: by_category
  type: GroupBy
  group_by:
    - category
  inputs:
    - classified_items
```

If items have categories ["health", "tech", "health"], creates 2 batches:
- Batch 1: Items with category="health"
- Batch 2: Items with category="tech"

**Multi-Field Example:**

```yaml
- name: by_category_and_sentiment
  type: GroupBy
  group_by:
    - category
    - sentiment
  inputs:
    - classified_items
```

Creates nested structure:
- health → positive → [items]
- health → negative → [items]
- tech → positive → [items]
- tech → neutral → [items]

**Use Cases:**

- Process items differently based on classification
- Analyze patterns within categories
- Create hierarchical groupings
- Prepare for category-specific transformations

### Ungroup

Flatten all BatchList nesting levels, returning a flat list.

**Type:** `Ungroup`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Node name |
| `inputs` | List[str] | Required | Input nodes (must be BatchList) |

**Input:** BatchList (any nesting level)
**Output:** Flat list of items

**How It Works:**

- Recursively flattens all batch nesting
- Preserves original item order
- Removes all grouping structure

**Example:**

```yaml
- name: flattened
  type: Ungroup
  inputs:
    - grouped_items
```

Converts nested structure:
```
[
  [item1, item2],      # Batch 1
  [item3],             # Batch 2
  [[item4], [item5]]   # Nested batches
]
```

To flat list:
```
[item1, item2, item3, item4, item5]
```

**Use Cases:**

- Remove grouping after category-specific processing
- Prepare batched results for non-batch-aware nodes
- Flatten before final output
- Combine results from multiple batch levels

### Filter

Filter items based on a boolean expression (no LLM call).

**Type:** `Filter`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Node name |
| `inputs` | List[str] | Required | Input nodes (must produce list) |
| `expression` | str | Required | Python expression using simpleeval |

**Input:** List of items (TrackedItem or ChatterResult)
**Output:** Filtered list (items where expression is truthy)

**Filter Modes:**

1. **LLM Mode** - Run template through LLM, filter on extracted fields
2. **Simple Mode** - Filter directly on item data

Mode auto-detected: if `template` provided, uses LLM mode.

**LLM Mode Example:**

```yaml
- name: filtered
  type: Filter
  template: "Is this relevant? [[bool:is_relevant]]"
  expression: "is_relevant == True"
  inputs: [chunks]
```

**Simple Mode Example:**

```yaml
- name: long_chunks
  type: Filter
  expression: "len(input) > 100"
  inputs: [chunks]
```

**Common Expressions:**

```python
# Boolean response (ChatterResult)
"item['decision_node'].response is True"

# Numeric threshold from outputs
"item['score_node'].outputs['score'] > 0.5"

# Multiple conditions
"item['category_node'].outputs['category'] == 'relevant' and item['score_node'].outputs['score'] > 0.3"
```

**Use Case:**

Filtering items based on LLM decisions or computed scores without additional LLM calls.

## Common Patterns

### Parallel Processing

```yaml
nodes:
  - name: chunks
    type: Split

  - name: process_chunks
    type: Map          # Processes all chunks in parallel
    inputs: [chunks]
```

### Collect and Consolidate

```yaml
nodes:
  - name: chunk_codes
    type: Map

  - name: all_codes
    type: Reduce       # Concatenate all outputs
    inputs: [chunk_codes]

  - name: final_codes
    type: Transform    # Consolidate into final result
    inputs: [all_codes]
```

### Multi-input Transform

```yaml
nodes:
  - name: codes
    type: Transform

  - name: themes
    type: Transform
    inputs: [codes]    # Uses codes output

  - name: narrative
    type: Transform
    inputs:
      - codes          # Access both in template
      - themes
```

### Classification Pipeline

```yaml
nodes:
  - name: chunks
    type: Split

  - name: classify
    type: Classifier
    inputs: [chunks]
```

### Nested Splits

```yaml
nodes:
  - name: chapters
    type: Split
    chunk_size: 50000

  - name: paragraphs
    type: Split
    chunk_size: 5000
    inputs: [chapters]    # Split the splits
```

Provenance: `book__chapters__0__paragraphs__2`

## Template Reference

### Available Variables

**In all nodes:**
- Pipeline `default_context` variables
- All previous node results (by node name)

**In ItemsNode (Map, Classifier, Transform):**
- `{{input}}` - Current item content
- `{{__source_id}}` - Provenance ID
- `{{__metadata}}` - Item metadata dict
- `{{__tracked_item}}` - Full TrackedItem object

**In Reduce:**
- `{{input}}` - Each item being reduced
- Named node variables (e.g., `{{input.codes}}`)

### Jinja2 Features

**Conditionals:**

```jinja
{% if research_question %}
Research question: {{research_question}}
{% endif %}
```

**Loops:**

```jinja
{% for item in items %}
- {{item}}
{% endfor %}
```

**Filters:**

```jinja
{{text|length}}
{{text|upper}}
{{list|join(", ")}}
```

### Struckdown Syntax

**Return types for thematic analysis:**

```
[[codes:codes]]          # List[Code]
[[themes:themes]]        # List[Theme]
[[extract:text]]         # Free text
[[report]]               # Free text (narrative)
```

**Return types for classification:**

```
[[pick:field|a,b,c]]     # Single choice
[[pick*:field|a,b]]      # Multiple choice (list)
[[int:field]]            # Integer
[[boolean:field]]        # True/False
[[text:field]]           # Free text string
```

**Context control:**

```
¡BEGIN                   # Start new context
¡OBLIVIATE               # Clear context between questions
```

## Node Configuration

### Global Config

Set in pipeline front matter:

```yaml
config:
  model_name: openai/gpt-4.1-mini
  llm_credentials:
    api_key: ${LLM_API_KEY}
    base_url: ${LLM_API_BASE}
```

### Per-node Overrides

```yaml
- name: detailed_analysis
  type: Map
  model_name: openai/gpt-4o     # Override for this node
  max_tokens: 16000
  temperature: 0.3
  inputs: [chunks]
```

## Next Steps

- [Pipeline Format](pipeline-format.md) - YAML structure
- [How-to: Thematic Analysis](../how-to/thematic-analysis.md) - Using nodes together
- [Node Types Explanation](../explanation/node-types.md) - When to use which node
