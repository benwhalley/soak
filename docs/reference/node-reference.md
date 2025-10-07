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
| `template_text` | str | Required | Jinja2 + struckdown template |
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
| `template_text` | str | Required | Template with structured outputs |
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
    - litellm/gpt-4o-mini
    - litellm/gpt-4.1-mini
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
| `template_text` | str | `"{{input}}\n"` | Template for each item |

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
| `template_text` | str | Required | Jinja2 + struckdown template |
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

Verify that extracted quotes appear in source documents using BM25 lexical search and embedding-based semantic similarity.

**Type:** `VerifyQuotes`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `"checkquotes"` | Node name |
| `inputs` | List[str] | `["codes"]` | Input nodes |
| `window_size` | int | `null` | Window size (auto: 1.1× max quote, cap 500) |
| `overlap` | int | `null` | Window overlap (auto: 30% of window_size) |
| `bm25_k1` | float | `1.5` | BM25 term frequency saturation |
| `bm25_b` | float | `0.4` | BM25 length normalization (lower = less penalty) |
| `ellipsis_max_gap` | int | `null` | Max windows between ellipsis head/tail |
| `trim_spans` | bool | `true` | Enable span refinement to quote boundaries |
| `trim_method` | str | `"fuzzy"` | Trimming method: `"fuzzy"`, `"sliding_bm25"`, `"hybrid"` |
| `min_fuzzy_ratio` | float | `0.6` | Minimum fuzzy match quality threshold |
| `expand_window_neighbors` | int | `1` | Expand search to ±N windows if truncated |

**Input:** Codes with quotes (expects node named "codes" in context)
**Output:** Verification results with BM25 scores, cosine similarity, and source locations

**How it Works:**

1. Creates overlapping windows from source documents (adaptive sizing)
2. Builds BM25 index over windows
3. For each quote:
   - If no ellipsis: Finds best BM25 window
   - If ellipsis: Matches head/tail separately, reconstructs span
   - Trims span to align with quote boundaries (optional)
   - Expands to neighbor windows if match appears truncated
4. Computes embedding similarity between quote and matched span
5. Tracks source document and character positions

**Example:**

```yaml
- name: checkquotes
  type: VerifyQuotes
  inputs:
    - codes
  window_size: 450
  overlap: 150
  trim_method: "fuzzy"
  expand_window_neighbors: 1
```

**Export:**

```
07_VerifyQuotes_checkquotes/
├── quote_verification.xlsx      # Formatted Excel (sorted by confidence)
├── stats.csv                    # Aggregate statistics
├── info.txt                     # Algorithm description
└── meta.txt
```

**Output Metrics:**

Each verified quote includes:
- `bm25_score`: Lexical relevance score
- `bm25_ratio`: Match uniqueness (top1/top2)
- `cosine_similarity`: Embedding similarity (0-1)
- `match_ratio`: Fuzzy alignment quality (if trimming enabled)
- `source_doc`: Source document name
- `global_start`, `global_end`: Character positions
- `span_text`: Matched text from source
- `context_window`: ±300 chars around match

**Interpreting Results:**

| BM25 Score | BM25 Ratio | Cosine Sim | Interpretation |
|-----------|-----------|-----------|----------------|
| High | High | ~1.0 | ✓ Perfect verbatim match |
| High | High | >0.9 | ✓ Near-exact (minor edits) |
| Low-Med | Low | >0.85 | ⚠ Paraphrase (verify manually) |
| Medium | Low | >0.8 | ⚠ Possible truncation |
| Low | Low | <0.7 | ✗ Likely hallucination |

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

### TransformReduce

Recursively reduce long inputs by splitting, transforming, and re-reducing until single output.

**Type:** `TransformReduce`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Node name |
| `inputs` | List[str] | Required | Input nodes |
| `template_text` | str | Required | Transform template |
| `reduce_template` | str | `"{{input}}\n\n"` | Reduce template |
| `chunk_size` | int | `20000` | Split size for long inputs |
| `split_unit` | str | `"tokens"` | Unit for splitting |
| `max_levels` | int | `5` | Max recursion depth |
| `max_tokens` | int | `4096` | Max response tokens |

**Input:** Single long text
**Output:** Recursively reduced result

**How it Works:**

1. If input fits in `chunk_size`, transform directly
2. If too long: split → transform each → concatenate → recurse
3. Repeat until single output or max_levels reached

**Example:**

```yaml
- name: long_summary
  type: TransformReduce
  chunk_size: 10000
  max_levels: 3
  inputs:
    - very_long_document

---#long_summary
Summarize this text concisely:

{{input}}

[[summary]]
```

**Use Case:**

Summarizing documents longer than LLM context window.

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
