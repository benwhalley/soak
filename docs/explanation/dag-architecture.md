---
layout: default
title: DAG Architecture
parent: Explanation
nav_order: 2
---

# DAG Architecture

This document explains why soak uses a DAG (Directed Acyclic Graph) architecture and how pipeline execution works.

## What is a DAG?

A **Directed Acyclic Graph** is a data structure where:

- **Directed**: Edges (connections) have a direction -- data flows from one node to another
- **Acyclic**: No loops -- you can't follow connections back to where you started
- **Graph**: Nodes (steps) connected by edges (dependencies)

In soak, each pipeline is a DAG where:
- **Nodes** are processing steps (Split, Map, Reduce, Transform, etc.)
- **Edges** represent data dependencies (node B depends on output from node A)

## Why Use a DAG?

### 1. Explicit Dependencies

Traditional scripts hide dependencies in execution order:

```python
# Traditional script - dependencies implicit
chunks = split_documents(docs)
codes = map_coding(chunks)  # Hidden dependency on chunks
themes = reduce_themes(codes)  # Hidden dependency on codes
```

DAG pipelines make dependencies explicit:

```yaml
nodes:
  - name: chunks
    type: Split

  - name: codes
    type: Map
    inputs:
      - chunks  # Explicit dependency

  - name: themes
    type: Reduce
    inputs:
      - codes  # Explicit dependency
```

### 2. Automatic Parallelization

The DAG structure enables automatic parallel execution. Nodes without dependencies can run simultaneously:

```yaml
nodes:
  - name: chunks
    type: Split

  # These two can run in parallel (both depend only on chunks)
  - name: extract_codes
    type: Map
    inputs:
      - chunks

  - name: extract_themes
    type: Map
    inputs:
      - chunks
```

The execution engine automatically identifies parallelizable nodes and runs them concurrently.

### 3. Reproducibility

DAG pipelines are declarative -- they describe **what** should happen, not **how**:

- Same pipeline + same data = same results
- No hidden state or ordering requirements
- Can be version controlled, shared, and re-run

### 4. Composability

Nodes are modular and reusable:

```yaml
# Reuse same Map node with different inputs
- name: code_chunks
  type: Map
  inputs:
    - chunks

- name: code_paragraphs
  type: Map
  inputs:
    - paragraphs
```

## Pipeline Execution Model

### Execution Phases

1. **Parse**: Load YAML, parse templates, build dependency graph
2. **Validate**: Check for cycles, missing nodes, invalid syntax
3. **Topological Sort**: Determine execution order respecting dependencies
4. **Execute**: Run nodes in batches, parallelizing independent nodes
5. **Export**: Write results to JSON, HTML, and dump directory

### Dependency Resolution

soak uses topological sorting to determine execution order:

```yaml
nodes:
  - name: A
    type: Split

  - name: B
    type: Map
    inputs: [A]

  - name: C
    type: Map
    inputs: [A]

  - name: D
    type: Reduce
    inputs: [B, C]
```

Execution batches:
1. **Batch 1**: A (no dependencies)
2. **Batch 2**: B, C (both depend only on A, run in parallel)
3. **Batch 3**: D (waits for B and C to complete)

### Concurrency Model

soak uses **structured concurrency** via `anyio`:

- **Batch-level parallelism**: Independent nodes in same batch run concurrently
- **Item-level parallelism**: Within a node, items process in parallel (e.g., Map processes multiple chunks simultaneously)
- **Semaphore control**: `MAX_CONCURRENCY` limits simultaneous operations (default: 20)

Example execution flow:

```
Documents → Split → [chunk1, chunk2, chunk3]
                          ↓
                    Map (parallel)
                          ↓
              [code1, code2, code3] ← All 3 process concurrently
                          ↓
                       Reduce
```

### Timeout and Error Handling

- **Max runtime**: Pipelines timeout after 30 minutes (configurable via `SOAK_MAX_RUNTIME`)
- **Context exceeded**: If LLM context limit exceeded, item can be skipped (configurable via `fail_on_context_exceeded`)
- **Content policy**: Content policy violations can be skipped or fail pipeline (configurable via `skip_content_policy_violations`)

## Data Flow Through the DAG

### TrackedItem Propagation

All data flows through the DAG as `TrackedItem` objects:

```python
TrackedItem(
    content="...",
    id="document_1",
    sources=["document_1"],
    metadata={"original_path": "data/doc.txt", ...}
)
```

As data transforms:
1. **Split**: One TrackedItem → Many TrackedItems (e.g., `doc1__chunks__0`)
2. **Map**: TrackedItems → Structured data (codes, classifications)
3. **Reduce**: Many items → Aggregated text
4. **Transform**: Text → Structured output

### Context Dictionary

Each node accesses results via the shared context dictionary:

```yaml
- name: all_codes
  type: Reduce
  inputs:
    - chunk_codes  # References previous node by name
```

In templates:
```jinja2
{{chunk_codes}}  # Access node result by name
{{input}}        # Default input (first in inputs list)
```

### Provenance Tracking

TrackedItem IDs preserve lineage:

```
document_1                    # Original document
  ↓
document_1__chunks__0         # After splitting
  ↓
document_1__chunks__0__codes  # After coding
```

The `sources` list tracks all original documents contributing to each item.

## Document Ordering and Bias Prevention

### The Problem: Systematic Ordering Bias

When using glob patterns to load documents, file system ordering can introduce systematic bias:

```bash
uv run soak zs data/group1/*.txt data/group2/*.txt
```

Without shuffling, all `group1` documents appear first in the context window, followed by all `group2` documents. This creates a problem because:

1. **Attention effects**: LLMs may attend differently to content at different positions in long prompts -- early content can receive less attention in very long contexts ("lost in the middle" effect)
2. **Systematic bias**: If document groups correlate with meaningful categories (e.g., treatment vs control, different time periods, different sources), ordering effects could systematically bias the analysis
3. **Hidden confounds**: Folder structure often reflects data collection order, source, or other factors that shouldn't influence coding

### The Solution: Randomised Document Order

By default, soak shuffles documents before processing:

```yaml
default_config:
  randomise_document_order: true  # default
  seed: 42                        # for reproducibility
```

This ensures documents from different sources are interleaved randomly, preventing any single group from being systematically positioned in a particular part of the context window.

To disable shuffling, in your `.soak` pipeline file:

```yaml
default_config:
  randomise_document_order: false
```

But for qualitative analyses, keep shuffling enabled to avoid unintentional ordering effects.


## DAG Optimization

### Batching

Nodes execute in dependency batches:

```
Batch 1: [A]           ← 1 node
Batch 2: [B, C, D]     ← 3 nodes in parallel
Batch 3: [E, F]        ← 2 nodes in parallel
Batch 4: [G]           ← 1 node
```

Minimizing batches (by reducing dependencies) speeds up execution.

### Lazy Evaluation

Nodes only execute when their outputs are needed:

- If you only need intermediate results, downstream nodes don't run
- DAG structure makes selective execution possible (future feature)

### Caching

Results cache between runs:

- **Embeddings**: Cached in `.embeddings/` directory
- **Documents**: Extracted text cached based on file mtime
- **Future**: Node-level caching for expensive operations

## Comparison with Other Approaches

### vs. Linear Scripts

**Script:**
```python
chunks = split(docs)
codes = code_chunks(chunks)
themes = generate_themes(codes)
```

**Limitations:**
- No automatic parallelization
- Dependencies implicit
- Hard to modify or extend
- No provenance tracking

**DAG:**
```yaml
nodes:
  - name: chunks
    type: Split
  - name: codes
    type: Map
    inputs: [chunks]
  - name: themes
    type: Transform
    inputs: [codes]
```

**Advantages:**
- Automatic parallelization
- Explicit dependencies
- Easy to modify (add nodes, change connections)
- Built-in provenance

### vs. Task Queues (Celery, Airflow)

**Task Queues** are designed for distributed systems:
- Multiple workers across machines
- Persistent task storage
- Complex scheduling

**soak DAG** is designed for single-machine LLM pipelines:
- All execution in one process (with concurrency)
- In-memory state
- Simple execution model

## DAG Configuration

DAG behavior configured via `DAGConfig`:

```python
DAGConfig(
    model_name="gpt-4o",
    show_progress=True,          # Show progress bars
    fail_on_context_exceeded=True,  # Fail if context too large
    export_enabled=True,          # Export nodes as they finish
    export_folder=Path("./results_dump")
)
```

## Visualization

Future feature: Generate visual DAG diagrams:

```
┌─────────┐
│Documents│
└────┬────┘
     │
     ▼
  ┌─────┐
  │Split│
  └──┬──┘
     │
   ┌─┴─┐
   ▼   ▼
┌──┴┐ ┌┴──┐
│Map│ │Map│
└─┬─┘ └─┬─┘
  │     │
  └──┬──┘
     ▼
  ┌──────┐
  │Reduce│
  └──────┘
```

## Next Steps

- [Node Types](node-types.md) - Understanding different node categories
- [Node Reference](../reference/node-reference.md) - Full node specifications
- [Template System](template-system.md) - How templates work with the DAG
