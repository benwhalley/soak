# Node Types

This document explains the different categories of nodes in soak and when to use each type.

## Node Categories

soak provides several node types that fall into distinct categories based on their role in data processing:

### 1. Input Processing Nodes

**Split** - Divide documents into smaller pieces

Use when:
- Documents are too large for LLM context windows
- You want to process text in manageable chunks
- You need granular analysis (sentence-level, paragraph-level)

```yaml
- name: chunks
  type: Split
  chunk_size: 30000
  split_unit: characters  # or sentences, paragraphs
```

### 2. Transformation Nodes

**Map** - Apply operation to each item independently in parallel

Use when:
- Processing each item separately (no cross-item information needed)
- Running the same prompt on multiple chunks
- Maximum parallelization is desired

```yaml
- name: code_chunks
  type: Map
  inputs:
    - chunks
```

**Transform** - Apply operation to single aggregated input

Use when:
- Consolidating multiple results into one output
- Generating summaries or final reports
- Processing needs context from all inputs

```yaml
- name: final_codes
  type: Transform
  inputs:
    - all_codes
```

**TransformReduce** - Reduce then transform in one step

Use when:
- You need both reduction and transformation
- Want to avoid intermediate node

```yaml
- name: consolidated
  type: TransformReduce
  inputs:
    - chunk_results
```

### 3. Aggregation Nodes

**Reduce** - Collect and concatenate results from multiple items

Use when:
- Gathering all outputs into single text
- Preparing for consolidation step
- Simple aggregation without LLM processing

```yaml
- name: all_codes
  type: Reduce
  inputs:
    - chunk_codes
```

### 4. Structuring Nodes

**Batch** - Group items by criteria

Use when:
- Processing items as groups
- Organizing by document, category, or metadata
- Creating hierarchical structure

```yaml
- name: by_document
  type: Batch
  batch_by: doc_index
  inputs:
    - chunks
```

**GroupBy** - Group items by field values

Use when:
- Creating multiple groups from single input
- Organizing by multi-field criteria
- Building nested batch structure

```yaml
- name: by_category
  type: GroupBy
  group_by:
    - category
    - subcategory
  inputs:
    - classified_items
```

**Ungroup** - Flatten all batch nesting

Use when:
- Converting BatchList back to flat list
- Removing all grouping structure

```yaml
- name: flattened
  type: Ungroup
  inputs:
    - grouped_items
```

### 5. Analysis Nodes

**Classifier** - Extract structured categorical data

Use when:
- Assigning categories or labels
- Extracting ratings or scores
- Running multi-model agreement analysis

```yaml
- name: classify
  type: Classifier
  model_names:
    - gpt-4o-mini
    - gpt-4o
  agreement_fields:
    - topic
  inputs:
    - documents
```

**VerifyQuotes** - Validate quotes against sources

Use when:
- Checking quote accuracy in qualitative analysis
- Ensuring LLM used verbatim quotes
- Identifying paraphrasing or hallucinations

```yaml
- name: checkquotes
  type: VerifyQuotes
  inputs:
    - codes
```

### 6. Filtering Nodes

**Filter** - Keep/remove items based on conditions

Use when:
- Removing irrelevant items
- Selecting subsets for further processing
- Implementing quality checks

```yaml
- name: relevant_only
  type: Filter
  inputs:
    - classified
    - relevance_check
```

## Choosing the Right Node Type

### Question: How many inputs, how many outputs?

**Many inputs → Many outputs**: Use **Map**
- Example: Code each chunk independently

**Many inputs → One output**: Use **Reduce** or **Transform**
- Example: Collect all codes into final codebook

**One input → Many outputs**: Use **Split**
- Example: Break document into paragraphs

**One input → One output**: Use **Transform**
- Example: Generate narrative report

### Question: Do I need an LLM?

**Yes**: Use **Map**, **Transform**, **TransformReduce**, or **Classifier**
- These nodes have templates and call LLMs

**No**: Use **Split**, **Reduce**, **Batch**, **GroupBy**, **Ungroup**, or **Filter**
- These nodes do structural operations only

### Question: Do items need context from other items?

**No** (independent): Use **Map**
- Faster due to parallelization
- Each item processed separately

**Yes** (dependent): Use **Transform**
- All inputs combined before processing
- Slower but enables cross-referencing

### Question: Am I organizing or analyzing?

**Organizing** data structure:
- **Split**: Break apart
- **Batch**/**GroupBy**: Group together
- **Ungroup**: Flatten structure
- **Filter**: Remove items

**Analyzing** content:
- **Map**: Process items in parallel
- **Transform**: Consolidate/generate
- **Classifier**: Extract structured data
- **VerifyQuotes**: Validate quotes

## Common Node Patterns

### Pattern 1: Split-Map-Reduce-Transform

Classic qualitative analysis pattern:

```yaml
nodes:
  # Break documents into chunks
  - name: chunks
    type: Split
    chunk_size: 30000

  # Code each chunk independently
  - name: chunk_codes
    type: Map
    inputs:
      - chunks

  # Collect all codes
  - name: all_codes
    type: Reduce
    inputs:
      - chunk_codes

  # Consolidate into final codebook
  - name: final_codes
    type: Transform
    inputs:
      - all_codes
```

### Pattern 2: Batch-Map-Reduce

Process documents separately, then combine:

```yaml
nodes:
  # Group chunks by document
  - name: by_document
    type: Batch
    batch_by: doc_index
    inputs:
      - chunks

  # Code within each document
  - name: document_codes
    type: Map
    inputs:
      - by_document

  # Aggregate across documents
  - name: all_codes
    type: Reduce
    inputs:
      - document_codes
```

### Pattern 3: Classify-GroupBy-Transform

Categorize then process by category:

```yaml
nodes:
  # Classify items
  - name: classified
    type: Classifier
    inputs:
      - items

  # Group by classification
  - name: by_category
    type: GroupBy
    group_by:
      - category
    inputs:
      - classified

  # Analyze each category
  - name: category_analysis
    type: Map
    inputs:
      - by_category
```

### Pattern 4: Map-Filter-Transform

Generate candidates, filter, consolidate:

```yaml
nodes:
  # Generate relevance checks
  - name: relevance
    type: Map
    inputs:
      - chunks

  # Keep only relevant items
  - name: relevant_chunks
    type: Filter
    inputs:
      - chunks
      - relevance

  # Process filtered items
  - name: analysis
    type: Transform
    inputs:
      - relevant_chunks
```

## Node Execution Behavior

### Parallelization

**Parallel execution** (multiple items at once):
- Map
- Classifier (within model)

**Sequential execution** (one item/batch at a time):
- Transform
- Reduce
- Split
- Filter
- VerifyQuotes

**Batch-level parallelization** (independent batches in parallel):
- All nodes respect DAG dependency batching

### Memory Considerations

**Low memory** (streaming):
- Reduce (concatenates text incrementally)
- Filter (drops items as processed)

**High memory** (accumulates):
- Transform (loads all inputs)
- Map (stores all results)
- Classifier (especially multi-model)

**Controlled memory**:
- Split (processes one document at a time)
- Batch (groups items but doesn't duplicate)

## Extending with Custom Nodes

All nodes inherit from base classes:

```python
from soak.models.nodes.base import CompletionDAGNode, ItemsNode

class MyCustomNode(ItemsNode, CompletionDAGNode):
    """Custom node with LLM completion."""

    type: Literal["MyCustomNode"] = "MyCustomNode"

    async def run(self):
        items = await self.get_items()
        # Custom processing logic
        return results
```

## Node Type Reference Table

| Node Type | Inputs | Outputs | Uses LLM | Parallelizes | Use Case |
|-----------|--------|---------|----------|--------------|----------|
| Split | 1 | Many | No | No | Break documents into chunks |
| Map | Many | Many | Yes | Yes | Process items independently |
| Reduce | Many | 1 | No | No | Collect results into text |
| Transform | 1+ | 1 | Yes | No | Consolidate and generate |
| TransformReduce | Many | 1 | Yes | No | Reduce + transform combined |
| Batch | Many | Many | No | No | Group by metadata field |
| GroupBy | Many | Many | No | No | Group by multiple fields |
| Ungroup | Many | Many | No | No | Flatten batch structure |
| Classifier | Many | Many | Yes | Yes | Structured classification |
| Filter | Many | Many | No | No | Remove items by condition |
| VerifyQuotes | 1 | 1 | No | Yes | Validate quotes vs sources |

## Next Steps

- [Node Reference](../reference/node-reference.md) - Detailed node parameters
- [DAG Architecture](dag-architecture.md) - How nodes execute in pipeline
- [Template System](template-system.md) - How nodes use templates
