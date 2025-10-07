# CLAUDE.md

Guide for Claude Code working with this repository.

## Project

**soak-llm**: DAG-based pipeline system for LLM-assisted qualitative text analysis.

## Environment

Package manager: **uv**

Required environment variables:
- `LLM_API_KEY`: API key
- `LLM_API_BASE`: OpenAI-compatible endpoint (optional)

Install:
```bash
uv pip install -e .
```

## Commands

**Run pipeline:**
```bash
uv run soak run <pipeline_name> <input_files> [options]
# Example:
uv run soak run demo data/*.txt --output results
```

**Options:**
- `--output, -o`: Output filename (generates .json and .html)
- `--format, -f`: Output format for stdout (json/html)
- `--model-name`: LLM model (default: gpt-4o-mini)
- `--include-documents`: Include source docs in output

**Build:**
```bash
rm -rf dist build *.egg-info
python -m build
twine check dist/*
```

## Architecture

### Pipeline System

**Definition** (`soak/specs.py`):
YAML format with embedded Jinja2 templates:
```yaml
name: pipeline_name
default_context:
  # context variables
nodes:
  - name: node_name
    type: NodeType
    inputs: [dependencies]
---#node_name
Template with {{variables}}
```

**Execution** (`soak/models.py`):
- `DAG` class orchestrates node execution via anyio
- Parallel execution in batches determined by dependencies
- Max runtime: 30 mins (configurable via `SOAK_MAX_RUNTIME`)

**Node Types** (`soak/models.py`):
- `Split`: Chunk documents
- `Map`: Apply LLM to each item in parallel
- `Reduce`: Aggregate inputs
- `Transform`: Single-item LLM transformation
- `TransformReduce`: Combined transform + reduce
- `Batch`: Group items
- `VerifyQuotes`: Validate quotes against sources (BM25 + embeddings)
- `Classifier`: Multi-model classification with agreement metrics
- `Filter`: Boolean filtering with LLM

**Templates**:
- Jinja2 with `StrictUndefined`
- `[[return_type]]` syntax for structured outputs (via struckdown)
- Return types: `theme`, `code`, `themes`, `codes` (see `get_action_lookup()`)

**Document Processing** (`soak/document_utils.py`):
- Formats: PDF, DOCX, TXT, ZIP
- Auto-detection via python-magic
- Caching based on mtime

### Data Models

Core models (`soak/models.py`):
- `Code`: Qualitative code with name, description, quotes
- `Theme`: Groups codes
- `QualitativeAnalysis`: Complete analysis (codes, themes, narrative)
- `DAGConfig`: Pipeline configuration
- `TrackedItem`: Document with provenance (source_id, metadata, lineage)

### Concurrency

- `anyio` for structured concurrency
- Global semaphore: `MAX_CONCURRENCY` (default: 20)
- Nodes in same batch run concurrently

### Caching

- Embeddings: `.embeddings/` (joblib)
- Document extraction: mtime-based

### CLI

Entry: `soak/cli.py` (Typer)
- Pipeline resolution: local dir → `soak/pipelines/`
- Trogon TUI support
- Output: JSON (model_dump_json), HTML (custom template)

### Comparison System

`soak/comparators/`:
- `SimilarityComparator`: Compare analyses via embeddings
- Heatmaps and network plots
- Evaluates consistency across runs

## Key Implementation Notes

**Quote Verification** (models.py:2273-2876):
- BM25-first lexical search + embedding verification
- Handles ellipses by matching head/tail separately
- Adaptive windowing (1.1× longest quote, 30% overlap)
- Outputs: BM25 score, cosine similarity, source positions

**Provenance Tracking** (`TrackedItem`):
- `source_id`: Hierarchical ID (e.g., "docA__split__0")
- `metadata`: Document properties, indices
- Preserved through all transformations

**Agreement Calculation** (`Classifier` node):
- Multi-model support
- Metrics: Krippendorff's alpha, Gwet's AC1, percent agreement
- CSV/HTML export per model + combined statistics
