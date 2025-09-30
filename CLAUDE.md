# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**soak-llm** is a Python package for LLM-based textual analysis and qualitative research. It implements a DAG (Directed Acyclic Graph) based pipeline system for processing text documents through various analysis stages using language models.

## Environment Setup

This project uses **uv** as the package manager. Environment variables required:
- `LLM_API_KEY`: API key for LLM provider
- `LLM_BASE_URL`: OpenAI-compatible endpoint URL (optional)

Install the package in development mode:
```bash
uv pip install -e .
```

## Common Commands

**Run a pipeline:**
```bash
uv run soak run <pipeline_name> <input_files> [options]
# Example:
uv run soak run demo data/*.txt --output results
```

**Command options:**
- `--output, -o`: Output file path (without extension; generates both .json and .html)
- `--format, -f`: Output format (json or html) when writing to stdout
- `--model-name`: LLM model to use (default: litellm/gpt-4.1-mini)
- `--include-documents`: Include original documents in output

**Build package:**
```bash
rm -rf dist build *.egg-info
python -m build
twine check dist/*
```

## Architecture

### Pipeline System

The core architecture is a **DAG-based execution system** for text analysis:

1. **Pipeline Definition** (`soak/specs.py`): Pipelines are defined in YAML files with embedded Jinja2 templates. Format:
   ```yaml
   name: pipeline_name
   default_context:
     # context variables for templates
   nodes:
     - name: node_name
       type: NodeType
       inputs: [dependency_nodes]
   ---#node_name
   Template content with {{variables}}
   ```

2. **DAG Execution** (`soak/models.py`):
   - `DAG` class orchestrates node execution using anyio for structured concurrency
   - Nodes execute in batches determined by dependency resolution
   - Uses `get_execution_order()` to calculate parallel execution batches
   - Max runtime: 3 hours (configurable via `SOAK_MAX_RUNTIME`)

3. **Node Types** (all in `soak/models.py`):
   - `Split`: Chunks input documents into smaller pieces
   - `Map`: Applies LLM completion to each item in parallel
   - `Reduce`: Aggregates multiple inputs into one
   - `Transform`: Transforms input through LLM completion
   - `TransformReduce`: Combined transform and reduce operation
   - `Batch`: Groups items for batch processing
   - `VerifyQuotes`: Validates quotes against source documents

4. **Template System**:
   - Uses Jinja2 with `StrictUndefined` mode
   - Templates use special `[[return_type]]` syntax for structured outputs (handled by `struckdown` library)
   - Return types map to Pydantic models: `theme`, `code`, `themes`, `codes` (defined in `get_action_lookup()`)

5. **Document Processing** (`soak/document_utils.py`):
   - Supports: PDF, Word (.docx), plain text, zip archives
   - Auto-detects file types using `python-magic`
   - Extracts and caches text from documents
   - Safe zip extraction with path traversal protection

### Data Models

Core models (`soak/models.py`):
- `Code`: Represents a qualitative code with name, description, and example quotes
- `Theme`: Groups codes with name, description, and code references
- `QualitativeAnalysis`: Complete analysis result with codes, themes, narrative
- `DAGConfig`: Pipeline configuration including documents, model settings, chunking
- `Document`: Source document representation

### Concurrency

- Uses `anyio` for structured concurrency
- Global semaphore limits concurrent LLM calls: `MAX_CONCURRENCY` (default: 20, set via env var)
- Nodes in the same execution batch run concurrently within task groups

### Caching

- Embedding cache: `.embeddings/` directory (uses joblib)
- Document text extraction cached based on file mtime

### CLI Architecture

Entry point: `soak/cli.py` → `app` (Typer instance)
- Pipeline resolution checks: local dir first, then `soak/pipelines/`
- Uses Trogon for TUI support
- Output formats: JSON (model_dump_json), HTML (custom template in `soak/templates/`)

### Comparison System

Location: `soak/comparators/`
- `SimilarityComparator`: Compares analysis results using embeddings
- Creates heatmaps and network plots for result comparison
- Used for evaluating consistency across multiple analysis runs