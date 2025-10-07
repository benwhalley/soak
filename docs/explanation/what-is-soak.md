# What is soak?

soak is a Python package for LLM-assisted qualitative text analysis. It automates coding, theme generation, and other text processing tasks while maintaining researcher control over the process.

## Purpose

Qualitative analysis—coding interviews, identifying themes, analyzing open-ended survey responses—is valuable but time-consuming. LLMs can assist, but using them effectively requires:

1. **Structured workflows**: Breaking analysis into clear stages
2. **Parallel processing**: Handling long documents by chunking
3. **Consolidation**: Merging results from parallel processes
4. **Provenance**: Tracking which text produced which codes
5. **Verification**: Checking quotes against sources

soak provides this infrastructure. You write analysis prompts; soak handles execution, data flow, and bookkeeping.

## What soak Does

### Thematic Analysis

Process interview transcripts or other qualitative data:

```bash
uv run soak run zs data/interviews/*.txt --output results
```

Produces:
- Codes with quotes from your data
- Themes grouping related codes
- Narrative report ready for publication
- Verified quotes with source tracking

### Classification

Extract structured data from text:

```bash
uv run soak run classifier data/documents/*.docx --output results
```

Produces:
- CSV with classifications for each document/chunk
- Summary statistics
- Source tracking showing which text was classified

### Custom Pipelines

Define your own analysis workflow:

```yaml
nodes:
  - name: summaries
    type: Map
    inputs: [documents]

  - name: comparison
    type: Transform
    inputs: [summaries]
```

## What soak Doesn't Do

**Not a black box**: You write the prompts. soak executes them but doesn't impose analysis methods.

**Not a replacement for expertise**: LLMs assist; researchers interpret. soak helps you work faster, not think less.

**Not a chatbot**: soak runs predefined pipelines. It's not interactive Q&A about your data.

## When to Use soak

Use soak when:

- You have qualitative text data (interviews, surveys, documents)
- You want systematic, reproducible analysis
- You need to process more data than manual coding allows
- You want to try multiple analysis approaches (different prompts, models)
- You need source tracking and quote verification

Don't use soak when:

- You have small datasets (< 5 documents) where manual coding is faster
- Your analysis requires deep contextual knowledge LLMs can't provide
- You need real-time, interactive exploration (use a chatbot instead)

## Design Philosophy

### DAG-based Pipelines

soak uses directed acyclic graphs (DAGs) to represent analysis workflows. Each node is a processing step; edges represent data dependencies.

Benefits:
- **Parallel execution**: Independent steps run concurrently
- **Reproducibility**: Same pipeline + data = same results
- **Modularity**: Reuse nodes across pipelines
- **Visibility**: See exactly what happens at each stage

See [DAG Architecture](dag-architecture.md) for details.

### Template-driven

Analysis logic lives in templates, not code:

```yaml
---#codes
Read this text and identify key themes:

{{input}}

Generate codes: [[codes:codes]]
```

Templates use Jinja2 for logic and struckdown for structured outputs. Non-programmers can write them.

See [Template System](template-system.md) for details.

### Provenance-first

Every piece of text tracks its origin:

```
interview_001.txt → chunks__0 → codes with source_id="chunks__0"
```

You can always trace results back to source text. Export formats include source tracking by default.

See [Provenance Tracking](provenance-tracking.md) for details.

## Common Workflows

### Inductive Thematic Analysis

1. Split documents into chunks
2. Code each chunk independently (Map)
3. Collect all codes (Reduce)
4. Consolidate duplicates (Transform)
5. Generate themes (Transform)
6. Verify quotes (VerifyQuotes)
7. Write narrative (Transform)

See [Thematic Analysis](../how-to/thematic-analysis.md)

### Classification with Agreement

1. Split documents if needed
2. Classify with multiple models (Classifier with model_names)
3. Calculate inter-rater agreement
4. Export results with source tracking

See [Multi-model Agreement](../how-to/multi-model-agreement.md)

### Pre-extraction Analysis

1. Extract relevant sections from long documents (Map)
2. Code the extracted sections (Map)
3. Continue as in standard thematic analysis

See [Pre-extraction Workflow](../how-to/pre-extract-workflow.md)

## Architecture Overview

```
User writes:           soak provides:
─────────────          ──────────────
Pipeline YAML    ───→  DAG executor
Templates        ───→  Jinja2 + struckdown rendering
Research Qs      ───→  Context injection

                       Document loading (PDF, DOCX, TXT)
                       Parallel processing
                       Rate limiting
                       Caching
                       Export (JSON, HTML, CSV)
                       Provenance tracking
```

## Example: From Pipeline to Results

**Pipeline** (`my_analysis.yaml`):

```yaml
nodes:
  - name: chunks
    type: Split
    chunk_size: 10000

  - name: codes
    type: Map
    inputs: [chunks]

---#codes
Generate codes from this text:
{{input}}

[[codes:codes]]
```

**Command**:

```bash
uv run soak run my_analysis.yaml data/interview.txt --output results
```

**What happens**:

1. Load `interview.txt`, create TrackedItem with source_id="interview"
2. Split into chunks: "interview__chunks__0", "interview__chunks__1", ...
3. Render template for each chunk with `{{input}}` = chunk content
4. Send prompts to LLM in parallel (respecting rate limits)
5. Parse responses into `Code` objects via struckdown
6. Export results with source tracking
7. Generate HTML view of all codes

**Result**:

```
results/
├── 01_Split_chunks/
│   ├── inputs/0000_interview.txt
│   └── outputs/0000_interview__chunks__0.txt
├── 02_Map_codes/
│   ├── inputs/0000_interview__chunks__0.txt
│   └── 0000_interview__chunks__0_response.json
└── results.html
```

## Next Steps

- [Getting Started](../tutorials/getting-started.md) - Run your first analysis
- [Thematic Analysis](../how-to/thematic-analysis.md) - Detailed workflow
- [Node Types](node-types.md) - Understanding processing nodes
- [DAG Architecture](dag-architecture.md) - How pipelines execute
