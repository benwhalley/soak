# CLI Reference

soak provides a command-line interface for running pipelines and working with results.

## Global Options

These options apply to all commands:

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Increase verbosity (`-v` = INFO, `-vv` = DEBUG) |
| `--install-completion` | | Install shell completion for the current shell |
| `--show-completion` | | Show completion script for the current shell |
| `--help` | | Show help message and exit |

## Commands

| Command | Description |
|---------|-------------|
| `run` | Run a pipeline on input files |
| `compare` | Compare analyses or string lists and generate comparison statistics |
| `show` | Show the contents of a built-in pipeline or template |
| `coverage` | Analyse how well themes from an analysis are represented across documents |

### run

Run a pipeline on input files.

```bash
uv run soak PIPELINE INPUT_FILES [OPTIONS]
```

**Arguments:**

- `PIPELINE` - Pipeline name (e.g., `zs`, `classifier`) or path to YAML file
- `INPUT_FILES` - One or more file paths or glob patterns (e.g., `data/*.txt`, `interviews.zip`)

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--model MODEL` | `-m` | Model configuration (see [Model Aliases](#model-aliases) below) |
| `--output PATH` | `-o` | Output file path without extension (default: derived from pipeline name) |
| `--template NAME` | `-t` | Template name in `soak/templates/` or path to custom template (default: `pipeline.html`, can be used multiple times) |
| `--include-documents` | | Include original document text in JSON output |
| `--context KEY=VALUE` | `-c` | Override context variables (can be used multiple times) |
| `--force` | `-f` | Overwrite existing output files/folders (checked before pipeline runs) |
| `--sample N` | `-S` | Randomly sample N rows/documents from input (mutually exclusive with `--head`) |
| `--head N` | `-H` | Take first N rows/documents from input (mutually exclusive with `--sample`) |
| `--seed N` | | Random seed for reproducible outputs and document shuffling (default: 42) |
| `--progress/--no-progress` | | Show progress bars (auto-detected: enabled for TTY, disabled with -vv) |
| `--timeout N` | | Timeout in seconds for individual LLM API calls (default: 90) |
| `--skip-node NAME` | | Skip specified node(s) during execution (can be used multiple times) |
| `--stop-at NAME` | | Stop execution before the specified node runs |

**Examples:**

```bash
# Basic usage (creates zs.json and zs_pipeline.html)
uv run soak zs data/interview.txt

# Specify custom output name
uv run soak zs --output results data/interview.txt

# Multiple files
uv run soak zs --output analysis data/*.txt

# Set default model
uv run soak zs --output results --model gpt-4.1 data/*.txt

# Override specific model aliases
uv run soak zs --output results --model default=gpt-4.1-mini --model best=gpt-4.1 data/*.txt

# Override context variables
uv run soak zspe -o results data/*.txt \
  -c research_question="What are recovery experiences?" \
  -c excerpt_topics="Exercise and rehabilitation"

# Use custom template
uv run soak zs -o results -t my_template.html data/*.txt

# Use multiple templates (creates results_pipeline.html and results_simple.html)
uv run soak zs -o results -t pipeline.html -t simple.html data/*.txt

# Process ZIP archive
uv run soak zs -o results interviews.zip

# Process CSV spreadsheet (each row becomes a document)
uv run soak classifier_tabular -o results soak/data/test_data.csv

# Sample first 10 rows from spreadsheet
uv run soak classifier_tabular --head 10 -o results data/survey.xlsx
```

**Pipeline Resolution:**

soak looks for pipeline files in this order:

1. `./PIPELINE` (exact path)
2. `./PIPELINE.soak`
3. `./PIPELINE.yml`
4. `soak/pipelines/PIPELINE`
5. `soak/pipelines/PIPELINE.soak`
6. `soak/pipelines/PIPELINE.yml`

**Input Files:**

- Supports: `.txt`, `.pdf`, `.docx`, `.csv`, `.xlsx`, `.zip`
- Glob patterns: `data/*.txt`, `**/*.docx`
- CSV/XLSX: Each row becomes a separate document with columns accessible as `{{column_name}}`
- ZIP files: Automatically extracted to temp directory
- Multiple files processed in parallel

**Output:**

Output files are always written to disk (never to stdout). All files go into a dump folder:

Without `--output` (pipeline `zs.soak`):
```
zs_dump/
├── zs.json              # Full pipeline data
├── zs_pipeline.html     # Rendered view (default template)
├── 01_Split_chunks/     # Per-node execution details
├── 02_Map_codes/
└── ...
```

With `--output results`:
```
results_dump/
├── results.json
├── results_pipeline.html
├── results_simple.html  # If -t simple specified
└── ...
```

**Conflict handling:**
- Before running the pipeline, soak checks if the output folder already exists
- If output folder exists with JSON but no `--force`: generates only new templates (template-only mode)
- If output folder exists with `--force`: warns and overwrites everything
- Template-only mode allows adding new templates without re-running the pipeline

#### Model Aliases

Pipelines can define model aliases in `default_config.models` to assign different models to different roles. The `--model` flag overrides these aliases at runtime.

**Syntax:**

| Form | Effect |
|------|--------|
| `--model gpt-4.1` | Sets the `default` alias |
| `--model alias=model` | Sets a specific alias |

**Example pipeline definition:**

```yaml
default_config:
  models:
    default: gpt-4.1-mini    # routine tasks
    best: gpt-4.1            # complex reasoning
  embeddings: text-embedding-3-large

nodes:
  - name: extract_codes
    type: Map
    model: default           # uses 'default' alias

  - name: synthesize
    type: Transform
    model: best              # uses 'best' alias
```

**Override from CLI:**

```bash
# Set default model only
uv run soak zs data/*.txt --model gpt-4.1

# Override specific aliases
uv run soak zs data/*.txt --model default=claude-3-5-sonnet --model best=claude-3-opus

# Multiple aliases
uv run soak zs data/*.txt \
  --model default=gpt-4.1-mini \
  --model best=gpt-4.1 \
  --model cheap=gpt-4.1-mini
```

See [Model Aliases](../explanation/model-aliases.md) for full documentation.

### compare

Compare analyses or string lists and generate comparison statistics.

Two modes:
1. **JSON mode**: Compare QualitativeAnalysis JSON files from pipeline runs
2. **Strings mode**: Compare columns of strings from XLSX/CSV files (useful for comparing theme lists)

```bash
# JSON mode
uv run soak compare results1.json results2.json -o comparison.html

# Strings mode
uv run soak compare --strings themes.xlsx --cols "A,B,C" -o comparison.html
```

**Key Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output PATH` | `-o` | Output file (.html for report, .txt for text stats) |
| `--strings PATH` | `-s` | XLSX/CSV file with columns to compare (enables strings mode) |
| `--cols NAMES` | `-c` | Comma-separated column names (default: `A,B`). All pairwise combinations computed. |
| `--threshold FLOAT` | | Similarity threshold for binary matching (default: 0.6) |
| `--similarity METRIC` | `-S` | Similarity metric: `angular` (default), `cosine`, `shepard` |
| `--ot-k FLOAT` | | Optimal transport mass penalty K (default: 0.25). Lower = more selective. |
| `--embedding-model` | | Embedding model (default: `text-embedding-3-large`). Use `local/model-name` for local models. |

**Examples:**

```bash
# Compare two pipeline results
uv run soak compare results1.json results2.json -o comparison.html

# Compare three sets of themes from spreadsheet columns
uv run soak compare --strings themes.xlsx --cols "Method1,Method2,Method3" -o comparison.html

# Use local embeddings (no API key needed)
uv run soak compare --embedding-model local/all-MiniLM-L6-v2 *.json

# Text output only (no HTML)
uv run soak compare --strings data.xlsx -o stats.txt
```

**Output:**

- HTML report with similarity heatmaps, optimal transport visualisations, and matching statistics
- Console output shows key metrics (suppressed when outputting to HTML)

### show

Display contents of built-in pipelines or templates.

```bash
uv run soak show TYPE [NAME]
```

**Arguments:**

- `TYPE` - `pipeline` or `template`
- `NAME` - Name of item to show (optional - lists all if omitted)

**Examples:**

```bash
# List all pipelines
uv run soak show pipeline

# List all templates
uv run soak show template

# Show a specific pipeline
uv run soak show pipeline zs

# Show a specific template
uv run soak show template default

# Save to file for customization
uv run soak show pipeline zs > my_analysis.soak
uv run soak show template default > my_template.html
```

**Built-in Pipelines:**

- `zs` - Zero-shot thematic analysis
- `zspe` - Pre-extraction thematic analysis
- `classifier` - Multi-level classification with agreement analysis
- `classifier_tabular` - Classification for spreadsheet data (CSV/Excel)
- `demo` - Simple demonstration pipeline
- `verify` - Quote verification workflow
- `test` - Testing pipeline

**Built-in Templates:**

- `pipeline.html` - Standard pipeline results view (default)
- `simple.html` - Simplified results view
- `narrative.html` - Narrative-focused output
- `comparison.html` - Multi-run comparison view

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for LLM provider | Required |
| `LLM_API_BASE` | Base URL for API | `https://api.openai.com/v1` |
| `MAX_CONCURRENCY` | Max parallel LLM calls | `20` |
| `SOAK_MAX_RUNTIME` | Max pipeline runtime (seconds) | `1800` (30 minutes) |

Set via:

```bash
export LLM_API_KEY=sk-...
export LLM_API_BASE=https://api.openai.com/v1
```

Or create `.env` file in working directory:

```
LLM_API_KEY=sk-...
LLM_API_BASE=https://api.openai.com/v1
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (missing files, invalid arguments) |
| 2 | Pipeline validation error |

## Tips

**View progress:**

```bash
uv run soak zs -o results -v data/*.txt
```

**Process results with jq:**

```bash
uv run soak zs data/test.txt  # Creates zs.json
cat zs.json | jq '.codes'
```

**Iterate on templates:**

```bash
# Get default template
uv run soak show template default > my_template.html

# Edit my_template.html

# Use it
uv run soak zs data/*.txt -o results -t my_template.html
```

**Add new templates to existing results:**

```bash
# Run once
uv run soak zs -o results data/*.txt

# Add another template without re-running (template-only mode)
uv run soak zs -o results -t my_template.html data/*.txt
```

**Process large datasets:**

```bash
# Reduce concurrency to avoid rate limits
export MAX_CONCURRENCY=5
uv run soak zs large_dataset/*.txt -o results
```
