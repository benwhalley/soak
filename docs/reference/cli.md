# CLI Reference

soak provides a command-line interface for running pipelines and working with results.

## Commands

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
| `--model MODEL` | `-m` | Override LLM model (default: from pipeline or `gpt-4o-mini`) |
| `--output PATH` | `-o` | Output file path without extension (default: derived from pipeline name) |
| `--template NAME` | `-t` | Template name in `soak/templates/` or path to custom template (default: `pipeline.html`, can be used multiple times) |
| `--include-documents` | | Include original document text in JSON output |
| `--context KEY=VALUE` | `-c` | Override context variables (can be used multiple times) |
| `--force` | `-f` | Overwrite existing output files/folders (checked before pipeline runs) |
| `--sample N` | `-S` | Randomly sample N rows/documents from input |
| `--head N` | `-H` | Take first N rows/documents from input |
| `--seed N` | | Random seed for reproducible outputs (overrides pipeline config) |
| `--progress/--no-progress` | | Show progress bars (auto-detected: enabled for TTY, disabled with -vv) |
| `--verbose` | `-v` | Increase verbosity (`-v` = INFO, `-vv` = DEBUG) |

**Examples:**

```bash
# Basic usage (creates zs.json and zs_pipeline.html)
uv run soak zs data/interview.txt

# Specify custom output name
uv run soak zs --output results data/interview.txt

# Multiple files
uv run soak zs --output analysis data/*.txt

# Custom model
uv run soak zs --output results --model openai/gpt-4o data/*.txt

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

Output files are always written to disk (never to stdout):

Without `--output` (pipeline `zs.soak`):
- `zs.json` - Full pipeline data
- `zs_pipeline.html` - Rendered view (default template)
- `zs_dump/` - Detailed execution folder with node outputs

With `--output results`:
- `results.json` - Full pipeline data
- `results_pipeline.html` - Rendered view (default template)
- Additional HTML files if multiple `-t` options specified (e.g., `results_simple.html`)
- `results_dump/` - Detailed execution folder

**Conflict handling:**
- Before running the pipeline, soak checks if any output files/folders already exist
- If conflicts found without `--force`: exits with error (prevents wasted pipeline execution)
- If conflicts found with `--force`: warns and overwrites all conflicting files/folders

### compare

Compare multiple analysis results.

```bash
uv run soak compare [OPTIONS] INPUT_FILES...
```

**Arguments:**

- `INPUT_FILES` - Two or more JSON files containing analysis results

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output PATH` | `-o` | Output HTML file (default: `comparison.html`) |
| `--threshold FLOAT` | | Similarity threshold for matching themes (default: 0.6) |
| `--method METHOD` | | Dimensionality reduction: `umap`, `mds`, `pca` (default: `umap`) |
| `--label TEMPLATE` | `-l` | Format string for labels: `{name}`, `{description}` (default: `{name}`) |
| `--embedding-template TEMPLATE` | `-e` | Format string for embeddings (default: `{name}`) |

**Examples:**

```bash
# Compare two analyses
uv run soak compare results2.json -o comparison.html results1.json

# Compare with custom similarity threshold
uv run soak compare run2.json run3.json --threshold 0.7 run1.json

# Use different visualization method
uv run soak compare --method pca -o comparison.html *.json

# Custom label template
uv run soak compare -l "{name}: {description}" -o comparison.html *.json
```

**Output:**

- HTML report with similarity heatmaps, network plots, and statistics
- Inter-rater agreement metrics (Gwet's AC1, Krippendorff's Alpha)

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
- `classifier` - Classification workflow
- `demo` - Simple demonstration

**Built-in Templates:**

- `default.html` - Standard results view

### dump

Export detailed DAG execution to folder structure.

```bash
uv run soak dump [OPTIONS] INPUT_JSON
```

**Arguments:**

- `INPUT_JSON` - Path to JSON file from previous run

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output-folder PATH` | `-o` | Output folder (default: `<input_stem>_dump`) |
| `--template NAME` | `-t` | Generate HTML files in dump using template(s) (can be used multiple times) |
| `--force` | `-f` | Overwrite existing folder |

**Examples:**

```bash
# Dump execution details
uv run soak dump

# Custom output folder results.json
uv run soak dump results.json -o detailed_analysis

# Overwrite existing dump
uv run soak dump --force results.json

# Dump with HTML generation
uv run soak dump results.json -t pipeline.html
```

**Output Structure:**

```
results_dump/
├── 01_Split_chunks/
│   ├── inputs/           # Original documents
│   ├── outputs/          # Generated chunks
│   └── meta.txt          # Node configuration
├── 02_Map_codes/
│   ├── inputs/           # Chunks processed
│   ├── 0000_*.json       # Full response for each chunk
│   └── ...
└── metadata.json         # Command and configuration
```

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

**Reuse saved results:**

```bash
# Run once
uv run soak zs -o results data/*.txt

# Inspect detailed execution
uv run soak dump results.json

# Generate HTML from dump with custom template
uv run soak dump results.json -t my_template.html
```

**Process large datasets:**

```bash
# Reduce concurrency to avoid rate limits
export MAX_CONCURRENCY=5
uv run soak zs large_dataset/*.txt -o results
```
