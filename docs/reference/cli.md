# CLI Reference

soak provides a command-line interface for running pipelines and working with results.

## Commands

### run

Run a pipeline on input files.

```bash
uv run soak run PIPELINE INPUT_FILES [OPTIONS]
```

**Arguments:**

- `PIPELINE` - Pipeline name (e.g., `zs`, `classifier`) or path to YAML file
- `INPUT_FILES` - One or more file paths or glob patterns (e.g., `data/*.txt`, `interviews.zip`)

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--model-name MODEL` | | Override LLM model (default: from pipeline or `litellm/gpt-4.1-mini`) |
| `--output PATH` | `-o` | Output file path without extension (generates `.json` and `.html`) |
| `--format FORMAT` | `-f` | Output format when writing to stdout: `json` or `html` (default: `json`) |
| `--template NAME` | `-t` | Template name in `soak/templates/` or path to custom template (default: `default.html`) |
| `--include-documents` | | Include original document text in JSON output |
| `--context KEY=VALUE` | `-c` | Override context variables (can be used multiple times) |
| `--dump` | `-d` | Export detailed execution to folder |
| `--dump-folder PATH` | | Custom dump folder path (default: `<output>_dump`) |
| `--force` | `-F` | Overwrite existing output/dump files |
| `--verbose` | `-v` | Increase verbosity (`-v` = INFO, `-vv` = DEBUG) |

**Examples:**

```bash
# Basic usage
uv run soak run zs data/interview.txt --output results

# Multiple files
uv run soak run zs data/*.txt --output analysis

# Custom model
uv run soak run zs data/*.txt --output results --model-name openai/gpt-4o

# Override context variables
uv run soak run zspe data/*.txt -o results \
  -c research_question="What are recovery experiences?" \
  -c excerpt_topics="Exercise and rehabilitation"

# Dump execution details
uv run soak run zs data/*.txt -o results --dump

# Use custom template
uv run soak run zs data/*.txt -o results -t my_template.html

# Write JSON to stdout
uv run soak run zs data/interview.txt -f json > output.json

# Process ZIP archive
uv run soak run zs interviews.zip -o results
```

**Pipeline Resolution:**

soak looks for pipeline files in this order:

1. `./PIPELINE` (exact path)
2. `./PIPELINE.yaml`
3. `./PIPELINE.yml`
4. `soak/pipelines/PIPELINE`
5. `soak/pipelines/PIPELINE.yaml`
6. `soak/pipelines/PIPELINE.yml`

**Input Files:**

- Supports: `.txt`, `.pdf`, `.docx`, `.zip`
- Glob patterns: `data/*.txt`, `**/*.docx`
- ZIP files: Automatically extracted to temp directory
- Multiple files processed in parallel

**Output:**

Without `--output`:
- Writes to stdout in format specified by `--format`

With `--output results`:
- `results.json` - Full pipeline data
- `results.html` - Rendered view (using template)

With `--dump`:
- Creates `results_dump/` directory with detailed node execution data

### compare

Compare multiple analysis results.

```bash
uv run soak compare INPUT_FILES... [OPTIONS]
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
uv run soak compare results1.json results2.json -o comparison.html

# Compare with custom similarity threshold
uv run soak compare run1.json run2.json run3.json --threshold 0.7

# Use different visualization method
uv run soak compare *.json --method pca -o comparison.html

# Custom label template
uv run soak compare *.json -l "{name}: {description}" -o comparison.html
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

# Show a pipeline
uv run soak show pipeline zs

# Show a template
uv run soak show template default

# Save to file for customization
uv run soak show pipeline zs > my_analysis.yaml
uv run soak show template default > my_template.html
```

**Built-in Pipelines:**

- `zs` - Zero-shot thematic analysis
- `zspe` - Pre-extraction thematic analysis
- `classifier` - Classification workflow
- `demo` - Simple demonstration

**Built-in Templates:**

- `default.html` - Standard results view

### export

Export saved JSON analysis to HTML.

```bash
uv run soak export INPUT_JSON [OPTIONS]
```

**Arguments:**

- `INPUT_JSON` - Path to JSON file from previous run

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output PATH` | `-o` | Output path without extension (default: same as input) |
| `--template NAME` | `-t` | Template name or path (default: `default.html`) |

**Examples:**

```bash
# Export with default template
uv run soak export results.json

# Export with custom template
uv run soak export results.json -t narrative.html

# Export to different file
uv run soak export results.json -o final_report
```

### dump

Export detailed DAG execution to folder structure.

```bash
uv run soak dump INPUT_JSON [OPTIONS]
```

**Arguments:**

- `INPUT_JSON` - Path to JSON file from previous run

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output-folder PATH` | `-o` | Output folder (default: `<input_stem>_dump`) |
| `--force` | `-F` | Overwrite existing folder |

**Examples:**

```bash
# Dump execution details
uv run soak dump results.json

# Custom output folder
uv run soak dump results.json -o detailed_analysis

# Overwrite existing dump
uv run soak dump results.json --force
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
| `SOAK_MAX_RUNTIME` | Max pipeline runtime (seconds) | `10800` (3 hours) |

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
uv run soak run zs data/*.txt -o results -v
```

**Test with stdout:**

```bash
uv run soak run zs data/test.txt -f json | jq '.codes'
```

**Iterate on templates:**

```bash
# Get default template
uv run soak show template default > my_template.html

# Edit my_template.html

# Use it
uv run soak run zs data/*.txt -o results -t my_template.html
```

**Reuse saved results:**

```bash
# Run once
uv run soak run zs data/*.txt -o results

# Re-export with different template
uv run soak export results.json -t my_template.html

# Inspect detailed execution
uv run soak dump results.json
```

**Process large datasets:**

```bash
# Reduce concurrency to avoid rate limits
export MAX_CONCURRENCY=5
uv run soak run zs large_dataset/*.txt -o results
```
