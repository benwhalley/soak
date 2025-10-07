# soak-llm

DAG-based pipelines for LLM-assisted qualitative text analysis.

## Quick Start

```bash
# Install
uv pip install -e .

# Set credentials
export LLM_API_KEY=your_api_key
export LLM_API_BASE=https://api.openai.com/v1

# Run analysis
uv run soak run demo data/5LC.docx --output results
open results.html
```

## Installation

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation)

```bash
git clone https://github.com/benwhalley/soak-package
cd soak-package
uv pip install -e .
```

## Usage

```bash
uv run soak run <pipeline> <files> --output <name>

# Examples
uv run soak run demo data/*.txt --output analysis
uv run soak run zspe data/interviews.docx -t short --output results
```

**Options:**
- `--output, -o`: Output filename (generates .json and .html)
- `--model-name`: LLM model (default: gpt-4o-mini)
- `--format, -f`: Output format when printing to stdout
- `-c, --context`: Pipeline context variables (e.g., `-c topic="Healthcare"`)

## Pipelines

Built-in pipelines in `soak/pipelines/`:
- `demo`: Basic thematic analysis
- `zspe`: Zero-shot with pre-extraction filtering
- Custom pipelines: Use local YAML file path

## Environment Variables

```bash
export LLM_API_KEY=sk-...           # Required
export LLM_API_BASE=https://...     # Optional (OpenAI-compatible endpoint)
export MAX_CONCURRENCY=20           # Optional (concurrent LLM calls)
```

## Documentation

See [CLAUDE.md](CLAUDE.md) for architecture details.

## License

MIT
