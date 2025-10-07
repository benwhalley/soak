# soak-llm Documentation

**soak** is a Python package for LLM-assisted qualitative text analysis. It uses DAG-based pipelines to perform thematic analysis, classification, and other text processing tasks.

## Quick Start

```bash
# Install
uv pip install -e .

# Set credentials
export LLM_API_KEY=your_api_key
export LLM_API_BASE=https://api.openai.com/v1

# Run thematic analysis
uv run soak run zs data/interviews.txt --output results
open results.html
```

## Documentation Structure

### Tutorials (Learning-oriented)

Start here if you're new to soak:

- [Getting Started](tutorials/getting-started.md) - Installation and your first analysis
- [Customizing Your Analysis](tutorials/customizing-analysis.md) - Adapting prompts to your research
- [Working with Results](tutorials/working-with-results.md) - Understanding codes, themes, and exports

### How-to Guides (Goal-oriented)

Complete workflows for specific tasks:

- [Thematic Analysis](how-to/thematic-analysis.md) - Inductive coding and theme generation
- [Pre-extraction Workflow](how-to/pre-extract-workflow.md) - Filter text before analysis
- [Build a Classifier](how-to/build-classifier.md) - Structured classification pipelines
- [Configure Text Splitting](how-to/configure-splits.md) - Chunking strategies
- [Track Provenance](how-to/track-provenance.md) - Source tracking through the pipeline
- [Multi-model Agreement](how-to/multi-model-agreement.md) - Inter-rater reliability

### Explanation (Understanding-oriented)

Conceptual background:

- [What is soak?](explanation/what-is-soak.md) - Purpose and design philosophy
- [DAG Architecture](explanation/dag-architecture.md) - Why pipelines, execution model
- [Node Types](explanation/node-types.md) - Understanding different processing nodes
- [Template System](explanation/template-system.md) - Jinja2 and struckdown syntax
- [Provenance Tracking](explanation/provenance-tracking.md) - How lineage works
- [Quote Verification Approach](explanation/quote-verification-approach.md) - Design and rationale for quote validation

### Reference (Information-oriented)

Technical specifications:

- [CLI Reference](reference/cli.md) - Command-line interface
- [Node Reference](reference/node-reference.md) - All node types and parameters
- [Pipeline Format](reference/pipeline-format.md) - YAML specification
- [Configuration](reference/configuration.md) - Environment variables
- [Quote Verification Algorithm](reference/quote-verification.md) - Technical specification of quote validation

## Common Use Cases

### Thematic Analysis

Analyze interview transcripts, survey responses, or other qualitative data:

```bash
uv run soak run zs data/*.txt --output analysis
```

See [Thematic Analysis](how-to/thematic-analysis.md)

### Classification

Extract structured data from text:

```bash
uv run soak run classifier data/*.docx --output results
```

See [Build a Classifier](how-to/build-classifier.md)

## Support

- GitHub: https://github.com/benwhalley/soak-package
- Issues: https://github.com/benwhalley/soak-package/issues
