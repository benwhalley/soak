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
uv run soak zs data/interviews.txt --output results
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
- [Working with Spreadsheet Data](how-to/working-with-spreadsheet-data.md) - Process CSV and Excel files
- [Pre-extraction Workflow](how-to/pre-extract-workflow.md) - Filter text before analysis
- [Build a Classifier](how-to/build-classifier.md) - Structured classification pipelines

### Explanation (Understanding-oriented)

Conceptual background:

- [What is soak?](explanation/what-is-soak.md) - Purpose and design philosophy
- [DAG Architecture](explanation/dag-architecture.md) - Why pipelines, execution model
- [Node Types](explanation/node-types.md) - Understanding different processing nodes
- [Model Aliases](explanation/model-aliases.md) - Configuring models for different pipeline roles
- [Template System](explanation/template-system.md) - Jinja2 and struckdown syntax
- [Quote Verification Approach](explanation/quote-verification-approach.md) - Design and rationale for quote validation

### Reference (Information-oriented)

Technical specifications:

- [CLI Reference](reference/cli.md) - Command-line interface
- [Node Reference](reference/node-reference.md) - All node types and parameters
- [Quote Verification Algorithm](reference/quote-verification.md) - Technical specification of quote validation

## Common Use Cases

### Thematic Analysis

Analyze interview transcripts, survey responses, or other qualitative data:

```bash
uv run soak zs data/*.txt --output analysis
```

See [Thematic Analysis](how-to/thematic-analysis.md)

### Classification

Extract structured data from text:

```bash
uv run soak classifier data/*.docx --output results
```

See [Build a Classifier](how-to/build-classifier.md)

### Spreadsheet Analysis

Process CSV or Excel files (each row becomes a document):

```bash
uv run soak classifier_tabular data/survey.csv --output results
```

See [Working with Spreadsheet Data](how-to/working-with-spreadsheet-data.md)

## Support

- GitHub: https://github.com/benwhalley/soak
- Issues: https://github.com/benwhalley/soak/issues
