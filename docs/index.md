---
layout: default
title: Home
nav_order: 1
---




# soak


**DAG-based pipelines for LLM-assisted qualitative text analysis.**


![](images/u3311749543_httpss.mj.runKSun0h8R-vs_httpss.mj.rungggOgHBnZsM_8e94b8b6-f495-428b-bc3b-d4c70bdf18f9_3.png){: .main-image}


soak helps qualitative researchers rapidly define and run text analysis pipelines -- thematic analysis, classification, and structured data extraction from interviews, surveys, and documents.


## Quick start

Runs a simple thematic analysis on a set of example data (interview transcripts):

```bash
uv tool install soaking

soak test  # set up credentials

soak zs "soak-data/cfs/a*" -t simple -o my-analysis
open my-analysis_dump/my-analysis_simple.html
```




## Tutorials

Start here if you're new to soak.

- [Getting Started](tutorials/getting-started.md) -- Installation and your first analysis
- [Customizing Your Analysis](tutorials/customizing-analysis.md) -- Adapting prompts to your research
- [Working with Results](tutorials/working-with-results.md) -- Understanding codes, themes, and exports


## How-to Guides

Practical guides for specific tasks.

### Analysis workflows

- [Thematic Analysis](how-to/thematic-analysis.md) -- Inductive coding and theme generation
- [Build a Classifier](how-to/build-classifier.md) -- Structured classification pipelines
- [Ground Truth Validation](how-to/ground-truth-validation.md) -- Validate against labelled data

### Working with data

- [Working with Spreadsheet Data](how-to/working-with-spreadsheet-data.md) -- Process CSV and Excel files
- [Pre-extraction Workflow](how-to/pre-extract-workflow.md) -- Filter text before analysis
- [Adapting Pipelines](how-to/adapting-pipelines.md) -- Customizing pipeline workflows


## Explanation

Conceptual background on how soak works.

- [What is soak?](explanation/what-is-soak.md) -- Purpose and design philosophy
- [DAG Architecture](explanation/dag-architecture.md) -- Why pipelines, execution model
- [Node Types](explanation/node-types.md) -- Understanding different processing nodes
- [Template System](explanation/template-system.md) -- Jinja2 and struckdown syntax
- [Quote Verification](explanation/quote-verification-approach.md) -- How quote validation works
- [Model Aliases](explanation/model-aliases.md) -- Configuring models for pipeline roles


## Reference

Technical specifications and quick reference.

- [CLI Reference](reference/cli.md) -- Command-line interface
- [Node Reference](reference/node-reference.md) -- All node types and parameters
- [Quote Verification Algorithm](reference/quote-verification.md) -- Technical specification


---

## Sample outputs

- [Thematic analysis (simple)](https://benwhalley.github.io/soak/samples/cfs1_simple.html) -- Analysis of patient interviews
- [Thematic analysis (extended)](https://benwhalley.github.io/soak/samples/cfs2_simple.html) -- Same data, different model
- [Analysis comparison](https://benwhalley.github.io/soak/samples/comparison.html) -- Comparing two analyses
- [Classifier output](https://benwhalley.github.io/soak/samples/classifier/20251008_085446_5db6_pipeline.html) -- Structured data extraction


## Support

- GitHub: [github.com/benwhalley/soak](https://github.com/benwhalley/soak)
- Issues: [github.com/benwhalley/soak/issues](https://github.com/benwhalley/soak/issues)
