---
layout: default
title: Explanation
nav_order: 4
has_children: true
---

# Explanation

![](../images/ink-explanation.png){: .main-image}


Conceptual background on how soak works. These documents explain the design decisions and architecture.

## Core concepts

- [What is soak?](what-is-soak.md) -- Purpose and design philosophy
- [DAG Architecture](dag-architecture.md) -- Why pipelines, how execution works
- [Node Types](node-types.md) -- Understanding different processing nodes

## Templates and prompts

- [Template System](template-system.md) -- Jinja2 and struckdown syntax
- [Model Aliases](model-aliases.md) -- Configuring models for pipeline roles

## Verification and quality

- [Quote Verification](quote-verification-approach.md) -- How quote validation works
- [Similarity Metrics](similarity-metrics.md) -- Comparing analyses
- [Calibration](calibration.md) -- Calibrating analysis parameters

## Advanced topics

- [Filtering](filtering.md) -- Boolean filtering with LLM
- [HyDE Coverage](hyde-coverage.md) -- Hypothetical document embeddings
