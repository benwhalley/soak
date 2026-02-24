---
layout: default
title: Explanation
nav_order: 4
has_children: true
---

# Explanation

![](../images/u3311749543_httpss.mj.runKSun0h8R-vs_httpss.mj.rungggOgHBnZsM_8e94b8b6-f495-428b-bc3b-d4c70bdf18f9_2.png){: .main-image}


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
