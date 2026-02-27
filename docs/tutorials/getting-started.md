---
layout: default
title: Getting Started
parent: Tutorials
nav_order: 1
---

# Getting Started with soak

This tutorial walks you through installing soak and running your first thematic analysis.

## Prerequisites

- 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- API key for an OpenAI-compatible LLM provider

## Installation

```bash
uv tool install soaking
```

Run the test command to set environment variables:

```bash
soak test
```

This will save these variables to your `.env` file:

```bash
LLM_API_KEY=your_api_key
LLM_API_BASE=https://api.openai.com/v1  # Optional
```

## Your First Analysis

### 1. Prepare Your Data

Create a text file with some interview data. For this example, create `data/interview.txt`:

```txt
I started feeling ill about two years ago. At first it was just fatigue,
but then I couldn't get out of bed. The doctors didn't know what was wrong.

Eventually I was diagnosed with CFS. It was a relief to have a name for it,
but also scary because there's no cure. I've had to completely change my life.

The hardest part is that people don't understand. They think I'm just tired.
But this is different - it's like my body just stopped working properly.
```

### 2. Run the Analysis

Use the built-in `zs` (zero-shot) pipeline for thematic analysis, based on [Raza et al 2025](https://arxiv.org/abs/2502.01620).

```bash
uv run soak zs "soak-data/UKDA-2000-tab/rtf/2000int00*.rtf" --output social_history -f
```


You can also run the analysis in a Python script. In this instance, we run multiple versions of the analysis, each with different directions:

```python
#!/usr/bin/env python3

from soak import api
import json
from pathlib import Path

cfg = json.loads(Path("perspectives.json").read_text())

for i, (name, block) in enumerate(cfg["perspectives"].items(), 1):
    result = api.run(
        "zs",
        "soak-data/UKDA-2000-tab/rtf/2000int00*.rtf",
        context=block["context"],
        output=f"history_{name}",
        seed=i,
        skip_nodes=["checkquotes"],
        force=True,
        progress=True
    )
    print(f"{name}: {len(result.themes)} themes")

```


This will:

- Split the text into chunks
- Generate codes from each chunk
- Identify themes
- Consolidate codes and themes
- Verify quotes
- Write a narrative report

The process takes a few minutes depending on text length of the interview.


### 3. View Results

Open the HTML output:

```bash
open my_first_analysis.html
```

You'll see:

- **Codes**: Specific concepts identified in the text (with quotes)
- **Themes**: Broader patterns grouping codes
- **Narrative**: A written report of findings

A JSON file containing all the model output and logging all LLM calls is also available:

```bash
cat my_first_analysis.json | jq '.codes'
```




## Understanding the Output

### Codes

Each code has:

- **slug**: Short identifier (e.g., `illness_onset`)
- **name**: Descriptive name (e.g., "Gradual onset of unexplained symptoms")
- **description**: What the code represents
- **quotes**: Example text from your data

Example:

```json
{
  "slug": "social_misunderstanding",
  "name": "Others fail to grasp the severity of the condition",
  "description": "Participants describe frustration when family, friends...",
  "quotes": [
    "people don't understand. They think I'm just tired."
  ]
}
```

### Themes

Themes group related codes:

```json
{
  "name": "Living with chronic illness uncertainty",
  "description": "Participants navigate the challenges of...",
  "code_slugs": ["illness_onset", "diagnosis_relief", "lifestyle_changes"]
}
```

### Narrative

A written report (sort of) ready for your results section:

> **Living with chronic illness uncertainty**: Participants described a gradual
> onset of symptoms that were initially unexplained. As one participant noted,
> "At first it was just fatigue, but then I couldn't get out of bed"...

## Next Steps

- **Customize the analysis**: See [Customizing Your Analysis](customizing-analysis.md)
- **Understand the pipeline**: See [Thematic Analysis How-to](../how-to/thematic-analysis.md)
- **Work with multiple files**: `soak zs data/*.txt --output results`
- **Try classification**: See [Build a Classifier](../how-to/build-classifier.md)

