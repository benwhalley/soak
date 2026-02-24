---
layout: default
title: Model Aliases
parent: Explanation
nav_order: 5
---

# Model Aliases

Model aliases provide a flexible way to configure different LLM models for different roles within a pipeline without hardcoding model IDs. This allows pipelines to define semantic roles (e.g., `default`, `best`, `embeddings`) that can be assigned different models at runtime.

## Why Use Model Aliases?

When building analysis pipelines, different tasks may benefit from different models:

- **Fast, cheap model** for routine operations (chunking, initial filtering)
- **High-capability model** for complex reasoning (theme synthesis, narrative generation)
- **Embedding model** for semantic similarity and clustering

Hardcoding model names makes pipelines inflexible. Model aliases let pipeline authors define *what kind* of model is needed, while users choose *which specific model* to use.

## Defining Aliases in Pipelines

Pipelines define model aliases in the `default_config.models` section:

```yaml
name: thematic_analysis
default_config:
  models:
    default: gpt-4.1-mini    # fast, routine tasks
    best: gpt-4.1            # complex reasoning
  embeddings: text-embedding-3-large
  seed: 42

nodes:
  - name: extract_codes
    type: Map
    model: default           # uses the 'default' alias

  - name: synthesize_themes
    type: Transform
    model: best              # uses the 'best' alias
```

Alias names are arbitrary -- you can define any aliases that make sense for your pipeline. Common conventions:

| Alias | Typical Use |
|-------|-------------|
| `default` | General-purpose tasks, used when no alias specified |
| `best` | High-quality outputs requiring stronger reasoning |
| `cheap` | Cost-sensitive operations, high-volume tasks |
| `fast` | Low-latency operations |

## Using Aliases from the CLI

Override model aliases when running pipelines with the `--model` flag:

```bash
# Set the default model (simple form)
uv run soak run pipeline data/*.txt --model gpt-4.1

# Override specific aliases
uv run soak run pipeline data/*.txt --model default=gpt-4.1-mini --model best=gpt-4.1

# Multiple overrides
uv run soak run pipeline data/*.txt \
  --model default=claude-3-5-sonnet \
  --model best=claude-3-opus \
  --model cheap=gpt-4.1-mini
```

### Syntax

| Form | Effect |
|------|--------|
| `--model gpt-4.1` | Sets the `default` alias |
| `--model alias=model` | Sets a specific alias |

You can combine multiple `--model` flags to override several aliases at once.

## Using Aliases in the Web UI

When creating a run in the web interface:

1. Select a pipeline -- the UI loads its defined aliases
2. For each alias, a dropdown appears showing available models
3. Select which model to use for each role
4. The embedding model has its own selector

The UI pre-populates dropdowns with the pipeline's default values when available.

## How Aliases are Resolved

At execution time:

1. **Pipeline defaults** -- aliases defined in `default_config.models` provide base values
2. **User overrides** -- CLI flags or web UI selections override specific aliases
3. **Node lookup** -- each node's `model` field references an alias name
4. **Resolution** -- the alias is resolved to an actual model ID for API calls

```
Pipeline YAML:           default_config.models.best = "gpt-4.1"
                                    ↓
User override:           --model best=claude-3-5-sonnet
                                    ↓
Node definition:         model: best
                                    ↓
Resolved at runtime:     "claude-3-5-sonnet"
```

## Example Pipeline

A complete pipeline using multiple aliases:

```yaml
name: multi_model_analysis
default_config:
  models:
    default: gpt-4.1-mini
    best: gpt-4.1
    cheap: gpt-4.1-mini
  embeddings: text-embedding-3-large
  seed: 1

nodes:
  # Split uses no model (structural operation)
  - name: chunks
    type: Split
    chunk_size: 30000

  # High-volume coding -- use cheap/fast model
  - name: initial_codes
    type: Map
    model: cheap
    inputs: [chunks]

  # Cluster uses embeddings (configured separately)
  - name: grouped_codes
    type: Cluster
    inputs: [initial_codes]

  # Theme synthesis -- use best model for quality
  - name: themes
    type: Transform
    model: best
    inputs: [grouped_codes]

  # Final narrative -- use best model
  - name: narrative
    type: Transform
    model: best
    inputs: [themes]
```

Running with different model configurations:

```bash
# Use defaults from pipeline
uv run soak run multi_model_analysis data/*.txt

# Use Claude for everything
uv run soak run multi_model_analysis data/*.txt \
  --model default=claude-3-5-sonnet \
  --model best=claude-3-5-sonnet \
  --model cheap=claude-3-5-sonnet

# Mix providers
uv run soak run multi_model_analysis data/*.txt \
  --model cheap=gpt-4.1-mini \
  --model best=claude-3-opus
```

## Embeddings

The `embeddings` configuration is separate from LLM model aliases. It's used by:

- `Cluster` nodes for semantic grouping
- `VerifyQuotes` for quote similarity matching
- `compare` command for analysis comparison

Set in pipeline:

```yaml
default_config:
  embeddings: text-embedding-3-large
```

Override from CLI:

```bash
uv run soak compare --embedding-model text-embedding-3-small *.json
```

## Best Practices

1. **Use meaningful alias names** -- `best` and `cheap` are clearer than `model1` and `model2`

2. **Set sensible defaults** -- pipelines should work out of the box with their default models

3. **Document your aliases** -- add comments explaining what each alias is used for

4. **Consider cost** -- assign cheaper models to high-volume operations (Map nodes with many items)

5. **Test with different models** -- verify your pipeline works across different providers

## See Also

- [CLI Reference](../reference/cli.md) -- full `--model` flag documentation
- [Node Types](node-types.md) -- which nodes use models
- [DAG Architecture](dag-architecture.md) -- how pipelines execute
