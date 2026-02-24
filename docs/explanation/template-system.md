---
layout: default
title: Template System
parent: Explanation
nav_order: 4
---

# Template System

This document explains how soak's template system works, combining Jinja2 for variable substitution with struckdown for structured LLM outputs.

## Overview

soak templates have two components:

1. **Jinja2** - Template engine for inserting variables and logic
2. **struckdown** - Syntax for extracting structured data from LLM responses

## Jinja2 Templates

### Basic Variable Substitution

Insert context variables using double curly braces:

```jinja2
You are a: {{persona}}
Your research question is: {{research_question}}

Analyze this text:
{{input}}
```

### Accessing Node Results

Reference previous node outputs by name:

```yaml
nodes:
  - name: chunks
    type: Split

  - name: codes
    type: Map
    inputs:
      - chunks

  - name: themes
    type: Transform
    inputs:
      - codes
```

In templates:

```jinja2
---#themes

Here are all the codes from previous stage:

{{codes}}

Now generate themes...
```

### Default Input Variable

The `{{input}}` variable references the first item in the `inputs` list:

```yaml
- name: my_node
  type: Transform
  inputs:
    - previous_result
```

```jinja2
---#my_node

Process this:
{{input}}  <!-- Equivalent to {{previous_result}} -->
```

### Strict Undefined Behavior

soak uses `StrictUndefined` mode -- referencing undefined variables causes an error:

```jinja2
{{undefined_var}}  <!-- ERROR: 'undefined_var' is undefined -->
```

This catches typos and missing context variables early.

### Conditionals and Loops

Standard Jinja2 control structures work:

```jinja2
{% if research_question %}
Research question: {{research_question}}
{% endif %}

{% for code in codes %}
- {{code.name}}
{% endfor %}
```

However, most templates don't need control structures -- use simple variable substitution.

## struckdown Syntax

struckdown extracts structured data from LLM text responses using special `[[syntax]]`.

### Return Type Syntax

General format:

```
[[return_type:field_name]]
```

or with options:

```
[[return_type:field_name|option1,option2,option3]]
```

### Available Return Types

#### Thematic Analysis Types

**[[codes:field_name]]** - Extract list of Code objects

```jinja2
Identify codes in the text:

[[code*:codes]]
```

Expected LLM output format:
```
# frustration_with_doctors

Frustration with medical professionals who dismiss symptoms

> "The doctor said it was all in my head"
```

Returns:
```python
{
  "codes": [
    Code(
      slug="frustration_with_doctors",
      name="Frustration with medical professionals",
      description="...",
      quotes=["The doctor said..."]
    )
  ]
}
```

**[[themes:field_name]]** - Extract list of Theme objects

```jinja2
Group codes into themes:

[[theme*:themes]]
```

Expected LLM output format:
```
# Medical System Barriers

Participants struggle to access appropriate care

Codes: frustration_with_doctors, diagnostic_delay, treatment_access
```

Returns:
```python
{
  "themes": [
    Theme(
      name="Medical System Barriers",
      description="...",
      code_slugs=["frustration_with_doctors", ...]
    )
  ]
}
```

#### Classification Types

See the struckdown package docs for more information on how to use struckdown syntax for classificaiton, but know that you can extract structured information using pick, bool and other types:


```jinja2
What is the sentiment?
[[pick:sentiment|positive,negative,neutral,mixed]]
```

Returns:
```python
{"sentiment": "negative"}
```


```jinja2
Which symptoms are mentioned?
[[pick*:symptoms|fatigue,pain,insomnia,headache]]
```

Returns:
```python
{"symptoms": ["fatigue", "pain"]}
```


Etc...



## Template Sections

### Inline vs External Templates

Templates can be defined inline (within .soak files) or in separate .sd files. External templates enable reuse across pipelines. See [Hybrid Template System](template_resolution.md) for details.

### Section Separators

Templates are separated by triple-dash headers:

```yaml
---#node_name_1

Template content for node_name_1...

---#node_name_2

Template content for node_name_2...
```

The `#node_name` must match a node name in the YAML header.

### Nodes Without Templates

Not all nodes need templates:

- **Split**, **Reduce**, **Batch**, **Filter** - No LLM, no template needed
- **Map**, **Transform**, **Classifier** - Require templates

If a node with LLM has no template, you'll get an error.

## Context Variables

### Default Context

Pipelines define default context in YAML:

```yaml
name: my_pipeline

default_context:
  persona: Experienced qualitative researcher
  research_question: None
```

Access in templates:

```jinja2
You are a: {{persona}}
```

### CLI Context Variables

Override or add context via `-c` flag:

```bash
uv run soak my_pipeline.soak data/*.txt \
  -c research_question="What are recovery experiences?" \
  -c persona="Clinical psychologist"
```

### Node-Specific Context

Access node results as context:

```yaml
- name: themes
  type: Transform
  inputs:
    - codes
    - preliminary_themes
```

```jinja2
---#themes

Codes:
{{codes}}

Preliminary themes:
{{preliminary_themes}}

Consolidate into final themes...
```

### Special Variables

**{{input}}** - First input in node's `inputs` list

**{{item}}** - Current item in Map node (within iteration)

**{{metadata}}** - TrackedItem metadata (available in item context)




## Debugging Templates

### Inspect Rendered Templates

Check the dump directory to see what LLM received:

```bash
uv run soak my_pipeline.soak data/test.txt -o test

# View prompt sent to LLM
cat test_dump/02_Map_code_chunks/0000_*_prompt.md
```




### Conditional Templates

Use Jinja2 to adapt template based on context:

```jinja2
{% if detailed_analysis %}
Provide extensive descriptions (100-150 words per code).
{% else %}
Provide concise descriptions (30-50 words per code).
{% endif %}

[[code*:codes]]
```


## Next Steps

- [Node Reference](../reference/node-reference.md) - See which nodes use templates
- [Customizing Your Analysis](../tutorials/customizing-analysis.md) - Practical template editing
- [DAG Architecture](dag-architecture.md) - How templates fit into pipeline execution
