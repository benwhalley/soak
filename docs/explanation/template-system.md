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

[[codes:codes]]
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

[[themes:themes]]
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

**[[pick:field_name|options]]** - Single choice from options

```jinja2
What is the sentiment?
[[pick:sentiment|positive,negative,neutral,mixed]]
```

Returns:
```python
{"sentiment": "negative"}
```

**[[pick*:field_name|options]]** - Multiple choice (zero or more)

```jinja2
Which symptoms are mentioned?
[[pick*:symptoms|fatigue,pain,insomnia,headache]]
```

Returns:
```python
{"symptoms": ["fatigue", "pain"]}
```

**[[int:field_name]]** - Integer value

```jinja2
How many years ill?
[[int:years]]
```

Returns:
```python
{"years": 5}
```

**[[boolean:field_name]]** - True/false

```jinja2
Is the patient employed?
[[boolean:employed]]
```

Returns:
```python
{"employed": False}
```

**[[text:field_name]]** - Short free text

```jinja2
Summarize the main complaint:
[[text:complaint]]
```

Returns:
```python
{"complaint": "Chronic fatigue and unrefreshing sleep"}
```

#### Extraction Types

**[[extract:field_name]]** - Free-form text extraction

```jinja2
Extract only participant speech:

[[extract:participant_text]]
```

Returns raw text, no structure.

**[[report]]** - Free-form narrative (no field name)

```jinja2
Write a narrative report:

[[report]]
```

Returns raw text in `report` field.

### Multiple Return Types in One Template

Templates can extract multiple structured outputs:

```jinja2
---#analyze

{{input}}

First, identify codes:
[[codes:codes]]

Next, rate the sentiment:
[[pick:sentiment|positive,negative,neutral]]

Finally, write a summary:
[[text:summary]]
```

Returns:
```python
{
  "codes": [...],
  "sentiment": "negative",
  "summary": "..."
}
```

## Template Sections

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

## struckdown Processing

### How struckdown Works

1. Template sent to LLM
2. LLM responds with structured text
3. struckdown parser extracts data using [[syntax]] markers
4. Structured objects returned

Example:

**Template:**
```jinja2
Identify the topic:
[[pick:topic|health,tech,business]]
```

**LLM Response:**
```
Based on the text, the topic is:

health

The text discusses chronic illness...
```

**struckdown Extraction:**
```python
{"topic": "health"}
```

### struckdown Features

**¡BEGIN** marker - Ignore everything before this:

```jinja2
Here is background context...

¡BEGIN

What is the sentiment?
[[pick:sentiment|positive,negative]]
```

LLM sees all context, but struckdown only parses after `¡BEGIN`.

**¡OBLIVIATE** marker - Reset context between fields:

```jinja2
¡BEGIN

What is the topic?
[[pick:topic|health,tech]]

¡OBLIVIATE

What is the sentiment?
[[pick:sentiment|positive,negative]]
```

Each `¡OBLIVIATE` tells struckdown to not let prior extractions influence next extraction.

## Template Best Practices

### 1. Be Explicit

**Bad:**
```jinja2
Code this:
{{input}}
[[codes:codes]]
```

**Good:**
```jinja2
You are a qualitative researcher conducting thematic analysis.

Read the following interview transcript:

<text>
{{input}}
</text>

Identify all codes. A 'code' should capture specific participant experiences.

Provide:
- A slug (short identifier, e.g., medical_dismissal)
- A name (8-15 words, e.g., "Frustration with doctors who dismiss symptoms")
- A description (50 words explaining what this code represents)
- Direct verbatim quotes from the text

[[codes:codes]]
```

### 2. Structure Input Clearly

Use XML-style tags for clarity:

```jinja2
<research_question>
{{research_question}}
</research_question>

<text_to_analyze>
{{input}}
</text_to_analyze>
```

### 3. Provide Examples

```jinja2
Classify the sentiment as:
- positive: Expresses satisfaction, hope, or positive outcomes
- negative: Expresses frustration, disappointment, or negative outcomes
- neutral: Factual statements without emotional content

Example:
Text: "The treatment helped me return to work"
Sentiment: positive

Now classify this text:
{{input}}

[[pick:sentiment|positive,negative,neutral]]
```

### 4. Specify Format Requirements

```jinja2
Provide VERBATIM quotes (do not paraphrase or summarize).
Use "..." to indicate omitted sections.
Keep quotes under 150 words each.

[[codes:codes]]
```

### 5. Use Context Variables for Flexibility

```yaml
default_context:
  code_criteria: "related to participant experiences"
  quote_requirements: "3-5 short direct quotes per code"
```

```jinja2
Identify codes that are {{code_criteria}}.

For each code, provide {{quote_requirements}}.

[[codes:codes]]
```

## Debugging Templates

### Inspect Rendered Templates

Check the dump directory to see what LLM received:

```bash
uv run soak my_pipeline.soak data/test.txt -o test

# View prompt sent to LLM
cat test_dump/02_Map_code_chunks/0000_*_prompt.md
```

### Common Template Errors

**Undefined variable:**
```
jinja2.exceptions.UndefinedError: 'reserach_question' is undefined
```

Fix: Check spelling, ensure variable in context.

**Invalid struckdown syntax:**
```
struckdown error: Could not parse [[pick]] - missing options
```

Fix: Ensure `[[pick:name|opt1,opt2]]` format.

**No template found:**
```
Template for node 'my_node' not found
```

Fix: Add `---#my_node` section.

## Advanced Techniques

### Nested Extractions

Extract codes, then themes from those codes in one template:

```jinja2
First, identify codes:
[[codes:codes]]

Now group these codes into themes:
[[themes:themes]]
```

### Conditional Templates

Use Jinja2 to adapt template based on context:

```jinja2
{% if detailed_analysis %}
Provide extensive descriptions (100-150 words per code).
{% else %}
Provide concise descriptions (30-50 words per code).
{% endif %}

[[codes:codes]]
```

### Multi-Stage Prompts

Guide LLM through stages:

```jinja2
STAGE 1: Review the text and make notes on key themes.

[[text:notes]]

STAGE 2: Using your notes, identify formal codes.

[[codes:codes]]
```

## Next Steps

- [Node Reference](../reference/node-reference.md) - See which nodes use templates
- [Customizing Your Analysis](../tutorials/customizing-analysis.md) - Practical template editing
- [DAG Architecture](dag-architecture.md) - How templates fit into pipeline execution
