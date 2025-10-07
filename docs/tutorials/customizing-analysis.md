# Customizing Your Analysis

This tutorial shows how to adapt soak pipelines to your research needs by modifying prompts and pipeline structure.

## Quick Customization: Context Variables

The fastest way to customize is using context variables with `-c`:

```bash
uv run soak run zs data/*.txt \
  --output results \
  -c research_question="What factors influence treatment adherence?" \
  -c persona="Health psychologist specializing in chronic illness"
```

Context variables inject into templates via `{{variable_name}}`.

### Available Variables (zs/zspe)

Check pipeline defaults:

```bash
uv run soak show pipeline zs | grep -A 5 "default_context"
```

Common variables:
- `persona` - Who the LLM should act as (default: "Experienced qual researcher")
- `research_question` - Your specific research question (default: None)
- `excerpt_topics` - Topics to extract (zspe only)

## Deep Customization: Editing Pipelines

For more control, copy and modify pipeline files.

### Step 1: Get the Pipeline

```bash
uv run soak show pipeline zs > my_analysis.yaml
```

### Step 2: Edit the YAML

Open `my_analysis.yaml` in your editor. The file has two sections:

**Front matter** (YAML):

```yaml
name: zero_shot
default_context:
  persona: Experienced qual researcher
  research_question: None

nodes:
  - name: chunks
    type: Split
    chunk_size: 30000
  # ... more nodes
```

**Templates** (Jinja2 + struckdown):

```yaml
---#chunk_codes_and_themes

You are a: {{persona}}

{{research_question}}

Identify all relevant codes in the text...

[[codes:codes]]
```

### Step 3: Modify for Your Domain

Example: Adapting for education research

```yaml
name: education_analysis
default_context:
  persona: Education researcher studying student engagement
  research_question: How do students experience online learning?

nodes:
  - name: chunks
    type: Split
    chunk_size: 20000  # Smaller chunks for detailed coding

  - name: chunk_codes_and_themes
    type: Map
    max_tokens: 16000
    inputs:
      - chunks
```

Then edit template:

```yaml
---#chunk_codes_and_themes

You are a: {{persona}}

Research question: {{research_question}}

Code this student interview transcript. Focus on:
- Learning experiences (positive and negative)
- Technology use and challenges
- Social interaction and isolation
- Motivation and engagement

A 'code' should capture specific aspects of the student experience.

Identify all codes, with:
- Name (8-15 words)
- Description (50 words)
- Direct quotes from the student

<text>
{{input}}
</text>

[[codes:codes]]

Now identify themes that group related codes...

[[themes:themes]]
```

### Step 4: Run Your Custom Pipeline

```bash
uv run soak run my_analysis.yaml data/student_interviews/*.txt --output results
```

## Common Customizations

### Change Code/Theme Criteria

**Original (zs.yaml):**

```
A 'code' should be related to the desires, needs, and meaningful outcomes
for participants.
```

**Modified for behavior analysis:**

```
A 'code' should identify specific behaviors, actions, or practices mentioned
by participants. Focus on what people do, not just what they feel.
```

### Adjust Number of Themes

**Original:**

```
Review and consolidate into ~7 (+/- 2) overarching major themes.
```

**Modified:**

```
Review and consolidate into ~4 major themes. Use fewer themes to capture
only the most prominent patterns.
```

### Change Quote Requirements

**Original:**

```
Give a dense Description of the code in 50 words and direct quotes from
the participant for each code.
```

**Modified:**

```
Give a dense Description of the code in 50 words and 2-3 SHORT direct quotes
(max 2 sentences each) from the participant for each code.
```

### Add Custom Instructions

Insert domain-specific guidance:

```yaml
---#chunk_codes_and_themes

You are analyzing clinical interviews about treatment experiences.

IMPORTANT CONTEXT:
- Participants have chronic fatigue syndrome (CFS/ME)
- Many tried multiple treatments before finding help
- Recovery is often partial, not complete
- Medical dismissal is a common theme

When coding:
- Distinguish between complete/partial/no recovery
- Note treatments tried (medical, alternative, self-directed)
- Flag experiences of medical gaslighting or dismissal

{{input}}

[[codes:codes]]
```

## Working with Return Types

soak uses struckdown syntax for structured outputs: `[[return_type:field_name]]`

### Available Return Types

**Thematic analysis:**
- `[[codes:codes]]` - List of Code objects
- `[[themes:themes]]` - List of Theme objects
- `[[extract:text]]` - Free-form text extraction
- `[[report]]` - Free-form narrative

**Classification (see classifier.yaml):**
- `[[pick:field|option1,option2]]` - Single choice
- `[[pick*:field|option1,option2]]` - Multiple choice
- `[[int:field]]` - Integer
- `[[boolean:field]]` - Yes/no
- `[[text:field]]` - Free text

### Example: Custom Structured Output

```yaml
---#assessment

Read this clinical note and extract structured information:

{{input}}

Patient diagnosis:
[[pick:diagnosis|cfs,me,both,unclear]]

Severity level:
[[pick:severity|mild,moderate,severe,very_severe]]

Primary symptoms (select all that apply):
[[pick*:symptoms|fatigue,pain,cognitive_issues,sleep_problems,pem]]

Duration of illness in years:
[[int:years_ill]]

Currently employed:
[[boolean:employed]]

Clinical notes:
[[text:notes]]
```

This creates a dictionary with typed fields.

## Pipeline Structure Changes

### Add a New Node

Insert a filtering step:

```yaml
nodes:
  - name: chunks
    type: Split
    chunk_size: 30000

  - name: filter_relevant      # NEW NODE
    type: Map
    inputs:
      - chunks

  - name: chunk_codes_and_themes
    type: Map
    inputs:
      - filter_relevant        # Changed from 'chunks'
```

Then add template:

```yaml
---#filter_relevant

Remove any text that is:
- Interviewer speech (unless needed for context)
- Off-topic small talk
- Administrative content

Keep only substantive participant responses.

{{input}}

[[extract:filtered_text]]
```

### Remove a Node

Delete the node definition and its template. Update dependent nodes:

```yaml
# Remove checkquotes node
nodes:
  # ... other nodes
  # - name: checkquotes      # REMOVED
  #   type: VerifyQuotes
  #   inputs:
  #     - codes
```

Delete the template section:

```yaml
# ---#checkquotes           # DELETE THIS SECTION
```

### Change Node Parameters

Adjust processing behavior:

```yaml
- name: chunks
  type: Split
  chunk_size: 15000          # Smaller chunks
  overlap: 500               # Add overlap to preserve context

- name: chunk_codes_and_themes
  type: Map
  max_tokens: 8000           # Reduce max tokens
  temperature: 0.3           # Lower temperature = more consistent
  inputs:
    - chunks
```

## Testing Your Changes

### Test on Small Data

```bash
# Test with single file first
uv run soak run my_analysis.yaml data/test_interview.txt -f json | jq '.codes'
```

### Check Intermediate Outputs

```bash
# Dump execution to inspect each stage
uv run soak run my_analysis.yaml data/test.txt -o test --dump

# Review specific node output
cat test_dump/02_Map_chunk_codes_and_themes/0000_*_response.json | jq
```

### Validate Templates

Templates use Jinja2. Test syntax:

```python
from jinja2 import Template

template = Template("{{research_question}}")
print(template.render(research_question="What is recovery?"))
```

## Example: Complete Custom Pipeline

Here's a focused pipeline for analyzing treatment experiences:

```yaml
name: treatment_analysis
default_context:
  persona: Medical anthropologist
  condition: chronic fatigue syndrome

nodes:
  - name: chunks
    type: Split
    chunk_size: 25000

  - name: treatment_codes
    type: Map
    max_tokens: 12000
    inputs:
      - chunks

  - name: all_treatments
    type: Reduce
    inputs:
      - treatment_codes

  - name: consolidated_treatments
    type: Transform
    inputs:
      - all_treatments

  - name: narrative
    type: Transform
    inputs:
      - consolidated_treatments

---#treatment_codes

You are a {{persona}} analyzing patient experiences with {{condition}}.

Focus exclusively on TREATMENT experiences. Code for:
- Treatments tried (name them specifically)
- Effectiveness (helped/hurt/no effect)
- Side effects
- Reasons for starting/stopping
- Provider relationships

{{input}}

[[codes:codes]]

---#all_treatments

{{input.codes}}

---#consolidated_treatments

Merge duplicate treatments from different transcripts.

{{all_treatments}}

[[codenotes]]

[[codes:codes]]

---#narrative

Summarize treatment patterns:

{{consolidated_treatments}}

[[report]]
```

Run it:

```bash
uv run soak run treatment_analysis.yaml data/*.txt -o treatment_results
```

## Tips

**Keep original pipeline:**

```bash
cp my_analysis.yaml my_analysis_backup.yaml
```

**Version your pipelines:**

```yaml
name: education_analysis_v2
# Note: Changed theme consolidation prompt
```

**Document your changes:**

```yaml
# Added filtering node to remove interviewer speech
# Reduced chunk_size from 30k to 20k for finer granularity
# Modified code criteria to focus on behaviors not feelings
```

**Start small:**

Change one thing at a time and test. Don't modify multiple nodes simultaneously.

**Use verbose mode:**

```bash
uv run soak run my_analysis.yaml data/test.txt -o test -v
```

Shows what's happening at each stage.

## Next Steps

- [Working with Results](working-with-results.md) - Analyzing output data
- [Node Types](../explanation/node-types.md) - Understanding different nodes
- [Template System](../explanation/template-system.md) - Advanced template features
- [Node Reference](../reference/node-reference.md) - All node parameters
