---
layout: default
title: Build a Classifier
parent: How-to Guides
nav_order: 2
---

# Building Classification Pipelines

This guide shows how to build pipelines that extract structured data from text using the Classifier node.

## When to Use Classification

**Use classifiers when you need:**

- Structured categorical data (topic, sentiment, category)
- Ratings or scores (1-5 scales, severity levels)
- Yes/no decisions (contains feature X?)
- Multi-label classification (select all that apply)
- Consistent coding across many documents

**Don't use classifiers for:**

- Exploratory analysis (use thematic analysis instead)
- Open-ended text generation (use Transform nodes)
- Extracting quotes or themes (use Map/Reduce patterns)

## Basic Classifier Pipeline

### Step 1: Create Pipeline Structure

```yaml
name: my_classifier

nodes:
  - name: classify
    type: Classifier
    inputs:
      - documents
```

### Step 2: Define Classification Template

```yaml
---#classify

Read the following text:

<text>
{{input}}
</text>

What is the primary topic?
[[pick:topic|health,technology,education,business,other]]

What is the sentiment?
[[pick:sentiment|positive,negative,neutral,mixed]]

Rate the overall positivity (1=very negative, 5=very positive):
[[pick:positivity|1,2,3,4,5]]
```

### Step 3: Run the Pipeline

```bash
uv run soak my_classifier.soak data/*.txt --output results
```

This produces:
- `results.json` - Structured classification data
- `results.html` - Formatted view
- `results_dump/01_Classifier_classify/classifications.csv` - CSV export

## Classification Syntax

### Single Choice (pick)

Select exactly one option:

```
[[pick:field_name|option1,option2,option3]]
```

Example:
```
[[pick:diagnosis|cfs,me,both,unclear]]
[[pick:severity|mild,moderate,severe]]
```

### Multiple Choice (pick*)

Select zero or more options:

```
[[pick*:field_name|option1,option2,option3]]
```

Example:
```
[[pick*:symptoms|fatigue,pain,sleep_issues,cognitive_problems]]
```

### Integer (int)

Extract a numeric value:

```
[[int:field_name]]
```

Example:
```
How many years has the patient been ill?
[[int:years_ill]]
```

### Boolean (boolean)

Yes/no question:

```
[[boolean:field_name]]
```

Example:
```
Is the patient currently employed?
[[boolean:employed]]
```

### Free Text (text)

Short free-form response:

```
[[text:field_name]]
```

Example:
```
Summarize the main complaint in one sentence:
[[text:chief_complaint]]
```

## Multi-Model Classification

Run the same classification with multiple models to assess agreement:

```yaml
- name: classify
  type: Classifier
  model_names:
    - gpt-4o-mini
    - gpt-4o
    - claude-sonnet-4
  agreement_fields:
    - topic
    - sentiment
  inputs:
    - documents
```

This will:
1. Run classification with each model
2. Calculate agreement statistics (Krippendorff's alpha, percentage agreement)
3. Export per-model results and combined statistics

### Viewing Agreement Results

After running with multiple models:

```bash
# View agreement statistics
cat results_dump/01_Classifier_classify/agreement_stats.json | jq

# Check per-model CSV files
cat results_dump/01_Classifier_classify/classifications_gpt-4o-mini.csv
cat results_dump/01_Classifier_classify/classifications_gpt-4o.csv
cat results_dump/01_Classifier_classify/classifications_claude-sonnet-4.csv

# View combined results with agreement metrics
cat results_dump/01_Classifier_classify/classifications_combined.csv
```

## Splitting Strategies

### Classify Sentences

```yaml
- name: sentences
  type: Split
  split_unit: sentences
  chunk_size: 3  # Group 3 sentences together

- name: classify
  type: Classifier
  inputs:
    - sentences
```

Good for: Sentiment analysis, utterance classification

### Classify Paragraphs

```yaml
- name: paragraphs
  type: Split
  split_unit: paragraphs
  chunk_size: 1  # Each paragraph separately

- name: classify
  type: Classifier
  inputs:
    - paragraphs
```

Good for: Topic classification, section categorization

### Classify Whole Documents

```yaml
- name: classify
  type: Classifier
  inputs:
    - documents  # No splitting
```

Good for: Overall document classification, metadata extraction

## Working with Results

### Python Analysis

```python
import pandas as pd

# Load CSV
df = pd.read_csv('results_dump/01_Classifier_classify/classifications.csv')

# Count by category
print(df['topic'].value_counts())
print(df['sentiment'].value_counts())

# Cross-tabulation
print(pd.crosstab(df['topic'], df['sentiment']))

# Filter specific classifications
health_negative = df[(df['topic'] == 'health') & (df['sentiment'] == 'negative')]
print(health_negative[['source_id', 'positivity']])

# Average scores
print(df.groupby('topic')['positivity'].mean())
```

### jq Analysis

```bash
# Get all topics
cat results.json | jq '.nodes[] | select(.name=="classify") | .result[].topic'

# Count sentiments
cat results.json | jq '[.nodes[] | select(.name=="classify") | .result[].sentiment] | group_by(.) | map({sentiment: .[0], count: length})'

# Filter by criteria
cat results.json | jq '.nodes[] | select(.name=="classify") | .result[] | select(.topic=="health" and .positivity >= 4)'
```

## Example: Clinical Note Classification

```yaml
name: clinical_classifier

nodes:
  - name: classify
    type: Classifier
    inputs:
      - documents

---#classify

Read this clinical note:

{{input}}

Patient diagnosis:
[[pick:diagnosis|cfs,me,both,unclear,other]]

Severity level:
[[pick:severity|mild,moderate,severe,very_severe]]

Primary symptoms (select all that apply):
[[pick*:symptoms|fatigue,pain,cognitive_issues,sleep_problems,pem,headaches]]

Duration of illness in years:
[[int:years_ill]]

Currently employed:
[[boolean:employed]]

Main treatment goal:
[[text:treatment_goal]]
```

Run it:

```bash
uv run soak clinical_classifier.soak data/notes/*.txt --output clinical_results
```

Analyze results:

```python
import pandas as pd

df = pd.read_csv('clinical_results_dump/01_Classifier_classify/classifications.csv')

# Severity distribution
print(df['severity'].value_counts(normalize=True))

# Average illness duration by severity
print(df.groupby('severity')['years_ill'].mean())

# Most common symptoms
# Note: symptoms is a list stored as string, need to parse
from ast import literal_eval
all_symptoms = df['symptoms'].apply(literal_eval).explode()
print(all_symptoms.value_counts())

# Employment by severity
print(pd.crosstab(df['severity'], df['employed']))
```

## Best Practices

### Clear Classification Criteria

**Bad:**
```
What type is this?
[[pick:type|A,B,C]]
```

**Good:**
```
Based on the primary complaint, classify this clinical note:

- health_anxiety: Primarily concerned about having an illness
- symptom_management: Focused on managing existing symptoms
- treatment_seeking: Looking for new treatments or interventions

[[pick:type|health_anxiety,symptom_management,treatment_seeking]]
```

### Provide Examples

```
Classify the communication style:

Examples:
- directive: "Take this medication twice daily"
- supportive: "I understand this is difficult"
- informational: "CFS affects the immune system"

<text>
{{input}}
</text>

[[pick:style|directive,supportive,informational,mixed]]
```

### Use Appropriate Field Names

**Bad:**
```
[[pick:thing1|yes,no]]
[[pick:thing2|1,2,3]]
```

**Good:**
```
[[boolean:mentions_fatigue]]
[[pick:pain_severity|none,mild,moderate,severe]]
```

### Test on Small Dataset First

```bash
# Test with single file
uv run soak classifier.soak data/test.txt -f json | jq '.nodes[] | select(.name=="classify")'

# Review first 5 results
uv run soak classifier.soak data/*.txt -o test
head -10 test_dump/01_Classifier_classify/classifications.csv
```

## Common Issues

**Empty classifications:**

Check if template uses correct syntax:
- `[[pick:name|opts]]` not `[[pick|opts]]`
- Options separated by commas, no spaces
- Field names use underscores, not spaces

**Agreement too low:**

- Check if categories are clearly defined
- Provide examples in template
- Ensure categories are mutually exclusive
- Consider combining ambiguous categories

**CSV has weird formatting:**

- Field names with special chars cause issues
- Use `snake_case` for field names
- Avoid spaces, hyphens, or special characters

## Next Steps

- [Node Reference: Classifier](../reference/node-reference.md#classifier) - Full parameter reference
- [Working with Results](../tutorials/working-with-results.md) - Analyzing classification data
- [Ground Truth Validation](./ground-truth-validation.md) - Validating against known answers
