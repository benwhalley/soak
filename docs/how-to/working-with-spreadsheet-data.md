---
layout: default
title: Working with Spreadsheet Data
parent: How-to Guides
nav_order: 4
---

# Working with Spreadsheet Data (CSV/XLSX)

soak can process CSV and Excel files directly, treating each row as a separate document. This is useful for survey data, coded transcripts, or any tabular data you want to analyze with LLMs.

## Quick Start

```bash
# Process CSV file
soak run classifier_tabular data/responses.csv -o results

# Process Excel file
soak run my_pipeline data/survey.xlsx -o analysis
```

## How It Works

When you provide a CSV or XLSX file as input:

1. **Each row becomes a document** -- soak creates one `TrackedItem` per row
2. **Columns become template variables** -- All column values are accessible in your pipeline templates as `{{column_name}}`
3. **Content is in metadata** -- Unlike text files, the row data is stored in metadata, not in `content`
4. **Provenance is tracked** -- Each row gets a unique `source_id` like `filename__row_0`, `filename__row_1`, etc.

## Accessing Column Data in Templates

### Example Data (responses.csv)

```csv
participant_id,age,condition,response
P001,25,control,I felt very relaxed during the session
P002,32,treatment,The intervention helped me focus better
P003,28,control,No significant changes noticed
P004,45,treatment,Remarkable improvement in my daily routine
```

### Pipeline Template

```yaml
name: analyze_responses

nodes:
  - name: classify
    type: Map
    inputs: [documents]

---#classify

Analyze this participant response:

**Participant:** {{participant_id}}
**Age:** {{age}}
**Condition:** {{condition}}
**Response:** {{response}}

Based on the response, classify the sentiment and extract key themes.

[[classification]]
```

All column names (`participant_id`, `age`, `condition`, `response`) are automatically available as template variables.

## Sampling and Filtering

### Take First N Rows

```bash
# Process first 10 rows only
soak run my_pipeline data/survey.csv --head 10 -o test_run
```

### Random Sample

```bash
# Randomly sample 50 rows
soak run my_pipeline data/large_survey.csv --sample 50 -o pilot_analysis
```

This is useful for:
- Testing pipelines on large datasets
- Pilot studies
- Cost estimation before full runs

## Ground Truth Validation

When your CSV contains ground truth labels, you can automatically validate LLM predictions:

### Example Data (coded_data.csv)

```csv
text,sentiment_actual,topic_actual
Great product!,positive,product
Terrible service,negative,service
```

### Pipeline with Ground Truth

```yaml
name: validate_classifier

nodes:
  - name: classify
    type: Classifier
    inputs: [documents]
    ground_truth:
      sentiment:
        existing: sentiment_actual  # Compare to this column
        mapping: null  # Auto-detect mapping
      topic:
        existing: topic_actual

---#classify

Text: {{text}}

Classify the sentiment (positive/negative/neutral) and topic: [[classification]]
```

This will automatically:
- Compare LLM predictions to ground truth labels
- Calculate precision, recall, F1 scores
- Generate confusion matrices
- Export results to CSV with accuracy metrics

See [Ground Truth Validation](ground-truth-validation.md) for details.

## Multi-Column Access

You can access multiple columns in a single template:

```yaml
---#analyze

**Demographics:**
- ID: {{participant_id}}
- Age: {{age}}
- Gender: {{gender}}
- Education: {{education_level}}

**Study Data:**
- Condition: {{condition}}
- Session: {{session_number}}
- Response: {{response}}

Analyze the response considering the participant's demographics: [[analysis]]
```

## Export Preserves Metadata

When you export classifier results, the original CSV columns are preserved:

```bash
soak run classifier_tabular data/responses.csv -o results --dump
```

Output CSV (`results_dump/classify/classifications.csv`) includes:

```csv
index,source_id,participant_id,age,condition,response,sentiment,topic
0,responses__row_0,P001,25,control,I felt very relaxed...,positive,relaxation
1,responses__row_1,P002,32,treatment,The intervention...,positive,focus
```

## Supported Formats

- **CSV** (`.csv`) -- via pandas `read_csv()`
- **Excel** (`.xlsx`) -- via pandas `read_excel()` with openpyxl

## Common Use Cases

### Survey Analysis

```bash
soak run classifier_tabular survey_responses.csv -o survey_analysis
```

### Coded Transcripts

If you have pre-coded interview data in a spreadsheet:

```yaml
# Each row is a coded segment
nodes:
  - name: analyze_codes
    type: Map
    inputs: [documents]

---#analyze_codes

**Segment:** {{segment_id}}
**Speaker:** {{speaker}}
**Existing Code:** {{manual_code}}
**Text:** {{text}}

Compare the manual code with the text and suggest refinements: [[analysis]]
```

### Longitudinal Data

For repeated measures:

```yaml
---#analyze_change

**Participant:** {{participant_id}}
**Timepoint:** {{timepoint}}
**Measure:** {{score}}
**Notes:** {{clinician_notes}}

Analyze change over time: [[analysis]]
```


## Example: End-to-End Analysis

```bash
# 1. Test on small sample
soak run classifier_tabular survey.csv --head 20 -o test

# 2. Review test results
open test_pipeline.html

# 3. Run on full dataset
soak run classifier_tabular survey.csv -o full_analysis -v

# 4. Check CSV output (results are in the dump folder)
open full_analysis_dump/01_Classifier_classify/classifications.csv
```

