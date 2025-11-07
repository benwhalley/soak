# Ground Truth Validation

Validate your LLM classifications against ground truth labels using the Classifier node's built-in metrics.

## Overview

The Classifier node can automatically:
- Compare LLM predictions to ground truth labels from your data
- Calculate precision, recall, F1 scores (macro, micro, weighted, binary)
- Generate confusion matrices
- Support multiple models with inter-rater agreement

## Quick Start

```yaml
nodes:
  - name: classify_utterances
    type: Classifier
    model_names: [gpt-4o-mini]
    inputs: [documents]
    ground_truths:
      reflection:                      # LLM output field name
        existing: reflection_exists    # Ground truth column name
        mapping:
          yes: 1                       # Map LLM outputs to GT values
          no: 0
          unclear: 0
```

## Configuration

### Basic Structure

```yaml
ground_truths:
  <llm_field_name>:           # Must match LLM output field
    existing: <gt_column>     # Ground truth column in your data
    mapping:                  # How to map LLM outputs
      <llm_value>: <gt_value>
    drop: [<values_to_exclude>]  # Optional: exclude from metrics
```

### Field Names

**Critical:** The ground truth config key must match the LLM output field name.

```yaml
# Template outputs: [[pick:utterance_type|...]]
ground_truths:
  utterance_type:      # ✓ Matches output field
    existing: question_exists

  # utterance_classification:  # ✗ Wrong - doesn't match
  #   existing: question_exists
```

**Error if mismatch:**
```
Field 'utterance_classification' not found in model outputs.
Available output fields: ['utterance_type', 'reflection']
```

## Mapping Strategies

### 1. Binary Classification

Map multiple LLM categories to binary ground truth:

```yaml
utterance_type:
  existing: question_exists    # Values: 1.0, 0.0, NaN
  mapping:
    open_question: 1           # Questions → 1
    closed_question: 1
    statement: 0               # Non-questions → 0
    advice_suggestion: 0
    other: 0
```

**Result:** 2-class confusion matrix (0, 1)

### 2. Multi-class with Wildcards

Collapse unmapped categories using `"*"`:

```yaml
utterance_type:
  existing: question_subtype   # Values: open, closed, facilitating, NaN
  mapping:
    open_question: open
    closed_question: closed
    "*": "*"                   # All others → "other" category
```

**Result:** 3-class confusion matrix (closed, open, other)

### 3. Exclude Categories

Drop predictions you don't want to validate:

```yaml
utterance_type:
  existing: question_exists
  mapping:
    open_question: 1
    closed_question: 1
    statement: ""              # Map to empty string
    advice_suggestion: ""
    other: ""
  drop: [""]                   # Exclude empty string
```

**Result:** Only open_question and closed_question validated

Alternative with wildcard:

```yaml
mapping:
  open_question: 1
  closed_question: 1
  "*": ""                      # All unmapped → ""
drop: [""]                     # Then drop
```

## Boolean Handling

YAML parses `yes`/`no` as booleans. This is handled automatically:

```yaml
mapping:
  yes: 1      # YAML parses as {True: 1, False: 0}
  no: 0       # But matches LLM outputs "yes"/"no"
```

**Supported variations:** yes/Yes/YES/no/No/NO/true/True/TRUE/false/False/FALSE

**To be explicit, use quotes:**

```yaml
mapping:
  "yes": 1    # Stays as string key
  "no": 0
```

## Missing Values

Missing values (NaN, None) are handled as "NA" category:

```yaml
# Ground truth: [1, 0, NaN, 1]
# Becomes:      [1, 0, NA, 1]
```

**To exclude missing:**

```yaml
drop: ["NA"]
```

**To collapse to "other":**

```yaml
mapping:
  yes: 1
  no: 0
  "*": "*"     # NaN → "other"
```

## Output Files

### Metrics CSV

`ground_truth_metrics.csv`:
```csv
field,model,ground_truth_column,mapping,n_samples,precision_macro,recall_macro,f1_macro,...
reflection,gpt-4o-mini,reflection_exists,"{True: 1, False: 0}",50,0.85,0.82,0.83,...
```

### Confusion Matrices

`confusion_matrix_<field>_<model>.csv`:
```csv
# Confusion Matrix: reflection
# Model: gpt-4o-mini
# Sample size: 50
# Ground truth column: reflection_exists
# Mapping applied:
#   yes -> 1
#   no -> 0
#
True \ Predicted,0,1
0,38,2
1,3,7
```

### Detailed JSON

`ground_truth_metrics.json` includes:
- Full confusion matrices
- Per-class precision/recall/F1
- Classification reports
- Configuration metadata

## Metrics Explained

### Macro F1
- Average F1 across all classes
- **Equal weight** to each class
- Use when: Rare classes as important as common ones

### Micro F1
- Global average across all predictions
- **Weighted by frequency** (dominated by common classes)
- Equals overall accuracy
- Use when: Overall correctness matters most

### Weighted F1
- Average F1 weighted by class support
- **Balance between macro and micro**
- Use when: Default choice for reporting

### Binary F1
- F1 for positive class only (2-class problems)
- Uses highest-sorted label as positive (e.g., "1" in ["0", "1"])

## Common Patterns

### Binary with Multiple Predictors

```yaml
model_names: [gpt-4o-mini, gpt-4o, claude-3-5-sonnet]
ground_truths:
  reflection:
    existing: reflection_exists
    mapping: {yes: 1, no: 0, unclear: 0}
```

**Output:** Metrics for each model + inter-rater agreement

### Multi-class Collapsed

```yaml
ground_truths:
  question_type:
    existing: question_subtype
    mapping:
      open_question: open
      closed_question: closed
      "*": other              # statement/advice/etc → "other"
```

### Exclude Non-applicable

```yaml
ground_truths:
  client_utterance:
    existing: client_talk_type
    mapping:
      positive: positive
      negative: negative
      neutral: neutral
      "*": ""
    drop: ["", "NA"]          # Exclude therapist utterances
```

## Error Messages

### Unmapped Predictions

```
Unmapped prediction value(s) found in field 'reflection':
  Values: ['unclear']

Fix option 1 - Map to target values:
    "unclear": 0

  Option 2 - Drop/exclude from metrics:
    mapping:
      "unclear": ""
    drop: [""]

  Option 3 - Use wildcard:
    "*": other
```

### Unpredicted Ground Truth

```
WARNING: Ground truth categories never predicted in 'utterance_type/gpt-4o-mini':
  Categories: ['facilitating', 'NA']
  These will appear in confusion matrix with zero prediction counts.
  To collapse these to 'other' category, add wildcard to mapping:
    "*": "*"  or  "*": other
```

This is informational -- metrics continue, but zero-count categories appear in matrix.

## Duplicate Keys

**YAML doesn't allow duplicate keys** -- only the last one is kept:

```yaml
ground_truths:
  utterance_type:        # First definition
    existing: question_exists
    mapping: {...}

  utterance_type:        # ✗ Overwrites first!
    existing: question_subtype
    mapping: {...}
```

**Fix:** Use unique names:

```yaml
ground_truths:
  utterance_type_binary:
    existing: question_exists

  utterance_type_subtype:
    existing: question_subtype
```

## Best Practices

1. **Match field names** -- Ground truth key must match LLM output field
2. **Use wildcards** -- Avoid explicit mapping of every category
3. **Drop vs map** -- Drop unwanted predictions, don't force-map them
4. **Quote booleans** -- Use `"yes"` and `"no"` if you want explicit control
5. **Test incrementally** -- Start with one field, add more once working
6. **Check confusion matrix** -- Verify categories make sense before trusting metrics
7. **Multiple models** -- Compare different models on same ground truth

## Full Example

```yaml
name: mi_classification
nodes:
  - name: classify
    type: Classifier
    model_names:
      - gpt-4o-mini
      - gpt-4o
    inputs: [documents]
    ground_truths:
      # Binary: is it a question?
      utterance_type:
        existing: question_exists
        mapping:
          open_question: 1
          closed_question: 1
          "*": 0

      # Multi-class: question subtype
      question_subtype:
        existing: question_subtype_gt
        mapping:
          open_question: open
          closed_question: closed
          "*": "*"
        drop: ["NA"]

      # Binary: reflection present?
      reflection:
        existing: reflection_exists
        mapping:
          "yes": 1
          "no": 0
          "unclear": 0
---#classify
[[pick:utterance_type|open_question,closed_question,statement,other]]
[[pick:question_subtype|open_question,closed_question]]
[[pick:reflection|yes,no,unclear]]
```

## See Also

- [Classifier Node Reference](../reference/nodes.md#classifier)
- [Inter-rater Agreement](./inter-rater-agreement.md)
- [Working with Results](../tutorials/working-with-results.md)
