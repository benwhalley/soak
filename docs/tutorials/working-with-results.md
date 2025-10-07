# Working with Results

This tutorial shows how to analyze and interpret soak output data.

## Output Formats

After running a pipeline, you get two files:

```bash
uv run soak run zs data/*.txt --output results

# Creates:
results.json    # Full pipeline data
results.html    # Rendered view
```

### JSON Structure

```json
{
  "name": "zero_shot",
  "config": { ... },
  "nodes": [
    {
      "name": "chunks",
      "type": "Split",
      "result": [ ... ]
    },
    {
      "name": "codes",
      "type": "Transform",
      "result": {
        "codes": [ ... ]
      }
    },
    ...
  ]
}
```

Each node's result is stored. Access via:

```bash
# Get final codes
cat results.json | jq '.nodes[] | select(.name=="codes") | .result.codes'

# Get narrative
cat results.json | jq '.nodes[] | select(.name=="narrative") | .result.report'

# Count themes
cat results.json | jq '.nodes[] | select(.name=="themes") | .result.themes | length'
```

### HTML View

Open in browser:

```bash
open results.html
```

Shows codes, themes, and narrative in readable format.

## Understanding Codes

A Code object:

```json
{
  "slug": "medical_dismissal",
  "name": "Experiences of being dismissed by healthcare providers",
  "description": "Participants describe frustration when doctors minimize...",
  "quotes": [
    "The doctor said it was all in my head",
    "They told me to just exercise more..."
  ]
}
```

**Fields:**

- `slug` - Short identifier (max 20 chars, a-z only)
- `name` - Descriptive name (8-15 words)
- `description` - What the code represents (~50 words)
- `quotes` - Example text from your data

### Extracting Codes

**All code names:**

```bash
cat results.json | jq '.nodes[] | select(.name=="codes") | .result.codes[].name'
```

**Codes with quotes:**

```bash
cat results.json | jq '.nodes[] | select(.name=="codes") | .result.codes[] | {name, quotes}'
```

**Find specific code:**

```bash
cat results.json | jq '.nodes[] | select(.name=="codes") | .result.codes[] | select(.slug=="medical_dismissal")'
```

### Working with Codes in Python

```python
import json

with open("results.json") as f:
    data = json.load(f)

# Find codes node
codes_node = next(n for n in data["nodes"] if n["name"] == "codes")
codes = codes_node["result"]["codes"]

# Print all code names
for code in codes:
    print(f"- {code['name']}")

# Codes by quote count
sorted_codes = sorted(codes, key=lambda c: len(c["quotes"]), reverse=True)
print(f"Most quoted: {sorted_codes[0]['name']} ({len(sorted_codes[0]['quotes'])} quotes)")

# Export to CSV
import csv

with open("codes.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["slug", "name", "description", "quote_count"])
    writer.writeheader()
    for code in codes:
        writer.writerow({
            "slug": code["slug"],
            "name": code["name"],
            "description": code["description"],
            "quote_count": len(code["quotes"])
        })
```

## Understanding Themes

A Theme object:

```json
{
  "name": "Navigating medical system barriers",
  "description": "Participants struggle to access appropriate care due to...",
  "code_slugs": ["medical_dismissal", "diagnostic_delay", "treatment_access"]
}
```

**Fields:**

- `name` - Theme name (8-15 words)
- `description` - What the theme represents (60-80 words)
- `code_slugs` - References to codes (by slug)

### Extracting Themes

**All theme names:**

```bash
cat results.json | jq '.nodes[] | select(.name=="themes") | .result.themes[].name'
```

**Themes with codes:**

```bash
cat results.json | jq '.nodes[] | select(.name=="themes") | .result.themes[] | {name, code_slugs}'
```

### Linking Themes to Codes

```python
import json

with open("results.json") as f:
    data = json.load(f)

codes_node = next(n for n in data["nodes"] if n["name"] == "codes")
themes_node = next(n for n in data["nodes"] if n["name"] == "themes")

codes = {c["slug"]: c for c in codes_node["result"]["codes"]}
themes = themes_node["result"]["themes"]

# Print themes with their codes
for theme in themes:
    print(f"\n{theme['name']}")
    print(f"  {theme['description']}")
    print(f"  Codes:")
    for slug in theme["code_slugs"]:
        code = codes[slug]
        print(f"    - {code['name']}")
        print(f"      {code['quotes'][0][:100]}...")  # First quote preview
```

## Understanding the Narrative

The narrative is formatted text ready for publication:

```bash
cat results.json | jq -r '.nodes[] | select(.name=="narrative") | .result.report'
```

Output:

```markdown
**Theme 1: Living with uncertainty**: Participants described prolonged periods
without diagnosis, leading to anxiety... "I didn't know what was wrong with me
for three years."

**Theme 2: Medical system barriers**: Access to appropriate care was
challenging...
```

Copy directly into your results section.

## Detailed Execution Dump

For detailed inspection, use `--dump`:

```bash
uv run soak run zs data/*.txt --output results --dump
```

Creates `results_dump/` folder:

```
results_dump/
├── 01_Split_chunks/
│   ├── inputs/
│   │   ├── 0000_interview_001.txt
│   │   └── 0000_interview_001_metadata.json
│   ├── outputs/
│   │   ├── 0000_interview_001__chunks__0.txt
│   │   └── 0000_interview_001__chunks__0_metadata.json
│   └── split_summary.txt
├── 02_Map_chunk_codes_and_themes/
│   ├── inputs/
│   ├── 0000_interview_001__chunks__0_prompt.md
│   ├── 0000_interview_001__chunks__0_response.json
│   └── ...
└── metadata.json
```

### Inspecting Node Outputs

**View a specific chunk:**

```bash
cat results_dump/01_Split_chunks/outputs/0000_interview_001__chunks__0.txt
```

**See LLM prompt:**

```bash
cat results_dump/02_Map_chunk_codes_and_themes/0000_*_prompt.md
```

**See LLM response:**

```bash
cat results_dump/02_Map_chunk_codes_and_themes/0000_*_response.json | jq
```

### Tracking Provenance

Each file includes source_id in filename:

```
0000_interview_001__chunks__0.txt
     └─────┬─────┘  └──┬──┘  └┬┘
       document     node   chunk
```

Trace a code's quote back to source:

```python
# Find code
code = codes[0]
quote = code["quotes"][0]

# Search in chunk outputs
import os
for file in os.listdir("results_dump/01_Split_chunks/outputs/"):
    if file.endswith(".txt"):
        with open(f"results_dump/01_Split_chunks/outputs/{file}") as f:
            if quote in f.read():
                print(f"Found in: {file}")
```

## Analyzing Classifications

For classifier pipelines, outputs include CSV:

```bash
uv run soak run classifier data/*.txt --output results --dump
```

Check `results_dump/XX_Classifier_*/classifications.csv`:

```csv
index,source_id,doc_index,original_file,topic,sentiment,positivity
0,interview_001__sentences__0,0,data/interview_001.txt,health,negative,2
1,interview_001__sentences__1,0,data/interview_001.txt,health,neutral,3
2,interview_002__sentences__0,1,data/interview_002.txt,tech,positive,4
```

### Analyzing Classifications in Python

```python
import pandas as pd

df = pd.read_csv("results_dump/XX_Classifier_*/classifications.csv")

# Distribution
print(df['topic'].value_counts())
print(df['sentiment'].value_counts())

# Cross-tabulation
print(pd.crosstab(df['topic'], df['sentiment']))

# By document
by_doc = df.groupby('original_file').agg({
    'topic': lambda x: x.mode()[0],  # Most common topic
    'sentiment': lambda x: x.mode()[0],
    'positivity': 'mean'
})
print(by_doc)

# Find specific classifications
health_negative = df[(df['topic'] == 'health') & (df['sentiment'] == 'negative')]
print(health_negative['source_id'])
```

## Quote Verification

If your pipeline includes `VerifyQuotes`:

```bash
cat results_dump/XX_VerifyQuotes_*/verification.txt
```

Shows which quotes failed verification:

```
Verifying 45 quotes...
✓ 42 quotes verified
✗ 3 quotes failed:

Code: medical_dismissal
Quote: "The doctor said it was in my head"
Reason: Not found in source documents (possible paraphrase)
```

Fix by:
- Checking original quote in LLM response
- Emphasizing "verbatim quotes" in template
- Reviewing chunks for quote boundaries

## Comparing Multiple Runs

Run same pipeline multiple times with different parameters:

```bash
uv run soak run zs data/*.txt -o run1
uv run soak run zs data/*.txt -o run2 --model-name openai/gpt-4o
uv run soak run zs data/*.txt -o run3 -c persona="Clinical psychologist"
```

Compare:

```bash
uv run soak compare run1.json run2.json run3.json -o comparison.html
open comparison.html
```

Shows:
- Theme similarity heatmaps
- Network plots of overlapping themes
- Agreement statistics

## Exporting for Analysis

### Convert to DataFrame

```python
import json
import pandas as pd

with open("results.json") as f:
    data = json.load(f)

codes_node = next(n for n in data["nodes"] if n["name"] == "codes")
codes = codes_node["result"]["codes"]

# Flatten codes
rows = []
for code in codes:
    for quote in code["quotes"]:
        rows.append({
            "code_slug": code["slug"],
            "code_name": code["name"],
            "quote": quote
        })

df = pd.DataFrame(rows)
df.to_csv("codes_with_quotes.csv", index=False)
```

### Import to NVivo/Atlas.ti

Export codes as CSV for import:

```python
import csv

with open("codes_for_nvivo.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Code", "Description", "Example"])
    for code in codes:
        writer.writerow([
            code["name"],
            code["description"],
            "; ".join(code["quotes"][:3])  # First 3 quotes
        ])
```

## Tips

**Find longest codes:**

```bash
cat results.json | jq '.nodes[] | select(.name=="codes") | .result.codes | sort_by(.description | length) | reverse | .[0:3] | .[].name'
```

**Count total quotes:**

```bash
cat results.json | jq '[.nodes[] | select(.name=="codes") | .result.codes[].quotes[]] | length'
```

**Extract just narrative:**

```bash
cat results.json | jq -r '.nodes[] | select(.name=="narrative") | .result.report' > narrative.md
```

**Re-export with different template:**

```bash
uv run soak export results.json -t my_custom.html
```

**Inspect specific node:**

```bash
# What did the 'all_codes' Reduce produce?
cat results.json | jq '.nodes[] | select(.name=="all_codes") | .result'
```

## Next Steps

- [Thematic Analysis](../how-to/thematic-analysis.md) - Understanding the pipeline
- [Customizing Your Analysis](customizing-analysis.md) - Adapting prompts
- [Node Reference](../reference/node-reference.md) - All node types and outputs
