# Pre-extraction Workflow

The `zspe` (zero-shot pre-extract) pipeline filters text before analysis. Use this when documents contain irrelevant content or when you want to focus on specific topics.

## When to Use Pre-extraction

**Use zspe when:**

- Documents are long with mixed content (e.g., full interview transcripts with off-topic discussion)
- You want to analyze specific topics mentioned throughout documents
- Source material includes interviewer speech you want to exclude
- You need to reduce processing time/cost by filtering early

**Use standard zs when:**

- All content is relevant
- You want comprehensive analysis
- Documents are already focused on your topic

## Pipeline Overview

The `zspe` pipeline adds a pre-extraction step before coding:

```
1. Split documents into chunks
2. Extract relevant excerpts from each chunk  ← NEW
3. Generate codes and themes from excerpts
4. Reduce and consolidate (as in zs)
5. Final themes and narrative
```

## Running the Pipeline

```bash
uv run soak run zspe data/interviews/*.txt \
  --output results \
  -c excerpt_topics="Exercise, physical rehabilitation, and recovery through movement"
```

The `excerpt_topics` context variable tells the LLM what to extract.

## Pipeline Stages

### Stage 1: chunks (Split)

Same as `zs` pipeline - splits documents into manageable chunks.

```yaml
- name: chunks
  type: Split
  chunk_size: 30000
```

### Stage 2: extract_relevant_excepts (Map)

This node filters each chunk to relevant content.

```yaml
- name: extract_relevant_excepts
  type: Map
  inputs:
    - chunks
```

**Template:**

```
You are pre-reading qualitative transcripts to identify the most relevant
exceptions to the research question.

You will extract text from the interview according to the following criteria:
- only patients' speech (not the interviewer, unless necessary for context)
- EXACTLY the text from the transcript, VERBATIM, with no amendments
- only the sections that are relevant to the research question

<transcribed text>
{{input}}
</transcribed text>

Extract elements of the text relevant to the research question:

{{excerpt_topics}}

Copy text related to this question/topic verbatim.

[[extract:relevant_content]]
```

**Key points:**

- `{{excerpt_topics}}` comes from CLI `-c` option
- Template emphasizes VERBATIM extraction (no paraphrasing)
- Filters to participant speech only
- Uses `[[extract:relevant_content]]` for free-form text output

**Input:** Each chunk
**Output:** Filtered text containing only relevant excerpts

### Stage 3+: chunk_codes_and_themes (Map)

Same as `zs` pipeline, but operates on extracted content instead of full chunks.

```yaml
- name: chunk_codes_and_themes
  type: Map
  max_tokens: 16000
  inputs:
    - extract_relevant_excepts  # Note: uses filtered content
```

The rest of the pipeline (all_codes, all_themes, codes, themes, narrative) is identical to `zs`.

## Comparison: zs vs zspe

### Standard zs

```
Document → Chunks → Code all chunks → Consolidate
```

Codes everything, including off-topic content.

### Pre-extraction zspe

```
Document → Chunks → Extract relevant → Code excerpts → Consolidate
```

Codes only content matching your topic.

## Example Workflow

### Step 1: Identify Your Topic

What aspect of the data matters for your research question?

Examples:
- "Recovery experiences and symptom improvement"
- "Work and employment challenges"
- "Social relationships and support systems"
- "Treatment experiences and medical interactions"

### Step 2: Run with Topic

```bash
uv run soak run zspe data/*.txt \
  --output results \
  -c excerpt_topics="Recovery experiences and symptom improvement"
```

### Step 3: Review Extractions

Check `results_dump/02_Map_extract_relevant_excepts/` to see what was extracted:

```bash
# Dump detailed execution
uv run soak dump results.json

# View an extraction
cat results_dump/02_Map_extract_relevant_excepts/0000_*_response.json | jq '.relevant_content'
```

If extractions are too narrow/broad, adjust `excerpt_topics` and re-run.

### Step 4: Analyze Results

View codes and themes as usual:

```bash
open results.html
```

Codes will focus on your specified topics.

## Customization

### Multiple Topics

```bash
uv run soak run zspe data/*.txt \
  --output results \
  -c excerpt_topics="1) Physical symptoms and energy levels, 2) Social isolation, 3) Medical treatment"
```

### Adjust Extraction Instructions

Copy and modify the template:

```bash
uv run soak show pipeline zspe > my_zspe.yaml
```

Edit `extract_relevant_excepts` section:

```yaml
---#extract_relevant_excepts

Extract ONLY direct quotes where participants discuss:

{{excerpt_topics}}

Rules:
- Participant speech only
- Keep full sentences for context
- Include emotional language
- Preserve hesitations and emphasis

[[extract:relevant_content]]
```

### Include Interviewer Context

Modify template to keep interviewer questions:

```
Extract text according to criteria:
- Participant responses about: {{excerpt_topics}}
- Include interviewer questions immediately before responses
- Keep verbatim

[[extract:relevant_content]]
```

## Tips

**Check extraction quality:**

Always dump execution to verify extractions match expectations:

```bash
uv run soak run zspe data/*.txt -o results --dump
```

**Too little extracted:**

- Broaden `excerpt_topics` description
- Check if topic actually appears in data
- Review original documents

**Too much extracted:**

- Narrow `excerpt_topics` to specific aspects
- Add exclusion criteria to template
- Use more specific language

**Extraction misses context:**

Increase `chunk_size` so related content stays together:

```yaml
- name: chunks
  type: Split
  chunk_size: 50000  # Larger chunks = better context
```

**Paraphrasing instead of verbatim:**

Emphasize in template:

```
CRITICAL: Copy text EXACTLY as written. Do not paraphrase, summarize, or edit.
Use "..." to mark skipped sections.
```

## When Pre-extraction Helps

### Long Mixed Documents

Interview transcripts often wander off-topic. Pre-extraction keeps analysis focused:

```
Full transcript: 50,000 words
After extraction: 8,000 words (relevant content only)
Result: Faster, cheaper, more focused codes
```

### Multi-topic Datasets

Run multiple analyses on same data with different topics:

```bash
# Analysis 1: Physical health
uv run soak run zspe data/*.txt -o health_analysis \
  -c excerpt_topics="Physical symptoms and bodily experiences"

# Analysis 2: Social impact
uv run soak run zspe data/*.txt -o social_analysis \
  -c excerpt_topics="Relationships and social interactions"

# Compare results
uv run soak compare health_analysis.json social_analysis.json
```

### Filtering Irrelevant Speakers

Transcripts with multiple speakers:

```
Extract criteria:
- Patient speech only
- Exclude clinician, researcher, family members
- Include only when discussing: {{excerpt_topics}}
```

## Common Issues

**Empty extractions:**

Check if `excerpt_topics` matches actual content:

```bash
# Verify topic exists in data
grep -i "recovery" data/*.txt
```

**Extraction too aggressive:**

LLM might filter out important context. Review and adjust:

```bash
cat results_dump/02_Map_extract_relevant_excepts/0000_*_response.json
```

**Quotes don't verify:**

VerifyQuotes may fail if extraction modified text. Ensure template emphasizes verbatim copying.

## Next Steps

- [Thematic Analysis](thematic-analysis.md) - Understanding the full pipeline
- [Customizing Your Analysis](../tutorials/customizing-analysis.md) - Adapting prompts
- [Node Types](../explanation/node-types.md) - Understanding Map nodes
