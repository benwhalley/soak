# Thematic Analysis with soak

This guide explains how soak performs inductive thematic analysis using the `zs.yaml` pipeline.

## Pipeline Overview

The `zs` pipeline implements a standard thematic analysis workflow:

1. **Split**: Break documents into manageable chunks
2. **Map**: Generate codes and themes from each chunk independently
3. **Reduce**: Collect all codes and themes
4. **Transform**: Consolidate into final codebook
5. **Transform**: Generate final themes
6. **VerifyQuotes**: Check quotes against source
7. **Transform**: Write narrative report

## Running the Pipeline

```bash
uv run soak run zs data/interviews/*.txt --output results
```

The pipeline processes all files in parallel, then consolidates results.

## Pipeline Stages

### Stage 1: chunks (Split)

```yaml
- name: chunks
  type: Split
  chunk_size: 30000
```

Splits documents into ~30,000 character chunks. This keeps LLM context manageable while preserving coherence.

**Input**: Raw documents
**Output**: List of text chunks with provenance tracking

### Stage 2: chunk_codes_and_themes (Map)

This is the core coding stage. Each chunk is analyzed independently.

```yaml
- name: chunk_codes_and_themes
  type: Map
  max_tokens: 16000
  inputs:
    - chunks
```

**Template** (excerpt):

```
You will code the transcript independently, without using a pre-existing codebook.

Identify all relevant codes in the text, provide a Name for each code in 8 to 15 words.

Give a dense Description of the code in 50 words and direct quotes from the participant.

[[codes:codes]]
```

The `[[codes:codes]]` syntax uses struckdown to extract structured `Code` objects:

```python
class Code(BaseModel):
    slug: str  # Short identifier (max 20 chars)
    name: str  # Descriptive name
    description: str
    quotes: List[str]  # Example quotes
```

After coding, the same template generates themes:

```
Your task now is to group the initial codes into distinct themes.

Provide a descriptive and specific name of 8 to 15 words for each theme.

[[themes:themes]]
```

**Input**: Each chunk
**Output**: List of `ChatterResult` objects, each containing codes and themes for one chunk

### Stage 3: all_codes (Reduce)

```yaml
- name: all_codes
  type: Reduce
  inputs:
    - chunk_codes_and_themes
```

**Template**:

```
{{input.codes}}
```

Collects the `.codes` field from all chunk results into a single text.

**Input**: All chunk_codes_and_themes results
**Output**: Concatenated text of all codes

### Stage 4: all_themes (Reduce)

```yaml
- name: all_themes
  type: Reduce
  inputs:
    - chunk_codes_and_themes
```

**Template**:

```
{{input.themes}}
```

Collects the `.themes` field from all chunk results.

**Input**: All chunk_codes_and_themes results
**Output**: Concatenated text of all themes

### Stage 5: codes (Transform)

Consolidates duplicate codes from different chunks into a single codebook.

```yaml
- name: codes
  type: Transform
  max_tokens: 32000
  inputs:
    - all_codes
    - all_themes
```

**Template** (excerpt):

```
We are now going to rationalise the set of codes identified across multiple documents.

## Preliminary codes

{{all_codes}}

First, make a short list of notes on the codes we want to keep. Avoid duplicates.

[[codenotes]]

Now, form this list of new/aligned codes into the required format.

[[codes:codes]]
```

This two-step process (notes → structured output) helps the LLM deduplicate effectively.

**Input**: All codes and themes
**Output**: Final consolidated `CodeList`

### Stage 6: themes (Transform)

Generates final themes that reference the consolidated codes.

```yaml
- name: themes
  type: Transform
  max_tokens: 32000
  inputs:
    - codes
    - all_themes
```

**Template** (excerpt):

```
Review and consolidate the following preliminary themes into ~7 (+/- 2) overarching major themes.

Complete list of codes identified:

{{codes}}

Preliminary themes:

{{all_themes}}

[[themes:themes]]
```

Each theme includes `code_slugs` that reference codes by their slug identifier.

**Input**: Final codes and preliminary themes
**Output**: Final `Themes` object

### Stage 7: checkquotes (VerifyQuotes)

Validates that quotes in codes actually appear in source documents.

```yaml
- name: checkquotes
  type: VerifyQuotes
  inputs:
    - codes
```

Checks each quote against original documents. Invalid quotes are flagged.

**Input**: Final codes
**Output**: Verification report

### Stage 8: narrative (Transform)

Writes a narrative report suitable for publication.

```yaml
- name: narrative
  type: Transform
  inputs:
    - themes
    - codes
```

**Template**:

```
Results of a thematic analysis:

## CODES
{{codes}}

## THEMES
{{themes}}

Write this up as a standard qualitative report, ready for copying into
the results section of an academic journal.

Be brief - about 1 paragraph per theme. Include quotes for each theme.

[[report]]
```

**Input**: Final codes and themes
**Output**: Formatted narrative text

## Understanding the Flow

### Parallel Processing

```
Document 1 ──┐
Document 2 ──┼─→ Split ──→ Chunk 1 ──┐
Document 3 ──┘                       │
                          Chunk 2 ──┼─→ Map (codes+themes for each)
                          Chunk 3 ──┘
                                │
                                ↓
                          all_codes, all_themes
                                │
                                ↓
                          Consolidate codes
                                │
                                ↓
                          Generate final themes
                                │
                                ↓
                          Verify quotes
                                │
                                ↓
                          Write narrative
```

### Why This Architecture?

**Split first**: Large documents exceed LLM context limits. Chunking enables processing long texts.

**Map chunks independently**: Parallel processing is faster. Each chunk gets full attention.

**Reduce then consolidate**: Collecting all codes first, then deduplicating, produces better results than streaming consolidation.

**Two-stage themes**: Initial themes from chunks provide diversity. Final consolidation ensures coherence.

## Outputs

After running, check `results/`:

```
results/
├── 01_Split_chunks/
│   ├── inputs/           # Original documents
│   └── outputs/          # Generated chunks
├── 02_Map_chunk_codes_and_themes/
│   ├── inputs/           # Chunks that were coded
│   ├── 0000_*.json       # Full ChatterResult for each chunk
│   └── ...
├── 03_Reduce_all_codes/
├── 04_Reduce_all_themes/
├── 05_Transform_codes/
│   └── result.json       # Final CodeList
├── 06_Transform_themes/
│   └── result.json       # Final Themes
├── 07_VerifyQuotes_checkquotes/
│   └── verification.txt
└── 08_Transform_narrative/
    └── result.txt        # Report text
```

The `results.json` and `results.html` files combine key outputs for easy viewing.

## Customization

### Adjust Chunk Size

Smaller chunks = more granular coding, longer runtime:

```yaml
- name: chunks
  type: Split
  chunk_size: 15000  # Half the default
```

### Change Model

```bash
uv run soak run zs data/*.txt --output results --model-name openai/gpt-4o
```

### Modify Prompts

Copy `soak/pipelines/zs.yaml` locally and edit templates:

```bash
cp soak/pipelines/zs.yaml my_analysis.yaml
# Edit my_analysis.yaml
uv run soak run my_analysis.yaml data/*.txt --output results
```

See [Customizing Your Analysis](../tutorials/customizing-analysis.md) for details.

### Add Context Variables

```bash
uv run soak run zs data/*.txt \
  --output results \
  -c research_question="What are participants' experiences of recovery?"
```

The `research_question` variable is injected into templates via `{{research_question}}`.

## Common Issues

**Codes are too similar across chunks**

The consolidation step should deduplicate, but if codes remain too granular, try:
- Larger chunk_size (fewer chunks = fewer duplicate codes)
- More specific instructions in the consolidation template

**Quotes don't verify**

VerifyQuotes fails when:
- LLM paraphrased instead of quoting verbatim
- Quotes span chunk boundaries

Solutions:
- Emphasize "exact quotes" in template
- Use larger chunks to reduce boundary issues
- Review `07_VerifyQuotes_checkquotes/verification.txt` for details

**Too many/few themes**

Adjust the template instruction:

```
Review and consolidate into ~5 (+/- 1) overarching themes.  # Fewer themes
```

**Out of memory**

- Process fewer files at once
- Reduce `chunk_size`
- Lower `MAX_CONCURRENCY`: `export MAX_CONCURRENCY=5`

## Next Steps

- [Pre-extraction Workflow](pre-extract-workflow.md) - Filter text before analysis
- [Customizing Your Analysis](../tutorials/customizing-analysis.md) - Adapt prompts
- [Node Types](../explanation/node-types.md) - Understand Map/Reduce/Transform
