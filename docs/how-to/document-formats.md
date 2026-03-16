---
layout: default
title: Supported Document Formats
parent: How-to Guides
nav_order: 1
---

# Supported Document Formats

soak extracts text from common document formats and converts everything to Markdown for consistent processing. This page explains which formats are supported and how extraction works.

## Supported formats

| Format | Extension | Extraction method |
|--------|-----------|-------------------|
| Word | `.docx` | pandoc |
| RTF | `.rtf` | pandoc |
| Plain text | `.txt` | pandoc |
| Markdown | `.md`, `.markdown` | pandoc |
| PDF | `.pdf` | pdfplumber |
| CSV | `.csv` | pandas (structured) |
| Excel | `.xlsx`, `.xls` | pandas (structured) |

## Requirements

**pandoc** must be installed for document extraction (Word, RTF, text, Markdown):

```bash
# macOS
brew install pandoc

# Ubuntu/Debian
sudo apt install pandoc

# Windows
choco install pandoc
```

PDF extraction uses pdfplumber (installed automatically with soak).

## How extraction works

### Documents → Markdown

Word, RTF, text, and Markdown files are converted to GitHub-Flavoured Markdown (GFM) using pandoc. This preserves:

- Headings and structure
- Lists (bulleted and numbered)
- Bold, italic, and other emphasis
- Tables
- Links and footnotes

Example conversion from a Word document:

```
Original DOCX:
  [Heading 1] Interview with Participant 5
  [Bold] Date: [/Bold] 15 March 2024
  [Bullet] First point
  [Bullet] Second point

Extracted Markdown:
  # Interview with Participant 5

  **Date:** 15 March 2024

  - First point
  - Second point
```

### PDF extraction

PDFs are processed with pdfplumber to extract embedded text only:

- No OCR (scanned documents won't work)
- No layout reconstruction
- Paragraph breaks preserved where detectable
- Page breaks become double newlines

For scanned PDFs, convert to searchable PDF first using OCR software.

### Whitespace normalisation

All extracted text is normalised:

- Windows line endings (`\r\n`) converted to Unix (`\n`)
- Multiple spaces collapsed to single space
- Three or more consecutive newlines collapsed to two

## Usage examples

### Single file

```bash
soak run thematic interview.docx -o analysis
```

### Multiple files

```bash
soak run thematic interviews/*.rtf -o analysis
```

### Mixed formats

```bash
soak run thematic data/interview1.docx data/interview2.pdf notes.txt -o analysis
```

### ZIP archives

soak can extract files from ZIP archives:

```bash
soak run thematic interviews.zip -o analysis
```

Files inside the ZIP are extracted to a temporary directory, processed, then cleaned up.

## Troubleshooting

### "pandoc not found"

Install pandoc using your system package manager (see Requirements above).

### RTF files have no paragraph breaks

Old RTF files may use non-standard paragraph markers. If paragraphs run together, try opening in Word/LibreOffice and re-saving as `.docx`.

### PDF extraction returns empty text

The PDF likely contains scanned images rather than embedded text. Use OCR software to create a searchable PDF first.

### Encoding errors

soak expects UTF-8 encoded files. For files with other encodings, convert first:

```bash
iconv -f ISO-8859-1 -t UTF-8 input.txt > input_utf8.txt
```

## Programmatic access

You can use the extraction functions directly in Python:

```python
from soak.document_utils import extract_text, get_supported_extensions

# Check supported formats
print(get_supported_extensions())
# ['.docx', '.rtf', '.txt', '.md', '.markdown', '.pdf', '.csv', '.xlsx', '.xls']

# Extract text from a document
text = extract_text("interview.docx")
print(text)  # Markdown string

# Spreadsheets return structured data
rows = extract_text("survey.csv")
print(rows)  # List of dicts, one per row
```

## Format-specific notes

### Word (.docx)

- Comments and tracked changes are ignored
- Document metadata is stripped
- Headers/footers are included in extraction
- Embedded images are ignored (text only)

### RTF

- Structural elements preserved where possible
- Complex formatting may be simplified
- Older RTF variants may have reduced fidelity

### Plain text (.txt)

- Treated as Markdown for normalisation
- No formatting interpretation
- UTF-8 encoding expected

### Markdown (.md, .markdown)

- Passed through with whitespace normalisation only
- Frontmatter (YAML headers) preserved
- Valid GFM output guaranteed

### PDF

- Text extraction only, no OCR
- Reading order follows PDF structure
- Multi-column layouts may interleave incorrectly
- Tables extracted as plain text (no structure)

## See also

- [Working with Spreadsheet Data](working-with-spreadsheet-data.md) -- CSV and Excel processing
- [Pre-extraction Workflow](pre-extract-workflow.md) -- Filter and prepare text before analysis
