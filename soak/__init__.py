"""Automated qualitative analysis using language models."""

import nltk

# ensure NLTK punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
