"""Automated qualitative analysis using language models."""

import nltk

# ensure NLTK punkt tokenizer data is available
# wrapped in error handling to avoid race conditions when multiple workers start simultaneously
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except (FileExistsError, OSError):
        # another process downloaded it concurrently -- verify it's now available
        nltk.data.find("tokenizers/punkt_tab")
