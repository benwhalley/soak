"""Text extraction utilities for PDF, Word, text documents, and spreadsheets.

Document extraction strategy:
- pandoc (via pypandoc) for: .docx, .rtf, .txt, .md, .markdown
- pdfplumber for: .pdf (text extraction only, no OCR)

All document formats are converted to GitHub-Flavoured Markdown (GFM).
"""

import glob
import logging
import os
import re
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import pdfplumber
import pypandoc

logger = logging.getLogger(__name__)


# plain text extensions - read directly without pandoc
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown",  # plain text / markdown
    ".vtt", ".srt",              # subtitles (video transcripts)
    ".log",                      # log files
    ".rst",                      # reStructuredText
}

# binary document extensions - need pandoc or special handling
BINARY_DOCUMENT_EXTENSIONS = {".docx", ".rtf", ".pdf"}

# all supported document extensions for text extraction
DOCUMENT_EXTENSIONS = TEXT_EXTENSIONS | BINARY_DOCUMENT_EXTENSIONS


def normalise_whitespace(text: str) -> str:
    """Normalise whitespace in extracted text.

    - Convert Windows/old Mac line endings to Unix
    - Collapse multiple spaces/tabs to single space
    - Collapse 3+ newlines to double newline
    - Strip leading/trailing whitespace
    """
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_text_file(path: Path) -> str:
    """Read plain text file directly, trying common encodings."""
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # last resort: read with errors ignored
    return path.read_text(encoding="utf-8", errors="ignore")


def _convert_with_pandoc(path: Path) -> str:
    """Convert document to GFM using pandoc.

    For .docx/.rtf: uses pandoc to preserve structure.
    For text files (.txt, .md, .vtt, .srt, etc.): reads directly.
    """
    suffix = path.suffix.lower()

    # plain text files: read directly, skip pandoc overhead
    if suffix in TEXT_EXTENSIONS:
        return _read_text_file(path)

    # .docx, .rtf: use pandoc for actual format conversion
    extra_args = ["--wrap=none", "--strip-comments"]
    pandoc_data_home = os.environ.get("PANDOC_DATA_HOME")
    if pandoc_data_home:
        extra_args.append(f"--data-dir={pandoc_data_home}")

    return pypandoc.convert_file(
        str(path),
        to="gfm",
        format=None,  # let pandoc auto-detect
        extra_args=extra_args,
    )


def _extract_pdf_text(path: Path) -> str:
    """Extract text from PDF using pdfplumber.

    - Extracts embedded text only (no OCR)
    - Does not attempt layout or column reconstruction
    - Preserves paragraph breaks where detectable
    """
    with pdfplumber.open(path) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)


def resolve_path_with_package_data(path_pattern: str) -> list[str]:
    """Resolve a file path or glob pattern, checking package data if not found locally.

    Args:
        path_pattern: File path or glob pattern (e.g., 'data/cfs/*.txt')

    Returns:
        List of resolved file paths

    Raises:
        FileNotFoundError: If pattern matches no files in either location
    """
    # First try current working directory
    matches = glob.glob(path_pattern)

    if matches:
        return matches

    # If no matches, try package data directory
    package_dir = Path(__file__).parent
    package_pattern = str(package_dir / path_pattern)
    package_matches = glob.glob(package_pattern)

    if package_matches:
        logger.info(f"Using package data files: {path_pattern}")
        return package_matches

    # No matches in either location
    raise FileNotFoundError(
        f"No files found matching '{path_pattern}' in current directory or package data"
    )


def strip_null_bytes(obj):
    """Recursively strip null bytes from strings in nested structures."""
    if isinstance(obj, str):
        return obj.replace("\u0000", "")
    elif isinstance(obj, dict):
        return {strip_null_bytes(k): strip_null_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_null_bytes(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(strip_null_bytes(i) for i in obj)
    elif isinstance(obj, set):
        return {strip_null_bytes(i) for i in obj}
    else:
        return obj


def get_scrubber(salt, model="en_core_web_md"):
    """Create PII scrubber with spaCy NER and strict email detection.

    Returns:
        Configured scrubadub.Scrubber with hashed replacements
    """

    import scrubadub
    from scrubadub.detectors import EmailDetector
    from scrubadub.post_processors import FilthReplacer, PrefixSuffixReplacer
    from scrubadub_spacy.detectors import SpacyEntityDetector

    class StrictEmailDetector(EmailDetector):
        """Only match proper RFC-style emails, not things like 'vague @ times'."""

        name = "strict_email"
        regex = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.UNICODE
        )

    scrubber = scrubadub.Scrubber(
        post_processor_list=[
            FilthReplacer(include_hash=True, hash_salt=salt, hash_length=4),
            PrefixSuffixReplacer(prefix="[", suffix="]"),
        ],
        detector_list=[],  # start empty to avoid default detectors
    )
    spacy_detector = SpacyEntityDetector(model=model)
    scrubber.add_detector(spacy_detector)
    scrubber.add_detector(StrictEmailDetector())

    return scrubber


def safer_extract(zip_ref, dest_dir, max_files: int = 1000):
    """Safely extract zip archive with path traversal and symlink checks.

    Raises:
        Exception: If zip contains too many files, unsafe paths, or symlinks
    """
    members = zip_ref.infolist()

    if len(members) > max_files:
        raise Exception(f"Zip contains too many files ({len(members)} > {max_files})")

    for member in members:
        # Avoid path traversal
        dest_path = os.path.abspath(os.path.join(dest_dir, member.filename))
        if not dest_path.startswith(os.path.abspath(dest_dir)):
            raise Exception(f"Unsafe path in zip: {member.filename}")

        # Block symlinks
        is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
        if is_symlink:
            raise Exception(f"Symlink found in zip: {member.filename}")

    zip_ref.extractall(dest_dir)


@contextmanager
def unpack_zip_to_temp_paths_if_needed(
    paths: list[str | Path],
) -> list[tuple[str, dict]]:
    """Unpack zip files to temp dir and yield file paths with metadata.

    Returns:
        List of (file_path, metadata) tuples. Metadata includes zip_source and zip_path.
        Temp dirs are cleaned up on context exit.
    """
    # Convert all paths to strings for consistent handling
    paths = [str(p) for p in paths]

    expanded_items = []
    temp_dirs = []

    try:
        for path in paths:
            # Check if it's a zip file (in current dir or package data)
            if path.endswith(".zip"):
                zip_path = None
                if os.path.isfile(path):
                    zip_path = path
                else:
                    # Try package data
                    package_dir = Path(__file__).parent
                    package_zip = package_dir / path
                    if package_zip.is_file():
                        zip_path = str(package_zip)
                        logger.info(f"Using package data zip: {path}")

                if zip_path:
                    zip_stem = Path(zip_path).stem  # "archive.zip" -> "archive"
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        tmpdir = tempfile.mkdtemp(prefix="unpacked_zip_")
                        temp_dirs.append(tmpdir)
                        safer_extract(zip_ref, tmpdir)
                        for root, _, files in os.walk(tmpdir):
                            for f in files:
                                file_path = os.path.join(root, f)
                                metadata = {
                                    "zip_source": zip_stem,
                                    "zip_path": zip_path,
                                }
                                expanded_items.append((file_path, metadata))
                    continue

                # Zip file not found
                logger.warning(f"Zip file not found: {path}")
            else:
                # Expand globs, checking package data if not found locally
                try:
                    resolved_paths = resolve_path_with_package_data(path)
                    for expanded_path in resolved_paths:
                        metadata = {"zip_source": None, "zip_path": None}
                        expanded_items.append((expanded_path, metadata))
                except FileNotFoundError:
                    # If no matches found, add the original path anyway
                    # This allows proper error messages from downstream code
                    logger.warning(f"No files found for pattern: {path}")
                    # Don't add anything - will cause empty list if all patterns fail

        yield expanded_items

    finally:
        for tmpdir in temp_dirs:
            shutil.rmtree(tmpdir, ignore_errors=True)


def is_spreadsheet(path: Union[str, Path]) -> bool:
    """Check if file is spreadsheet (CSV, XLS, or XLSX) by extension."""
    suffix = Path(path).suffix.lower()
    return suffix in [".csv", ".xlsx", ".xls"]


def extract_spreadsheet_rows(path: str) -> List[Dict[str, Any]]:
    """Extract rows from CSV or XLSX file as list of dictionaries.

    Each row becomes a dictionary with column names as keys.
    NaN values are converted to None.

    Args:
        path: Path to CSV or XLSX file

    Returns:
        List of dictionaries, one per row (excluding header)
    """
    suffix = Path(path).suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".xlsx":
            df = pd.read_excel(path, engine="openpyxl")
        elif suffix == ".xls":
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported spreadsheet format: {suffix}")

        # Convert NaN to None, convert to list of dicts
        rows = df.where(pd.notna(df), None).to_dict("records")

        logger.info(
            f"Loaded {len(rows)} rows from {path} with columns: {list(df.columns)}"
        )
        return rows

    except Exception as e:
        logger.error(f"Failed to read spreadsheet {path}: {e}")
        raise


def extract_text(path: str) -> Union[str, List[Dict[str, Any]]]:
    """Extract text from document, converting to GitHub-Flavoured Markdown.

    Supports:
    - Documents (.docx, .rtf, .txt, .md, .markdown, .pdf)
    - Spreadsheets (.csv, .xlsx, .xls) -- returns list of row dictionaries

    Args:
        path: Path to the document file

    Returns:
        Extracted markdown text (str) or list of row dictionaries for spreadsheets
    """
    path_obj = Path(path)

    # spreadsheets return structured data
    if is_spreadsheet(path_obj):
        return extract_spreadsheet_rows(path)

    # documents return markdown text
    mtime = path_obj.stat().st_mtime
    return strip_null_bytes(_extract_text_cached(str(path), mtime))


def _extract_text_cached(path: str, mtime: float) -> str:
    """Extract text from document as GFM. Cached by mtime.

    Uses:
    - pandoc for: .docx, .rtf, .txt, .md, .markdown
    - pdfplumber for: .pdf
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix not in DOCUMENT_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {suffix}")

    if suffix == ".pdf":
        text = _extract_pdf_text(path_obj)
    else:
        text = _convert_with_pandoc(path_obj)

    return normalise_whitespace(text)


def get_supported_extensions() -> list[str]:
    """Return list of supported file extensions for documents and spreadsheets."""
    return list(DOCUMENT_EXTENSIONS) + [".csv", ".xlsx", ".xls"]


def is_supported_file(path: Path) -> bool:
    """Check if file is supported for text extraction."""
    suffix = path.suffix.lower()
    return suffix in DOCUMENT_EXTENSIONS or suffix in {".csv", ".xlsx", ".xls"}


def detect_file_type(path: Path) -> str:
    """Detect document type for logging."""
    suffix = path.suffix.lower()

    type_map = {
        ".pdf": "PDF",
        ".docx": "Word Document",
        ".rtf": "RTF Document",
        ".txt": "Text File",
        ".md": "Markdown",
        ".markdown": "Markdown",
        ".csv": "CSV Spreadsheet",
        ".xlsx": "Excel Spreadsheet",
        ".xls": "Excel Spreadsheet",
    }

    return type_map.get(suffix, "Unknown")
