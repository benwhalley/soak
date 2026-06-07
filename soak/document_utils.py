"""Text extraction utilities for PDF, Word, text documents, and spreadsheets.

Document extraction strategy:
- pandoc (via pypandoc) for: .docx, .rtf, .epub, .txt, .md, .markdown
- docling for: .pdf, .pptx (markdown with structure, layout-aware)
- trafilatura with pandoc fallback for: .html, .htm

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
import pypandoc

logger = logging.getLogger(__name__)


def _harden_xml_parsers() -> None:
    """Disable external-entity / DTD resolution in stdlib XML parsers (anti-XXE).

    Untrusted office documents (DOCX/XLSX/PPTX/EPUB) and HTML are parsed
    transitively by openpyxl, docling and others. defusedxml.defuse_stdlib()
    monkeypatches xml.etree, xml.sax, xml.dom and pyexpat so external entities,
    DTD retrieval and entity-expansion bombs are refused process-wide. lxml-based
    parsers (trafilatura HTML, python-pptx) already default to resolve_entities
    off / HTML parsing that ignores entity definitions; this covers the stdlib
    paths defusedxml can reach. Best-effort -- never block import if unavailable.
    """
    import defusedxml

    defusedxml.defuse_stdlib()


_harden_xml_parsers()


# plain text extensions - read directly without pandoc
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",  # plain text / markdown
    ".log",  # log files
    ".rst",  # reStructuredText
}

# subtitle/transcript formats - cue-timing markers stripped
SUBTITLE_EXTENSIONS = {".vtt", ".srt"}

# email message formats
EMAIL_EXTENSIONS = {".eml", ".msg"}

# binary document extensions - need pandoc or special handling
BINARY_DOCUMENT_EXTENSIONS = {".docx", ".rtf", ".pdf", ".pptx", ".epub"}

# docling handles structural extraction for these
DOCLING_EXTENSIONS = {".pdf", ".pptx", ".eml"}

# HTML extensions - extracted with trafilatura (readability) or pandoc
HTML_EXTENSIONS = {".html", ".htm"}

# all supported document extensions for text extraction
DOCUMENT_EXTENSIONS = (
    TEXT_EXTENSIONS
    | SUBTITLE_EXTENSIONS
    | EMAIL_EXTENSIONS
    | BINARY_DOCUMENT_EXTENSIONS
    | HTML_EXTENSIONS
)

# PDF extraction backends. Docling handles layout-aware markdown with
# table extraction; the dispatcher / kwarg remain in place so additional
# backends can be slotted in later without breaking callers.
PDF_BACKEND_DOCLING = "docling"
PDF_BACKENDS = (PDF_BACKEND_DOCLING,)


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


_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")


def _visible_text_length(html: str) -> int:
    """Approximate the length of human-visible text in an HTML document.

    Strips script/style blocks first (their contents shouldn't count as
    content), then drops all remaining tags. Whitespace is collapsed so
    the result roughly tracks what a reader would actually see.
    """
    cleaned = _SCRIPT_STYLE_RE.sub(" ", html)
    cleaned = _TAG_RE.sub(" ", cleaned)
    return len(re.sub(r"\s+", " ", cleaned).strip())


def _extract_html_text(
    path: Path,
    use_readability: bool = True,
    fallback_min_chars: int = 200,
    fallback_min_recall: float = 0.01,
) -> str:
    """Extract main content from an HTML file as markdown.

    With use_readability=True (default), trafilatura identifies the article
    body and emits markdown. We then sanity-check the output and fall back
    to pandoc on the full document when extraction looks broken:

    - **fallback_min_chars**: absolute minimum extracted character count.
      Anything shorter is treated as a miss regardless of input size.
    - **fallback_min_recall**: minimum share of the page's *visible text*
      (after stripping `<script>`, `<style>` and remaining tags) that the
      extractor must keep. A value of `0.01` means trafilatura has to hold
      onto at least 1% of what a reader would see; less than that and we
      assume the heuristic latched onto a sidebar/menu. The ratio check
      only applies when the visible text is large enough (>1000 chars) to
      make the ratio meaningful.

    These thresholds are intentionally loose -- they catch hard failures
    (e.g. database-results pages where readability returns just the
    navigation) without rejecting genuinely short articles.

    With use_readability=False, runs pandoc on the full HTML document --
    deterministic, no heuristics, but you keep whatever nav/footer/ads
    are in the source.
    """
    html = _read_text_file(path)

    if use_readability:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            include_formatting=True,
        )
        ext_len = len((extracted or "").strip())
        visible_len = _visible_text_length(html)
        too_small = ext_len < fallback_min_chars
        low_recall = (
            visible_len > 1000 and ext_len < visible_len * fallback_min_recall
        )
        if extracted and not (too_small or low_recall):
            return extracted
        logger.info(
            "trafilatura kept %d chars vs %d visible-text chars (min=%d, "
            "recall>=%.3f); falling back to pandoc",
            ext_len,
            visible_len,
            fallback_min_chars,
            fallback_min_recall,
        )

    extra_args = ["--wrap=none", "--strip-comments"]
    pandoc_data_home = os.environ.get("PANDOC_DATA_HOME")
    if pandoc_data_home:
        extra_args.append(f"--data-dir={pandoc_data_home}")
    return pypandoc.convert_text(
        html, to="gfm", format="html", extra_args=extra_args
    )


_SUBTITLE_TIMING_RE = re.compile(r"-->")
_SUBTITLE_SEQ_RE = re.compile(r"^\d+$")
_SUBTITLE_TAG_RE = re.compile(r"<[^>]+>")


def _extract_subtitle_text(path: Path) -> str:
    """Strip cue timings/sequence numbers from a VTT or SRT file.

    Returns the dialogue text with timing markers, cue identifiers and
    inline styling tags removed. Useful for qualitative analysis where
    the spoken content matters but the timing noise doesn't.
    """
    raw = _read_text_file(path)
    out = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "WEBVTT" or stripped.startswith("WEBVTT "):
            continue
        if _SUBTITLE_TIMING_RE.search(stripped):
            continue
        if _SUBTITLE_SEQ_RE.match(stripped):
            continue
        if stripped.startswith("NOTE ") or stripped == "NOTE":
            continue
        out.append(_SUBTITLE_TAG_RE.sub("", stripped))
    return "\n".join(out)




def _extract_msg_text(path: Path) -> str:
    """Extract subject/from/to/body from an Outlook .msg file."""
    import extract_msg

    msg = extract_msg.Message(str(path))

    header_lines = []
    if msg.subject:
        header_lines.append(f"**Subject:** {msg.subject}")
    if msg.sender:
        header_lines.append(f"**From:** {msg.sender}")
    if msg.to:
        header_lines.append(f"**To:** {msg.to}")
    if msg.date:
        header_lines.append(f"**Date:** {msg.date}")

    body = msg.body or ""
    if not body and msg.htmlBody:
        html = msg.htmlBody
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="ignore")
        body = pypandoc.convert_text(html, to="gfm", format="html",
                                      extra_args=["--wrap=none"])

    return "\n".join(header_lines + ["", body]) if header_lines else body


_DOCLING_CONVERTER = None


def _get_docling_converter():
    """Lazy-initialise a shared DocumentConverter.

    Constructing the converter loads layout/table models from disk (or
    downloads them on first run). Reusing one instance across calls
    avoids repeated model load on every document.

    OCR is disabled by default. Image-only / scanned PDFs will extract
    as empty text. Re-enable per-call if and when a scanned-PDF backend
    is wired up; for now the assumption is born-digital input.
    """
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_opts = PdfPipelineOptions(do_ocr=False)
        _DOCLING_CONVERTER = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            }
        )
    return _DOCLING_CONVERTER


def _extract_with_docling(path: Path) -> str:
    """Extract document text as markdown via docling.

    Handles PDF and PPTX with layout-aware structural extraction
    (headings, lists, tables). For PDF, OCR is applied to image-only
    pages when the docling OCR pipeline is enabled.
    """
    converter = _get_docling_converter()
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


def _extract_pdf_text(path: Path, backend: str = PDF_BACKEND_DOCLING) -> str:
    """Dispatch PDF extraction to the requested backend.

    Only docling is wired today; the `backend` kwarg exists so future
    backends can be added without changing the call sites.
    """
    if backend not in PDF_BACKENDS:
        logger.warning(
            "Unknown PDF backend %r; falling back to %s.", backend, PDF_BACKEND_DOCLING
        )
    return _extract_with_docling(path)


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


class ZipSecurityError(Exception):
    """A zip archive failed a security check (path traversal, symlink, or zip bomb).

    Raised before any extraction occurs so a malicious archive never touches disk.
    """


# default ceiling on total uncompressed size of a zip archive (200 MB).
# override with SOAK_MAX_UNCOMPRESSED_ZIP_BYTES to raise/lower for trusted inputs.
DEFAULT_MAX_UNCOMPRESSED_ZIP_BYTES = 200 * 1024 * 1024


def _max_uncompressed_zip_bytes() -> int:
    return int(
        os.environ.get(
            "SOAK_MAX_UNCOMPRESSED_ZIP_BYTES", DEFAULT_MAX_UNCOMPRESSED_ZIP_BYTES
        )
    )


def safer_extract(zip_ref, dest_dir, max_files: int = 1000):
    """Safely extract zip archive with path traversal, symlink, and zip-bomb checks.

    All checks run against the central directory before any member is written, so a
    hostile archive is rejected without writing to disk.

    Raises:
        ZipSecurityError: If the zip contains too many files, unsafe paths, symlinks,
            or its total uncompressed size exceeds the configured ceiling (a potential
            zip bomb).
    """
    members = zip_ref.infolist()

    if len(members) > max_files:
        raise ZipSecurityError(
            f"Zip contains too many files ({len(members)} > {max_files})"
        )

    max_uncompressed = _max_uncompressed_zip_bytes()
    total_uncompressed = 0

    for member in members:
        # avoid path traversal
        dest_path = os.path.abspath(os.path.join(dest_dir, member.filename))
        if not dest_path.startswith(os.path.abspath(dest_dir)):
            raise ZipSecurityError(f"Unsafe path in zip: {member.filename}")

        # block symlinks
        is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
        if is_symlink:
            raise ZipSecurityError(f"Symlink found in zip: {member.filename}")

        # guard against zip bombs: cap total uncompressed size
        total_uncompressed += member.file_size
        if total_uncompressed > max_uncompressed:
            raise ZipSecurityError(
                f"Zip uncompressed size exceeds limit "
                f"({total_uncompressed} > {max_uncompressed} bytes); "
                f"possible zip bomb. Raise SOAK_MAX_UNCOMPRESSED_ZIP_BYTES to allow."
            )

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


def extract_text(
    path: str,
    html_use_readability: bool = True,
    html_fallback_min_chars: int = 200,
    html_fallback_min_recall: float = 0.01,
    pdf_backend: str = PDF_BACKEND_DOCLING,
) -> Union[str, List[Dict[str, Any]]]:
    """Extract text from document, converting to GitHub-Flavoured Markdown.

    Supports:
    - Documents (.docx, .rtf, .epub, .txt, .md, .markdown, .pdf, .pptx)
    - HTML (.html, .htm) -- main content via trafilatura by default,
      with pandoc fallback when extraction looks suspicious
    - Spreadsheets (.csv, .xlsx, .xls) -- returns list of row dictionaries

    Args:
        path: Path to the document file
        html_use_readability: For HTML files, extract main article content
            via trafilatura. If False, convert the full HTML via pandoc.
        html_fallback_min_chars: Trafilatura outputs shorter than this trigger
            the pandoc fallback. See `_extract_html_text` for details.
        html_fallback_min_recall: Minimum share of the page's visible text
            that trafilatura must keep before its output is trusted.
        pdf_backend: For PDF files, which extractor to use. Only "docling"
            (default, layout-aware markdown with table extraction) is
            currently supported. The kwarg is kept for forward
            compatibility with future backends.

    Returns:
        Extracted markdown text (str) or list of row dictionaries for spreadsheets
    """
    path_obj = Path(path)

    # spreadsheets return structured data
    if is_spreadsheet(path_obj):
        return extract_spreadsheet_rows(path)

    # documents return markdown text
    mtime = path_obj.stat().st_mtime
    return strip_null_bytes(
        _extract_text_cached(
            str(path),
            mtime,
            html_use_readability,
            html_fallback_min_chars,
            html_fallback_min_recall,
            pdf_backend,
        )
    )


def _extract_text_cached(
    path: str,
    mtime: float,
    html_use_readability: bool = True,
    html_fallback_min_chars: int = 200,
    html_fallback_min_recall: float = 0.01,
    pdf_backend: str = PDF_BACKEND_DOCLING,
) -> str:
    """Extract text from document as GFM. Cached by mtime.

    Uses:
    - pandoc for: .docx, .rtf, .epub
    - direct read for: .txt, .md, .markdown, .log, .rst
    - docling for: .pdf, .pptx, .eml
    - custom cleaner for: .vtt, .srt (docling drops cues with unclosed
      <v Speaker> voice tags, which are common in Zoom/Teams transcripts)
    - extract-msg for: .msg (no docling backend)
    - trafilatura (with pandoc fallback) for: .html, .htm
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix not in DOCUMENT_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {suffix}")

    if suffix == ".pdf":
        text = _extract_pdf_text(path_obj, backend=pdf_backend)
    elif suffix in (".pptx", ".eml"):
        text = _extract_with_docling(path_obj)
    elif suffix == ".msg":
        text = _extract_msg_text(path_obj)
    elif suffix in SUBTITLE_EXTENSIONS:
        text = _extract_subtitle_text(path_obj)
    elif suffix in HTML_EXTENSIONS:
        text = _extract_html_text(
            path_obj,
            use_readability=html_use_readability,
            fallback_min_chars=html_fallback_min_chars,
            fallback_min_recall=html_fallback_min_recall,
        )
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
        ".pptx": "PowerPoint",
        ".epub": "EPUB",
        ".eml": "Email",
        ".msg": "Outlook Message",
        ".vtt": "WebVTT Subtitles",
        ".srt": "SRT Subtitles",
        ".txt": "Text File",
        ".md": "Markdown",
        ".markdown": "Markdown",
        ".html": "HTML",
        ".htm": "HTML",
        ".csv": "CSV Spreadsheet",
        ".xlsx": "Excel Spreadsheet",
        ".xls": "Excel Spreadsheet",
    }

    return type_map.get(suffix, "Unknown")
