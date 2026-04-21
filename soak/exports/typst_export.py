"""Typst/PDF export functionality for qualitative analysis results."""

import importlib.metadata
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jinja2 import Environment, FileSystemLoader, StrictUndefined

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
SOAK_GITHUB_URL = "https://github.com/benwhalley/soak"


def get_soak_version() -> str:
    """Get the current soak version."""
    try:
        return importlib.metadata.version("soaking")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def escape_typst(text: str) -> str:
    """Escape special Typst characters in user content.

    Typst uses these characters with special meaning:
    - # for function calls
    - [ ] for content blocks
    - * for bold
    - _ for emphasis (in some contexts)
    - $ for math mode
    - @ for references
    - \\ for escape sequences
    - < > for labels
    """
    if not text:
        return ""
    text = str(text)
    text = text.replace("\\", "\\\\")
    text = text.replace("#", "\\#")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace("*", "\\*")
    text = text.replace("$", "\\$")
    text = text.replace("@", "\\@")
    text = text.replace("<", "\\<")
    text = text.replace(">", "\\>")
    text = text.replace("_", "\\_")
    return text


def markdown_to_typst(text: str) -> str:
    """Convert markdown formatting to Typst formatting.

    Handles:
    - **bold** -> *bold*
    - *italic* -> _italic_
    - # Heading -> = Heading
    - - list items (same in Typst)
    - Paragraphs (blank lines)

    Also escapes remaining special characters that aren't part of formatting.
    """
    if not text:
        return ""

    text = str(text)

    # escape characters that should never be interpreted (before conversion)
    text = text.replace("\\", "\\\\")
    text = text.replace("$", "\\$")
    text = text.replace("@", "\\@")
    text = text.replace("<", "\\<")
    text = text.replace(">", "\\>")

    # use unique placeholders to avoid double-conversion
    BOLD_START = "\x00B\x00"
    BOLD_END = "\x00/B\x00"
    ITALIC_START = "\x00I\x00"
    ITALIC_END = "\x00/I\x00"

    # convert markdown bold **text** to placeholder
    text = re.sub(
        r"\*\*(.+?)\*\*", lambda m: f"{BOLD_START}{m.group(1)}{BOLD_END}", text
    )

    # convert markdown bold __text__ to placeholder
    text = re.sub(r"__(.+?)__", lambda m: f"{BOLD_START}{m.group(1)}{BOLD_END}", text)

    # convert markdown italic *text* to placeholder (single asterisks only)
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        lambda m: f"{ITALIC_START}{m.group(1)}{ITALIC_END}",
        text,
    )

    # convert markdown italic _text_ to placeholder (single underscores)
    text = re.sub(
        r"(?<!_)_(?!_)([^_]+?)(?<!_)_(?!_)",
        lambda m: f"{ITALIC_START}{m.group(1)}{ITALIC_END}",
        text,
    )

    # replace placeholders with Typst syntax
    text = text.replace(BOLD_START, "*")
    text = text.replace(BOLD_END, "*")
    text = text.replace(ITALIC_START, "_")
    text = text.replace(ITALIC_END, "_")

    # convert markdown headers to typst headers (process longer ones first)
    text = re.sub(r"^#### (.+)$", r"==== \1", text, flags=re.MULTILINE)
    text = re.sub(r"^### (.+)$", r"=== \1", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"== \1", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"= \1", text, flags=re.MULTILINE)

    # markdown links [text](url) -> #link("url")[text]
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'#link("{m.group(2)}")[{m.group(1)}]',
        text,
    )

    return text


def render_typst_template(template_name: str, context: dict) -> str:
    """Render a Typst template with Jinja2.

    Args:
        template_name: Name of template file in templates/
        context: Variables to inject into template

    Returns:
        Rendered Typst source code
    """
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        autoescape=False,
    )
    env.filters["e"] = escape_typst
    env.filters["escape_typst"] = escape_typst
    env.filters["md"] = markdown_to_typst
    env.filters["markdown"] = markdown_to_typst
    template = env.get_template(template_name)
    return template.render(**context)


def compile_typst_to_pdf(
    typst_source: str, output_path: Path, images_dir: Optional[Path] = None
) -> None:
    """Compile Typst source to PDF.

    Args:
        typst_source: Typst source code
        output_path: Output path for PDF file
        images_dir: Optional directory containing images referenced in the source
    """
    import typst

    if images_dir:
        source_path = images_dir / "source.typ"
        source_path.write_text(typst_source)
        root_dir = str(images_dir)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".typ", delete=False) as f:
            f.write(typst_source)
            source_path = Path(f.name)
        root_dir = None

    try:
        pdf_bytes = typst.compile(str(source_path), root=root_dir)
        output_path.write_bytes(pdf_bytes)
    finally:
        if not images_dir and source_path.exists():
            source_path.unlink()


class RunPdfExporter:
    """Export analysis results as PDF using Typst.

    Works with QualitativeAnalysis objects or dicts from JSON deserialization.
    """

    def __init__(
        self,
        analysis: Union["QualitativeAnalysis", Dict[str, Any]],
        pipeline_name: str = "Analysis",
        model_name: str = "",
        doc_count: int = 0,
        word_count: int = 0,
        run_name: Optional[str] = None,
        template: str = "default",
        model_details: Optional[List[Dict]] = None,
        embedding_model: Optional[Dict] = None,
        tldr: str = "",
    ):
        """Initialise the exporter.

        Args:
            analysis: QualitativeAnalysis object or dict with themes/codes/narrative
            pipeline_name: Name of the pipeline used
            model_name: LLM model used (simple string, used if model_details not provided)
            doc_count: Number of documents analysed
            word_count: Total word count
            run_name: Optional run name (defaults to pipeline_name)
            template: Template style - "default" or "apa" for APA 7th edition format
            model_details: List of model dicts with alias, name, model_name, provider
            embedding_model: Dict with name, model_name, provider for embedding model
            tldr: One-line summary of the analysis
        """
        self.analysis = analysis
        self.pipeline_name = pipeline_name
        self.model_name = model_name
        self.doc_count = doc_count
        self.word_count = word_count
        self.run_name = run_name or pipeline_name
        self.template = template
        self.model_details = model_details or []
        self.embedding_model = embedding_model or {}
        self.tldr = tldr

    def _get_themes(self) -> List[Dict]:
        """Extract themes from analysis."""
        if hasattr(self.analysis, "themes"):
            themes = self.analysis.themes
            if hasattr(themes, "__iter__"):
                return [
                    t.model_dump() if hasattr(t, "model_dump") else t for t in themes
                ]
        elif isinstance(self.analysis, dict):
            return self.analysis.get("themes", [])
        return []

    def _get_codes(self) -> List[Dict]:
        """Extract codes from analysis."""
        if hasattr(self.analysis, "codes"):
            codes = self.analysis.codes
            if hasattr(codes, "__iter__"):
                return [
                    c.model_dump() if hasattr(c, "model_dump") else c for c in codes
                ]
        elif isinstance(self.analysis, dict):
            return self.analysis.get("codes", [])
        return []

    def _get_narrative(self) -> str:
        """Extract narrative from analysis."""
        if hasattr(self.analysis, "narrative"):
            return self.analysis.narrative or ""
        elif isinstance(self.analysis, dict):
            return self.analysis.get("narrative", "") or ""
        return ""

    def _get_meta_narrative(self) -> str:
        """Extract meta-narrative from analysis."""
        if hasattr(self.analysis, "meta_narrative"):
            return self.analysis.meta_narrative or ""
        elif isinstance(self.analysis, dict):
            return self.analysis.get("meta_narrative", "") or ""
        return ""

    def _build_context(self) -> dict:
        """Build template context from analysis data."""
        themes = self._get_themes()
        codes = self._get_codes()

        # process themes for template
        theme_list = []
        for theme in themes:
            codes_list = theme.get(
                "codes", theme.get("code_hashes", theme.get("code_slugs", []))
            )
            codes_set = set(codes_list)
            quotes = []
            for code in codes:
                code_slug = code.get("slug", "")
                code_name = code.get("name", "").lower().replace(" ", "-")
                from soak.models.base import compute_code_hash
                code_hash = compute_code_hash(code.get("name", ""), code.get("description", ""))
                if code_slug in codes_set or code_name in codes_set or code_hash in codes_set:
                    for q in code.get("quotes", [])[:2]:
                        text = q.get("text", str(q)) if isinstance(q, dict) else str(q)
                        quotes.append(text)
                        if len(quotes) >= 3:
                            break
                if len(quotes) >= 3:
                    break

            theme_list.append(
                {
                    "name": theme.get("name", ""),
                    "description": theme.get("description", ""),
                    "codes": codes_list,
                    "quotes": quotes,
                }
            )

        # process codes for template
        code_list = []
        for code in codes:
            quote_list = []
            for q in code.get("quotes", []):
                if isinstance(q, dict):
                    quote_list.append({"text": q.get("text", str(q))})
                else:
                    quote_list.append({"text": str(q)})

            code_list.append(
                {
                    "name": code.get("name", ""),
                    "description": code.get("description", ""),
                    "quotes": quote_list,
                }
            )

        methods_text = self._generate_methods_text(themes, codes)

        # compute self-similarity matrix if we have multiple themes
        self_similarity = None
        if len(themes) > 1:
            try:
                from ..analysis.self_similarity import \
                    compute_self_similarity_matrix

                self_similarity = compute_self_similarity_matrix(themes)
            except ImportError:
                logger.debug("Self-similarity module not available")
            except Exception as e:
                logger.warning(f"Could not compute self-similarity: {e}")

        return {
            "version": get_soak_version(),
            "github_url": SOAK_GITHUB_URL,
            "run_name": self.run_name,
            "pipeline_name": self.pipeline_name,
            "model_name": self.model_name,
            "model_details": self.model_details,
            "embedding_model": self.embedding_model,
            "completed_at": "",
            "doc_count": self.doc_count,
            "word_count": self.word_count,
            "duration": "",
            "cost": "",
            "theme_count": len(themes),
            "code_count": len(codes),
            "themes": theme_list,
            "codes": code_list,
            "tldr": self.tldr,
            "narrative": self._get_narrative(),
            "meta_narrative": self._get_meta_narrative(),
            "methods_text": methods_text,
            "self_similarity": self_similarity,
            "sample_n": None,
            "seed": None,
            "prompt_tokens": "",
            "completion_tokens": "",
            "completion_ratio": "",
        }

    def _generate_methods_text(self, themes: List[Dict], codes: List[Dict]) -> str:
        """Generate auto-generated methods section."""
        version = get_soak_version()
        total_quotes = sum(len(code.get("quotes", [])) for code in codes)

        lines = [
            "METHODS",
            "",
            f"Qualitative analysis was conducted using Soak v{version} ({SOAK_GITHUB_URL}),",
            f"an LLM-assisted thematic analysis tool. The analysis used the {self.pipeline_name} pipeline",
            (
                f"with {self.model_name} as the language model."
                if self.model_name
                else "."
            ),
            "",
            f"{self.doc_count} documents ({self.word_count:,} words) were analysed. The analysis",
            f"identified {len(themes)} themes and {len(codes)} codes, with {total_quotes} supporting",
            "quotations extracted from the source material.",
        ]

        return "\n".join(lines)

    def generate_pdf(self) -> bytes:
        """Generate PDF from analysis results.

        Returns:
            PDF file contents as bytes.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # copy logo to temp directory
            logo_src = TEMPLATES_DIR / "soak-logo.svg"
            if logo_src.exists():
                shutil.copy(logo_src, temp_path / "soak-logo.svg")

            context = self._build_context()

            # select template based on style
            if self.template == "apa":
                template_name = "run_results_apa.typ"
            else:
                template_name = "run_results.typ"

            typst_source = render_typst_template(template_name, context)

            output_path = temp_path / "output.pdf"
            compile_typst_to_pdf(typst_source, output_path, images_dir=temp_path)

            return output_path.read_bytes()


def export_analysis_pdf(
    analysis: Union["QualitativeAnalysis", Dict[str, Any]],
    output_path: Path,
    pipeline_name: str = "Analysis",
    model_name: str = "",
    doc_count: int = 0,
    word_count: int = 0,
    run_name: Optional[str] = None,
    template: str = "default",
    model_details: Optional[List[Dict]] = None,
    embedding_model: Optional[Dict] = None,
    tldr: str = "",
) -> None:
    """Export analysis results to PDF file.

    Convenience function that creates a RunPdfExporter and writes to file.

    Args:
        analysis: QualitativeAnalysis object or dict
        output_path: Path to write PDF file
        pipeline_name: Name of the pipeline used
        model_name: LLM model used (simple string, used if model_details not provided)
        doc_count: Number of documents analysed
        word_count: Total word count
        run_name: Optional run name
        template: Template style - "default" or "apa"
        model_details: List of model dicts with alias, name, model_name, provider
        embedding_model: Dict with name, model_name, provider for embedding model
        tldr: One-line summary of the analysis
    """
    exporter = RunPdfExporter(
        analysis=analysis,
        pipeline_name=pipeline_name,
        model_name=model_name,
        doc_count=doc_count,
        word_count=word_count,
        run_name=run_name,
        template=template,
        model_details=model_details,
        embedding_model=embedding_model,
        tldr=tldr,
    )
    pdf_bytes = exporter.generate_pdf()
    output_path.write_bytes(pdf_bytes)
