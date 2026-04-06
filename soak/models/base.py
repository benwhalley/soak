"""Base models and utilities for qualitative analysis pipelines."""

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import anyio
from decouple import config as env_config
from joblib import Memory
from pydantic import BaseModel, ConfigDict, Field, constr
from struckdown import (StruckdownResult, LLMCredentials, get_embedding,
                        get_embedding_async)
from struckdown.response_types import ResponseTypes
from struckdown.return_type_models import ResponseModel

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# Configuration: Quote hash length (base32 encoding)
QUOTE_HASH_LENGTH = int(os.getenv("QUOTE_HASH_LENGTH", "8"))

# Jinja2 environment for __str__ templates
_str_template_dir = Path(__file__).resolve().parent.parent / "templates" / "__str__"
_str_env = Environment(
    loader=FileSystemLoader(_str_template_dir),
    keep_trailing_newline=False,
)


def render_str_template(template_name: str, context: Dict[str, Any]) -> str:
    """Render a str_template by name (e.g. 'Code.jinja')."""
    return _str_env.get_template(template_name).render(context)

SOAK_MAX_RUNTIME = 60 * 30  # 30 mins

# Caching for embeddings
# verbose=0 by default, but will show cache info at DEBUG level via our logging
memory = Memory(Path(".embeddings"), verbose=0)


# Concurrency settings
MAX_CONCURRENCY = env_config("SOAK_MAX_CONCURRENCY", default=20, cast=int)
_semaphore = None
_max_concurrency_value = MAX_CONCURRENCY


def get_semaphore() -> anyio.Semaphore:
    """Get the shared concurrency semaphore (lazy initialisation)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = anyio.Semaphore(MAX_CONCURRENCY)
    return _semaphore


def get_max_concurrency() -> int:
    """Get the current maximum concurrency value."""
    return _max_concurrency_value


def set_max_concurrency(n: int) -> None:
    """Set the maximum concurrency for LLM calls. Replaces the semaphore."""
    global _semaphore, _max_concurrency_value
    _semaphore = anyio.Semaphore(n)
    _max_concurrency_value = n


# Exception classes for backward compatibility
class CancelledRun(Exception):
    """Exception raised when a flow run is cancelled."""

    pass


class Cancelled(Exception):
    """Exception raised when a task is cancelled."""

    pass


class QuoteProvenanceError(Exception):
    """Exception raised when quotes cannot determine their source provenance.

    This typically occurs when trying to extract quotes in a Map node after a Reduce,
    where the context has multiple sources. Use QuoteReference instead of Quote
    in these situations.
    """

    pass


class BatchList(BaseModel):
    """Container for batched results."""

    batches: Union[List[List[StruckdownResult]], List[List[str]]]

    def __iter__(self):
        return iter(self.batches)


# Type definitions
CodeSlugStr = constr(min_length=8, max_length=150, pattern=r"^[a-zA-Z0-9-]+$")


# Helper for post-processed fields
def PostProcessedField(**kwargs):
    """Field populated by post-processing, excluded from LLM schema."""
    kwargs.setdefault("json_schema_extra", {})
    kwargs["json_schema_extra"]["exclude_from_completion"] = True
    return Field(**kwargs)


# Quote and QuoteReference classes for provenance tracking
class Quote(BaseModel):
    """Quote with source document tracking.

    A Quote always comes from a single source (chunk). For quotes that reference
    multiple sources, use QuoteReference with a hash pointing to an existing Quote.
    """

    type: Literal["quote"] = "quote"  # Discriminator for Union
    text: str
    source: str = ""  # Single source ID (quotes always from one chunk)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def hash(self) -> str:
        """Generate base32 hash from text + source (lowercase, case-insensitive)."""
        content = f"{self.text}|{self.source}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        # Base32 encode, take first N chars, lowercase
        return base64.b32encode(hash_bytes).decode("ascii")[:QUOTE_HASH_LENGTH].lower()

    def __str__(self) -> str:
        """Render as 'hash: text' for templates."""
        return f"{self.hash()}: '{self.text}'"


class QuoteReference(BaseModel):
    """Reference to existing quote by hash."""

    type: Literal["quotereference"] = "quotereference"  # Discriminator for Union
    hash: str = Field(
        ...,
        min_length=QUOTE_HASH_LENGTH,
        max_length=QUOTE_HASH_LENGTH,
        description=f"{QUOTE_HASH_LENGTH}-character lowercase base32 hash matching a Quote",
    )

    def __str__(self) -> str:
        """Render as just the hash for templates."""
        return "ref: " + self.hash


# Type alias for Union
QuoteOrRef = Union[Quote, QuoteReference]


# Base models for qualitative analysis
@ResponseTypes.register("code")
class Code(ResponseModel):
    slug: CodeSlugStr = Field(
        ...,
        description="A very short abbreviated unique slug/reference for this Code. MAXIMUM of 40 ascii characters, never more.",
    )
    name: str = Field(..., min_length=1, description="A short name for the code.")
    description: str = Field(
        ..., min_length=5, description="A description of the code."
    )

    # LLM generates Union[Quote, QuoteReference]
    # pydantic handles discriminated union via 'type' field
    quotes: List[Union[Quote, QuoteReference]] = Field(
        default_factory=list,
        max_length=24,
        description="Example quotes from the text which illustrate the code. Choose the best examples. Maximum 24 quotes allowed, but typically fewer--be selective.",
    )

    # post-processed field (excluded from LLM schema)
    resolved_quotes: Optional[List[Dict]] = PostProcessedField(default=None)
    
    def __str__(self):
        return render_str_template("Code.jinja", {"code": self})

    @property
    def all_quotes(self) -> List[Quote]:
        """Get all quotes as Quote objects (resolves references)."""
        if self.resolved_quotes:
            return [Quote(**q) for q in self.resolved_quotes]
        # Fallback: convert quotes directly
        return [q for q in self.quotes if isinstance(q, Quote)]

    @property
    def source_ids(self) -> Set[str]:
        """Get all unique source IDs from quotes."""
        return {q.source for q in self.all_quotes if q.source}

    def post_process(self, context: Dict[str, Any]):
        """Post-process quotes: resolve QuoteReferences and capture source in quotes."""
        from soak.models.utils import post_process_code_quotes

        post_process_code_quotes(self, context)

    @classmethod
    def customize_schema_for_context(
        cls, schema: dict, context: Dict[str, Any]
    ) -> dict:
        """Customize quotes field: use QuoteReference when rationalizing."""
        use_refs = _has_code_inputs(context)

        if use_refs and "properties" in schema and "quotes" in schema["properties"]:
            quotes_schema = schema["properties"]["quotes"]
            if "items" in quotes_schema:
                quotes_schema["items"] = (
                    QuoteReference.__pydantic_model__().model_json_schema()
                )

        return schema


def _has_code_inputs(context: Dict[str, Any]) -> bool:
    """Private utility: check if context contains Code/CodeList objects."""

    def check_value(val) -> bool:
        if isinstance(val, (Code, CodeList)):
            return True
        if isinstance(val, list):
            return any(check_value(item) for item in val)
        return False

    return any(check_value(val) for val in context.values())


class CodeList(BaseModel):
    """Container for a list of Code objects. Internal wrapper, not an LLM return type."""

    codes: List[Code] = Field(..., min_length=0)

    def __str__(self):
        return render_str_template("CodeList.jinja", {"codes": self.codes})

    def __iter__(self):
        return iter(self.codes)

    def __len__(self):
        return len(self.codes)

    def post_process(self, context: Dict[str, Any]):
        """Post-process each code in the list."""
        for code in self.codes:
            code.post_process(context)


@ResponseTypes.register("theme")
class Theme(ResponseModel):
    name: str = Field(..., min_length=10)
    description: str = Field(..., min_length=10)
    # refer to codes by slug/identifier
    code_slugs: List[CodeSlugStr] = Field(
        ...,
        min_length=0,
        max_length=60,
        description="A List of the code-references that are part of this theme. Identify them accurately by the slug/hash from the text. Each code slug MUST BE no more than 40 ascii characters. Alphanumeric only, no spaces. Only refer to codes in the input text above. An absolute MAXIMUM of 60 codes is allowed per theme, but normally far fewer--be selective.",
    )

    def __str__(self):
        return render_str_template("Theme.jinja", {"theme": self})

    # Post-processed fields (excluded from LLM schema)
    resolved_code_refs: Optional[List[Dict]] = PostProcessedField(default=None)
    label: Optional[str] = PostProcessedField(default=None)

    @property
    def resolved_codes(self) -> List[Code]:
        """Get Code objects matched from context."""
        if self.resolved_code_refs:
            return [Code(**c) for c in self.resolved_code_refs]
        return []

    def set_label(self, template: str, index: int) -> None:
        """Set label from template string with index, name, and description.

        Args:
            template: Format string with {i}, {name}, {description} placeholders
            index: 1-based index for this theme
        """
        self.label = template.format(
            i=index, name=self.name, description=self.description
        )

    def post_process(self, context: Dict[str, Any]):
        """Post-process code_slugs to match actual Codes."""
        from soak.models.utils import post_process_theme_code_refs

        post_process_theme_code_refs(self, context)


class Themes(BaseModel):
    """Container for a list of Theme objects. Internal wrapper, not an LLM return type."""

    themes: List[Theme] = Field(..., min_length=1)

    def __str__(self):
        return render_str_template("Themes.jinja", {"themes": self.themes})

    def __iter__(self):
        return iter(self.themes)

    def __len__(self):
        return len(self.themes)

    def post_process(self, context: Dict[str, Any]):
        """Post-process each theme in the list."""
        for theme in self.themes:
            theme.post_process(context)


@dataclass
class Document:
    """Represents a source document (transcript)."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrackedItem(BaseModel):
    """Wrapper for content with provenance metadata.

    Tracks items through the pipeline with unique IDs and source provenance.

    Fields:
        content: str (for documents) or StruckdownResult (for LLM outputs)
        id: Unique identifier for THIS item
        sources: List of source item IDs that contributed to this (provenance trail)
        metadata: Additional metadata (filename, etc.)

    ID generation:
        - Original docs: filename (e.g., "doc1")
        - After node: "{parent_id}__{node_name}__{index}" for indexed items
        - After reduce: "{node_name}" (simple, since multiple sources)

    Examples:
        TrackedItem(content="text", id="doc1", sources=["doc1"])  # Original document
        TrackedItem(content="chunk", id="doc1__chunks__0", sources=["doc1"])  # After split
        TrackedItem(content=StruckdownResult(...), id="doc1__chunks__0__coded", sources=["doc1__chunks__0"])  # After map
        TrackedItem(content="combined", id="all_codes", sources=["doc1__coded", "doc2__coded"])  # After reduce
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: Union[
        str, Any
    ]  # str for documents, StruckdownResult for LLM outputs, Any for flexibility
    id: str  # THIS item's unique ID
    sources: List[str]  # IDs that contributed to this item
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_excluding_overlap: Optional[Tuple[int, int]] = (
        None  # (start, end) indices for core content without overlap
    )

    def __str__(self) -> str:
        """Return content for template rendering (includes overlap for LLM context)."""
        return str(self.content)

    def get_core_content(self) -> str:
        """Return content excluding overlap regions.

        Returns content[start:end] if content_excluding_overlap is set,
        otherwise returns full content. Use this when joining chunks
        to avoid duplicating overlapped content.
        """
        if self.content_excluding_overlap is not None and isinstance(self.content, str):
            start, end = self.content_excluding_overlap
            return self.content[start:end]
        return str(self.content)

    def __repr__(self) -> str:
        """Return representation showing ID and sources."""
        return f"TrackedItem(id={self.id!r}, sources={self.sources!r})"

    def __hash__(self) -> int:
        """Hash based on id for use in sets and as dict keys."""
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """Equality based on id."""
        if isinstance(other, TrackedItem):
            return self.id == other.id
        return NotImplemented

    @property
    def lineage(self) -> List[str]:
        """List of IDs showing full lineage (e.g., "A__0__2" -> ["A", "A__0", "A__0__2"])."""
        if "__" not in self.id:
            return [self.id]

        parts = self.id.split("__")
        lineage = [parts[0]]
        for i in range(1, len(parts)):
            lineage.append("__".join(parts[: i + 1]))
        return lineage

    @property
    def depth(self) -> int:
        """Nesting depth (number of splits from original)."""
        return len(self.lineage) - 1

    @property
    def safe_id(self) -> str:
        """ID safe for filesystem (slashes replaced with underscores)."""
        return self.id.replace("/", "_").replace("\\", "_")

    @property
    def root_document(self) -> str:
        """Root document name (first part before splits, e.g., "A__0__2" -> "A")."""
        return self.id.split("__")[0] if "__" in self.id else self.id

    def get_export_metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary suitable for export (CSV/JSON rows).

        Includes ALL metadata fields (e.g., spreadsheet columns).

        Returns:
            Dict with all metadata plus guaranteed item_id, document fields
        """
        # Start with ALL metadata
        metadata = dict(self.metadata) if self.metadata else {}

        # Override/ensure standard fields are set correctly
        metadata.update(
            {
                "item_id": self.id,
                "document": self.root_document,
            }
        )

        # Ensure filename is set (try metadata first, then fallback)
        if "filename" not in metadata and "original_path" in metadata:
            metadata["filename"] = metadata["original_path"]

        return metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert TrackedItem to dict with serializable metadata.

        Automatically converts non-serializable metadata values to strings.

        Returns:
            Dict suitable for JSON serialization
        """
        result = self.model_dump(mode="json")

        # clean metadata to ensure JSON serializability
        if result.get("metadata"):
            clean_metadata = {}
            for k, v in result["metadata"].items():
                # keep simple serializable types as-is
                if isinstance(v, (str, int, float, bool, type(None))):
                    clean_metadata[k] = v
                elif isinstance(v, (list, dict)):
                    # try to serialize, fallback to str if fails
                    try:
                        json.dumps(v)
                        clean_metadata[k] = v
                    except (TypeError, ValueError):
                        clean_metadata[k] = str(v)
                else:
                    # convert other types to string
                    clean_metadata[k] = str(v)
            result["metadata"] = clean_metadata

        return result

    def to_json(self) -> str:
        """Convert TrackedItem to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @staticmethod
    def extract_source_id(item: Any) -> str:
        """Extract ID from TrackedItem/Box or return 'unknown'.

        Note: Name kept as extract_source_id for backward compatibility during transition.
        """
        if isinstance(item, TrackedItem):
            return item.id
        elif hasattr(item, "tracked_item") and isinstance(
            item.tracked_item, TrackedItem
        ):
            return item.tracked_item.id
        elif hasattr(item, "id"):
            return item.id
        return "unknown"

    @staticmethod
    def make_safe_id(item_id: str) -> str:
        """Make ID safe for filesystem (replace slashes)."""
        return item_id.replace("/", "_").replace("\\", "_")

    @staticmethod
    def extract_export_metadata(item: Any, idx: int) -> Dict[str, Any]:
        """Extract export metadata from any item type (TrackedItem, Box, or fallback).

        Args:
            item: Item to extract metadata from
            idx: Index to use as fallback

        Returns:
            Dict with keys: item_id, document, filename (optional), index
        """
        # Try TrackedItem first
        if isinstance(item, TrackedItem):
            meta = item.get_export_metadata()
            meta["index"] = idx
            return meta

        # Try Box with tracked_item
        if hasattr(item, "tracked_item") and isinstance(item.tracked_item, TrackedItem):
            meta = item.tracked_item.get_export_metadata()
            meta["index"] = idx
            return meta

        # Fallback for unknown types
        return {"item_id": f"item_{idx}", "document": f"item_{idx}", "index": idx}


def order_export_columns(
    df: "pd.DataFrame",
    output_keys: Optional[List[str]] = None,
    template: Optional[str] = None,
    ground_truth_columns: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """Order DataFrame columns: identifiers → template-used → ground_truth → outputs → other metadata.

    Args:
        df: DataFrame to reorder
        output_keys: List of output field names (classification results, etc.)
        template: Optional template to detect which metadata columns were used
        ground_truth_columns: Optional list of ground truth column names to place before outputs

    Returns:
        DataFrame with reordered columns
    """
    import pandas as pd

    # Core identifiers (always first)
    core_identifiers = [
        "item_id",
        "document",
        "filename",
        "index",
        "original_path",
        "doc_index",
        "row_index",
    ]

    # Detect template-used columns
    template_used = set()
    if template:
        try:
            from jinja2 import Environment, meta

            env = Environment()
            ast = env.parse(template)
            # Get all variables referenced in template
            all_vars = meta.find_undeclared_variables(ast)
            # Filter to actual column names that exist in df
            template_used = {
                v
                for v in all_vars
                if v in df.columns
                and v not in core_identifiers
                and not v.startswith("__")
            }
        except Exception:
            pass  # Silently fail - column ordering is nice-to-have

    # Build ordered column list
    ordered = []

    # 1. Add core identifiers
    ordered.extend([c for c in core_identifiers if c in df.columns])

    # 2. Add template-used metadata columns
    ordered.extend([c for c in template_used if c not in ordered])

    # 3. Add ground truth columns (before outputs)
    if ground_truth_columns:
        ordered.extend(
            [c for c in ground_truth_columns if c in df.columns and c not in ordered]
        )

    # 4. Add ALL output columns together
    if output_keys:
        ordered.extend([c for c in output_keys if c in df.columns and c not in ordered])

    # 5. Remember where to insert separator
    separator_position = len(ordered)

    # 6. Add remaining columns (other metadata)
    remaining = [c for c in df.columns if c not in ordered]
    ordered.extend(remaining)

    # Create result DataFrame with reordered columns
    result_df = df[ordered].copy()

    # Insert blank separator column before unused metadata
    result_df.insert(separator_position, "", "")

    return result_df


class QualitativeAnalysis(BaseModel):
    """Container for qualitative analysis results including codes, themes, and narrative."""

    label: Optional[str] = None
    name: Optional[str] = None
    codes: Optional[List[Code]] = Field(default_factory=list)
    themes: Optional[List[Theme]] = Field(default_factory=list)
    narrative: Optional[str] = None
    quotes: Optional[Any] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)

    pipeline: Optional[str] = None

    def theme_text_for_comparison(self):
        """Extract theme names for similarity comparison."""
        return [i.name for i in self.themes]

    def sha256(self):
        """Generate short hash of analysis for unique identification."""
        return hashlib.sha256(json.dumps(self.model_dump()).encode()).hexdigest()[:8]

    def model_post_init(self, __context):
        """Set name field if not provided."""
        if self.name is None:
            self.name = self.label or self.sha256()[:8]

    def __str__(self):
        return f"Themes: {self.themes}\nCodes: {self.codes}"


class QualitativeAnalysisComparison(BaseModel):
    results: List["QualitativeAnalysis"]
    combinations: Any  # Dict[str, Tuple["QualitativeAnalysis", "QualitativeAnalysis"]]
    statistics: Dict[str, Dict]
    comparison_plots: Dict[str, Dict[str, Any]]  # eg. heatmaps.xxxyyy = List
    additional_plots: Dict[str, Any]
    config: dict
    embedded_strings: Dict[str, List[Dict[str, str]]] = (
        {}
    )  # analysis_name -> list of theme embedding details

    def by_comparisons(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a nested dict keyed by comparison key, with inner dict containing:
            - 'a', 'b' (the QualitativeAnalysis objects)
            - 'stats'
            - 'plots': { plot_type: plot_object, ... }
            - 'embedded_a', 'embedded_b': lists of embedded string details for easier template access
        """
        out = {}
        for key, (a, b) in self.combinations.items():
            out[key] = {
                "a": a,
                "b": b,
                "stats": self.statistics.get(key),
                "plots": {},
                "embedded_a": self.embedded_strings.get(a.name, []),
                "embedded_b": self.embedded_strings.get(b.name, []),
            }

        for plot_type in self.comparison_plots.keys():
            for k in out.keys():
                out[k]["plots"][plot_type] = self.comparison_plots[plot_type][k]

        return out


# Utility functions
def get_action_lookup():
    """Get ACTION_LOOKUP with soak's registered response types.

    This function is kept for backward compatibility but simply
    returns the lookup from ResponseTypes registry.
    """
    return ResponseTypes.get_lookup()


def safe_json_dump(obj: Any, indent: int = 2) -> str:
    """Serialize to JSON, handling StruckdownResult and Pydantic models."""
    try:
        # Handle Pydantic models
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(mode="json"), indent=indent)
        # Handle regular dicts/lists
        return json.dumps(obj, indent=indent, default=str)
    except Exception as e:
        logger.debug(f"Failed to serialize to JSON: {e}, using repr fallback")
        return json.dumps(
            {"__repr__": repr(obj), "__str__": str(obj), "__error__": str(e)},
            indent=indent,
        )


def extract_content(obj) -> Optional[str]:
    """Extract text content from TrackedItem/str/dict."""
    if hasattr(obj, "content"):
        return obj.content
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, dict) and "content" in obj:
        return obj["content"]
    return None


def extract_prompt(complete_result) -> Optional[str]:
    """Extract prompt text from StruckdownResult.results."""
    if not complete_result or not hasattr(complete_result, "results"):
        return None

    if complete_result.results:
        first_seg = next(iter(complete_result.results.values()), None)
        return getattr(first_seg, "prompt", None) if first_seg else None

    return None


def get_default_llm_credentials():
    """Create LLMCredentials from environment variables (LLM_API_KEY, LLM_API_BASE)."""
    return LLMCredentials()
