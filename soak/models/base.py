"""Base models and utilities for qualitative analysis pipelines."""

import base64
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

import anyio
from decouple import config as env_config
from joblib import Memory
from pydantic import BaseModel, Field, constr
from struckdown import ChatterResult, LLMCredentials
from struckdown import get_embedding as get_embedding_
from struckdown.response_types import ResponseTypes
from struckdown.return_type_models import ResponseModel

logger = logging.getLogger(__name__)

# Configuration: Quote hash length (base32 encoding)
QUOTE_HASH_LENGTH = int(os.getenv("QUOTE_HASH_LENGTH", "6"))

SOAK_MAX_RUNTIME = 60 * 30  # 30 mins

# Caching for embeddings
memory = Memory(Path(".embeddings"), verbose=0)


@memory.cache
def get_embedding(*args, **kwargs):
    """Wrapper for struckdown get_embedding with cached embeddings."""
    return get_embedding_(*args, **kwargs)


# Concurrency settings
MAX_CONCURRENCY = env_config("SOAK_MAX_CONCURRENCY", default=20, cast=int)
semaphore = anyio.Semaphore(MAX_CONCURRENCY)


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

    batches: Union[List[List[ChatterResult]], List[List[str]]]

    def __iter__(self):
        return iter(self.batches)


# Type definitions
CodeSlugStr = constr(min_length=12, max_length=128)


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
        description="A very short abbreviated unique slug/reference for this Code (MAX 20 ascii characters).",
    )
    name: str = Field(..., min_length=1, description="A short name for the code.")
    description: str = Field(
        ..., min_length=5, description="A description of the code."
    )

    # LLM generates Union[Quote, QuoteReference]
    # Pydantic handles discriminated union via 'type' field
    quotes: List[Union[Quote, QuoteReference]] = Field(
        default_factory=list,
        description="Example quotes from the text which illustrate the code. Choose the best examples.",
    )

    # Post-processed field (excluded from LLM schema)
    resolved_quotes: Optional[List[Dict]] = PostProcessedField(default=None)

    def __str__(self):
        quotes_text = "".join([f"""\n- {i}""" for i in self.all_quotes])
        return f"""Code: `{self.slug}`: {self.name}
Description: {self.description}
Supporting quotes: {quotes_text}"""

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


@ResponseTypes.register("codes")
class CodeList(BaseModel):
    codes: List[Code] = Field(..., min_length=0)

    # def to_markdown(self):
    #     return "\n\n".join(
    #         [f"- {i.name}: {i.description}\n{[str(q) for q in i.quotes]}" for i in self.codes]
    #     )
    def __str__(self):
        return "\n\n".join([str(i) for i in self.codes])

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
        max_length=64,
        description="A List of the code-references that are part of this theme. Identify them accurately by slug/hash code from the text. Each code slug MUST BE no more than 20 ascii characters. Only refer to codes in the input text above.",
    )

    # Post-processed fields (excluded from LLM schema)
    resolved_code_refs: Optional[List[Dict]] = PostProcessedField(default=None)

    @property
    def resolved_codes(self) -> List[Code]:
        """Get Code objects matched from context."""
        if self.resolved_code_refs:
            return [Code(**c) for c in self.resolved_code_refs]
        return []

    def post_process(self, context: Dict[str, Any]):
        """Post-process code_slugs to match actual Codes."""
        from soak.models.utils import post_process_theme_code_refs

        post_process_theme_code_refs(self, context)


@ResponseTypes.register("themes")
class Themes(BaseModel):
    themes: List[Theme] = Field(..., min_length=1)

    # def to_markdown(self):
    #     return "\n- ".join([i.name for i in self.themes])

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


@dataclass
class TrackedItem:
    """Wrapper for content with provenance metadata.

    Tracks items through the pipeline with unique IDs and source provenance.

    Fields:
        content: str (for documents) or ChatterResult (for LLM outputs)
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
        TrackedItem(content=ChatterResult(...), id="doc1__chunks__0__coded", sources=["doc1__chunks__0"])  # After map
        TrackedItem(content="combined", id="all_codes", sources=["doc1__coded", "doc2__coded"])  # After reduce
    """

    content: Union[
        str, Any
    ]  # str for documents, ChatterResult for LLM outputs, Any for flexibility
    id: str  # THIS item's unique ID
    sources: List[str]  # IDs that contributed to this item
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return content for template rendering."""
        return str(self.content)

    def __repr__(self) -> str:
        """Return representation showing ID and sources."""
        return f"TrackedItem(id={self.id!r}, sources={self.sources!r})"

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
        result = asdict(self)

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
    template_text: Optional[str] = None,
    ground_truth_columns: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """Order DataFrame columns: identifiers → template-used → ground_truth → outputs → other metadata.

    Args:
        df: DataFrame to reorder
        output_keys: List of output field names (classification results, etc.)
        template_text: Optional template to detect which metadata columns were used
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
    if template_text:
        try:
            from jinja2 import Environment, meta

            env = Environment()
            ast = env.parse(template_text)
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

    def by_comparisons(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a nested dict keyed by comparison key, with inner dict containing:
            - 'a', 'b' (the QualitativeAnalysis objects)
            - 'stats'
            - 'plots': { plot_type: plot_object, ... }
        """
        out = {}
        for key, (a, b) in self.combinations.items():
            out[key] = {"a": a, "b": b, "stats": self.statistics.get(key), "plots": {}}

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
    """Serialize to JSON, handling ChatterResult and Pydantic models."""
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


def extract_prompt(chatter_result) -> Optional[str]:
    """Extract prompt text from ChatterResult.results."""
    if not chatter_result or not hasattr(chatter_result, "results"):
        return None

    if chatter_result.results:
        first_seg = next(iter(chatter_result.results.values()), None)
        return getattr(first_seg, "prompt", None) if first_seg else None

    return None


def get_default_llm_credentials():
    """Create LLMCredentials from environment variables (LLM_API_KEY, LLM_API_BASE)."""
    return LLMCredentials()
