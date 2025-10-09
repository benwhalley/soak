"""Base models and utilities for qualitative analysis pipelines."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
from decouple import config as env_config
from joblib import Memory
from pydantic import BaseModel, Field, constr
from struckdown import LLMCredentials
from struckdown import get_embedding as get_embedding_
from struckdown.return_type_models import ACTION_LOOKUP

logger = logging.getLogger(__name__)

SOAK_MAX_RUNTIME = 60 * 30  # 30 mins

# Caching for embeddings
memory = Memory(Path(".embeddings"), verbose=0)


@memory.cache
def get_embedding(*args, **kwargs):
    return get_embedding_(*args, **kwargs)


# Concurrency settings
MAX_CONCURRENCY = env_config("MAX_CONCURRENCY", default=20, cast=int)
semaphore = anyio.Semaphore(MAX_CONCURRENCY)


# Exception classes for backward compatibility
class CancelledRun(Exception):
    """Exception raised when a flow run is cancelled."""

    pass


class Cancelled(Exception):
    """Exception raised when a task is cancelled."""

    pass


# Type definitions
CodeSlugStr = constr(min_length=12, max_length=128)


# Base models for qualitative analysis
class Code(BaseModel):
    slug: CodeSlugStr = Field(
        ...,
        description="A very short abbreviated unique slug/reference for this Code (MAX 20 ascii characters).",
    )
    name: str = Field(..., min_length=1, description="A short name for the code.")
    description: str = Field(
        ..., min_length=5, description="A description of the code."
    )
    quotes: List[str] = Field(
        ...,
        min_length=0,
        description="Example quotes from the text which illustrate the code. Choose the best examples.",
    )


class CodeList(BaseModel):
    codes: List[Code] = Field(..., min_length=0)

    def to_markdown(self):
        return "\n\n".join(
            [f"- {i.name}: {i.description}\n{i.quotes}" for i in self.codes]
        )


class Theme(BaseModel):
    name: str = Field(..., min_length=10)
    description: str = Field(..., min_length=10)
    # refer to codes by slug/identifier
    code_slugs: List[CodeSlugStr] = Field(
        ...,
        min_length=0,
        max_length=64,
        description="A List of the code-references that are part of this theme. Identify them accurately by slug/hash code from the text. Each code slug MUST BE no more than 20 ascii characters. Only refer to codes in the input text above.",
    )


class Themes(BaseModel):
    themes: List[Theme] = Field(..., min_length=1)

    def to_markdown(self):
        return "\n- ".join([i.name for i in self.themes])


@dataclass
class Document:
    """Represents a source document (transcript)."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedItem:
    """Wrapper for content with provenance metadata.

    Tracks the source document and any splitting operations through the pipeline.

    Examples:
        "A.txt" -> original document with source_id="A"
        "A__0" -> first chunk after splitting A
        "A__0__2" -> third sub-chunk after splitting A__0
    """

    content: str
    source_id: str  # e.g., "A", "A__0", "A__0__2"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return content for template rendering."""
        return self.content

    def __repr__(self) -> str:
        """Return content for representation."""
        return self.content

    @property
    def lineage(self) -> List[str]:
        """Return list of source IDs showing full lineage.

        Example: "A__0__2" -> ["A", "A__0", "A__0__2"]
        """
        if "__" not in self.source_id:
            return [self.source_id]

        parts = self.source_id.split("__")
        lineage = [parts[0]]
        for i in range(1, len(parts)):
            lineage.append("__".join(parts[: i + 1]))
        return lineage

    @property
    def depth(self) -> int:
        """Return nesting depth (number of splits from original)."""
        return len(self.lineage) - 1

    @property
    def safe_id(self) -> str:
        """Return source_id safe for filesystem use."""
        return self.source_id.replace("/", "_").replace("\\", "_")

    @property
    def root_document(self) -> str:
        """Return root document name (first part of source_id before splits).

        Example: "A__0__2" -> "A"
        """
        return (
            self.source_id.split("__")[0] if "__" in self.source_id else self.source_id
        )

    def get_export_metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary suitable for export (CSV/JSON rows).

        Returns:
            Dict with keys: item_id, document, filename (optional), index (optional)
        """
        metadata = {
            "item_id": self.source_id,
            "document": self.root_document,
        }

        # Add filename if present in metadata
        if "filename" in self.metadata:
            metadata["filename"] = self.metadata["filename"]
        elif "original_path" in self.metadata:
            metadata["filename"] = self.metadata["original_path"]

        # Add index if present
        if "index" in self.metadata:
            metadata["index"] = self.metadata["index"]

        return metadata

    @staticmethod
    def extract_source_id(item: Any) -> str:
        """Extract source_id from TrackedItem, Box with tracked_item, or return 'unknown'."""
        if isinstance(item, TrackedItem):
            return item.source_id
        elif hasattr(item, "tracked_item") and isinstance(
            item.tracked_item, TrackedItem
        ):
            return item.tracked_item.source_id
        elif hasattr(item, "source_id"):
            return item.source_id
        return "unknown"

    @staticmethod
    def make_safe_id(source_id: str) -> str:
        """Make a source_id safe for filesystem use."""
        return source_id.replace("/", "_").replace("\\", "_")

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


class QualitativeAnalysis(BaseModel):
    label: Optional[str] = None
    name: Optional[str] = None
    codes: Optional[List[Code]] = Field(default_factory=list)
    themes: Optional[List[Theme]] = Field(default_factory=list)
    narrative: Optional[str] = None
    quotes: Optional[Any] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)

    pipeline: Optional[str] = None

    def theme_text_for_comparison(self):
        return [i.name for i in self.themes]

    def sha256(self):
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
    SOAK_ACTION_LOOKUP = dict(ACTION_LOOKUP.copy())
    SOAK_ACTION_LOOKUP.update(
        {
            "theme": Theme,
            "code": Code,
            "themes": Themes,
            "codes": CodeList,
        }
    )
    return SOAK_ACTION_LOOKUP


def safe_json_dump(obj: Any, indent: int = 2) -> str:
    """Safely serialize objects to JSON, handling ChatterResult and Pydantic models."""
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
    """Extract text content from TrackedItem, str, or dict."""
    if hasattr(obj, "content"):
        return obj.content
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, dict) and "content" in obj:
        return obj["content"]
    return None


def extract_prompt(chatter_result) -> Optional[str]:
    """Extract prompt from ChatterResult."""
    if not chatter_result or not hasattr(chatter_result, "results"):
        return None

    if chatter_result.results:
        first_seg = next(iter(chatter_result.results.values()), None)
        return getattr(first_seg, "prompt", None) if first_seg else None

    return None


def get_default_llm_credentials():
    return LLMCredentials()
