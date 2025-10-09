"""Data models for qualitative analysis pipelines."""

import hashlib
import itertools
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (TYPE_CHECKING, Annotated, Any, Callable, Dict, List,
                    Literal, Optional, Set, Tuple, Union)

import anyio
import nltk
import numpy as np
import pandas as pd
import tiktoken
from box import Box
from decouple import config as env_config
from jinja2 import (Environment, FileSystemLoader, StrictUndefined,
                    TemplateSyntaxError, meta)
from joblib import Memory
from pydantic import BaseModel, Field, PrivateAttr, constr, model_validator
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from struckdown import (LLM, ChatterResult, LLMCredentials, chatter,
                        chatter_async)
from struckdown import get_embedding as get_embedding_
from struckdown.parsing import parse_syntax
from struckdown.return_type_models import ACTION_LOOKUP

from .agreement import (export_agreement_stats, kappam_fleiss, kripp_alpha,
                        percent_agreement)
from .agreement_scripts import (collect_field_categories,
                                generate_human_rater_template,
                                write_agreement_scripts)
from .document_utils import (extract_text, get_scrubber,
                             unpack_zip_to_temp_paths_if_needed)
from .export_utils import export_to_csv, export_to_html, export_to_json

if TYPE_CHECKING:
    from .dag import QualitativeAnalysisPipeline


logger = logging.getLogger(__name__)

SOAK_MAX_RUNTIME = 60 * 30  # 30 mins


memory = Memory(Path(".embeddings"), verbose=0)


@memory.cache
def get_embedding(*args, **kwargs):
    return get_embedding_(*args, **kwargs)


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


MAX_CONCURRENCY = env_config("MAX_CONCURRENCY", default=20, cast=int)
semaphore = anyio.Semaphore(MAX_CONCURRENCY)


# exception classes for backward compatibility
class CancelledRun(Exception):
    """Exception raised when a flow run is cancelled."""

    pass


class Cancelled(Exception):
    """Exception raised when a task is cancelled."""

    pass


CodeSlugStr = constr(min_length=12, max_length=128)


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


def get_default_llm_credentials():
    return LLMCredentials()


class DAGConfig(BaseModel):
    document_paths: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]] = []
    documents: List[Union[str, "TrackedItem"]] = []
    model_name: str = "litellm/gpt-4.1-mini"
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words
    extra_context: Dict[str, Any] = {}
    llm_credentials: LLMCredentials = Field(
        default_factory=get_default_llm_credentials, repr=False, exclude=True
    )
    scrub_pii: bool = False
    scrubber_model: str = "en_core_web_md"
    scrubber_salt: Optional[str] = Field(default="42", exclude=True)
    seed: int = 42

    def get_model(self):
        return LLM(model_name=self.model_name)

    def load_documents(self) -> List["TrackedItem"]:
        """Load documents and wrap them in TrackedItem for provenance tracking."""
        if hasattr(self, "documents") and self.documents:
            logger.debug("Using cached documents")
            # Ensure cached docs are TrackedItems
            if self.documents and isinstance(self.documents[0], TrackedItem):
                return self.documents
            # Upgrade cached string documents to TrackedItems
            logger.debug("Upgrading cached documents to TrackedItems")
            self.documents = [
                (
                    TrackedItem(
                        content=doc, source_id=f"doc_{idx}", metadata={"doc_index": idx}
                    )
                    if isinstance(doc, str)
                    else doc
                )
                for idx, doc in enumerate(self.documents)
            ]
            return self.documents

        # Check if document_paths contains tuples (already unpacked) or strings (need unpacking)
        if self.document_paths and isinstance(self.document_paths[0], tuple):
            # Already unpacked by CLI - document_paths contains (path, metadata) tuples
            items = self.document_paths
            texts = [extract_text(path) for path, _ in items]

            # Create TrackedItems with source_id from filename
            tracked_docs = []
            for idx, ((path, path_metadata), text) in enumerate(zip(items, texts)):
                file_stem = Path(path).stem  # "A.txt" -> "A"

                # Construct source_id: include zip name if from zip
                if path_metadata.get("zip_source"):
                    source_id = f"{path_metadata['zip_source']}__{file_stem}"
                else:
                    source_id = file_stem

                # Build metadata
                metadata = {
                    "original_path": str(path),
                    "doc_index": idx,
                    "filename": Path(path).name,
                }

                # Add zip info to metadata if present
                if path_metadata.get("zip_source"):
                    metadata["zip_source"] = path_metadata["zip_source"]
                    metadata["zip_path"] = path_metadata["zip_path"]

                tracked_docs.append(
                    TrackedItem(content=text, source_id=source_id, metadata=metadata)
                )

            self.documents = tracked_docs
        else:
            # Need to unpack - document_paths contains string paths
            with unpack_zip_to_temp_paths_if_needed(self.document_paths) as items:
                # items is now list of (path, metadata) tuples
                texts = [extract_text(path) for path, _ in items]

                # Create TrackedItems with source_id from filename
                tracked_docs = []
                for idx, ((path, path_metadata), text) in enumerate(zip(items, texts)):
                    file_stem = Path(path).stem  # "A.txt" -> "A"

                    # Construct source_id: include zip name if from zip
                    if path_metadata.get("zip_source"):
                        source_id = f"{path_metadata['zip_source']}__{file_stem}"
                    else:
                        source_id = file_stem

                    # Build metadata
                    metadata = {
                        "original_path": str(path),
                        "doc_index": idx,
                        "filename": Path(path).name,
                    }

                    # Add zip info to metadata if present
                    if path_metadata.get("zip_source"):
                        metadata["zip_source"] = path_metadata["zip_source"]
                        metadata["zip_path"] = path_metadata["zip_path"]

                    tracked_docs.append(
                        TrackedItem(
                            content=text, source_id=source_id, metadata=metadata
                        )
                    )

                self.documents = tracked_docs

        if self.scrub_pii:
            logger.debug("Scrubbing PII")
            if self.scrubber_salt == 42:
                logger.warning(
                    "Scrubber salt is default, consider setting to a random value"
                )

            scrubber = get_scrubber(model=self.scrubber_model, salt=self.scrubber_salt)
            # Apply scrubbing to TrackedItem content
            for doc in self.documents:
                if isinstance(doc, TrackedItem):
                    doc.content = scrubber.clean(doc.content)
                    doc.metadata["scrubbed"] = True

        return self.documents


async def run_node(node):
    try:
        result = await node.run()
        logger.debug(f"COMPLETED: {node.name}\n")
        return result
    except Exception as e:
        logger.error(f"Node {node.name} failed: {e}")
        raise e


def get_template_variables(template_string: str) -> Set[str]:
    """Extract all variables from a jinja template string.
    get_template_variables('{a} {{b}} c ')
    """
    env = Environment()
    ast = env.parse(template_string)
    return meta.find_undeclared_variables(ast)


def render_strict_template(template_str: str, context: dict) -> str:
    # try:
    env = Environment(undefined=StrictUndefined)
    template = env.from_string(template_str)
    return template.render(**context)
    # except Exception as e:
    #     import pdb; pdb.set_trace()


@dataclass(frozen=True)
class Edge:
    from_node: str
    to_node: str


DAGNodeUnion = Annotated[
    Union[
        "Map",
        "Reduce",
        "Transform",
        "Batch",
        "Split",
        "TransformReduce",
        "VerifyQuotes",
        "Classifier",
        "Filter",
    ],
    Field(discriminator="type"),
]


class DAG(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    default_context: Dict[str, Any] = {}
    default_config: Dict[str, Union[str, int, float]] = {}

    nodes: List["DAGNodeUnion"] = Field(default_factory=list)
    config: DAGConfig = DAGConfig()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add defaults for config fields
        for k, v in self.default_config.items():
            if hasattr(self.config, k) and k not in self.config.model_fields_set:
                setattr(self.config, k, v)

    @model_validator(mode="after")
    def validate_node_templates(self) -> "DAG":
        """Validate that nodes requiring templates have them defined."""
        # Node types that require templates
        template_required_types = {"Map", "Transform", "Classifier", "Filter"}

        for node in self.nodes:
            if node.type in template_required_types:
                # Check if template_text exists and is not None/empty
                if not hasattr(node, "template_text") or not node.template_text:
                    raise ValueError(
                        f"Node '{node.name}' of type '{node.type}' requires a template, "
                        f"but none was found. Add a template section like '---#{node.name}' "
                        f"in your YAML file."
                    )

        return self

    @property
    def edges(self) -> List["Edge"]:
        all_edges = []
        for node in self.nodes:
            for input_ref in node.inputs:
                if input_ref in [i.name for i in self.nodes]:
                    all_edges.append(Edge(from_node=input_ref, to_node=node.name))

        return all_edges

    def to_mermaid(self) -> str:
        """Generate a Mermaid diagram of the DAG structure with shapes by node type."""
        from soak.visualization import dag_to_mermaid

        return dag_to_mermaid(self)

    def get_execution_order(self) -> List[List[str]]:
        """Get the execution order as batches of nodes that can run in parallel."""
        remaining = set([i.name for i in self.nodes])
        execution_order = []

        while remaining:
            # Find nodes with no unprocessed dependencies
            ready = set()
            for node_name in remaining:
                deps = self.get_dependencies_for_node(node_name)
                if all(dep not in remaining for dep in deps):
                    ready.add(node_name)

            if not ready and remaining:
                # Circular dependency detected
                raise ValueError(f"Circular dependency detected in nodes: {remaining}")

            execution_order.append(list(ready))
            remaining -= ready

        return execution_order

    @property
    def nodes_dict(self):
        return {i.name: i for i in self.nodes}

    def cancel(self):
        if self.cancel_scope is not None:
            self.cancel_scope.cancel()
            logger.warning(f"DAG {self.name} cancelled")

    async def run(self):
        try:
            self.config.load_documents()
            if not self.config.llm_credentials:
                raise Exception("LLMCredentials must be set for DAG")
            for batch in self.get_execution_order():
                # use anyio structured concurrency - start all tasks in batch concurrently
                with anyio.fail_after(SOAK_MAX_RUNTIME):
                    async with anyio.create_task_group() as tg:
                        for name in batch:
                            tg.start_soon(run_node, self.nodes_dict[name])
                # all tasks in batch complete when task group exits
            return self, None
        except Exception as e:
            import traceback

            err = f"DAG execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(err)
            return self, str(e)

    def get_dependencies_for_node(self, node_name: str) -> Set[str]:
        """Get nodes that must complete before a node can run."""

        dependencies = set()

        # set[edge for edge in self.edges if edge.to_node == node_name]
        for edge in self.edges:
            if edge.to_node == node_name:
                dependencies.add(edge.from_node)

        return dependencies

    def add_node(self, node: "DAGNode"):
        # if self.nodes_dict.get(node.name):
        #     raise ValueError(f"Node '{node.name}' already exists in DAG")
        node.dag = self
        self.nodes.append(node)

    def get_required_context_variables(self):
        node_names = [i.name for i in self.nodes]
        tmplts = list(
            itertools.chain(
                *[get_template_variables(i.template) for i in self.nodes if i.template]
            )
        )
        return set(tmplts).difference(node_names)

    def __str__(self):
        return f"DAG: {self.name}"

    def __repr__(self):
        return f"DAG: {self.name}"

    @property
    def context(self) -> Dict[str, Any]:
        """Backward compatibility: return node outputs as dict"""
        results = {v.name: v.output for v in self.nodes if v and v.output is not None}
        conf = self.config.extra_context.copy()
        conf.update(results)
        return conf

    def export_execution(self, output_dir: Path, metadata: Dict[str, Any] = None):
        """Export detailed execution information to a folder structure.

        Args:
            output_dir: Directory to export to
            metadata: Optional metadata to include in meta.txt (e.g., CLI command, runtime info)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Exporting execution details to {output_dir}")

        # Extract unique_id from metadata if provided
        unique_id = metadata.get("unique_id", "") if metadata else ""

        # Write metadata file
        meta_content = f"""DAG Execution Export
====================
DAG Name: {self.name}
Export Time: {datetime.now().isoformat()}

"""
        if metadata:
            meta_content += "Runtime Configuration:\n"
            for key, value in metadata.items():
                meta_content += f"  {key}: {value}\n"

        meta_content += f"\nDefault Context:\n"
        for key, value in self.default_context.items():
            meta_content += f"  {key}: {value}\n"

        meta_content += f"  Documents: {len(self.config.documents)}\n"

        (output_dir / "meta.txt").write_text(meta_content)

        # Get execution order for numbering
        execution_order = self.get_execution_order()

        # Create node_to_order mapping
        node_order = {}
        for batch_idx, batch in enumerate(execution_order):
            for node_name in batch:
                node_order[node_name] = batch_idx + 1

        # Export each node
        for node in self.nodes:
            order = node_order.get(node.name, 0)
            folder_name = f"{order:02d}_{node.type}_{node.name}"
            node_folder = output_dir / folder_name

            try:
                node.export(node_folder, unique_id=unique_id)
                logger.debug(f"  Exported node: {folder_name}")
            except Exception as e:
                logger.error(f"  Failed to export node {node.name}: {e}")
                import traceback

                traceback.print_exc()

        logger.debug(f"Export complete: {output_dir}")


OutputUnion = Union[
    str,
    List[str],
    List[List[str]],
    ChatterResult,
    List[ChatterResult],
    List[List[ChatterResult]],
    # for top matches
    List[Dict[str, Union[str, List[Tuple[str, float]]]]],
    # for multi-model classifier
    Dict[str, List[ChatterResult]],
]


class DAGNode(BaseModel):
    # this used for reserialization
    model_config = {"discriminator": "type"}
    type: str = Field(default_factory=lambda self: type(self).__name__, exclude=False)

    dag: Optional["DAG"] = Field(default=None, exclude=True)

    name: str
    inputs: Optional[List[str]] = []
    template_text: Optional[str] = None
    output: Optional[OutputUnion] = Field(default=None)

    def get_model(self):
        model_name = getattr(self, 'model_name', None)
        if model_name:
            m = LLM(model_name=model_name)
            return m
        return self.dag.config.get_model()

    def validate_template(self):
        try:
            Environment().parse(self.template_text)
            return True
        except TemplateSyntaxError as e:
            raise e
            logger.error(f"Template syntax error: {e}")
            return False

    async def run(self, items: List[Any] = None) -> List[Any]:
        logger.debug(f"\n\nRunning `{self.name}` ({self.__class__.__name__})\n\n")

    @property
    def context(self) -> Dict[str, Any]:
        ctx = self.dag.default_context.copy()

        # merge in extra_context from config (includes persona, research_question, etc.)
        ctx.update(self.dag.config.extra_context)

        # if there are no inputs, assume we'll be using input documents
        if not self.inputs:
            self.inputs = ["documents"]

        if "documents" in self.inputs:
            ctx["documents"] = self.dag.config.load_documents()

        prev_nodes = {k: self.dag.nodes_dict.get(k) for k in self.inputs}
        prev_output = {
            k: v.output for k, v in prev_nodes.items() if v and v.output is not None
        }
        ctx.update(prev_output)

        return ctx

    @property
    def template(self) -> str:
        return self.template_text

    def get_input_items(self) -> Optional[List[Any]]:
        """Get the input items that were processed by this node."""
        if not self.inputs or not self.dag:
            return None

        # Get the first input (most common case)
        input_name = self.inputs[0]
        if input_name == "documents":
            return self.dag.config.documents
        elif input_name in self.dag.nodes_dict:
            input_node = self.dag.nodes_dict[input_name]
            return input_node.output if input_node.output else None

        return None

    def result(self) -> Dict[str, Any]:
        """Extract results from this node. Override in subclasses for node-specific results."""
        metadata = {
            "name": self.name,
            "type": self.type,
            "inputs": self.inputs,
        }

        # Add template text if available
        if hasattr(self, "template_text") and self.template_text:
            metadata["template_text"] = self.template_text

        # Add model info if available
        if hasattr(self, "model_name") and self.model_name:
            metadata["model_name"] = self.model_name
        if hasattr(self, "temperature") and self.temperature is not None:
            metadata["temperature"] = self.temperature

        return {
            "metadata": metadata,
        }

    def export(self, folder: Path, unique_id: str = ""):
        """Export node execution details to a folder. Override in subclasses.

        Args:
            folder: Folder to export to
            unique_id: Optional unique identifier to append to file names
        """
        folder.mkdir(parents=True, exist_ok=True)

        # Write comprehensive node config including all fields and defaults
        config_data = self.model_dump(
            exclude={
                "output",
                "context",
                "dag",
                "_processed_items",
                "_model_results",
                "_agreement_stats",
            },
            exclude_none=False,  # Include fields with None to show defaults
        )

        # Add type info for clarity
        config_text = f"""Node Configuration
==================
Type: {self.type}
Name: {self.name}

Parameters:
"""
        for key, value in sorted(config_data.items()):
            if key not in ["name", "type"]:
                config_text += f"  {key}: {value}\n"

        (folder / "meta.txt").write_text(config_text)

        # Export input items for traceability
        input_items = self.get_input_items()
        if input_items and isinstance(input_items, list):
            inputs_folder = folder / "inputs"
            inputs_folder.mkdir(exist_ok=True)

            for idx, item in enumerate(input_items):
                if isinstance(item, TrackedItem):
                    # Export with source_id in filename
                    (inputs_folder / f"{idx:04d}_{item.safe_id}.txt").write_text(
                        item.content
                    )

                    # Export metadata
                    if item.metadata:
                        (
                            inputs_folder / f"{idx:04d}_{item.safe_id}_metadata.json"
                        ).write_text(json.dumps(item.metadata, indent=2, default=str))
                elif isinstance(item, str):
                    # Backward compatibility
                    (inputs_folder / f"{idx:04d}_input.txt").write_text(item)
                else:
                    # Try to convert to string
                    try:
                        (inputs_folder / f"{idx:04d}_input.txt").write_text(str(item))
                    except Exception as e:
                        logger.warning(f"Failed to export input {idx}: {e}")


class CompletionDAGNode(DAGNode):
    model_name: Optional[str] = None
    temperature: float = 1
    max_tokens: Optional[int] = None

    async def run(self, items: List[Any] = None) -> List[Any]:
        await super().run()

    def output_keys(self) -> List[str]:
        """Return the list of output keys provided by this node."""
        try:
            sections = parse_syntax(self.template)
            keys = []
            for section in sections:
                keys.extend(section.keys())
            return keys or [self.name]
        except Exception as e:
            logger.warning(f"Failed to parse template for output keys: {e}")
            return ["input"]


class Split(DAGNode):
    type: Literal["Split"] = "Split"

    name: str = "chunks"
    template_text: str = "{{input}}"

    chunk_size: int = 20000
    min_split: int = 500
    overlap: int = 0
    split_unit: Literal["chars", "tokens", "words", "sentences", "paragraphs"] = (
        "tokens"
    )
    encoding_name: str = "cl100k_base"

    @property
    def template(self):
        return None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.inputs) > 1:
            raise ValueError("Split node can only have one input")

        if self.chunk_size < self.min_split:
            logger.warning(
                f"Chunk size must be larger than than min_split. Setting min_split to chunk_size // 2 = {self.chunk_size // 2}"
            )
            self.min_split = self.chunk_size // 2

    async def run(self) -> List[Union[str, "TrackedItem"]]:
        import numpy as np

        await super().run()
        input_docs = self.context[self.inputs[0]]

        # Split each document and track provenance
        all_chunks = []
        for doc in input_docs:
            if isinstance(doc, TrackedItem):
                chunks = self.split_tracked_document(doc)
            else:
                # Backward compatibility: wrap plain strings
                temp_tracked = TrackedItem(
                    content=doc, source_id="unknown_doc", metadata={}
                )
                chunks = self.split_tracked_document(temp_tracked)
            all_chunks.extend(chunks)

        self.output = all_chunks

        # Calculate stats on content
        lens = [
            len(
                self.tokenize(
                    chunk.content if isinstance(chunk, TrackedItem) else chunk,
                    method=self.split_unit,
                )
            )
            for chunk in self.output
        ]
        logger.debug(
            f"CREATED {len(self.output)} chunks; average length ({self.split_unit}): {np.mean(lens).round(1)}, max: {max(lens)}, min: {min(lens)}."
        )

        return self.output

    def split_tracked_document(self, doc: TrackedItem) -> List[TrackedItem]:
        """Split a TrackedItem into multiple TrackedItems with nested source_ids.

        The node name is included in the source_id to track which Split node created the chunk.
        Format: parent_source_id__nodename__index
        Example: doc_A__sentences__0 (first chunk from 'sentences' Split node)
        """
        text_chunks = self.split_document(doc.content)

        tracked_chunks = []
        for idx, chunk in enumerate(text_chunks):
            # Include node name in source_id for tracking which split created this
            new_source_id = f"{doc.source_id}__{self.name}__{idx}"

            tracked_chunks.append(
                TrackedItem(
                    content=chunk,
                    source_id=new_source_id,
                    metadata={
                        **doc.metadata,
                        "split_index": idx,
                        "split_node": self.name,
                        "parent_source_id": doc.source_id,
                    },
                )
            )

        return tracked_chunks

    def split_document(self, doc: str) -> List[str]:
        """Split a document string into chunks (backward compatibility)."""
        if self.split_unit == "chars":
            return self._split_by_length(doc, len_fn=len, overlap=self.overlap)

        tokens = self.tokenize(doc, method=self.split_unit)
        spans = self._compute_spans(
            len(tokens), self.chunk_size, self.min_split, self.overlap
        )
        return [
            self._format_chunk(tokens[start:end]) for start, end in spans if end > start
        ]

    def tokenize(self, doc: str, method: str) -> List[Union[int, str]]:
        if method == "tokens":
            return self.token_encoder.encode(doc)

        elif method == "sentences":
            import nltk

            return nltk.sent_tokenize(doc)
        elif method == "words":
            import nltk

            return nltk.word_tokenize(doc)
        elif method == "paragraphs":
            from nltk.tokenize import BlanklineTokenizer

            return BlanklineTokenizer().tokenize(doc)
        else:
            raise ValueError(f"Unsupported tokenization method: {method}")

    def _compute_spans(
        self, n: int, chunk_size: int, min_split: int, overlap: int
    ) -> List[Tuple[int, int]]:
        if n <= chunk_size:
            return [(0, n)]

        n_chunks = max(1, math.ceil(n / chunk_size))
        target = max(min_split, math.ceil(n / n_chunks))
        spans = []
        start = 0
        for _ in range(n_chunks - 1):
            end = min(n, start + target)
            spans.append((start, end))
            start = max(0, end - overlap)
        spans.append((start, n))
        return spans

    def _format_chunk(self, chunk_tokens: List[Union[int, str]]) -> str:
        if not chunk_tokens:
            return ""
        if self.split_unit == "tokens":
            return self.token_encoder.decode(chunk_tokens).strip()
        elif self.split_unit in {"words", "sentences"}:
            return " ".join(chunk_tokens).strip()
        elif self.split_unit == "paragraphs":
            return "\n\n".join(chunk_tokens).strip()
        else:
            raise ValueError(
                f"Unexpected split_unit in format_chunk: {self.split_unit}"
            )

    @property
    def token_encoder(self):
        return tiktoken.get_encoding(self.encoding_name)

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and DataFrame of chunks with length statistics."""
        # Get base metadata from parent
        result = super().result()

        # Build DataFrame of chunks
        rows = []
        for idx, chunk in enumerate(self.output or []):
            content = extract_content(chunk)
            rows.append(
                {
                    "index": idx,
                    "source_id": TrackedItem.extract_source_id(chunk),
                    "content": content,
                    "metadata": getattr(chunk, "metadata", None),
                    "length": (
                        len(self.tokenize(content, method=self.split_unit))
                        if content
                        else 0
                    ),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["split_unit"] = self.split_unit
            df["chunk_size"] = self.chunk_size

        # Add Split-specific data
        result["data"] = df
        result["metadata"]["split_unit"] = self.split_unit
        result["metadata"]["chunk_size"] = self.chunk_size
        result["metadata"]["num_chunks"] = len(self.output or [])

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Split node details."""
        super().export(folder, unique_id=unique_id)

        if self.output:
            import numpy as np

            # Extract content from TrackedItems for statistics
            lens = []
            for doc in self.output:
                # Handle TrackedItem, string, or dict (from JSON deserialization)
                if isinstance(doc, TrackedItem):
                    content = doc.content
                elif isinstance(doc, str):
                    content = doc
                elif isinstance(doc, dict) and "content" in doc:
                    content = doc["content"]
                else:
                    # Skip items that can't be processed (e.g., ChatterResult from wrong node type)
                    logger.debug(
                        f"Skipping non-text output in Split export: {type(doc)}"
                    )
                    continue

                if content and isinstance(content, str):
                    lens.append(len(self.tokenize(content, method=self.split_unit)))

            if lens:
                summary = f"""Split Summary
==============
Chunks created: {len(self.output)}
Split unit: {self.split_unit}
Chunk size: {self.chunk_size}
Average length: {np.mean(lens).round(1)}
Max length: {max(lens)}
Min length: {min(lens)}
"""
            else:
                summary = f"""Split Summary
==============
Chunks created: {len(self.output)}
Split unit: {self.split_unit}
Chunk size: {self.chunk_size}
(Statistics unavailable - output not in expected format)
"""
            (folder / "split_summary.txt").write_text(summary)

            # Export output chunks with source_id naming
            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            for idx, chunk in enumerate(self.output):
                if isinstance(chunk, TrackedItem):
                    # Use source_id in filename
                    (outputs_folder / f"{idx:04d}_{chunk.safe_id}.txt").write_text(
                        chunk.content
                    )

                    # Export metadata if present
                    if chunk.metadata:
                        (
                            outputs_folder / f"{idx:04d}_{chunk.safe_id}_metadata.json"
                        ).write_text(json.dumps(chunk.metadata, indent=2, default=str))
                else:
                    # Backward compatibility: plain strings
                    (outputs_folder / f"{idx:04d}_chunk.txt").write_text(str(chunk))


class ItemsNode(DAGNode):
    """Any note which applies to multiple items at once"""

    async def get_items(self) -> List[Dict[str, Any]]:
        """Resolve all inputs, then zip together, combining multiple inputs.

        Handles TrackedItem transparently: extracts content for templates while
        preserving source_id and metadata for provenance tracking.
        """
        # resolve futures now (it's lazy to this point)
        input_data = {k: self.context[k] for k in self.inputs}

        lengths = {
            k: len(v) if isinstance(v, list) else 1 for k, v in input_data.items()
        }
        max_len = max(lengths.values())

        for k, v in input_data.items():
            if isinstance(v, list):
                if len(v) != max_len:
                    raise ValueError(
                        f"Length mismatch for input '{k}': expected {max_len}, got {len(v)}"
                    )
            else:
                input_data[k] = [v] * max_len

        zipped = list(zip(*[input_data[k] for k in self.inputs]))

        # make the first input available as {{input}} in any template
        items = []
        for idx, values in enumerate(zipped):
            item_dict = {}

            for key, val in zip(self.inputs, values):
                if isinstance(val, TrackedItem):
                    # Extract content for template, preserve metadata
                    item_dict[key] = val.content
                    item_dict[f"__{key}__source_id"] = val.source_id
                    item_dict[f"__{key}__metadata"] = val.metadata
                    item_dict[f"__{key}__tracked_item"] = val  # Keep reference
                else:
                    item_dict[key] = val

            # Make first input available as {{input}}
            if self.inputs:
                first_val = values[0]
                if isinstance(first_val, TrackedItem):
                    item_dict["input"] = first_val.content
                    item_dict["source_id"] = first_val.source_id
                    item_dict["metadata"] = first_val.metadata
                    item_dict["tracked_item"] = first_val
                else:
                    item_dict["input"] = first_val

            items.append(Box(item_dict))

        return items


async def default_map_task(template, context, model, credentials, **kwargs):
    """Default map task renders the Step template for each input item and calls the LLM."""

    rt = render_strict_template(template, context)

    # call chatter as async function within the main event loop
    result = await chatter_async(
        multipart_prompt=rt,
        context=context,
        model=model,
        credentials=credentials,
        action_lookup=get_action_lookup(),
        extra_kwargs=kwargs,
    )
    return result


# TODO implement scrubber as a node?
# class Scrub(ItemsNode):
#     type: Literal["Scrub"] = "Scrub"

#     def run(self) -> List[Any]:


class Map(ItemsNode, CompletionDAGNode):
    model_config = {
        "discriminator": "type",
    }

    type: Literal["Map"] = "Map"

    task: Callable = Field(default=default_map_task, exclude=True)
    template_text: str = None

    @property
    def template(self) -> str:
        return self.template_text

    def validate_template(self):
        try:
            parse_syntax(self.template_text)
            return True
        except Exception as e:
            logger.error(f"Template syntax error: {e}")
            return False

    async def run(self) -> List[Any]:
        # await super().run()

        input_data = self.context[self.inputs[0]] if self.inputs else None
        is_batch = isinstance(input_data, BatchList)

        # Flatten batch input if needed
        if is_batch:
            all_items = []
            batch_sizes = []
            for batch in input_data.batches:
                batch_items = [Box({"input": item}) for item in batch]
                all_items.extend(batch_items)
                batch_sizes.append(len(batch))
            items = all_items
            filtered_context = {
                k: v for k, v in self.context.items() if not isinstance(v, BatchList)
            }
        else:
            items = await self.get_items()
            filtered_context = self.context

        results = [None] * len(items)

        async with anyio.create_task_group() as tg:
            for idx, item in enumerate(items):

                async def run_and_store(index=idx, item=item):
                    async with semaphore:
                        # Collect extra kwargs for LLM
                        extra_kwargs = {}
                        if self.max_tokens is not None:
                            extra_kwargs['max_tokens'] = self.max_tokens
                        extra_kwargs['temperature'] = self.temperature
                        extra_kwargs['seed'] = self.dag.config.seed

                        results[index] = await self.task(
                            template=self.template,
                            context={**filtered_context, **item},
                            model=self.get_model(),
                            credentials=self.dag.config.llm_credentials,
                            **extra_kwargs,
                        )

                tg.start_soon(run_and_store)

        if is_batch:
            # Reconstruct BatchList structure
            reconstructed_batches = []
            result_idx = 0
            for batch_size in batch_sizes:
                batch_results = results[result_idx : result_idx + batch_size]
                reconstructed_batches.append(batch_results)
                result_idx += batch_size
            batch_list_result = BatchList(batches=reconstructed_batches)
            self.output = batch_list_result
            return batch_list_result
        else:
            self.output = results
            return results

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and DataFrame of mapped items."""
        # Get base metadata from parent
        result = super().result()

        input_items = self.get_input_items()
        rows = []

        output_list = self.output if isinstance(self.output, list) else []

        for idx, chatter_result in enumerate(output_list):
            item = input_items[idx] if input_items and idx < len(input_items) else None

            row = TrackedItem.extract_export_metadata(item, idx)
            row.update(
                {
                    "prompt": extract_prompt(chatter_result),
                    "response_text": (
                        str(chatter_result.response)
                        if hasattr(chatter_result, "response")
                        else None
                    ),
                    "response_obj": (
                        chatter_result.response
                        if hasattr(chatter_result, "response")
                        else None
                    ),
                    "chatter_result": chatter_result,
                }
            )

            rows.append(row)

        # Add Map-specific data
        result["data"] = pd.DataFrame(rows)
        result["metadata"]["num_items"] = len(output_list)

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Map node details with numbered prompts and responses."""
        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

        # Get input items for source tracking
        input_items = self.get_input_items()

        # Write each prompt/response pair with source tracking
        if self.output and isinstance(self.output, list):
            for idx, result in enumerate(self.output):
                # Get source_id if available
                item = (
                    input_items[idx] if input_items and idx < len(input_items) else None
                )
                safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))
                file_prefix = f"{idx:04d}_{safe_id}"

                # Try to extract prompt from ChatterResult
                try:
                    if hasattr(result, "results") and result.results:
                        # Get first segment's prompt
                        first_seg = next(iter(result.results.values()))
                        if hasattr(first_seg, "prompt"):
                            (folder / f"{file_prefix}_prompt.md").write_text(
                                first_seg.prompt
                            )

                    # Write response text
                    if hasattr(result, "response"):
                        response_text = str(result.response)
                        (folder / f"{file_prefix}_response.txt").write_text(
                            response_text
                        )

                    # Write full ChatterResult JSON
                    (folder / f"{file_prefix}_response.json").write_text(
                        safe_json_dump(result)
                    )
                except Exception as e:
                    logger.warning(f"Failed to export Map item {idx}: {e}")


class Classifier(ItemsNode, CompletionDAGNode):
    """
    Apply a classification prompt to each input item and extract structured outputs.

    Returns a list of dictionaries where each dict contains the classification results
    for one input item. Fields are extracted from [[name]] and [[pick:name|...]] syntax
    in the template.

    Example output: [{"diagnosis": "cfs", "severity": "high"}, ...]

    If model_names is provided with multiple models, runs classification with each model
    and calculates inter-rater agreement statistics.
    """

    type: Literal["Classifier"] = "Classifier"
    temperature: float = 0.5
    template_text: str = None
    model_names: Optional[List[str]] = None  # Multiple models for agreement analysis
    agreement_fields: Optional[List[str]] = None  # Fields to calculate agreement on
    _processed_items: Optional[List[Any]] = None  # Store items for export
    _model_results: Optional[Dict[str, List[Any]]] = None  # Results keyed by model name
    _agreement_stats: Optional[Dict[str, Dict[str, float]]] = (
        None  # Agreement statistics
    )

    @property
    def template(self) -> str:
        return self.template_text

    def validate_template(self):
        try:
            parse_syntax(self.template_text)
            return True
        except Exception as e:
            logger.error(f"Template syntax error: {e}")
            return False

    async def run(self) -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """Process each item through the classification template.

        Returns:
            If single model: List of ChatterResults
            If multiple models: Dict mapping model_name -> List of ChatterResults
        """

        input_data = self.context[self.inputs[0]] if self.inputs else None
        if not input_data:
            raise Exception("Classifier node must have input data")

        if isinstance(input_data, BatchList):
            raise Exception("Classifier node does not support batch input")

        items = await self.get_items()
        filtered_context = self.context

        # Store items for export to access TrackedItem metadata
        self._processed_items = items

        # Normalize model_names to always be a list
        if not self.model_names:
            self.model_names = (
                [self.model_name] if self.model_name else [self.dag.config.model_name]
            )

        # Initialize result storage
        self._model_results = {}

        # Run classification for each model
        for model_name in self.model_names:
            logger.debug(f"Running classification with model: {model_name}")

            results = [None] * len(items)

            async with anyio.create_task_group() as tg:
                for idx, item in enumerate(items):

                    async def run_and_store(
                        index=idx, item=item, current_model=model_name
                    ):
                        async with semaphore:
                            # Create model instance for this specific model
                            model = LLM(model_name=current_model)

                            # Run the classification template
                            # Collect extra kwargs for LLM
                            extra_kwargs = {}
                            if self.max_tokens is not None:
                                extra_kwargs['max_tokens'] = self.max_tokens
                            extra_kwargs['temperature'] = self.temperature
                            extra_kwargs['seed'] = self.dag.config.seed

                            chatter_result = await chatter_async(
                                multipart_prompt=self.template,
                                context={**filtered_context, **item},
                                model=model,
                                credentials=self.dag.config.llm_credentials,
                                action_lookup=get_action_lookup(),
                                extra_kwargs=extra_kwargs,
                            )

                            # Extract structured outputs from ChatterResult
                            results[index] = chatter_result

                    tg.start_soon(run_and_store)

            self._model_results[model_name] = results

        # Set output: single model returns list, multiple models return dict
        if len(self.model_names) == 1:
            self.output = self._model_results[self.model_names[0]]
        else:
            self.output = self._model_results

        return self.output

    def _build_dataframes_from_results(self):
        """Build DataFrames from in-memory results for agreement calculation."""
        from .agreement import calculate_agreement_from_dataframes

        model_dfs = {}
        for model_name, results in self._model_results.items():
            # Ensure results is a list, not a generator
            if not isinstance(results, list):
                results = (
                    list(results)
                    if hasattr(results, "__iter__")
                    and not isinstance(results, (str, dict))
                    else [results]
                )

            rows = []
            for idx, output_item in enumerate(results):
                # Get metadata using TrackedItem helper
                if self._processed_items and idx < len(self._processed_items):
                    row = TrackedItem.extract_export_metadata(
                        self._processed_items[idx], idx
                    )
                else:
                    row = {"item_id": f"item_{idx}", "index": idx}

                # Add classification outputs
                if hasattr(output_item, "outputs"):
                    output_dict = output_item.outputs
                elif isinstance(output_item, dict):
                    output_dict = {
                        k: v for k, v in output_item.items() if not k.startswith("__")
                    }
                else:
                    continue

                # Convert non-hashable types (lists, BoxList) to strings for agreement calculation
                for k, v in output_dict.items():
                    if (
                        isinstance(v, (list, tuple))
                        or hasattr(v, "__iter__")
                        and not isinstance(v, str)
                    ):
                        row[k] = str(v)  # Convert to string for CSV/agreement
                    else:
                        row[k] = v

                rows.append(row)

            if rows:
                model_dfs[model_name] = pd.DataFrame(rows)

        return model_dfs if model_dfs else None

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and classification results per model."""
        # Get base metadata from parent
        result = super().result()

        # Reconstruct _model_results if needed
        if not hasattr(self, "_model_results") or not self._model_results:
            if isinstance(self.output, dict):
                self._model_results = self.output
            else:
                model_name = (
                    self.model_names[0]
                    if hasattr(self, "model_names") and self.model_names
                    else (self.model_name if hasattr(self, "model_name") else None)
                    or "default"
                )
                self._model_results = {model_name: self.output}

        # Use the same method as export() to build DataFrames
        model_dfs = self._build_dataframes_from_results()

        # Calculate agreement statistics if multiple models
        agreement_stats = None
        agreement_stats_df = None
        if model_dfs and len(model_dfs) >= 2:
            try:
                from .agreement import calculate_agreement_from_dataframes

                # Auto-detect agreement fields if not set
                if not self.agreement_fields:
                    metadata_cols = {"item_id", "document", "filename", "index"}
                    all_fields = {c for df in model_dfs.values() for c in df.columns}
                    self.agreement_fields = sorted(
                        f
                        for f in all_fields
                        if f not in metadata_cols and not f.endswith("__evidence")
                    )

                agreement_stats = calculate_agreement_from_dataframes(
                    model_dfs, self.agreement_fields
                )

                # Convert to DataFrame for easier HTML rendering
                if agreement_stats:
                    agreement_stats_df = pd.DataFrame(agreement_stats).T
                    agreement_stats_df.index.name = "field"
            except Exception as e:
                logger.warning(f"Could not calculate agreement statistics: {e}")
                agreement_stats = None
                agreement_stats_df = None

        # Add Classifier-specific data
        result["model_dfs"] = model_dfs if model_dfs else {}
        result["agreement_stats"] = agreement_stats  # Dict format
        result["agreement_stats_df"] = agreement_stats_df  # DataFrame format
        result["metadata"]["num_models"] = len(model_dfs) if model_dfs else 0
        result["metadata"]["model_names"] = list(model_dfs.keys()) if model_dfs else []
        result["metadata"]["agreement_fields"] = self.agreement_fields or []

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Classifier node with CSV output and individual responses."""
        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

        # Reconstruct _model_results if needed (e.g., from JSON deserialization)
        if not self._model_results:
            if isinstance(self.output, dict):
                self._model_results = self.output
            else:
                # Single model case - wrap in dict
                model_name = (
                    self.model_names[0]
                    if self.model_names
                    else (self.model_name or "default")
                )
                self._model_results = {model_name: self.output}

        # Early return if no results to export (e.g., node failed)
        if not self._model_results or all(
            v is None for v in self._model_results.values()
        ):
            logger.warning(f"No results to export for {self.name}")
            return

        # Export prompts once (same for all models)
        prompts_folder = folder / "prompts"
        prompts_folder.mkdir(parents=True, exist_ok=True)
    
        first_results = next(iter(self._model_results.values()))
        if not first_results:
            logger.warning(f"No results to export for {self.name}")
            return

        for i, result_item in enumerate(first_results):
            item = (
                self._processed_items[i]
                if self._processed_items and i < len(self._processed_items)
                else None
            )
            safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))

            # Export prompts and responses
            if hasattr(result_item, "outputs"):
                (prompts_folder / f"{i:04d}_{safe_id}_response.json").write_text(
                    result_item.outputs.to_json()
                )
                if hasattr(result_item, "results"):
                    for k, v in result_item.results.items():
                        (
                            prompts_folder / f"{i:04d}_{safe_id}_{k}_prompt.txt"
                        ).write_text(v.prompt)
            else:
                # Plain dict from JSON deserialization
                (prompts_folder / f"{i:04d}_{safe_id}_response.json").write_text(
                    json.dumps(result_item, indent=2, default=str)
                )
        
        # Export responses for each model
        responses_folder = folder / "responses"
        responses_folder.mkdir(parents=True, exist_ok=True)

        for model_name, results in self._model_results.items():
            safe_model_name = model_name.replace("/", "_").replace(":", "_")

            for i, result_item in enumerate(results):
                item = (
                    self._processed_items[i]
                    if self._processed_items and i < len(self._processed_items)
                    else None
                )
                safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))
                file_prefix = f"{i:04d}_{safe_id}_{safe_model_name}"

                # Export response for this model
                if hasattr(result_item, "outputs"):
                    (responses_folder / f"{file_prefix}_response.json").write_text(
                        result_item.outputs.to_json()
                    )
                else:
                    # Plain dict from JSON deserialization
                    (responses_folder / f"{file_prefix}_response.json").write_text(
                        json.dumps(result_item, indent=2, default=str)
                    )

        # Export CSV for each model
        for model_name, results in self._model_results.items():
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            suffix = f"_{safe_model_name}" if len(self._model_results) > 1 else ""

            # Build rows with source_id tracking
            rows = []
            for idx, output_item in enumerate(results):
                # Get metadata using TrackedItem helper
                if self._processed_items and idx < len(self._processed_items):
                    row = TrackedItem.extract_export_metadata(
                        self._processed_items[idx], idx
                    )
                else:
                    row = {
                        "item_id": f"item_{idx}",
                        "document": f"item_{idx}",
                        "index": idx,
                    }

                # Add classification outputs
                if hasattr(output_item, "outputs"):
                    output_dict = output_item.outputs
                elif isinstance(output_item, dict):
                    output_dict = {
                        k: v for k, v in output_item.items() if not k.startswith("__")
                    }
                else:
                    continue

                # Convert non-hashable types to strings
                for k, v in output_dict.items():
                    if (
                        isinstance(v, (list, tuple))
                        or hasattr(v, "__iter__")
                        and not isinstance(v, str)
                    ):
                        row[k] = str(v)
                    else:
                        row[k] = v

                rows.append(row)

            if not rows:
                logger.warning(
                    f"No valid classification results to export for {model_name}"
                )
                continue

            # Export CSV, HTML, JSON using utility functions
            df = pd.DataFrame(rows)
            # Add unique_id to filename if provided
            uid_suffix = f"_{unique_id}" if unique_id else ""
            export_to_csv(
                df, folder / f"classifications_{self.name}{suffix}{uid_suffix}.csv"
            )
            export_to_html(
                df, folder / f"classifications_{self.name}{suffix}{uid_suffix}.html"
            )
            export_to_json(
                rows, folder / f"classifications_{self.name}{suffix}{uid_suffix}.json"
            )

        # Calculate and export agreement statistics if multiple models
        if len(self._model_results) >= 2:
            from .agreement import calculate_agreement_from_dataframes

            model_dfs = self._build_dataframes_from_results()
            if model_dfs:
                stats = calculate_agreement_from_dataframes(
                    model_dfs, self.agreement_fields
                )

                # Auto-detect agreement fields if not set (same logic as in calculate_agreement_from_dataframes)
                if not self.agreement_fields:
                    metadata_cols = {"item_id", "document", "filename", "index"}
                    all_fields = {c for df in model_dfs.values() for c in df.columns}
                    self.agreement_fields = sorted(
                        f
                        for f in all_fields
                        if f not in metadata_cols and not f.endswith("__evidence")
                    )

                if stats:
                    stats_prefix = str(folder / "agreement_stats")
                    export_agreement_stats(stats, stats_prefix)
                    logger.info(
                        f"Exported agreement statistics to {stats_prefix}.csv and {stats_prefix}.json"
                    )

                # Generate agreement calculation script and template
                self._generate_agreement_script(folder)

    def _generate_agreement_script(self, folder: Path):
        """Generate a standalone Python script for agreement calculation."""

        if not self._model_results or not self.agreement_fields:
            return

        # Get list of CSV files
        csv_files = []
        for model_name in self._model_results.keys():
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            csv_files.append(f"classifications_{self.name}_{safe_model_name}.csv")

        # Collect valid categories using extracted function
        field_categories = collect_field_categories(
            self._model_results, self.agreement_fields
        )

        # Generate template CSV using extracted function
        first_results = next(iter(self._model_results.values()))
        first_model_name = next(iter(self._model_results.keys()))

        # Build DataFrame for template generation
        rows = []
        for idx, output_item in enumerate(first_results):
            # Get metadata using TrackedItem helper
            if self._processed_items and idx < len(self._processed_items):
                row = TrackedItem.extract_export_metadata(
                    self._processed_items[idx], idx
                )
            else:
                row = {
                    "item_id": f"item_{idx}",
                    "document": f"item_{idx}",
                    "index": idx,
                }

            # Add classification outputs
            if hasattr(output_item, "outputs"):
                output_dict = output_item.outputs
            elif isinstance(output_item, dict):
                output_dict = {
                    k: v for k, v in output_item.items() if not k.startswith("__")
                }
            else:
                continue

            for k, v in output_dict.items():
                if (
                    isinstance(v, (list, tuple))
                    or hasattr(v, "__iter__")
                    and not isinstance(v, str)
                ):
                    row[k] = str(v)
                else:
                    row[k] = v

            rows.append(row)

        df = pd.DataFrame(rows)
        generate_human_rater_template(
            folder, self.name, first_model_name, df, field_categories
        )

        # Generate scripts using extracted function
        write_agreement_scripts(
            folder, self.name, self.agreement_fields, csv_files, field_categories
        )


class Filter(ItemsNode, CompletionDAGNode):
    """
    Filter items based on a boolean LLM completion.

    Each input item is processed through an LLM prompt containing at least one boolean
    completion slot. Only items where the filter field evaluates to True are passed to output.

    Filter field identification:
    1. If a field named 'filter' exists (must be boolean), it's used
    2. Otherwise, the last completion slot is used (must be boolean)
    3. Validation error if the identified field is not boolean

    Output: List[TrackedItem] containing only included items (filter == True)
    """

    type: Literal["Filter"] = "Filter"
    template_text: str = None

    # Internal state for export (not Pydantic fields, just plain attributes)
    _excluded_items: Optional[List[Any]] = None
    _filter_results: Optional[List[Any]] = None
    _processed_items: Optional[List[Any]] = None

    @property
    def template(self) -> str:
        return self.template_text

    def validate_template(self):
        """Validate template has at least one boolean field."""
        try:
            # Just check for [[bool: or [[boolean: or [[decide: syntax without parsing
            bool_pattern = r"\[\[(bool|boolean|decide):"
            if not re.search(bool_pattern, self.template_text, re.IGNORECASE):
                raise ValueError(
                    f"Filter node '{self.name}' template must have at least one boolean completion slot (use [[bool:fieldname]])"
                )
            return True
        except Exception as e:
            logger.error(f"Template validation error: {e}")
            raise e

    def _identify_filter_field(self) -> str:
        """Identify which boolean field to use for filtering.

        Returns:
            str: The field name to use for filtering
        """
        import re

        # Find all boolean field names using regex
        # Pattern matches: [[bool:name]], [[boolean:name]], [[decide:name]]
        bool_pattern = r"\[\[(bool|boolean|decide):(\w+)\]"
        matches = re.findall(bool_pattern, self.template_text, re.IGNORECASE)

        if not matches:
            raise ValueError(f"Filter node '{self.name}' has no boolean fields")

        # Extract field names (second group from each match)
        boolean_fields = [m[1] for m in matches]

        # Convention: prefer field named 'filter'
        if "filter" in boolean_fields:
            return "filter"

        # Fallback: use last boolean field
        return boolean_fields[-1]

    def _extract_filter_value(self, result: ChatterResult, filter_field: str) -> bool:
        """Extract the boolean filter value from a ChatterResult.

        Args:
            result: ChatterResult from LLM
            filter_field: Name of the field to extract

        Returns:
            bool: The filter decision (defaults to False if extraction fails)
        """
        try:
            if hasattr(result, "outputs") and filter_field in result.outputs:
                return bool(result.outputs[filter_field])
            elif hasattr(result, "results") and filter_field in result.results:
                return bool(result.results[filter_field].output)
            else:
                logger.warning(
                    f"Could not find filter field '{filter_field}' in result, defaulting to False (exclude)"
                )
                return False
        except Exception as e:
            logger.warning(
                f"Error extracting filter value: {e}, defaulting to False (exclude)"
            )
            return False

    async def run(self) -> List[Any]:
        """Process each item through filter template, split into included/excluded."""
        await super().run()

        input_data = self.context[self.inputs[0]] if self.inputs else None
        if not input_data:
            raise Exception("Filter node must have input data")

        if isinstance(input_data, BatchList):
            raise Exception("Filter node does not support batch input")

        items = await self.get_items()
        filtered_context = self.context

        # Store for export
        self._processed_items = items

        # Identify which field to use for filtering
        filter_field = self._identify_filter_field()
        logger.info(
            f"Filter node '{self.name}' using field '{filter_field}' for filtering"
        )

        # Process all items concurrently
        results = [None] * len(items)

        async with anyio.create_task_group() as tg:
            for idx, item in enumerate(items):

                async def run_and_store(index=idx, item=item):
                    async with semaphore:
                        # Collect extra kwargs for LLM
                        extra_kwargs = {}
                        if self.max_tokens is not None:
                            extra_kwargs['max_tokens'] = self.max_tokens
                        extra_kwargs['temperature'] = self.temperature
                        extra_kwargs['seed'] = self.dag.config.seed

                        # Run the filter template
                        chatter_result = await chatter_async(
                            multipart_prompt=self.template,
                            context={**filtered_context, **item},
                            model=self.get_model(),
                            credentials=self.dag.config.llm_credentials,
                            action_lookup=get_action_lookup(),
                            extra_kwargs=extra_kwargs,
                        )
                        results[index] = chatter_result

                tg.start_soon(run_and_store)

        # Store all results for export
        self._filter_results = results

        # Split items based on filter field
        included_items = []
        excluded_items = []

        for idx, (item, result) in enumerate(zip(items, results)):
            filter_value = self._extract_filter_value(result, filter_field)

            # Get the original TrackedItem from the item dict
            if hasattr(item, "tracked_item") and isinstance(
                item.tracked_item, TrackedItem
            ):
                tracked = item.tracked_item
            elif "tracked_item" in item:
                tracked = item["tracked_item"]
            elif isinstance(item, TrackedItem):
                tracked = item
            else:
                # Fallback: create TrackedItem from content
                content = (
                    item.get("input", str(item))
                    if isinstance(item, dict)
                    else str(item)
                )
                tracked = TrackedItem(
                    content=content, source_id=f"item_{idx}", metadata={}
                )

            # Create a clean copy of TrackedItem to avoid serialization issues
            # Filter out any non-serializable objects from metadata
            clean_metadata = {}
            if tracked.metadata:
                for k, v in tracked.metadata.items():
                    # Only include simple serializable types
                    if isinstance(v, (str, int, float, bool, type(None))):
                        clean_metadata[k] = v
                    elif isinstance(v, (list, dict)):
                        # Include simple containers (assume they're serializable)
                        try:
                            json.dumps(v)
                            clean_metadata[k] = v
                        except (TypeError, ValueError):
                            # Skip non-serializable items
                            pass

            clean_tracked = TrackedItem(
                content=tracked.content,
                source_id=tracked.source_id,
                metadata=clean_metadata,
            )

            if filter_value:
                included_items.append(clean_tracked)
            else:
                excluded_items.append(clean_tracked)

        # Store excluded items for export
        self._excluded_items = excluded_items

        # Store included items as output (convert to dicts for serialization)
        self.output = [
            {
                "content": item.content,
                "source_id": item.source_id,
                "metadata": item.metadata,
            }
            for item in included_items
        ]

        logger.info(
            f"Filter '{self.name}': {len(included_items)} included, {len(excluded_items)} excluded"
        )

        return self.output

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, included/excluded items and filter statistics."""
        # Get base metadata from parent
        result = super().result()

        included_rows = []
        if self.output:
            for idx, item in enumerate(self.output):
                included_rows.append(
                    {
                        "index": idx,
                        "source_id": (
                            item.get("source_id")
                            if isinstance(item, dict)
                            else getattr(item, "source_id", None)
                        ),
                        "content": (
                            item.get("content")
                            if isinstance(item, dict)
                            else getattr(item, "content", None)
                        ),
                        "metadata": (
                            item.get("metadata")
                            if isinstance(item, dict)
                            else getattr(item, "metadata", None)
                        ),
                    }
                )

        excluded_rows = []
        if self._excluded_items:
            for idx, item in enumerate(self._excluded_items):
                excluded_rows.append(
                    {
                        "index": idx,
                        "source_id": getattr(item, "source_id", None),
                        "content": getattr(item, "content", None),
                        "metadata": getattr(item, "metadata", None),
                    }
                )

        # Add Filter-specific data
        result["included"] = pd.DataFrame(included_rows)
        result["excluded"] = pd.DataFrame(excluded_rows)
        result["filter_field"] = (
            self._identify_filter_field() if self.template_text else None
        )
        result["metadata"]["num_included"] = len(included_rows)
        result["metadata"]["num_excluded"] = len(excluded_rows)

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Filter node with included/excluded items and all prompts."""
        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

        # Write summary statistics
        n_included = len(self.output) if self.output else 0
        n_excluded = len(self._excluded_items) if self._excluded_items else 0
        n_total = n_included + n_excluded

        pct_included = (n_included / n_total * 100) if n_total > 0 else 0
        pct_excluded = (n_excluded / n_total * 100) if n_total > 0 else 0

        summary = f"""Filter Summary
==============
Total items processed: {n_total}
Included (filter=True): {n_included} ({pct_included:.1f}%)
Excluded (filter=False): {n_excluded} ({pct_excluded:.1f}%)

Filter field: {self._identify_filter_field() if self.template_text else 'unknown'}
"""
        (folder / "filter_summary.txt").write_text(summary)

        # Export included items (outputs folder) - reconstruct TrackedItems from output dicts
        if self.output:
            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            for idx, item in enumerate(self.output):
                # Handle both dict (from self.output) and TrackedItem (from _excluded_items)
                if isinstance(item, dict):
                    # Reconstruct from dict
                    safe_id = item["source_id"].replace("/", "_").replace("\\", "_")
                    (outputs_folder / f"{idx:04d}_{safe_id}.txt").write_text(
                        item["content"]
                    )

                    if item.get("metadata"):
                        (
                            outputs_folder / f"{idx:04d}_{safe_id}_metadata.json"
                        ).write_text(
                            json.dumps(item["metadata"], indent=2, default=str)
                        )
                elif isinstance(item, TrackedItem):
                    safe_id = item.safe_id
                    (outputs_folder / f"{idx:04d}_{safe_id}.txt").write_text(
                        item.content
                    )

                    if item.metadata:
                        (
                            outputs_folder / f"{idx:04d}_{safe_id}_metadata.json"
                        ).write_text(json.dumps(item.metadata, indent=2, default=str))

        # Export excluded items (excluded folder)
        if self._excluded_items:
            excluded_folder = folder / "excluded"
            excluded_folder.mkdir(exist_ok=True)

            for idx, item in enumerate(self._excluded_items):
                if isinstance(item, TrackedItem):
                    safe_id = item.safe_id
                    (excluded_folder / f"{idx:04d}_{safe_id}.txt").write_text(
                        item.content
                    )

                    if item.metadata:
                        (
                            excluded_folder / f"{idx:04d}_{safe_id}_metadata.json"
                        ).write_text(json.dumps(item.metadata, indent=2, default=str))

        # Export all prompts and responses (prompts folder)
        if self._filter_results and self._processed_items:
            prompts_folder = folder / "prompts"
            prompts_folder.mkdir(exist_ok=True)

            for idx, (item, result) in enumerate(
                zip(self._processed_items, self._filter_results)
            ):
                # Get source_id for filename
                safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))
                file_prefix = f"{idx:04d}_{safe_id}"

                try:
                    # Extract and write prompt
                    if hasattr(result, "results") and result.results:
                        first_seg = next(iter(result.results.values()))
                        if hasattr(first_seg, "prompt"):
                            (prompts_folder / f"{file_prefix}_prompt.md").write_text(
                                first_seg.prompt
                            )

                    # Write response text
                    if hasattr(result, "response"):
                        (prompts_folder / f"{file_prefix}_response.txt").write_text(
                            str(result.response)
                        )

                    # Write full ChatterResult JSON
                    (prompts_folder / f"{file_prefix}_response.json").write_text(
                        safe_json_dump(result)
                    )

                except Exception as e:
                    logger.warning(f"Failed to export prompt/response {idx}: {e}")


class Transform(ItemsNode, CompletionDAGNode):
    type: Literal["Transform"] = "Transform"
    # TODO: allow for arbitrary python functions instead of chatter?
    # fn: Field(Callable[..., Iterable[Any]],  exclude=True) = None
    template_text: str = Field(default="{{input}} <prompt>: [[output]]")

    @property
    def template(self) -> str:
        return self.template_text

    async def run(self):
        items = await self.get_items()

        if not isinstance(items, str):
            assert len(items) == 1, "Transform nodes must have exactly one input item"

        rt = render_strict_template(self.template, {**self.context, **items[0]})

        # Collect extra kwargs for LLM
        extra_kwargs = {}
        if self.max_tokens is not None:
            extra_kwargs['max_tokens'] = self.max_tokens
        extra_kwargs['temperature'] = self.temperature
        extra_kwargs['seed'] = self.dag.config.seed

        # call chatter as async function within the main event loop
        self.output = await chatter_async(
            multipart_prompt=rt,
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            action_lookup=get_action_lookup(),
            extra_kwargs=extra_kwargs,
        )
        return self.output

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, prompt, response object, and raw ChatterResult."""
        # Get base metadata from parent
        result = super().result()

        # Add Transform-specific data
        result["prompt"] = extract_prompt(self.output)
        result["response_obj"] = (
            self.output.response if hasattr(self.output, "response") else None
        )
        result["response_text"] = (
            str(self.output.response) if hasattr(self.output, "response") else None
        )
        result["chatter_result"] = self.output

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Transform node details with single prompt/response."""
        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

        # Write prompt and response
        if self.output:
            try:
                if hasattr(self.output, "results") and self.output.results:
                    # Get first segment's prompt
                    first_seg = next(iter(self.output.results.values()))
                    if hasattr(first_seg, "prompt"):
                        (folder / "prompt.md").write_text(first_seg.prompt)

                # Write response text
                if hasattr(self.output, "response"):
                    response_text = str(self.output.response)
                    (folder / "response.txt").write_text(response_text)

                # Write full ChatterResult JSON
                (folder / "response.json").write_text(safe_json_dump(self.output))
            except Exception as e:
                logger.warning(f"Failed to export Transform node: {e}")


class Reduce(ItemsNode):
    type: Literal["Reduce"] = "Reduce"
    template_text: str = "{{input}}\n"

    @property
    def template(self) -> str:
        return self.template_text

    def get_items(self):
        if len(self.inputs) > 1:
            raise ValueError("Reduce nodes can only have one input")

        if self.inputs:
            input_data = self.dag.context[self.inputs[0]]
        else:
            input_data = self.dag.config.documents

        # if input is a BatchList, return it directly for special handling in run()
        if isinstance(input_data, BatchList):
            return input_data

        # otherwise, wrap individual items in the expected format
        nk = self.inputs and self.inputs[0] or "input_"
        items = [{"input": v, nk: v} for v in input_data]
        return items

    async def run(self, items=None) -> Any:
        await super().run()

        items = items or self.get_items()

        # if items is a BatchList, run on each batch
        if isinstance(items, BatchList):
            self.output = []
            for batch in items.batches:
                result = await self.run(items=batch)
                self.output.append(result)

            return self.output

        else:
            # handle both dictionaries and strings
            rendered = []
            for item in items:
                if isinstance(item, dict):
                    context = {**item}
                else:
                    # item is a string, wrap it for template processing
                    context = {"input": item}
                rendered.append(render_strict_template(self.template, context))
            self.output = "\n".join(rendered)
            return self.output

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and reduced output."""
        # Get base metadata from parent
        result = super().result()

        # Add Reduce-specific data
        result["output"] = self.output
        result["output_type"] = type(self.output).__name__ if self.output else None

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Reduce node details."""
        super().export(folder, unique_id=unique_id)

        # Write reduce template
        if self.template_text:
            (folder / "reduce_template.md").write_text(self.template_text)

        # Write reduced output
        if self.output:
            if isinstance(self.output, str):
                (folder / "reduced.txt").write_text(self.output)
            elif isinstance(self.output, list):
                # Handle list of reduced outputs
                for idx, item in enumerate(self.output, 1):
                    (folder / f"reduced_{idx:03d}.txt").write_text(str(item))


@dataclass
class BatchList(object):
    batches: List[Any]

    def __iter__(self):
        return iter(self.batches)


class Batch(ItemsNode):
    type: Literal["Batch"] = "Batch"
    # batch_fn: Optional[Callable] = None
    batch_size: int = 10

    async def run(self) -> List[List[Any]]:
        await super().run()

        batches_ = self.default_batch(await self.get_items(), self.batch_size)
        self.output = BatchList(batches=batches_)
        return self.output

    def default_batch(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Batch items into lists of size batch_size."""
        return list(itertools.batched(items, batch_size))


class QualitativeAnalysisPipeline(DAG):
    name: Optional[str] = None

    def to_html(self, template_path: Optional[str] = None) -> str:
        """Render the analysis as HTML using Jinja2 template from file.

        Args:
            template_path: Path to the HTML template file. If None, uses default template.

        Returns:
            Rendered HTML string.
        """
        if template_path is None:
            # Use default template in soak/templates directory
            template_dir = Path(__file__).parent / "templates"
            template_name = "pipeline.html"
        else:
            # Use provided template path
            template_path = Path(template_path)
            template_dir = template_path.parent
            template_name = template_path.name

        # Create Jinja2 environment and load template
        env = Environment(
            loader=FileSystemLoader([template_dir, template_dir / "nodes"]),
            extensions=["jinja_markdown.MarkdownExtension"],
        )

        # Add custom filter to convert DataFrames to HTML
        def df_to_html(df, show_index=None):
            """Convert pandas DataFrame to HTML table.

            Args:
                df: DataFrame to convert
                show_index: Whether to show index. If None, auto-detects based on index name or type.
            """
            if df is None or (hasattr(df, "empty") and df.empty):
                return "<p><em>No data</em></p>"

            # Auto-detect if index should be shown
            if show_index is None:
                # Show index if it has a name or is not a simple RangeIndex
                show_index = df.index.name is not None or not isinstance(
                    df.index, pd.RangeIndex
                )

            return df.to_html(
                classes="table table-sm table-striped", index=show_index, escape=True
            )

        env.filters["df_to_html"] = df_to_html

        # Add enumerate filter for templates
        def enumerate_filter(iterable):
            """Enumerate filter for Jinja2."""
            return list(enumerate(iterable))

        env.filters["enumerate"] = enumerate_filter

        # Add custom function to render individual nodes
        def render_node(node):
            """Render a node using its type-specific template."""
            node_template_name = f"{node.type.lower()}.html"
            nodes_template_dir = Path(__file__).parent / "templates" / "nodes"

            try:
                # Try to load node-specific template
                if (nodes_template_dir / node_template_name).exists():
                    node_template = env.get_template(node_template_name)
                else:
                    # Fall back to default node template
                    node_template = env.get_template("default.html")

                # Get node result with metadata
                try:
                    node_result = node.result()
                except Exception as e:
                    logger.warning(f"Error getting result for node {node.name}: {e}")
                    node_result = {
                        "metadata": {"name": node.name, "type": node.type},
                        "error": str(e),
                    }

                return node_template.render(node=node, result=node_result)
            except Exception as e:
                logger.error(f"Error rendering node {node.name}: {e}")
                return f"<div class='alert alert-danger'>Error rendering node {node.name}: {e}</div>"

        env.globals["render_node"] = render_node

        template = env.get_template(template_name)

        # Get execution order for display
        execution_order = self.get_execution_order()

        # Render template with data
        dd = self.model_dump()
        dd["config"]["documents"] = []
        return template.render(
            pipeline=self,
            result=self.result(),
            detail=dd,
            execution_order=execution_order,
        )

    def result(self):
        def safe_get_output(name):
            try:
                return self.nodes_dict.get(name).output.response
            except Exception:
                logger.warning(f"Error getting output {name}")
                return None

        try:
            codes = self.nodes_dict.get("codes").output.response["codes"]
        except Exception:
            codes = []
        try:
            themes = self.nodes_dict.get("themes").output.response["themes"]
        except Exception:
            themes = []

        try:
            narrative = self.nodes_dict.get("narrative").output.response
        except Exception:
            narrative = ""

        try:
            quotes = self.nodes_dict.get("quotes").output.response
        except Exception:
            quotes = []

        return QualitativeAnalysis.model_validate(
            {
                "themes": themes,
                "codes": codes,
                "narrative": narrative,
                "detail": self.model_dump_json(),
                "quotes": quotes,
            }
        )


def make_windows(
    text: str,
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    extracted_sentences: Optional[List[str]] = None,
) -> List[Tuple[str, int, int]]:
    """Create overlapping windows of text.

    Returns list of tuples: (window_text, start_pos, end_pos)

    Defaults:
    - overlap: 30% of window_size (helps catch quotes spanning window boundaries)
    """

    if not overlap:
        overlap = int(window_size * 0.3)  # 30% overlap for better boundary coverage

    windows = []
    i = 0
    while i < len(text):
        start = i
        end = min(i + window_size, len(text))
        windows.append((text[start:end], start, end))
        i += window_size - overlap
    return windows


ELLIPSIS_RE = re.compile(r"\.{3,}|…")


def create_document_boundaries(
    documents: List["TrackedItem"],
) -> Tuple[List[Tuple[str, int, int]], Dict[str, str]]:
    """Create a list of (doc_name, start_pos, end_pos) for each document in concatenated text.

    Assumes documents are joined with "\n\n" separator.

    Returns:
        Tuple of (boundaries, doc_content_map) where:
        - boundaries: List of (doc_name, start_pos, end_pos)
        - doc_content_map: Dict mapping doc_name to full document content
    """
    boundaries = []
    doc_content_map = {}
    current_pos = 0

    for doc in documents:
        content_len = len(doc.content)

        doc_name = None
        doc_name = (
            doc.metadata.get("filename")
            if hasattr(doc, "metadata") and doc.metadata
            else (
                doc.source_id
                if hasattr(doc, "source_id")
                else getattr(doc, "path", "unknown")
            )
        )
        # Fall back to source_id or path attribute
        if not doc_name:
            doc_name = (
                doc.source_id
                if hasattr(doc, "source_id")
                else getattr(doc, "path", "unknown")
            )

        doc_name_str = str(doc_name)
        boundaries.append((doc_name_str, current_pos, current_pos + content_len))
        doc_content_map[doc_name_str] = doc.content
        current_pos += content_len + 2  # +2 for "\n\n" separator

    return boundaries, doc_content_map


def find_source_document(
    position: int,
    doc_boundaries: List[Tuple[str, int, int]],
    doc_content_map: Dict[str, str],
) -> Tuple[str, str]:
    """Find which document a character position belongs to.

    Returns:
        Tuple of (doc_name, doc_content)
    """
    for doc_name, start, end in doc_boundaries:
        if start <= position < end:
            return doc_name, doc_content_map.get(doc_name, "")
    return "unknown", ""


def find_alignment_fuzzy(
    quote: str, span: str, min_ratio: float = 0.6, context_pad: int = 30
) -> Dict[str, Any]:
    """Find best character offset in span where quote aligns using fuzzy matching.

    Uses difflib.SequenceMatcher.get_matching_blocks() to find exact positions.

    Returns dict with: start_char, end_char, match_ratio, matched_text
    """
    from difflib import SequenceMatcher

    # Normalize for matching
    clean = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    quote_clean = clean(quote)
    span_clean = clean(span)

    if not quote_clean or not span_clean:
        return {"start_char": 0, "end_char": 0, "match_ratio": 0.0, "matched_text": ""}

    # Fast path: exact substring
    if quote_clean in span_clean:
        offset = span_clean.index(quote_clean)
        start = offset
        end = offset + len(quote_clean)

        # Snap to word boundaries
        start, end = snap_to_boundaries(span, start, end, snap_to="word")

        return {
            "start_char": start,
            "end_char": end,
            "match_ratio": 1.0,
            "matched_text": span[start:end],
        }

    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, span_clean, quote_clean)

    # Get all matching blocks above minimum length
    blocks = [b for b in matcher.get_matching_blocks() if b[2] > 5]

    if not blocks:
        # Fallback: no good match
        return {
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": 0.0,
            "matched_text": span,
        }

    # Find the best block (longest contiguous match)
    best_block = max(blocks, key=lambda b: b[2])
    i1, i2, n = best_block
    match_ratio = matcher.ratio()

    # Add context padding
    start = max(i1 - context_pad, 0)
    end = min(i1 + n + context_pad, len(span))

    # Snap to word boundaries to avoid mid-word cuts
    start, end = snap_to_boundaries(span, start, end, snap_to="word")

    return {
        "start_char": start,
        "end_char": end,
        "match_ratio": float(match_ratio),
        "matched_text": span[start:end],
    }


def find_alignment_sliding_bm25(quote: str, span: str) -> Tuple[int, float]:
    """Find best alignment using word-level sliding BM25.

    Returns (start_char_offset, bm25_score)
    """
    quote_tokens = nltk.word_tokenize(quote.lower())
    span_tokens = nltk.word_tokenize(span.lower())

    if len(quote_tokens) > len(span_tokens):
        return 0, 0.0

    # Build windows
    window_size = max(len(quote_tokens), int(len(quote_tokens) * 1.2))
    windows = []
    window_starts = []  # token positions

    for i in range(len(span_tokens) - window_size + 1):
        windows.append(span_tokens[i : i + window_size])
        window_starts.append(i)

    if not windows:
        return 0, 0.0

    # Score with BM25
    bm25 = BM25Okapi(windows)
    scores = bm25.get_scores(quote_tokens)
    best_idx = int(np.argmax(scores))
    best_token_offset = window_starts[best_idx] if best_idx < len(window_starts) else 0

    # Convert token offset to character offset (approximate)
    # Rebuild span up to that token
    if best_token_offset > 0:
        char_offset = len(" ".join(span_tokens[:best_token_offset])) + 1  # +1 for space
    else:
        char_offset = 0

    return char_offset, float(scores[best_idx])


def trim_span_to_quote(
    quote: str,
    span: str,
    method: Literal["fuzzy", "sliding_bm25", "hybrid"] = "hybrid",
    min_fuzzy_ratio: float = 0.6,
    min_span_length_multiplier: float = 1.2,
    context_pad: int = 30,
) -> Dict[str, Any]:
    """Trim span to align with quote start using matching blocks.

    Returns dict with: matched_text, start_char, end_char, match_ratio
    """
    if not span or not quote:
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span) if span else 0,
            "match_ratio": 0.0,
        }

    # Don't trim very short quotes (unreliable)
    if len(quote) < 20:
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": 0.0,
        }

    if method == "fuzzy":
        result = find_alignment_fuzzy(quote, span, min_fuzzy_ratio, context_pad)

    elif method == "sliding_bm25":
        offset, confidence = find_alignment_sliding_bm25(quote, span)
        start, end = offset, len(span)
        # Snap to boundaries
        start, end = snap_to_boundaries(span, start, end, snap_to="word")
        # Convert to dict format
        result = {
            "start_char": start,
            "end_char": end,
            "match_ratio": confidence,
            "matched_text": span[start:end],
        }

    elif method == "hybrid":
        # Try fuzzy first
        result = find_alignment_fuzzy(quote, span, min_fuzzy_ratio, context_pad)
        if result["match_ratio"] < min_fuzzy_ratio:
            # Fall back to BM25
            offset_bm25, confidence_bm25 = find_alignment_sliding_bm25(quote, span)
            # Use BM25 result if it seems reasonable
            if offset_bm25 < len(span) * 0.5:  # heuristic: not too far into span
                start, end = offset_bm25, len(span)
                # Snap to boundaries
                start, end = snap_to_boundaries(span, start, end, snap_to="word")
                result = {
                    "start_char": start,
                    "end_char": end,
                    "match_ratio": confidence_bm25,
                    "matched_text": span[start:end],
                }
    else:
        result = {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": 0.0,
        }

    # Safety: don't trim if result would be too short (preserve some context)
    min_result_length = int(len(quote) * min_span_length_multiplier)
    matched_len = result["end_char"] - result["start_char"]
    if matched_len < min_result_length and matched_len < len(span):
        return {
            "matched_text": span,
            "start_char": 0,
            "end_char": len(span),
            "match_ratio": result["match_ratio"],
        }

    return result


def snap_to_boundaries(
    text: str, start: int, end: int, snap_to: Literal["word", "sentence"] = "word"
) -> Tuple[int, int]:
    """Expand start/end to nearest word or sentence boundary.

    Prevents ugly mid-word cuts by snapping outward to natural boundaries.

    Args:
        text: Full text
        start: Start index (inclusive)
        end: End index (exclusive, as in text[start:end])
    """
    if snap_to == "word":
        # Define word boundary characters (whitespace and punctuation)
        boundaries = {
            " ",
            "\n",
            "\t",
            "\r",
            ".",
            "!",
            "?",
            ",",
            ";",
            ":",
            "-",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            '"',
            "'",
            "/",
            "\\",
        }

        # Expand left until we hit a boundary (or reach start of text)
        while start > 0 and text[start - 1] not in boundaries:
            start -= 1

        # Expand right until we hit a boundary (or reach end of text)
        # Note: end is exclusive (text[start:end]), so we check text[end] if it exists
        while end < len(text) and text[end] not in boundaries:
            end += 1

        # Trim leading whitespace
        while start < end and text[start] in (" ", "\n", "\t", "\r"):
            start += 1

        # Trim trailing whitespace
        while end > start and text[end - 1] in (" ", "\n", "\t", "\r"):
            end -= 1

    elif snap_to == "sentence":
        # Find sentence boundaries (periods, exclamation, question marks followed by space/newline)
        sentence_pattern = r"[.!?][\s\n]+"
        sentence_ends = [m.end() for m in re.finditer(sentence_pattern, text)]

        # Expand left to previous sentence end (or beginning)
        prev_end = 0
        for pos in sentence_ends:
            if pos < start:
                prev_end = pos
            else:
                break
        start = prev_end

        # Expand right to next sentence end (or end of text)
        next_end = len(text)
        for pos in sentence_ends:
            if pos > end:
                next_end = pos
                break
        end = next_end

    return start, end


def is_match_truncated(
    match_result: Dict[str, Any], span_text: str, boundary_threshold: int = 30
) -> bool:
    """Detect if a match looks truncated and might benefit from window expansion.

    Only checks boundary positions (not match_ratio) because low ratio can mean
    either truncation OR too much context (we can't distinguish).

    Returns True if:
    - Matched text starts very close to span beginning (might extend left)
    - Matched text ends very close to span end (might extend right)
    """
    start_char = match_result.get("start_char", 0)
    end_char = match_result.get("end_char", len(span_text))

    # Match starts at/near beginning → might be left-truncated
    starts_at_boundary = start_char < boundary_threshold

    # Match ends at/near end → might be right-truncated
    ends_at_boundary = end_char > len(span_text) - boundary_threshold

    return starts_at_boundary or ends_at_boundary


def verify_quotes_bm25_first(
    extracted_sentences: List[str],
    original_windows: List[Tuple[str, int, int]],
    original_text: str,
    doc_boundaries: List[Tuple[str, int, int]],
    doc_content_map: Dict[str, str],
    get_embedding,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.4,
    ellipsis_max_gap: Optional[int] = 300,
    trim_spans: bool = True,
    trim_method: Literal["fuzzy", "sliding_bm25", "hybrid"] = "hybrid",
    min_fuzzy_ratio: float = 0.6,
    expand_window_neighbors: int = 1,
) -> List[Dict[str, Any]]:
    """Verify quotes using BM25-first matching with ellipsis support.

    For each quote:
    - If no ellipsis: find best BM25 window
    - If ellipsis: match head/tail separately, reconstruct span
    - Always compute embedding similarity on combined span
    - Return BM25 score, ratio (top1/top2), cosine similarity, and full source document content

    Args:
        original_windows: List of (window_text, start_pos, end_pos) tuples
        original_text: Full concatenated source text
        doc_boundaries: List of (doc_name, start_pos, end_pos) for document tracking
        doc_content_map: Dict mapping doc_name to full document content
    """

    if not extracted_sentences:
        return []

    # Extract window texts and build BM25 index
    window_texts = [w[0] for w in original_windows]
    tokenized = [nltk.word_tokenize(w.lower()) for w in window_texts]
    bm25 = BM25Okapi(tokenized, k1=bm25_k1, b=bm25_b)

    results = []

    for quote in extracted_sentences:
        has_ellipsis = bool(ELLIPSIS_RE.search(quote))

        if not has_ellipsis:
            # Simple case: single BM25 match
            query_tokens = nltk.word_tokenize(quote.lower())
            scores = bm25.get_scores(query_tokens)
            sorted_scores = np.sort(scores)[::-1]

            top1 = float(sorted_scores[0]) if len(sorted_scores) > 0 else 0.0
            top2 = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            ratio = top1 / (top2 + 1e-6)

            best_idx = int(np.argmax(scores))
            bm25_score = top1

            # Get window with position info
            single_window_text, window_start_pos, window_end_pos = original_windows[
                best_idx
            ]

            if trim_spans:
                # Try matching in single window
                match_result = trim_span_to_quote(
                    quote, single_window_text, trim_method, min_fuzzy_ratio
                )

                # Check if match looks truncated
                if expand_window_neighbors > 0 and is_match_truncated(
                    match_result, single_window_text
                ):
                    # Expand to neighbors and retry using original text
                    start_idx = max(0, best_idx - expand_window_neighbors)
                    end_idx = min(
                        len(original_windows), best_idx + expand_window_neighbors + 1
                    )

                    # Get global positions for expanded range
                    expanded_start_pos = original_windows[start_idx][1]
                    expanded_end_pos = original_windows[end_idx - 1][2]
                    expanded_span = original_text[expanded_start_pos:expanded_end_pos]

                    # Retry matching in expanded span
                    expanded_result = trim_span_to_quote(
                        quote, expanded_span, trim_method, min_fuzzy_ratio
                    )

                    # Use expanded result if it's better
                    if expanded_result.get("match_ratio", 0) > match_result.get(
                        "match_ratio", 0
                    ):
                        match_result = expanded_result
                        window_start_pos = expanded_start_pos  # Update base position

                span_text = match_result["matched_text"]
                window_relative_start = match_result["start_char"]
                window_relative_end = match_result["end_char"]
                match_ratio = match_result["match_ratio"]

                # Convert to global positions
                global_start = window_start_pos + window_relative_start
                global_end = window_start_pos + window_relative_end
            else:
                span_text = single_window_text
                global_start = window_start_pos
                global_end = window_end_pos
                match_ratio = None

            # Find source document and get full document content
            source_doc, source_doc_content = find_source_document(
                global_start, doc_boundaries, doc_content_map
            )

        else:
            # Ellipsis case: match head and tail
            parts = [p.strip() for p in ELLIPSIS_RE.split(quote) if p.strip()]

            if len(parts) < 2:
                # Degenerate case: ellipsis but only one part
                parts = [quote.replace("...", "").replace("…", "").strip()]

            head = parts[0]
            tail = parts[-1]

            # BM25 for head
            head_tokens = nltk.word_tokenize(head.lower())
            head_scores = bm25.get_scores(head_tokens)
            head_idx = int(np.argmax(head_scores))
            head_score = float(head_scores[head_idx])

            # BM25 for tail
            tail_tokens = nltk.word_tokenize(tail.lower())
            tail_scores = bm25.get_scores(tail_tokens)
            tail_idx = int(np.argmax(tail_scores))
            tail_score = float(tail_scores[tail_idx])

            # Ensure head comes before tail
            if tail_idx < head_idx:
                head_idx, tail_idx = tail_idx, head_idx
                head_score, tail_score = tail_score, head_score

            # Check gap constraint
            if (
                ellipsis_max_gap is not None
                and (tail_idx - head_idx) > ellipsis_max_gap
            ):
                # Fall back to single best window
                best_idx = head_idx if head_score >= tail_score else tail_idx

                # Expand search space to include neighbor windows using original text
                if expand_window_neighbors > 0:
                    start_idx = max(0, best_idx - expand_window_neighbors)
                    end_idx = min(
                        len(original_windows), best_idx + expand_window_neighbors + 1
                    )

                    expanded_start_pos = original_windows[start_idx][1]
                    expanded_end_pos = original_windows[end_idx - 1][2]
                    expanded_span = original_text[expanded_start_pos:expanded_end_pos]
                    window_start_pos = expanded_start_pos
                else:
                    expanded_span, window_start_pos, _ = original_windows[best_idx]

                bm25_score = max(head_score, tail_score)

                # Calculate ratio for the chosen part
                scores = head_scores if head_score >= tail_score else tail_scores
                sorted_scores = np.sort(scores)[::-1]
                top1 = float(sorted_scores[0]) if len(sorted_scores) > 0 else 0.0
                top2 = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
                ratio = top1 / (top2 + 1e-6)

                # Trim span to align with the chosen part (using expanded search space)
                if trim_spans:
                    part_to_match = head if head_score >= tail_score else tail
                    match_result = trim_span_to_quote(
                        part_to_match, expanded_span, trim_method, min_fuzzy_ratio
                    )
                    span_text = match_result["matched_text"]
                    window_relative_start = match_result["start_char"]
                    window_relative_end = match_result["end_char"]
                    match_ratio = match_result["match_ratio"]

                    # Convert to global positions
                    global_start = window_start_pos + window_relative_start
                    global_end = window_start_pos + window_relative_end
                else:
                    span_text = expanded_span
                    global_start = window_start_pos
                    global_end = window_start_pos + len(expanded_span)
                    match_ratio = None

                # Find source document and get full document content
                source_doc, source_doc_content = find_source_document(
                    global_start, doc_boundaries, doc_content_map
                )
            else:
                # Reconstruct span from head to tail using global positions
                head_start_pos = original_windows[head_idx][1]
                tail_end_pos = original_windows[tail_idx][2]
                span_text = original_text[head_start_pos:tail_end_pos]

                bm25_score = (head_score + tail_score) / 2.0

                # Calculate ratio as average of head and tail ratios
                head_sorted = np.sort(head_scores)[::-1]
                tail_sorted = np.sort(tail_scores)[::-1]

                head_top1 = float(head_sorted[0]) if len(head_sorted) > 0 else 0.0
                head_top2 = float(head_sorted[1]) if len(head_sorted) > 1 else 0.0
                tail_top1 = float(tail_sorted[0]) if len(tail_sorted) > 0 else 0.0
                tail_top2 = float(tail_sorted[1]) if len(tail_sorted) > 1 else 0.0

                head_ratio = head_top1 / (head_top2 + 1e-6)
                tail_ratio = tail_top1 / (tail_top2 + 1e-6)
                ratio = (head_ratio + tail_ratio) / 2.0

                # Trim span to align with head part
                if trim_spans:
                    head_part = parts[0]
                    match_result = trim_span_to_quote(
                        head_part, span_text, trim_method, min_fuzzy_ratio
                    )
                    span_text = match_result["matched_text"]
                    window_relative_start = match_result["start_char"]
                    window_relative_end = match_result["end_char"]
                    match_ratio = match_result["match_ratio"]

                    # Convert to global positions
                    global_start = head_start_pos + window_relative_start
                    global_end = head_start_pos + window_relative_end
                else:
                    global_start = head_start_pos
                    global_end = tail_end_pos
                    match_ratio = None

                # Find source document and get full document content
                source_doc, source_doc_content = find_source_document(
                    global_start, doc_boundaries, doc_content_map
                )

        # Compute embedding similarity between original quote and identified span
        # For long spans (e.g., ellipsis quotes spanning multiple windows), we need to
        # avoid exceeding the embedding model's context window (8192 tokens).
        # Strategy: embed comparable-length excerpts - use 2x quote length (bounded by 2000 chars)
        # to allow for context while staying within limits
        max_embed_chars = min(max(len(quote) * 2, 500), 4000)
        quote_truncated = quote[:max_embed_chars]

        # For very long spans, take start portion (where quote likely appears)
        # rather than truncating the quote comparison unfairly
        span_truncated = span_text[:max_embed_chars]

        quote_emb = np.array(get_embedding([quote_truncated]))
        span_emb = np.array(get_embedding([span_truncated]))
        cosine_sim = float(cosine_similarity(quote_emb, span_emb)[0][0])

        results.append(
            {
                "quote": quote,
                "source_doc": source_doc,
                "span_text": span_text,
                "source_doc_content": source_doc_content,
                "bm25_score": float(bm25_score),
                "bm25_ratio": float(ratio),
                "global_start": int(global_start),
                "global_end": int(global_end),
                "match_ratio": float(match_ratio) if match_ratio is not None else None,
                "cosine_similarity": float(cosine_sim),
            }
        )

    return results


class VerifyQuotes(CompletionDAGNode):
    """Quote verification using BM25-first matching with ellipsis support.

    Uses BM25 (lexical search) to identify candidate spans, including handling
    quotes with ellipses by matching head/tail separately and reconstructing
    the full span. Embeddings are computed on identified spans for verification.

    Configurable windowing (default: 1.1 × longest quote, capped at 500 chars,
    30% overlap). Returns BM25 scores, ratios (top1/top2), and cosine similarity.

    Automatic neighbor expansion (±1 window by default) catches quotes that span
    window boundaries.
    """

    type: Literal["VerifyQuotes"] = "VerifyQuotes"
    window_size: Optional[int] = 300
    overlap: Optional[int] = None
    bm25_k1: float = 1.5
    bm25_b: float = 0.4  # Lower value reduces long-doc penalty
    ellipsis_max_gap: Optional[int] = 3  # Max windows between head/tail
    trim_spans: bool = True
    trim_method: Literal["fuzzy", "sliding_bm25", "hybrid"] = "fuzzy"
    min_fuzzy_ratio: float = 0.6
    expand_window_neighbors: int = 1  # Search ±N windows around BM25 best match
    template_text: Optional[str] = None  # Custom LLM-as-judge prompt template

    stats: Optional[Dict[str, Any]] = None
    original_sentences: Optional[List[str]] = None
    extracted_sentences: Optional[List[str]] = None
    sentence_matches: Optional[List[Dict[str, Union[str, Any]]]] = None

    def validate_template(self):
        """Validate template_text if provided."""
        if self.template_text:
            try:
                parse_syntax(self.template_text)
                return True
            except Exception as e:
                logger.error(f"Judge template syntax error: {e}")
                return False
        return True

    async def llm_as_judge(self, quote: str, source: str) -> Dict[str, Any]:
        """Use LLM to verify if a quote is truly contained in the source text.

        This is a last-resort if the lexical and semantic matches are low. We use the LLM to verify if the quote is truly contained in the source text.

        Args:
            quote: The extracted quote to verify
            source: The source text where the quote should appear

        Returns:
            Dict with 'explanation' and 'is_contained' keys
        """
        # Use custom template if provided, otherwise load default from file
        if self.template_text:
            prompt = self.template_text
        else:
            template_path = Path(__file__).parent / "templates" / "llm_as_judge.md"
            prompt = template_path.read_text()

        try:
            result = await chatter_async(
                multipart_prompt=prompt,
                context={"text1": quote, "text2": source},
                model=self.get_model(),
                credentials=self.dag.config.llm_credentials,
                action_lookup=get_action_lookup(),
            )

            # Extract the parsed results
            explanation = (
                result.results.get("explanation", {}).output
                if hasattr(result.results.get("explanation", {}), "output")
                else ""
            )
            is_contained = (
                result.results.get("is_contained", {}).output
                if hasattr(result.results.get("is_contained", {}), "output")
                else None
            )

            return {"explanation": explanation, "is_contained": is_contained}
        except Exception as e:
            logger.error(f"Error in llm_as_judge: {e}")
            return {"explanation": f"Error: {str(e)}", "is_contained": None}

    async def run(self) -> List[Any]:
        await super().run()

        alldocs = "\n\n".join(doc.content for doc in self.dag.config.documents)

        # Create document boundaries for tracking source documents
        doc_boundaries, doc_content_map = create_document_boundaries(
            self.dag.config.documents
        )

        codes = self.context.get("codes")
        if not codes:
            raise Exception("VerifyQuotes must be run after node called `codes`")

        # collect extracted quotes
        self.extracted_sentences = []
        # Wrap in list if it's a single result
        codes_list = [codes] if not isinstance(codes, list) else codes
        for result in codes_list:
            try:
                # Try different ways to extract codes from the result
                cset = None
                if hasattr(result, "outputs"):
                    if hasattr(result.outputs, "codes"):
                        cset = result.outputs.codes
                    elif isinstance(result.outputs, dict) and "codes" in result.outputs:
                        cset = result.outputs["codes"]
                elif hasattr(result, "results") and result.results.get("codes"):
                    output = result.results.get("codes").output
                    if hasattr(output, "codes"):
                        cset = output.codes
                    elif isinstance(output, dict) and "codes" in output:
                        cset = output["codes"]

                if not cset:
                    logger.warning(
                        f"Could not extract codes from result: {type(result)}"
                    )
                    continue

                # Handle CodeList wrapper
                codes_to_process = cset.codes if hasattr(cset, "codes") else cset
                for code in codes_to_process:
                    if hasattr(code, "quotes"):
                        self.extracted_sentences.extend(code.quotes)
                    else:
                        logger.warning(
                            f"Code object missing 'quotes' attribute: {type(code)}"
                        )
            except Exception as e:
                logger.error(f"Error extracting quotes from result: {e}")
                raise

        # Check if we found any quotes - fail early if not
        if not self.extracted_sentences:
            logger.error("No quotes found in codes to verify")
            import pdb

            pdb.set_trace()
            raise ValueError(
                "No quotes found in codes to verify. Check that the 'codes' node produces Code objects with quotes."
            )

        # --- Create windowed slices through haystack ---
        windows_with_positions = make_windows(
            alldocs,
            window_size=self.window_size,
            overlap=self.overlap,
            extracted_sentences=self.extracted_sentences,
        )

        # Store just the text for serialization (not the position tuples)
        self.original_sentences = [w[0] for w in windows_with_positions]

        # --- Run quote matching with BM25-first approach ---
        matches = verify_quotes_bm25_first(
            self.extracted_sentences,
            windows_with_positions,
            alldocs,
            doc_boundaries,
            doc_content_map,
            get_embedding,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            ellipsis_max_gap=self.ellipsis_max_gap,
            trim_spans=self.trim_spans,
            trim_method=self.trim_method,
            min_fuzzy_ratio=self.min_fuzzy_ratio,
            expand_window_neighbors=self.expand_window_neighbors,
        )

        # --- Convert to dataframe and compute stats ---
        df = pd.DataFrame(matches)

        # --- LLM-based verification for poor matches ---
        # Identify poor matches based on multiple criteria
        # TODO formalise why I picked this heuristic
        poor_match_mask = ((df["bm25_score"] < 30) & (df["bm25_ratio"] < 2)) | (
            (df["bm25_score"] < 20) & (df["cosine_similarity"] < 0.7)
        )
        poor_matches = df[poor_match_mask]

        if len(poor_matches) > 0:
            logger.info(f"Running LLM verification on {len(poor_matches)} poor matches")

            # Initialize columns for all rows
            df["llm_explanation"] = None
            df["llm_is_contained"] = None

            # Run LLM judge on poor matches in parallel
            async with anyio.create_task_group() as tg:

                async def check_match(idx, quote, span_text):
                    async with semaphore:
                        result = await self.llm_as_judge(quote, span_text)
                        df.at[idx, "llm_explanation"] = result["explanation"]
                        df.at[idx, "llm_is_contained"] = result["is_contained"]

                for idx, row in poor_matches.iterrows():
                    tg.start_soon(
                        check_match, idx, row["quote"], row["source_doc_content"]
                    )

        self.sentence_matches = df.to_dict(orient="records")

        # Compute statistics
        n_quotes = len(df)
        n_with_ellipses = df["quote"].apply(lambda q: bool(ELLIPSIS_RE.search(q))).sum()

        self.stats = {
            "n_quotes": int(n_quotes),
            "n_with_ellipses": int(n_with_ellipses),
            "mean_bm25_score": float(df["bm25_score"].mean()),
            "median_bm25_score": float(df["bm25_score"].median()),
            "mean_bm25_ratio": float(df["bm25_ratio"].mean()),
            "median_bm25_ratio": float(df["bm25_ratio"].median()),
            "mean_cosine": float(df["cosine_similarity"].mean()),
            "median_cosine": float(df["cosine_similarity"].median()),
            "min_cosine": float(df["cosine_similarity"].min()),
            "max_cosine": float(df["cosine_similarity"].max()),
        }

        # Add match_ratio stats if available
        if "match_ratio" in df.columns and df["match_ratio"].notna().any():
            valid_match = df["match_ratio"].dropna()
            self.stats.update(
                {
                    "mean_match_ratio": (
                        float(valid_match.mean()) if len(valid_match) > 0 else None
                    ),
                    "median_match_ratio": (
                        float(valid_match.median()) if len(valid_match) > 0 else None
                    ),
                    "min_match_ratio": (
                        float(valid_match.min()) if len(valid_match) > 0 else None
                    ),
                    "n_low_match_confidence": (
                        int((valid_match < self.min_fuzzy_ratio).sum())
                        if len(valid_match) > 0
                        else 0
                    ),
                    "mean_span_length": float(
                        (df["global_end"] - df["global_start"]).mean()
                    ),
                }
            )

        return matches

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, DataFrame of quote matches and statistics."""
        # Get base metadata from parent
        result = super().result()

        df = pd.DataFrame(self.sentence_matches)

        # Reorder columns for readability
        if df.empty:
            raise Exception("No matches found when verifying quotes.")

        # Sort by confidence
        if "bm25_ratio" in df.columns:
            df = df.sort_values(
                ["bm25_score", "bm25_ratio", "cosine_similarity"],
                ascending=[True, True, True],
            )

        # Add VerifyQuotes-specific data
        result["matches_df"] = df
        result["stats"] = self.stats
        result["metadata"]["num_quotes"] = len(df)
        result["metadata"]["min_fuzzy_ratio"] = self.min_fuzzy_ratio

        return result

    def export(self, folder: Path, unique_id: str = ""):
        super().export(folder, unique_id=unique_id)
        (folder / "info.txt").write_text(VerifyQuotes.__doc__)
        pd.DataFrame(self.stats, index=[0]).melt().to_csv(
            folder / "stats.csv", index=False
        )

        # Export quote verification as Excel with formatting
        uid_suffix = f"_{unique_id}" if unique_id else ""
        excel_path = folder / f"quote_verification{uid_suffix}.xlsx"
        df = pd.DataFrame(self.sentence_matches)

        # Rename columns for clarity
        df = df.rename(
            columns={
                "quote": "extracted_quote",
                "span_text": "found_in_original",
                "source_doc_content": "full_original_text",
            }
        )

        # Reorder columns: text columns first, then source_doc, then LLM judge, then metrics
        priority_cols = [
            "extracted_quote",
            "found_in_original",
            "source_doc",
            "full_original_text",
        ]

        # Add LLM columns after full_original_text if they exist
        llm_cols = []
        if "llm_is_contained" in df.columns:
            llm_cols.append("llm_is_contained")
        if "llm_explanation" in df.columns:
            llm_cols.append("llm_explanation")

        priority_cols.extend(llm_cols)
        other_cols = [col for col in df.columns if col not in priority_cols]
        df = df[priority_cols + other_cols]

        # Sort by LLM verification (False first) then BM25 metrics, so most problematic quotes appear at top
        sort_cols = []
        sort_ascending = []

        if "llm_is_contained" in df.columns:
            sort_cols.append("llm_is_contained")
            sort_ascending.append(
                True
            )  # False sorts before True, putting failed verifications first

        sort_cols.extend(["bm25_score", "bm25_ratio", "cosine_similarity"])
        sort_ascending.extend([True, True, True])

        df = df.sort_values(sort_cols, ascending=sort_ascending)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Quote Verification", index=False)

            # Access the worksheet to apply formatting
            worksheet = writer.sheets["Quote Verification"]

            # Apply text wrapping and set column widths
            from openpyxl.styles import Alignment, Font

            # Default font size increased by 20% (11pt -> 13pt)
            default_font = Font(size=13)

            for column in worksheet.columns:
                column_letter = column[0].column_letter
                header_value = column[0].value

                # Set width and wrapping for text columns
                if header_value in [
                    "extracted_quote",
                    "found_in_original",
                    "full_original_text",
                ]:
                    worksheet.column_dimensions[column_letter].width = 80
                    for cell in column:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                        cell.font = default_font
                elif header_value == "llm_explanation":
                    # LLM explanation - wide with wrapping
                    worksheet.column_dimensions[column_letter].width = 60
                    for cell in column:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                        cell.font = default_font
                elif header_value == "llm_is_contained":
                    # Boolean column - narrow
                    worksheet.column_dimensions[column_letter].width = 18
                    for cell in column:
                        cell.font = default_font
                elif header_value in [
                    "bm25_score",
                    "bm25_ratio",
                    "cosine_similarity",
                    "global_start",
                    "global_end",
                    "match_ratio",
                ]:
                    # Number columns - narrower
                    worksheet.column_dimensions[column_letter].width = 15
                    for cell in column:
                        cell.font = default_font
                elif header_value == "source_doc":
                    # Source document column - medium width
                    worksheet.column_dimensions[column_letter].width = 25
                    for cell in column:
                        cell.font = default_font
                else:
                    # Auto-width for other columns
                    max_length = 0
                    for cell in column:
                        if cell.value:
                            max_length = max(max_length, len(str(cell.value)))
                        cell.font = default_font
                    worksheet.column_dimensions[column_letter].width = min(
                        max_length + 2, 30
                    )


class TransformReduce(CompletionDAGNode):
    """
    Recursively reduce a list into a single item by transforming it with an LLM template.

    If inputs are ChatterResult with specific keys, then the TransformReduce must produce a ChatterResult with the same keys.

    """

    type: Literal["TransformReduce"] = "TransformReduce"

    chunk_size: int = 20000
    min_split: int = 500
    overlap: int = 0
    split_unit: Literal["chars", "tokens", "words", "sentences", "paragraphs"] = (
        "tokens"
    )
    encoding_name: str = "cl100k_base"
    reduce_template: str = "{{input}}\n\n"
    template_text: str = (
        "<text>\n{{input}}\n</text>\n\n-----\nSummarise the text: [[output]]"
    )
    max_levels: int = 5

    reduction_tree: List[List[OutputUnion]] = Field(default_factory=list, exclude=False)

    @property
    def token_encoder(self):
        return tiktoken.get_encoding(self.encoding_name)

    def tokenize(self, doc: str) -> List[Union[int, str]]:
        if self.split_unit == "tokens":
            return self.token_encoder.encode(doc)
        elif self.split_unit == "sentences":
            import nltk

            return nltk.sent_tokenize(doc)
        elif self.split_unit == "words":
            import nltk

            return nltk.word_tokenize(doc)
        elif self.split_unit == "paragraphs":
            from nltk.tokenize import BlanklineTokenizer

            return BlanklineTokenizer().tokenize(doc)
        else:
            raise ValueError(f"Unsupported tokenization method: {self.split_unit}")

    def _compute_spans(self, n: int) -> List[Tuple[int, int]]:
        if n <= self.chunk_size:
            return [(0, n)]
        n_chunks = max(1, math.ceil(n / self.chunk_size))
        target = max(self.min_split, math.ceil(n / n_chunks))
        spans = []
        start = 0
        for _ in range(n_chunks - 1):
            end = min(n, start + target)
            spans.append((start, end))
            start = max(0, end - self.overlap)
        spans.append((start, n))
        return spans

    def chunk_document(self, doc: str) -> List[str]:
        if self.split_unit == "chars":
            if len(doc) <= self.chunk_size:
                return [doc.strip()]
            spans = self._compute_spans(len(doc))
            return [doc[start:end].strip() for start, end in spans]
        tokens = self.tokenize(doc)
        spans = self._compute_spans(len(tokens))

        if self.split_unit == "tokens":
            return [
                self.token_encoder.decode(tokens[start:end]).strip()
                for start, end in spans
            ]
        else:
            return [" ".join(tokens[start:end]).strip() for start, end in spans]

    def chunk_items(self, items: List[str]) -> List[str]:
        # import pdb; pdb.set_trace()
        joined = "\n".join(
            [
                render_strict_template(
                    self.reduce_template, {**self.context, **i.outputs, "input": i}
                )
                for i in items
            ]
        )
        return self.chunk_document(joined)

    def batch_by_units(self, items: List[str]) -> List[List[str]]:
        batches = []
        current_batch = []
        current_units = 0

        for item in items:
            if self.split_unit == "chars":
                item_len = len(item)
            elif self.split_unit == "tokens":
                item_len = len(self.token_encoder.encode(item))
            elif self.split_unit == "words":
                item_len = len(item.split())
            elif self.split_unit == "sentences":
                import nltk

                item_len = len(nltk.sent_tokenize(item))
            elif self.split_unit == "paragraphs":
                from nltk.tokenize import BlanklineTokenizer

                item_len = len(BlanklineTokenizer().tokenize(item))
            else:
                raise ValueError(f"Unknown split_unit: {self.split_unit}")

            # If adding this item would exceed batch size, flush
            if current_units + item_len > self.chunk_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_units = 0

            current_batch.append(item)
            current_units += item_len

        if current_batch:
            batches.append(current_batch)

        return batches

    async def run(self) -> str:
        await super().run()

        raw_items = self.context[self.inputs[0]]

        if isinstance(raw_items, str):
            raw_items = [raw_items]
        elif not isinstance(raw_items, list):
            raise ValueError("Input to TransformReduce must be a list of strings")

        # Initial chunking
        current = self.chunk_items(raw_items)

        self.reduction_tree = [current]

        level = 0
        nbatches = len(current)
        logger.info(f"Starting with {nbatches} batches")

        while len(current) > 1:
            level += 1
            logger.warning(f"TransformReduce level: {level}")
            if level > self.max_levels:
                raise RuntimeError("Exceeded max cascade depth")

            # Chunk input into batches
            batches = self.batch_by_units(current)
            if len(batches) > (nbatches and nbatches or 1e10):
                import pdb

                pdb.set_trace()
                raise Exception(
                    f"Batches {len(batches)} equals or exceeded original batch size {nbatches} so likely won't ever converge to 1 item. Try increasing the chunk_size or rewriting the prompt to be more concise."
                )
            nbatches = len(batches)

            logger.info(f"Cascade Reducing {len(batches)} batches")
            results: List[Any] = [None] * len(batches)

            async with anyio.create_task_group() as tg:
                for idx, batch in enumerate(batches):

                    async def run_and_store(index=idx, batch=batch):
                        prompt = render_strict_template(
                            self.template_text, {**self.context, "input": batch}
                        )
                        # Collect extra kwargs for LLM
                        extra_kwargs = {}
                        if self.max_tokens is not None:
                            extra_kwargs['max_tokens'] = self.max_tokens
                        extra_kwargs['temperature'] = self.temperature
                        extra_kwargs['seed'] = self.dag.config.seed

                        async with semaphore:
                            results[index] = await chatter_async(
                                multipart_prompt=prompt,
                                model=self.get_model(),
                                credentials=self.dag.config.llm_credentials,
                                action_lookup=get_action_lookup(),
                                extra_kwargs=extra_kwargs,
                            )

                    tg.start_soon(run_and_store)

            # Extract string results and re-chunk for next level
            current = self.chunk_items(results)
            self.reduction_tree.append(current)

        final_prompt = render_strict_template(
            self.template_text, {"input": current[0], **self.context}
        )

        # Collect extra kwargs for LLM
        extra_kwargs = {}
        if self.max_tokens is not None:
            extra_kwargs['max_tokens'] = self.max_tokens
        extra_kwargs['temperature'] = self.temperature
        extra_kwargs['seed'] = self.dag.config.seed

        final_response = await chatter_async(
            multipart_prompt=final_prompt,
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            action_lookup=get_action_lookup(),
            extra_kwargs=extra_kwargs,
        )

        self.output = final_response
        return final_response

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, reduction tree structure and final result."""
        # Get base metadata from parent
        result = super().result()

        # Build tree structure with DataFrames per level
        tree_dfs = []

        for level_idx, level_items in enumerate(self.reduction_tree):
            level_rows = []
            for item_idx, item in enumerate(level_items):
                level_rows.append(
                    {
                        "level": level_idx,
                        "item_idx": item_idx,
                        "text": str(item),
                        "response_obj": (
                            item.response if hasattr(item, "response") else None
                        ),
                        "chatter_result": item if hasattr(item, "results") else None,
                    }
                )
            tree_dfs.append(pd.DataFrame(level_rows))

        # Add TransformReduce-specific data
        result["tree_levels"] = tree_dfs
        result["final_prompt"] = extract_prompt(self.output)
        result["final_response"] = (
            self.output.response if hasattr(self.output, "response") else None
        )
        result["final_chatter_result"] = self.output
        result["metadata"]["num_levels"] = len(tree_dfs)
        result["metadata"]["batch_size"] = self.batch_size

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export TransformReduce node with multi-level structure."""
        super().export(folder, unique_id=unique_id)

        # Write templates
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)
        if self.reduce_template:
            (folder / "reduce_template.md").write_text(self.reduce_template)

        # Export each level of the reduction tree
        if self.reduction_tree:
            for level_idx, level_items in enumerate(self.reduction_tree):
                level_folder = folder / f"level_{level_idx}"
                level_folder.mkdir(exist_ok=True)

                for item_idx, item in enumerate(level_items, 1):
                    try:
                        # Write the item as text
                        item_text = str(item)
                        (level_folder / f"{item_idx:03d}_item.txt").write_text(
                            item_text
                        )

                        # If it's a ChatterResult, export it fully
                        if hasattr(item, "results"):
                            (level_folder / f"{item_idx:03d}_result.json").write_text(
                                safe_json_dump(item)
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to export TransformReduce level {level_idx} item {item_idx}: {e}"
                        )

        # Export final result
        if self.output:
            final_folder = folder / "final"
            final_folder.mkdir(exist_ok=True)

            try:
                if hasattr(self.output, "results") and self.output.results:
                    first_seg = next(iter(self.output.results.values()))
                    if hasattr(first_seg, "prompt"):
                        (final_folder / "prompt.md").write_text(first_seg.prompt)

                if hasattr(self.output, "response"):
                    (final_folder / "response.txt").write_text(
                        str(self.output.response)
                    )

                (final_folder / "response.json").write_text(safe_json_dump(self.output))
            except Exception as e:
                logger.warning(f"Failed to export TransformReduce final result: {e}")


# Resolve forward references after QualitativeAnalysisPipeline is defined

ItemsNode.model_rebuild(force=True)
DAGNode.model_rebuild(force=True)
DAG.model_rebuild(force=True)
QualitativeAnalysis.model_rebuild(force=True)
Filter.model_rebuild(force=True)
