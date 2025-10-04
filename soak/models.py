"""Data models for qualitative analysis pipelines."""
import pandas as pd
import hashlib
import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import anyio
import tiktoken
from box import Box
from decouple import config as env_config
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateSyntaxError,
    meta,
)
from joblib import Memory
from pydantic import BaseModel, Field, constr
from struckdown import LLM, ChatterResult, LLMCredentials, chatter, chatter_async
from struckdown import get_embedding as get_embedding_
from struckdown.parsing import parse_syntax
from struckdown.return_type_models import ACTION_LOOKUP

from .document_utils import (
    extract_text,
    get_scrubber,
    unpack_zip_to_temp_paths_if_needed,
)

if TYPE_CHECKING:
    from .dag import QualitativeAnalysisPipeline


logger = logging.getLogger(__name__)

SOAK_MAX_RUNTIME = 60 * 60 * 3  # 3 hours


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
        logger.warning(f"Failed to serialize to JSON: {e}, using repr fallback")
        return json.dumps(
            {"__repr__": repr(obj), "__str__": str(obj), "__error__": str(e)},
            indent=indent,
        )


MAX_CONCURRENCY = env_config("MAX_CONCURRENCY", default=20, cast=int)
semaphore = anyio.Semaphore(MAX_CONCURRENCY)


# exception classes for backward compatibility
class CancelledRun(Exception):
    """Exception raised when a flow run is cancelled."""

    pass


class Cancelled(Exception):
    """Exception raised when a task is cancelled."""

    pass


CodeSlugStr = constr(min_length=12, max_length=64)


class Code(BaseModel):
    slug: CodeSlugStr = Field(
        ...,
        description="A very short abbreviated unique slug/reference for this Code (max 20 letters a-Z).",
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
        max_length=20,
        description="A List of the code-references that are part of this theme. Identify them accurately by slug/hash code from the text. Each code will be around 8 to 20 a-Z characters long. Only refer to codes in the input text above.",
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
            lineage.append("__".join(parts[:i+1]))
        return lineage

    @property
    def depth(self) -> int:
        """Return nesting depth (number of splits from original)."""
        return len(self.lineage) - 1

    @property
    def safe_id(self) -> str:
        """Return source_id safe for filesystem use."""
        return self.source_id.replace("/", "_").replace("\\", "_")

    @staticmethod
    def extract_source_id(item: Any) -> str:
        """Extract source_id from TrackedItem, Box with tracked_item, or return 'unknown'."""
        if isinstance(item, TrackedItem):
            return item.source_id
        elif hasattr(item, 'tracked_item') and isinstance(item.tracked_item, TrackedItem):
            return item.tracked_item.source_id
        elif hasattr(item, 'source_id'):
            return item.source_id
        return "unknown"

    @staticmethod
    def make_safe_id(source_id: str) -> str:
        """Make a source_id safe for filesystem use."""
        return source_id.replace("/", "_").replace("\\", "_")


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
    model_name: str = "gpt-5-mini"
    temperature: float = 1.0
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words
    extra_context: Dict[str, Any] = {}
    llm_credentials: LLMCredentials = Field(default_factory=get_default_llm_credentials)
    scrub_pii: bool = False
    scrubber_model: str = "en_core_web_md"
    scrubber_salt: Optional[str] = Field(default="42", exclude=True)

    def get_model(self):
        return LLM(model_name=self.model_name, temperature=self.temperature)

    def load_documents(self) -> List["TrackedItem"]:
        """Load documents and wrap them in TrackedItem for provenance tracking."""
        if hasattr(self, "documents") and self.documents:
            logger.info("Using cached documents")
            # Ensure cached docs are TrackedItems
            if self.documents and isinstance(self.documents[0], TrackedItem):
                return self.documents
            # Upgrade cached string documents to TrackedItems
            logger.info("Upgrading cached documents to TrackedItems")
            self.documents = [
                TrackedItem(
                    content=doc,
                    source_id=f"doc_{idx}",
                    metadata={"doc_index": idx}
                ) if isinstance(doc, str) else doc
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
                if path_metadata.get('zip_source'):
                    source_id = f"{path_metadata['zip_source']}__{file_stem}"
                else:
                    source_id = file_stem

                # Build metadata
                metadata = {
                    "original_path": str(path),
                    "doc_index": idx,
                    "filename": Path(path).name
                }

                # Add zip info to metadata if present
                if path_metadata.get('zip_source'):
                    metadata["zip_source"] = path_metadata['zip_source']
                    metadata["zip_path"] = path_metadata['zip_path']

                tracked_docs.append(TrackedItem(
                    content=text,
                    source_id=source_id,
                    metadata=metadata
                ))

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
                    if path_metadata.get('zip_source'):
                        source_id = f"{path_metadata['zip_source']}__{file_stem}"
                    else:
                        source_id = file_stem

                    # Build metadata
                    metadata = {
                        "original_path": str(path),
                        "doc_index": idx,
                        "filename": Path(path).name
                    }

                    # Add zip info to metadata if present
                    if path_metadata.get('zip_source'):
                        metadata["zip_source"] = path_metadata['zip_source']
                        metadata["zip_path"] = path_metadata['zip_path']

                    tracked_docs.append(TrackedItem(
                        content=text,
                        source_id=source_id,
                        metadata=metadata
                    ))

                self.documents = tracked_docs

        if self.scrub_pii:
            logger.info("Scrubbing PII")
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
        logger.info(f"COMPLETED: {node.name}\n")
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

    def progress(self):
        last_complete = self.nodes[0]
        return f"Last completed: {last_complete.name}"

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
        lines = ["flowchart TD"]

        shape_map = {
            "Split": ("(", ")"),  # round edges
            "Map": ("[[", "]]"),  # standard rectangle
            "Reduce": ("{{", "}}"),  # hexagon
            "Transform": (">", "]"),  #
            "TransformReduce": (">", "]"),  #
            "VerifyQuotes": ("[[", "]]"),  #
            "Batch": ("[[", "]]"),  # subroutine shape
            "Classifier": ("[/", "\\]"),  # parallelogram (for input/output operations)
        }

        for node in self.nodes:
            le, ri = shape_map.get(node.type, ("[", "]"))  # fallback to rectangle
            label = f"{node.type}: {node.name}"
            lines.append(f"    {node.name}{le}{label}{ri}")

        for edge in self.edges:
            lines.append(f"    {edge.from_node} --> {edge.to_node}")

        lines.append(
            """classDef heavyDotted stroke-dasharray: 4 4, stroke-width: 2px;"""
        )
        for node in self.nodes:
            if node.type == "TransformReduce":
                lines.append("""class all_themes heavyDotted;""")

        return "\n".join(lines)

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
                with anyio.fail_after(
                    SOAK_MAX_RUNTIME
                ):  # 2 hours, to cleanup if needed
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

        meta_content += f"\nDAG Configuration:\n"
        meta_content += f"  Model: {self.config.model_name}\n"
        meta_content += f"  Temperature: {self.config.temperature}\n"
        meta_content += f"  Chunk size: {self.config.chunk_size}\n"
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
                node.export(node_folder)
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
        if self.model_name or self.temperature:
            m = LLM(model_name=self.model_name, temperature=self.temperature)
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
        logger.info(f"\n\nRunning `{self.name}` ({self.__class__.__name__})\n\n")

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

    def export(self, folder: Path):
        """Export node execution details to a folder. Override in subclasses."""
        folder.mkdir(parents=True, exist_ok=True)

        # Write node config
        config_data = {
            "name": self.name,
            "type": self.type,
            "inputs": self.inputs,
        }
        (folder / "meta.txt").write_text(safe_json_dump(config_data))

        # Export input items for traceability
        input_items = self.get_input_items()
        if input_items and isinstance(input_items, list):
            inputs_folder = folder / "inputs"
            inputs_folder.mkdir(exist_ok=True)

            for idx, item in enumerate(input_items):
                if isinstance(item, TrackedItem):
                    # Export with source_id in filename
                    (inputs_folder / f"{idx:04d}_{item.safe_id}.txt").write_text(item.content)

                    # Export metadata
                    if item.metadata:
                        (inputs_folder / f"{idx:04d}_{item.safe_id}_metadata.json").write_text(
                            json.dumps(item.metadata, indent=2, default=str)
                        )
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
    temperature: Optional[float] = None
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
    split_unit: Literal["chars", "tokens", "words", "sentences", "paragraphs"] = "tokens"
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
                    content=doc,
                    source_id="unknown_doc",
                    metadata={}
                )
                chunks = self.split_tracked_document(temp_tracked)
            all_chunks.extend(chunks)

        self.output = all_chunks

        # Calculate stats on content
        lens = [
            len(self.tokenize(
                chunk.content if isinstance(chunk, TrackedItem) else chunk,
                method=self.split_unit
            ))
            for chunk in self.output
        ]
        logger.info(
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

            tracked_chunks.append(TrackedItem(
                content=chunk,
                source_id=new_source_id,
                metadata={
                    **doc.metadata,
                    "split_index": idx,
                    "split_node": self.name,
                    "parent_source_id": doc.source_id
                }
            ))

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

    def export(self, folder: Path):
        """Export Split node details."""
        super().export(folder)

        if self.output:
            import numpy as np

            # Extract content from TrackedItems for statistics
            lens = []
            for doc in self.output:
                content = doc.content if isinstance(doc, TrackedItem) else doc
                lens.append(len(self.tokenize(content, method=self.split_unit)))

            summary = f"""Split Summary
==============
Chunks created: {len(self.output)}
Split unit: {self.split_unit}
Chunk size: {self.chunk_size}
Average length: {np.mean(lens).round(1)}
Max length: {max(lens)}
Min length: {min(lens)}
"""
            (folder / "split_summary.txt").write_text(summary)

            # Export output chunks with source_id naming
            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            for idx, chunk in enumerate(self.output):
                if isinstance(chunk, TrackedItem):
                    # Use source_id in filename
                    (outputs_folder / f"{idx:04d}_{chunk.safe_id}.txt").write_text(chunk.content)

                    # Export metadata if present
                    if chunk.metadata:
                        (outputs_folder / f"{idx:04d}_{chunk.safe_id}_metadata.json").write_text(
                            json.dumps(chunk.metadata, indent=2, default=str)
                        )
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
                        results[index] = await self.task(
                            template=self.template,
                            context={**filtered_context, **item},
                            model=self.get_model(),
                            credentials=self.dag.config.llm_credentials,
                            max_tokens=self.max_tokens,
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

    def export(self, folder: Path):
        """Export Map node details with numbered prompts and responses."""
        super().export(folder)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

        # Get input items for source tracking
        input_items = self.get_input_items()

        # Write each prompt/response pair with source tracking
        if self.output and isinstance(self.output, list):
            for idx, result in enumerate(self.output):
                # Get source_id if available
                item = input_items[idx] if input_items and idx < len(input_items) else None
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
                        (folder / f"{file_prefix}_response.txt").write_text(response_text)

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
    """

    type: Literal["Classifier"] = "Classifier"
    template_text: str = None
    _processed_items: Optional[List[Any]] = None  # Store items for export

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

    async def run(self) -> List[Dict[str, Any]]:
        """Process each item through the classification template."""

        input_data = self.context[self.inputs[0]] if self.inputs else None
        if not input_data:
            raise Exception("Classifier node must have input data")

        if isinstance(input_data, BatchList):
            raise Exception("Classifier node does not support batch input")

        items = await self.get_items()
        filtered_context = self.context

        # Store items for export to access TrackedItem metadata
        self._processed_items = items

        results = [None] * len(items)

        async with anyio.create_task_group() as tg:
            for idx, item in enumerate(items):

                async def run_and_store(index=idx, item=item):
                    async with semaphore:
                        # Run the classification template
                        chatter_result = await chatter_async(
                            multipart_prompt=self.template,
                            context={**filtered_context, **item},
                            model=self.get_model(),
                            credentials=self.dag.config.llm_credentials,
                            action_lookup=get_action_lookup(),
                            extra_kwargs={"max_tokens": self.max_tokens},
                        )

                        # Extract structured outputs from ChatterResult
                        # ChatterResult.response is typically a dict with the extracted fields
                        results[index] = chatter_result

                tg.start_soon(run_and_store)

        self.output = results
        return results

    def export(self, folder: Path):
        """Export Classifier node with CSV output and individual responses."""
        super().export(folder)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

        # Export individual responses with source tracking
        for i, j in enumerate(self.output):
            # Get source_id from processed items
            item = self._processed_items[i] if self._processed_items and i < len(self._processed_items) else None
            safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))

            # Export response with source_id in filename
            (folder/"prompts").mkdir(parents=True, exist_ok=True)
            (folder / "prompts" / f"{i:04d}_{safe_id}_response.json").write_text(j.outputs.to_json())
            for k, v in j.results.items():
                (folder / "prompts" / f"{i:04d}_{safe_id}_{k}_prompt.txt").write_text(v.prompt)

        # Write results as CSV (primary format) with source tracking
        if self.output and isinstance(self.output, list):
            try:
                # Build rows with source_id tracking
                rows = []
                for idx, output_item in enumerate(self.output):
                    row = {}

                    # Add source tracking first (will be first columns)
                    if self._processed_items and idx < len(self._processed_items):
                        source_id = TrackedItem.extract_source_id(self._processed_items[idx])
                        row["item_id"] = source_id

                        # Parse out document name (first part before __)
                        parts = source_id.split("__")
                        row["document"] = parts[0] if parts else source_id

                        # Add original filename if available from tracked_item metadata
                        item = self._processed_items[idx]
                        metadata = None
                        if isinstance(item, TrackedItem):
                            metadata = item.metadata
                        elif hasattr(item, 'tracked_item') and isinstance(item.tracked_item, TrackedItem):
                            metadata = item.tracked_item.metadata

                        if metadata:
                            if "filename" in metadata:
                                row["filename"] = metadata["filename"]
                            elif "original_path" in metadata:
                                row["filename"] = metadata["original_path"]
                    else:
                        row["item_id"] = f"item_{idx}"
                        row["document"] = f"item_{idx}"

                    # Add index
                    row["index"] = idx

                    # Add classification outputs
                    row.update(output_item.outputs)

                    rows.append(row)

                # Convert to DataFrame and export as CSV
                df = pd.DataFrame(rows)
                df.to_csv(folder / f"classifications_{self.name}.csv", index=False)
                html = df.to_html(
                index=False,
                classes="dataframe",
                escape=False
                )

                styled = f"""
                <html>
                <head>
                <style>
                table {{
                border-collapse: collapse;
                width: 100%;
                }}
                th, td {{
                border: 1px solid #ccc;
                padding: 6px 8px;
                text-align: left;
                vertical-align: top;   /* this is the key */
                font-family: sans-serif;
                font-size: 14px;
                }}
                th {{
                background: #f0f0f0;
                }}
                </style>
                </head>
                <body>
                {html}
                </body>
                </html>
                """

                (folder / f"classifications_{self.name}.html").write_text(styled, encoding="utf-8")

                # Also write as JSON
                (folder / f"classifications_{self.name}.json").write_text(
                    json.dumps(rows, indent=2, default=str)
                )

                # Write summary statistics if there are categorical fields
                summary = f"""Classifier Summary
==================
Total items classified: {len(self.output)}

Field Summary:
"""
                if self.output and isinstance(self.output[0], dict):
                    for field in self.output[0].keys():
                        values = [item.get(field) for item in self.output if field in item]
                        unique_vals = set(str(v) for v in values if v is not None)
                        summary += f"\n{field}:\n  Unique values: {len(unique_vals)}\n  Values: {', '.join(sorted(unique_vals))}\n"

                (folder / "summary.txt").write_text(summary)

            except Exception as e:
                logger.warning(f"Failed to export Classifier results as CSV: {e}")


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

        # call chatter as async function within the main event loop
        self.output = await chatter_async(
            multipart_prompt=rt,
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            action_lookup=get_action_lookup(),
            extra_kwargs={"max_tokens": self.max_tokens},
        )
        return self.output

    def export(self, folder: Path):
        """Export Transform node details with single prompt/response."""
        super().export(folder)

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

    def export(self, folder: Path):
        """Export Reduce node details."""
        super().export(folder)

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
            template_name = "default.html"
        else:
            # Use provided template path
            template_path = Path(template_path)
            template_dir = template_path.parent
            template_name = template_path.name

        # Create Jinja2 environment and load template
        env = Environment(
            loader=FileSystemLoader(template_dir),
            extensions=["jinja_markdown.MarkdownExtension"],
        )

        template = env.get_template(template_name)

        # Render template with data
        dd = self.model_dump()
        dd["config"]["documents"] = []
        return template.render(pipeline=self, result=self.result(), detail=dd)

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


class VerifyQuotes(DAGNode):
    
    """Quote Verifiction, comparing extracted with original texts.
    
    Quote verification statistics are calculated by extracting 'matches' for each quote used to support a Code. Matches are found by using text-embeddings of extracted quotes and a sentence-by-sentence split of the source documents. The similarity value represents the semantic similarity of the extracted and original quote (range 0 to 1, higher scores are better). Matches > 0.9 are typically very close, and normally explained by the LLM tidying punctuation or using ellipses. Checking against the original in this way mitigates the risk that quotes have been hallucinated or changed in the interview transcript: we can verify that an exact match or similar match exists in the original text."""
    
    
    type: Literal["VerifyQuotes"] = "VerifyQuotes"
    threshold: float = 0.6
    stats: Optional[Dict[str, Any]] = None
    original_sentences: Optional[List[str]] = None
    extracted_sentences: Optional[List[str]] = None

    sentence_matches: Optional[List[Dict[str, Union[str,Any]]]] = None
    
    async def run(self) -> List[Any]:
        await super().run()

        import nltk
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        alldocs = "\n\n".join(self.dag.config.documents)
        
        self.original_sentences = nltk.sent_tokenize(alldocs)

        codes = self.context.get("codes", None)
        if not codes:
            raise Exception("VerifyQuotes must be run after node called `codes`")

        self.extracted_sentences = list(itertools.chain(*[i.quotes for i in codes.response.codes]))

        # embed real and extracted quotes
        # import pdb; pdb.set_trace()
        try:
            real_emb = np.array(get_embedding(self.original_sentences))
            extr_emb = np.array(get_embedding(self.extracted_sentences))
        except Exception as e:
            print(e)
        
        # calculate cosine similarity
        similarities = cosine_similarity(extr_emb, real_emb)
        # self.similarity_matrix = similarities.tolist()
        
        # find top matches and sort by similarity
        top_matches = []
        max_matches = 10
        for quote, row in zip(self.extracted_sentences, similarities):
            real_and_sim = list(zip(self.original_sentences, row))
            above_thresh = [
                (r, float(s)) for r, s in real_and_sim if s >= self.threshold
            ]

            if above_thresh:
                matches = sorted(above_thresh, key=lambda x: -x[1])[:max_matches]
            else:
                top3_idx = row.argsort()[-3:]
                matches = sorted(
                    [(self.original_sentences[j], float(row[j])) for j in top3_idx],
                    key=lambda x: -x[1],
                )[:max_matches]

            top_matches.append(
                {
                    "quote": quote,
                    "matches": matches,
                }
            )
        
        # TODO tie the quotes back to the original source doc?

        # expand top matches into a df to export late
        df = pd.DataFrame(top_matches)
        # expand: one row per (quote, match, similarity)
        df = df.explode("matches", ignore_index=True)
        df[["match", "similarity"]] = pd.DataFrame(df["matches"].tolist(), index=df.index)
        df = df.drop(columns=["matches"])
        sentence_matches = df.to_dict(orient="records")
        
        best_matches = (
            df.sort_values(["quote", "similarity"], ascending=[True, False])
            .groupby("quote", as_index=False)
            .first()
        )        
        # calulate stats on matches
        try:   
            self.stats = best_matches["similarity"].describe(percentiles=[0.1, 0.9])        
            self.stats.update({
            "n_no_match_above_threshold": (best_matches["similarity"] < self.threshold).sum(),
            "pct_no_match_above_threshold": (best_matches["similarity"] < self.threshold).mean() * 100,
            })
            # for pydantic serialisation
            self.stats = {k: float(v) for k, v in self.stats.items()}
            
        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
            self.stats = {"error": str(e)}
    
        # import pdb; pdb.set_trace()
        self.sentence_matches = sentence_matches
        
        return top_matches

    def export(self, folder: Path):
        """Export a VerifyQuotes node, including details of quote matches with original sources."""
        super().export(folder)
        # Write statistics
        
        """"""
        (folder/'info.txt').write_text(VerifyQuotes.__doc__)
        
        pd.DataFrame(self.stats, index=[1]).melt().to_csv(folder/'stats.csv')
        # Write CSV with quote verification details
        pd.DataFrame(self.sentence_matches).to_csv(folder / "quote_verification.csv")
        
            
class TransformReduce(CompletionDAGNode):
    """
    Recursively reduce a list into a single item by transforming it with an LLM template.

    If inputs are ChatterResult with specific keys, then the TransformReduce must produce a ChatterResult with the same keys.

    """

    type: Literal["TransformReduce"] = "TransformReduce"

    chunk_size: int = 20000
    min_split: int = 500
    overlap: int = 0
    split_unit: Literal["chars", "tokens", "words", "sentences", "paragraphs"] = "tokens"
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
                        async with semaphore:
                            results[index] = await chatter_async(
                                multipart_prompt=prompt,
                                model=self.get_model(),
                                credentials=self.dag.config.llm_credentials,
                                action_lookup=get_action_lookup(),
                                extra_kwargs={"max_tokens": self.max_tokens},
                            )

                    tg.start_soon(run_and_store)

            # Extract string results and re-chunk for next level
            current = self.chunk_items(results)
            self.reduction_tree.append(current)

        final_prompt = render_strict_template(
            self.template_text, {"input": current[0], **self.context}
        )
        final_response = await chatter_async(
            multipart_prompt=final_prompt,
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            action_lookup=get_action_lookup(),
            extra_kwargs={"max_tokens": self.max_tokens},
        )

        self.output = final_response
        return final_response

    def export(self, folder: Path):
        """Export TransformReduce node with multi-level structure."""
        super().export(folder)

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
