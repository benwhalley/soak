"""DAG (Directed Acyclic Graph) execution engine for pipelines."""

import itertools
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Set, Union

import anyio
from jinja2 import Environment, StrictUndefined, meta
from pydantic import BaseModel, Field, model_validator
from struckdown import ChatterResult, LLM, LLMCredentials

from ..document_utils import extract_text, get_scrubber, unpack_zip_to_temp_paths_if_needed
from .base import TrackedItem, get_default_llm_credentials, SOAK_MAX_RUNTIME

if TYPE_CHECKING:
    from .nodes.base import DAGNode

logger = logging.getLogger(__name__)


class DAGConfig(BaseModel):
    document_paths: List[Union[str, tuple[str, Dict[str, Any]]]] = []
    documents: List[Union[str, "TrackedItem"]] = []
    model_name: str = "litellm/gpt-4.1-mini"
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words
    extra_context: Dict[str, Any] = {}
    llm_credentials: LLMCredentials = Field(
        default_factory=get_default_llm_credentials, repr=False, exclude=True
    )
    scrub_pii: bool = False
    scrubber_model: str = "en_core_web_md"
    scrubber_salt: str | None = Field(default="42", exclude=True)
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
    env = Environment(undefined=StrictUndefined)
    template = env.from_string(template_str)
    return template.render(**context)


@dataclass(frozen=True)
class Edge:
    from_node: str
    to_node: str


# Forward references to node types - will be defined in nodes/ module
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

# Type alias for node outputs
OutputUnion = Union[
    str,
    List[str],
    List[List[str]],
    ChatterResult,
    List[ChatterResult],
    List[List[ChatterResult]],
    # for top matches
    List[Dict[str, Union[str, List[tuple[str, float]]]]],
    # for multi-model classifier
    Dict[str, List[ChatterResult]],
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
