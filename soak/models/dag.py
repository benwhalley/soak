"""DAG (Directed Acyclic Graph) execution engine for pipelines."""

import itertools
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import anyio
from jinja2 import Environment, StrictUndefined, meta
from pydantic import BaseModel, Field, model_validator
from struckdown import LLM, ChatterResult, LLMCredentials

from soak.document_utils import (
    extract_text,
    get_scrubber,
    is_spreadsheet,
    unpack_zip_to_temp_paths_if_needed,
)
from soak.models.base import SOAK_MAX_RUNTIME, TrackedItem, get_default_llm_credentials

if TYPE_CHECKING:
    from .nodes.base import DAGNode
    from .nodes.batch import BatchList

logger = logging.getLogger(__name__)


class DAGConfig(BaseModel):
    """Configuration for DAG execution including documents, model settings, and error handling."""

    document_paths: List[Union[str, tuple[str, Dict[str, Any]]]] = []
    documents: List[Union[str, "TrackedItem"]] = []
    model_name: str = "gpt-4.1-mini"
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words
    extra_context: Dict[str, Any] = {}
    llm_credentials: LLMCredentials = Field(
        default_factory=get_default_llm_credentials, repr=False, exclude=True
    )
    scrub_pii: bool = False
    scrubber_model: str = "en_core_web_md"
    scrubber_salt: str | None = Field(default="42", exclude=True)
    seed: int = 42
    sample_n: int | None = None  # randomly sample N documents/rows
    head_n: int | None = None  # take first N documents/rows
    show_progress: bool = False  # show progress bars during execution

    # error handling configuration
    fail_on_context_exceeded: bool = True  # if False, skip item with warning
    skip_content_policy_violations: bool = True  # if False, fail pipeline
    log_failed_prompts: bool = True  # log offending prompts to stderr

    def get_model(self):
        """Create LLM instance with configured model_name."""
        return LLM(model_name=self.model_name)

    def _create_tracked_items_from_file(
        self, path: str, path_metadata: Dict[str, Any], doc_index: int
    ) -> List["TrackedItem"]:
        """Create TrackedItem(s) from a single file.

        Regular files (PDF, DOCX, TXT) produce one TrackedItem.
        Spreadsheets (CSV, XLSX) produce one TrackedItem per row.

        Args:
            path: File path
            path_metadata: Metadata from zip extraction (zip_source, zip_path)
            doc_index: Global document index counter

        Returns:
            List of TrackedItem objects
        """
        file_stem = Path(path).stem
        file_name = Path(path).name

        # Extract content (str for regular files, list of dicts for spreadsheets)
        content = extract_text(path)

        # Check if this is a spreadsheet
        if is_spreadsheet(path):
            # Spreadsheet: create one TrackedItem per row
            tracked_items = []
            for row_idx, row_data in enumerate(content):
                # Build item_id: filename__row_0, filename__row_1, etc.
                if path_metadata.get("zip_source"):
                    item_id = (
                        f"{path_metadata['zip_source']}__{file_stem}__row_{row_idx}"
                    )
                else:
                    item_id = f"{file_stem}__row_{row_idx}"

                # Build metadata: merge column data with file metadata
                metadata = {
                    "original_path": str(path),
                    "filename": file_name,
                    "doc_index": doc_index,
                    "row_index": row_idx,
                    **row_data,  # Spread all column values into metadata
                }

                # Add zip info if present
                if path_metadata.get("zip_source"):
                    metadata["zip_source"] = path_metadata["zip_source"]
                    metadata["zip_path"] = path_metadata["zip_path"]

                # Content is empty string for spreadsheet rows
                # (all data is in metadata/columns)
                tracked_items.append(
                    TrackedItem(
                        content="", id=item_id, sources=[item_id], metadata=metadata
                    )
                )

            logger.info(
                f"Created {len(tracked_items)} TrackedItems from spreadsheet {file_name}"
            )
            return tracked_items

        else:
            # Regular file: create single TrackedItem
            if path_metadata.get("zip_source"):
                item_id = f"{path_metadata['zip_source']}__{file_stem}"
            else:
                item_id = file_stem

            metadata = {
                "original_path": str(path),
                "doc_index": doc_index,
                "filename": file_name,
            }

            if path_metadata.get("zip_source"):
                metadata["zip_source"] = path_metadata["zip_source"]
                metadata["zip_path"] = path_metadata["zip_path"]

            return [
                TrackedItem(
                    content=content, id=item_id, sources=[item_id], metadata=metadata
                )
            ]

    def load_documents(self) -> List["TrackedItem"]:
        """Load documents and wrap in TrackedItem for provenance tracking.

        Returns:
            List of TrackedItem objects. Cached after first load.
        """
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
                        content=doc,
                        id=f"doc_{idx}",
                        sources=[f"doc_{idx}"],
                        metadata={"doc_index": idx},
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
            tracked_docs = []
            doc_idx = 0

            for path, path_metadata in items:
                tracked_docs.extend(
                    self._create_tracked_items_from_file(path, path_metadata, doc_idx)
                )
                doc_idx += 1

            self.documents = tracked_docs
        else:
            # Need to unpack - document_paths contains string paths
            with unpack_zip_to_temp_paths_if_needed(self.document_paths) as items:
                tracked_docs = []
                doc_idx = 0

                for path, path_metadata in items:
                    tracked_docs.extend(
                        self._create_tracked_items_from_file(
                            path, path_metadata, doc_idx
                        )
                    )
                    doc_idx += 1

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

        # Apply sampling/slicing if requested
        original_count = len(self.documents)
        if self.sample_n is not None:
            random.seed(self.seed)
            if self.sample_n < original_count:
                self.documents = random.sample(self.documents, self.sample_n)
                logger.info(
                    f"Randomly sampled {self.sample_n} from {original_count} documents/rows"
                )
            else:
                logger.warning(
                    f"sample_n ({self.sample_n}) >= document count ({original_count}), using all documents"
                )
        elif self.head_n is not None:
            if self.head_n < original_count:
                self.documents = self.documents[: self.head_n]
                logger.info(
                    f"Taking first {self.head_n} from {original_count} documents/rows"
                )
            else:
                logger.warning(
                    f"head_n ({self.head_n}) >= document count ({original_count}), using all documents"
                )

        return self.documents


async def run_node(node):
    """Execute DAG node and update its output.

    Raises:
        Exception: Propagates node execution failures
    """
    try:
        result = await node.run()
        logger.debug(f"COMPLETED: {node.name}\n")
        return result
    except Exception as e:
        logger.error(f"Node {node.name} failed: {e}")
        raise e


def get_template_variables(template_string: str) -> Set[str]:
    """Extract all variables from Jinja2 template (e.g., '{a} {{b}}' -> {'a', 'b'})."""
    env = Environment()
    ast = env.parse(template_string)
    return meta.find_undeclared_variables(ast)


def render_strict_template(template_str: str, context: dict) -> str:
    """Render Jinja2 template with StrictUndefined (fails on missing variables)."""
    env = Environment(undefined=StrictUndefined)
    template = env.from_string(template_str)
    return template.render(**context)


@dataclass(frozen=True)
class Edge:
    """DAG edge representing dependency between nodes."""

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
    "BatchList",  # Forward reference to avoid circular import
]


class DAG(BaseModel):
    """Directed Acyclic Graph for pipeline execution with parallel batch processing."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    default_context: Dict[str, Any] = {}
    default_config: Dict[str, Union[str, int, float]] = {}

    nodes: List["DAGNodeUnion"] = Field(default_factory=list)
    config: Optional[DAGConfig] = Field(default_factory=DAGConfig, exclude=False)

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
        """Compute dependency edges from node inputs."""
        all_edges = []
        for node in self.nodes:
            for input_ref in node.inputs:
                if input_ref in [i.name for i in self.nodes]:
                    all_edges.append(Edge(from_node=input_ref, to_node=node.name))

        return all_edges

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of DAG structure with node type shapes."""
        from soak.visualization import dag_to_mermaid

        return dag_to_mermaid(self)

    def get_execution_order(self) -> List[List[str]]:
        """Get execution order as batches of nodes that can run in parallel.

        Returns:
            List of batches (lists of node names). Nodes within a batch can run concurrently.

        Raises:
            ValueError: If circular dependency detected
        """
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
        """Node name → node instance mapping."""
        return {i.name: i for i in self.nodes}

    def cancel(self):
        """Cancel running DAG execution."""
        if self.cancel_scope is not None:
            self.cancel_scope.cancel()
            logger.warning(f"DAG {self.name} cancelled")

    async def run(self):
        """Execute DAG by running nodes in dependency-ordered batches.

        Returns:
            Tuple of (DAG instance, error string or None)

        Raises:
            Exception: On timeout (SOAK_MAX_RUNTIME) or missing credentials
        """
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

            # aggregate costs after all nodes complete
            self._aggregate_costs()

            return self, None
        except Exception as e:
            import traceback

            err = f"DAG execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(err)
            return self, str(e)

    def _aggregate_costs(self) -> None:
        """Aggregate costs from all completion nodes and store summary."""
        total_cost = 0.0
        fresh_cost = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        fresh_count = 0
        cached_count = 0
        has_unknown_costs = False
        all_costs_unknown = True
        by_node = {}

        for node in self.nodes:
            # only CompletionDAGNode instances have cost tracking
            if hasattr(node, "_total_cost"):
                node_cost = node._total_cost
                node_prompt_tokens = node._prompt_tokens
                node_completion_tokens = node._completion_tokens

                total_cost += node_cost
                total_prompt_tokens += node_prompt_tokens
                total_completion_tokens += node_completion_tokens

                # check if this node has unknown costs by examining its output
                node_has_unknown = self._node_has_unknown_costs(node)
                node_all_unknown = self._node_all_costs_unknown(node)

                # extract cache stats from node output
                node_fresh_cost, node_fresh_count, node_cached_count = (
                    self._node_cache_stats(node)
                )
                fresh_cost += node_fresh_cost
                fresh_count += node_fresh_count
                cached_count += node_cached_count

                if node_has_unknown:
                    has_unknown_costs = True
                if not node_all_unknown:
                    all_costs_unknown = False

                by_node[node.name] = {
                    "cost": node_cost,
                    "fresh_cost": node_fresh_cost,
                    "prompt_tokens": node_prompt_tokens,
                    "completion_tokens": node_completion_tokens,
                    "fresh_count": node_fresh_count,
                    "cached_count": node_cached_count,
                    "has_unknown": node_has_unknown,
                }

        # store as instance attribute (not in model fields)
        self._cost_summary = {
            "total_cost": total_cost,
            "fresh_cost": fresh_cost,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "fresh_count": fresh_count,
            "cached_count": cached_count,
            "has_unknown_costs": has_unknown_costs,
            "all_costs_unknown": all_costs_unknown,
            "by_node": by_node,
        }

    def _node_has_unknown_costs(self, node) -> bool:
        """Check if node has any unknown costs by examining its ChatterResult outputs"""
        if not hasattr(node, "output") or node.output is None:
            return False

        # handle different output types
        from struckdown import ChatterResult

        outputs = []
        if isinstance(node.output, list):
            outputs = [o for o in node.output if isinstance(o, ChatterResult)]
        elif isinstance(node.output, dict):
            outputs = [o for o in node.output.values() if isinstance(o, ChatterResult)]
        elif isinstance(node.output, ChatterResult):
            outputs = [node.output]

        return any(result.has_unknown_costs for result in outputs)

    def _node_all_costs_unknown(self, node) -> bool:
        """Check if all node costs are unknown"""
        if not hasattr(node, "output") or node.output is None:
            return True

        from struckdown import ChatterResult

        outputs = []
        if isinstance(node.output, list):
            outputs = [o for o in node.output if isinstance(o, ChatterResult)]
        elif isinstance(node.output, dict):
            outputs = [o for o in node.output.values() if isinstance(o, ChatterResult)]
        elif isinstance(node.output, ChatterResult):
            outputs = [node.output]

        if not outputs:
            return True

        return all(result.all_costs_unknown for result in outputs)

    def _node_cache_stats(self, node) -> Tuple[float, int, int]:
        """Extract cache statistics from node outputs

        Returns:
            Tuple of (fresh_cost, fresh_count, cached_count)
        """
        if not hasattr(node, "output") or node.output is None:
            return 0.0, 0, 0

        from struckdown import ChatterResult

        outputs = []
        if isinstance(node.output, list):
            outputs = [o for o in node.output if isinstance(o, ChatterResult)]
        elif isinstance(node.output, dict):
            outputs = [o for o in node.output.values() if isinstance(o, ChatterResult)]
        elif isinstance(node.output, ChatterResult):
            outputs = [node.output]

        fresh_cost = sum(result.fresh_cost for result in outputs)
        fresh_count = sum(result.fresh_call_count for result in outputs)
        cached_count = sum(result.cached_call_count for result in outputs)

        return fresh_cost, fresh_count, cached_count

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for the pipeline run.

        Returns:
            Dict with total_cost, total_prompt_tokens, total_completion_tokens, and per-node breakdown
        """
        return getattr(
            self,
            "_cost_summary",
            {
                "total_cost": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "by_node": {},
            },
        )

    def get_dependencies_for_node(self, node_name: str) -> Set[str]:
        """Get nodes that must complete before this node can run."""

        dependencies = set()

        # set[edge for edge in self.edges if edge.to_node == node_name]
        for edge in self.edges:
            if edge.to_node == node_name:
                dependencies.add(edge.from_node)

        return dependencies

    def add_node(self, node: "DAGNode"):
        """Add node to DAG and set its dag reference."""
        # if self.nodes_dict.get(node.name):
        #     raise ValueError(f"Node '{node.name}' already exists in DAG")
        node.dag = self
        self.nodes.append(node)

    def get_required_context_variables(self):
        """Extract context variables required by node templates (excluding node names)."""
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


# Import BatchList at end of module to avoid circular import
# (DAG is now fully defined, so BatchList can safely import from it)
from .nodes.batch import BatchList  # noqa: E402, F401
