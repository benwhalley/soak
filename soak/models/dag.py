"""DAG (Directed Acyclic Graph) execution engine for pipelines."""

import itertools
import logging
import pdb
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (TYPE_CHECKING, Annotated, Any, Callable, Dict, List, Optional, Set,
                    Tuple, Union)

import anyio
from jinja2 import Environment, StrictUndefined, meta
from pydantic import BaseModel, Field, model_validator
from struckdown import LLM, ChatterResult, LLMCredentials

from soak.document_utils import (extract_text, get_scrubber, is_spreadsheet,
                                 unpack_zip_to_temp_paths_if_needed)
from soak.export_utils import export_to_csv
from soak.models.base import (SOAK_MAX_RUNTIME, TrackedItem,
                              get_default_llm_credentials)
from soak.models.context import ContextVariable, get_context_defaults, normalize_context
from soak.models.cost_tracker import GlobalCostTracker
from soak.models.progress import ProgressManager

if TYPE_CHECKING:
    from .nodes.base import DAGNode
    from .nodes.batch import BatchList

logger = logging.getLogger(__name__)


class DAGConfig(BaseModel):
    """Configuration for DAG execution including documents, model settings, and error handling."""

    document_paths: List[Union[str, tuple[str, Dict[str, Any]]]] = []
    documents: List[Union[str, "TrackedItem"]] = []
    input_source: Optional[str] = None  # summary of input (e.g., "data/*.txt")
    model_name: str = "gpt-4.1-mini"
    models: Dict[str, str] = Field(
        default_factory=dict,
        description="Model alias mappings (e.g., {'default': 'gpt-4.1-mini', 'best': 'gpt-4.1'})",
    )
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words
    extra_context: Dict[str, Any] = {}
    llm_credentials: LLMCredentials = Field(
        default_factory=get_default_llm_credentials, repr=False, exclude=True
    )
    scrub_pii: bool = False
    scrubber_model: str = "en_core_web_md"
    scrubber_salt: str | None = Field(default="42", exclude=True)
    seed: int = 42
    randomise_document_order: bool = (
        True  # shuffle documents to avoid folder-name ordering
    )
    sample_n: int | None = None  # randomly sample N documents/rows
    head_n: int | None = None  # take first N documents/rows
    show_progress: bool = False  # show progress bars during execution

    # embedding configuration
    embedding_model: str = (
        "text-embedding-3-large"  # model for embeddings (use "local/model-name" for sentence-transformers)
    )

    # error handling configuration
    fail_on_context_exceeded: bool = True  # if False, skip item with warning
    skip_content_policy_violations: bool = True  # if False, fail pipeline
    log_failed_prompts: bool = True  # log offending prompts to stderr

    # LLM call configuration
    llm_timeout: float = 90.0  # timeout in seconds for individual LLM calls

    # incremental export configuration
    export_enabled: bool = False  # export nodes as they finish
    export_folder: Optional[Path] = None  # folder to export to
    export_metadata: Dict[str, Any] = Field(default_factory=dict)  # metadata for export

    # node skipping and stopping
    skip_nodes: List[str] = Field(
        default_factory=list
    )  # nodes to skip during execution
    stop_at_node: Optional[str] = None  # stop execution before this node runs

    # debugging
    pdb_on_exception: bool = False  # drop into pdb on unhandled exceptions

    # progress callback for web UI (called when items complete)
    progress_callback: Optional[Callable[[str, int, int, float], None]] = Field(
        default=None, exclude=True
    )  # callback(node_name, done, total, cost)

    # node completion callback for web UI (called when each node finishes)
    node_complete_callback: Optional[Callable[["DAGNode"], None]] = Field(
        default=None, exclude=True
    )  # callback(node) - called when node completes (success or failure)

    # report configuration
    report_texts: List[str] = Field(
        default_factory=lambda: ["narrative"]
    )  # Transform nodes to show as text in HTML reports

    def get_model(self, alias: Optional[str] = None):
        """Create LLM instance with configured model.

        Args:
            alias: Optional model alias (e.g., 'default', 'best').
                   If provided and exists in models dict, uses that model.
                   Otherwise falls back to model_name.
        """
        model_name = self.resolve_model_alias(alias)
        return LLM(model_name=model_name)

    def resolve_model_alias(self, alias: Optional[str] = None) -> str:
        """Resolve a model alias to an actual model ID.

        Args:
            alias: Model alias (e.g., 'default', 'best').
                   If None, returns model_name.
                   If alias exists in models dict, returns mapped model ID.
                   Otherwise returns alias directly (assumes it's a model ID).

        Returns:
            Resolved model ID string.
        """
        if alias is None:
            return self.model_name

        # Check if alias exists in models mapping
        if alias in self.models:
            return self.models[alias]

        # Check if it's actually a model ID (not an alias)
        # Return as-is to support both alias and direct model ID usage
        return alias

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

        # Shuffle documents to avoid folder-name ordering bias
        if self.randomise_document_order:
            random.seed(self.seed)
            random.shuffle(self.documents)
            logger.debug(
                f"Shuffled {len(self.documents)} documents with seed={self.seed}"
            )

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
    logger.info(f"Starting node: {node.name} ({node.type})")
    try:
        result = await node.run()
        logger.debug(f"COMPLETED: {node.name}\n")

        # Export node if incremental export is enabled
        if (
            node.dag.config.export_enabled
            and hasattr(node.dag, "_node_export_folders")
            and node.name in node.dag._node_export_folders
        ):
            node_folder = node.dag._node_export_folders[node.name]
            unique_id = node.dag.config.export_metadata.get("unique_id", "")
            await anyio.to_thread.run_sync(node.export, node_folder, unique_id)
            logger.info(f"✓ Exported node: {node.name} to {node_folder.name}")

        # Call node completion callback if set (for web UI incremental saving)
        if node.dag.config.node_complete_callback:
            await anyio.to_thread.run_sync(
                node.dag.config.node_complete_callback, node
            )

        return result
    except Exception as e:
        logger.error(f"Node {node.name} failed: {e}")
        # Store error on node for callback access
        node._error = e
        # Call node completion callback on failure too (for partial result saving)
        if node.dag.config.node_complete_callback:
            try:
                await anyio.to_thread.run_sync(
                    node.dag.config.node_complete_callback, node
                )
            except Exception as callback_err:
                logger.warning(f"Node completion callback failed: {callback_err}")
        raise e


def get_template_variables(template_string: str) -> Set[str]:
    """Extract all variables from Jinja2 template (e.g., '{a} {{b}}' -> {'a', 'b'})."""
    env = Environment()
    ast = env.parse(template_string)
    return meta.find_undeclared_variables(ast)


def render_strict_template(template_str: str, context: dict) -> str:
    """Render Jinja2 template with StrictUndefined and auto-escaping for struckdown syntax.

    Uses struckdown's finalize function to automatically escape special syntax in
    context variables. This prevents prompt injection attacks where user-provided
    content could contain struckdown commands.
    """
    from struckdown import struckdown_finalize

    env = Environment(
        undefined=StrictUndefined,
        finalize=struckdown_finalize,  # Auto-escape struckdown syntax
    )
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
        "VerifyQuotes",
        "Classifier",
        "Filter",
        "GroupBy",
        "Ungroup",
        "Scrub",
        "Cluster",
    ],
    Field(discriminator="type"),
]

# Type alias for node outputs
# Note: Any at end handles deserialized ChatterResult data (ChatterResult isn't Pydantic)
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
    List[Any],  # fallback for serialized outputs (may contain None or dicts)
    Dict[str, Any],  # fallback for other serialized outputs
]


class DAG(BaseModel):
    """Directed Acyclic Graph for pipeline execution with parallel batch processing."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    default_context: Dict[str, Any] = {}
    default_config: Dict[str, Any] = {}  # Allow any value including nested dicts (e.g., models: {default: gpt-4})
    template_dirs: List[str] = []  # additional template search directories
    scrub: Optional[bool] = None  # if False, suppress PII warning

    nodes: List["DAGNodeUnion"] = Field(default_factory=list)
    config: Optional[DAGConfig] = Field(default_factory=DAGConfig, exclude=False)
    cost_tracker: Optional[GlobalCostTracker] = Field(default=None, exclude=True)
    progress_manager: Optional[ProgressManager] = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add defaults for config fields
        for k, v in self.default_config.items():
            if hasattr(self.config, k) and k not in self.config.model_fields_set:
                setattr(self.config, k, v)
        # initialize cost tracker
        if self.cost_tracker is None:
            self.cost_tracker = GlobalCostTracker()

    @model_validator(mode="after")
    def validate_node_templates(self) -> "DAG":
        """Validate that nodes requiring templates have them defined."""
        # Node types that require templates
        template_required_types = {"Map", "Transform", "Classifier"}

        for node in self.nodes:
            if node.type in template_required_types:
                # Check if template exists and is not None/empty
                if not hasattr(node, "template") or not node.template:
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

    def _validate_execution_options(self) -> Tuple[Set[str], Optional[str]]:
        """Validate skip_nodes and stop_at_node configuration.

        Returns:
            Tuple of (skip_nodes set, stop_at_node name or None)
        """
        node_names = {n.name for n in self.nodes}

        # Validate skip_nodes
        skip_nodes = set(self.config.skip_nodes)
        if skip_nodes:
            unknown_skips = skip_nodes - node_names
            if unknown_skips:
                logger.warning(
                    f"Skip nodes not found in DAG: {', '.join(unknown_skips)}"
                )
            valid_skips = skip_nodes & node_names
            if valid_skips:
                logger.info(f"Skipping nodes: {', '.join(valid_skips)}")

        # Validate stop_at_node
        stop_at_node = self.config.stop_at_node
        if stop_at_node:
            if stop_at_node not in node_names:
                logger.warning(f"Stop-at node not found in DAG: {stop_at_node}")
                stop_at_node = None
            else:
                logger.info(f"Will stop at node: {stop_at_node}")

        return skip_nodes, stop_at_node

    async def _execute_batches(
        self, skip_nodes: Set[str], stop_at_node: Optional[str]
    ) -> None:
        """Execute DAG batches with skip and stop-at logic.

        Args:
            skip_nodes: Set of node names to skip
            stop_at_node: Stop execution before this node runs (None to run all)
        """
        for batch in self.get_execution_order():
            # Check if we should stop before this batch
            if stop_at_node and stop_at_node in batch:
                logger.info(f"Stopping execution at node: {stop_at_node}")
                break

            # Execute batch concurrently
            with anyio.fail_after(SOAK_MAX_RUNTIME):
                async with anyio.create_task_group() as tg:
                    for name in batch:
                        if name in skip_nodes:
                            logger.debug(f"Skipping node: {name}")
                            continue
                        tg.start_soon(run_node, self.nodes_dict[name])

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

            # Validate skip_nodes and stop_at_node
            skip_nodes, stop_at_node = self._validate_execution_options()

            # Pre-compute node export folders if incremental export is enabled
            if self.config.export_enabled and self.config.export_folder:
                self._prepare_incremental_export()

            # Create progress manager if progress is enabled
            if self.config.show_progress:
                self.progress_manager = ProgressManager(self.cost_tracker)
                with self.progress_manager:
                    await self._execute_batches(skip_nodes, stop_at_node)
                self.progress_manager = None
            else:
                await self._execute_batches(skip_nodes, stop_at_node)

            # aggregate costs after all nodes complete
            self._aggregate_costs()

            # finalize incremental export with cost data
            if self.config.export_enabled:
                self._finalize_incremental_export()

            return self, None
        except Exception as e:
            import traceback

            err = f"DAG execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(err)

            if self.config.pdb_on_exception:
                # unwrap ExceptionGroups from async TaskGroup to get the actual exception
                exc = e
                while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                    exc = exc.exceptions[0]
                pdb.post_mortem(exc.__traceback__)

            return self, str(e)

    def _prepare_incremental_export(self) -> None:
        """Prepare for incremental node export by creating folder structure and metadata.

        Creates the export folder, writes meta.txt, and computes node export paths.
        Stores node_export_folders mapping for use by run_node().
        """
        output_dir = Path(self.config.export_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preparing incremental export to {output_dir}")

        # Write metadata file
        metadata = self.config.export_metadata
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

        # Create node_to_order mapping and store export folders
        self._node_export_folders = {}
        for batch_idx, batch in enumerate(execution_order):
            for node_name in batch:
                order = batch_idx + 1
                node = self.nodes_dict[node_name]
                folder_name = f"{order:02d}_{node.type}_{node.name}"
                self._node_export_folders[node_name] = output_dir / folder_name

        logger.debug(
            f"Prepared export folders for {len(self._node_export_folders)} nodes"
        )

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
        """Extract cache statistics from node outputs and _llm_results

        Returns:
            Tuple of (fresh_cost, fresh_count, cached_count)
        """
        from struckdown import ChatterResult

        outputs = []

        # Primary: Check _llm_results attribute (CompletionDAGNode standard)
        if hasattr(node, "_llm_results") and node._llm_results:
            outputs.extend(node._llm_results)
        # Fallback: Check node.output ONLY if _llm_results is empty/missing (backward compatibility)
        elif hasattr(node, "output") and node.output is not None:
            if isinstance(node.output, list):
                outputs.extend([o for o in node.output if isinstance(o, ChatterResult)])
            elif isinstance(node.output, dict):
                outputs.extend(
                    [o for o in node.output.values() if isinstance(o, ChatterResult)]
                )
            elif isinstance(node.output, ChatterResult):
                outputs.append(node.output)

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

    def _format_cost_summary_text(self, cost_summary: Dict[str, Any]) -> str:
        """Format cost summary as text for meta.txt display.

        Args:
            cost_summary: Dict from get_cost_summary()

        Returns:
            Formatted text block with cost information
        """
        lines = [
            "\nCost Summary",
            "============",
            f"Total Cost: ${cost_summary['total_cost']:.4f}",
            f"Fresh Cost: ${cost_summary.get('fresh_cost', 0.0):.4f}",
            f"Total Tokens: {cost_summary['total_prompt_tokens']:,} in / {cost_summary['total_completion_tokens']:,} out",
            f"Calls: {cost_summary.get('fresh_count', 0)} fresh, {cost_summary.get('cached_count', 0)} cached",
        ]

        if cost_summary.get("has_unknown_costs", False):
            lines.append("Note: Some costs are unknown (check costs.csv for details)")

        return "\n".join(lines)

    def _export_costs_csv(self, output_dir: Path) -> None:
        """Export per-node cost breakdown as CSV file.

        Creates costs.csv in the output directory with columns:
        node_name, node_type, cost, fresh_cost, prompt_tokens, completion_tokens,
        fresh_count, cached_count, has_unknown

        Args:
            output_dir: Directory to write costs.csv to
        """
        import pandas as pd

        cost_summary = self.get_cost_summary()
        by_node = cost_summary.get("by_node", {})

        if not by_node:
            logger.debug("No node costs to export")
            return

        rows = []
        for node_name, node_data in by_node.items():
            # get node type from nodes_dict
            node = self.nodes_dict.get(node_name)
            node_type = node.type if node else "Unknown"

            rows.append(
                {
                    "node_name": node_name,
                    "node_type": node_type,
                    "cost": node_data["cost"],
                    "fresh_cost": node_data.get("fresh_cost", 0.0),
                    "prompt_tokens": node_data["prompt_tokens"],
                    "completion_tokens": node_data["completion_tokens"],
                    "fresh_count": node_data.get("fresh_count", 0),
                    "cached_count": node_data.get("cached_count", 0),
                    "has_unknown": node_data.get("has_unknown", False),
                }
            )

        df = pd.DataFrame(rows)
        csv_path = output_dir / "costs.csv"
        export_to_csv(df, csv_path)
        logger.info(f"✓ Exported cost breakdown to costs.csv")

    def _export_diagram(self, output_dir: Path) -> None:
        """Export pipeline diagram as DOT and PDF files.

        Args:
            output_dir: Directory to write diagram files to
        """
        try:
            from soak.visualization import (dag_to_graphviz,
                                            render_graphviz_to_pdf)

            dot_definition = dag_to_graphviz(self)

            # save .dot file
            dot_path = output_dir / "pipeline_diagram.dot"
            dot_path.write_text(dot_definition)
            logger.info(f"✓ Saved DOT definition to {dot_path.name}")

            # render to PDF
            pdf_path = output_dir / "pipeline_diagram.pdf"
            render_graphviz_to_pdf(dot_definition, pdf_path)
        except Exception as e:
            logger.warning(f"Could not export Graphviz diagram: {e}")

    def _finalize_incremental_export(self) -> None:
        """Finalize incremental export with cost data after all nodes complete.

        Updates meta.txt with cost summary and creates costs.csv.
        Called at the end of run() after _aggregate_costs().
        """
        if not self.config.export_enabled or not self.config.export_folder:
            return

        output_dir = Path(self.config.export_folder)

        # update meta.txt with cost summary
        cost_summary = self.get_cost_summary()
        cost_text = self._format_cost_summary_text(cost_summary)

        meta_path = output_dir / "meta.txt"
        if meta_path.exists():
            content = meta_path.read_text()
            content += f"\n{cost_text}"
            meta_path.write_text(content)

        # export costs.csv
        self._export_costs_csv(output_dir)

        # export diagram
        self._export_diagram(output_dir)

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

    def get_context_variables(self) -> Dict[str, ContextVariable]:
        """Get normalized context variables with metadata.

        Converts default_context to ContextVariable instances, supporting both
        simple values and rich definitions with description/help_text.

        Returns:
            Dict mapping variable names to ContextVariable instances
        """
        return normalize_context(self.default_context)

    def get_context_defaults(self) -> Dict[str, Any]:
        """Get default values from context variables.

        Extracts just the default values for use in template rendering,
        handling both simple values and rich ContextVariable definitions.

        Returns:
            Dict mapping variable names to their default values
        """
        return get_context_defaults(self.default_context)

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

        # add cost summary
        cost_summary = self.get_cost_summary()
        cost_text = self._format_cost_summary_text(cost_summary)
        meta_content += f"\n{cost_text}\n"

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

        # export cost breakdown
        self._export_costs_csv(output_dir)

        # export diagram
        self._export_diagram(output_dir)

        logger.debug(f"Export complete: {output_dir}")


# Import BatchList at end of module to avoid circular import
# (DAG is now fully defined, so BatchList can safely import from it)
from .nodes.batch import BatchList  # noqa: E402, F401
