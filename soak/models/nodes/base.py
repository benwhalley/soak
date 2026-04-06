"""Base node classes for DAG execution."""

import json
import logging
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import anyio
import nltk
import numpy as np
import pandas as pd
import tiktoken
from box import Box
from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from struckdown import LLM, StruckdownResult, LLMError, complete_async
from struckdown.parsing import parse_syntax
from tqdm import tqdm

from soak.error_handlers import log_error_to_stderr, should_continue_pipeline
from soak.models.base import TrackedItem, extract_content, get_action_lookup
from soak.models.context import get_context_defaults
from soak.models.dag import (DAG, OutputUnion, render_strict_template,
                             render_template_preserve_undefined)

logger = logging.getLogger(__name__)


class _TemplateProxy:
    """Read-only proxy over a StruckdownResult for template use.

    Wraps list outputs in container types for __str__ rendering
    (e.g. [Theme, ...] -> Themes, [Code, ...] -> CodeList).
    Resolves theme.code_slugs -> theme.resolved_code_refs using codes from the DAG.
    """

    def __init__(self, cr, all_codes: list = None):
        self._cr = cr
        self._all_codes = all_codes or []

    def _resolve_theme_codes(self, theme) -> None:
        """Resolve code_slugs on a Theme using collected codes from the DAG."""
        if not self._all_codes or not theme.code_slugs or theme.resolved_code_refs:
            return
        code_by_slug = {c.slug: c for c in self._all_codes}
        matched = [
            c.model_dump() for slug in theme.code_slugs
            if (c := code_by_slug.get(slug))
        ]
        if matched:
            theme.resolved_code_refs = matched

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        from soak.models.base import Code, CodeList, Theme, Themes
        if name in self._cr.results:
            output = self._cr.results[name].output
            # wrap typed lists in container types for __str__ rendering
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, Theme):
                    for t in output:
                        self._resolve_theme_codes(t)
                    return Themes(themes=output)
                if isinstance(first, Code):
                    return CodeList(codes=output)
            return output
        raise AttributeError(
            f"No slot '{name}' in StruckdownResult "
            f"(slots: {list(self._cr.results.keys())})"
        )

    def __str__(self):
        return str(self._cr)

    def __repr__(self):
        return f"<_TemplateProxy slots={list(self._cr.results.keys())}>"


def _collect_codes_from_dag(dag) -> list:
    """Scan all completed DAG node outputs for Code objects."""
    from struckdown import StruckdownResult
    from soak.models.base import Code
    codes = []

    for node in dag.nodes_dict.values():
        if node.output is None:
            continue
        outputs = node.output if isinstance(node.output, list) else [node.output]
        for item in outputs:
            if isinstance(item, Code):
                codes.append(item)
            elif isinstance(item, StruckdownResult):
                for seg in item.results.values():
                    out = seg.output
                    if isinstance(out, list):
                        codes.extend(c for c in out if isinstance(c, Code))
                    elif isinstance(out, Code):
                        codes.append(out)
    return codes


def _prepare_for_template(output, source_node_type: str = "", all_codes: list = None):
    """Wrap StruckdownResult outputs in _TemplateProxy for downstream templates.

    Transform/TransformReduce: unwrap single-element list, return one proxy.
    Map/Filter/Classifier: return list of proxies.
    Bare StruckdownResult: return one proxy.
    Anything else: pass through unchanged.
    """
    from struckdown import StruckdownResult
    if source_node_type in ("Transform", "TransformReduce"):
        if (isinstance(output, list) and len(output) == 1
                and isinstance(output[0], StruckdownResult)):
            return _TemplateProxy(output[0], all_codes)
    if isinstance(output, list) and output and isinstance(output[0], StruckdownResult):
        return [_TemplateProxy(cr, all_codes) for cr in output]
    if isinstance(output, StruckdownResult):
        return _TemplateProxy(output, all_codes)
    return output


class DAGNode(BaseModel):
    # this used for reserialization
    model_config = {"discriminator": "type", "extra": "forbid"}
    type: str = Field(default_factory=lambda self: type(self).__name__, exclude=False)

    dag: Optional["DAG"] = Field(default=None, exclude=True)

    name: str
    inputs: Optional[List[str]] = []
    template: Optional[str] = None
    output: Optional[OutputUnion] = Field(default=None)
    model: Optional[str] = Field(
        default=None,
        description="Model alias (e.g., 'default', 'best') or model ID to use for this node",
    )

    def get_model(self):
        # First check explicit model_name attribute (legacy)
        model_name = getattr(self, "model_name", None)
        if model_name:
            return LLM(model_name=model_name)

        # Check model alias/ID field
        if self.model:
            # Resolve alias to actual model ID via DAG config
            resolved_model = self.dag.config.resolve_model_alias(self.model)
            return LLM(model_name=resolved_model)

        # Fall back to default model from DAG config
        return self.dag.config.get_model()

    def validate_template(self):
        try:
            Environment().parse(self.template)
            return True
        except TemplateSyntaxError as e:
            raise e
            logger.error(f"Template syntax error: {e}")
            return False

    async def run(self, items: List[Any] = None) -> List[Any]:
        logger.debug(f"\n\nRunning `{self.name}` ({self.__class__.__name__})\n\n")

    async def skip(self) -> Optional[List[Any]]:
        """Called when node is in skip_nodes list.

        Default behavior: return None (no output, downstream sees nothing).
        Nodes can override for passthrough behavior.

        Returns:
            None by default. Override to return processed output for passthrough.
        """
        logger.debug(f"Skipping node '{self.name}' (default: no output)")
        return None

    @property
    def context(self) -> Dict[str, Any]:
        # Extract just the default values from context variables (not the full dicts)
        ctx = get_context_defaults(self.dag.default_context)

        # merge in extra_context from config (includes persona, research_question, etc.)
        ctx.update(self.dag.config.extra_context)

        # if there are no inputs, assume we'll be using input documents
        if not self.inputs:
            self.inputs = ["documents"]

        if "documents" in self.inputs:
            ctx["documents"] = self.dag.config.load_documents()

        # collect all codes from completed DAG nodes for theme resolution
        all_codes = _collect_codes_from_dag(self.dag)

        prev_nodes = {k: self.dag.nodes_dict.get(k) for k in self.inputs}
        prev_output = {}
        for k, v in prev_nodes.items():
            if v and v.output is not None:
                prev_output[k] = _prepare_for_template(
                    v.output, source_node_type=v.type, all_codes=all_codes,
                )
        ctx.update(prev_output)

        return ctx

    def get_input_items(self) -> Optional[List[Any]]:
        """Get the input items that were processed by this node.

        Returns a flat list, flattening any BatchList structures.
        """
        from .batch import BatchList

        if not self.inputs or not self.dag:
            return None

        # Get the first input (most common case)
        input_name = self.inputs[0]
        if input_name == "documents":
            return self.dag.config.documents
        elif input_name in self.dag.nodes_dict:
            input_node = self.dag.nodes_dict[input_name]
            output = input_node.output if input_node.output else None

            # Flatten BatchList if needed
            if isinstance(output, BatchList):
                return output.flatten_all()

            return output

        return None

    def result(self) -> Dict[str, Any]:
        """Extract results from this node. Override in subclasses for node-specific results."""
        metadata = {
            "name": self.name,
            "type": self.type,
            "inputs": self.inputs,
        }

        # Add template text if available
        if hasattr(self, "template") and self.template:
            metadata["template"] = self.template

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
                        ).write_text(item.to_json())
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
    thinking: Optional[str] = None

    @field_validator("thinking", mode="before")
    @classmethod
    def coerce_thinking(cls, v):
        if v is False:
            return "off"
        return v

    # cost tracking (private attributes, not serialized)
    _total_cost: float = PrivateAttr(default=0.0)
    _fresh_cost: float = PrivateAttr(default=0.0)  # cost of non-cached calls only
    _prompt_tokens: int = PrivateAttr(default=0)
    _completion_tokens: int = PrivateAttr(default=0)
    _llm_results: List[StruckdownResult] = PrivateAttr(default_factory=list)
    _input_count: int = PrivateAttr(
        default=0
    )  # tracks expected input count for progress

    def _accumulate_costs(self, result: StruckdownResult) -> None:
        """Extract and accumulate costs from a StruckdownResult"""
        self._total_cost += result.total_cost
        self._fresh_cost += result.fresh_cost  # only non-cached calls
        self._prompt_tokens += result.prompt_tokens
        self._completion_tokens += result.completion_tokens

        # notify global cost tracker if available
        if self.dag and self.dag.cost_tracker:
            self.dag.cost_tracker.add_result(result)

    def get_llm_kwargs(self) -> Dict[str, Any]:
        """Build extra_kwargs dict for LLM API calls.

        Constructs the standard kwargs used across all LLM calls including
        temperature, seed, timeout, and optional max_tokens.

        Returns:
            Dict with LLM call parameters
        """
        kwargs = {
            "temperature": self.temperature,
            "seed": self.dag.config.seed,
            "timeout": self.dag.config.llm_timeout,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        return kwargs

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


class ItemsNode(DAGNode):
    """Base class for nodes that process multiple items.

    Provides automatic BatchList handling:
    - Operates at the innermost level of nested BatchLists
    - Preserves grouping structure through processing
    - Subclasses implement process_items() for their logic
    """

    def get_input_data(self) -> Union[List[Any], Any]:
        """Get raw input data (may be flat list, BatchList, or single item)."""
        if not self.inputs:
            return self.dag.config.load_documents()

        input_name = self.inputs[0]
        if input_name == "documents":
            return self.dag.config.load_documents()

        return self.context[input_name]

    async def run(self) -> Union[List[Any], Any]:
        """Orchestrate processing with automatic BatchList handling."""
        await super().run()
        input_data = self.get_input_data()

        # Create single progress bar for all items (including batched)
        progress_bar = None
        if self.dag.config.show_progress:
            # Calculate total items (BatchList.__len__ uses flatten_all())
            # Avoid counting string length - single string inputs are 1 item
            from .batch import BatchList

            if isinstance(input_data, (list, BatchList)):
                total_items = len(input_data)
            else:
                total_items = 1

            # Use progress manager if available for coordinated display
            if self.dag.progress_manager:
                if isinstance(self, CompletionDAGNode) and self.dag.cost_tracker:
                    progress_bar = self.dag.progress_manager.create_cost_progress_bar(
                        total=total_items,
                        node_name=f"{self.type}: {self.name}",
                        unit="item",
                    )
                else:
                    progress_bar = self.dag.progress_manager.create_progress_bar(
                        total=total_items,
                        desc=f"{self.type}: {self.name}",
                        unit="item",
                    )
            # fallback to direct tqdm if no progress manager
            elif isinstance(self, CompletionDAGNode) and self.dag.cost_tracker:
                from soak.models.progress import CostProgressBar

                progress_bar = CostProgressBar(
                    tracker=self.dag.cost_tracker,
                    node_name=f"{self.type}: {self.name}",
                    total=total_items,
                    unit="item",
                )
            else:
                # pad description to match CostProgressBar alignment
                desc = f"{self.type}: {self.name}".ljust(35)
                progress_bar = tqdm(
                    total=total_items,
                    desc=desc,
                    unit="item",
                    file=sys.stderr,
                    ncols=120,
                    leave=True,
                    mininterval=0.1,
                )

        try:
            result = await self._run_recursive(input_data, progress_bar)
            self.output = result
            return result
        finally:
            if progress_bar:
                progress_bar.close()

    async def _run_recursive(
        self, data: Union[List[Any], Any], progress_bar: Optional[Any] = None
    ) -> Union[List[Any], Any]:
        """Recursively process nested BatchLists with concurrent batch processing.

        Args:
            data: Input data (flat list, BatchList, or items)
            progress_bar: Optional tqdm progress bar to update

        Returns:
            Processed data maintaining the same structure
        """
        # Import here to avoid circular import
        from .batch import BatchList

        if not isinstance(data, BatchList):
            # Flat input: process directly
            items = data if isinstance(data, list) else [data]
            return await self.process_items(items, progress_bar)

        # Check if nested
        if data.is_nested():
            # Nested BatchList: recurse on each inner BatchList concurrently
            inner_results = [None] * len(data.batches)

            async with anyio.create_task_group() as tg:
                for idx, inner_batch in enumerate(data.batches):

                    async def process_inner(index=idx, batch=inner_batch):
                        inner_results[index] = await self._run_recursive(
                            batch, progress_bar
                        )

                    tg.start_soon(process_inner)

            return BatchList(
                batches=inner_results,
                group_field=data.group_field,
                group_keys=data.group_keys,
            )
        else:
            # Single-level BatchList: process each batch concurrently
            batch_results = [None] * len(data.batches)

            async with anyio.create_task_group() as tg:
                for idx, batch in enumerate(data.batches):

                    async def process_batch(index=idx, batch_items=batch):
                        batch_results[index] = await self.process_items(
                            batch_items, progress_bar
                        )

                    tg.start_soon(process_batch)

            # Remove empty batches (Filter can create these)
            non_empty_results = [r for r in batch_results if r]

            return BatchList(
                batches=non_empty_results,
                group_field=data.group_field,
                group_keys=data.group_keys,
            )

    async def process_items(
        self, items: List[Any], progress_bar: Optional[Any] = None
    ) -> List[Any]:
        """Process a flat list of items.

        Subclasses must override this method to implement their processing logic.
        This method is called on each batch at the innermost level of nesting.

        Args:
            items: Flat list of items to process
            progress_bar: Optional tqdm progress bar to update after processing items

        Returns:
            Processed list of items
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process_items()"
        )

    async def get_items(self) -> List[Dict[str, Any]]:
        """Resolve all inputs, then zip together, combining multiple inputs.

        Handles TrackedItem transparently: extracts content for templates while
        preserving source_id and metadata for provenance tracking.
        """
        # Ensure inputs default is set (accessing context property sets it)
        context = self.context

        # resolve futures now (it's lazy to this point)
        input_data = {k: context[k] for k in self.inputs}

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
                    # Extract content for template, preserve metadata and provenance
                    item_dict[key] = val.content
                    item_dict[f"__{key}__tracked_item"] = val  # Keep full reference

                    # Maintain backward compatibility for templates using source_id
                    item_dict[f"__{key}__source_id"] = val.id  # Backward compat
                    item_dict[f"__{key}__metadata"] = val.metadata

                    # Spread metadata into top-level context for easy access
                    # Column names from spreadsheets will be available as {{column_name}}
                    if val.metadata:
                        for meta_key, meta_val in val.metadata.items():
                            # don't override existing keys with metadata
                            if meta_key not in item_dict:
                                item_dict[meta_key] = meta_val
                else:
                    item_dict[key] = val

            # Make first input available as {{input}}
            if self.inputs:
                first_val = values[0]
                if isinstance(first_val, TrackedItem):
                    item_dict["input"] = first_val.content
                    item_dict["source_id"] = (
                        first_val.id
                    )  # Backward compat for templates
                    item_dict["metadata"] = first_val.metadata
                    item_dict["tracked_item"] = (
                        first_val  # Full TrackedItem for post_process
                    )
                else:
                    item_dict["input"] = first_val

            items.append(Box(item_dict))

        return items

    @contextmanager
    def progress_bar(self, items: List, node_type: str = None):
        """Context manager for optional progress bars.

        Creates a progress bar if show_progress is enabled in config,
        otherwise yields None. This eliminates repetitive progress bar
        setup code across nodes.

        Args:
            items: List of items to process (for total count)
            node_type: Optional node type label (defaults to self.type)

        Yields:
            tqdm progress bar instance or None. The bar has a `slots_per_item`
            attribute indicating how many slots per item (for update increments).
        """
        if not self.dag.config.show_progress:
            yield None
            return

        node_type = node_type or self.type
        node_name = f"{node_type}: {self.name}" if node_type != self.type else self.name

        # estimate slots per item from template (for better progress estimation)
        slots_per_item = 1
        if isinstance(self, CompletionDAGNode) and self.template:
            slots_per_item = max(1, len(self.output_keys()))

        total_completions = len(items) * slots_per_item

        # Use progress manager if available for coordinated display
        if self.dag.progress_manager:
            if isinstance(self, CompletionDAGNode) and self.dag.cost_tracker:
                pbar = self.dag.progress_manager.create_cost_progress_bar(
                    total=total_completions,
                    node_name=node_name,
                    unit="call",
                )
            else:
                pbar = self.dag.progress_manager.create_progress_bar(
                    total=total_completions,
                    desc=node_name,
                    unit="call",
                )
        # fallback to direct tqdm if no progress manager
        elif isinstance(self, CompletionDAGNode) and self.dag.cost_tracker:
            from soak.models.progress import CostProgressBar

            pbar = CostProgressBar(
                tracker=self.dag.cost_tracker,
                node_name=node_name,
                total=total_completions,
                unit="call",
            )
        else:
            from tqdm import tqdm

            # pad description to match CostProgressBar alignment
            padded_name = node_name.ljust(35)
            pbar = tqdm(
                total=total_completions,
                desc=padded_name,
                unit="call",
                file=sys.stderr,
                ncols=120,
                leave=True,
                mininterval=0.1,
            )

        # store slots count so callers know how much to increment
        pbar.slots_per_item = slots_per_item

        try:
            yield pbar
        finally:
            pbar.close()


async def default_map_task(template, context, model, credentials, **kwargs):
    """Default map task renders the Step template for each input item and calls the LLM.

    Note: This function does NOT catch LLMError - it lets errors propagate
    to the calling node (Map) which handles them appropriately via managed_llm_call.
    The connection error counter is reset by managed_llm_call wrapper.
    """
    # Pre-render context variables but preserve undefined ones for struckdown.
    # This allows {{min_themes}} to render while keeping {{narrative}} (from [[narrative]])
    # intact for struckdown to fill after checkpoint processing.
    rt = render_template_preserve_undefined(template, context)

    # call complete as async function within the main event loop
    # errors will propagate to the calling node for handling
    result = await complete_async(
        multipart_prompt=rt,
        context=context,
        model=model,
        credentials=credentials,
        extra_kwargs=kwargs,
    )

    return result


async def template_map_task(template, context, **kwargs):
    """Template-only map task that renders Jinja2 template without LLM call.

    Returns the rendered string directly (no StruckdownResult wrapper).
    Used when Map node has function="template" for pure templating.
    """
    return render_strict_template(template, context)


class Split(ItemsNode):
    type: Literal["Split"] = "Split"

    name: str = "chunks"

    chunk_size: int = 20000
    min_split: int = 500
    overlap: int = 0
    split_unit: Literal["chars", "tokens", "words", "sentences", "paragraphs"] = (
        "tokens"
    )
    encoding_name: str = "cl100k_base"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.inputs) > 1:
            raise ValueError("Split node can only have one input")

        if self.chunk_size < self.min_split:
            logger.warning(
                f"Chunk size must be larger than than min_split. Setting min_split to chunk_size // 2 = {self.chunk_size // 2}"
            )
            self.min_split = self.chunk_size // 2

    async def process_items(
        self, items: List[Any], progress_bar: Optional[Any] = None
    ) -> List[TrackedItem]:
        """Split each document in the batch into chunks.

        Args:
            items: List of documents (TrackedItem or strings) in this batch
            progress_bar: Optional progress bar to update after processing each document

        Returns:
            Flat list of all chunks from all documents in the batch
        """
        import numpy as np

        # Split each document and track provenance
        all_chunks = []
        for doc in items:
            if isinstance(doc, TrackedItem):
                chunks = self.split_tracked_document(doc)
            else:
                # Backward compatibility: wrap plain strings
                temp_tracked = TrackedItem(
                    content=doc, id="unknown_doc", sources=["unknown_doc"], metadata={}
                )
                chunks = self.split_tracked_document(temp_tracked)
            all_chunks.extend(chunks)

            # update progress bar after processing each document
            if progress_bar is not None:
                progress_bar.update(1)

        # Calculate stats on content
        if all_chunks:
            lens = [
                len(self.tokenize(chunk.content, method=self.split_unit))
                for chunk in all_chunks
            ]
            logger.debug(
                f"Split '{self.name}': Created {len(all_chunks)} chunks; "
                f"average length ({self.split_unit}): {np.mean(lens).round(1)}, "
                f"max: {max(lens)}, min: {min(lens)}"
            )

        return all_chunks

    def split_tracked_document(self, doc: TrackedItem) -> List[TrackedItem]:
        """Split a TrackedItem into multiple TrackedItems with hierarchical IDs.

        The node name is included in the ID to track which Split node created the chunk.
        Format: parent_id__nodename__index
        Example: doc_A__sentences__0 (first chunk from 'sentences' Split node)
        """
        text_chunks = self.split_document(doc.content)

        tracked_chunks = []
        for idx, chunk in enumerate(text_chunks):
            # Include node name in ID for tracking which split created this
            new_id = f"{doc.id}__{self.name}__{idx}"

            # Calculate overlap region if overlap > 0
            content_excluding_overlap = None
            if self.overlap > 0 and len(text_chunks) > 1:
                content_excluding_overlap = self._calculate_overlap_region(
                    chunk, idx, len(text_chunks)
                )

            tracked_chunks.append(
                TrackedItem(
                    content=chunk,
                    id=new_id,
                    sources=[doc.id],  # This chunk comes from single parent
                    metadata={
                        **doc.metadata,
                        "split_index": idx,
                        "split_node": self.name,
                        "parent_id": doc.id,
                    },
                    content_excluding_overlap=content_excluding_overlap,
                )
            )

        return tracked_chunks

    def _calculate_overlap_region(
        self, chunk: str, chunk_idx: int, total_chunks: int
    ) -> Tuple[int, int]:
        """Calculate the (start, end) indices for content excluding overlap.

        Args:
            chunk: The chunk text (including overlap)
            chunk_idx: Index of this chunk (0-based)
            total_chunks: Total number of chunks

        Returns:
            (start, end) tuple indicating core content region
        """
        chunk_len = len(chunk)

        if chunk_idx == 0:
            # First chunk: no pre-overlap, may have post-overlap
            # For simplicity, we only track pre-overlap, so first chunk is fully core
            return (0, chunk_len)

        elif chunk_idx == total_chunks - 1:
            # Last chunk: has pre-overlap, no post-overlap
            # The pre-overlap is approximately self.overlap chars/tokens
            # For char-level split, this is exact
            # For token/word/sentence split, this is approximate
            if self.split_unit == "chars":
                overlap_chars = self.overlap
            else:
                # Approximate: assume average token length for estimation
                # This is not perfect but works reasonably well
                overlap_chars = self._estimate_overlap_chars(chunk, self.overlap)

            return (min(overlap_chars, chunk_len), chunk_len)

        else:
            # Middle chunk: has both pre and post overlap
            # We only exclude the pre-overlap region
            if self.split_unit == "chars":
                overlap_chars = self.overlap
            else:
                overlap_chars = self._estimate_overlap_chars(chunk, self.overlap)

            return (min(overlap_chars, chunk_len), chunk_len)

    def _estimate_overlap_chars(self, chunk: str, overlap_units: int) -> int:
        """Estimate character length of overlap region for token/word/sentence splits.

        Args:
            chunk: The chunk text
            overlap_units: Number of overlap units (tokens/words/sentences)

        Returns:
            Estimated character count for overlap region
        """
        if self.split_unit == "tokens":
            # Tokenize the chunk and find where overlap_units tokens end
            tokens = self.token_encoder.encode(chunk)
            if len(tokens) <= overlap_units:
                return len(chunk)
            # Decode first overlap_units tokens to get character length
            overlap_tokens = tokens[:overlap_units]
            overlap_text = self.token_encoder.decode(overlap_tokens)
            return len(overlap_text)

        elif self.split_unit == "words":
            # Split by words and find where overlap_units words end
            import nltk

            words = nltk.word_tokenize(chunk)
            if len(words) <= overlap_units:
                return len(chunk)
            # Find character position after overlap_units words
            # Reconstruct text from first overlap_units words
            overlap_words = words[:overlap_units]
            # Simple estimate: find position in original text
            rejoined = " ".join(overlap_words)
            return len(rejoined)

        elif self.split_unit == "sentences":
            # Split by sentences and find where overlap_units sentences end
            from pysbd import Segmenter

            seg = Segmenter(language="en", clean=True)
            sentences = seg.segment(chunk)
            if len(sentences) <= overlap_units:
                return len(chunk)
            # Sum length of first overlap_units sentences
            overlap_text = " ".join(sentences[:overlap_units])
            return len(overlap_text)

        # Fallback: return 0 if can't estimate
        return 0

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
            from pysbd import Segmenter

            seg = Segmenter(language="en", clean=True)
            return seg.segment(doc)
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
        from .batch import BatchList

        # Get base metadata from parent
        result = super().result()

        if not self.output:
            logger.warning(
                f"Split node '{self.name}' has no output to create statistics from"
            )
            result["chunks_df"] = pd.DataFrame()
            result["summary_stats"] = {}
            result["metadata"]["split_unit"] = self.split_unit
            result["metadata"]["chunk_size"] = self.chunk_size
            result["metadata"]["num_chunks"] = 0
            return result

        # Flatten BatchList if needed
        if isinstance(self.output, BatchList):
            all_items = self.output.flatten_all()
        else:
            all_items = self.output

        # Debug: log what types we're seeing
        type_counts = {}
        for chunk in all_items:
            type_name = type(chunk).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        logger.debug(f"Split node '{self.name}' output types: {type_counts}")

        # Build DataFrame of chunks with comprehensive statistics
        rows = []
        for idx, chunk in enumerate(all_items):
            # Handle both TrackedItem objects and deserialized dicts
            if isinstance(chunk, dict):
                # Deserialized from JSON
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {}) or {}
                item_id = chunk.get("id", "")
            elif isinstance(chunk, TrackedItem):
                # Live object
                content = chunk.content
                metadata = chunk.metadata or {}
                item_id = chunk.id
            else:
                # Try extract_content as fallback, but this likely won't work for wrong types
                content = extract_content(chunk)
                if not content:
                    logger.debug(
                        f"Skipping chunk {idx} - wrong type: {type(chunk).__name__}"
                    )
                    continue
                metadata = getattr(chunk, "metadata", {}) or {}
                item_id = getattr(chunk, "id", "")

            if not content:
                logger.debug(f"Skipping chunk {idx} - no content")
                continue

            # Extract filename from metadata or item_id
            filename = metadata.get("filename", "")
            if not filename:
                # Try to extract filename from item_id (format: filename__node__index)
                if item_id and "__" in item_id:
                    filename = item_id.split("__")[0]
                else:
                    filename = item_id or "unknown"

            # Get chunk index from metadata
            chunk_index = metadata.get("split_index", idx)

            # Compute various length statistics
            length_chars = len(content)
            length_words = len(content.split())

            # Tokenize to get token count using tiktoken (OpenAI tokenizer)
            try:
                length_tokens = len(self.token_encoder.encode(content))
            except Exception as e:
                logger.debug(f"Failed to tokenize chunk {idx}: {e}")
                length_tokens = 0

            # Count sentences using nltk?
            try:
                from pysbd import Segmenter

                seg = Segmenter(language="en", clean=True)
                length_sentences = len(seg.segment(content))
            except Exception as e:
                logger.debug(f"Failed to count sentences in chunk {idx}: {e}")
                length_sentences = 0

            # Count paragraphs (split on blank lines)
            length_paragraphs = len([p for p in content.split("\n\n") if p.strip()])

            rows.append(
                {
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "length_chars": length_chars,
                    "length_words": length_words,
                    "length_sentences": length_sentences,
                    "length_paragraphs": length_paragraphs,
                    "length_tokens": length_tokens,
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            logger.warning(
                f"Split node '{self.name}' created empty DataFrame from {len(all_items)} chunks - check if output is the correct type"
            )
        else:
            logger.debug(
                f"Split node '{self.name}' created DataFrame with {len(df)} rows"
            )

        # Compute summary statistics
        summary_stats = {}
        if not df.empty:
            for col in [
                "length_chars",
                "length_words",
                "length_sentences",
                "length_paragraphs",
                "length_tokens",
            ]:
                if col in df.columns:
                    summary_stats[col] = {
                        "min": int(df[col].min()),
                        "max": int(df[col].max()),
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                    }

        # Add Split-specific data
        result["chunks_df"] = df
        result["summary_stats"] = summary_stats
        result["metadata"]["split_unit"] = self.split_unit
        result["metadata"]["chunk_size"] = self.chunk_size
        result["metadata"]["num_chunks"] = len(all_items)

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Split node details."""
        super().export(folder, unique_id=unique_id)

        if self.output:
            import numpy as np

            from .batch import BatchList

            # Flatten BatchList if needed
            if isinstance(self.output, BatchList):
                all_items = self.output.flatten_all()
            else:
                all_items = self.output

            # Extract content from TrackedItems for statistics
            lens = []
            for doc in all_items:
                # Handle TrackedItem, string, or dict (from JSON deserialization)
                if isinstance(doc, TrackedItem):
                    content = doc.content
                elif isinstance(doc, str):
                    content = doc
                elif isinstance(doc, dict) and "content" in doc:
                    content = doc["content"]
                else:
                    # Skip items that can't be processed (e.g., StruckdownResult from wrong node type)
                    logger.debug(
                        f"Skipping non-text output in Split export: {type(doc)}"
                    )
                    continue

                if content and isinstance(content, str):
                    lens.append(len(self.tokenize(content, method=self.split_unit)))

            if lens:
                summary = f"""Split Summary
==============
Chunks created: {len(all_items)}
Split unit: {self.split_unit}
Chunk size: {self.chunk_size}
Average length: {np.mean(lens).round(1)}
Max length: {max(lens)}
Min length: {min(lens)}
"""
            else:
                summary = f"""Split Summary
==============
Chunks created: {len(all_items)}
Split unit: {self.split_unit}
Chunk size: {self.chunk_size}
(Statistics unavailable - output not in expected format)
"""
            (folder / "split_summary.txt").write_text(summary)

            # Export output chunks with source_id naming
            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            for idx, chunk in enumerate(all_items):
                if isinstance(chunk, TrackedItem):
                    # Use source_id in filename
                    (outputs_folder / f"{idx:04d}_{chunk.safe_id}.txt").write_text(
                        chunk.content
                    )

                    # Export metadata if present
                    if chunk.metadata:
                        (
                            outputs_folder / f"{idx:04d}_{chunk.safe_id}_metadata.json"
                        ).write_text(chunk.to_json())
                else:
                    # Backward compatibility: plain strings
                    (outputs_folder / f"{idx:04d}_chunk.txt").write_text(str(chunk))
