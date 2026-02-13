"""Transform node for single-item LLM transformations."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

from pydantic import Field
from struckdown import LLMError, chatter_async
from tqdm import tqdm

from soak.error_handlers import managed_llm_call
from soak.models.base import (extract_prompt, get_action_lookup,
                              safe_json_dump, semaphore)
from soak.models.dag import render_strict_template
from soak.models.utils import post_process_chatter_result

from .base import CompletionDAGNode, ItemsNode

logger = logging.getLogger(__name__)


class Transform(ItemsNode, CompletionDAGNode):
    """Single-item transformation node using LLM."""

    type: Literal["Transform"] = "Transform"
    template: str = Field(default="{{input}} <prompt>: [[output]]")

    async def run(self) -> Union[List[Any], Any]:
        """Override run() to set progress bar total to 1 (Transform makes 1 LLM call).

        Transform processes all input items as a single batch with one LLM call,
        so the progress bar should show 1/1 regardless of input count.
        """
        await super(ItemsNode, self).run()  # Call DAGNode.run(), skip ItemsNode.run()
        input_data = self.get_input_data()

        # Create progress bar with total=1 (Transform always makes exactly 1 LLM call)
        progress_bar = None
        if self.dag.config.show_progress:
            # Use progress manager if available for coordinated display
            if self.dag.progress_manager:
                if isinstance(self, CompletionDAGNode) and self.dag.cost_tracker:
                    progress_bar = self.dag.progress_manager.create_cost_progress_bar(
                        total=1,
                        node_name=self.name,
                        unit="item",
                    )
                else:
                    progress_bar = self.dag.progress_manager.create_progress_bar(
                        total=1,
                        desc=f"{self.type}: {self.name}",
                        unit="item",
                    )
            elif isinstance(self, CompletionDAGNode) and self.dag.cost_tracker:
                from soak.models.progress import CostProgressBar

                progress_bar = CostProgressBar(
                    tracker=self.dag.cost_tracker,
                    node_name=self.name,
                    total=1,  # Always 1 for Transform
                    unit="item",
                )
            else:
                # pad description to match CostProgressBar alignment
                desc = f"{self.type}: {self.name}".ljust(35)
                progress_bar = tqdm(
                    total=1,  # Always 1 for Transform
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

    async def process_items(
        self, items: List[Any], progress_bar: Any = None
    ) -> List[Any]:
        """Process exactly one item (Transform requires single-item batches).

        Args:
            items: Must contain exactly 1 item
            progress_bar: Optional progress bar to update after processing

        Returns:
            List with single ChatterResult
        """

        # assert len(items) == 1, (
        #     f"Transform node '{self.name}' requires exactly one input item per batch, "
        #     f"got {len(items)}. Use Batch with batch_size=1 or GroupBy before Transform."
        # )

        # Get items with proper context

        input_context = {node: self.dag.nodes_dict[node].output for node in self.inputs}
        merged_context = {**self.context, **input_context}

        # Add node/DAG reference for quote resolution via DAG traversal
        # This allows collect_input_codes to find codes in ancestor nodes
        merged_context["_node"] = self
        merged_context["_dag"] = self.dag

        rt = render_strict_template(self.template, {**self.context, **merged_context})

        # Get LLM kwargs using helper method
        extra_kwargs = self.get_llm_kwargs()

        # Call chatter with semaphore to limit concurrency
        async with semaphore:
            try:
                result = await managed_llm_call(
                    node_name=self.name,
                    config=self.dag.config,
                    llm_func=chatter_async,
                    item_index=None,
                    multipart_prompt=rt,
                    context=merged_context,
                    model=self.get_model(),
                    credentials=self.dag.config.llm_credentials,
                    extra_kwargs=extra_kwargs,
                )
            except Exception as e:
                # catch-all for any non-struckdown errors
                logger.error(f"Unexpected error in node '{self.name}': {e}")
                # default to skip + continue for unknown errors
                result = None

        # accumulate costs and track for cache statistics
        if result is not None:
            self._accumulate_costs(result)
            self._llm_results.append(result)

            # Post-process outputs to populate resolved_quotes/resolved_code_refs
            post_process_chatter_result(result, merged_context)

            # update progress bar with per-node cost if using CostProgressBar
            from soak.models.progress import CostProgressBar

            if isinstance(progress_bar, CostProgressBar):
                progress_bar.update_cost(
                    result.fresh_cost, result.prompt_tokens + result.completion_tokens
                )

        # update progress bar after processing the item
        if progress_bar is not None:
            progress_bar.update(getattr(progress_bar, "slots_per_item", 1))

        return [result] if result else []

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, prompt, response object, raw ChatterResult, and slot dataframes."""
        import pandas as pd

        from soak.models.base import Code, CodeList, Theme, Themes

        # Get base metadata from parent
        result = super().result()

        # Handle list output (Transform wraps ChatterResult in a list)
        chatter = (
            self.output[0]
            if isinstance(self.output, list) and len(self.output) > 0
            else self.output
        )

        # Add Transform-specific data
        result["prompt"] = extract_prompt(chatter)
        result["response_obj"] = (
            chatter.response if hasattr(chatter, "response") else None
        )
        result["response_text"] = (
            str(chatter.response) if hasattr(chatter, "response") else None
        )
        result["chatter_result"] = chatter

        # Extract slot dataframes for HTML display
        result["slot_dataframes"] = {}
        if hasattr(chatter, "results") and chatter.results:
            for slot_name, segment_result in chatter.results.items():
                if hasattr(segment_result, "output"):
                    output = segment_result.output

                    # Convert Themes to DataFrame
                    if isinstance(output, Themes):
                        rows = []
                        for theme in output.themes:
                            rows.append(
                                {
                                    "name": (
                                        theme.name if hasattr(theme, "name") else ""
                                    ),
                                    "description": (
                                        theme.description
                                        if hasattr(theme, "description")
                                        else ""
                                    ),
                                    "code_slugs": (
                                        ", ".join(theme.code_slugs)
                                        if hasattr(theme, "code_slugs")
                                        else ""
                                    ),
                                    "num_codes": (
                                        len(theme.code_slugs)
                                        if hasattr(theme, "code_slugs")
                                        else 0
                                    ),
                                }
                            )
                        if rows:
                            result["slot_dataframes"][slot_name] = pd.DataFrame(rows)

                    elif isinstance(output, Theme):
                        rows = [
                            {
                                "name": output.name if hasattr(output, "name") else "",
                                "description": (
                                    output.description
                                    if hasattr(output, "description")
                                    else ""
                                ),
                                "code_slugs": (
                                    ", ".join(output.code_slugs)
                                    if hasattr(output, "code_slugs")
                                    else ""
                                ),
                                "num_codes": (
                                    len(output.code_slugs)
                                    if hasattr(output, "code_slugs")
                                    else 0
                                ),
                            }
                        ]
                        result["slot_dataframes"][slot_name] = pd.DataFrame(rows)

                    # Convert Codes to DataFrame
                    elif isinstance(output, CodeList):
                        rows = []
                        for code in output.codes:
                            quotes_text = []
                            if hasattr(code, "all_quotes"):
                                quotes_text = [
                                    q.text if hasattr(q, "text") else str(q)
                                    for q in code.all_quotes
                                ]
                            elif hasattr(code, "quotes"):
                                quotes_text = [
                                    q.text if hasattr(q, "text") else str(q)
                                    for q in code.quotes
                                ]

                            rows.append(
                                {
                                    "slug": code.slug if hasattr(code, "slug") else "",
                                    "name": code.name if hasattr(code, "name") else "",
                                    "description": (
                                        code.description
                                        if hasattr(code, "description")
                                        else ""
                                    ),
                                    "quotes": " | ".join(quotes_text),
                                    "num_quotes": len(quotes_text),
                                }
                            )
                        if rows:
                            result["slot_dataframes"][slot_name] = pd.DataFrame(rows)

                    elif isinstance(output, Code):
                        quotes_text = []
                        if hasattr(output, "all_quotes"):
                            quotes_text = [
                                q.text if hasattr(q, "text") else str(q)
                                for q in output.all_quotes
                            ]
                        elif hasattr(output, "quotes"):
                            quotes_text = [
                                q.text if hasattr(q, "text") else str(q)
                                for q in output.quotes
                            ]

                        rows = [
                            {
                                "slug": output.slug if hasattr(output, "slug") else "",
                                "name": output.name if hasattr(output, "name") else "",
                                "description": (
                                    output.description
                                    if hasattr(output, "description")
                                    else ""
                                ),
                                "quotes": " | ".join(quotes_text),
                                "num_quotes": len(quotes_text),
                            }
                        ]
                        result["slot_dataframes"][slot_name] = pd.DataFrame(rows)

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Transform node details.

        For unbatched: exports slots as individual text files in the main folder
        For batched: creates batch_N subfolders with slots as text files
        """
        from ..utils import export_slots_as_text_files
        from .batch import BatchList

        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template:
            (folder / "prompt_template.sd").write_text(self.template)

        if not self.output:
            return

        # Handle different output types
        if isinstance(self.output, BatchList):
            # Batched output: create batch_N subfolders
            batches = self.output.flatten_one_level()
            for batch_idx, batch in enumerate(batches):
                batch_folder = folder / f"batch_{batch_idx}"
                batch_folder.mkdir(parents=True, exist_ok=True)

                # Each batch contains a single ChatterResult (may be wrapped in list)
                result = (
                    batch[0] if isinstance(batch, list) and len(batch) > 0 else batch
                )
                export_slots_as_text_files(result, batch_folder)

                # Also export full JSON for reference
                (batch_folder / "response.json").write_text(safe_json_dump(result))

        elif isinstance(self.output, list):
            # Unbatched output: plain list with single ChatterResult
            if len(self.output) > 0:
                result = self.output[0]
                export_slots_as_text_files(result, folder)
                (folder / "response.json").write_text(safe_json_dump(result))
            else:
                logger.warning(f"Transform node '{self.name}' has empty output list")

        else:
            # Single ChatterResult (edge case)
            export_slots_as_text_files(self.output, folder)
            (folder / "response.json").write_text(safe_json_dump(self.output))
