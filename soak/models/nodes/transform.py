"""Transform node for single-item LLM transformations."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import Field
from struckdown import StruckdownLLMError, chatter_async

from soak.error_handlers import managed_llm_call
from soak.models.base import (
    extract_prompt,
    get_action_lookup,
    safe_json_dump,
    semaphore,
)
from soak.models.dag import render_strict_template

from .base import CompletionDAGNode, ItemsNode

logger = logging.getLogger(__name__)


class Transform(ItemsNode, CompletionDAGNode):
    """Single-item transformation node using LLM."""

    type: Literal["Transform"] = "Transform"
    template: str = Field(default="{{input}} <prompt>: [[output]]")

    async def process_items(self, items: List[Any], progress_bar: Any = None) -> List[Any]:
        """Process exactly one item (Transform requires single-item batches).

        Args:
            items: Must contain exactly 1 item
            progress_bar: Optional progress bar to update after processing

        Returns:
            List with single ChatterResult
        """
        assert len(items) == 1, (
            f"Transform node '{self.name}' requires exactly one input item per batch, "
            f"got {len(items)}. Use Batch with batch_size=1 or GroupBy before Transform."
        )

        # Get items with proper context
        items_with_context = await self.get_items()
        assert len(items_with_context) == 1, "Context mismatch in Transform"

        rt = render_strict_template(self.template, {**self.context, **items_with_context[0]})

        # Get LLM kwargs using helper method
        extra_kwargs = self.get_llm_kwargs()

        # Call chatter with semaphore to limit concurrency
        async with semaphore:
            try:
                # Include ALL node outputs in context for post_process to find previous codes
                full_dag_context = {
                    node.name: node.output
                    for node in self.dag.nodes
                    if node.output is not None
                }
                merged_context = {**self.context, **full_dag_context, **items_with_context[0]}

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

        # update progress bar after processing the item
        if progress_bar is not None:
            progress_bar.update(1)

        return [result] if result else []

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, prompt, response object, and raw ChatterResult."""
        # Get base metadata from parent
        result = super().result()

        # Handle list output (Transform wraps ChatterResult in a list)
        chatter = self.output[0] if isinstance(self.output, list) and len(self.output) > 0 else self.output

        # Add Transform-specific data
        result["prompt"] = extract_prompt(chatter)
        result["response_obj"] = (
            chatter.response if hasattr(chatter, "response") else None
        )
        result["response_text"] = (
            str(chatter.response) if hasattr(chatter, "response") else None
        )
        result["chatter_result"] = chatter

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Transform node details.

        For unbatched: exports slots as individual text files in the main folder
        For batched: creates batch_N subfolders with slots as text files
        """
        from .batch import BatchList
        from ..utils import export_slots_as_text_files

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
                result = batch[0] if isinstance(batch, list) and len(batch) > 0 else batch
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
