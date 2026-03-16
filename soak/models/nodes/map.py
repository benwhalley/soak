"""Map node for applying LLM transformations to multiple items."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import anyio
import pandas as pd
from box import Box
from pydantic import Field
from struckdown import LLMError
from struckdown.parsing import parse_syntax

from soak.error_handlers import managed_llm_call
from soak.models.base import (TrackedItem, extract_prompt, safe_json_dump,
                              semaphore)
from soak.models.utils import post_process_chatter_result

from .base import (CompletionDAGNode, ItemsNode, default_map_task,
                   template_map_task)

logger = logging.getLogger(__name__)


class Map(ItemsNode, CompletionDAGNode):
    model_config = {
        "discriminator": "type",
    }

    type: Literal["Map"] = "Map"

    mode: Literal["llm", "template"] = "llm"  # llm = LLM call, template = Jinja2 only
    task: Callable = Field(default=default_map_task, exclude=True)
    template: str = None

    def validate_template(self):
        try:
            parse_syntax(self.template)
            return True
        except Exception as e:
            logger.error(f"Template syntax error: {e}")
            return False

    async def process_items(
        self, items: List[Any], progress_bar: Optional[Any] = None
    ) -> List[Any]:
        """Process items in parallel at this batch level.

        Args:
            items: Flat list of items to process in this batch
            progress_bar: Optional tqdm progress bar to update

        Returns:
            Flat list of results (ChatterResult or rendered strings)
        """
        # Track input count for progress reporting
        self._input_count = len(items)

        # Convert items to Box items for templates
        boxed_items = []
        for item in items:
            if isinstance(item, TrackedItem):
                # Unpack metadata (contains Excel/CSV column data) into template context
                boxed_items.append(
                    Box({"input": item.content, "tracked_item": item, **item.metadata})
                )
            elif isinstance(item, dict):
                boxed_items.append(Box(item))
            else:
                boxed_items.append(Box({"input": item}))

        # Filter context to remove BatchList objects
        from .batch import BatchList

        filtered_context = {
            k: v for k, v in self.context.items() if not isinstance(v, BatchList)
        }

        # Add node/DAG reference for quote resolution via DAG traversal
        # This allows collect_input_codes to find codes in ancestor nodes
        filtered_context["_node"] = self
        filtered_context["_dag"] = self.dag

        results = [None] * len(boxed_items)

        # Use passed-in progress bar if available, otherwise create local one
        pbar = progress_bar
        if pbar is None:
            # Create local progress bar for backward compatibility (non-batched case)
            if self.dag.config.show_progress:
                import sys

                # Use progress manager if available for coordinated display
                if self.dag.progress_manager:
                    pbar = self.dag.progress_manager.create_progress_bar(
                        total=len(boxed_items),
                        desc=f"{self.type}: {self.name}",
                        unit="item",
                    )
                else:
                    from tqdm import tqdm

                    desc = f"{self.type}: {self.name}".ljust(35)
                    pbar = tqdm(
                        total=len(boxed_items),
                        desc=desc,
                        unit="item",
                        file=sys.stderr,
                        ncols=120,
                        leave=True,
                        mininterval=0.1,
                    )

        try:
            async with anyio.create_task_group() as tg:
                for idx, item in enumerate(boxed_items):

                    async def run_and_store(index=idx, item=item, progress_bar=pbar):
                        async with semaphore:
                            try:
                                if self.mode == "template":
                                    # Template-only mode: just render Jinja2, no LLM call
                                    results[index] = await template_map_task(
                                        template=self.template,
                                        context={**filtered_context, **item},
                                    )
                                else:
                                    # Default LLM mode
                                    extra_kwargs = self.get_llm_kwargs()
                                    llm_context = {**filtered_context, **item}
                                    results[index] = await managed_llm_call(
                                        node_name=self.name,
                                        config=self.dag.config,
                                        llm_func=self.task,
                                        item_index=index,
                                        template=self.template,
                                        context=llm_context,
                                        model=self.get_model(),
                                        credentials=self.dag.config.llm_credentials,
                                        **extra_kwargs,
                                    )
                                    # Post-process outputs to populate resolved_quotes/resolved_code_refs
                                    post_process_chatter_result(
                                        results[index], llm_context
                                    )
                            except Exception as e:
                                # re-raise all other errors to fail the pipeline
                                logger.error(
                                    f"Error in node '{self.name}' for item {index}: {e}"
                                )
                                raise
                            finally:
                                # Update progress bar on completion
                                if progress_bar is not None:
                                    progress_bar.update(
                                        getattr(progress_bar, "slots_per_item", 1)
                                    )

                                # Call progress callback for real-time web UI updates
                                if self.dag.config.progress_callback:
                                    try:
                                        # Count completed items (non-None results)
                                        done = sum(1 for r in results if r is not None)
                                        logger.debug(f"Map {self.name}: progress {done}/{len(results)}")
                                        self.dag.config.progress_callback(
                                            self.name,
                                            done,
                                            len(results),
                                            self._total_cost,
                                            self._fresh_cost,  # for cache savings display
                                        )
                                    except Exception as e:
                                        logger.warning(f"Progress callback error: {e}")

                    tg.start_soon(run_and_store)
        finally:
            # Only close progress bar if we created it locally
            if progress_bar is None and pbar is not None:
                pbar.close()

        # accumulate costs and track for cache statistics (only for LLM mode)
        if self.mode == "llm":
            for result in results:
                if result is not None:
                    self._accumulate_costs(result)
                    self._llm_results.append(result)

                    # update progress bar with per-node cost if using CostProgressBar
                    from soak.models.progress import CostProgressBar

                    if isinstance(pbar, CostProgressBar):
                        pbar.update_cost(
                            result.fresh_cost,
                            result.prompt_tokens + result.completion_tokens,
                        )


        return results

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and DataFrame of mapped items."""
        from .batch import BatchList

        # Get base metadata from parent
        result = super().result()

        input_items = self.get_input_items()
        rows = []

        # Flatten BatchList if needed
        if isinstance(self.output, BatchList):
            output_list = self.output.flatten_all()
        elif isinstance(self.output, list):
            output_list = self.output
        else:
            output_list = []

        for idx, output_item in enumerate(output_list):
            item = input_items[idx] if input_items and idx < len(input_items) else None

            row = TrackedItem.extract_export_metadata(item, idx)

            if self.mode == "template":
                # Template mode: output is plain string
                row.update(
                    {
                        "rendered_text": str(output_item) if output_item else None,
                    }
                )
            else:
                # LLM mode: output is ChatterResult
                row.update(
                    {
                        "prompt": extract_prompt(output_item),
                        "response_text": (
                            str(output_item.response)
                            if hasattr(output_item, "response")
                            else None
                        ),
                        "response_obj": (
                            output_item.response
                            if hasattr(output_item, "response")
                            else None
                        ),
                        "chatter_result": output_item,
                    }
                )

            rows.append(row)

        # Add Map-specific data
        result["data"] = pd.DataFrame(rows)
        result["metadata"]["num_items"] = len(output_list)
        result["metadata"]["mode"] = self.mode

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Map node details with numbered prompts and responses."""
        from ..utils import export_chatter_result
        from .batch import BatchList

        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template:
            template_filename = (
                "template.md" if self.mode == "template" else "prompt_template.sd"
            )
            (folder / template_filename).write_text(self.template)

        # Get input items for source tracking
        input_items = self.get_input_items()

        # Flatten BatchList if needed
        if isinstance(self.output, BatchList):
            output_list = self.output.flatten_all()
        elif isinstance(self.output, list):
            output_list = self.output
        else:
            output_list = None

        # Write each output with source tracking
        if output_list:
            for idx, result in enumerate(output_list):
                # Get source_id if available
                item = (
                    input_items[idx] if input_items and idx < len(input_items) else None
                )
                safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))
                file_prefix = f"{idx:04d}_{safe_id}"

                if self.mode == "template":
                    # Template mode: export rendered text
                    (folder / f"{file_prefix}_rendered.txt").write_text(str(result))
                else:
                    # LLM mode: export ChatterResult
                    export_chatter_result(result, folder, file_prefix)

    async def skip(self) -> Optional[List[Any]]:
        """When skipped (e.g., due to bypass), pass input through unchanged.

        Returns:
            The input items unchanged, flattening any BatchList structures.
        """
        from .batch import BatchList

        if self.inputs:
            input_name = self.inputs[0]
            if input_name == "documents":
                input_data = self.dag.config.load_documents()
            else:
                input_data = self.context[input_name]
        else:
            input_data = self.dag.config.load_documents()

        if isinstance(input_data, BatchList):
            self.output = input_data.flatten_all()
        elif isinstance(input_data, list):
            self.output = input_data
        else:
            self.output = [input_data] if input_data else []

        logger.info(f"Map '{self.name}': skipped, passing through {len(self.output)} items")
        return self.output
