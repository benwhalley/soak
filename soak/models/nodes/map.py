"""Map node for applying LLM transformations to multiple items."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

import anyio
import pandas as pd
from box import Box
from pydantic import Field
from struckdown import StruckdownLLMError
from struckdown.parsing import parse_syntax

from soak.error_handlers import managed_llm_call
from soak.models.base import TrackedItem, extract_prompt, safe_json_dump, semaphore

from .base import CompletionDAGNode, ItemsNode, default_map_task, template_map_task

logger = logging.getLogger(__name__)


class Map(ItemsNode, CompletionDAGNode):
    model_config = {
        "discriminator": "type",
    }

    type: Literal["Map"] = "Map"

    function: Literal["llm", "template"] = (
        "llm"  # llm = LLM call, template = Jinja2 only
    )
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
        # Import here to avoid circular import
        from .batch import BatchList

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

        # Use progress bar context manager
        with self.progress_bar(items) as pbar:
            async with anyio.create_task_group() as tg:
                for idx, item in enumerate(items):

                    async def run_and_store(index=idx, item=item, progress_bar=pbar):
                        async with semaphore:
                            try:
                                if self.function == "template":
                                    # Template-only mode: just render Jinja2, no LLM call
                                    results[index] = await template_map_task(
                                        template=self.template,
                                        context={**filtered_context, **item},
                                    )
                                else:
                                    # Default LLM mode
                                    extra_kwargs = self.get_llm_kwargs()
                                    results[index] = await managed_llm_call(
                                        node_name=self.name,
                                        config=self.dag.config,
                                        llm_func=self.task,
                                        item_index=index,
                                        template=self.template,
                                        context={**filtered_context, **item},
                                        model=self.get_model(),
                                        credentials=self.dag.config.llm_credentials,
                                        **extra_kwargs,
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
                                    progress_bar.update(1)

                    tg.start_soon(run_and_store)

        # accumulate costs from all results (only for LLM mode)
        if self.function == "llm":
            for result in results:
                if result is not None:
                    self._accumulate_costs(result)

        if is_batch:
            # Reconstruct BatchList structure
            from .batch import BatchList

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

        for idx, output_item in enumerate(output_list):
            item = input_items[idx] if input_items and idx < len(input_items) else None

            row = TrackedItem.extract_export_metadata(item, idx)

            if self.function == "template":
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
        result["metadata"]["function"] = self.function

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Map node details with numbered prompts and responses."""
        from ..utils import export_chatter_result

        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template_text:
            template_filename = (
                "template.md"
                if self.function == "template"
                else "prompt_template.sd.md"
            )
            (folder / template_filename).write_text(self.template_text)

        # Get input items for source tracking
        input_items = self.get_input_items()

        # Write each output with source tracking
        if self.output and isinstance(self.output, list):
            for idx, result in enumerate(self.output):
                # Get source_id if available
                item = (
                    input_items[idx] if input_items and idx < len(input_items) else None
                )
                safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))
                file_prefix = f"{idx:04d}_{safe_id}"

                if self.function == "template":
                    # Template mode: export rendered text
                    (folder / f"{file_prefix}_rendered.txt").write_text(str(result))
                else:
                    # LLM mode: export ChatterResult
                    export_chatter_result(result, folder, file_prefix)
