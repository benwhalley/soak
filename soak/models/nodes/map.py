"""Map node for applying LLM transformations to multiple items."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

import anyio
import pandas as pd
from box import Box
from pydantic import Field
from struckdown.parsing import parse_syntax

from ..base import TrackedItem, extract_prompt, safe_json_dump, semaphore
from .base import CompletionDAGNode, ItemsNode, default_map_task

logger = logging.getLogger(__name__)


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

        async with anyio.create_task_group() as tg:
            for idx, item in enumerate(items):

                async def run_and_store(index=idx, item=item):
                    async with semaphore:
                        # Collect extra kwargs for LLM
                        extra_kwargs = {}
                        if self.max_tokens is not None:
                            extra_kwargs["max_tokens"] = self.max_tokens
                        extra_kwargs["temperature"] = self.temperature
                        extra_kwargs["seed"] = self.dag.config.seed

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
