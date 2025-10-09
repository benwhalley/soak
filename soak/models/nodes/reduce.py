"""Reduce node for aggregating inputs."""

import logging
from pathlib import Path
from typing import Any, Dict, Literal

from ..dag import render_strict_template
from .base import ItemsNode

logger = logging.getLogger(__name__)


class Reduce(ItemsNode):
    """Node that reduces multiple items into a single output."""
    type: Literal["Reduce"] = "Reduce"
    template_text: str = "{{input}}\n"

    @property
    def template(self) -> str:
        return self.template_text

    def get_items(self):
        # Import here to avoid circular import
        from .batch import BatchList

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

        # Import here to avoid circular import
        from .batch import BatchList

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

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and reduced output."""
        # Get base metadata from parent
        result = super().result()

        # Add Reduce-specific data
        result["output"] = self.output
        result["output_type"] = type(self.output).__name__ if self.output else None

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Reduce node details."""
        super().export(folder, unique_id=unique_id)

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
