"""Transform node for single-item LLM transformations."""

import logging
from pathlib import Path
from typing import Any, Dict, Literal

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
    template_text: str = Field(default="{{input}} <prompt>: [[output]]")

    @property
    def template(self) -> str:
        return self.template_text

    async def run(self):
        items = await self.get_items()

        if not isinstance(items, str):
            assert len(items) == 1, "Transform nodes must have exactly one input item"

        rt = render_strict_template(self.template, {**self.context, **items[0]})

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
                merged_context = {**self.context, **full_dag_context, **items[0]}

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
                self.output = result
            except Exception as e:
                # catch-all for any non-struckdown errors
                logger.error(f"Unexpected error in node '{self.name}': {e}")
                # default to skip + continue for unknown errors
                self.output = None

        # accumulate costs if we got a result
        if self.output is not None:
            self._accumulate_costs(self.output)

        return self.output

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, prompt, response object, and raw ChatterResult."""
        # Get base metadata from parent
        result = super().result()

        # Add Transform-specific data
        result["prompt"] = extract_prompt(self.output)
        result["response_obj"] = (
            self.output.response if hasattr(self.output, "response") else None
        )
        result["response_text"] = (
            str(self.output.response) if hasattr(self.output, "response") else None
        )
        result["chatter_result"] = self.output

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Transform node details with single prompt/response."""
        from ..utils import export_chatter_result

        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

        # Write prompt and response using utility function
        # Note: for Transform, we use simple filenames without index prefix
        if self.output:
            export_chatter_result(self.output, folder, "")
            # Rename files to Transform's preferred names (without prefix)
            if (folder / "_prompt.md").exists():
                (folder / "_prompt.md").rename(folder / "prompt.md")
            if (folder / "_response.txt").exists():
                (folder / "_response.txt").rename(folder / "response.txt")
            if (folder / "_response.json").exists():
                (folder / "_response.json").rename(folder / "response.json")
