"""Transform node for single-item LLM transformations."""

import logging
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import Field
from struckdown import chatter_async

from ..base import extract_prompt, get_action_lookup, safe_json_dump, semaphore
from ..dag import render_strict_template
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

        # Collect extra kwargs for LLM
        extra_kwargs = {}
        if self.max_tokens is not None:
            extra_kwargs["max_tokens"] = self.max_tokens
        extra_kwargs["temperature"] = self.temperature
        extra_kwargs["seed"] = self.dag.config.seed

        # Call chatter with semaphore to limit concurrency
        async with semaphore:
            self.output = await chatter_async(
                multipart_prompt=rt,
                model=self.get_model(),
                credentials=self.dag.config.llm_credentials,
                action_lookup=get_action_lookup(),
                extra_kwargs=extra_kwargs,
            )
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
        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

        # Write prompt and response
        if self.output:
            try:
                if hasattr(self.output, "results") and self.output.results:
                    # Get first segment's prompt
                    first_seg = next(iter(self.output.results.values()))
                    if hasattr(first_seg, "prompt"):
                        (folder / "prompt.md").write_text(first_seg.prompt)

                # Write response text
                if hasattr(self.output, "response"):
                    response_text = str(self.output.response)
                    (folder / "response.txt").write_text(response_text)

                # Write full ChatterResult JSON
                (folder / "response.json").write_text(safe_json_dump(self.output))
            except Exception as e:
                logger.warning(f"Failed to export Transform node: {e}")
