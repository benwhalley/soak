"""Filter node for boolean filtering with LLM."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import anyio
import pandas as pd
from struckdown import ChatterResult, chatter_async

from ..base import TrackedItem, get_action_lookup, safe_json_dump, semaphore
from .base import ItemsNode, CompletionDAGNode

logger = logging.getLogger(__name__)


class Filter(ItemsNode, CompletionDAGNode):
    """
    Filter items based on a boolean LLM completion.

    Each input item is processed through an LLM prompt containing at least one boolean
    completion slot. Only items where the filter field evaluates to True are passed to output.

    Filter field identification:
    1. If a field named 'filter' exists (must be boolean), it's used
    2. Otherwise, the last completion slot is used (must be boolean)
    3. Validation error if the identified field is not boolean

    Output: List[TrackedItem] containing only included items (filter == True)
    """

    type: Literal["Filter"] = "Filter"
    template_text: str = None

    # Internal state for export (not Pydantic fields, just plain attributes)
    _excluded_items: Optional[List[Any]] = None
    _filter_results: Optional[List[Any]] = None
    _processed_items: Optional[List[Any]] = None

    @property
    def template(self) -> str:
        return self.template_text

    def validate_template(self):
        """Validate template has at least one boolean field."""
        try:
            # Just check for [[bool: or [[boolean: or [[decide: syntax without parsing
            bool_pattern = r"\[\[(bool|boolean|decide):"
            if not re.search(bool_pattern, self.template_text, re.IGNORECASE):
                raise ValueError(
                    f"Filter node '{self.name}' template must have at least one boolean completion slot (use [[bool:fieldname]])"
                )
            return True
        except Exception as e:
            logger.error(f"Template validation error: {e}")
            raise e

    def _identify_filter_field(self) -> str:
        """Identify which boolean field to use for filtering.

        Returns:
            str: The field name to use for filtering
        """
        # Find all boolean field names using regex
        # Pattern matches: [[bool:name]], [[boolean:name]], [[decide:name]]
        bool_pattern = r"\[\[(bool|boolean|decide):(\w+)\]"
        matches = re.findall(bool_pattern, self.template_text, re.IGNORECASE)

        if not matches:
            raise ValueError(f"Filter node '{self.name}' has no boolean fields")

        # Extract field names (second group from each match)
        boolean_fields = [m[1] for m in matches]

        # Convention: prefer field named 'filter'
        if "filter" in boolean_fields:
            return "filter"

        # Fallback: use last boolean field
        return boolean_fields[-1]

    def _extract_filter_value(self, result: ChatterResult, filter_field: str) -> bool:
        """Extract the boolean filter value from a ChatterResult.

        Args:
            result: ChatterResult from LLM
            filter_field: Name of the field to extract

        Returns:
            bool: The filter decision (defaults to False if extraction fails)
        """
        try:
            if hasattr(result, "outputs") and filter_field in result.outputs:
                return bool(result.outputs[filter_field])
            elif hasattr(result, "results") and filter_field in result.results:
                return bool(result.results[filter_field].output)
            else:
                logger.warning(
                    f"Could not find filter field '{filter_field}' in result, defaulting to False (exclude)"
                )
                return False
        except Exception as e:
            logger.warning(
                f"Error extracting filter value: {e}, defaulting to False (exclude)"
            )
            return False

    async def run(self) -> List[Any]:
        """Process each item through filter template, split into included/excluded."""
        await super().run()

        # Import here to avoid circular import
        from .batch import BatchList

        input_data = self.context[self.inputs[0]] if self.inputs else None
        if not input_data:
            raise Exception("Filter node must have input data")

        if isinstance(input_data, BatchList):
            raise Exception("Filter node does not support batch input")

        items = await self.get_items()
        filtered_context = self.context

        # Store for export
        self._processed_items = items

        # Identify which field to use for filtering
        filter_field = self._identify_filter_field()
        logger.info(
            f"Filter node '{self.name}' using field '{filter_field}' for filtering"
        )

        # Process all items concurrently
        results = [None] * len(items)

        async with anyio.create_task_group() as tg:
            for idx, item in enumerate(items):

                async def run_and_store(index=idx, item=item):
                    async with semaphore:
                        # Collect extra kwargs for LLM
                        extra_kwargs = {}
                        if self.max_tokens is not None:
                            extra_kwargs['max_tokens'] = self.max_tokens
                        extra_kwargs['temperature'] = self.temperature
                        extra_kwargs['seed'] = self.dag.config.seed

                        # Run the filter template
                        chatter_result = await chatter_async(
                            multipart_prompt=self.template,
                            context={**filtered_context, **item},
                            model=self.get_model(),
                            credentials=self.dag.config.llm_credentials,
                            action_lookup=get_action_lookup(),
                            extra_kwargs=extra_kwargs,
                        )
                        results[index] = chatter_result

                tg.start_soon(run_and_store)

        # Store all results for export
        self._filter_results = results

        # Split items based on filter field
        included_items = []
        excluded_items = []

        for idx, (item, result) in enumerate(zip(items, results)):
            filter_value = self._extract_filter_value(result, filter_field)

            # Get the original TrackedItem from the item dict
            if hasattr(item, "tracked_item") and isinstance(
                item.tracked_item, TrackedItem
            ):
                tracked = item.tracked_item
            elif "tracked_item" in item:
                tracked = item["tracked_item"]
            elif isinstance(item, TrackedItem):
                tracked = item
            else:
                # Fallback: create TrackedItem from content
                content = (
                    item.get("input", str(item))
                    if isinstance(item, dict)
                    else str(item)
                )
                tracked = TrackedItem(
                    content=content, source_id=f"item_{idx}", metadata={}
                )

            # Create a clean copy of TrackedItem to avoid serialization issues
            # Filter out any non-serializable objects from metadata
            clean_metadata = {}
            if tracked.metadata:
                for k, v in tracked.metadata.items():
                    # Only include simple serializable types
                    if isinstance(v, (str, int, float, bool, type(None))):
                        clean_metadata[k] = v
                    elif isinstance(v, (list, dict)):
                        # Include simple containers (assume they're serializable)
                        try:
                            json.dumps(v)
                            clean_metadata[k] = v
                        except (TypeError, ValueError):
                            # Skip non-serializable items
                            pass

            clean_tracked = TrackedItem(
                content=tracked.content,
                source_id=tracked.source_id,
                metadata=clean_metadata,
            )

            if filter_value:
                included_items.append(clean_tracked)
            else:
                excluded_items.append(clean_tracked)

        # Store excluded items for export
        self._excluded_items = excluded_items

        # Store included items as output (convert to dicts for serialization)
        self.output = [
            {
                "content": item.content,
                "source_id": item.source_id,
                "metadata": item.metadata,
            }
            for item in included_items
        ]

        logger.info(
            f"Filter '{self.name}': {len(included_items)} included, {len(excluded_items)} excluded"
        )

        return self.output

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, included/excluded items and filter statistics."""
        # Get base metadata from parent
        result = super().result()

        included_rows = []
        if self.output:
            for idx, item in enumerate(self.output):
                included_rows.append(
                    {
                        "index": idx,
                        "source_id": (
                            item.get("source_id")
                            if isinstance(item, dict)
                            else getattr(item, "source_id", None)
                        ),
                        "content": (
                            item.get("content")
                            if isinstance(item, dict)
                            else getattr(item, "content", None)
                        ),
                        "metadata": (
                            item.get("metadata")
                            if isinstance(item, dict)
                            else getattr(item, "metadata", None)
                        ),
                    }
                )

        excluded_rows = []
        if self._excluded_items:
            for idx, item in enumerate(self._excluded_items):
                excluded_rows.append(
                    {
                        "index": idx,
                        "source_id": getattr(item, "source_id", None),
                        "content": getattr(item, "content", None),
                        "metadata": getattr(item, "metadata", None),
                    }
                )

        # Add Filter-specific data
        result["included"] = pd.DataFrame(included_rows)
        result["excluded"] = pd.DataFrame(excluded_rows)
        result["filter_field"] = (
            self._identify_filter_field() if self.template_text else None
        )
        result["metadata"]["num_included"] = len(included_rows)
        result["metadata"]["num_excluded"] = len(excluded_rows)

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Filter node with included/excluded items and all prompts."""
        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

        # Write summary statistics
        n_included = len(self.output) if self.output else 0
        n_excluded = len(self._excluded_items) if self._excluded_items else 0
        n_total = n_included + n_excluded

        pct_included = (n_included / n_total * 100) if n_total > 0 else 0
        pct_excluded = (n_excluded / n_total * 100) if n_total > 0 else 0

        summary = f"""Filter Summary
==============
Total items processed: {n_total}
Included (filter=True): {n_included} ({pct_included:.1f}%)
Excluded (filter=False): {n_excluded} ({pct_excluded:.1f}%)

Filter field: {self._identify_filter_field() if self.template_text else 'unknown'}
"""
        (folder / "filter_summary.txt").write_text(summary)

        # Export included items (outputs folder) - reconstruct TrackedItems from output dicts
        if self.output:
            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            for idx, item in enumerate(self.output):
                # Handle both dict (from self.output) and TrackedItem (from _excluded_items)
                if isinstance(item, dict):
                    # Reconstruct from dict
                    safe_id = item["source_id"].replace("/", "_").replace("\\", "_")
                    (outputs_folder / f"{idx:04d}_{safe_id}.txt").write_text(
                        item["content"]
                    )

                    if item.get("metadata"):
                        (
                            outputs_folder / f"{idx:04d}_{safe_id}_metadata.json"
                        ).write_text(
                            json.dumps(item["metadata"], indent=2, default=str)
                        )
                elif isinstance(item, TrackedItem):
                    safe_id = item.safe_id
                    (outputs_folder / f"{idx:04d}_{safe_id}.txt").write_text(
                        item.content
                    )

                    if item.metadata:
                        (
                            outputs_folder / f"{idx:04d}_{safe_id}_metadata.json"
                        ).write_text(json.dumps(item.metadata, indent=2, default=str))

        # Export excluded items (excluded folder)
        if self._excluded_items:
            excluded_folder = folder / "excluded"
            excluded_folder.mkdir(exist_ok=True)

            for idx, item in enumerate(self._excluded_items):
                if isinstance(item, TrackedItem):
                    safe_id = item.safe_id
                    (excluded_folder / f"{idx:04d}_{safe_id}.txt").write_text(
                        item.content
                    )

                    if item.metadata:
                        (
                            excluded_folder / f"{idx:04d}_{safe_id}_metadata.json"
                        ).write_text(json.dumps(item.metadata, indent=2, default=str))

        # Export all prompts and responses (prompts folder)
        if self._filter_results and self._processed_items:
            prompts_folder = folder / "prompts"
            prompts_folder.mkdir(exist_ok=True)

            for idx, (item, result) in enumerate(
                zip(self._processed_items, self._filter_results)
            ):
                # Get source_id for filename
                safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))
                file_prefix = f"{idx:04d}_{safe_id}"

                try:
                    # Extract and write prompt
                    if hasattr(result, "results") and result.results:
                        first_seg = next(iter(result.results.values()))
                        if hasattr(first_seg, "prompt"):
                            (prompts_folder / f"{file_prefix}_prompt.md").write_text(
                                first_seg.prompt
                            )

                    # Write response text
                    if hasattr(result, "response"):
                        (prompts_folder / f"{file_prefix}_response.txt").write_text(
                            str(result.response)
                        )

                    # Write full ChatterResult JSON
                    (prompts_folder / f"{file_prefix}_response.json").write_text(
                        safe_json_dump(result)
                    )

                except Exception as e:
                    logger.warning(f"Failed to export prompt/response {idx}: {e}")
