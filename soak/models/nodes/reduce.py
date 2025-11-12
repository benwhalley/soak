"""Reduce node for aggregating inputs."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

from soak.models.dag import render_strict_template

from .base import DAGNode

logger = logging.getLogger(__name__)


class Reduce(DAGNode):
    """Node that reduces items by peeling one layer of grouping.

    Behavior:
    - Nested BatchList: Reduce each inner BatchList, return flat list (peel one layer)
    - Single-level BatchList: Reduce each batch, return flat list
    - Flat list: Reduce to single string

    When exclude_overlap=True, uses core content from TrackedItems (excluding overlap regions)
    to avoid duplicating overlapped content when rejoining chunks from Split nodes.
    """

    type: Literal["Reduce"] = "Reduce"
    template: str = "{{input}} "
    exclude_overlap: bool = True  # Exclude overlap by default when joining chunks

    async def run(
        self,
    ) -> Union[List[Union[str, "TrackedItem"]], Union[str, "TrackedItem"]]:
        """Reduce items, peeling one layer of BatchList nesting."""
        await super().run()

        # Import here to avoid circular import
        from .batch import BatchList

        # Get input data
        if len(self.inputs) > 1:
            raise ValueError("Reduce nodes must only have one input.")

        if self.inputs:
            input_data = self.context[self.inputs[0]]
        else:
            input_data = self.dag.config.documents

        # Handle different input types
        if isinstance(input_data, BatchList):
            if input_data.is_nested():
                # Nested: reduce each inner BatchList
                batch_results = []
                for inner_batch in input_data.batches:
                    # Recurse to handle the inner BatchList
                    reduced = await self._reduce_batchlist(inner_batch)
                    batch_results.append(reduced)

                self.output = batch_results
                logger.info(
                    f"Reduce '{self.name}': Reduced nested BatchList to {len(batch_results)} items"
                )
                return batch_results
            else:
                # Single level: reduce each batch
                batch_results = []
                for batch in input_data.batches:
                    reduced = await self._reduce_items(batch)
                    batch_results.append(reduced)

                self.output = batch_results
                logger.info(
                    f"Reduce '{self.name}': Reduced {len(input_data.batches)} batches"
                )
                return batch_results
        else:
            # Flat input: reduce to single value
            items = input_data if isinstance(input_data, list) else [input_data]
            result = await self._reduce_items(items)
            self.output = result
            logger.info(
                f"Reduce '{self.name}': Reduced {len(items)} items to single value"
            )
            return result

    async def _reduce_batchlist(self, batchlist) -> Union[str, "TrackedItem"]:
        """Reduce a BatchList (or nested list) to a single value.

        Args:
            batchlist: BatchList or list to reduce

        Returns:
            TrackedItem if inputs are TrackedItems, otherwise string
        """
        from .batch import BatchList

        if isinstance(batchlist, BatchList):
            # Flatten the BatchList one level and reduce
            items = batchlist.flatten_one_level()
        else:
            items = batchlist

        return await self._reduce_items(items)

    async def _reduce_items(self, items: List[Any]) -> Union[str, "TrackedItem"]:
        """Reduce a list of items to single value using template.

        When exclude_overlap=True and items are TrackedItems with overlap metadata,
        uses core content (excluding overlap regions) to avoid duplicating content.

        When input items are TrackedItems, returns a TrackedItem to preserve source tracking.

        Args:
            items: Flat list of items to reduce

        Returns:
            TrackedItem if input items are TrackedItems, otherwise string
        """
        from soak.models.base import TrackedItem

        rendered = []
        all_sources = []
        has_tracked_items = False

        for item in items:
            # Use core content if excluding overlap and item has overlap info
            if self.exclude_overlap and isinstance(item, TrackedItem):
                has_tracked_items = True
                all_sources.extend(item.sources)

                if item.content_excluding_overlap is not None:
                    # Create modified TrackedItem with core content for rendering
                    core_content = item.get_core_content()
                    # Create a temporary item for template context
                    temp_item = TrackedItem(
                        content=core_content,
                        id=item.id,
                        sources=item.sources,
                        metadata=item.metadata,
                    )
                    context = {"input": temp_item}
                else:
                    # No overlap metadata, use full item
                    context = {"input": item}
            elif isinstance(item, TrackedItem):
                has_tracked_items = True
                all_sources.extend(item.sources)
                context = {"input": item}
            elif isinstance(item, dict):
                context = {**item}
            elif hasattr(item, "__dict__"):
                # Handle objects with attributes
                context = {"input": item}
            else:
                # Plain values (strings, etc.)
                context = {"input": item}

            rendered.append(render_strict_template(self.template, context))

        combined_content = "\n".join(rendered)

        # Return TrackedItem if inputs were TrackedItems to preserve provenance
        if has_tracked_items:
            # Use first item's ID as base, or construct from node name
            first_id = items[0].id if isinstance(items[0], TrackedItem) else self.name
            return TrackedItem(
                content=combined_content,
                id=f"{first_id}__{self.name}",
                sources=list(
                    dict.fromkeys(all_sources)
                ),  # deduplicate while preserving order
                metadata={},
            )

        return combined_content

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
        if self.template:
            (folder / "reduce_template.md").write_text(self.template)

        # Write reduced output to outputs/ folder for consistency
        if self.output:
            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            if isinstance(self.output, str):
                (outputs_folder / "reduced.txt").write_text(self.output)
            elif isinstance(self.output, list):
                # Handle list of reduced outputs
                for idx, item in enumerate(self.output, 1):
                    (outputs_folder / f"reduced_{idx:03d}.txt").write_text(str(item))
