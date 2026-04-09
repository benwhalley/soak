"""Reduce node for aggregating inputs."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from soak.models.dag import render_strict_template
from soak.models.utils import unwrap_complete_items

from .base import DAGNode

logger = logging.getLogger(__name__)


class Reduce(DAGNode):
    """Node that reduces items by peeling one layer of grouping.

    Behavior (template mode, default):
    - Nested BatchList: Reduce each inner BatchList, return flat list (peel one layer)
    - Single-level BatchList: Reduce each batch, return flat list
    - Flat list: Reduce to single string

    Behavior (items_field mode):
    - When items_field is set, extracts structured items from containers (e.g., StruckdownResult)
    - Flattens all inputs into a single list of extracted items
    - Returns the list directly without template rendering
    - Useful for collecting Code/Theme objects from multiple Map outputs

    When exclude_overlap=True, uses core content from TrackedItems (excluding overlap regions)
    to avoid duplicating overlapped content when rejoining chunks from Split nodes.

    Example YAML:
        # Template mode (default): join text
        - name: combined_text
          type: Reduce
          template: "{{input.content}}"
          inputs: [chunks]

        # Items field mode: extract and flatten Code objects
        - name: all_codes
          type: Reduce
          items_field: codes
          inputs: [coded_chunks]  # List of StruckdownResults with codes
    """

    type: Literal["Reduce"] = "Reduce"
    template: str = "{{input}} "
    exclude_overlap: bool = True  # Exclude overlap by default when joining chunks
    items_field: Optional[str] = (
        None  # When set, extract items instead of rendering template
    )

    async def run(
        self,
    ) -> Union[List[Any], str, "TrackedItem"]:
        """Reduce items, peeling one layer of BatchList nesting."""
        await super().run()

        # Import here to avoid circular import
        from .batch import BatchList

        # Get input data
        if len(self.inputs) > 1:
            raise ValueError("Reduce nodes must only have one input.")

        # Items field mode: extract and flatten structured items.
        # Bypass self.context (which wraps StruckdownResults in _TemplateProxy for
        # template rendering, hiding individual items from unwrap_complete_items).
        if self.items_field:
            if self.inputs:
                input_name = self.inputs[0]
                if input_name == "documents":
                    raw_input = self.dag.config.documents
                else:
                    input_node = self.dag.nodes_dict.get(input_name)
                    raw_input = input_node.output if input_node else None
            else:
                raw_input = self.dag.config.documents
            return await self._reduce_with_items_field(raw_input)

        if self.inputs:
            input_data = self.context[self.inputs[0]]
        else:
            input_data = self.dag.config.documents

        # Template mode: original behavior
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

    async def _reduce_with_items_field(self, input_data: Any) -> List[Any]:
        """Extract and flatten structured items using items_field.

        Flattens all input (including BatchList) and extracts items from containers.

        Args:
            input_data: Input data (list, BatchList, or single item)

        Returns:
            Flat list of extracted items (e.g., Code objects)
        """
        from .batch import BatchList

        # Flatten all input to a single list
        if isinstance(input_data, BatchList):
            raw_items = input_data.flatten_all()
        elif isinstance(input_data, list):
            raw_items = input_data
        else:
            raw_items = [input_data]

        # Extract items using items_field
        extracted = unwrap_complete_items(raw_items, self.items_field)

        self.output = extracted
        logger.info(
            f"Reduce '{self.name}': Extracted {len(extracted)} items "
            f"(items_field='{self.items_field}') from {len(raw_items)} inputs"
        )
        return extracted

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

        # filter out None items (from skipped/failed upstream items)
        items = [item for item in items if item is not None]

        if not items:
            logger.warning(
                f"Reduce '{self.name}': No valid items to reduce (all were None/skipped)"
            )
            return ""

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
        """Returns dict with metadata and reduced output.

        When items_field is used, wraps output in appropriate container types
        (CodeList for codes, Themes for themes) to provide consistent interface
        with Transform nodes.
        """
        from soak.models.base import Code, CodeList, Theme, Themes

        # Get base metadata from parent
        result = super().result()

        # Add Reduce-specific data
        result["output"] = self.output
        result["output_type"] = type(self.output).__name__ if self.output else None

        # Wrap list outputs in container types for template consistency
        response_obj = None
        if self.items_field and isinstance(self.output, list) and self.output:
            first_item = self.output[0]
            if isinstance(first_item, Code):
                response_obj = CodeList(codes=self.output)
            elif isinstance(first_item, Theme):
                response_obj = Themes(themes=self.output)

        result["response_obj"] = response_obj
        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Reduce node details."""
        import json

        super().export(folder, unique_id=unique_id)

        # Write reduce template (only in template mode)
        if self.template and not self.items_field:
            (folder / "reduce_template.md").write_text(self.template)

        # Write reduced output to outputs/ folder for consistency
        if self.output:
            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            if isinstance(self.output, str):
                (outputs_folder / "reduced.txt").write_text(self.output)
            elif isinstance(self.output, list):
                # Handle list of outputs
                if self.items_field:
                    # Items field mode: export as JSON for structured items
                    items_json = []
                    for item in self.output:
                        if hasattr(item, "model_dump"):
                            items_json.append(item.model_dump())
                        else:
                            items_json.append(str(item))
                    (outputs_folder / "items.json").write_text(
                        json.dumps(items_json, indent=2, default=str)
                    )
                    # Also write human-readable text
                    (outputs_folder / "items.txt").write_text(
                        "\n\n---\n\n".join(str(item) for item in self.output)
                    )
                else:
                    # Template mode: write each reduced output
                    for idx, item in enumerate(self.output, 1):
                        (outputs_folder / f"reduced_{idx:03d}.txt").write_text(
                            str(item)
                        )
