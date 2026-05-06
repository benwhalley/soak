"""GroupBy node for grouping items by field values."""

import logging
from typing import Any, List, Literal, Union

from soak.models.base import TrackedItem

from .base import DAGNode
from .batch import BatchList

logger = logging.getLogger(__name__)


class GroupBy(DAGNode):
    """Group items by field value(s).

    Adds a layer of grouping to the input:
    - Flat input → BatchList (first grouping layer)
    - BatchList input → nested BatchList (adds grouping layer)

    Multi-field grouping:
    - group_by: ["doc", "category"] creates nested structure
    - Equivalent to sequential GroupBy operations
    """

    type: Literal["GroupBy"] = "GroupBy"
    group_by: Union[str, List[str]]  # Field path or list of field paths

    async def run(self) -> BatchList:
        """Group items by field value(s)."""
        await super().run()

        # Get input data (may be flat list or BatchList)
        if self.inputs:
            input_data = self.context[self.inputs[0]]
        else:
            input_data = self.dag.config.load_documents()

        # If input is BatchList, flatten first
        if isinstance(input_data, BatchList):
            items = input_data.flatten_one_level()
        else:
            items = input_data if isinstance(input_data, list) else [input_data]

        # Handle multi-field grouping (nested structure)
        if isinstance(self.group_by, list) and len(self.group_by) > 1:
            # Sequential grouping: group by first field, then by second within each group, etc.
            result = self._group_sequential(items, self.group_by)
        else:
            # Single field grouping
            field = (
                self.group_by if isinstance(self.group_by, str) else self.group_by[0]
            )
            result = self._group_by_field(items, field)

        self.output = result
        return result

    def _group_sequential(self, items: List[Any], fields: List[str]) -> BatchList:
        """Group by multiple fields sequentially (creates nested structure).

        Args:
            items: Flat list of items
            fields: List of field paths to group by

        Returns:
            Nested BatchList with depth = len(fields)
        """
        if not fields:
            raise ValueError("fields list cannot be empty")

        # Group by first field
        result = self._group_by_field(items, fields[0])

        # If more fields, recurse on each batch
        if len(fields) > 1:
            nested_batches = []
            for batch in result.batches:
                # Group this batch by remaining fields
                nested = self._group_sequential(batch, fields[1:])
                nested_batches.append(nested)

            return BatchList(
                batches=nested_batches,
                group_field=",".join(fields),
                group_keys=result.group_keys,
            )
        else:
            return result

    def _group_by_field(self, items: List[Any], field_path: str) -> BatchList:
        """Group items by a single field value.

        Args:
            items: Flat list of items to group
            field_path: Path to field (e.g., "sources[0]", "metadata.category")

        Returns:
            BatchList with one batch per unique field value
        """
        # Group items by field value
        groups = {}
        group_keys_order = []  # Preserve order of first appearance

        for item in items:
            try:
                key = self._extract_field(item, field_path)
                key_str = str(key) if key is not None else "None"

                if key_str not in groups:
                    groups[key_str] = []
                    group_keys_order.append(key_str)

                groups[key_str].append(item)

            except Exception as e:
                raise ValueError(
                    f"GroupBy '{self.name}': Failed to extract field '{field_path}' "
                    f"from item {item}: {e}. All items must have the specified field."
                )

        # Convert to BatchList
        batches = [groups[key] for key in group_keys_order]

        logger.info(
            f"GroupBy '{self.name}': Created {len(batches)} groups by '{field_path}'"
        )

        return BatchList(
            batches=batches,
            group_field=field_path,
            group_keys=group_keys_order,
        )

    def _get_available_fields(self, obj: Any) -> str:
        """Get formatted list of available fields/properties for an object.

        Args:
            obj: Object to inspect

        Returns:
            Formatted string listing available fields
        """
        if isinstance(obj, TrackedItem):
            fields = ["content", "id", "sources", "metadata"]
            properties = ["lineage", "depth", "safe_id", "root_document"]
            return (
                f"Available fields: {', '.join(fields)}\n"
                f"Available properties: {', '.join(properties)}"
            )
        elif isinstance(obj, dict):
            keys = list(obj.keys())
            if keys:
                return f"Available keys: {', '.join(str(k) for k in keys[:20])}"
            else:
                return "Available keys: (empty dict)"
        elif isinstance(obj, (list, tuple)):
            return (
                f"Available indices: 0-{len(obj)-1}"
                if obj
                else "Available indices: (empty list)"
            )
        else:
            # For other objects, list attributes (excluding private/magic methods)
            attrs = [a for a in dir(obj) if not a.startswith("_")]
            if attrs:
                return f"Available attributes: {', '.join(attrs[:20])}"
            else:
                return "No accessible attributes"

    def _extract_field(self, item: Any, field_path: str) -> Any:
        """Extract field value from item using path like 'sources[0]' or 'metadata.category'.

        Args:
            item: Item to extract from (TrackedItem or dict)
            field_path: Dot-notation path with optional array indices

        Returns:
            Field value

        Raises:
            ValueError: If field doesn't exist or path is invalid
        """
        if isinstance(item, TrackedItem):
            obj = item
        elif isinstance(item, dict):
            obj = item
        elif hasattr(item, "model_dump"):
            # Pydantic model (Code, Theme, etc.) -- access attributes directly
            obj = item
        else:
            raise ValueError(f"Cannot extract field from {type(item)}")

        # Parse field_path: "sources[0]" → ["sources", 0], "metadata.category" → ["metadata", "category"]
        # Handle both bracket and dot notation
        parts = (
            field_path.replace("][", "].[")
            .replace("[", ".")
            .replace("]", "")
            .split(".")
        )

        for part in parts:
            if not part:  # Skip empty parts from split
                continue

            # Try as integer index
            if part.isdigit():
                idx = int(part)
                if isinstance(obj, (list, tuple)):
                    if idx < len(obj):
                        obj = obj[idx]
                    else:
                        raise ValueError(
                            f"Index {idx} out of range for field path '{field_path}'"
                        )
                else:
                    raise ValueError(
                        f"Cannot index {type(obj)} with integer in path '{field_path}'"
                    )
            # Try as attribute
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            # Try as dict key
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                available = self._get_available_fields(obj)
                obj_type = (
                    "TrackedItem"
                    if isinstance(obj, TrackedItem)
                    else type(obj).__name__
                )
                raise ValueError(
                    f"Field '{part}' not found in {obj_type} (path: '{field_path}')\n{available}"
                )

        return obj
