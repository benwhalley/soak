"""Ungroup node for flattening all BatchList levels."""

import logging
from typing import Any, List, Literal

from .base import DAGNode
from .batch import BatchList

logger = logging.getLogger(__name__)


class Ungroup(DAGNode):
    """Flatten all BatchList nesting levels without processing items.

    Removes all grouping structure:
    - BatchList([[a, b], [c, d]]) → [a, b, c, d]
    - Nested BatchLists are recursively flattened
    - Flat input passes through unchanged
    """

    type: Literal["Ungroup"] = "Ungroup"

    async def run(self) -> List[Any]:
        """Flatten all BatchList levels."""
        await super().run()

        # Get input data
        if self.inputs:
            input_data = self.context[self.inputs[0]]
        else:
            input_data = self.dag.config.load_documents()

        # If BatchList, flatten completely
        if isinstance(input_data, BatchList):
            result = input_data.flatten_all()
            logger.info(
                f"Ungroup '{self.name}': Flattened {input_data.depth()} levels "
                f"to {len(result)} items"
            )
        else:
            # Already flat, pass through
            result = input_data if isinstance(input_data, list) else [input_data]
            logger.debug(
                f"Ungroup '{self.name}': Input already flat ({len(result)} items)"
            )

        self.output = result
        return result
