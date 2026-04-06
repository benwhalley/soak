"""Batch node for grouping items into batches."""

import itertools
import logging
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel
from struckdown import StruckdownResult

from .base import DAGNode

logger = logging.getLogger(__name__)


class BatchList(BaseModel):
    """Container for batched results with arbitrary nesting depth.

    Supports nested grouping where batches can contain other BatchLists.
    Tracks grouping metadata for export and debugging.
    """

    batches: List[Any]  # Can contain items or nested BatchLists
    group_field: Optional[str] = None  # Field used for grouping
    group_keys: Optional[List[str]] = None  # Keys for each batch

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        """Return total number of items across all batches."""
        return len(self.flatten_all())

    def depth(self) -> int:
        """Calculate nesting depth of this BatchList."""
        if not self.batches or not isinstance(self.batches[0], BatchList):
            return 1
        return 1 + self.batches[0].depth()

    def is_nested(self) -> bool:
        """Check if this contains nested BatchLists."""
        return (
            self.batches
            and len(self.batches) > 0
            and isinstance(self.batches[0], BatchList)
        )

    def flatten_one_level(self) -> List[Any]:
        """Peel off one layer of nesting.

        Returns:
            Flat list of items from all batches at this level
        """
        result = []
        for batch in self.batches:
            if isinstance(batch, list):
                result.extend(batch)
            else:
                result.append(batch)
        return result

    def flatten_all(self) -> List[Any]:
        """Recursively flatten all nesting levels.

        Returns:
            Completely flat list with no BatchList structure
        """
        result = []
        for batch in self.batches:
            if isinstance(batch, BatchList):
                # Recursive: flatten nested BatchList
                result.extend(batch.flatten_all())
            elif isinstance(batch, list):
                # Flat batch: extend items
                result.extend(batch)
            else:
                # Single item
                result.append(batch)
        return result


class Batch(DAGNode):
    """Node that groups items into fixed-size batches.

    Adds a layer of grouping to the input:
    - Flat input → BatchList (first grouping layer)
    - BatchList input → nested BatchList (adds grouping layer)
    """

    type: Literal["Batch"] = "Batch"
    batch_size: int = 10

    async def run(self) -> BatchList:
        """Create batches from input."""
        await super().run()

        # Get input data (may be flat list or BatchList)
        if self.inputs:
            input_data = self.context[self.inputs[0]]
        else:
            input_data = self.dag.config.load_documents()

        # If input is BatchList, flatten first then rebatch (adds nesting)
        if isinstance(input_data, BatchList):
            items = input_data.flatten_one_level()
        else:
            items = input_data if isinstance(input_data, list) else [input_data]

        # Create fixed-size batches
        batches = list(itertools.batched(items, self.batch_size))

        # Create BatchList with metadata
        result = BatchList(
            batches=batches,
            group_field=f"batch_size_{self.batch_size}",
            group_keys=[f"batch_{i}" for i in range(len(batches))],
        )

        self.output = result
        logger.info(
            f"Batch '{self.name}': Created {len(batches)} batches of size {self.batch_size}"
        )
        return result
