"""Batch node for grouping items into batches."""

import itertools
import logging
from typing import Any, List, Literal

from .base import ItemsNode

logger = logging.getLogger(__name__)


class BatchList(object):
    """Container for batched results."""
    batches: List[Any]

    def __init__(self, batches: List[Any]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


class Batch(ItemsNode):
    """Node that groups items into fixed-size batches."""
    type: Literal["Batch"] = "Batch"
    batch_size: int = 10

    async def run(self) -> List[List[Any]]:
        await super().run()

        batches_ = self.default_batch(await self.get_items(), self.batch_size)
        self.output = BatchList(batches=batches_)
        return self.output

    def default_batch(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Batch items into lists of size batch_size."""
        return list(itertools.batched(items, batch_size))
