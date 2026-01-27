"""Custom progress display components for real-time cost tracking."""

import sys
from typing import Callable, Optional

from tqdm import tqdm

from .cost_tracker import GlobalCostTracker

# fixed width for description to align progress bars
DESC_WIDTH = 35
# fixed terminal width for consistent layout
NCOLS = 120
# minimum interval between refreshes (seconds)
MIN_INTERVAL = 0.1


class ProgressManager:
    """Central manager for creating consistently configured progress bars.

    Provides factory methods for creating progress bars with consistent styling.
    Bars stack naturally with leave=True so completed bars remain visible.
    """

    def __init__(self, cost_tracker: Optional[GlobalCostTracker] = None) -> None:
        self.cost_tracker = cost_tracker

    def __enter__(self) -> "ProgressManager":
        """Context manager entry - no special setup needed."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - no special cleanup needed."""
        pass

    def create_progress_bar(
        self,
        total: int,
        desc: str,
        unit: str = "item",
        leave: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> tqdm:
        """Create a consistently configured progress bar.

        Args:
            total: Total number of items
            desc: Description (will be padded to DESC_WIDTH)
            unit: Unit name for items
            leave: Whether to leave the bar after completion (default True)
            progress_callback: Optional callback called on each update

        Returns:
            Configured tqdm progress bar
        """
        padded_desc = desc.ljust(DESC_WIDTH)

        bar = _ManagedProgressBar(
            total=total,
            desc=padded_desc,
            unit=unit,
            file=sys.stderr,
            ncols=NCOLS,
            leave=leave,
            progress_callback=progress_callback,
            mininterval=MIN_INTERVAL,
        )
        return bar

    def create_cost_progress_bar(
        self,
        total: int,
        node_name: str,
        unit: str = "item",
        leave: bool = True,
    ) -> "CostProgressBar":
        """Create a progress bar with cost tracking.

        Args:
            total: Total number of items
            node_name: Name of the node (used in description)
            unit: Unit name for items
            leave: Whether to leave the bar after completion (default True)

        Returns:
            CostProgressBar instance
        """
        bar = CostProgressBar(
            tracker=self.cost_tracker,
            node_name=node_name,
            total=total,
            unit=unit,
            file=sys.stderr,
            ncols=NCOLS,
            leave=leave,
            mininterval=MIN_INTERVAL,
        )
        return bar


class _ManagedProgressBar(tqdm):
    """Progress bar with optional progress callback."""

    def __init__(
        self,
        progress_callback: Optional[Callable[[int], None]] = None,
        **kwargs,
    ):
        self._progress_callback = progress_callback
        super().__init__(**kwargs)

    def update(self, n: int = 1) -> None:
        super().update(n)
        if self._progress_callback:
            self._progress_callback(n)


class CostProgressBar(tqdm):
    """tqdm progress bar with per-node cost display."""

    def __init__(
        self,
        tracker: GlobalCostTracker,
        node_name: str,
        *args,
        **kwargs,
    ):
        self.tracker = tracker
        self.node_name = node_name
        self._node_fresh_cost: float = 0.0
        self._node_tokens: int = 0

        # set fixed bar width if not specified
        if "ncols" not in kwargs:
            kwargs["ncols"] = NCOLS

        super().__init__(*args, **kwargs)
        # pad description to fixed width for alignment
        padded_name = node_name.ljust(DESC_WIDTH)
        self.set_description(padded_name)

    def update_cost(self, fresh_cost: float, tokens: int) -> None:
        """Update node-level cost display with fresh cost only."""
        self._node_fresh_cost += fresh_cost
        self._node_tokens += tokens
        tokens_k = self._node_tokens / 1000
        # refresh=False to let mininterval throttle updates
        self.set_postfix_str(f"${self._node_fresh_cost:.4f} | {tokens_k:.1f}k tokens", refresh=False)
