"""Custom progress display components for real-time cost tracking."""

from typing import Optional

from tqdm import tqdm

from .cost_tracker import GlobalCostTracker


class GlobalCostDisplay:
    """Persistent status line showing pipeline-wide costs."""

    def __init__(self, tracker: GlobalCostTracker) -> None:
        self.tracker = tracker
        self.bar: Optional[tqdm] = None

    def __enter__(self):
        """Create persistent progress bar for global cost display."""
        import sys

        # Initial display
        snapshot = self.tracker.get_snapshot()
        self.bar = tqdm(
            total=0,
            desc="Pipeline",
            position=0,
            leave=True,
            bar_format="{desc}: {postfix}",
            file=sys.stderr,
            postfix=snapshot.format_compact(),
        )
        # register callback to update display when costs change
        self.tracker.register_callback(self.refresh)
        return self

    def __exit__(self, *args):
        """Clean up progress bar."""
        if self.bar:
            self.bar.close()

    def refresh(self) -> None:
        """Update display with current costs."""
        if self.bar and not self.bar.disable:
            snapshot = self.tracker.get_snapshot()
            self.bar.set_postfix_str(snapshot.format_compact())
            self.bar.refresh()


class CostProgressBar(tqdm):
    """tqdm progress bar with per-node cost display."""

    # Fixed width for description to align progress bars
    DESC_WIDTH = 35

    def __init__(self, tracker: GlobalCostTracker, node_name: str, *args, **kwargs):
        self.tracker = tracker
        self.node_name = node_name
        self._node_fresh_cost: float = 0.0
        self._node_tokens: int = 0

        # offset position to make room for global display
        if "position" not in kwargs:
            kwargs["position"] = 1

        # Set fixed bar width if not specified
        if "ncols" not in kwargs:
            kwargs["ncols"] = 120

        super().__init__(*args, **kwargs)
        # Pad description to fixed width for alignment
        padded_name = node_name.ljust(self.DESC_WIDTH)
        self.set_description(padded_name)

    def update_cost(self, fresh_cost: float, tokens: int) -> None:
        """Update node-level cost display with fresh cost only."""
        self._node_fresh_cost += fresh_cost
        self._node_tokens += tokens
        tokens_k = self._node_tokens / 1000
        self.set_postfix_str(f"${self._node_fresh_cost:.4f} | {tokens_k:.1f}k tokens")
