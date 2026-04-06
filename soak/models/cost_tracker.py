"""Thread-safe cost tracking for pipeline execution."""

import threading
from dataclasses import dataclass
from typing import Optional

from struckdown import StruckdownResult


@dataclass
class CostSnapshot:
    """Immutable snapshot of current costs."""

    total_cost: float
    fresh_cost: float
    prompt_tokens: int
    completion_tokens: int
    fresh_count: int
    cached_count: int

    def format_compact(self) -> str:
        """Format for display in progress bars."""
        tokens_k = (self.prompt_tokens + self.completion_tokens) / 1000
        return (
            f"${self.fresh_cost:.4f} fresh / ${self.total_cost:.4f} total | "
            f"{self.fresh_count} fresh / {self.cached_count} cached | "
            f"{tokens_k:.1f}k tokens"
        )


class GlobalCostTracker:
    """Thread-safe accumulator for LLM costs across all nodes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_cost: float = 0.0
        self._fresh_cost: float = 0.0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._fresh_count: int = 0
        self._cached_count: int = 0
        self._callbacks: list = []

    def add_result(self, result: StruckdownResult) -> None:
        """Add a StruckdownResult to the running totals."""
        with self._lock:
            self._total_cost += result.total_cost
            self._fresh_cost += result.fresh_cost
            self._prompt_tokens += result.prompt_tokens
            self._completion_tokens += result.completion_tokens
            self._fresh_count += result.fresh_call_count
            self._cached_count += result.cached_call_count

        # notify callbacks outside lock to avoid deadlock
        for callback in self._callbacks:
            callback()

    def add_embedding_cost(
        self, fresh_cost: float, fresh_tokens: int, fresh_count: int
    ) -> None:
        """Add embedding API costs to the running totals.

        This is called by the embedding cost callback after each embedding batch.
        Embeddings are tracked as 'prompt tokens' since they're input-only.

        Args:
            fresh_cost: USD cost from fresh (non-cached) API calls
            fresh_tokens: Token count from fresh API calls
            fresh_count: Number of fresh embedding calls
        """
        with self._lock:
            self._total_cost += fresh_cost
            self._fresh_cost += fresh_cost
            self._prompt_tokens += fresh_tokens
            self._fresh_count += fresh_count

        # notify callbacks outside lock to avoid deadlock
        for callback in self._callbacks:
            callback()

    def get_snapshot(self) -> CostSnapshot:
        """Get immutable snapshot of current costs."""
        with self._lock:
            return CostSnapshot(
                total_cost=self._total_cost,
                fresh_cost=self._fresh_cost,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                fresh_count=self._fresh_count,
                cached_count=self._cached_count,
            )

    def register_callback(self, callback) -> None:
        """Register callback to be called when costs update."""
        self._callbacks.append(callback)
