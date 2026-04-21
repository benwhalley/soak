"""Typed event system for DAG execution.

Nodes emit events during execution. Consumers (Django web UI, CLI tqdm, tests)
register handlers to observe progress, errors, cost changes, and streaming output.
"""

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union

from tqdm import tqdm

if TYPE_CHECKING:
    from soak.models.cost_tracker import GlobalCostTracker
    from soak.models.nodes.base import DAGNode

logger = logging.getLogger(__name__)


# -- event types --


@dataclass
class NodeProgress:
    """Emitted when a node reports item-level progress."""

    node_name: str
    done: int
    total: int
    cost: float
    fresh_cost: float = 0.0


@dataclass
class NodeCompleted:
    """Emitted when a node finishes (success, failure, or skip)."""

    node: "DAGNode"


@dataclass
class NodeError:
    """Emitted when an LLM error occurs during node execution."""

    node_name: str
    item_index: Optional[int]
    error: Exception  # typically LLMError with .prompt, .model_name


@dataclass
class RateLimitHit:
    """Emitted on each rate-limit (429) retry."""

    node_name: str
    model_name: str


@dataclass
class StreamingEvent:
    """Emitted for each streaming token/lifecycle event."""

    node_name: str
    item_index: Optional[int]
    event: Any  # struckdown SlotStreamStart / TokenDelta / SlotCompleted


@dataclass
class CostUpdated:
    """Emitted when the global cost tracker records new costs."""

    total_cost: float
    fresh_cost: float
    prompt_tokens: int
    completion_tokens: int
    fresh_count: int
    cached_count: int


DAGEvent = Union[
    NodeProgress, NodeCompleted, NodeError, RateLimitHit, StreamingEvent, CostUpdated
]


class DAGEventHandler(Protocol):
    def on_event(self, event: DAGEvent) -> None: ...


# -- built-in CLI handler --

_DESC_WIDTH = 35
_NCOLS = 120
_MIN_INTERVAL = 0.1


class CLIEventHandler:
    """Event handler that displays tqdm progress bars for CLI usage.

    Replaces the old ProgressManager. Creates/updates/closes tqdm bars
    in response to NodeProgress and NodeCompleted events.
    """

    def __init__(self, cost_tracker: Optional["GlobalCostTracker"] = None) -> None:
        self.cost_tracker = cost_tracker
        self._bars: dict[str, tqdm] = {}
        self._node_fresh_costs: dict[str, float] = {}

    def on_event(self, event: DAGEvent) -> None:
        match event:
            case NodeProgress(node_name=name, done=done, total=total,
                              cost=cost, fresh_cost=fresh_cost):
                if name not in self._bars:
                    self._bars[name] = self._create_bar(name, total)
                bar = self._bars[name]
                if isinstance(bar, _CostProgressBar):
                    cost_delta = fresh_cost - self._node_fresh_costs.get(name, 0.0)
                    if cost_delta > 0:
                        self._node_fresh_costs[name] = fresh_cost
                        bar.update_cost(cost_delta, 0)
                increment = done - bar.n
                if increment > 0:
                    bar.update(increment)
                elif done != bar.n:
                    bar.n = done
                    bar.refresh()
            case NodeCompleted(node=node):
                if node.name in self._bars:
                    bar = self._bars[node.name]
                    if bar.n < bar.total:
                        bar.update(bar.total - bar.n)
                    bar.close()
                    del self._bars[node.name]
            case _:
                pass

    def _create_bar(self, node_name: str, total: int) -> tqdm:
        padded = node_name.ljust(_DESC_WIDTH)
        if self.cost_tracker:
            return _CostProgressBar(
                total=total,
                desc=padded,
                unit="item",
                file=sys.stderr,
                ncols=_NCOLS,
                leave=True,
                mininterval=_MIN_INTERVAL,
            )
        return tqdm(
            total=total,
            desc=padded,
            unit="item",
            file=sys.stderr,
            ncols=_NCOLS,
            leave=True,
            mininterval=_MIN_INTERVAL,
        )

    def close(self) -> None:
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()


class _CostProgressBar(tqdm):
    """tqdm bar with per-node cost display."""

    def __init__(self, *args, **kwargs):
        self._node_fresh_cost: float = 0.0
        self._node_tokens: int = 0
        super().__init__(*args, **kwargs)

    def update_cost(self, fresh_cost: float, tokens: int) -> None:
        self._node_fresh_cost += fresh_cost
        self._node_tokens += tokens
        tokens_k = self._node_tokens / 1000
        self.set_postfix_str(
            f"${self._node_fresh_cost:.4f} | {tokens_k:.1f}k tokens", refresh=False
        )
