"""Declarative output invariants for DAG nodes.

A pipeline author annotates a node in YAML with `invariants: [name1, name2]`.
After the node's `run()` completes successfully, `DAGNode.validate_output()`
looks each name up in `INVARIANT_REGISTRY` and calls it. If a check is
violated it raises `NodeInvariantError`, which propagates through the same
exception path as any other node failure -- so downstream consumers
(notably the web app's auto-retry policy) can react to bad output without
needing a separate signal.

Register a check with the `@invariant` decorator. The function takes
`(node, context)` and either returns (pass) or raises `NodeInvariantError`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from soak.error_handlers import NodeInvariantError

logger = logging.getLogger(__name__)


INVARIANT_REGISTRY: Dict[str, Callable[[Any, Dict[str, Any]], None]] = {}


def invariant(name: str):
    """Register a callable as a named invariant available to YAML pipelines."""

    def decorator(fn: Callable[[Any, Dict[str, Any]], None]):
        if name in INVARIANT_REGISTRY:
            logger.warning(f"Invariant '{name}' is being re-registered")
        INVARIANT_REGISTRY[name] = fn
        return fn

    return decorator


def run_invariants(node, names: List[str], context: Dict[str, Any]) -> None:
    """Run each named invariant. First failure raises NodeInvariantError."""
    for name in names or []:
        fn = INVARIANT_REGISTRY.get(name)
        if fn is None:
            raise NodeInvariantError(
                node_name=node.name,
                invariant_name=name,
                message=f"unknown invariant '{name}' (not in registry)",
            )
        try:
            fn(node, context)
        except NodeInvariantError:
            raise
        except Exception as e:
            # Buggy invariant -- surface clearly rather than silently passing
            raise NodeInvariantError(
                node_name=node.name,
                invariant_name=name,
                message=f"invariant raised unexpected {type(e).__name__}: {e}",
            )


# --- extractors ---------------------------------------------------------------
# Walk a node's output (which may be a Pydantic object, a StruckdownResult, or
# a list of either) and pull out Theme / Code instances. Mirrors the pattern
# used in `soak.models.nodes.base._collect_codes_from_dag`.


def extract_themes(output) -> List:
    from struckdown import StruckdownResult

    from soak.models.base import Theme

    if output is None:
        return []
    items = output if isinstance(output, list) else [output]
    themes = []
    for item in items:
        if isinstance(item, Theme):
            themes.append(item)
        elif isinstance(item, StruckdownResult):
            for seg in item.results.values():
                out = seg.output
                if isinstance(out, list):
                    themes.extend(t for t in out if isinstance(t, Theme))
                elif isinstance(out, Theme):
                    themes.append(out)
    return themes


def extract_codes(output) -> List:
    from struckdown import StruckdownResult

    from soak.models.base import Code

    if output is None:
        return []
    items = output if isinstance(output, list) else [output]
    codes = []
    for item in items:
        if isinstance(item, Code):
            codes.append(item)
        elif isinstance(item, StruckdownResult):
            for seg in item.results.values():
                out = seg.output
                if isinstance(out, list):
                    codes.extend(c for c in out if isinstance(c, Code))
                elif isinstance(out, Code):
                    codes.append(out)
    return codes


# --- built-in invariants ------------------------------------------------------


@invariant("min_themes_with_codes")
def _check_min_themes_with_codes(node, context: Dict[str, Any]) -> None:
    """Themes node produced at least `min_themes` themes, each linking >= 1 code.

    Reads `min_themes` from pipeline context (the same value the YAML
    `[[theme{min,max}]]` slot uses). Defaults to 3 if absent.
    """
    themes = extract_themes(node.output)
    min_themes = int(context.get("min_themes", 3))
    if len(themes) < min_themes:
        raise NodeInvariantError(
            node_name=node.name,
            invariant_name="min_themes_with_codes",
            message=(
                f"only {len(themes)} themes generated, "
                f"expected at least {min_themes}"
            ),
        )
    no_codes = [t.name for t in themes if not t.code_hashes]
    if no_codes:
        sample = ", ".join(no_codes[:3])
        more = f" (+{len(no_codes) - 3} more)" if len(no_codes) > 3 else ""
        raise NodeInvariantError(
            node_name=node.name,
            invariant_name="min_themes_with_codes",
            message=(
                f"{len(no_codes)} theme(s) have no linked codes: {sample}{more}"
            ),
        )


@invariant("code_retention_and_quotes")
def _check_code_retention_and_quotes(node, context: Dict[str, Any]) -> None:
    """Map/Reduce node retained >= 10% of input codes, each with >= 1 quote.

    Walks the first input node's output to count input codes. Useful on
    code-consolidation steps where the LLM occasionally collapses too
    aggressively or drops quote provenance entirely.
    """
    out_codes = extract_codes(node.output)

    in_codes: List = []
    if node.inputs:
        # use the first non-`documents` input
        for input_name in node.inputs:
            if input_name == "documents":
                continue
            upstream = node.dag.nodes_dict.get(input_name)
            if upstream is not None and upstream.output is not None:
                in_codes = extract_codes(upstream.output)
                if in_codes:
                    break

    min_ratio = float(context.get("min_code_retention_ratio", 0.10))
    if in_codes:
        ratio = len(out_codes) / max(1, len(in_codes))
        if ratio < min_ratio:
            raise NodeInvariantError(
                node_name=node.name,
                invariant_name="code_retention_and_quotes",
                message=(
                    f"retained only {len(out_codes)}/{len(in_codes)} codes "
                    f"({ratio:.0%}), expected >= {min_ratio:.0%}"
                ),
            )

    no_quotes = [
        c.name for c in out_codes if not (c.quotes or getattr(c, "resolved_quotes", None))
    ]
    if no_quotes:
        sample = ", ".join(no_quotes[:3])
        more = f" (+{len(no_quotes) - 3} more)" if len(no_quotes) > 3 else ""
        raise NodeInvariantError(
            node_name=node.name,
            invariant_name="code_retention_and_quotes",
            message=f"{len(no_quotes)} code(s) have no quotes: {sample}{more}",
        )


@invariant("non_empty_output")
def _check_non_empty_output(node, context: Dict[str, Any]) -> None:
    """Output must not be None or an empty list. Useful as a baseline check."""
    out = node.output
    if out is None:
        raise NodeInvariantError(
            node_name=node.name,
            invariant_name="non_empty_output",
            message="output is None",
        )
    if isinstance(out, list) and len(out) == 0:
        raise NodeInvariantError(
            node_name=node.name,
            invariant_name="non_empty_output",
            message="output is an empty list",
        )
