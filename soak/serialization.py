"""Serialization and deserialization for pipeline node outputs.

Provides a consistent, type-preserving serialization layer used by both CLI and Django.
Objects are tagged with a ``__type__`` key so they can be losslessly round-tripped.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# lazy imports to avoid circular dependency at module load
_TYPE_REGISTRY: Optional[Dict[str, type]] = None


def _get_type_registry() -> Dict[str, type]:
    """Build the type registry on first use (avoids circular imports)."""
    global _TYPE_REGISTRY
    if _TYPE_REGISTRY is not None:
        return _TYPE_REGISTRY

    from struckdown import StruckdownResult
    from struckdown.results import SlotResult

    from soak.models.base import (BatchList, Code, CodeList, Quote,
                                  QuoteReference, Theme, Themes, TrackedItem)

    _TYPE_REGISTRY = {
        "TrackedItem": TrackedItem,
        "StruckdownResult": StruckdownResult,
        "SlotResult": SlotResult,
        "Code": Code,
        "Theme": Theme,
        "CodeList": CodeList,
        "Themes": Themes,
        "BatchList": BatchList,
        "Quote": Quote,
        "QuoteReference": QuoteReference,
    }
    return _TYPE_REGISTRY


# --------------------------------------------------------------------------- #
#  Serialization
# --------------------------------------------------------------------------- #


def serialize_value(obj: Any) -> Any:
    """Recursively serialize a value to a JSON-safe form, preserving type info.

    Pydantic models get a ``__type__`` tag so they can be reconstructed later.
    StruckdownResult is stripped of debug data (prompts/completions) first.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    from struckdown import StruckdownResult
    from struckdown.results import SlotResult

    from soak.models.base import TrackedItem

    # StruckdownResult -- strip debug data, preserve structure
    if isinstance(obj, StruckdownResult):
        stripped = obj.strip_debug_data()
        serialized_results = {}
        for name, seg in stripped.results.items():
            serialized_results[name] = {
                "name": seg.name,
                "action": seg.action,
                "output": serialize_value(seg.output),
            }
        return {
            "__type__": "StruckdownResult",
            "results": serialized_results,
        }

    # SlotResult (standalone -- unusual but handle it)
    if isinstance(obj, SlotResult):
        return {
            "__type__": "SlotResult",
            "name": obj.name,
            "action": obj.action,
            "output": serialize_value(obj.output),
        }

    # TrackedItem -- content may itself be a StruckdownResult
    if isinstance(obj, TrackedItem):
        return {
            "__type__": "TrackedItem",
            "content": serialize_value(obj.content),
            "id": obj.id,
            "sources": obj.sources,
            "metadata": _serialize_metadata(obj.metadata),
            "content_excluding_overlap": obj.content_excluding_overlap,
        }

    # other registered Pydantic models (Code, Theme, Quote, etc.)
    registry = _get_type_registry()
    for type_name, type_cls in registry.items():
        if type_name in ("StruckdownResult", "SlotResult", "TrackedItem"):
            continue  # handled above
        if isinstance(obj, type_cls):
            return {"__type__": type_name, **obj.model_dump(mode="json", serialize_as_any=True)}

    # generic Pydantic model (not in our registry)
    if isinstance(obj, BaseModel):
        return {"__type__": type(obj).__name__, **obj.model_dump(mode="json", serialize_as_any=True)}

    # lists
    if isinstance(obj, (list, tuple)):
        return [serialize_value(item) for item in obj]

    # dicts
    if isinstance(obj, dict):
        return {k: serialize_value(v) for k, v in obj.items()}

    # fallback
    return str(obj)


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata dict for JSON serialization.

    Recurses through values via serialize_value so registered Pydantic models
    (Code, Theme, Quote, ...) keep their ``__type__`` envelope and can be
    rebuilt on restart. Without this, e.g. Cluster's ``metadata["items"]``
    (a list of Code objects) would be stringified and quote/code provenance
    would be lost across checkpoints.
    """
    import json as _json

    clean = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            clean[k] = v
            continue
        serialized = serialize_value(v)
        try:
            _json.dumps(serialized)
            clean[k] = serialized
        except (TypeError, ValueError):
            clean[k] = str(v)
    return clean


# --------------------------------------------------------------------------- #
#  Deserialization
# --------------------------------------------------------------------------- #


def deserialize_value(data: Any) -> Any:
    """Recursively deserialize a value, reconstructing typed objects from ``__type__`` tags.

    Plain dicts without ``__type__`` are returned as-is (handles legacy data).
    """
    if data is None or isinstance(data, (str, int, float, bool)):
        return data

    if isinstance(data, dict):
        type_name = data.get("__type__")

        if type_name == "StruckdownResult":
            return _deserialize_complete_result(data)

        if type_name == "SlotResult":
            return _deserialize_segment_result(data)

        if type_name == "TrackedItem":
            return _deserialize_tracked_item(data)

        if type_name is not None:
            return _deserialize_registered_type(type_name, data)

        # no __type__ tag -- check for natural StruckdownResult discriminator
        if data.get("type") == "complete" and "results" in data:
            return _deserialize_complete_result_natural(data)

        # plain dict (legacy data or non-typed) -- recurse on values
        return {k: deserialize_value(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        return [deserialize_value(item) for item in data]

    return data


def _deserialize_complete_result(data: dict) -> Any:
    """Reconstruct StruckdownResult from our envelope format."""
    from struckdown import StruckdownResult
    from struckdown.results import SlotResult

    results = {}
    for name, seg_data in data.get("results", {}).items():
        output = deserialize_value(seg_data.get("output"))
        results[name] = SlotResult(
            name=seg_data.get("name", name),
            action=seg_data.get("action"),
            output=output,
            prompt="",
        )
    return StruckdownResult(results=results)


def _deserialize_complete_result_natural(data: dict) -> Any:
    """Reconstruct StruckdownResult from Pydantic's native model_dump format.

    This handles the case where StruckdownResult was serialized via
    model_dump(mode="json") without our envelope.
    """
    from struckdown import StruckdownResult

    try:
        cr = StruckdownResult.model_validate(data)
        # recursively deserialize segment outputs
        for seg in cr.results.values():
            seg.output = deserialize_value(seg.output)
        return cr
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Failed to reconstruct StruckdownResult from natural format: {e}"
        )
        return data


def _deserialize_segment_result(data: dict) -> Any:
    """Reconstruct a standalone SlotResult."""
    from struckdown.results import SlotResult

    output = deserialize_value(data.get("output"))
    return SlotResult(
        name=data.get("name"),
        action=data.get("action"),
        output=output,
        prompt="",
    )


def _deserialize_tracked_item(data: dict) -> Any:
    """Reconstruct TrackedItem with recursive content + metadata deserialization."""
    from soak.models.base import TrackedItem

    content = deserialize_value(data.get("content"))
    raw_metadata = data.get("metadata", {})
    metadata = {k: deserialize_value(v) for k, v in raw_metadata.items()}
    return TrackedItem(
        content=content,
        id=data["id"],
        sources=data.get("sources", []),
        metadata=metadata,
        content_excluding_overlap=data.get("content_excluding_overlap"),
    )


def _deserialize_registered_type(type_name: str, data: dict) -> Any:
    """Reconstruct a registered Pydantic model from its __type__ tag."""
    registry = _get_type_registry()
    type_cls = registry.get(type_name)

    if type_cls is None:
        logger.debug(f"Unknown __type__ '{type_name}', returning raw dict")
        return {k: v for k, v in data.items() if k != "__type__"}

    # strip __type__ before passing to model_validate
    clean_data = {k: v for k, v in data.items() if k != "__type__"}
    try:
        return type_cls.model_validate(clean_data)
    except Exception as e:
        logger.warning(f"Failed to reconstruct {type_name}: {e}")
        return clean_data


# --------------------------------------------------------------------------- #
#  Node-level serialization (replaces django_soak's serialise_node_output)
# --------------------------------------------------------------------------- #


def serialize_node_output(node) -> Any:
    """Serialize a DAG node's output for storage.

    Returns JSON-safe data with ``__type__`` tags for type reconstruction.
    """
    from struckdown import StruckdownResult

    if node.output is None:
        return None

    # Split/Scrub: serialize the full TrackedItem list (not just metadata)
    if node.type in ("Split", "Scrub"):
        return serialize_value(node.output)

    # Transform/TransformReduce: unwrap single-element list
    if node.type in ("Transform", "TransformReduce"):
        output = node.output
        if isinstance(output, list) and len(output) == 1:
            output = output[0]
        return serialize_value(output)

    # Map: list of StruckdownResults
    if node.type == "Map":
        # for Map nodes, flatten single-slot StruckdownResults for compactness
        results = []
        for item in node.output:
            if isinstance(item, StruckdownResult) and len(item.results) == 1:
                seg = next(iter(item.results.values()))
                results.append(serialize_value(seg.output))
            else:
                results.append(serialize_value(item))
        return results

    # everything else
    return serialize_value(node.output)


def deserialize_node_output(data: Any, node_type: str = "") -> Any:
    """Deserialize node output from storage back to typed objects.

    Handles both new format (with ``__type__`` tags) and legacy format
    (plain dicts/lists without tags).
    """
    if data is None:
        return data

    return deserialize_value(data)


# --------------------------------------------------------------------------- #
#  Display extraction (backward compat for Django consumers)
# --------------------------------------------------------------------------- #


def extract_display_output(data: Any) -> Any:
    """Convert envelope format to the flat format existing consumers expect.

    StruckdownResult envelopes are extracted to their slot outputs.
    ``__type__`` keys are left in dicts (harmless -- ignored by templates).
    Strings and plain dicts pass through unchanged.
    """
    if data is None:
        return data

    if isinstance(data, dict):
        type_name = data.get("__type__")

        if type_name == "StruckdownResult":
            results = data.get("results", {})
            if len(results) == 1:
                # single slot -- return just the output
                seg = next(iter(results.values()))
                return _extract_slot_output(seg)
            # multiple slots -- return {slot_name: output}
            return {name: _extract_slot_output(seg) for name, seg in results.items()}

        if type_name == "TrackedItem":
            # for display, return a plain dict; recurse on content first
            content = data.get("content", "")
            if isinstance(content, dict):
                content = extract_display_output(content)
            return {
                "id": data.get("id", ""),
                "content": str(content) if not isinstance(content, str) else content,
                "sources": data.get("sources", []),
                "metadata": data.get("metadata", {}),
            }

        # other __type__ dicts -- pass through as-is
        # __type__ key is harmless (ignored by Django templates and views)
        return data

    if isinstance(data, list):
        return [extract_display_output(item) for item in data]

    return data


def _extract_slot_output(seg_data: dict) -> Any:
    """Extract the output value from a serialized SlotResult."""
    output = seg_data.get("output")
    if isinstance(output, list):
        return [extract_display_output(item) for item in output]
    if isinstance(output, dict):
        return extract_display_output(output)
    return output
