import hashlib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def format_exception_concise(exc: Exception) -> str:
    """Format exception with minimal context - just the error and relevant code location."""
    tb = traceback.extract_tb(exc.__traceback__)

    # Get the last frame (where the error actually occurred)
    if tb:
        last_frame = tb[-1]
        error_msg = f"\n{type(exc).__name__}: {exc}\n"
        error_msg += f"  File: {last_frame.filename}:{last_frame.lineno}\n"
        error_msg += f"  In: {last_frame.name}\n"
        if last_frame.line:
            error_msg += f"    {last_frame.line}\n"
    else:
        error_msg = f"\n{type(exc).__name__}: {exc}\n"

    return error_msg


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load environment variables from a .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    return env_vars


def save_env_file(env_path: Path, env_vars: dict[str, str]):
    """Save environment variables to a .env file."""
    with open(env_path, "w") as f:
        for key, value in env_vars.items():
            f.write(f'{key}="{value}"\n')


def resolve_pipeline(pipeline: str, localdir: Path, pipelinedir: Path) -> Path:
    candidates = [
        localdir / pipeline,
        localdir / f"{pipeline}.yaml",
        localdir / f"{pipeline}.yml",
        pipelinedir / f"{pipeline}",
        pipelinedir / f"{pipeline}.yaml",
        pipelinedir / f"{pipeline}.yml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Pipeline file not found. Tried: {candidates}")


def hash_run_config(
    input_files: list[str],
    model_name: str | None = None,
    context: list[str] | None = None,
    template: str | None = None,
    length: int = 4,
) -> str:
    """Generate a short hash from run configuration.

    Args:
        input_files: List of input file paths
        model_name: Model name if specified
        context: Context overrides if specified
        template: Template name if specified
        length: Length of hash to return (default: 4)

    Returns:
        Short hash string of specified length (hex chars only - always safe)
    """
    # Build configuration string from all parameters
    parts = [
        "files:" + "|".join(sorted(input_files)),
    ]
    if model_name:
        parts.append(f"model:{model_name}")
    if context:
        parts.append("context:" + "|".join(sorted(context)))
    if template:
        parts.append(f"template:{template}")

    config_str = "||".join(parts)
    hash_obj = hashlib.sha256(config_str.encode("utf-8"))
    return hash_obj.hexdigest()[:length]


def sanitize_for_filename(text: str) -> str:
    """Remove or replace characters that are problematic in filenames.

    Args:
        text: Input string

    Returns:
        Sanitized string safe for use in filenames
    """
    # Replace problematic characters with underscores
    dangerous_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]
    result = text
    for char in dangerous_chars:
        result = result.replace(char, "_")
    return result


def build_combined_long_form_dataset(
    model_results: Dict[str, List[Any]],
    processed_items: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """Build a long-form dataset combining results from multiple models.

    Creates a dataset where each row represents one model's response to one slot
    for one input item, enabling easy comparison across models.

    Args:
        model_results: Dict mapping model_name -> list of result items
        processed_items: Optional list of input items for metadata extraction

    Returns:
        DataFrame with columns: model_name, slot_name, filename, document,
        item_id, index, slot_response_type, slot_options, response, and other metadata
    """
    from .models import TrackedItem  # Avoid circular import

    combined_rows = []

    for model_name, results in model_results.items():
        for idx, output_item in enumerate(results):
            # Extract base metadata
            if processed_items and idx < len(processed_items):
                base_metadata = TrackedItem.extract_export_metadata(
                    processed_items[idx], idx
                )
            else:
                base_metadata = {
                    "item_id": f"item_{idx}",
                    "document": f"item_{idx}",
                    "index": idx,
                }

            # Extract slot-level responses
            if hasattr(output_item, "results") and output_item.results:
                # Use results dict for detailed slot information
                for slot_name, segment_result in output_item.results.items():
                    row = {
                        **base_metadata,
                        "model_name": model_name,
                        "slot_name": slot_name,
                        "slot_response_type": (
                            segment_result.action
                            if hasattr(segment_result, "action")
                            else None
                        ),
                        "slot_options": (
                            ",".join(segment_result.options)
                            if hasattr(segment_result, "options")
                            and segment_result.options
                            else None
                        ),
                        "response": (
                            str(segment_result.output)
                            if hasattr(segment_result, "output")
                            and segment_result.output is not None
                            else (
                                str(getattr(output_item.outputs, slot_name))
                                if hasattr(output_item.outputs, slot_name)
                                else None
                            )
                        ),
                    }
                    combined_rows.append(row)
            elif hasattr(output_item, "outputs"):
                # Fallback: use outputs dict if results not available
                output_dict = (
                    output_item.outputs if hasattr(output_item.outputs, "items") else {}
                )
                for slot_name, response_value in output_dict.items():
                    if not slot_name.startswith("__"):
                        row = {
                            **base_metadata,
                            "model_name": model_name,
                            "slot_name": slot_name,
                            "slot_response_type": None,
                            "slot_options": None,
                            "response": (
                                str(response_value)
                                if response_value is not None
                                else None
                            ),
                        }
                        combined_rows.append(row)
            elif isinstance(output_item, dict):
                # Plain dict from JSON deserialization
                for slot_name, response_value in output_item.items():
                    if not slot_name.startswith("__"):
                        row = {
                            **base_metadata,
                            "model_name": model_name,
                            "slot_name": slot_name,
                            "slot_response_type": None,
                            "slot_options": None,
                            "response": (
                                str(response_value)
                                if response_value is not None
                                else None
                            ),
                        }
                        combined_rows.append(row)

    if not combined_rows:
        return pd.DataFrame()

    df = pd.DataFrame(combined_rows)

    # Reorder columns for readability
    col_order = ["model_name", "slot_name"]
    for col in ["filename", "document", "item_id", "index"]:
        if col in df.columns:
            col_order.append(col)
    col_order.extend(["slot_response_type", "slot_options", "response"])
    remaining_cols = [c for c in df.columns if c not in col_order]
    col_order.extend(remaining_cols)
    df = df[[c for c in col_order if c in df.columns]]

    # Sort by filename, document identifiers, slot_name, model_name
    sort_cols = [
        c
        for c in ["filename", "document", "item_id", "index", "slot_name", "model_name"]
        if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    return df
