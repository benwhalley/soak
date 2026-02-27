import hashlib
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def format_exception_concise(exc: Exception) -> str:
    """Format exception with minimal context.

    Returns:
        Error message with exception type, message, file location, and code line
    """
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
    """Load environment variables from .env file.

    Returns:
        Dict of key-value pairs. Empty dict if file missing.
    """
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


def save_env_file(env_path: Path, env_vars: dict[str, str]) -> None:
    """Save environment variables to .env file with quoted values."""
    with open(env_path, "w") as f:
        for key, value in env_vars.items():
            # Strip existing quotes to avoid double-quoting
            value = str(value).strip('"').strip("'")
            f.write(f'{key}="{value}"\n')


def resolve_pipeline(pipeline: str, localdir: Path, pipelinedir: Path) -> Path:
    """Resolve pipeline name to file path.

    Searches localdir first, then builtin pipelinedir.
    Tries with/without .soak extension, including subfolders.

    Raises:
        FileNotFoundError: If pipeline file not found in any location
    """
    candidates = [
        localdir / pipeline,
        localdir / f"{pipeline}.soak",
        pipelinedir / f"{pipeline}",
        pipelinedir / f"{pipeline}.soak",
    ]
    # also search subfolders (e.g., "zs" finds "thematic_analysis/zs.soak")
    candidates.extend(pipelinedir.glob(f"**/{pipeline}"))
    candidates.extend(pipelinedir.glob(f"**/{pipeline}.soak"))

    for cand in candidates:
        if isinstance(cand, Path) and cand.is_file():
            return cand
    raise FileNotFoundError(f"Pipeline file not found. Tried: {candidates}")


def hash_run_config(
    input_files: list[str | Path],
    model_name: str | None = None,
    context: list[str] | None = None,
    template: str | None = None,
    length: int = 4,
) -> str:
    """Generate a short hash from run configuration.

    Args:
        input_files: List of input file paths (str or Path)
        model_name: Model name if specified
        context: Context overrides if specified
        template: Template name if specified
        length: Length of hash to return (default: 4)

    Returns:
        Short hash string of specified length (hex chars only - always safe)
    """
    # Convert paths to strings for consistent handling
    input_files_str = [str(f) for f in input_files]

    # Build configuration string from all parameters
    parts = [
        "files:" + "|".join(sorted(input_files_str)),
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


def derive_input_source(docfiles: list) -> str:
    """Derive a summary string from document paths.

    Attempts to find common directory and file extension pattern.
    E.g., [("data/a.txt", {}), ("data/b.txt", {})] -> "data/*.txt"

    Args:
        docfiles: List of paths or (path, metadata) tuples

    Returns:
        Summary string like "data/*.txt"
    """
    if not docfiles:
        return ""

    # extract paths from tuples
    paths = [Path(p[0]) if isinstance(p, tuple) else Path(p) for p in docfiles]

    # find common parent directory
    parents = [p.parent for p in paths]
    if parents and all(p == parents[0] for p in parents):
        common_dir = str(parents[0])
    else:
        # try to find longest common prefix
        parent_strs = [str(p) for p in parents]
        common_prefix = os.path.commonpath(parent_strs) if parent_strs else ""
        common_dir = common_prefix if common_prefix else "."

    # find common extension
    extensions = {p.suffix for p in paths}
    if len(extensions) == 1 and extensions != {""}:
        ext_pattern = f"*{list(extensions)[0]}"
    else:
        ext_pattern = "*"

    return f"{common_dir}/{ext_pattern}"


def print_comparison_stats(
    result: dict,
    name_a: str,
    name_b: str,
    list_a: list,
    list_b: list,
    threshold: float,
    embedding_model: str,
    shepard_k: float,
    ot_k_values: list,
    similarity: str = "angular",
) -> str:
    """Generate text output for comparison statistics.

    Args:
        result: Comparison result dict with statistics
        name_a: Name of first analysis
        name_b: Name of second analysis
        list_a: List of theme names from first analysis
        list_b: List of theme names from second analysis
        threshold: Similarity threshold used
        embedding_model: Embedding model used
        shepard_k: Shepard k parameter
        ot_k_values: List of OT k values to display
        similarity: Similarity metric name

    Returns:
        Formatted text string with comparison statistics
    """
    sim_metric = result.get("similarity_metric", similarity)

    lines = []

    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Comparison: {name_a} vs {name_b}")
    lines.append("=" * 70)
    lines.append(
        f"Similarity: {sim_metric}  |  Threshold: {threshold}  |  Shepard k: {shepard_k}"
    )
    lines.append(f"Embedding: {embedding_model}")
    lines.append(f"Items: {len(list_a)} ({name_a}) x {len(list_b)} ({name_b})")

    lines.append("")
    lines.append("-" * 70)
    lines.append("COVERAGE (Hit Rates)")
    lines.append("-" * 70)
    lines.append(
        f"  Hit Rate {name_a}: {result['hit_rate_a']:.1%}  (items with >=1 match above threshold)"
    )
    lines.append(f"  Hit Rate {name_b}: {result['hit_rate_b']:.1%}")
    lines.append(
        f"  Jaccard:         {result['jaccard']:.3f}  (proportion of pairs above threshold)"
    )

    lines.append("")
    lines.append("-" * 70)
    lines.append("FIDELITY (Mean Best-Match Similarity)")
    lines.append("-" * 70)
    lines.append(f"  {name_a} -> {name_b}: {result['mean_max_sim_a_to_b']:.3f}")
    lines.append(f"  {name_b} -> {name_a}: {result['mean_max_sim_b_to_a']:.3f}")
    lines.append(f"  Fidelity:        {result['fidelity']:.3f}  (harmonic mean)")

    lines.append("")
    lines.append("-" * 70)
    lines.append("HUNGARIAN MATCHING (Optimal 1-to-1 Assignment)")
    lines.append("-" * 70)
    hungarian = result.get("hungarian", {})
    thresh_metrics = hungarian.get("thresholded_metrics", {})
    lines.append(f"  Coverage {name_a}: {thresh_metrics.get('coverage_a', 0):.1%}")
    lines.append(f"  Coverage {name_b}: {thresh_metrics.get('coverage_b', 0):.1%}")
    lines.append(
        f"  1-to-1 Jaccard:  {thresh_metrics.get('true_jaccard', 0):.3f}  "
        "(matched pairs / total unique items)"
    )

    lines.append("")
    lines.append("-" * 70)
    lines.append("OPTIMAL TRANSPORT (Many-to-Many Alignment)")
    lines.append("-" * 70)

    ot_by_k = result.get("ot_by_k", {})
    default_k = result.get("default_k", 0.25)
    elbow_k = result.get("elbow_k")

    if default_k in ot_by_k:
        default_ot = ot_by_k[default_k]["ot"]
        ceiling_sim = default_ot.get("paraphrase_upper_bound")
        ceiling_cost = default_ot.get("paraphrase_cost_lower_bound")
        floor_sim = default_ot.get("null_shared_mass_mean")
        floor_cost = default_ot.get("null_avg_cost_mean")

        if ceiling_sim is not None or floor_sim is not None:
            lines.append("  Baselines:")
            if ceiling_sim is not None:
                ceiling_align = 1 - (ceiling_cost or 0)
                lines.append(
                    f"    Paraphrase ceiling:   {ceiling_sim:.1%} shared mass, {ceiling_align:.1%} alignment"
                )
            if floor_sim is not None:
                floor_align = 1 - (floor_cost or 0)
                lines.append(
                    f"    Word-salad floor:     {floor_sim:.1%} shared mass, {floor_align:.1%} alignment"
                )

    if ot_k_values:
        display_k_values = ot_k_values
    else:
        display_k_values = sorted(set([default_k] + ([elbow_k] if elbow_k else [])))

    for k_val in display_k_values:
        if k_val in ot_by_k:
            ot_data = ot_by_k[k_val]["ot"]
            marker = ""
            if k_val == default_k:
                marker = " (default)"
            if k_val == elbow_k:
                marker = " (knee/diminishing returns)"
            if k_val == default_k == elbow_k:
                marker = " (default, knee)"

            lines.append("")
            lines.append(f"  K = {k_val}{marker}")

            shared_mass = ot_data.get("shared_mass", 0)
            null_shared_mass = ot_data.get("null_shared_mass_mean")
            paraphrase_ceiling = ot_data.get("paraphrase_upper_bound")

            if paraphrase_ceiling is not None and null_shared_mass is not None:
                lines.append(
                    f"    Shared Mass:          {shared_mass:.1%}  (floor: {null_shared_mass:.1%}, ceiling: {paraphrase_ceiling:.1%})"
                )
            else:
                lines.append(f"    Shared Mass:          {shared_mass:.1%}")

            pct_ceiling = ot_data.get("shared_mass_pct_of_ceiling")
            pct_improvement = ot_data.get("shared_mass_improvement_vs_null")
            if pct_ceiling is not None:
                lines.append(
                    f"      % of ceiling:       {pct_ceiling:.0%}  (vs paraphrase upper bound)"
                )
            if pct_improvement is not None:
                lines.append(
                    f"      vs word-salad:      {pct_improvement:.0%} better, relative to ceiling"
                )

            avg_cost = ot_data.get("avg_cost", 0)
            alignment = 1 - avg_cost
            alignment_ceiling = ot_data.get("alignment_paraphrase_ceiling")
            alignment_floor = ot_data.get("alignment_null_floor")

            if alignment_ceiling is not None and alignment_floor is not None:
                lines.append(
                    f"    Semantic Alignment:   {alignment:.1%}  (floor: {alignment_floor:.1%}, ceiling: {alignment_ceiling:.1%})"
                )
            else:
                lines.append(
                    f"    Semantic Alignment:   {alignment:.1%}  (quality of matches)"
                )

            align_pct_ceiling = ot_data.get("alignment_pct_of_ceiling")
            align_improvement = ot_data.get("alignment_improvement_vs_null")
            if align_pct_ceiling is not None:
                lines.append(
                    f"      % of ceiling:       {align_pct_ceiling:.0%}  (vs paraphrase upper bound)"
                )
            if align_improvement is not None:
                lines.append(
                    f"      vs word-salad:      {align_improvement:.0%} better, relative to ceiling"
                )

    if elbow_k:
        lines.append("")
        lines.append(f"  Knee detected at K = {elbow_k} (point of diminishing returns)")

    lines.append("")
    lines.append("-" * 70)
    lines.append(
        f"TRANSPORT FLOWS (cost labels: low <K, med K-2K, high >2K where K={default_k})"
    )
    lines.append("-" * 70)

    if default_k in ot_by_k:
        transport_plan = np.array(ot_by_k[default_k]["ot"]["transport_plan"])
        selected_sim = result.get(
            "selected_similarity_matrix", result.get("angle_similarity_matrix", [])
        )
        cost_matrix = 1 - np.array(selected_sim)
        total_mass = transport_plan.sum()

        flow_threshold = 0.01 * transport_plan.max() if transport_plan.max() > 0 else 0
        k_threshold = default_k

        for i, item_a in enumerate(list_a):
            flows = []
            for j, item_b in enumerate(list_b):
                flow = transport_plan[i, j]
                if flow > flow_threshold:
                    cost = cost_matrix[i, j] if len(cost_matrix) > 0 else 0
                    pct = (flow / total_mass * 100) if total_mass > 0 else 0
                    if cost < k_threshold:
                        cost_label = "low"
                    elif cost < 2 * k_threshold:
                        cost_label = "med"
                    else:
                        cost_label = "high"
                    flows.append((item_b, pct, cost_label))

            item_a_display = (
                item_a[:25] + "..." if len(str(item_a)) > 28 else str(item_a)
            )

            if flows:
                flows.sort(key=lambda x: -x[1])
                flow_strs = [
                    f"{b[:20]}({cost},{pct:.0f}%)" for b, pct, cost in flows[:3]
                ]
                extra = f" +{len(flows)-3} more" if len(flows) > 3 else ""
                lines.append(
                    f"  {item_a_display:28s} --> {', '.join(flow_strs)}{extra}"
                )
            else:
                lines.append(f"  {item_a_display:28s} --> (unmatched)")

    lines.append("")
    lines.append("-" * 70)
    lines.append("TRANSPORT MASS MATRIX (%)")
    lines.append("-" * 70)

    if default_k in ot_by_k:
        transport_plan = np.array(ot_by_k[default_k]["ot"]["transport_plan"])
        total_mass = transport_plan.sum()
        if total_mass > 0:
            pct_matrix = transport_plan / total_mass * 100
        else:
            pct_matrix = transport_plan

        _print_compact_matrix(
            lines,
            pct_matrix,
            list_a,
            list_b,
            fmt="{:5.1f}",
            name_a=name_a,
            name_b=name_b,
        )

    lines.append("")
    lines.append("-" * 70)
    lines.append(f"SIMILARITY MATRIX ({sim_metric.upper()} -- used for all metrics)")
    lines.append("-" * 70)

    if "selected_similarity_matrix" in result:
        _print_compact_matrix(
            lines,
            np.array(result["selected_similarity_matrix"]),
            list_a,
            list_b,
            fmt="{:5.2f}",
            name_a=name_a,
            name_b=name_b,
        )
    elif "angle_similarity_matrix" in result:
        _print_compact_matrix(
            lines,
            np.array(result["angle_similarity_matrix"]),
            list_a,
            list_b,
            fmt="{:5.2f}",
            name_a=name_a,
            name_b=name_b,
        )

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def _print_compact_matrix(
    lines: list,
    matrix,
    list_a: list,
    list_b: list,
    fmt: str = "{:5.2f}",
    name_a: str = "A",
    name_b: str = "B",
    max_label_len: int = 12,
):
    """Append compact matrix representation to lines list.

    Args:
        lines: List to append formatted lines to
        matrix: 2D numpy array
        list_a: Row labels
        list_b: Column labels
        fmt: Format string for values
        name_a: Name for row dimension
        name_b: Name for column dimension
        max_label_len: Maximum label length before truncation
    """
    n_a, n_b = matrix.shape

    def trunc(s, max_len):
        s = str(s)
        return s[: max_len - 1] + "..." if len(s) > max_len else s

    labels_a = [trunc(s, max_label_len) for s in list_a]
    labels_b = [trunc(s, max_label_len) for s in list_b]

    if n_a > 15 or n_b > 10:
        lines.append(f"  (Matrix {n_a}x{n_b} -- showing top 5 pairs by value)")
        flat_indices = np.argsort(matrix.ravel())[::-1][:5]
        for idx in flat_indices:
            i, j = np.unravel_index(idx, matrix.shape)
            lines.append(
                f"    {labels_a[i]:12s} <-> {labels_b[j]:12s}: {fmt.format(matrix[i, j])}"
            )
        return

    header = " " * (max_label_len + 2) + "|"
    for label in labels_b:
        header += f" {label:>{max_label_len}}"
    lines.append(f"  {header}")

    sep = "-" * (max_label_len + 1) + "+" + "-" * (len(labels_b) * (max_label_len + 1))
    lines.append(f"  {sep}")

    for i, label_a in enumerate(labels_a):
        row = f"{label_a:>{max_label_len}} |"
        for j in range(n_b):
            val = fmt.format(matrix[i, j])
            row += f" {val:>{max_label_len}}"
        lines.append(f"  {row}")
