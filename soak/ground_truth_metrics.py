"""Ground truth comparison metrics for classification results."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def _extract_wildcard_config(
    mapping: Dict[str, Any],
) -> tuple[bool, str, Dict[str, Any]]:
    """Extract wildcard configuration from mapping dict.

    Args:
        mapping: Mapping dictionary that may contain "*" key

    Returns:
        Tuple of (has_wildcard, wildcard_target, explicit_mappings)
        - has_wildcard: Whether "*" key exists
        - wildcard_target: What to map wildcard values to (e.g., "other")
        - explicit_mappings: Dict with "*" key removed
    """
    if "*" not in mapping:
        return False, None, mapping

    wildcard_value = mapping["*"]

    # If wildcard maps to "*", use "other" as the target
    if wildcard_value == "*":
        wildcard_target = "other"
    else:
        wildcard_target = str(wildcard_value)

    # Build explicit mappings (exclude wildcard)
    explicit_mappings = {k: v for k, v in mapping.items() if k != "*"}

    return True, wildcard_target, explicit_mappings


def apply_mapping(
    values: Union[pd.Series, List[Any]],
    mapping: Optional[Dict[str, Any]] = None,
    apply_wildcard_to_targets: bool = False,
) -> List[Any]:
    """Apply value mapping to convert categorical values.

    Args:
        values: Series or list of values to map
        mapping: Dict mapping original values to new values. If None, returns values as-is
        apply_wildcard_to_targets: If True, also map values that match wildcard targets

    Returns:
        List of mapped values (converted to strings for consistency)
    """
    if mapping is None:
        return list(values) if isinstance(values, pd.Series) else values

    # Extract wildcard configuration
    has_wildcard, wildcard_target, explicit_mapping = _extract_wildcard_config(mapping)

    # Normalize mapping keys: convert boolean keys to their string equivalents
    # This handles YAML parsing yes/no as True/False
    normalized_mapping = {}
    for k, v in explicit_mapping.items():
        if isinstance(k, bool):
            # Map both the boolean and its string representation
            normalized_mapping[k] = v
            normalized_mapping[str(k).lower()] = v  # True -> "true", False -> "false"
            # Also map common boolean string representations
            if k is True:
                normalized_mapping["yes"] = v
                normalized_mapping["Yes"] = v
                normalized_mapping["YES"] = v
                normalized_mapping["true"] = v
                normalized_mapping["True"] = v
                normalized_mapping["TRUE"] = v
            else:  # False
                normalized_mapping["no"] = v
                normalized_mapping["No"] = v
                normalized_mapping["NO"] = v
                normalized_mapping["false"] = v
                normalized_mapping["False"] = v
                normalized_mapping["FALSE"] = v
        else:
            normalized_mapping[k] = v

    # Get explicit target values (for wildcard target matching)
    explicit_targets = set(str(v) for v in normalized_mapping.values())

    result = []
    values_list = list(values) if isinstance(values, pd.Series) else values

    for v in values_list:
        # First try explicit mapping
        if v in normalized_mapping:
            mapped_value = normalized_mapping[v]
        elif has_wildcard:
            # If value not explicitly mapped and wildcard exists
            # Check if we should map this value to wildcard target
            if apply_wildcard_to_targets:
                # For ground truth: map values not in explicit targets to wildcard
                v_str = (
                    str(v)
                    if not (isinstance(v, float) and v.is_integer())
                    else str(int(v))
                )
                if v_str not in explicit_targets:
                    mapped_value = wildcard_target
                else:
                    mapped_value = v
            else:
                # For predictions: map unmapped values to wildcard target
                mapped_value = wildcard_target
        else:
            # No mapping found, keep original
            mapped_value = v

        # Convert to string, handling floats like we do for ground truth
        if isinstance(mapped_value, float) and mapped_value.is_integer():
            mapped_value = str(int(mapped_value))
        else:
            mapped_value = str(mapped_value)

        result.append(mapped_value)

    return result


def extract_ground_truth_from_metadata(
    processed_items: List[Any],
    column_name: str,
    mapping: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Extract ground truth values from TrackedItem metadata.

    Args:
        processed_items: List of TrackedItem objects with metadata
        column_name: Name of metadata column containing ground truth
        mapping: Optional mapping to apply to ground truth values

    Returns:
        List of ground truth values (mapped if mapping provided)
    """
    from .models.base import TrackedItem

    ground_truth = []
    for item in processed_items:
        # Extract metadata from TrackedItem
        if hasattr(item, "metadata") and isinstance(item.metadata, dict):
            value = item.metadata.get(column_name)
        elif isinstance(item, dict):
            value = item.get(column_name)
        else:
            value = None

        # Convert to string for consistent comparison
        # Check for NaN/None BEFORE converting to string to avoid "nan" string
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = "NA"  # Use R-style NA for compatibility with Excel/R/Python
        elif isinstance(value, float) and value.is_integer():
            value = str(int(value))  # 1.0 -> "1" not "1.0"
        else:
            value = str(value)

        ground_truth.append(value)

    # Apply mapping if provided
    # Use apply_wildcard_to_targets=True for ground truth to map unmapped GT values to wildcard
    if mapping:
        ground_truth = apply_mapping(
            ground_truth, mapping, apply_wildcard_to_targets=True
        )

    return ground_truth


def calculate_classification_metrics(
    y_true: List[Any],
    y_pred: List[Any],
    field_name: str,
    model_name: str = "model",
    labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Calculate classification metrics comparing predictions to ground truth.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        field_name: Name of the classification field
        model_name: Name of the model making predictions
        labels: Optional explicit list of labels for confusion matrix ordering

    Returns:
        Dict containing:
        - precision_macro, precision_micro, precision_weighted
        - recall_macro, recall_micro, recall_weighted
        - f1_macro, f1_micro, f1_weighted
        - confusion_matrix
        - classification_report (detailed per-class metrics)
        - n_samples
        - labels (unique classes)
    """
    # Convert to lists and handle None/NaN values
    # We treat missing values as a category rather than filtering them out
    y_true_clean = []
    y_pred_clean = []

    for true_val, pred_val in zip(y_true, y_pred):
        # Convert None and NaN to "NA" category (R-style)
        if true_val is None or (isinstance(true_val, float) and np.isnan(true_val)):
            true_val = "NA"
        else:
            true_val = str(true_val)

        if pred_val is None or (isinstance(pred_val, float) and np.isnan(pred_val)):
            pred_val = "NA"
        else:
            pred_val = str(pred_val)

        y_true_clean.append(true_val)
        y_pred_clean.append(pred_val)

    if not y_true_clean:
        logger.warning(f"No data for field '{field_name}' in model '{model_name}'")
        return {
            "field": field_name,
            "model": model_name,
            "n_samples": 0,
            "error": "No data",
        }

    # Log unique values for debugging
    unique_true = sorted(set(y_true_clean))
    unique_pred = sorted(set(y_pred_clean))

    logger.info(
        f"Ground truth metrics for {field_name}/{model_name}:\n"
        f"  Unique ground truth values: {unique_true}\n"
        f"  Unique predicted values: {unique_pred}\n"
        f"  Sample size: {len(y_true_clean)}"
    )

    # Check for unpredicted ground truth categories (WARNING only)
    unpredicted_gt = set(unique_true) - set(unique_pred)

    if unpredicted_gt:
        logger.warning(
            f"Ground truth categories never predicted in '{field_name}/{model_name}':\n"
            f"  Categories: {sorted(unpredicted_gt)}\n"
            f"  These will appear in confusion matrix with zero prediction counts.\n"
            f"  To collapse these to 'other' category, add wildcard to mapping:\n"
            f'    "*": "*"  or  "*": other'
        )

    # Determine unique labels
    if labels is None:
        labels = sorted(set(y_true_clean) | set(y_pred_clean))

    n_classes = len(labels)
    n_samples = len(y_true_clean)

    logger.debug(
        f"Calculating metrics for {field_name}/{model_name}: {n_samples} samples, {n_classes} classes"
    )

    try:
        # Use sklearn's classification_report to get all metrics in one call
        report = classification_report(
            y_true_clean,
            y_pred_clean,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        # Confusion matrix
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)

        # Build metrics dict extracting from classification report
        metrics = {
            "field": field_name,
            "model": model_name,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "labels": labels,
            # Weighted averages only
            "precision": round(report["weighted avg"]["precision"], 4),
            "recall": round(report["weighted avg"]["recall"], 4),
            "f1": round(report["weighted avg"]["f1-score"], 4),
            # Overall accuracy
            "accuracy": round(report["accuracy"], 4),
            # Full report and confusion matrix
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        return metrics

    except Exception as e:
        logger.error(
            f"Error calculating metrics for {field_name}/{model_name}: {e}",
            exc_info=True,
        )
        return {
            "field": field_name,
            "model": model_name,
            "n_samples": n_samples,
            "error": str(e),
        }


def calculate_ground_truth_metrics(
    model_dfs: Dict[str, pd.DataFrame],
    processed_items: List[Any],
    ground_truth_config: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Calculate ground truth metrics for all models and fields.

    Args:
        model_dfs: Dict mapping model_name -> DataFrame with classifications
        processed_items: List of TrackedItem objects with metadata
        ground_truth_config: Dict mapping field_name -> {column: str, mapping: dict}

    Returns:
        Nested dict: {field_name: {model_name: metrics_dict}}
    """
    results = {}

    for field_name, config in ground_truth_config.items():
        # Support both 'column' and 'existing' as key names
        column_name = config.get("column") or config.get("existing")
        mapping = config.get("mapping")
        drop_values = config.get("drop", [])  # Values to exclude from metrics

        if not column_name:
            logger.warning(
                f"No column specified for ground truth field '{field_name}'. "
                f"Use 'column' or 'existing' key in config. Config keys: {list(config.keys())}"
            )
            continue

        # Extract ground truth from metadata
        try:
            y_true = extract_ground_truth_from_metadata(
                processed_items, column_name, mapping=mapping
            )
        except Exception as e:
            logger.error(f"Error extracting ground truth for {field_name}: {e}")
            continue

        # Calculate metrics for each model
        field_results = {}
        for model_name, df in model_dfs.items():
            if field_name not in df.columns:
                available_fields = [
                    c
                    for c in df.columns
                    if not c.startswith(("item_id", "document", "filename", "index"))
                ]
                logger.warning(
                    f"Field '{field_name}' not found in model '{model_name}' outputs. "
                    f"Ground truth config field names must match LLM output field names. "
                    f"Available output fields: {available_fields[:10]}"
                )
                continue

            # Get predictions and apply mapping if specified
            y_pred_raw = df[field_name].tolist()
            logger.debug(
                f"Raw predictions for {field_name}/{model_name}: {set(y_pred_raw)}"
            )

            if mapping:
                logger.debug(f"Applying mapping: {mapping}")

                # Check for unmapped prediction values (ERROR if no wildcard)
                # Need to account for boolean key normalization (yes/no → True/False in YAML)
                has_wildcard = "*" in mapping

                # Build normalized key set (same logic as apply_mapping)
                normalized_keys = set()
                for k in mapping.keys():
                    if isinstance(k, bool):
                        # Boolean keys map to yes/no/true/false variations
                        normalized_keys.add(k)
                        if k is True:
                            normalized_keys.update(
                                ["yes", "Yes", "YES", "true", "True", "TRUE"]
                            )
                        else:  # False
                            normalized_keys.update(
                                ["no", "No", "NO", "false", "False", "FALSE"]
                            )
                    else:
                        normalized_keys.add(k)

                unmapped_predictions = set(y_pred_raw) - normalized_keys

                if unmapped_predictions and not has_wildcard:
                    raise ValueError(
                        f"\nUnmapped prediction value(s) found in field '{field_name}' model '{model_name}':\n"
                        f"  Values: {sorted(unmapped_predictions)}\n\n"
                        f"Fix option 1 - Map to target values:\n"
                        + "\n".join(
                            f'    "{v}": <target_value>'
                            for v in sorted(unmapped_predictions)
                        )
                        + "\n\n  Option 2 - Drop/exclude from metrics:\n"
                        + "    mapping:\n"
                        + "\n".join(
                            f'      "{v}": ""' for v in sorted(unmapped_predictions)
                        )
                        + '\n    drop: [""]\n'
                        + "\n  Option 3 - Use wildcard:\n"
                        + '    "*": other  # or "*": "" with drop: [""]'
                    )

                y_pred = apply_mapping(y_pred_raw, mapping)
                logger.debug(f"Mapped predictions: {set(y_pred)}")
            else:
                y_pred = y_pred_raw

            # Filter out dropped values if specified
            if drop_values:
                original_count = len(y_true)
                # Create list of (true, pred) pairs, excluding dropped values
                valid_pairs = [
                    (t, p)
                    for t, p in zip(y_true, y_pred)
                    if t not in drop_values and p not in drop_values
                ]

                if valid_pairs:
                    y_true_filtered, y_pred_filtered = zip(*valid_pairs)
                    y_true = list(y_true_filtered)
                    y_pred = list(y_pred_filtered)
                else:
                    y_true = []
                    y_pred = []

                dropped_count = original_count - len(y_true)
                if dropped_count > 0:
                    logger.info(
                        f"Dropped {dropped_count}/{original_count} items in '{field_name}/{model_name}' "
                        f"(values: {drop_values})"
                    )

            # Calculate metrics
            metrics = calculate_classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                field_name=field_name,
                model_name=model_name,
            )

            field_results[model_name] = metrics

        if field_results:
            results[field_name] = field_results

    return results


def export_ground_truth_metrics(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    output_prefix: str,
    ground_truth_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Export ground truth metrics to CSV and JSON files.

    Args:
        metrics: Nested dict from calculate_ground_truth_metrics()
        output_prefix: Path prefix for output files
        ground_truth_config: Optional ground truth configuration with mappings

    Returns:
        DataFrame with metrics (one row per field/model combination)
    """
    import json

    # Flatten metrics for DataFrame export
    rows = []
    for field_name, model_metrics in metrics.items():
        for model_name, metric_dict in model_metrics.items():
            row = {"field": field_name, "model": model_name}

            # Add configuration information if available
            if ground_truth_config and field_name in ground_truth_config:
                config = ground_truth_config[field_name]
                row["ground_truth_column"] = config.get("column") or config.get(
                    "existing"
                )
                row["mapping"] = str(config.get("mapping", {}))
            else:
                row["ground_truth_column"] = None
                row["mapping"] = None

            # Add scalar metrics
            for key, value in metric_dict.items():
                if key not in [
                    "confusion_matrix",
                    "classification_report",
                    "labels",
                    "field",
                    "model",
                ]:
                    row[key] = value

            rows.append(row)

    # Export summary metrics as CSV
    if rows:
        df = pd.DataFrame(rows)
        # Reorder columns to put config info first
        config_cols = ["field", "model", "ground_truth_column", "mapping"]
        other_cols = [c for c in df.columns if c not in config_cols]
        df = df[config_cols + other_cols]

        csv_path = f"{output_prefix}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported ground truth metrics to {csv_path}")

        # Export full metrics (including confusion matrices) as JSON
        # Add config info to each metric dict
        enriched_metrics = {}
        for field_name, model_metrics in metrics.items():
            enriched_metrics[field_name] = {}
            for model_name, metric_dict in model_metrics.items():
                enriched_dict = metric_dict.copy()

                # Add config information
                if ground_truth_config and field_name in ground_truth_config:
                    config = ground_truth_config[field_name]
                    enriched_dict["ground_truth_column"] = config.get(
                        "column"
                    ) or config.get("existing")
                    enriched_dict["mapping_config"] = config.get("mapping", {})

                enriched_metrics[field_name][model_name] = enriched_dict

        json_path = f"{output_prefix}.json"
        with open(json_path, "w") as f:
            json.dump(enriched_metrics, f, indent=2, default=str)
        logger.info(f"Exported detailed ground truth metrics to {json_path}")

        return df

    return pd.DataFrame()


def export_confusion_matrices(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    output_folder: Path,
    ground_truth_config: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Export confusion matrices as separate CSV files.

    Args:
        metrics: Nested dict from calculate_ground_truth_metrics()
        output_folder: Folder to write confusion matrix files
        ground_truth_config: Optional ground truth configuration with mappings
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for field_name, model_metrics in metrics.items():
        for model_name, metric_dict in model_metrics.items():
            if "confusion_matrix" not in metric_dict:
                continue

            cm = metric_dict["confusion_matrix"]
            labels = metric_dict.get("labels", [])

            # Create DataFrame with labeled rows/columns
            df = pd.DataFrame(cm, index=labels, columns=labels)
            df.index.name = "True \\ Predicted"  # Row label (true values)
            df.columns.name = None  # Don't set column name to avoid offset

            # Safe filename
            safe_model = model_name.replace("/", "_").replace(":", "_")
            safe_field = field_name.replace("/", "_").replace(":", "_")
            filename = f"confusion_matrix_{safe_field}_{safe_model}.csv"

            # Build metadata header
            mapping_info = f"# Confusion Matrix: {field_name}\n"
            mapping_info += f"# Model: {model_name}\n"
            mapping_info += f"# Sample size: {metric_dict.get('n_samples', 'N/A')}\n"

            # Add performance metrics
            mapping_info += f"# Precision: {metric_dict.get('precision', 'N/A')}\n"
            mapping_info += f"# Recall: {metric_dict.get('recall', 'N/A')}\n"
            mapping_info += f"# F1: {metric_dict.get('f1', 'N/A')}\n"
            mapping_info += f"# Accuracy: {metric_dict.get('accuracy', 'N/A')}\n"

            # Add mapping information if available
            if ground_truth_config and field_name in ground_truth_config:
                config = ground_truth_config[field_name]
                column_name = config.get("column") or config.get("existing")
                mapping = config.get("mapping", {})

                if column_name:
                    mapping_info += f"# Ground truth column: {column_name}\n"

                if mapping:
                    mapping_info += "# Mapping applied:\n"
                    for k, v in mapping.items():
                        mapping_info += f"#   {k} -> {v}\n"

            mapping_info += "#\n"

            # Add column headers manually for clarity
            # Format: First column is "Actual/Predicted", then predicted labels
            output_path = output_folder / filename
            with open(output_path, "w") as f:
                # Write metadata as comments
                f.write(mapping_info)

                # Write header with predicted labels
                f.write(
                    "Actual \\ Predicted," + ",".join(str(lbl) for lbl in labels) + "\n"
                )
                # Write data rows with actual labels
                for actual_label, row in zip(labels, cm):
                    f.write(
                        str(actual_label)
                        + ","
                        + ",".join(str(val) for val in row)
                        + "\n"
                    )

            logger.info(f"Exported confusion matrix to {output_path}")
