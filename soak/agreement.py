"""Inter-rater agreement statistics for classification results."""

import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from irrCAC.raw import CAC

logger = logging.getLogger(__name__)


def kappam_fleiss(ratings: pd.DataFrame) -> float:
    """Calculate Fleiss' Kappa coefficient.

    Args:
        ratings: DataFrame where each row is a subject and each column is a rater

    Returns:
        Fleiss' Kappa coefficient (0-1, higher is better)
    """
    try:
        # Check for perfect agreement (no variance)
        if ratings.nunique(axis=1).max() == 1:
            # All raters agree on all items = perfect agreement
            return 1.0

        cac = CAC(ratings, weights="identity")
        result = cac.fleiss()
        return round(float(result["est"]["coefficient_value"]), 4)
    except ZeroDivisionError:
        # No variance in data = perfect agreement
        logger.debug("Perfect agreement detected (no variance)")
        return 1.0
    except Exception as e:
        logger.warning(f"Failed to calculate Fleiss' Kappa: {e}")
        return float("nan")


def kripp_alpha(ratings: pd.DataFrame) -> float:
    """Calculate Krippendorff's Alpha coefficient.

    Args:
        ratings: DataFrame where each row is a subject and each column is a rater

    Returns:
        Krippendorff's Alpha coefficient (0-1, higher is better)
    """
    try:
        # Check for perfect agreement (no variance)
        if ratings.nunique(axis=1).max() == 1:
            # All raters agree on all items = perfect agreement
            return 1.0

        cac = CAC(ratings, weights="identity")
        result = cac.krippendorff()
        return round(float(result["est"]["coefficient_value"]), 3)
    except ZeroDivisionError:
        # No variance in data = perfect agreement
        logger.debug("Perfect agreement detected (no variance)")
        return 1.0
    except Exception as e:
        logger.warning(f"Failed to calculate Krippendorff's Alpha: {e}")
        return float("nan")


def percent_agreement(ratings: pd.DataFrame) -> float:
    """Calculate simple percent agreement across raters.

    Args:
        ratings: DataFrame where each row is a subject and each column is a rater

    Returns:
        Proportion of subjects where all raters agree (0-1)
    """
    try:
        # Each row = one subject, across raters
        # True if all raters gave same label
        agree = ratings.nunique(axis=1) == 1
        return round(agree.mean(), 3)
    except Exception as e:
        logger.warning(f"Failed to calculate percent agreement: {e}")
        return float("nan")


def gwet_ac1(ratings: pd.DataFrame) -> float:
    """Calculate Gwet's AC1 coefficient.

    Args:
        ratings: DataFrame where each row is a subject and each column is a rater

    Returns:
        Gwet's AC1 coefficient (0-1, higher is better)
    """
    try:
        # Check for perfect agreement (no variance)
        if ratings.nunique(axis=1).max() == 1:
            # All raters agree on all items = perfect agreement
            return 1.0

        cac = CAC(ratings, weights="identity")
        result = cac.gwet()
        return round(float(result["est"]["coefficient_value"]), 4)
    except ZeroDivisionError:
        # No variance in data = perfect agreement
        logger.debug("Perfect agreement detected (no variance)")
        return 1.0
    except Exception as e:
        logger.warning(f"Failed to calculate Gwet's AC1: {e}")
        return float("nan")


def _calculate_agreement(
    ratings_dict: Dict[str, pd.Series],
    fields: List[str],
    item_id_col: str,
    use_gwet: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Private helper to calculate agreement statistics from ratings data.

    Args:
        ratings_dict: Dict mapping rater_name -> DataFrame with classifications
        fields: List of field names to calculate agreement for
        item_id_col: Column name containing item identifiers
        use_gwet: If True, use Gwet's AC1; if False, use Fleiss' Kappa

    Returns:
        Dictionary mapping field names to agreement statistics
    """
    results = {}
    n_raters = len(ratings_dict)

    for field in fields:
        # Build ratings matrix: rows=items, cols=raters
        ratings_data = {}

        for rater_name, df in ratings_dict.items():
            if field not in df.columns:
                continue
            # Use item_id as index, field value as rating
            field_ratings = df.set_index(item_id_col)[field]
            ratings_data[rater_name] = field_ratings

        if len(ratings_data) < 2:
            logger.warning(f"Field '{field}' not present in enough raters, skipping")
            continue

        # Combine into single DataFrame
        ratings = pd.DataFrame(ratings_data)

        # Handle missing items across raters
        # Only include items present in ALL raters
        ratings = ratings.dropna()

        if len(ratings) == 0:
            logger.warning(f"No common items found for field '{field}'")
            metric_name = "Gwet_AC1" if use_gwet else "Fleiss_Kappa"
            results[field] = {
                metric_name: float("nan"),
                "Kripp_alpha": float("nan"),
                "Percent_Agreement": float("nan"),
                "n_items": 0,
                "n_raters": len(ratings_data),
            }
            continue

        # Calculate statistics
        metric_name = "Gwet_AC1" if use_gwet else "Fleiss_Kappa"
        metric_func = gwet_ac1 if use_gwet else kappam_fleiss

        results[field] = {
            metric_name: metric_func(ratings),
            "Kripp_alpha": kripp_alpha(ratings),
            "Percent_Agreement": percent_agreement(ratings),
            "n_items": len(ratings),
            "n_raters": len(ratings_data),
        }

    return results


def calculate_agreement_stats(
    csv_paths: List[Union[str, Path]],
    agreement_fields: List[str],
    item_id_col: str = "item_id",
) -> Dict[str, Dict[str, float]]:
    """Calculate inter-rater agreement statistics across multiple CSV files.

    Args:
        csv_paths: List of paths to CSV files containing classification results
        agreement_fields: List of column names to calculate agreement for
        item_id_col: Column name containing item identifiers (default: "item_id")

    Returns:
        Dictionary mapping field names to agreement statistics:
        {
            "field_name": {
                "AC1": float,
                "Kripp_alpha": float,
                "Percent_Agreement": float,
                "n_items": int,
                "n_raters": int
            }
        }
    """

    dfs = {}
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            # Use filename as rater identifier
            rater_name = Path(path).stem
            dfs[rater_name] = df
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            raise

    if not dfs:
        raise ValueError("No valid CSV files provided")

    # Verify all CSVs have the required columns
    for rater_name, df in dfs.items():
        if item_id_col not in df.columns:
            raise ValueError(
                f"{rater_name} missing required column '{item_id_col}'. Available columns: {list(df.columns)}"
            )
        missing_fields = [f for f in agreement_fields if f not in df.columns]
        if missing_fields:
            raise ValueError(
                f"{rater_name} missing agreement fields: {missing_fields}. Available columns: {list(df.columns)}"
            )

    return _calculate_agreement(dfs, agreement_fields, item_id_col, use_gwet=True)


def calculate_agreement_from_dataframes(
    model_dfs: Dict[str, pd.DataFrame],
    agreement_fields: List[str] = None,
    item_id_col: str = "item_id",
) -> Dict[str, Dict[str, float]]:
    """Calculate inter-rater agreement statistics from DataFrames.

    Args:
        model_dfs: Dict mapping model_name -> DataFrame with classifications
        agreement_fields: List of field names to calculate agreement for (auto-detected if None)
        item_id_col: Column name containing item identifiers (default: "item_id")

    Returns:
        Dictionary mapping field names to agreement statistics:
        {
            "field_name": {
                "AC1": float,
                "Kripp_alpha": float,
                "Percent_Agreement": float,
                "n_items": int,
                "n_raters": int
            }
        }
    """
    if not model_dfs or len(model_dfs) < 2:
        logger.warning("Need at least 2 raters to calculate agreement")
        return None

    # Auto-detect agreement fields if not specified
    if not agreement_fields:
        metadata_cols = {"item_id", "document", "filename", "index"}
        all_fields = {c for df in model_dfs.values() for c in df.columns}
        agreement_fields = sorted(
            f
            for f in all_fields
            if f not in metadata_cols and not f.endswith("__evidence")
        )
        logger.debug("Auto-detected agreement fields: %s", agreement_fields)

    if not agreement_fields:
        logger.warning("No agreement fields found")
        return None

    # Calculate agreement using shared helper
    try:
        stats = _calculate_agreement(
            model_dfs, agreement_fields, item_id_col, use_gwet=False
        )
    except Exception as e:
        logger.error(f"Error calculating agreement: {e}")
        return None

    if stats:
        logger.debug(f"Calculated agreement statistics for {len(stats)} fields")
    return stats if stats else None


def export_agreement_stats(
    stats: Dict[str, Dict[str, float]], output_prefix: str = "agreement_stats"
) -> pd.DataFrame:
    """Export agreement statistics to CSV and JSON files.

    Args:
        stats: Agreement statistics from calculate_agreement_stats()
        output_prefix: Prefix for output filenames (default: "agreement_stats")

    Returns:
        DataFrame with agreement statistics (transposed, fields as rows)
    """
    import json

    # Export as CSV (transposed for readability)
    df = pd.DataFrame(stats).T
    df.index.name = "field"
    df.to_csv(f"{output_prefix}.csv")

    # Export as JSON
    with open(f"{output_prefix}.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Exported agreement stats to {output_prefix}.csv and {output_prefix}.json"
    )

    return df
