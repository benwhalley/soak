"""Inter-rater agreement statistics for classification results."""

import logging
from pathlib import Path
from typing import Dict, List, Union

import krippendorff
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

logger = logging.getLogger(__name__)


def gwet_ac1(ratings: pd.DataFrame) -> float:
    """Calculate Fleiss' Kappa (replaced Gwet's AC1).

    Args:
        ratings: DataFrame where each row is a subject and each column is a rater

    Returns:
        Fleiss' Kappa coefficient (0-1, higher is better)
    """
    try:
        # Convert to format expected by statsmodels: (n_subjects, n_categories)
        table, _ = aggregate_raters(ratings.values)
        kappa = fleiss_kappa(table, method="fleiss")
        return round(float(kappa), 3)
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
        # krippendorff expects shape (n_raters, n_subjects)
        alpha = krippendorff.alpha(
            reliability_data=ratings.T.values, level_of_measurement="nominal"
        )
        return round(float(alpha), 3)
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
    # Load all CSVs
    dfs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            # Use filename as rater identifier
            rater_name = Path(path).stem
            dfs.append((rater_name, df))
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            raise

    if not dfs:
        raise ValueError("No valid CSV files provided")

    # Verify all CSVs have the required columns
    for rater_name, df in dfs:
        if item_id_col not in df.columns:
            raise ValueError(
                f"{rater_name} missing required column '{item_id_col}'. Available columns: {list(df.columns)}"
            )
        missing_fields = [f for f in agreement_fields if f not in df.columns]
        if missing_fields:
            raise ValueError(
                f"{rater_name} missing agreement fields: {missing_fields}. Available columns: {list(df.columns)}"
            )

    results = {}

    for field in agreement_fields:
        # Build ratings matrix: rows=items, cols=raters
        ratings_data = {}

        for rater_name, df in dfs:
            # Use item_id as index, field value as rating
            field_ratings = df.set_index(item_id_col)[field]
            ratings_data[rater_name] = field_ratings

        # Combine into single DataFrame
        ratings = pd.DataFrame(ratings_data)

        # Handle missing items across raters
        # Only include items present in ALL CSVs
        ratings = ratings.dropna()

        if len(ratings) == 0:
            logger.warning(f"No common items found for field '{field}'")
            results[field] = {
                "AC1": float("nan"),
                "Kripp_alpha": float("nan"),
                "Percent_Agreement": float("nan"),
                "n_items": 0,
                "n_raters": len(dfs),
            }
            continue

        # Calculate statistics
        results[field] = {
            "Fleiss_Kappa": gwet_ac1(ratings),
            "Kripp_alpha": kripp_alpha(ratings),
            "Percent_Agreement": percent_agreement(ratings),
            "n_items": len(ratings),
            "n_raters": len(dfs),
        }

    return results


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

    # Calculate agreement for each field
    stats = {}
    for field in agreement_fields:
        try:
            # Build ratings matrix: rows=items, cols=models
            ratings_data = {
                model_name: df.set_index(item_id_col)[field]
                for model_name, df in model_dfs.items()
                if field in df.columns
            }

            if len(ratings_data) < 2:
                logger.warning(
                    f"Field '{field}' not present in enough models, skipping"
                )
                continue

            ratings = pd.DataFrame(ratings_data).dropna()

            if len(ratings) == 0:
                logger.warning(f"No common items for field '{field}'")
                continue

            # Calculate statistics
            stats[field] = {
                "Fleiss_Kappa": gwet_ac1(ratings),
                "Kripp_alpha": kripp_alpha(ratings),
                "Percent_Agreement": percent_agreement(ratings),
                "n_items": len(ratings),
                "n_raters": len(ratings_data),
            }
        except Exception as e:
            logger.error(f"Error calculating agreement for field '{field}': {e}")
            stats[field] = {"error": str(e)}

    if stats:
        logger.debug(f"Calculated agreement statistics for {len(stats)} fields")

    return stats if stats else None


def export_agreement_stats(
    stats: Dict[str, Dict[str, float]], output_prefix: str = "agreement_stats"
) -> None:
    """Export agreement statistics to CSV and JSON files.

    Args:
        stats: Agreement statistics from calculate_agreement_stats()
        output_prefix: Prefix for output filenames (default: "agreement_stats")
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
