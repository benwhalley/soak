"""Utility functions for exporting data from DAG nodes."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def export_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Export DataFrame to CSV file.

    Args:
        df: DataFrame to export
        path: Output path (should have .csv extension)
    """
    try:
        df.to_csv(path, index=False)
        logger.info(f"Exported CSV to {path}")
    except Exception as e:
        logger.error(f"Failed to export CSV to {path}: {e}")
        raise


def export_to_html(df: pd.DataFrame, path: Path) -> None:
    """Export DataFrame to HTML file with styling.

    Args:
        df: DataFrame to export
        path: Output path (should have .html extension)
    """
    try:
        html = df.to_html(index=False, escape=False, na_rep='')
        styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }}
        th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""
        path.write_text(styled_html)
        logger.info(f"Exported HTML to {path}")
    except Exception as e:
        logger.error(f"Failed to export HTML to {path}: {e}")
        raise


def export_to_json(rows: List[Dict], path: Path) -> None:
    """Export list of dictionaries to JSON file.

    Args:
        rows: List of row dictionaries to export
        path: Output path (should have .json extension)
    """
    try:
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2, default=str)
        logger.info(f"Exported JSON to {path}")
    except Exception as e:
        logger.error(f"Failed to export JSON to {path}: {e}")
        raise
