"""Utility functions for similarity comparison."""

import base64
import csv
import io
from typing import List

import numpy as np
import pandas as pd


def create_embeddings_csv_base64(
    embeddings_a: dict, embeddings_b: dict, name_a: str, name_b: str
) -> str:
    """Create a base64-encoded CSV of embeddings for download.

    Args:
        embeddings_a: Dict with 'labels', 'texts', 'vectors' for set A
        embeddings_b: Dict with 'labels', 'texts', 'vectors' for set B
        name_a: Name of analysis A
        name_b: Name of analysis B

    Returns:
        Base64-encoded CSV string
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # determine embedding dimension
    dim = len(embeddings_a["vectors"][0]) if embeddings_a["vectors"] else 0

    # header row
    header = ["analysis", "label", "text"] + [f"dim_{i}" for i in range(dim)]
    writer.writerow(header)

    # write A embeddings
    for label, text, vec in zip(
        embeddings_a["labels"], embeddings_a["texts"], embeddings_a["vectors"]
    ):
        writer.writerow([name_a, label, text] + vec)

    # write B embeddings
    for label, text, vec in zip(
        embeddings_b["labels"], embeddings_b["texts"], embeddings_b["vectors"]
    ):
        writer.writerow([name_b, label, text] + vec)

    csv_content = output.getvalue()
    return base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")


def format_similarity_matrix(
    matrix,
    row_names: List[str],
    col_names: List[str],
    round_dp: int = 2,
    set_a_name: str = "A",
    set_b_name: str = "B",
    show_legend: bool = False,
) -> str:
    """Format similarity matrix for console output with numbered indices and legend.

    Args:
        matrix: Similarity matrix to format (numpy array)
        row_names: Names for rows (theme names from set A)
        col_names: Names for columns (theme names from set B)
        round_dp: Number of decimal places for rounding
        set_a_name: Name of set A (for legend)
        set_b_name: Name of set B (for legend)
        show_legend: Whether to include the legend mapping indices to theme names

    Returns:
        Formatted string with optional legend and numbered matrix
    """
    output_parts = []

    # create legend if requested
    if show_legend:
        legend_lines = []
        legend_lines.append(f"\n{set_a_name} Themes (rows):")
        for i, name in enumerate(row_names):
            legend_lines.append(f"  {i}: {name}")

        legend_lines.append(f"\n{set_b_name} Themes (columns):")
        for i, name in enumerate(col_names):
            legend_lines.append(f"  {i}: {name}")

        output_parts.append("\n".join(legend_lines))

    # create numbered matrix
    df = pd.DataFrame(
        np.round(matrix, round_dp),
        index=[str(i) for i in range(len(row_names))],
        columns=[str(i) for i in range(len(col_names))],
    )

    output_parts.append(str(df))

    return "\n\n".join(output_parts) if show_legend else output_parts[0]
