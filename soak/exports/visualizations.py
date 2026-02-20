"""Visualisation utilities for export."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def create_sankey_figure(
    transport_plan: list,
    labels_a: List[str],
    labels_b: List[str],
    sim_matrix: Optional[list] = None,
    threshold: float = 0.01,
):
    """Create a Sankey diagram figure for optimal transport.

    Args:
        transport_plan: Transport plan matrix (list of lists)
        labels_a: Labels for source nodes
        labels_b: Labels for target nodes
        sim_matrix: Optional similarity matrix for colouring
        threshold: Minimum flow proportion to show

    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go

    plan = np.array(transport_plan)
    total_mass = plan.sum()

    if total_mass == 0:
        return go.Figure()

    n_a = len(labels_a)
    n_b = len(labels_b)

    def truncate(s: str, max_len: int = 25) -> str:
        return s[: max_len - 3] + "..." if len(s) > max_len else s

    node_labels = [truncate(label) for label in labels_a] + [truncate(label) for label in labels_b]

    sources = []
    targets = []
    values = []
    colors = []

    min_flow = threshold * total_mass

    for i in range(n_a):
        for j in range(n_b):
            flow = plan[i, j]
            if flow > min_flow:
                sources.append(i)
                targets.append(n_a + j)
                values.append(flow)

                if sim_matrix and i < len(sim_matrix) and j < len(sim_matrix[i]):
                    sim = sim_matrix[i][j]
                    if sim >= 0.7:
                        colors.append("rgba(76, 175, 80, 0.7)")
                    elif sim >= 0.5:
                        colors.append("rgba(255, 193, 7, 0.7)")
                    else:
                        colors.append("rgba(244, 67, 54, 0.7)")
                else:
                    colors.append("rgba(150, 150, 150, 0.5)")

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=["rgba(66, 133, 244, 0.8)"] * n_a + ["rgba(52, 168, 83, 0.8)"] * n_b,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=colors,
                ),
            )
        ]
    )

    fig.update_layout(
        font_size=10,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def create_heatmap_figure(
    matrix: list,
    labels_a: List[str],
    labels_b: List[str],
    title: str = "Similarity Matrix",
):
    """Create a heatmap figure for similarity matrix.

    Args:
        matrix: Similarity matrix (list of lists)
        labels_a: Row labels
        labels_b: Column labels
        title: Figure title

    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go

    def truncate(s: str, max_len: int = 20) -> str:
        return s[: max_len - 3] + "..." if len(s) > max_len else s

    short_a = [truncate(label) for label in labels_a]
    short_b = [truncate(label) for label in labels_b]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=short_b,
            y=short_a,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Similarity"),
            hovertemplate="%{y}<br>%{x}<br>Similarity: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=120, r=20, t=60, b=120),
    )

    return fig


def create_self_similarity_heatmap(
    self_similarity: dict,
    use_calibrated: bool = True,
    title: str = "Theme Self-Similarity",
):
    """Create a heatmap for theme self-similarity matrix.

    Args:
        self_similarity: Output from compute_self_similarity_matrix()
        use_calibrated: Whether to use calibrated scores
        title: Figure title

    Returns:
        Plotly figure object
    """
    matrix = self_similarity.get(
        "calibrated_matrix" if use_calibrated else "raw_matrix", []
    )
    themes = self_similarity.get("themes", [])

    return create_heatmap_figure(matrix, themes, themes, title)


def export_figure_png(
    fig, path: Path, dpi: int = 300, width: int = 800, height: int = 600
) -> None:
    """Export a Plotly figure as PNG at specified DPI.

    Args:
        fig: Plotly figure object
        path: Output path for PNG file
        dpi: Dots per inch (default 300 for print quality)
        width: Image width in pixels at 100 DPI base
        height: Image height in pixels at 100 DPI base
    """
    scale = dpi / 100
    fig.write_image(
        str(path),
        format="png",
        scale=scale,
        width=width,
        height=height,
    )
