"""Visualization functions for similarity comparison (Sankey, heatmaps, scree plots)."""

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _cost_to_color(norm_cost: float, opacity: float = 0.6) -> str:
    """Convert normalised cost [0,1] to RGBA colour string using green-amber-red gradient.

    Args:
        norm_cost: Normalised cost value (0 = best match/green, 1 = worst match/red)
        opacity: Alpha value for the colour (0-1)

    Returns:
        RGBA colour string for Plotly
    """
    # Three-colour gradient: green -> amber -> red
    # Green (good match): RGB(39, 174, 96) - #27ae60
    # Amber (medium match): RGB(243, 156, 18) - #f39c12
    # Red (poor match): RGB(231, 76, 60) - #e74c3c
    green = (39, 174, 96)
    amber = (243, 156, 18)
    red = (231, 76, 60)

    if norm_cost <= 0.5:
        # Interpolate green -> amber (0 to 0.5)
        t = norm_cost * 2  # scale to 0-1
        r = int(green[0] + (amber[0] - green[0]) * t)
        g = int(green[1] + (amber[1] - green[1]) * t)
        b = int(green[2] + (amber[2] - green[2]) * t)
    else:
        # Interpolate amber -> red (0.5 to 1)
        t = (norm_cost - 0.5) * 2  # scale to 0-1
        r = int(amber[0] + (red[0] - amber[0]) * t)
        g = int(amber[1] + (red[1] - amber[1]) * t)
        b = int(amber[2] + (red[2] - amber[2]) * t)

    return f"rgba({r}, {g}, {b}, {opacity})"


# CSS to force opaque hover labels (Plotly Sankey inherits link opacity by default)
_SANKEY_HOVER_CSS = """
<style>
.hoverlayer .hovertext path {
    fill: white !important;
    fill-opacity: 1 !important;
    stroke: #ccc !important;
    stroke-opacity: 1 !important;
}
.hoverlayer .hovertext text {
    fill: black !important;
    fill-opacity: 1 !important;
}
</style>
"""

# Plotly config with export buttons
_SANKEY_PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "staticPlot": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    ],
    "modeBarButtonsToAdd": [],
    "toImageButtonOptions": {
        "format": "svg",  # default to SVG for print quality
        "filename": "sankey_diagram",
        "height": None,
        "width": None,
        "scale": 2,
    },
}

# System UI font stack (consistent with struckdown online editor)
_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif"


class SankeyHTML:
    """Wrapper for Sankey diagram that provides both HTML and base64 PNG."""

    def __init__(
        self, html_content: str, png_buffer: BytesIO, name: str = "transport_sankey"
    ):
        self.html_content = html_content
        self.png_buffer = png_buffer
        self.name = name

    @property
    def html(self) -> str:
        """Return full HTML with CSS for embedding."""
        return self.html_content

    @property
    def base64(self) -> str:
        """Return base64-encoded PNG for static display."""
        self.png_buffer.seek(0)
        return base64.b64encode(self.png_buffer.read()).decode("utf-8")


class Base64ImageFile:
    """Simple wrapper for BytesIO that provides base64 encoding."""

    def __init__(self, buffer, name=None):
        self.buffer = buffer
        self.name = name

    @property
    def base64(self):
        self.buffer.seek(0)
        return base64.b64encode(self.buffer.read()).decode("utf-8")


def create_transport_sankey(
    transport_plan,
    theme_names_a: List[str],
    theme_names_b: List[str],
    cost_matrix=None,
    cost_matrix_original=None,
    analysis_name_a: str = "A",
    analysis_name_b: str = "B",
    threshold_ratio: float = 0.01,
    link_opacity: float = 0.6,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
) -> "SankeyHTML":
    """Create interactive Sankey diagram visualising optimal transport flow.

    Features:
    - Green-amber-red colour scale for alignment quality (green = good, red = poor)
    - Labels outside plot area with text wrapping and hyphenation
    - A nodes in fixed alphabetical order, B nodes positioned to minimise crossings
    - Hover text showing mass proportions, cost contribution, and unit cost
    - Opaque hover labels

    Args:
        transport_plan: (n_A x n_B) transport coupling matrix P
        theme_names_a: Theme names for set A (left side)
        theme_names_b: Theme names for set B (right side)
        cost_matrix: Cost matrix for colouring links (1 - rescaled_similarity)
        cost_matrix_original: Original cost matrix before rescaling (for hover display)
        analysis_name_a: Name of analysis A
        analysis_name_b: Name of analysis B
        threshold_ratio: Drop links below this fraction of max flow
        link_opacity: Opacity of link colours (0-1)
        cost_min: Minimum cost for color scale (green). If None, computed from links.
        cost_max: Maximum cost for color scale (red). If None, computed from links.

    Returns:
        SankeyHTML object with .html and .base64 properties
    """
    import plotly.graph_objects as go
    import pyphen

    P = np.asarray(transport_plan)
    n_A, n_B = P.shape

    # handle empty case
    if n_A == 0 or n_B == 0:
        empty_html = "<div>No themes to display</div>"
        empty_buffer = BytesIO()
        return SankeyHTML(empty_html, empty_buffer, name="transport_sankey")

    threshold = threshold_ratio * P.max()

    # sort nodes alphanumerically for consistency
    a_order = np.argsort([n.lower() for n in theme_names_a])
    b_order = np.argsort([n.lower() for n in theme_names_b])

    P_sorted = P[a_order, :][:, b_order]
    names_a_sorted = [theme_names_a[i] for i in a_order]
    names_b_sorted = [theme_names_b[i] for i in b_order]

    if cost_matrix is not None:
        M_sorted = np.asarray(cost_matrix)[a_order, :][:, b_order]
    else:
        M_sorted = None

    if cost_matrix_original is not None:
        M_orig_sorted = np.asarray(cost_matrix_original)[a_order, :][:, b_order]
    else:
        M_orig_sorted = None

    # text wrapping with hyphenation and widow control
    dic = pyphen.Pyphen(lang="en_GB")

    def hyphenate_word(word, max_len=10):
        if len(word) <= max_len:
            return [word]
        pairs = dic.pairs(word)
        if not pairs:
            return [word]
        mid = len(word) // 2
        best_pair = min(pairs, key=lambda p: abs(len(p[0]) - mid))
        return [best_pair[0] + "-", best_pair[1]]

    def wrap_with_hyphenation(text, width):
        words = text.split()
        lines = []
        current_line = []
        current_len = 0

        for word in words:
            word_len = len(word)
            space_needed = 1 if current_line else 0

            if current_len + space_needed + word_len <= width:
                current_line.append(word)
                current_len += space_needed + word_len
            elif word_len > width:
                parts = hyphenate_word(word)
                for part in parts:
                    if current_len + (1 if current_line else 0) + len(part) <= width:
                        current_line.append(part)
                        current_len += (1 if current_line else 0) + len(part)
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [part]
                        current_len = len(part)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_len = word_len

        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def wrap_text(text, max_width=35, min_last_line_ratio=0.6):
        if len(text) <= max_width:
            return text

        best_lines = None
        best_score = float("inf")

        for width in range(max(20, max_width - 8), max_width + 1):
            lines = wrap_with_hyphenation(text, width)
            if not lines:
                continue

            lengths = [len(line) for line in lines]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

            last_len = lengths[-1]
            max_len = max(lengths)
            widow_ratio = last_len / max_len if max_len > 0 else 1
            widow_penalty = (
                50 * (1 - widow_ratio) ** 2 if widow_ratio < min_last_line_ratio else 0
            )

            score = len(lines) * 5 + variance * 0.5 + widow_penalty

            if score < best_score:
                best_score = score
                best_lines = lines

        return "<br>".join(best_lines) if best_lines else text

    # node hover texts (show full analysis name in hover)
    hover_texts = [
        f"{analysis_name_a}: {names_a_sorted[i]}" for i in range(n_A)
    ] + [f"{analysis_name_b}: {names_b_sorted[j]}" for j in range(n_B)]

    # collect links and unit costs (both rescaled and original)
    sources, targets, values, link_costs, link_costs_orig = [], [], [], [], []
    for i in range(n_A):
        for j in range(n_B):
            flow = P_sorted[i, j]
            if flow > threshold:
                sources.append(i)
                targets.append(n_A + j)
                values.append(float(flow))
                if M_sorted is not None:
                    link_costs.append(M_sorted[i, j])
                if M_orig_sorted is not None:
                    link_costs_orig.append(M_orig_sorted[i, j])
                elif M_sorted is not None:
                    link_costs_orig.append(M_sorted[i, j])

    # map costs to colours
    if link_costs:
        if cost_min is None:
            cost_min = min(link_costs)
        if cost_max is None:
            cost_max = max(link_costs)
        cost_range = cost_max - cost_min
        if cost_range < 1e-9:
            cost_range = 1.0

        colors = []
        for cost in link_costs:
            norm_cost = (cost - cost_min) / cost_range
            norm_cost = max(0.0, min(1.0, norm_cost))
            colors.append(_cost_to_color(norm_cost, link_opacity))
    else:
        colors = [f"rgba(100, 150, 200, {link_opacity})"] * len(sources)
        link_costs = [0.5] * len(sources)
        cost_min, cost_max = 0.0, 1.0

    # A positions: fixed top to bottom
    y_a = np.linspace(0.02, 0.98, n_A).tolist() if n_A > 1 else [0.5]

    # B positions: flow-weighted to minimise crossings
    b_weighted_y = []
    for j in range(n_B):
        total_flow = 0
        weighted_sum = 0
        for i in range(n_A):
            flow = P_sorted[i, j]
            if flow > 0:
                weighted_sum += flow * y_a[i]
                total_flow += flow
        b_weighted_y.append(weighted_sum / total_flow if total_flow > 0 else 0.5)

    b_order_by_y = np.argsort(b_weighted_y)
    y_b = [0.0] * n_B
    y_positions = np.linspace(0.02, 0.98, n_B) if n_B > 1 else [0.5]
    for rank, b_idx in enumerate(b_order_by_y):
        y_b[b_idx] = y_positions[rank]

    node_x = [0.05] * n_A + [0.95] * n_B
    node_y = y_a + y_b

    # build link hover text
    total_cost = (
        sum(values[i] * link_costs[i] for i in range(len(values))) if link_costs else 1
    )
    a_mass_totals = P_sorted.sum(axis=1)
    b_mass_totals = P_sorted.sum(axis=0)

    link_hovers = []
    for idx in range(len(sources)):
        src_idx = sources[idx]
        tgt_idx = targets[idx] - n_A
        flow = values[idx]
        cost = link_costs[idx]
        cost_orig = link_costs_orig[idx] if link_costs_orig else cost

        mass_prop_a = flow / a_mass_totals[src_idx] if a_mass_totals[src_idx] > 0 else 0
        mass_prop_b = flow / b_mass_totals[tgt_idx] if b_mass_totals[tgt_idx] > 0 else 0

        link_cost_contribution = flow * cost
        cost_prop = link_cost_contribution / total_cost if total_cost > 0 else 0

        similarity_orig = 1 - cost_orig
        similarity_rescaled = 1 - cost
        link_hovers.append(
            f"A{src_idx+1} -> B{tgt_idx+1}<br>"
            f"<b>Similarity:</b> {similarity_orig:.3f} (rescaled: {similarity_rescaled:.3f})<br>"
            f"<b>Mass:</b> {mass_prop_a:.0%} of A{src_idx+1}, "
            f"{mass_prop_b:.0%} of B{tgt_idx+1}<br>"
            f"<b>Cost contribution:</b> {cost_prop:.1%} of total"
        )

    node_labels = [""] * (n_A + n_B)

    hoverlabel_style = dict(
        bgcolor="white",
        bordercolor="#ccc",
        font=dict(family=_FONT_STACK, size=12, color="black"),
    )

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=20,
                    thickness=2,
                    line=dict(color="#666666", width=0.5),
                    label=node_labels,
                    color=["#666666"] * (n_A + n_B),
                    x=node_x,
                    y=node_y,
                    customdata=hover_texts,
                    hovertemplate="%{customdata}<extra></extra>",
                    hoverlabel=hoverlabel_style,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=colors,
                    customdata=link_hovers,
                    hovertemplate="%{customdata}<extra></extra>",
                    hoverlabel=hoverlabel_style,
                ),
            )
        ]
    )

    # annotations for labels outside plot area
    annotations = []

    for i, name in enumerate(names_a_sorted):
        annotations.append(
            dict(
                x=-0.01,
                y=1 - y_a[i],
                xref="paper",
                yref="paper",
                text=f"{wrap_text(name, 50)}",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(size=13),
                align="right",
            )
        )

    for j, name in enumerate(names_b_sorted):
        annotations.append(
            dict(
                x=1.01,
                y=1 - y_b[j],
                xref="paper",
                yref="paper",
                text=f"{wrap_text(name, 50)}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=13),
                align="left",
            )
        )

    padding = 48

    # add continuous colorbar
    if link_costs:
        sim_max = 1 - cost_min
        sim_min = 1 - cost_max

        colorscale = [
            [0.0, "#e74c3c"],
            [0.5, "#f39c12"],
            [1.0, "#27ae60"],
        ]

        sim_range = sim_max - sim_min
        tick_vals = [sim_min + sim_range * i / 4 for i in range(5)]

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale=colorscale,
                    cmin=sim_min,
                    cmax=sim_max,
                    color=[(sim_min + sim_max) / 2],
                    colorbar=dict(
                        title=dict(text="Similarity", side="top", font=dict(size=13)),
                        orientation="h",
                        x=0.5,
                        y=-0.08,
                        xanchor="center",
                        yanchor="top",
                        len=0.4,
                        thickness=15,
                        tickfont=dict(size=11),
                        tickformat=".2f",
                        tickvals=tick_vals,
                    ),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title_text="",
        font=dict(family=_FONT_STACK, size=9),
        width=1100,
        height=max(600, 60 * max(n_A, n_B)),
        margin=dict(l=380 + padding, r=380 + padding, t=padding, b=padding + 80),
        annotations=annotations,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 1)",
            bordercolor="rgba(200, 200, 200, 1)",
            font=dict(family=_FONT_STACK, size=12, color="rgba(0, 0, 0, 1)"),
            namelength=-1,
        ),
    )

    html_str = fig.to_html(
        config=_SANKEY_PLOTLY_CONFIG, include_plotlyjs="cdn", full_html=True
    )
    html_str = html_str.replace("<head>", f"<head>{_SANKEY_HOVER_CSS}")

    download_buttons_html = """
<div style="text-align: center; margin: 20px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;">
    <span style="margin-right: 10px; color: #666; font-size: 13px;">Download:</span>
    <button onclick="downloadSankey('svg')" style="
        padding: 8px 16px; margin: 0 5px; cursor: pointer;
        background: #3498db; color: white; border: none; border-radius: 4px;
        font-size: 13px; font-family: inherit;
    ">SVG (vector)</button>
    <button onclick="downloadSankey('png')" style="
        padding: 8px 16px; margin: 0 5px; cursor: pointer;
        background: #27ae60; color: white; border: none; border-radius: 4px;
        font-size: 13px; font-family: inherit;
    ">PNG (2x)</button>
    <button onclick="downloadSankey('pdf')" style="
        padding: 8px 16px; margin: 0 5px; cursor: pointer;
        background: #9b59b6; color: white; border: none; border-radius: 4px;
        font-size: 13px; font-family: inherit;
    ">PDF</button>
</div>
<script>
function downloadSankey(format) {
    var gd = document.querySelector('.plotly-graph-div');
    var filename = 'sankey_diagram';
    if (format === 'pdf') {
        Plotly.downloadImage(gd, {format: 'svg', filename: filename + '_for_pdf', scale: 2});
        alert('SVG downloaded. Open in Inkscape, Illustrator, or use an online converter to save as PDF.');
    } else {
        Plotly.downloadImage(gd, {format: format, filename: filename, scale: 2});
    }
}
</script>
"""
    html_str = html_str.replace("</body>", download_buttons_html + "</body>")

    try:
        img_bytes = fig.to_image(format="png", scale=2)
        png_buffer = BytesIO(img_bytes)
    except Exception as e:
        logger.debug(f"PNG export failed: {e}")
        png_buffer = BytesIO()

    return SankeyHTML(html_str, png_buffer, name="transport_sankey")


def create_transport_heatmap(
    transport_plan,
    theme_names_a: List[str],
    theme_names_b: List[str],
    analysis_name_a: str = "A",
    analysis_name_b: str = "B",
) -> "Base64ImageFile":
    """Create heatmap of transport plan P, sorted alphanumerically.

    Args:
        transport_plan: (n_A x n_B) transport coupling matrix P
        theme_names_a: Theme names for set A (rows)
        theme_names_b: Theme names for set B (columns)
        analysis_name_a: Name of analysis A
        analysis_name_b: Name of analysis B

    Returns:
        Base64ImageFile containing the heatmap
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    P = np.asarray(transport_plan)
    n_A, n_B = P.shape

    if n_A == 0 or n_B == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No themes to display", ha="center", va="center")
        ax.axis("off")
        buffer = BytesIO()
        fig.savefig(buffer, dpi=150, bbox_inches="tight", format="png")
        plt.close(fig)
        buffer.seek(0)
        return Base64ImageFile(buffer, name="transport_heatmap.png")

    names_a = list(theme_names_a)
    names_b = list(theme_names_b)

    # sort alphanumerically for consistency
    a_order = np.argsort([n.lower() for n in names_a])
    b_order = np.argsort([n.lower() for n in names_b])
    P = P[a_order, :][:, b_order]
    names_a = [names_a[i] for i in a_order]
    names_b = [names_b[i] for i in b_order]

    def truncate(s, max_len=30):
        return s if len(s) <= max_len else s[: max_len - 3] + "..."

    names_a_display = [truncate(n) for n in names_a]
    names_b_display = [truncate(n) for n in names_b]

    shared_mass = P.sum()
    if shared_mass > 1e-9:
        P_pct = (P / shared_mass) * 100
    else:
        P_pct = P * 100
    logger.info(
        f"Transport heatmap: max={P_pct.max():.1f}%, sum={P_pct.sum():.1f}% (shared_mass={shared_mass:.1%})"
    )

    fig_height = max(6, n_A * 0.4)
    fig_width = max(10, n_B * 0.5)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    df = pd.DataFrame(P_pct, index=names_a_display, columns=names_b_display)

    sns.heatmap(
        df,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "% of Transported Mass"},
        ax=ax,
        vmin=0,
    )

    ax.set_title(f"Transport Plan: {analysis_name_a} -> {analysis_name_b}")
    ax.set_xlabel(analysis_name_b)
    ax.set_ylabel(analysis_name_a)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)
    return Base64ImageFile(buffer, name="transport_heatmap.png")


def find_elbow_points(
    k_values: List[float],
    shared_mass: List[float],
    *,
    n_interp: int = 100,
    eps: float = 1e-12,
    plateau_threshold: float = 0.20,
    min_k_for_elbow: float = 0.1,
    max_k_for_elbow: float = 2.0,
) -> dict:
    """Find elbow points using maximum curvature and diminishing returns methods.

    Uses maximum curvature to find the elbow point where the curve bends most sharply.

    Args:
        k_values: K parameter values (should be positive)
        shared_mass: Corresponding shared mass values (should increase with K)
        n_interp: Number of points for uniform grid (default: 100)
        eps: Small constant for numerical stability (default: 1e-12)
        plateau_threshold: Threshold for diminishing returns (default: 0.20)
        min_k_for_elbow: Minimum K to consider for elbow detection (default: 0.1)
        max_k_for_elbow: Maximum K to consider for elbow detection (default: 2.0)

    Returns:
        Dictionary with elbow_idx, elbow_k, diminishing_idx, diminishing_k, plateau_reached
    """
    from scipy.interpolate import UnivariateSpline

    K = np.asarray(k_values, float)
    s = np.asarray(shared_mass, float)

    elbow_mask = (K >= min_k_for_elbow) & (K <= max_k_for_elbow)
    K_elbow = K[elbow_mask]
    s_elbow = s[elbow_mask]

    K_uniform = np.linspace(K_elbow.min(), K_elbow.max(), n_interp)
    s_uniform = np.interp(K_uniform, K_elbow, s_elbow)

    if len(s_uniform) >= 3:
        kernel = np.ones(3) / 3
        s_padded = np.pad(s_uniform, (1, 1), mode="edge")
        s_uniform = np.convolve(s_padded, kernel, mode="valid")

    K_all_uniform = np.linspace(K.min(), K.max(), n_interp)
    s_all_uniform = np.interp(K_all_uniform, K, s)
    if len(s_all_uniform) >= 3:
        kernel = np.ones(3) / 3
        s_all_padded = np.pad(s_all_uniform, (1, 1), mode="edge")
        s_all_uniform = np.convolve(s_all_padded, kernel, mode="valid")

    slope = np.diff(s_all_uniform)
    n_window = min(5, len(slope) // 4) if len(slope) > 4 else 1
    initial_slope = np.mean(np.abs(slope[:n_window])) if len(slope) >= n_window else 0
    final_slope = np.mean(np.abs(slope[-n_window:])) if len(slope) >= n_window else 0

    relative_plateau = (initial_slope <= 0) or (
        final_slope / (initial_slope + 1e-12) < 0.25
    )
    n_tail = max(1, len(s_all_uniform) // 5)
    tail_range = s_all_uniform[-1] - s_all_uniform[-n_tail]
    absolute_plateau = abs(tail_range) < 4.0
    high_value_plateau = s_all_uniform[-1] > 85.0
    if len(slope) >= 4:
        mid = len(slope) // 2
        first_half_slope = np.mean(np.abs(slope[:mid]))
        second_half_slope = np.mean(np.abs(slope[mid:]))
        decelerating = second_half_slope < 0.5 * first_half_slope
    else:
        decelerating = False
    plateau_reached = (
        relative_plateau or absolute_plateau or high_value_plateau or decelerating
    )

    x_min, x_max = K_uniform.min(), K_uniform.max()
    y_min, y_max = s_uniform.min(), s_uniform.max()
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    x_norm = (K_uniform - x_min) / x_range
    y_norm = (s_uniform - y_min) / y_range

    try:
        spline = UnivariateSpline(x_norm, y_norm, s=0.01, k=4)
        dy = spline.derivative(1)(x_norm)
        d2y = spline.derivative(2)(x_norm)
    except Exception:
        dy = np.gradient(y_norm, x_norm)
        d2y = np.gradient(dy, x_norm)

    curvature = np.abs(d2y) / (1 + dy**2) ** 1.5

    start_margin = max(1, n_interp // 20)
    end_margin = max(1, n_interp // 10)
    inner_curvature = curvature[start_margin:-end_margin]
    max_curv_idx = start_margin + int(np.argmax(inner_curvature))

    elbow_K = K_uniform[max_curv_idx]
    elbow_orig_idx = int(np.argmin(np.abs(K - elbow_K)))

    k_arr = np.asarray(k_values, float)
    sm_arr = np.asarray(shared_mass, float)

    dk = np.diff(k_arr)
    dsm = np.diff(sm_arr)
    slopes_original = dsm / (dk + 1e-12)

    n_init = min(3, len(slopes_original))
    initial_slope_orig = np.mean(slopes_original[:n_init]) if n_init > 0 else 0

    thr_orig = plateau_threshold * initial_slope_orig if initial_slope_orig > 0 else 0
    diminishing_orig_idx = None

    run = 0
    for i, slope_val in enumerate(slopes_original):
        if slope_val < thr_orig:
            run += 1
            if run >= 2:
                diminishing_orig_idx = i - 1
                break
        else:
            run = 0

    if diminishing_orig_idx is None:
        above_thr = np.where(slopes_original >= thr_orig)[0]
        if len(above_thr) > 0:
            diminishing_orig_idx = above_thr[-1] + 1
        else:
            diminishing_orig_idx = len(k_arr) - 1

    diminishing_orig_idx = min(diminishing_orig_idx, len(k_arr) - 1)

    return {
        "chord_idx": elbow_orig_idx,
        "chord_k": K[elbow_orig_idx],
        "elbow_idx": elbow_orig_idx,
        "elbow_k": K[elbow_orig_idx],
        "diminishing_idx": diminishing_orig_idx,
        "diminishing_k": K[diminishing_orig_idx],
        "plateau_reached": plateau_reached,
    }
