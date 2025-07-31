"""Theme and code similarity comparison using embeddings."""

import base64
import itertools
import textwrap
from collections import OrderedDict
from io import BytesIO
from typing import Any, Dict, List

import matplotlib
from chatter import get_embedding
from django.core.files.images import ImageFile
from prefect import flow, task
from prefect.cache_policies import INPUTS
from prefect.task_runners import ConcurrentTaskRunner
from soak.models import QualitativeAnalysis, QualitativeAnalysisComparison, resolve

matplotlib.use("Agg")  # Non-GUI backend for headless use (saves to file only)


class Base64ImageFile(ImageFile):
    @property
    def base64(self):
        if self.closed:
            self.open()
        self.seek(0)
        return base64.b64encode(self.read()).decode("utf-8")


@task(persist_result=True, cache_policy=INPUTS)
def compare_result_similarity(
    A: QualitativeAnalysis, B: QualitativeAnalysis, threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Compare two sets of theme embeddings.

    Allows many-to-one matches: each theme may match multiple from the other set.

    Returns:
        - precision: % of B themes with at least one A match
        - recall: % of A themes with at least one B match
        - f1: harmonic mean of precision and recall
        - jaccard: proportion of theme pairs with similarity > threshold
        - match_matrix: binary matrix [n_A x n_B], 1 = similarity above threshold
        - similarity_matrix: raw cosine similarity values
    """

    A = [i.name for i in A.themes]
    B = [i.name for i in B.themes]

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    emb_A = get_embedding(list(map(lambda x: x.strip(), A)))
    emb_B = get_embedding(list(map(lambda x: x.strip(), B)))

    assert len(emb_A) == len(A), f"Mismatch in emb_A: {len(emb_A)} != {len(A)}"
    assert len(emb_B) == len(B), f"Mismatch in emb_B: {len(emb_B)} != {len(B)}"

    # Handle empty theme sets
    if len(emb_A) == 0 or len(emb_B) == 0:
        n_A = len(emb_A)
        n_B = len(emb_B)
        return {
            "error": "No themes found in any results. Cannot perform similarity comparison.",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "match_matrix": np.zeros((n_A, n_B), dtype=int),
            "similarity_matrix": np.zeros((n_A, n_B)),
            "a_b_most_similar": 0.0,
            "b_a_most_similar": 0.0,
            "similarity_f1": 0.0,
        }

    sim_matrix = cosine_similarity(emb_A, emb_B)
    match_matrix = sim_matrix >= threshold

    # Recall: % of A themes with any match
    recall_hits = match_matrix.any(axis=1).sum()
    recall = recall_hits / len(emb_A) if len(emb_A) > 0 else 0

    # Precision: % of B themes with any match
    precision_hits = match_matrix.any(axis=0).sum()
    precision = precision_hits / len(emb_B) if len(emb_B) > 0 else 0

    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

    # Jaccard: intersection / union across all pairwise theme comparisons
    intersection = match_matrix.sum()
    union = match_matrix.size  # total possible pairs = n_A * n_B
    jaccard = intersection / union if union > 0 else 0

    # best match of A themes with any B score
    a_b_most_similar = sim_matrix.max(axis=1).mean().round(3) if len(emb_A) > 0 else 0

    # best match of B themes with any A score
    b_a_most_similar = sim_matrix.max(axis=0).mean().round(3) if len(emb_B) > 0 else 0

    similarity_f1 = (
        2 * (a_b_most_similar * b_a_most_similar) / (a_b_most_similar + b_a_most_similar)
        if (a_b_most_similar + b_a_most_similar) > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "match_matrix": match_matrix.astype(int),
        "similarity_matrix": np.round(sim_matrix, 3),
        "a_b_most_similar": a_b_most_similar,
        "b_a_most_similar": b_a_most_similar,
        "similarity_f1": similarity_f1,
    }


@task(persist_result=True, cache_policy=INPUTS)
def network_similarity_plot(
    pipeline_results: List[QualitativeAnalysis],
    method="umap",
    n_neighbors=5,
    min_dist=0.01,
    threshold=0.6,
) -> str:
    """Create similarity plot using embedding visualization."""
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_similarity
    from umap import UMAP

    theme_sets_ = [[j.name for j in i.themes] for i in pipeline_results]
    theme_sets = [i for i in theme_sets_ if i]

    pipeline_names = [i.name() for i in pipeline_results]

    # Get embeddings for all sets
    embeddings = [get_embedding(set_str) for set_str in theme_sets]
    all_emb = np.vstack(embeddings)
    sim_matrix = cosine_similarity(all_emb)

    # Create graph
    G = nx.Graph()
    start_index = 0

    colors = [plt.cm.Set1(i) for i in range(len(embeddings))]
    valid_indices = list(range(len(embeddings)))  # Track which sets have themes

    for plot_idx, (emb, original_idx) in enumerate(zip(embeddings, valid_indices)):
        set_str = theme_sets[original_idx]
        lines = set_str
        for i, phrase in enumerate(lines, start=start_index):
            if not phrase.strip():
                continue
            G.add_node(i, label=phrase, set=chr(65 + plot_idx))
        start_index += len(emb)

    # Create 2D embedding
    if method == "umap":
        # Adjust n_neighbors if it's too large for the dataset
        effective_n_neighbors = min(n_neighbors, len(all_emb) - 1)
        effective_n_neighbors = max(2, effective_n_neighbors)  # Ensure at least 2

        reducer = UMAP(
            n_components=2,
            n_neighbors=effective_n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        pos_2d = reducer.fit_transform(all_emb)
    elif method == "mds":
        # Classical MDS expects a distance matrix, so convert similarity
        dist_matrix = 1 - sim_matrix
        reducer = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        pos_2d = reducer.fit_transform(dist_matrix)
    else:
        reducer = PCA(n_components=2)
        pos_2d = reducer.fit_transform(all_emb)

    pos = {i: pos_2d[i] for i in range(len(all_emb))}

    # Add edges based on threshold
    for i in range(len(all_emb)):
        for j in range(i + 1, len(all_emb)):
            if sim_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=sim_matrix[i, j])

    # Create plot
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 10))
    node_colors = [
        colors[ord(G.nodes[n].get("set", "?")) - 65] for n in G.nodes
    ]  # Assign colors based on set

    # Draw nodes with colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=200, ax=ax)

    # Add legend for sets (only valid ones with themes)
    legend_labels = [pipeline_names[idx] for idx in valid_indices]
    for i, label in enumerate(legend_labels):
        ax.scatter([], [], color=colors[i], label=label)
    ax.legend(title="Pipeline Results", loc="upper right")

    # Draw edges with alpha proportional to similarity weight
    edges = G.edges(data=True)
    for u, v, d in edges:
        weight = d["weight"]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            alpha=np.clip(weight, 0.1, 1.0),  # clip alpha to [0.1,1]
            width=2,
            ax=ax,
        )

    labels = nx.get_node_attributes(G, "label")

    wrapped_labels = {k: textwrap.fill(label, width=20) for k, label in labels.items()}
    label_pos = {k: (v[0] + 0.05, v[1]) for k, v in pos.items()}  # Add offset to x-coordinate
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=wrapped_labels,
        font_size=7,
        verticalalignment="top",
        horizontalalignment="left",
        ax=ax,
    )
    ax.text(
        1.0,
        -0.15,
        f"Theme similarity network with {method} layout" f" (threshold={threshold})",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
    )
    ax.axis("off")
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Save plot
    buffer = BytesIO()
    fig.savefig(buffer, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)
    return Base64ImageFile(buffer, name="similarity_plot.png")


@task(persist_result=True, cache_policy=INPUTS)
def create_pairwise_heatmap(
    a: QualitativeAnalysis, b: QualitativeAnalysis, threshold=0.6, use_threshold=True
) -> str:
    """Create a heatmap visualization for a single pair of pipeline results."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    def truncate_theme(theme: str, max_len: int = 25) -> str:  # Reduced from 40
        if len(theme) <= max_len:
            return theme
        return theme[: max_len - 3] + "..."

    themes_a = [i.name for i in a.themes]
    themes_b = [i.name for i in b.themes]
    themes_a_display = [truncate_theme(t) for t in themes_a]
    themes_b_display = [truncate_theme(t) for t in themes_b]

    # Better figure sizing accounting for label length
    avg_label_len_b = np.mean([len(label) for label in themes_b_display])
    fig_height = max(8, len(themes_a) * 0.4)
    fig_width = max(12, len(themes_b) * 0.5 + avg_label_len_b * 0.1)  # Account for label width

    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    comparison = compare_result_similarity(
        a,
        b,
        threshold=threshold or 0.5,  # ensure not None
    )

    similarity_matrix = comparison["similarity_matrix"]
    df_sim = pd.DataFrame(similarity_matrix, index=themes_a_display, columns=themes_b_display)

    assert similarity_matrix.shape == (
        len(themes_a_display),
        len(themes_b_display),
    ), f"Shape mismatch: {similarity_matrix.shape} vs {len(themes_a_display)} x {len(themes_b_display)}"

    if use_threshold:
        df_binary = (df_sim >= threshold).astype(int)
        cmap = LinearSegmentedColormap.from_list("threshold_cmap", ["white", "green"], N=2)
        # raise Exception(df_sim, threshold, df_binary)
        data = df_binary
        annot = False
        vmin = 0  # Explicitly set minimum
        vmax = 1  # Explicitly set maximum
    else:
        data = df_sim
        cmap = "viridis"
        annot = True
        vmin = None  # Let seaborn auto-scale
        vmax = None

    # Create heatmap with better spacing
    sns.heatmap(
        data,
        annot=annot,
        fmt=".2f" if annot else None,
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={"label": "Match" if use_threshold else "Cosine Similarity"},
        ax=ax,
        square=False,  # Don't force square aspect ratio
        vmin=vmin,  # Add explicit scale limits
        vmax=vmax,  # Add explicit scale limits
    )

    # TODO: set nicer titles

    ax.set_title(f"Theme Similarity Matrix\n{a.name()} vs {b.name()}. Threshold: {threshold}")
    ax.set_xlabel(b.name())
    ax.set_ylabel(a.name())

    # Better tick label handling
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    ax.set_aspect("equal")

    # Ensure labels are properly positioned
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    # Use tight_layout with padding
    fig.tight_layout(pad=2.0)

    # Additional spacing adjustment if needed
    plt.subplots_adjust(bottom=0.2)  # Add extra space at bottom for rotated labels

    suffix = threshold and f"_threshold={threshold}" or ""
    plot_name = f"heatmap_{a.name()}_{b.name()}{suffix}.png"
    buffer = BytesIO()
    fig.savefig(buffer, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    buffer.seek(0)

    return Base64ImageFile(buffer, name=plot_name)


class SimilarityComparator:
    """Comparator calculates similarity statistics and makes plot/heatmaps."""

    @flow(persist_result=False)
    def compare(self, pipeline_results: List[QualitativeAnalysis], config={}):

        threshold = config.get("threshold", 0.6)
        n_neighbors = config.get("n_neighbors", 5)
        min_dist = config.get("min_dist", 0.01)
        method = config.get("method", "umap")

        result_combinations = list(itertools.combinations(pipeline_results, 2))

        similarity_results = [
            compare_result_similarity.submit(
                i,
                j,
                threshold=threshold,
            )
            for i, j in result_combinations
        ]

        heatmaps = [
            create_pairwise_heatmap.submit(a, b, threshold=threshold, use_threshold=False)
            for a, b in result_combinations
        ]

        thresholded_heatmaps = [
            create_pairwise_heatmap.submit(a, b, threshold=threshold)
            for a, b in result_combinations
        ]

        network_plot = network_similarity_plot(
            pipeline_results,
            method=method,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            threshold=threshold,
        )

        result_combinations_dict = OrderedDict(
            {i.name() + "_" + j.name(): (i, j) for i, j in result_combinations}
        )

        stats_dict = {
            k: v for k, v in zip(result_combinations_dict.keys(), resolve(similarity_results))
        }

        heatmap_dict = {k: v for k, v in zip(result_combinations_dict.keys(), resolve(heatmaps))}

        thresh_heatmap_dict = {
            k: v for k, v in zip(result_combinations_dict.keys(), resolve(thresholded_heatmaps))
        }

        return QualitativeAnalysisComparison(
            results=pipeline_results,
            combinations=result_combinations_dict,
            statistics=stats_dict,
            comparison_plots={
                "heatmaps": heatmap_dict,
                "thresholded_heatmaps": thresh_heatmap_dict,
            },
            additional_plots={
                "network_plot": network_plot,
            },
            config=config,
        )


if False:
    from wellspring.models import Analysis

    pipeline_results = [
        QualitativeAnalysis(**j)
        for j in [i.result_json for i in Analysis.objects.filter(result_json__isnull=False)][-6:]
        if isinstance(j, dict)
    ]
    pipeline_results[0].name()

    x = list(reversed(pipeline_results))
    comp = SimilarityComparator().compare(pipeline_results)
