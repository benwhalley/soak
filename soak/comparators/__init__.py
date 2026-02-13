"""Comparison utilities for analysis results.

This package provides tools for comparing thematic analyses using
embedding similarity and optimal transport.

Submodules:
- rescaling: Similarity matrix calibration
- optimal_transport: Optimal transport computation
- baselines: Baseline generation (word-salad)
- paraphrasing: LLM-based paraphrase generation
- visualizations: Visualization functions (Sankey, heatmaps, etc.)
- utils: Utility functions
- similarity_comparator: Main comparison functions and SimilarityComparator class
"""

from .similarity_comparator import (  # Calibration; Optimal Transport; Baselines; Paraphrasing; Visualizations; Utils; Main functions; Main class
    Base64ImageFile, SankeyHTML, SimilarityComparator, apply_calibration,
    compare_result_similarity, compute_best_matches_for_k, compute_ot,
    compute_paraphrase_baseline, compute_paraphrase_ot_at_k,
    compute_split_join_stats, create_alignment_scree_plot,
    create_embeddings_csv_base64, create_pairwise_heatmap,
    create_shared_mass_scree_plot, create_splits_joins_scree_plot,
    create_transport_heatmap, create_transport_sankey, filter_transport_plan,
    find_elbow_points, format_similarity_matrix, generate_paraphrase_texts,
    generate_short_labels, generate_word_salad_texts, hungarian_matching,
    network_similarity_plot, prepare_paraphrase_cost_matrix)

__all__ = [
    "apply_calibration",
    "compute_ot",
    "compute_split_join_stats",
    "filter_transport_plan",
    "compute_best_matches_for_k",
    "hungarian_matching",
    "generate_word_salad_texts",
    "generate_paraphrase_texts",
    "generate_short_labels",
    "prepare_paraphrase_cost_matrix",
    "compute_paraphrase_ot_at_k",
    "compute_paraphrase_baseline",
    "SankeyHTML",
    "Base64ImageFile",
    "create_transport_sankey",
    "create_transport_heatmap",
    "find_elbow_points",
    "create_embeddings_csv_base64",
    "format_similarity_matrix",
    "create_shared_mass_scree_plot",
    "create_alignment_scree_plot",
    "create_splits_joins_scree_plot",
    "compare_result_similarity",
    "network_similarity_plot",
    "create_pairwise_heatmap",
    "SimilarityComparator",
]
