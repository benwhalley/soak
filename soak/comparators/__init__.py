"""Comparison utilities for analysis results.

This package provides tools for comparing thematic analyses using
embedding similarity and optimal transport.

Submodules:
- rescaling: Similarity matrix rescaling methods
- optimal_transport: Optimal transport computation
- baselines: Baseline generation (word-salad, permutation)
- paraphrasing: LLM-based paraphrase generation
- visualizations: Visualization functions (Sankey, heatmaps, etc.)
- utils: Utility functions
- similarity_comparator: Main comparison functions and SimilarityComparator class
"""

from .similarity_comparator import (
    # Types
    RescaleMethod,
    # Rescaling
    rescale_similarity,
    # Optimal Transport
    compute_ot,
    compute_split_join_stats,
    filter_transport_plan,
    compute_best_matches_for_k,
    hungarian_matching,
    # Baselines
    generate_word_salad_texts,
    compute_permutation_baseline,
    # Paraphrasing
    generate_paraphrase_texts,
    generate_short_labels,
    prepare_paraphrase_cost_matrix,
    compute_paraphrase_ot_at_k,
    compute_paraphrase_baseline,
    # Visualizations
    SankeyHTML,
    Base64ImageFile,
    create_transport_sankey,
    create_transport_heatmap,
    find_elbow_points,
    # Utils
    create_embeddings_csv_base64,
    format_similarity_matrix,
    # Main functions
    create_shared_mass_scree_plot,
    create_alignment_scree_plot,
    create_splits_joins_scree_plot,
    compare_result_similarity,
    network_similarity_plot,
    create_pairwise_heatmap,
    # Main class
    SimilarityComparator,
)

__all__ = [
    "RescaleMethod",
    "rescale_similarity",
    "compute_ot",
    "compute_split_join_stats",
    "filter_transport_plan",
    "compute_best_matches_for_k",
    "hungarian_matching",
    "generate_word_salad_texts",
    "compute_permutation_baseline",
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
