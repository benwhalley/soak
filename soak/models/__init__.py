"""
Data models and pipeline execution for qualitative analysis.

This module has been refactored from a monolithic 4,032-line file into a modular package.
All imports remain backward compatible.
"""

# Base models and utilities
from .base import (
    # Exceptions
    Cancelled,
    CancelledRun,
    # Base models
    Code,
    CodeList,
    CodeSlugStr,
    Document,
    QualitativeAnalysis,
    QualitativeAnalysisComparison,
    Theme,
    Themes,
    TrackedItem,
    # Utility functions
    extract_content,
    extract_prompt,
    get_action_lookup,
    get_default_llm_credentials,
    get_embedding,
    MAX_CONCURRENCY,
    memory,
    safe_json_dump,
    semaphore,
    SOAK_MAX_RUNTIME,
)

# DAG execution
from .dag import (
    DAG,
    DAGConfig,
    DAGNodeUnion,
    Edge,
    get_template_variables,
    OutputUnion,
    render_strict_template,
    run_node,
)

# All node types
from .nodes import (
    Batch,
    BatchList,
    Classifier,
    CompletionDAGNode,
    DAGNode,
    Filter,
    ItemsNode,
    Map,
    Reduce,
    Split,
    Transform,
    TransformReduce,
    VerifyQuotes,
    default_map_task,
)

# Pipeline
from .pipeline import QualitativeAnalysisPipeline

# Rebuild models to resolve forward references to node types
DAG.model_rebuild()
QualitativeAnalysisPipeline.model_rebuild()

# Export everything for backward compatibility
__all__ = [
    # Exceptions
    "Cancelled",
    "CancelledRun",
    # Base models
    "Code",
    "CodeList",
    "CodeSlugStr",
    "Document",
    "QualitativeAnalysis",
    "QualitativeAnalysisComparison",
    "Theme",
    "Themes",
    "TrackedItem",
    # DAG
    "DAG",
    "DAGConfig",
    "DAGNode",
    "DAGNodeUnion",
    "Edge",
    "OutputUnion",
    # Nodes
    "Batch",
    "BatchList",
    "Classifier",
    "CompletionDAGNode",
    "Filter",
    "ItemsNode",
    "Map",
    "Reduce",
    "Split",
    "Transform",
    "TransformReduce",
    "VerifyQuotes",
    # Pipeline
    "QualitativeAnalysisPipeline",
    # Functions
    "default_map_task",
    "extract_content",
    "extract_prompt",
    "get_action_lookup",
    "get_default_llm_credentials",
    "get_embedding",
    "get_template_variables",
    "render_strict_template",
    "run_node",
    "safe_json_dump",
    # Constants
    "MAX_CONCURRENCY",
    "SOAK_MAX_RUNTIME",
    "memory",
    "semaphore",
]
