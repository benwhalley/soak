"""Base classes for comparing analysis results."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from soak.models import QualitativeAnalysis, QualitativeAnalysisComparison


@dataclass
class ComparatorConfig:
    """Configuration for comparators."""

    output_dir: str = "output"
    model_name: str = "text-embedding-3-large"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    threshold: float = 0.6
    n_neighbors: int = 5
    min_dist: float = 0.1
    method: str = "umap"  # umap, mds, or pca
    baseline_files: Optional[List[str]] = None
    baseline_themes: Optional[Dict[str, List[str]]] = None  # Inline themes
    cache_dir: str = ".cache/embeddings"
    cache_enabled: bool = True
    cache_verbose: bool = False
    debug: bool = False  # Enable LiteLLM debug mode


class Comparator(ABC):
    """Abstract base class for pipeline result comparators."""

    def __init__(
        self,
        config: Optional[ComparatorConfig] = None,
        execution_id: Optional[str] = None,
    ):
        self.config = config or ComparatorConfig()
        self.execution_id = execution_id or str(uuid.uuid4())[:8]

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def compare(
        self, pipeline_results: List[QualitativeAnalysis]
    ) -> QualitativeAnalysisComparison:
        """Compare pipeline results and generate output."""
        pass
