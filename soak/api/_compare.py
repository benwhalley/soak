"""Compare implementation for soak API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ._credentials import get_credentials
from ._results import CompareResult, RunResult

if TYPE_CHECKING:
    from ..models import QualitativeAnalysis


class CompareError(Exception):
    """Error during comparison."""

    pass


def _print_comparison_stats(
    result: dict,
    name_a: str,
    name_b: str,
    list_a: list,
    list_b: list,
    threshold: float,
    embedding_model: str,
    shepard_k: float,
    ot_k_values: Optional[list] = None,
    similarity: str = "angular",
) -> str:
    """Generate text output for comparison statistics."""
    from ..helpers import print_comparison_stats

    return print_comparison_stats(
        result,
        name_a,
        name_b,
        list_a,
        list_b,
        threshold,
        embedding_model,
        shepard_k,
        ot_k_values or [],
        similarity,
    )


def compare(
    *analyses: Union[str, Path, "QualitativeAnalysis"],
    threshold: float = 0.6,
    embedding_model: str = "text-embedding-3-large",
    similarity: str = "angular",
    shepard_k: float = 1.0,
    ot_k: float = 0.25,
    embedding_template: Optional[str] = None,
    calibration: Optional[Union[str, Path]] = None,
    compute_paraphrase_bound: bool = True,
    n_paraphrases: int = 7,
    cwd: Optional[Path] = None,
) -> CompareResult:
    """Compare two or more analyses.

    Args:
        *analyses: JSON file paths, directory paths, or QualitativeAnalysis objects
        threshold: Similarity threshold for matching themes
        embedding_model: Model for embeddings (use 'local/model-name' for local)
        similarity: Metric: "angular", "cosine", or "shepard"
        shepard_k: Shepard similarity decay parameter
        ot_k: Optimal transport mass penalty (lower = more selective)
        embedding_template: Template for theme text before embedding
        calibration: Path to calibration folder
        compute_paraphrase_bound: Compute paraphrase-based upper bound
        n_paraphrases: Number of paraphrases for upper bound
        cwd: Working directory for resolving paths

    Returns:
        CompareResult with .by_comparisons(), .to_html(), .to_text()

    Raises:
        CompareError: If comparison fails
    """
    from ..cli._common import resolve_analysis_path
    from ..comparators.similarity_comparator import SimilarityComparator
    from ..models import QualitativeAnalysis, QualitativeAnalysisPipeline

    cwd = cwd or Path.cwd()

    if len(analyses) < 2:
        raise CompareError("At least 2 analyses required for comparison")

    # check credentials for API embedding models
    if not embedding_model.startswith("local/"):
        get_credentials(cwd)

    # resolve calibration path
    calibration_path = None
    if calibration:
        cal_path = Path(calibration)
        if not cal_path.exists():
            raise CompareError(f"Calibration path not found: {calibration}")
        if cal_path.is_dir():
            yaml_path = cal_path / "calibration.yaml"
            json_path = cal_path / "calibration.json"
            if yaml_path.exists():
                calibration_path = yaml_path
            elif json_path.exists():
                calibration_path = json_path
            else:
                raise CompareError(f"calibration.yaml not found in: {calibration}")
        else:
            calibration_path = cal_path
    else:
        # try auto-detection
        from ..calibration import get_bundled_calibration

        bundled_path = get_bundled_calibration(embedding_model)
        if bundled_path:
            calibration_path = bundled_path

    # load analyses
    loaded_analyses = []
    for analysis in analyses:
        if isinstance(analysis, (str, Path)):
            input_path = resolve_analysis_path(str(analysis))
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "nodes" in data:
                pipeline = QualitativeAnalysisPipeline.model_validate(data)
                loaded = pipeline.result()
                loaded.name = input_path.stem
            else:
                loaded = QualitativeAnalysis.model_validate(data)
                if not loaded.name or loaded.name == loaded.sha256()[:8]:
                    loaded.name = input_path.stem
            loaded_analyses.append(loaded)
        elif isinstance(analysis, RunResult):
            # extract QualitativeAnalysis from RunResult
            loaded = analysis.analysis
            if not loaded.name or loaded.name == loaded.sha256()[:8]:
                loaded.name = analysis.pipeline.name or f"analysis_{len(loaded_analyses)}"
            loaded_analyses.append(loaded)
        else:
            # assume QualitativeAnalysis
            loaded_analyses.append(analysis)

    # default embedding template for JSON mode
    effective_embedding_template = embedding_template or "{name}: {description}"

    # run comparison
    comparator = SimilarityComparator()
    comparison = comparator.compare(
        loaded_analyses,
        config={
            "threshold": threshold,
            "method": "umap",
            "n_neighbors": 5,
            "min_dist": 0.01,
            "label_template": "{name}",
            "embedding_template": effective_embedding_template,
            "embedding_model": embedding_model,
            "k": shepard_k,
            "reg_m": ot_k,
            "distance": similarity,
            "compute_paraphrase_bound": compute_paraphrase_bound,
            "n_paraphrases": n_paraphrases,
            "calibration_path": str(calibration_path) if calibration_path else None,
        },
    )

    # generate stats text for each pairwise comparison
    stats_text = []
    for key, comp in comparison.by_comparisons().items():
        stats = comp["stats"]
        analysis_a, analysis_b = comp["a"], comp["b"]
        list_a = [t.name for t in analysis_a.themes]
        list_b = [t.name for t in analysis_b.themes]

        text = _print_comparison_stats(
            stats,
            name_a=analysis_a.name,
            name_b=analysis_b.name,
            list_a=list_a,
            list_b=list_b,
            threshold=threshold,
            embedding_model=embedding_model,
            shepard_k=shepard_k,
            similarity=similarity,
        )
        stats_text.append(text)

    return CompareResult(comparison=comparison, stats_text=stats_text)


def compare_strings(
    columns: dict[str, list[str]],
    *,
    threshold: float = 0.6,
    embedding_model: str = "text-embedding-3-large",
    similarity: str = "angular",
    shepard_k: float = 1.0,
    ot_k: float = 0.25,
    embedding_template: str = "{name}",
    calibration: Optional[Union[str, Path]] = None,
    compute_paraphrase_bound: bool = True,
    n_paraphrases: int = 7,
    cwd: Optional[Path] = None,
) -> CompareResult:
    """Compare lists of strings (e.g., theme names from different sources).

    Args:
        columns: Dict mapping column names to lists of strings
                 e.g., {"Method A": [...], "Method B": [...]}
        threshold: Similarity threshold for matching
        embedding_model: Model for embeddings
        similarity: Metric: "angular", "cosine", or "shepard"
        shepard_k: Shepard similarity decay parameter
        ot_k: Optimal transport mass penalty
        embedding_template: Template for string text before embedding
        calibration: Path to calibration folder
        compute_paraphrase_bound: Compute paraphrase-based upper bound
        n_paraphrases: Number of paraphrases for upper bound
        cwd: Working directory

    Returns:
        CompareResult
    """
    from ..comparators.similarity_comparator import SimilarityComparator
    from ..models import QualitativeAnalysis, Theme

    cwd = cwd or Path.cwd()

    if len(columns) < 2:
        raise CompareError("At least 2 columns required for comparison")

    # check credentials for API embedding models
    if not embedding_model.startswith("local/"):
        get_credentials(cwd)

    # resolve calibration path
    calibration_path = None
    if calibration:
        cal_path = Path(calibration)
        if not cal_path.exists():
            raise CompareError(f"Calibration path not found: {calibration}")
        if cal_path.is_dir():
            yaml_path = cal_path / "calibration.yaml"
            json_path = cal_path / "calibration.json"
            if yaml_path.exists():
                calibration_path = yaml_path
            elif json_path.exists():
                calibration_path = json_path
            else:
                raise CompareError(f"calibration.yaml not found in: {calibration}")
        else:
            calibration_path = cal_path
    else:
        from ..calibration import get_bundled_calibration

        bundled_path = get_bundled_calibration(embedding_model)
        if bundled_path:
            calibration_path = bundled_path

    # create QualitativeAnalysis objects for each column
    analyses = []
    for col_name, items in columns.items():
        themes = [Theme(name=s, description=s, code_slugs=[]) for s in items]
        analysis = QualitativeAnalysis(name=col_name, themes=themes)
        analyses.append(analysis)

    # run comparison
    comparator = SimilarityComparator()
    comparison = comparator.compare(
        analyses,
        config={
            "threshold": threshold,
            "method": "umap",
            "n_neighbors": 5,
            "min_dist": 0.01,
            "label_template": "{name}",
            "embedding_template": embedding_template,
            "embedding_model": embedding_model,
            "k": shepard_k,
            "reg_m": ot_k,
            "distance": similarity,
            "compute_paraphrase_bound": compute_paraphrase_bound,
            "n_paraphrases": n_paraphrases,
            "calibration_path": str(calibration_path) if calibration_path else None,
        },
    )

    # generate stats text
    stats_text = []
    for key, comp in comparison.by_comparisons().items():
        stats = comp["stats"]
        analysis_a, analysis_b = comp["a"], comp["b"]
        list_a = [t.name for t in analysis_a.themes]
        list_b = [t.name for t in analysis_b.themes]

        text = _print_comparison_stats(
            stats,
            name_a=analysis_a.name,
            name_b=analysis_b.name,
            list_a=list_a,
            list_b=list_b,
            threshold=threshold,
            embedding_model=embedding_model,
            shepard_k=shepard_k,
            similarity=similarity,
        )
        stats_text.append(text)

    return CompareResult(comparison=comparison, stats_text=stats_text)
