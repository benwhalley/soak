"""Coverage implementation for soak API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._credentials import get_credentials
from ._results import CoverageResult


class CoverageError(Exception):
    """Error during coverage analysis."""

    pass


def coverage(
    analysis: Union[str, Path, "QualitativeAnalysis"],
    documents: Optional[Union[str, list[Union[str, Path]]]] = None,
    *,
    output: Optional[Union[str, Path]] = None,
    groups: Optional[Union[str, Path]] = None,
    chunk_size: int = 500,
    overlap: int = 50,
    split_unit: str = "words",
    aggregation: str = "max",
    embedding_model: str = "text-embedding-3-large",
    embedding_template: str = "{name}: {description}",
    embed_source: str = "quotes",
    threshold: float = 0.75,
    calibration: Optional[Union[str, Path]] = None,
    cwd: Optional[Path] = None,
) -> CoverageResult:
    """Analyze theme coverage across documents.

    Args:
        analysis: JSON file path or QualitativeAnalysis object
        documents: Override documents (default: use documents from analysis)
        output: Output folder (writes files if provided)
        groups: XLSX file with 'filename' and 'group' columns for grouping
        chunk_size: Size of document chunks
        overlap: Overlap between chunks
        split_unit: Unit for chunking: words, tokens, or chars
        aggregation: Aggregation method: max, mean, or p95
        embedding_model: Model for embeddings
        embedding_template: Template for theme text before embedding
        embed_source: What to embed for themes: quotes, themes, or both
        threshold: Threshold for heatmap colouring
        calibration: Path to calibration folder
        cwd: Working directory

    Returns:
        CoverageResult with coverage data and visualization methods

    Raises:
        CoverageError: If coverage analysis fails
    """
    import pandas as pd

    from ..coverage import ThemeCoverageAnalyzer
    from ..coverage.analyzer import (
        compute_within_doc_variation,
        generate_absolute_chunk_heatmap,
        generate_chunk_heatmap,
        generate_coverage_heatmap,
        generate_group_heatmap,
        generate_normalized_chunk_heatmap,
        generate_theme_trajectories,
    )
    from ..document_utils import extract_text, unpack_zip_to_temp_paths_if_needed
    from ..models import QualitativeAnalysis, QualitativeAnalysisPipeline
    from ..models.base import TrackedItem

    cwd = cwd or Path.cwd()

    # check credentials for API embedding models
    if not embedding_model.startswith("local/"):
        get_credentials(cwd)

    # resolve calibration path
    calibration_path = None
    if calibration:
        cal_path = Path(calibration)
        if not cal_path.exists():
            raise CoverageError(f"Calibration path not found: {calibration}")
        if cal_path.is_dir():
            yaml_path = cal_path / "calibration.yaml"
            if yaml_path.exists():
                calibration_path = yaml_path
            else:
                raise CoverageError(f"calibration.yaml not found in: {calibration}")
        else:
            calibration_path = cal_path

    # load analysis
    if isinstance(analysis, (str, Path)):
        analysis_path = Path(analysis)
        if not analysis_path.exists():
            raise CoverageError(f"Analysis file not found: {analysis}")

        with open(analysis_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "nodes" in data:
            pipeline = QualitativeAnalysisPipeline.model_validate(data)
            qa = pipeline.result()
            if not qa:
                raise CoverageError("Pipeline has no result")
        else:
            qa = QualitativeAnalysis.model_validate(data)
    else:
        qa = analysis
        data = {}

    if not qa.themes:
        raise CoverageError("Analysis has no themes")

    # load documents
    doc_items = []

    if documents:
        # use documents specified via parameter
        if isinstance(documents, str):
            from glob import glob

            doc_paths = [Path(p) for p in glob(documents)]
        else:
            doc_paths = [Path(p) for p in documents]

        with unpack_zip_to_temp_paths_if_needed(doc_paths) as docfiles:
            if not docfiles:
                raise CoverageError(f"No files found: {documents}")

            for docpath, doc_metadata in docfiles:
                content = extract_text(str(docpath))
                if isinstance(content, list):
                    continue  # skip spreadsheets

                doc_id = Path(docpath).stem
                doc_items.append(
                    TrackedItem(
                        content=content,
                        id=doc_id,
                        sources=[doc_id],
                        metadata={
                            "filename": Path(docpath).name,
                            "original_path": str(docpath),
                            **doc_metadata,
                        },
                    )
                )
    else:
        # try to get documents from analysis config
        config = data.get("config", {})

        embedded_docs = config.get("documents", [])
        if embedded_docs:
            for doc in embedded_docs:
                if isinstance(doc, dict):
                    doc_items.append(
                        TrackedItem(
                            content=doc.get("content", ""),
                            id=doc.get("id", "unknown"),
                            sources=[doc.get("id", "unknown")],
                            metadata=doc.get("metadata", {}),
                        )
                    )
        else:
            doc_paths = config.get("document_paths", [])
            if doc_paths:
                for path_entry in doc_paths:
                    if isinstance(path_entry, (list, tuple)):
                        docpath = path_entry[0]
                        doc_metadata = path_entry[1] if len(path_entry) > 1 else {}
                    else:
                        docpath = path_entry
                        doc_metadata = {}

                    if not Path(docpath).exists():
                        continue

                    content = extract_text(str(docpath))
                    if isinstance(content, list):
                        continue

                    doc_id = Path(docpath).stem
                    doc_items.append(
                        TrackedItem(
                            content=content,
                            id=doc_id,
                            sources=[doc_id],
                            metadata={
                                "filename": Path(docpath).name,
                                "original_path": str(docpath),
                                **(doc_metadata or {}),
                            },
                        )
                    )

    if not doc_items:
        raise CoverageError(
            "No documents found. Specify documents parameter or ensure analysis contains document paths."
        )

    # load groups if provided
    groups_dict = None
    if groups:
        groups_path = Path(groups)
        if not groups_path.exists():
            raise CoverageError(f"Groups file not found: {groups}")

        groups_df = pd.read_excel(groups_path)
        if "filename" not in groups_df.columns or "group" not in groups_df.columns:
            raise CoverageError("Groups file must have 'filename' and 'group' columns")

        groups_dict = dict(
            zip(groups_df["filename"].astype(str), groups_df["group"].astype(str))
        )

    # run analysis
    analyzer = ThemeCoverageAnalyzer(
        chunk_size=chunk_size,
        overlap=overlap,
        split_unit=split_unit,
        aggregation=aggregation,
        embedding_template=embedding_template,
        embedding_model=embedding_model,
        embed_source=embed_source,
    )

    result = analyzer.analyze(qa, doc_items, groups=groups_dict)

    # apply calibration if provided
    if calibration_path:
        from ..calibration import calibrate, load_calibration

        model, cal_metadata = load_calibration(calibration_path, embedding_model)
        method = cal_metadata.get("method", "gam")

        chunk_sim = np.array(result.chunk_similarity_matrix)
        calibrated_sim = calibrate(chunk_sim, model, method=method)
        result.chunk_similarity_matrix = calibrated_sim.tolist()

        # recompute document-level aggregations
        from collections import defaultdict

        doc_chunks = defaultdict(list)
        for chunk_idx, info in enumerate(result.chunk_info):
            doc_chunks[info.document_id].append(chunk_idx)

        for doc_coverage in result.documents:
            doc_id = doc_coverage.document_id
            chunk_indices = doc_chunks[doc_id]
            chunk_sims = calibrated_sim[chunk_indices]

            doc_scores = {}
            for theme_idx, theme_name in enumerate(result.theme_names):
                theme_sims = chunk_sims[:, theme_idx]
                if aggregation == "max":
                    score = float(np.max(theme_sims))
                elif aggregation == "mean":
                    score = float(np.mean(theme_sims))
                elif aggregation == "p95":
                    score = float(np.percentile(theme_sims, 95))
                else:
                    score = float(np.max(theme_sims))
                doc_scores[theme_name] = score

            doc_coverage.theme_scores = doc_scores

        result.config["calibration"] = {
            "path": str(calibration_path),
            "method": method,
        }

    # generate heatmaps
    heatmaps = {
        "heatmap": generate_coverage_heatmap(result),
        "heatmap_thresholded": generate_coverage_heatmap(result, threshold=threshold),
        "group_heatmap": generate_group_heatmap(result) if result.groups else None,
        "chunk_heatmap": generate_chunk_heatmap(result),
        "normalized_heatmap": generate_normalized_chunk_heatmap(result),
        "normalized_heatmap_zscore": generate_normalized_chunk_heatmap(
            result, z_score=True
        ),
        "absolute_heatmap": generate_absolute_chunk_heatmap(result),
        "theme_trajectories": generate_theme_trajectories(result),
        "within_doc_variation": compute_within_doc_variation(result),
    }

    # write output if specified
    if output:
        output_folder = Path(output)
        output_folder.mkdir(parents=True, exist_ok=True)

        json_path = output_folder / "coverage.json"
        json_path.write_text(result.model_dump_json(indent=2))

        csv_path = output_folder / "coverage.csv"
        csv_path.write_text(result.to_csv())

    return CoverageResult(result=result, heatmaps=heatmaps)
