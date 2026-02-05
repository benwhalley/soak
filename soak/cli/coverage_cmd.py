"""Coverage command for analyzing theme coverage across documents."""

import base64
import json
import logging
import pdb
import traceback
from pathlib import Path

import numpy as np
import typer

from ._common import (
    TEMPLATES_DIR,
    check_and_prompt_credentials,
    get_pdb_on_exception,
)

logger = logging.getLogger(__name__)


def coverage(
    analysis_file: str = typer.Argument(
        ..., help="Path to JSON file containing QualitativeAnalysis or Pipeline results"
    ),
    output: str = typer.Option(
        "coverage", "--output", "-o", help="Output folder name"
    ),
    documents: list[Path] = typer.Option(
        None,
        "--documents",
        "-d",
        help="Override documents (default: use documents from analysis)",
    ),
    groups_file: Path = typer.Option(
        None,
        "--groups",
        "-g",
        help="XLSX file with 'filename' and 'group' columns for grouping",
    ),
    format: str = typer.Option(
        "all", "--format", "-f", help="Output format: json, html, csv, or all"
    ),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Size of document chunks"),
    overlap_size: int = typer.Option(50, "--overlap", help="Overlap between chunks"),
    split_unit: str = typer.Option(
        "words", "--split-unit", help="Unit for chunking: words, tokens, or chars"
    ),
    aggregation: str = typer.Option(
        "max", "--aggregation", "-a", help="Aggregation method: max, mean, or p95"
    ),
    embedding_template: str = typer.Option(
        "{name}: {description}",
        "--embedding-template",
        "-e",
        envvar="SOAK_EMBEDDING_TEMPLATE",
        help="Template for theme text before embedding",
    ),
    embedding_model: str = typer.Option(
        "text-embedding-3-large",
        "--embedding-model",
        envvar="SOAK_EMBEDDING_MODEL",
        help="Embedding model (use 'local/model-name' for sentence-transformers, e.g., 'local/all-MiniLM-L6-v2')",
    ),
    sample_frac: float = typer.Option(
        None,
        "--sample-frac",
        help="Sample fraction of chunks (e.g., 0.1 for 10%)",
    ),
    sample_n: int = typer.Option(
        None,
        "--sample-n",
        help="Sample fixed number of chunks",
    ),
    threshold: float = typer.Option(
        0.75,
        "--threshold",
        "-t",
        help="Threshold for heatmap colouring (values below shown grey)",
    ),
    embed: str = typer.Option(
        "quotes",
        "--embed",
        help="What to embed for themes: 'quotes' (default), 'themes', or 'both'",
    ),
    chunk_window: int = typer.Option(
        10,
        "--chunk-window",
        help="Window size for chunk-level heatmap aggregation (default: 10)",
    ),
    n_bins: int = typer.Option(
        20,
        "--n-bins",
        help="Number of bins for normalized chunk heatmaps (default: 20)",
    ),
    calibration: Path = typer.Option(
        None,
        "--calibration",
        help="Path to calibration folder or file. If folder, looks for calibration.yaml inside.",
    ),
):
    """Analyze how well themes from an analysis are represented across documents.

    By default uses documents from the analysis JSON. Override with --documents.

    Computes embedding-based cosine similarity between theme descriptions and
    document chunks, then aggregates to document level.

    Examples:
        soak coverage analysis.json -o coverage
        soak coverage analysis.json --documents new_data/*.txt
        soak coverage analysis.json --groups metadata.xlsx
    """
    import pandas as pd
    from jinja2 import Environment, FileSystemLoader

    from ..coverage import ThemeCoverageAnalyzer
    from ..coverage.analyzer import (compute_within_doc_variation,
                                     generate_absolute_chunk_heatmap,
                                     generate_chunk_heatmap,
                                     generate_coverage_heatmap,
                                     generate_group_heatmap,
                                     generate_normalized_chunk_heatmap,
                                     generate_theme_trajectories)
    from ..document_utils import unpack_zip_to_temp_paths_if_needed
    from ..helpers import format_exception_concise

    # validate format
    valid_formats = {"json", "html", "csv", "all"}
    if format not in valid_formats:
        logger.error(f"Invalid format '{format}'. Must be one of: {valid_formats}")
        raise typer.Exit(1)

    # validate split_unit
    valid_split_units = {"words", "tokens", "chars"}
    if split_unit not in valid_split_units:
        logger.error(
            f"Invalid split-unit '{split_unit}'. Must be one of: {valid_split_units}"
        )
        raise typer.Exit(1)

    # validate aggregation
    valid_aggregations = {"max", "mean", "p95"}
    if aggregation not in valid_aggregations:
        logger.error(
            f"Invalid aggregation '{aggregation}'. Must be one of: {valid_aggregations}"
        )
        raise typer.Exit(1)

    # validate embed
    valid_embed_sources = {"quotes", "themes", "both"}
    if embed not in valid_embed_sources:
        logger.error(
            f"Invalid embed source '{embed}'. Must be one of: {valid_embed_sources}"
        )
        raise typer.Exit(1)

    # check for API credentials if using api backend
    if not embedding_model.startswith("local/"):
        check_and_prompt_credentials(Path.cwd())

    # handle calibration option - resolve folder to calibration.yaml/.json
    calibration_path = None
    if calibration:
        if not calibration.exists():
            logger.error(f"Calibration path not found: {calibration}")
            raise typer.Exit(1)
        if calibration.is_dir():
            yaml_path = calibration / "calibration.yaml"
            json_path = calibration / "calibration.json"
            if yaml_path.exists():
                calibration_path = yaml_path
            elif json_path.exists():
                calibration_path = json_path
            else:
                logger.error(f"calibration.yaml not found in folder: {calibration}")
                raise typer.Exit(1)
        else:
            calibration_path = calibration
        logger.info(f"Using calibration from {calibration_path}")

    # load analysis JSON
    analysis_path = Path(analysis_file)
    if not analysis_path.exists():
        logger.error(f"Analysis file not found: {analysis_file}")
        raise typer.Exit(1)

    logger.info(f"Loading analysis from {analysis_file}...")
    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # extract QualitativeAnalysis from pipeline or direct
    from ..models import QualitativeAnalysis, QualitativeAnalysisPipeline

    try:
        if "nodes" in data:
            # this is a pipeline
            pipeline = QualitativeAnalysisPipeline.model_validate(data)
            analysis = pipeline.result()
            if not analysis:
                logger.error("Pipeline has no result. Run the pipeline first.")
                raise typer.Exit(1)
        else:
            # direct QualitativeAnalysis
            analysis = QualitativeAnalysis.model_validate(data)
    except Exception as e:
        if get_pdb_on_exception():
            traceback.print_exc()
            pdb.post_mortem()
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Error loading analysis:\n{error_msg}")

    if not analysis.themes:
        logger.error("Analysis has no themes. Cannot compute coverage.")
        raise typer.Exit(1)

    logger.info(f"Loaded analysis with {len(analysis.themes)} themes")

    # load documents - from CLI, or from analysis config
    from ..document_utils import extract_text
    from ..models.base import TrackedItem

    doc_items = []

    if documents:
        # use documents specified via --documents option
        logger.info("Using documents specified via --documents")
        try:
            with unpack_zip_to_temp_paths_if_needed(documents) as docfiles:
                if not docfiles:
                    logger.error(
                        f"No files found matching input patterns: {', '.join(map(str, documents))}"
                    )
                    raise typer.Exit(1)

                for docpath, doc_metadata in docfiles:
                    content = extract_text(str(docpath))
                    if isinstance(content, list):
                        logger.warning(
                            f"Skipping spreadsheet file {docpath} - use text documents for coverage analysis"
                        )
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
                                **doc_metadata,
                            },
                        )
                    )
        except FileNotFoundError as e:
            logger.error(str(e))
            raise typer.Exit(1)
    else:
        # try to get documents from analysis config
        config = data.get("config", {})

        # first check if documents are embedded in the config
        embedded_docs = config.get("documents", [])
        if embedded_docs:
            logger.info(f"Using {len(embedded_docs)} documents embedded in analysis")
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
            # try to load from document_paths in config
            doc_paths = config.get("document_paths", [])
            if doc_paths:
                logger.info(
                    f"Loading documents from {len(doc_paths)} paths in analysis config"
                )
                for path_entry in doc_paths:
                    # handle both tuple format [(path, metadata), ...] and plain paths
                    if isinstance(path_entry, (list, tuple)):
                        docpath = path_entry[0]
                        doc_metadata = path_entry[1] if len(path_entry) > 1 else {}
                    else:
                        docpath = path_entry
                        doc_metadata = {}

                    if not Path(docpath).exists():
                        logger.warning(
                            f"Document not found (may have moved): {docpath}"
                        )
                        continue

                    content = extract_text(str(docpath))
                    if isinstance(content, list):
                        logger.warning(f"Skipping spreadsheet file {docpath}")
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
        logger.error(
            "No documents found. Specify --documents or ensure analysis contains document paths."
        )
        raise typer.Exit(1)

    logger.info(f"Loaded {len(doc_items)} documents")

    # load groups if provided
    groups = None
    if groups_file:
        if not groups_file.exists():
            logger.error(f"Groups file not found: {groups_file}")
            raise typer.Exit(1)

        logger.info(f"Loading groups from {groups_file}...")
        try:
            groups_df = pd.read_excel(groups_file)

            # check for required columns
            if "filename" not in groups_df.columns or "group" not in groups_df.columns:
                logger.error("Groups file must have 'filename' and 'group' columns")
                logger.info(f"Available columns: {', '.join(groups_df.columns)}")
                raise typer.Exit(1)

            groups = dict(
                zip(groups_df["filename"].astype(str), groups_df["group"].astype(str))
            )
            logger.info(f"Loaded {len(groups)} group mappings")
        except Exception as e:
            logger.error(f"Error reading groups file: {e}")
            raise typer.Exit(1)

    # run analysis
    logger.info("Running coverage analysis...")
    analyzer = ThemeCoverageAnalyzer(
        chunk_size=chunk_size,
        overlap=overlap_size,
        split_unit=split_unit,
        aggregation=aggregation,
        embedding_template=embedding_template,
        embedding_model=embedding_model,
        sample_frac=sample_frac,
        sample_n=sample_n,
        embed_source=embed,
    )

    result = analyzer.analyze(analysis, doc_items, groups=groups)

    # apply calibration if provided
    calibration_info = None
    if calibration_path:
        from ..calibration import calibrate, load_calibration

        logger.info("Loading calibration model...")
        model, cal_metadata = load_calibration(calibration_path, embedding_model)
        method = cal_metadata.get("method", "gam")

        # calibrate chunk-level similarities
        chunk_sim = np.array(result.chunk_similarity_matrix)
        logger.info(f"Applying {method} calibration to {chunk_sim.shape[0]} chunks...")
        calibrated_sim = calibrate(chunk_sim, model, method=method)
        result.chunk_similarity_matrix = calibrated_sim.tolist()

        # recompute document-level aggregations from calibrated chunk similarities
        logger.info("Recomputing document-level aggregations with calibrated scores...")

        # group chunks by document (using chunk_info)
        from collections import defaultdict
        doc_chunks = defaultdict(list)
        for chunk_idx, info in enumerate(result.chunk_info):
            doc_chunks[info.document_id].append(chunk_idx)

        # rebuild coverage matrix using calibrated similarities
        new_coverage_matrix = []
        for doc_coverage in result.documents:
            doc_id = doc_coverage.document_id
            chunk_indices = doc_chunks[doc_id]
            chunk_sims = calibrated_sim[chunk_indices]  # [n_doc_chunks x n_themes]

            doc_scores = {}
            doc_scores_list = []
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
                doc_scores_list.append(score)

            # update document coverage scores
            doc_coverage.theme_scores = doc_scores
            new_coverage_matrix.append(doc_scores_list)

        result.coverage_matrix = new_coverage_matrix

        # update top_chunks_per_theme similarity scores using chunk_index and re-sort
        for theme_name in result.top_chunks_per_theme:
            theme_idx = result.theme_names.index(theme_name)
            for chunk_example in result.top_chunks_per_theme[theme_name]:
                # use chunk_index to get calibrated score directly
                chunk_example.similarity = float(calibrated_sim[chunk_example.chunk_index, theme_idx])
            # re-sort by calibrated similarity (descending)
            result.top_chunks_per_theme[theme_name] = sorted(
                result.top_chunks_per_theme[theme_name],
                key=lambda x: x.similarity,
                reverse=True
            )

        # recompute group statistics if groups were provided
        if result.groups:
            from ..coverage.models import GroupCoverage
            group_docs = defaultdict(list)
            for doc in result.documents:
                if doc.group:
                    group_docs[doc.group].append(doc)

            new_group_coverages = []
            for group_name, docs in sorted(group_docs.items()):
                theme_means = {}
                theme_stds = {}
                for theme_name in result.theme_names:
                    scores = [doc.theme_scores[theme_name] for doc in docs]
                    theme_means[theme_name] = float(np.mean(scores))
                    theme_stds[theme_name] = float(np.std(scores))
                new_group_coverages.append(
                    GroupCoverage(
                        group_name=group_name,
                        n_documents=len(docs),
                        theme_scores=theme_means,
                        theme_scores_std=theme_stds,
                    )
                )
            result.groups = new_group_coverages

        # store calibration info in result config
        result.config["calibration"] = {
            "path": str(calibration_path),
            "method": method,
            "embedding_model": cal_metadata.get("embedding_model"),
        }

        # prepare calibration info for template
        raw_values = np.linspace(0.5, 1.0, 21)
        calibrated_values = np.clip(calibrate(raw_values.reshape(-1, 1), model, method=method).flatten(), 0.0, 1.0)
        calibration_info = {
            "metadata": cal_metadata,
            "transformation": [
                {"raw": float(r), "calibrated": float(c)}
                for r, c in zip(raw_values, calibrated_values)
            ],
        }
        # load calibration plot image if it exists
        plot_path = calibration_path.with_suffix(".png")
        if plot_path.exists():
            with open(plot_path, "rb") as f:
                calibration_info["plot_base64"] = base64.b64encode(f.read()).decode("utf-8")

        logger.info("Calibration applied successfully")

    # generate outputs
    logger.info("Generating outputs...")

    # generate heatmaps (both continuous and thresholded versions)
    heatmap = generate_coverage_heatmap(result)
    heatmap_thresholded = generate_coverage_heatmap(result, threshold=threshold)
    group_heatmap = generate_group_heatmap(result) if result.groups else None
    chunk_heatmap = generate_chunk_heatmap(result, window_size=chunk_window)

    # generate new chunk-level visualizations
    normalized_heatmap = generate_normalized_chunk_heatmap(result, n_bins=n_bins)
    normalized_heatmap_zscore = generate_normalized_chunk_heatmap(
        result, n_bins=n_bins, z_score=True
    )
    absolute_heatmap = generate_absolute_chunk_heatmap(result, n_bins_target=n_bins)
    theme_trajectories = generate_theme_trajectories(result, n_bins=n_bins)
    within_doc_variation = compute_within_doc_variation(result, n_bins=n_bins)

    # compute threshold counts: for each theme, how many documents exceed each threshold
    count_thresholds = [0.6, 0.7, 0.8, 0.9]
    threshold_counts = {
        theme_name: {
            t: sum(1 for doc in result.documents if doc.theme_scores.get(theme_name, 0) >= t)
            for t in count_thresholds
        }
        for theme_name in result.theme_names
    }

    # write outputs into folder
    output_folder = Path(output)
    output_folder.mkdir(parents=True, exist_ok=True)

    if format in {"json", "all"}:
        json_path = output_folder / "coverage.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        logger.info(f"Wrote JSON output to {json_path}")

    if format in {"csv", "all"}:
        csv_path = output_folder / "coverage.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(result.to_csv())
        logger.info(f"Wrote CSV output to {csv_path}")

    if format in {"html", "all"}:
        html_path = output_folder / "coverage.html"

        # render HTML template
        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
        template = env.get_template("coverage.html")
        html_content = template.render(
            result=result,
            analysis_file=analysis_file,
            heatmap=heatmap,
            heatmap_thresholded=heatmap_thresholded,
            group_heatmap=group_heatmap,
            chunk_heatmap=chunk_heatmap,
            normalized_heatmap=normalized_heatmap,
            normalized_heatmap_zscore=normalized_heatmap_zscore,
            absolute_heatmap=absolute_heatmap,
            theme_trajectories=theme_trajectories,
            within_doc_variation=within_doc_variation,
            threshold=threshold,
            chunk_window=chunk_window,
            n_bins=n_bins,
            calibration_info=calibration_info,
            threshold_counts=threshold_counts,
            count_thresholds=count_thresholds,
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Wrote HTML output to {html_path}")

    logger.info(f"Coverage analysis complete")
