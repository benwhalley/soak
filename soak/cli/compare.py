"""Compare command for comparing analyses."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import hashlib
import json
import logging
import pdb
import sys
import traceback
from pathlib import Path

import typer

from ._common import (TEMPLATES_DIR, check_and_prompt_credentials,
                      get_pdb_on_exception, get_soak_version,
                      resolve_analysis_path)

logger = logging.getLogger(__name__)


def generate_compare_filename(
    input_names: list[str],
    embedding_model: str = "text-embedding-3-large",
    threshold: float = 0.6,
    similarity: str = "angular",
    shepard_k: float = 1.0,
    no_paraphrase_bound: bool = False,
    extension: str = ".html",
    extra_options: dict | None = None,
) -> str:
    """Generate an auto-filename for compare output based on inputs and options.

    Creates a concise but unique filename encoding key parameters.
    Format: {inputs}_{options}_{hash}.{ext}

    The hash is computed from all options (including extra_options) to ensure
    unique filenames when any parameter changes.

    Examples:
        PROF_vs_NOTPROF_e5b_t75_a3f2.html
        A_vs_B_vs_C_cos_t60_b7c1.html
    """

    # abbreviate input names (remove common suffixes, shorten)
    def abbreviate(name: str) -> str:
        for suffix in ["_dump", "_analysis", "_results", "_output"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        if len(name) > 12:
            name = name[:12]
        return name

    abbrev_inputs = [abbreviate(n) for n in input_names]
    inputs_part = "_vs_".join(abbrev_inputs)

    # build options part (only non-default values)
    opts = []

    # embedding model: strip "local/" prefix and take last part after "/"
    model_name = embedding_model.replace("local/", "").split("/")[-1]
    if model_name != "text-embedding-3-large":
        opts.append(model_name)

    # threshold (if not default 0.6)
    if threshold != 0.6:
        opts.append(f"t{int(threshold * 100)}")

    # similarity metric (if not default angular)
    sim_abbrevs = {"angular": "", "cosine": "cos", "shepard": "shep"}
    if similarity in sim_abbrevs:
        if sim_abbrevs[similarity]:
            opts.append(sim_abbrevs[similarity])
    else:
        opts.append(similarity[:3])

    # shepard_k (only if using shepard and not default 1.0)
    if similarity == "shepard" and shepard_k != 1.0:
        opts.append(f"sk{shepard_k:.1f}".replace(".", ""))

    # no paraphrase bound
    if no_paraphrase_bound:
        opts.append("nopara")

    # compute hash of all options for uniqueness
    hash_data = {
        "embedding_model": embedding_model,
        "threshold": threshold,
        "similarity": similarity,
        "shepard_k": shepard_k,
        "no_paraphrase_bound": no_paraphrase_bound,
    }
    if extra_options:
        hash_data.update(extra_options)
    hash_str = json.dumps(hash_data, sort_keys=True, default=str)
    options_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:6]

    # combine parts
    opts_part = "_".join(opts) if opts else ""
    if opts_part:
        filename = f"{inputs_part}_{opts_part}_{options_hash}{extension}"
    else:
        filename = f"{inputs_part}_{options_hash}{extension}"

    # sanitise: replace spaces and special chars
    filename = filename.replace(" ", "_").replace("/", "-")

    return filename


async def _generate_llm_labels(
    themes: list[str],
    model: str,
    api_key: str,
    base_url: str,
) -> list[str]:
    """Generate short, unique labels for themes using LLM.

    Args:
        themes: List of theme names/strings to generate labels for
        model: LLM model name to use
        api_key: API key for LLM
        base_url: Base URL for LLM API

    Returns:
        List of short labels, same length as input themes
    """
    from jinja2 import StrictUndefined, Template
    from struckdown import LLM, LLMCredentials, chatter_async

    # load prompt template from .sd file
    prompt_path = Path(__file__).parent.parent / "pipelines" / "make_labels.sd"
    prompt_template = prompt_path.read_text()

    # format themes as numbered list
    themes_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(themes))

    # render template with context
    template = Template(prompt_template, undefined=StrictUndefined)
    prompt = template.render(themes_text=themes_text, n_themes=len(themes))

    credentials = LLMCredentials(api_key=api_key, base_url=base_url)
    llm = LLM(model_name=model)

    result = await chatter_async(
        multipart_prompt=prompt,
        model=llm,
        credentials=credentials,
    )

    # extract labels from result
    if hasattr(result, "outputs") and "labels" in result.outputs:
        labels_output = result.outputs["labels"]
        if hasattr(labels_output, "labels"):
            # structured response object with labels attribute
            return labels_output.labels
        elif isinstance(labels_output, list):
            return labels_output

    # fallback: return original themes if LLM fails
    logger.warning("LLM label generation failed, using original theme names")
    return themes


def _print_comparison_stats(
    result: dict,
    name_a: str,
    name_b: str,
    list_a: list,
    list_b: list,
    threshold: float,
    embedding_model: str,
    shepard_k: float,
    ot_k_values: list,
    similarity: str = "angular",
) -> str:
    """Generate text output for comparison statistics.

    Returns the formatted text output as a string.
    """
    import numpy as np

    # get similarity metric from result if available (canonical source)
    sim_metric = result.get("similarity_metric", similarity)

    lines = []

    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Comparison: {name_a} vs {name_b}")
    lines.append("=" * 70)
    lines.append(
        f"Similarity: {sim_metric}  |  Threshold: {threshold}  |  Shepard k: {shepard_k}"
    )
    lines.append(f"Embedding: {embedding_model}")
    lines.append(f"Items: {len(list_a)} ({name_a}) x {len(list_b)} ({name_b})")

    # coverage / hit rates
    lines.append("")
    lines.append("-" * 70)
    lines.append("COVERAGE (Hit Rates)")
    lines.append("-" * 70)
    lines.append(
        f"  Hit Rate {name_a}: {result['hit_rate_a']:.1%}  (items with >=1 match above threshold)"
    )
    lines.append(f"  Hit Rate {name_b}: {result['hit_rate_b']:.1%}")
    lines.append(
        f"  Jaccard:         {result['jaccard']:.3f}  (proportion of pairs above threshold)"
    )

    # fidelity
    lines.append("")
    lines.append("-" * 70)
    lines.append("FIDELITY (Mean Best-Match Similarity)")
    lines.append("-" * 70)
    lines.append(f"  {name_a} -> {name_b}: {result['mean_max_sim_a_to_b']:.3f}")
    lines.append(f"  {name_b} -> {name_a}: {result['mean_max_sim_b_to_a']:.3f}")
    lines.append(f"  Fidelity:        {result['fidelity']:.3f}  (harmonic mean)")

    # hungarian matching
    lines.append("")
    lines.append("-" * 70)
    lines.append("HUNGARIAN MATCHING (Optimal 1-to-1 Assignment)")
    lines.append("-" * 70)
    hungarian = result.get("hungarian", {})
    thresh_metrics = hungarian.get("thresholded_metrics", {})
    lines.append(f"  Coverage {name_a}: {thresh_metrics.get('coverage_a', 0):.1%}")
    lines.append(f"  Coverage {name_b}: {thresh_metrics.get('coverage_b', 0):.1%}")
    lines.append(
        f"  1-to-1 Jaccard:  {thresh_metrics.get('true_jaccard', 0):.3f}  "
        "(matched pairs / total unique items)"
    )

    # optimal transport
    lines.append("")
    lines.append("-" * 70)
    lines.append("OPTIMAL TRANSPORT (Many-to-Many Alignment)")
    lines.append("-" * 70)

    # if multiple K values requested, show stats for each
    ot_by_k = result.get("ot_by_k", {})
    default_k = result.get("default_k", 0.25)
    elbow_k = result.get("elbow_k")

    # show baseline reference values from default K's OT data
    if default_k in ot_by_k:
        default_ot = ot_by_k[default_k]["ot"]
        ceiling_sim = default_ot.get("paraphrase_upper_bound")
        ceiling_cost = default_ot.get("paraphrase_cost_lower_bound")
        floor_sim = default_ot.get("null_shared_mass_mean")
        floor_cost = default_ot.get("null_avg_cost_mean")

        if ceiling_sim is not None or floor_sim is not None:
            lines.append(f"  Baselines:")
            if ceiling_sim is not None:
                ceiling_align = 1 - (ceiling_cost or 0)
                lines.append(
                    f"    Paraphrase ceiling:   {ceiling_sim:.1%} shared mass, {ceiling_align:.1%} alignment"
                )
            if floor_sim is not None:
                floor_align = 1 - (floor_cost or 0)
                lines.append(
                    f"    Word-salad floor:     {floor_sim:.1%} shared mass, {floor_align:.1%} alignment"
                )

    # determine which K values to show
    if ot_k_values:
        display_k_values = ot_k_values
    else:
        # show default and elbow
        display_k_values = sorted(set([default_k] + ([elbow_k] if elbow_k else [])))

    for k_val in display_k_values:
        if k_val in ot_by_k:
            ot_data = ot_by_k[k_val]["ot"]
            marker = ""
            if k_val == default_k:
                marker = " (default)"
            if k_val == elbow_k:
                marker = " (knee/diminishing returns)"
            if k_val == default_k == elbow_k:
                marker = " (default, knee)"

            lines.append(f"")
            lines.append(f"  K = {k_val}{marker}")

            # shared mass metrics
            shared_mass = ot_data.get("shared_mass", 0)
            null_shared_mass = ot_data.get("null_shared_mass_mean")
            paraphrase_ceiling = ot_data.get("paraphrase_upper_bound")

            if paraphrase_ceiling is not None and null_shared_mass is not None:
                lines.append(
                    f"    Shared Mass:          {shared_mass:.1%}  (floor: {null_shared_mass:.1%}, ceiling: {paraphrase_ceiling:.1%})"
                )
            else:
                lines.append(f"    Shared Mass:          {shared_mass:.1%}")

            # paraphrase-scaled metrics (if available)
            pct_ceiling = ot_data.get("shared_mass_pct_of_ceiling")
            pct_improvement = ot_data.get("shared_mass_improvement_vs_null")
            if pct_ceiling is not None:
                lines.append(
                    f"      % of ceiling:       {pct_ceiling:.0%}  (vs paraphrase upper bound)"
                )
            if pct_improvement is not None:
                lines.append(
                    f"      vs word-salad:      {pct_improvement:.0%} better, relative to ceiling"
                )

            # semantic alignment (1 - cost, so higher = better)
            avg_cost = ot_data.get("avg_cost", 0)
            alignment = 1 - avg_cost
            alignment_ceiling = ot_data.get("alignment_paraphrase_ceiling")
            alignment_floor = ot_data.get("alignment_null_floor")

            if alignment_ceiling is not None and alignment_floor is not None:
                lines.append(
                    f"    Semantic Alignment:   {alignment:.1%}  (floor: {alignment_floor:.1%}, ceiling: {alignment_ceiling:.1%})"
                )
            else:
                lines.append(
                    f"    Semantic Alignment:   {alignment:.1%}  (quality of matches)"
                )

            # alignment paraphrase-scaled metrics
            align_pct_ceiling = ot_data.get("alignment_pct_of_ceiling")
            align_improvement = ot_data.get("alignment_improvement_vs_null")
            if align_pct_ceiling is not None:
                lines.append(
                    f"      % of ceiling:       {align_pct_ceiling:.0%}  (vs paraphrase upper bound)"
                )
            if align_improvement is not None:
                lines.append(
                    f"      vs word-salad:      {align_improvement:.0%} better, relative to ceiling"
                )

    if elbow_k:
        lines.append(f"")
        lines.append(f"  Knee detected at K = {elbow_k} (point of diminishing returns)")

    # text sankey diagram
    lines.append("")
    lines.append("-" * 70)
    lines.append(
        f"TRANSPORT FLOWS (cost labels: low <K, med K-2K, high >2K where K={default_k})"
    )
    lines.append("-" * 70)

    # use default K for sankey
    if default_k in ot_by_k:
        transport_plan = np.array(ot_by_k[default_k]["ot"]["transport_plan"])
        # use selected similarity for cost (same as OT uses internally)
        selected_sim = result.get(
            "selected_similarity_matrix", result.get("angle_similarity_matrix", [])
        )
        cost_matrix = 1 - np.array(selected_sim)
        total_mass = transport_plan.sum()

        # threshold for showing a flow (1% of max flow)
        flow_threshold = 0.01 * transport_plan.max() if transport_plan.max() > 0 else 0

        # cost thresholds relative to K (the OT mass penalty)
        # low: cost < K (cheaper than unmatching penalty, OT favours transport)
        # med: cost K to 2K (borderline)
        # high: cost > 2K (expensive, OT may prefer unmatching)
        k_threshold = default_k

        for i, item_a in enumerate(list_a):
            # find all B items this A item sends mass to
            flows = []
            for j, item_b in enumerate(list_b):
                flow = transport_plan[i, j]
                if flow > flow_threshold:
                    cost = cost_matrix[i, j] if len(cost_matrix) > 0 else 0
                    pct = (flow / total_mass * 100) if total_mass > 0 else 0
                    # qualitative cost label relative to K
                    if cost < k_threshold:
                        cost_label = "low"
                    elif cost < 2 * k_threshold:
                        cost_label = "med"
                    else:
                        cost_label = "high"
                    flows.append((item_b, pct, cost_label))

            # truncate item name for display
            item_a_display = (
                item_a[:25] + "..." if len(str(item_a)) > 28 else str(item_a)
            )

            if flows:
                flows.sort(key=lambda x: -x[1])  # sort by % descending
                flow_strs = [
                    f"{b[:20]}({cost},{pct:.0f}%)" for b, pct, cost in flows[:3]
                ]
                extra = f" +{len(flows)-3} more" if len(flows) > 3 else ""
                lines.append(
                    f"  {item_a_display:28s} --> {', '.join(flow_strs)}{extra}"
                )
            else:
                lines.append(f"  {item_a_display:28s} --> (unmatched)")

    # transport mass matrix
    lines.append("")
    lines.append("-" * 70)
    lines.append("TRANSPORT MASS MATRIX (%)")
    lines.append("-" * 70)

    if default_k in ot_by_k:
        transport_plan = np.array(ot_by_k[default_k]["ot"]["transport_plan"])
        total_mass = transport_plan.sum()
        if total_mass > 0:
            pct_matrix = transport_plan / total_mass * 100
        else:
            pct_matrix = transport_plan

        # print compact matrix
        _print_compact_matrix(
            lines,
            pct_matrix,
            list_a,
            list_b,
            fmt="{:5.1f}",
            name_a=name_a,
            name_b=name_b,
        )

    # selected similarity matrix (used for all metrics)
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"SIMILARITY MATRIX ({sim_metric.upper()} -- used for all metrics)")
    lines.append("-" * 70)

    if "selected_similarity_matrix" in result:
        _print_compact_matrix(
            lines,
            np.array(result["selected_similarity_matrix"]),
            list_a,
            list_b,
            fmt="{:5.2f}",
            name_a=name_a,
            name_b=name_b,
        )
    elif "angle_similarity_matrix" in result:
        # fallback for older results
        _print_compact_matrix(
            lines,
            np.array(result["angle_similarity_matrix"]),
            list_a,
            list_b,
            fmt="{:5.2f}",
            name_a=name_a,
            name_b=name_b,
        )

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def _print_compact_matrix(
    lines: list,
    matrix,
    list_a: list,
    list_b: list,
    fmt: str = "{:5.2f}",
    name_a: str = "A",
    name_b: str = "B",
    max_label_len: int = 12,
):
    """Append compact matrix representation to lines list."""
    import numpy as np

    n_a, n_b = matrix.shape

    # truncate labels
    def trunc(s, max_len):
        s = str(s)
        return s[: max_len - 1] + "..." if len(s) > max_len else s

    labels_a = [trunc(s, max_label_len) for s in list_a]
    labels_b = [trunc(s, max_label_len) for s in list_b]

    # if matrix is too large, show summary
    if n_a > 15 or n_b > 10:
        lines.append(f"  (Matrix {n_a}x{n_b} -- showing top 5 pairs by value)")
        flat_indices = np.argsort(matrix.ravel())[::-1][:5]
        for idx in flat_indices:
            i, j = np.unravel_index(idx, matrix.shape)
            lines.append(
                f"    {labels_a[i]:12s} <-> {labels_b[j]:12s}: {fmt.format(matrix[i, j])}"
            )
        return

    # header row
    header = " " * (max_label_len + 2) + "|"
    for label in labels_b:
        header += f" {label:>{max_label_len}}"
    lines.append(f"  {header}")

    # separator
    sep = "-" * (max_label_len + 1) + "+" + "-" * (len(labels_b) * (max_label_len + 1))
    lines.append(f"  {sep}")

    # data rows
    for i, label_a in enumerate(labels_a):
        row = f"{label_a:>{max_label_len}} |"
        for j in range(n_b):
            val = fmt.format(matrix[i, j])
            row += f" {val:>{max_label_len}}"
        lines.append(f"  {row}")


def compare(
    input_files: list[str] = typer.Argument(
        None,
        help="JSON files or directories containing QualitativeAnalysis results to compare (minimum 2). Not needed if using --strings.",
    ),
    strings: str = typer.Option(
        None,
        "--strings",
        "-s",
        help="Path to XLSX/CSV file with columns of strings to compare (alternative to JSON files)",
    ),
    cols: str = typer.Option(
        None,
        "--cols",
        "-c",
        help="Comma-separated column names to compare (e.g., 'A,B,C'). Compares all pairwise combinations. Default: 'A,B'",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (.html for full report, .txt for text stats only)",
    ),
    threshold: float = typer.Option(
        0.6,
        "--threshold",
        envvar="SOAK_THRESHOLD",
        help="Similarity threshold for matching themes",
    ),
    method: str = typer.Option(
        "umap",
        "--method",
        envvar="SOAK_METHOD",
        help="Dimensionality reduction method (umap, mds, pca)",
    ),
    label: str = typer.Option(
        "{name}",
        "--label",
        "-l",
        help="Python format string for theme labels in visualizations. Available: {name}, {description}",
    ),
    embedding_template: str = typer.Option(
        None,
        "--embedding-template",
        "-e",
        envvar="SOAK_EMBEDDING_TEMPLATE",
        help="Python format string for generating theme embeddings. Default: '{name}' for strings, '{name}: {description}' for JSON.",
    ),
    embedding_model: str = typer.Option(
        "text-embedding-3-large",
        "--embedding-model",
        envvar="SOAK_EMBEDDING_MODEL",
        help="Embedding model (use 'local/model-name' for sentence-transformers, e.g., 'local/all-MiniLM-L6-v2')",
    ),
    shepard_k: float = typer.Option(
        1.0,
        "--shepard-k",
        envvar="SOAK_SHEPARD_K",
        help="Shepard similarity decay parameter (default: 1.0, higher = steeper decay)",
    ),
    ot_k: float = typer.Option(
        0.25,
        "--ot-k",
        envvar="SOAK_OT_K",
        help="Default K for optimal transport mass penalty. Lower = more selective matching.",
    ),
    ot_k_values: str = typer.Option(
        None,
        "--ot-k-values",
        help="Comma-separated K values for OT analysis (e.g., '0.1,0.25,0.5'). Shows stats for each.",
    ),
    similarity: str = typer.Option(
        "angular",
        "--similarity",
        "-S",
        envvar="SOAK_SIMILARITY",
        help="Similarity metric: angular (default), cosine, shepard. Angular is preferred as it satisfies the triangle inequality. Used consistently for coverage, fidelity, and OT.",
    ),
    llm_labels: bool = typer.Option(
        False,
        "--llm-labels",
        help="Use LLM to generate short, unique labels for themes in plots (requires API credentials)",
    ),
    llm_labels_model: str = typer.Option(
        "gpt-4.1-mini",
        "--llm-labels-model",
        envvar="SOAK_LLM_LABELS_MODEL",
        help="Model to use for generating labels (default: gpt-4.1-mini)",
    ),
    no_paraphrase_bound: bool = typer.Option(
        False,
        "--no-paraphrase-bound",
        help="Disable paraphrase-based upper bound for relative metrics",
    ),
    n_paraphrases: int = typer.Option(
        7,
        "--n-paraphrases",
        envvar="SOAK_N_PARAPHRASES",
        help="Number of paraphrases per theme for upper bound baseline (default: 7)",
    ),
    paraphrase_model: str = typer.Option(
        None,
        "--paraphrase-model",
        envvar="SOAK_PARAPHRASE_MODEL",
        help="Model for paraphrase generation (default: gpt-4.1-mini)",
    ),
    filter_threshold: float = typer.Option(
        0.05,
        "--filter-threshold",
        envvar="SOAK_FILTER_THRESHOLD",
        help="Filter weak edges from transport plan. Edges with mass < threshold * row/col sum are removed. Default 0.05 (5%). Set to 0 to disable.",
    ),
    color_green: float = typer.Option(
        0.75,
        "--color-green",
        envvar="SOAK_COLOR_GREEN",
        help="Similarity-based colouring: links with similarity >= this value appear green. Default 0.75.",
    ),
    color_red: float = typer.Option(
        0.4,
        "--color-red",
        envvar="SOAK_COLOR_RED",
        help="Similarity-based colouring: links with similarity <= this value appear red. Default 0.4.",
    ),
    k_color_green: float = typer.Option(
        0.2,
        "--k-color-green",
        envvar="SOAK_K_COLOR_GREEN",
        help="K-relative colouring: links with cost < this*K appear green (strong match). Default 0.2.",
    ),
    k_color_red: float = typer.Option(
        1.1,
        "--k-color-red",
        envvar="SOAK_K_COLOR_RED",
        help="K-relative colouring: links with cost > this*K appear red (marginal match). Default 1.1.",
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Clear all cached comparison results before running. Use when parameters have changed.",
    ),
    calibration: Path = typer.Option(
        None,
        "--calibration",
        help="Path to calibration folder or file. If folder, looks for calibration.yaml inside.",
    ),
):
    """Compare analyses or string lists and generate comparison statistics.

    Two modes:

    1. JSON mode (default): Compare two or more QualitativeAnalysis JSON files
       soak compare results1.json results2.json results3.json

    2. Strings mode: Compare columns from an XLSX/CSV file
       soak compare --strings data.xlsx --cols "A,B"
       soak compare --strings data.xlsx --cols "Method1,Method2,Method3"

    When comparing 3+ items, all pairwise combinations are computed.
    Statistics are always printed to stdout. Use --output to save HTML report or text file.
    """
    import pandas as pd
    from jinja2 import Environment, FileSystemLoader

    from ..comparators.similarity_comparator import (SimilarityComparator,
                                                     clear_comparison_cache,
                                                     compare_result_similarity)
    from ..helpers import format_exception_concise
    from ..models import (QualitativeAnalysis, QualitativeAnalysisPipeline,
                          Theme)

    # clear cache if requested
    if clear_cache:
        clear_comparison_cache()

    # parse ot_k_values if provided
    parsed_ot_k_values = None
    if ot_k_values:
        try:
            parsed_ot_k_values = [float(v.strip()) for v in ot_k_values.split(",")]
        except ValueError:
            logger.error(
                f"Invalid --ot-k-values format: {ot_k_values}. Use comma-separated floats."
            )
            raise typer.Exit(1)

    # handle calibration option
    # accepts either a folder (looks for calibration.yaml/.json inside) or a direct file path
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

    # determine mode: strings or JSON
    if strings:
        # STRINGS MODE: compare columns from XLSX/CSV
        xlsx_path = Path(strings)
        if not xlsx_path.exists():
            # try package soak-data directory
            package_data_path = Path(__file__).parent.parent / strings
            if package_data_path.exists():
                xlsx_path = package_data_path
            else:
                logger.error(f"File not found: {strings}")
                raise typer.Exit(1)

        logger.info(f"Reading {xlsx_path}...")
        try:
            if xlsx_path.suffix == ".csv":
                df = pd.read_csv(xlsx_path)
            else:
                df = pd.read_excel(xlsx_path)
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise typer.Exit(1)

        # parse column names (default to first two columns from file)
        if cols:
            col_names = [c.strip() for c in cols.split(",")]
        else:
            if len(df.columns) < 2:
                logger.error("File must have at least 2 columns for comparison")
                raise typer.Exit(1)
            col_names = list(df.columns[:2])
            logger.info(f"Using first two columns: {col_names[0]}, {col_names[1]}")

        # warn if file has more columns than being compared
        if not cols and len(df.columns) > 2:
            logger.warning(
                f"File has {len(df.columns)} columns ({', '.join(df.columns)}) but only comparing first two. "
                f"Use --cols to compare more, e.g. --cols {','.join(df.columns)}"
            )
        if len(col_names) < 2:
            logger.error("At least 2 columns required for comparison")
            raise typer.Exit(1)

        # extract each column
        column_data = {}
        for col_name in col_names:
            try:
                items = df[col_name].dropna().astype(str).tolist()
                if not items:
                    logger.error(f"Column '{col_name}' is empty")
                    raise typer.Exit(1)
                column_data[col_name] = items
                logger.info(f"  Column {col_name}: {len(items)} items")
            except KeyError:
                logger.error(f"Column not found: {col_name}")
                logger.info(f"Available columns: {', '.join(df.columns)}")
                raise typer.Exit(1)

        # check credentials
        if not embedding_model.startswith("local/"):
            check_and_prompt_credentials(Path.cwd())

        # create QualitativeAnalysis objects for each column
        analyses = []
        for col_name, items in column_data.items():
            themes = [Theme(name=s, description=s, code_slugs=[]) for s in items]
            analysis = QualitativeAnalysis(name=col_name, themes=themes)
            analyses.append(analysis)

        # generate LLM labels if requested
        if llm_labels:
            api_key, base_url = check_and_prompt_credentials(Path.cwd())
            logger.info("Generating LLM labels for themes...")

            for analysis in analyses:
                theme_names = [t.name for t in analysis.themes]
                labels = asyncio.run(
                    _generate_llm_labels(
                        themes=theme_names,
                        model=llm_labels_model,
                        api_key=api_key,
                        base_url=base_url,
                    )
                )
                # set labels on themes
                if len(labels) == len(analysis.themes):
                    for theme, lbl in zip(analysis.themes, labels):
                        theme.label = lbl
                    logger.info(f"  {analysis.name}: generated {len(labels)} labels")
                else:
                    logger.warning(
                        f"  {analysis.name}: label count mismatch ({len(labels)} vs {len(analysis.themes)}), using names"
                    )
                    for i, theme in enumerate(analysis.themes):
                        theme.set_label("{name}", i + 1)

        # default embedding template for strings mode
        effective_embedding_template = embedding_template or "{name}"

        logger.info(
            f"Comparing {len(analyses)} sets ({len(analyses) * (len(analyses) - 1) // 2} pairwise comparisons)..."
        )

        # use comparator for all pairwise combinations
        comparator = SimilarityComparator()
        comparison = comparator.compare(
            analyses,
            config={
                "threshold": threshold,
                "method": method,
                "n_neighbors": 5,
                "min_dist": 0.01,
                "label_template": label,
                "embedding_template": effective_embedding_template,
                "embedding_model": embedding_model,
                "k": shepard_k,
                "reg_m": ot_k,
                "distance": similarity,
                "compute_paraphrase_bound": not no_paraphrase_bound,
                "n_paraphrases": n_paraphrases,
                "paraphrase_model": paraphrase_model,
                "calibration_path": str(calibration_path) if calibration_path else None,
                "filter_threshold": filter_threshold,
                "color_green": color_green,
                "color_red": color_red,
                "k_color_green": k_color_green,
                "k_color_red": k_color_red,
            },
        )

        # auto-generate filename if output is just an extension (e.g., ".html")
        effective_output = output
        if output and output.startswith("."):
            effective_output = generate_compare_filename(
                input_names=col_names,
                embedding_model=embedding_model,
                threshold=threshold,
                similarity=similarity,
                shepard_k=shepard_k,
                no_paraphrase_bound=no_paraphrase_bound,
                extension=output,
                extra_options={
                    "embedding_template": effective_embedding_template,
                    "n_paraphrases": n_paraphrases,
                    "paraphrase_model": paraphrase_model,
                    "calibration_path": (
                        str(calibration_path) if calibration_path else None
                    ),
                    "filter_threshold": filter_threshold,
                    "color_green": color_green,
                    "color_red": color_red,
                    "k_color_green": k_color_green,
                    "k_color_red": k_color_red,
                },
            )
            logger.info(f"Auto-generated filename: {effective_output}")

        # determine if we should print stats to console
        # only print if: no output specified, or output is .txt
        output_path = Path(effective_output) if effective_output else None
        print_to_console = not effective_output or (
            output_path and output_path.suffix == ".txt"
        )

        # generate statistics for each pairwise comparison
        all_stats_text = []
        for key, comp in comparison.by_comparisons().items():
            stats = comp["stats"]
            analysis_a, analysis_b = comp["a"], comp["b"]
            list_a = [t.name for t in analysis_a.themes]
            list_b = [t.name for t in analysis_b.themes]

            stats_text = _print_comparison_stats(
                stats,
                name_a=analysis_a.name,
                name_b=analysis_b.name,
                list_a=list_a,
                list_b=list_b,
                threshold=threshold,
                embedding_model=embedding_model,
                shepard_k=shepard_k,
                ot_k_values=parsed_ot_k_values,
                similarity=similarity,
            )
            all_stats_text.append(stats_text)
            if print_to_console:
                print(stats_text, file=sys.stdout)

        # load calibration info for HTML display
        calibration_info = None
        if calibration_path:
            import base64

            import numpy as np

            from ..calibration import calibrate, load_calibration

            model, cal_metadata = load_calibration(calibration_path)
            method = cal_metadata.get("method", "gam")
            # generate sample transformation points using interpolation
            raw_values = np.linspace(0.5, 1.0, 21)
            calibrated_values = calibrate(raw_values, model, method=method)
            calibration_info = {
                "metadata": cal_metadata,
                "transformation": [
                    {"raw": float(r), "calibrated": float(c)}
                    for r, c in zip(raw_values, calibrated_values)
                ],
            }
            # load calibration plot image if it exists
            plot_path = Path(calibration_path).with_suffix(".png")
            if plot_path.exists():
                with open(plot_path, "rb") as f:
                    calibration_info["plot_base64"] = base64.b64encode(f.read()).decode(
                        "utf-8"
                    )

        # save output if requested
        if effective_output:
            if output_path.suffix == ".txt":
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(all_stats_text))
                logger.info(f"Statistics saved to: {effective_output}")
            else:
                # HTML output
                env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
                env.globals["enumerate"] = enumerate
                template = env.get_template("comparison.html")
                html_content = template.render(
                    comparison=comparison,
                    calibration_info=calibration_info,
                    soak_version=get_soak_version(),
                    text_output="\n\n".join(all_stats_text),
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"HTML report saved to: {effective_output}")

    else:
        # JSON MODE: compare analysis files
        if not input_files or len(input_files) < 2:
            logger.error(
                "At least 2 JSON files required for comparison (or use --strings)"
            )
            raise typer.Exit(1)

        logger.info(f"Loading {len(input_files)} analyses...")
        analyses = []

        for input_arg in input_files:
            input_path = resolve_analysis_path(input_arg)

            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            try:
                if "nodes" in data:
                    pipeline = QualitativeAnalysisPipeline.model_validate(data)
                    analysis = pipeline.result()
                    analysis.name = input_path.stem
                else:
                    analysis = QualitativeAnalysis.model_validate(data)
                    if not analysis.name or analysis.name == analysis.sha256()[:8]:
                        analysis.name = input_path.stem

                analyses.append(analysis)
            except Exception as e:
                if get_pdb_on_exception():
                    traceback.print_exc()
                    pdb.post_mortem()
                error_msg = format_exception_concise(e)
                raise typer.BadParameter(f"Error loading {input_path}:\n{error_msg}")
            logger.info(
                f"  Loaded: {analysis.name} ({len(analysis.themes)} themes, {len(analysis.codes)} codes)"
            )

        # generate LLM labels if requested
        if llm_labels:
            api_key, base_url = check_and_prompt_credentials(Path.cwd())
            logger.info("Generating LLM labels for themes...")

            for analysis in analyses:
                theme_names = [t.name for t in analysis.themes]
                labels = asyncio.run(
                    _generate_llm_labels(
                        themes=theme_names,
                        model=llm_labels_model,
                        api_key=api_key,
                        base_url=base_url,
                    )
                )
                # set labels on themes
                if len(labels) == len(analysis.themes):
                    for theme, lbl in zip(analysis.themes, labels):
                        theme.label = lbl
                    logger.info(f"  {analysis.name}: generated {len(labels)} labels")
                else:
                    logger.warning(
                        f"  {analysis.name}: label count mismatch ({len(labels)} vs {len(analysis.themes)}), using names"
                    )
                    for i, theme in enumerate(analysis.themes):
                        theme.set_label("{name}", i + 1)

        # default embedding template for JSON mode
        effective_embedding_template = embedding_template or "{name}: {description}"

        logger.info("Comparing analyses...")
        comparator = SimilarityComparator()
        comparison = comparator.compare(
            analyses,
            config={
                "threshold": threshold,
                "method": method,
                "n_neighbors": 5,
                "min_dist": 0.01,
                "label_template": label,
                "embedding_template": effective_embedding_template,
                "embedding_model": embedding_model,
                "k": shepard_k,
                "reg_m": ot_k,
                "distance": similarity,
                "compute_paraphrase_bound": not no_paraphrase_bound,
                "n_paraphrases": n_paraphrases,
                "paraphrase_model": paraphrase_model,
                "calibration_path": str(calibration_path) if calibration_path else None,
                "filter_threshold": filter_threshold,
                "color_green": color_green,
                "color_red": color_red,
                "k_color_green": k_color_green,
                "k_color_red": k_color_red,
            },
        )

        # auto-generate filename if output is just an extension (e.g., ".html")
        effective_output = output
        if output and output.startswith("."):
            input_names = [a.name for a in analyses]
            effective_output = generate_compare_filename(
                input_names=input_names,
                embedding_model=embedding_model,
                threshold=threshold,
                similarity=similarity,
                shepard_k=shepard_k,
                no_paraphrase_bound=no_paraphrase_bound,
                extension=output,
                extra_options={
                    "embedding_template": effective_embedding_template,
                    "n_paraphrases": n_paraphrases,
                    "paraphrase_model": paraphrase_model,
                    "calibration_path": (
                        str(calibration_path) if calibration_path else None
                    ),
                    "filter_threshold": filter_threshold,
                    "color_green": color_green,
                    "color_red": color_red,
                    "k_color_green": k_color_green,
                    "k_color_red": k_color_red,
                },
            )
            logger.info(f"Auto-generated filename: {effective_output}")

        # determine if we should print stats to console
        # only print if: no output specified, or output is .txt
        output_path = (
            Path(effective_output) if effective_output else Path("comparison.html")
        )
        print_to_console = not output or output_path.suffix == ".txt"

        # generate statistics for each pairwise comparison
        all_stats_text = []
        for key, comp in comparison.by_comparisons().items():
            stats = comp["stats"]
            analysis_a, analysis_b = comp["a"], comp["b"]
            list_a = [t.name for t in analysis_a.themes]
            list_b = [t.name for t in analysis_b.themes]

            stats_text = _print_comparison_stats(
                stats,
                name_a=analysis_a.name,
                name_b=analysis_b.name,
                list_a=list_a,
                list_b=list_b,
                threshold=threshold,
                embedding_model=embedding_model,
                shepard_k=shepard_k,
                ot_k_values=parsed_ot_k_values,
                similarity=similarity,
            )
            all_stats_text.append(stats_text)
            if print_to_console:
                print(stats_text, file=sys.stdout)

        # load calibration info for HTML display
        calibration_info = None
        if calibration_path:
            import base64

            import numpy as np

            from ..calibration import calibrate, load_calibration

            model, cal_metadata = load_calibration(calibration_path)
            method = cal_metadata.get("method", "gam")
            # generate sample transformation points using interpolation
            raw_values = np.linspace(0.5, 1.0, 21)
            calibrated_values = calibrate(raw_values, model, method=method)
            calibration_info = {
                "metadata": cal_metadata,
                "transformation": [
                    {"raw": float(r), "calibrated": float(c)}
                    for r, c in zip(raw_values, calibrated_values)
                ],
            }
            # load calibration plot image if it exists
            plot_path = Path(calibration_path).with_suffix(".png")
            if plot_path.exists():
                with open(plot_path, "rb") as f:
                    calibration_info["plot_base64"] = base64.b64encode(f.read()).decode(
                        "utf-8"
                    )

        # save output
        if output_path.suffix == ".txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_stats_text))
            logger.info(f"Statistics saved to: {output_path}")
        else:
            # HTML output
            logger.info("Generating HTML report...")
            env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
            env.globals["enumerate"] = enumerate
            template = env.get_template("comparison.html")
            html_content = template.render(
                comparison=comparison,
                calibration_info=calibration_info,
                soak_version=get_soak_version(),
                text_output="\n\n".join(all_stats_text),
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"HTML report saved to: {output_path}")
