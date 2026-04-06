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

import numpy as np
import typer

from ..helpers import print_comparison_stats
from ._common import (TEMPLATES_DIR, check_and_prompt_credentials,
                      get_pdb_on_exception, get_soak_version,
                      resolve_analysis_path)

# re-export for backwards compatibility
_print_comparison_stats = print_comparison_stats

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
    """Generate an auto-filename for compare output based on inputs and options."""

    def abbreviate(name: str) -> str:
        for suffix in ["_dump", "_analysis", "_results", "_output"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        if len(name) > 12:
            name = name[:12]
        return name

    abbrev_inputs = [abbreviate(n) for n in input_names]
    inputs_part = "_vs_".join(abbrev_inputs)

    opts = []
    model_name = embedding_model.replace("local/", "").split("/")[-1]
    if model_name != "text-embedding-3-large":
        opts.append(model_name)
    if threshold != 0.6:
        opts.append(f"t{int(threshold * 100)}")
    sim_abbrevs = {"angular": "", "cosine": "cos", "shepard": "shep"}
    if similarity in sim_abbrevs:
        if sim_abbrevs[similarity]:
            opts.append(sim_abbrevs[similarity])
    else:
        opts.append(similarity[:3])
    if similarity == "shepard" and shepard_k != 1.0:
        opts.append(f"sk{shepard_k:.1f}".replace(".", ""))
    if no_paraphrase_bound:
        opts.append("nopara")

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

    opts_part = "_".join(opts) if opts else ""
    if opts_part:
        filename = f"{inputs_part}_{opts_part}_{options_hash}{extension}"
    else:
        filename = f"{inputs_part}_{options_hash}{extension}"

    filename = filename.replace(" ", "_").replace("/", "-")
    return filename


async def _generate_llm_labels(
    themes: list[str],
    model: str,
    api_key: str,
    base_url: str,
) -> list[str]:
    """Generate short, unique labels for themes using LLM."""
    from jinja2 import StrictUndefined, Template
    from struckdown import LLM, LLMCredentials, complete_async

    prompt_path = Path(__file__).parent.parent / "templates" / "make_labels.sd"
    prompt_template = prompt_path.read_text()

    themes_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(themes))
    template = Template(prompt_template, undefined=StrictUndefined)
    prompt = template.render(themes_text=themes_text, n_themes=len(themes))

    credentials = LLMCredentials(api_key=api_key, base_url=base_url)
    llm = LLM(model_name=model)

    result = await complete_async(
        multipart_prompt=prompt,
        model=llm,
        credentials=credentials,
    )

    if hasattr(result, "outputs") and "labels" in result.outputs:
        labels_output = result.outputs["labels"]
        if hasattr(labels_output, "labels"):
            return labels_output.labels
        elif isinstance(labels_output, list):
            return labels_output

    logger.warning("LLM label generation failed, using original theme names")
    return themes


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

    from ..api import CompareError
    from ..api import compare as api_compare
    from ..api import compare_strings as api_compare_strings
    from ..comparators.similarity_comparator import clear_comparison_cache
    from ..helpers import format_exception_concise
    from ..models import (QualitativeAnalysis, QualitativeAnalysisPipeline,
                          Theme)

    # clear cache if requested
    if clear_cache:
        clear_comparison_cache()

    # parse ot_k_values
    parsed_ot_k_values = None
    if ot_k_values:
        try:
            parsed_ot_k_values = [float(v.strip()) for v in ot_k_values.split(",")]
        except ValueError:
            logger.error(
                f"Invalid --ot-k-values format: {ot_k_values}. Use comma-separated floats."
            )
            raise typer.Exit(1)

    # handle calibration path resolution
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
    else:
        from ..calibration import get_bundled_calibration

        bundled_path = get_bundled_calibration(embedding_model)
        if bundled_path:
            calibration_path = bundled_path
            logger.info(f"Using bundled calibration for {embedding_model}")

    # check credentials for API embedding models
    if not embedding_model.startswith("local/"):
        check_and_prompt_credentials(Path.cwd())

    try:
        if strings:
            # STRINGS MODE
            xlsx_path = Path(strings)
            if not xlsx_path.exists():
                package_data_path = Path(__file__).parent.parent / strings
                if package_data_path.exists():
                    xlsx_path = package_data_path
                else:
                    logger.error(f"File not found: {strings}")
                    raise typer.Exit(1)

            logger.info(f"Reading {xlsx_path}...")
            if xlsx_path.suffix == ".csv":
                df = pd.read_csv(xlsx_path)
            else:
                df = pd.read_excel(xlsx_path)

            if cols:
                col_names = [c.strip() for c in cols.split(",")]
            else:
                if len(df.columns) < 2:
                    logger.error("File must have at least 2 columns for comparison")
                    raise typer.Exit(1)
                col_names = list(df.columns[:2])
                logger.info(f"Using first two columns: {col_names[0]}, {col_names[1]}")

            if not cols and len(df.columns) > 2:
                logger.warning(
                    f"File has {len(df.columns)} columns ({', '.join(df.columns)}) but only comparing first two. "
                    f"Use --cols to compare more, e.g. --cols {','.join(df.columns)}"
                )
            if len(col_names) < 2:
                logger.error("At least 2 columns required for comparison")
                raise typer.Exit(1)

            columns_data = {}
            for col_name in col_names:
                try:
                    items = df[col_name].dropna().astype(str).tolist()
                    if not items:
                        logger.error(f"Column '{col_name}' is empty")
                        raise typer.Exit(1)
                    columns_data[col_name] = items
                    logger.info(f"  Column {col_name}: {len(items)} items")
                except KeyError:
                    logger.error(f"Column not found: {col_name}")
                    logger.info(f"Available columns: {', '.join(df.columns)}")
                    raise typer.Exit(1)

            effective_embedding_template = embedding_template or "{name}"

            result = api_compare_strings(
                columns_data,
                threshold=threshold,
                embedding_model=embedding_model,
                similarity=similarity,
                shepard_k=shepard_k,
                ot_k=ot_k,
                embedding_template=effective_embedding_template,
                calibration=calibration_path,
                compute_paraphrase_bound=not no_paraphrase_bound,
                n_paraphrases=n_paraphrases,
            )

            # auto-generate filename
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
                )
                logger.info(f"Auto-generated filename: {effective_output}")

        else:
            # JSON MODE
            if not input_files or len(input_files) < 2:
                logger.error(
                    "At least 2 JSON files required for comparison (or use --strings)"
                )
                raise typer.Exit(1)

            effective_embedding_template = embedding_template or "{name}: {description}"

            result = api_compare(
                *input_files,
                threshold=threshold,
                embedding_model=embedding_model,
                similarity=similarity,
                shepard_k=shepard_k,
                ot_k=ot_k,
                embedding_template=effective_embedding_template,
                calibration=calibration_path,
                compute_paraphrase_bound=not no_paraphrase_bound,
                n_paraphrases=n_paraphrases,
            )

            # auto-generate filename
            effective_output = output
            if output and output.startswith("."):
                input_names = [Path(f).stem for f in input_files]
                effective_output = generate_compare_filename(
                    input_names=input_names,
                    embedding_model=embedding_model,
                    threshold=threshold,
                    similarity=similarity,
                    shepard_k=shepard_k,
                    no_paraphrase_bound=no_paraphrase_bound,
                    extension=output,
                )
                logger.info(f"Auto-generated filename: {effective_output}")

    except CompareError as e:
        logger.error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        if get_pdb_on_exception():
            traceback.print_exc()
            pdb.post_mortem()
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Comparison error:\n{error_msg}")

    # print stats to console
    output_path = Path(effective_output) if effective_output else None
    print_to_console = not effective_output or (
        output_path and output_path.suffix == ".txt"
    )

    if print_to_console:
        print(result.to_text(), file=sys.stdout)

    # load calibration info for HTML
    calibration_info = None
    if calibration_path:
        import base64

        from ..calibration import calibrate, load_calibration

        model, cal_metadata = load_calibration(calibration_path)
        cal_method = cal_metadata.get("method", "gam")
        raw_values = np.linspace(0.5, 1.0, 21)
        calibrated_values = calibrate(raw_values, model, method=cal_method)
        calibration_info = {
            "metadata": cal_metadata,
            "transformation": [
                {"raw": float(r), "calibrated": float(c)}
                for r, c in zip(raw_values, calibrated_values)
            ],
        }
        plot_path = Path(calibration_path).with_suffix(".png")
        if plot_path.exists():
            with open(plot_path, "rb") as f:
                calibration_info["plot_base64"] = base64.b64encode(f.read()).decode(
                    "utf-8"
                )

    # save output
    if effective_output:
        if output_path.suffix == ".txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.to_text())
            logger.info(f"Statistics saved to: {effective_output}")
        else:
            html_content = result.to_html(calibration_info=calibration_info)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"HTML report saved to: {effective_output}")
