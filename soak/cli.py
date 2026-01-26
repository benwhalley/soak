"""Command-line interface for running qualitative analysis pipelines."""

import asyncio
import json
import logging
import os
import pdb
import shutil
import sys
import traceback
from pathlib import Path

import typer

app = typer.Typer()

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
logger = logging.getLogger(__name__)

# module-level flag for pdb on exception
_pdb_on_exception = False

PIPELINE_DIR = Path(__file__).parent / "pipelines"
COMMANDS = {"run", "export", "dump", "compare", "show", "tui", "compare-strings", "coverage"}


@app.callback()
def main(
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,  # allows -v -vv -vvv
        help="Increase verbosity (-v=INFO, -vv=DEBUG)",
    ),
    pdb_on_exception: bool = typer.Option(
        False,
        "--pdb",
        help="Drop into pdb debugger on unhandled exceptions",
    ),
):
    """soak: DAG-based pipeline system for LLM-assisted qualitative analysis.

    Try `soak run --help` to get started
    """
    global _pdb_on_exception
    _pdb_on_exception = pdb_on_exception
    setup_logging(verbose)


def generate_html_output(pipeline: "QualitativeAnalysisPipeline", template: str) -> str:
    """Generate HTML output from a pipeline using specified template.

    Args:
        pipeline: QualitativeAnalysisPipeline instance
        template: Template name or path

    Returns:
        Rendered HTML string
    """
    template_path = resolve_template(template)
    return pipeline.to_html(template_path=str(template_path))


def generate_all_html_outputs(
    pipeline: "QualitativeAnalysisPipeline",
    templates: list[str],
    on_error: str = "raise",
) -> dict[str, str]:
    """Generate HTML outputs for multiple templates.

    Args:
        pipeline: QualitativeAnalysisPipeline instance to render
        templates: List of template names or paths
        on_error: Error handling strategy -- "raise" to raise exceptions, "warn" to log warnings and continue

    Returns:
        Dict mapping template name to rendered HTML content
    """
    from .helpers import format_exception_concise

    html_outputs = {}
    for tmpl in templates:
        try:
            html_outputs[tmpl] = generate_html_output(pipeline, tmpl)
        except Exception as e:
            error_msg = format_exception_concise(e)
            if on_error == "raise":
                raise typer.BadParameter(
                    f"Error generating HTML output for template '{tmpl}':\n{error_msg}"
                )
            else:  # warn
                logger.warning(
                    f"Error generating HTML for template '{tmpl}': {error_msg}"
                )
    return html_outputs


def load_pipeline_json(input_json: str) -> "QualitativeAnalysisPipeline":
    """Load and validate a pipeline from a JSON file.

    Args:
        input_json: Path to JSON file

    Returns:
        QualitativeAnalysisPipeline instance

    Raises:
        typer.Exit: If file not found or validation fails
    """
    from .helpers import format_exception_concise
    from .models import QualitativeAnalysisPipeline

    input_path = Path(input_json)
    if not input_path.exists():
        logger.error(f"File not found: {input_json}")
        raise typer.Exit(1)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Read pipeline from {input_json}")

    try:
        pipeline = QualitativeAnalysisPipeline.model_validate(data)
        logger.info(f"Loaded pipeline: {pipeline.name}")
        return pipeline
    except Exception as e:
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Could not load as pipeline:\n{error_msg}")


def resolve_template(template: str) -> Path:
    """Resolve template name or path to actual template file path.

    Args:
        template: Template name (e.g., 'default', 'narrative') or path to template file

    Returns:
        Path to resolved template file

    Raises:
        typer.Exit: If template cannot be found
    """
    template_path = Path(template)
    templates_dir = Path(__file__).parent / "templates"

    if template_path.exists():
        # Absolute or relative path that exists
        logger.info(f"Using custom template: {template_path}")
        return template_path

    # Try to find it in soak/templates directory
    candidate = templates_dir / template
    if candidate.exists():
        logger.info(f"Using template: {template} from soak/templates")
        return candidate

    # If template doesn't have a file extension, try adding .html
    if not template_path.suffix:
        candidate_with_ext = templates_dir / f"{template}.html"
        if candidate_with_ext.exists():
            logger.info(f"Using template: {template}.html from soak/templates")
            return candidate_with_ext

    # Template not found
    logger.error(f"Template not found: {template}")
    logger.error(f"Looked in: {templates_dir}")
    raise typer.Exit(1)


# @app.command()
# def export(
#     input_json: str = typer.Argument(
#         ..., help="Path to JSON file containing QualitativeAnalysisPipeline"
#     ),
#     output: str = typer.Option(
#         None,
#         "--output",
#         "-o",
#         help="Output file path (without extension). If not specified, HTML goes to stdout (single template only)",
#     ),
#     template: list[str] = typer.Option(
#         ["pipeline.html"],
#         "--template",
#         "-t",
#         help="Template name (in soak/templates) or path to custom HTML template (can be used multiple times)",
#     ),
# ):
#     """Export a saved analysis result to HTML."""

#     # Load the pipeline
#     pipeline = load_pipeline_json(input_json)

#     # Generate HTML for each template
#     html_outputs = generate_all_html_outputs(pipeline, template, on_error="raise")

#     # If no output specified, write to stdout (only for single template)
#     if output is None:
#         if len(template) > 1:
#             print(
#                 "Error: Multiple templates specified but no --output given. Specify --output or use single template.",
#                 file=sys.stderr,
#             )
#             raise typer.Exit(1)
#         # Output single HTML to stdout
#         print(html_outputs[template[0]], file=sys.stdout)
#     else:
#         # Write HTML files when output is specified
#         for tmpl in template:
#             template_stem = Path(resolve_template(tmpl)).stem
#             output_html = f"{output}_{template_stem}.html"
#             logger.info(
#                 f"Writing HTML output with template '{template_stem}' to {output_html}"
#             )
#             with open(output_html, "w", encoding="utf-8") as f:
#                 f.write(html_outputs[tmpl])

#         if len(template) == 1:
#             print(
#                 f"Successfully exported to {output}_{Path(resolve_template(template[0])).stem}.html",
#                 file=sys.stderr,
#             )
#         else:
#             print(
#                 f"Successfully exported {len(template)} HTML files to {output}_*.html",
#                 file=sys.stderr,
#             )


@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Pipeline name to run (e.g., 'poc')"),
    input: list[Path] = typer.Argument(
        None,
        help="Input file paths (file patterns or zip files, supports globs like '*.txt')",
    ),
    context: list[str] = typer.Option(
        None,
        "--context",
        "-c",
        help="Override context variables (format: key=value, can be used multiple times)",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (without extensions). If not specified, derived from pipeline name",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing output files/folders"
    ),
    template: list[str] = typer.Option(
        ["pipeline.html"],
        "--template",
        "-t",
        envvar="SOAK_TEMPLATE",
        help="Template name (in soak/templates) or path to custom HTML template (can be used multiple times)",
    ),
    include_documents: bool = typer.Option(
        False, "--include-documents", help="Include original documents in output"
    ),
    sample: int = typer.Option(
        None,
        "--sample",
        "-S",
        help="Randomly sample N rows/documents from input (mutually exclusive with --head)",
    ),
    head: int = typer.Option(
        None,
        "--head",
        "-H",
        help="Take first N rows/documents from input (mutually exclusive with --sample)",
    ),
    seed: int = typer.Option(
        None,
        "--seed",
        envvar="SOAK_SEED",
        help="Random seed for reproducible outputs and document shuffling (default: 42)",
    ),
    model_name: str = typer.Option(
        None, "--model", "-m", envvar="SOAK_MODEL", help="LLM model name"
    ),
    progress: bool = typer.Option(
        None,
        "--progress/--no-progress",
        help="Show progress bars (auto-detected: enabled for TTY, disabled with -vv)",
    ),
    timeout: int = typer.Option(
        90,
        "--timeout",
        envvar="SOAK_LLM_TIMEOUT",
        help="Timeout in seconds for individual LLM API calls (default: 90)",
    ),
    skip_node: list[str] = typer.Option(
        None,
        "--skip-node",
        help="Skip specified node(s) during execution (can be used multiple times)",
    ),
    stop_at: str = typer.Option(
        None,
        "--stop-at",
        help="Stop execution before the specified node runs",
    ),
):
    """Run a pipeline on input files."""
    from struckdown import CostSummary, LLMCredentials, new_run

    from .document_utils import unpack_zip_to_temp_paths_if_needed
    from .helpers import format_exception_concise, hash_run_config, resolve_pipeline
    from .specs import load_template_bundle

    new_run()

    # Validate that input files are provided
    if not input:
        logger.error("No input files specified.")
        raise typer.Exit(1)

    # Validate that no bare directories are passed (must use glob patterns like dir/*.txt)
    for inp in input:
        if inp.is_dir():
            print(
                f"Error: '{inp}' is a directory. Use a glob pattern instead (e.g., '{inp}/*.txt')",
                file=sys.stderr,
            )
            raise typer.Exit(1)

    # Validate mutually exclusive options
    if sample is not None and head is not None:
        logger.error("--sample and --head are mutually exclusive")
        raise typer.Exit(1)

    # Auto-detect progress bar setting if not explicitly provided
    if progress is None:
        # Enable progress bars if:
        # 1. Output is a TTY (interactive terminal)
        # 2. Verbosity is not DEBUG (-vv)
        # Check if logger is at DEBUG level (verbosity >= 2)
        is_debug = logging.getLogger().level <= logging.DEBUG
        progress = sys.stderr.isatty() and not is_debug
    else:
        # User explicitly set --progress or --no-progress, respect their choice
        pass

    # Check and prompt for credentials first
    api_key, base_url = check_and_prompt_credentials(Path.cwd())

    # Save the string pipeline argument before it gets reassigned
    pipeline_arg = pipeline

    pipyml = resolve_pipeline(pipeline_arg, Path.cwd(), PIPELINE_DIR)
    logger.info(f"Loading pipeline from {pipyml}")

    # If no output specified, derive from pipeline filename
    if output is None:
        output = Path(pipyml).stem
        logger.info(f"Using default output name: {output}")

    # Check if ANY output files/folders exist BEFORE running pipeline
    output_json = Path(f"{output}.json")

    # Check for all template HTML files
    output_html_files = []
    for tmpl in template:
        template_stem = Path(resolve_template(tmpl)).stem
        output_html_files.append(Path(f"{output}_{template_stem}.html"))

    # Check for dump folder (always created)
    expected_dump_folder = Path(f"{output}_dump")

    # Collect all existing paths
    existing = []
    if output_json.exists():
        existing.append(str(output_json))
    for html_file in output_html_files:
        if html_file.exists():
            existing.append(str(html_file))
    if expected_dump_folder.exists():
        existing.append(str(expected_dump_folder) + "/")

    # Fail if any conflicts and not force
    if existing:
        if not force:
            print(
                f"Error: Output file(s)/folder(s) already exist: {', '.join(existing)}",
                file=sys.stderr,
            )
            print(f"Use --force/-f to overwrite", file=sys.stderr)
            raise typer.Exit(1)
        else:
            logger.warning(f"Overwriting existing output folder: {', '.join(existing)}")

    try:
        pipeline = load_template_bundle(pipyml)
    except ValueError as e:
        raise typer.BadParameter(f"Pipeline validation error: {e}")

    # Override default_context with CLI-provided values
    logger.info(f"Setting params: {context}")
    if context:
        for item in context:
            if "=" not in item:
                print(
                    f"Error: Context variable must be in format 'key=value', got: {item}",
                    file=sys.stderr,
                )
                raise typer.Exit(1)
            key, value = item.split("=", 1)
            pipeline.default_context[key] = value
            logger.info(f"Set context variable: {key}={value}")

    if model_name is not None:
        pipeline.config.model_name = model_name
    if seed is not None:
        pipeline.config.seed = seed
        logger.info(f"Set seed to {seed}")
    pipeline.config.llm_credentials = LLMCredentials(
        api_key=api_key,
        base_url=base_url,
    )

    # Set sampling options
    if sample is not None:
        pipeline.config.sample_n = sample
        logger.info(f"Will randomly sample {sample} rows/documents")
    if head is not None:
        pipeline.config.head_n = head
        logger.info(f"Will take first {head} rows/documents")

    # Set progress bar setting
    pipeline.config.show_progress = progress
    if progress:
        logger.debug("Progress bars enabled")

    # Set LLM timeout
    pipeline.config.llm_timeout = timeout
    logger.debug(f"LLM timeout set to {timeout} seconds")

    # Set skip nodes and stop-at
    if skip_node:
        pipeline.config.skip_nodes = skip_node
        logger.info(f"Will skip nodes: {', '.join(skip_node)}")
    if stop_at:
        pipeline.config.stop_at_node = stop_at
        logger.info(f"Will stop at node: {stop_at}")

    # Set pdb on exception
    pipeline.config.pdb_on_exception = _pdb_on_exception

    # Always create dump folder for incremental export
    dump_path = Path(f"{output}_dump")

    # Remove existing dump folder if force is enabled
    if dump_path.exists() and force:
        logger.info(f"Removing existing dump folder: {dump_path}")
        shutil.rmtree(dump_path)

    # Build command string for metadata
    cmd_parts = [f"soak {pipeline_arg}"]
    for inp in input:
        cmd_parts.append(f"--input {inp}")
    cmd_parts.append(f"--output {output}")
    if model_name:
        cmd_parts.append(f"--model-name {model_name}")
    if context:
        for ctx in context:
            cmd_parts.append(f"--context {ctx}")

    # Generate config hash for dump folder naming
    config_hash = hash_run_config(
        input_files=input,
        model_name=model_name,
        context=context,
        template=template,
    )

    metadata = {
        "command": " ".join(cmd_parts),
        "pipeline_file": str(pipyml),
        "model_name": model_name,
        "templates": template,
        "unique_id": config_hash,
    }
    if context:
        metadata["context_overrides"] = dict([c.split("=", 1) for c in context])

    # Enable incremental export (nodes export as they finish)
    pipeline.config.export_enabled = True
    pipeline.config.export_folder = dump_path
    pipeline.config.export_metadata = metadata
    logger.info(f"Incremental export enabled to {dump_path}")

    try:
        with unpack_zip_to_temp_paths_if_needed(input) as docfiles:
            if not docfiles:
                print(
                    f"Error: No files found matching input patterns: {', '.join(map(str,input))}",
                    file=sys.stderr,
                )
                print(
                    "Tip: Check file paths exist in current directory or package data",
                    file=sys.stderr,
                )
                raise typer.Exit(1)

            pipeline.config.document_paths = docfiles
            pipeline.config.documents = pipeline.config.load_documents()
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(1)

    try:
        analysis, errors = asyncio.run(pipeline.run())

        if errors:
            raise typer.BadParameter(f"Pipeline execution failed:\n{errors}")

        # print cost summary to stderr (always visible)
        cost_summary = analysis.get_cost_summary()
        if cost_summary:
            # use CostSummary for overall display
            summary = CostSummary(
                total_cost=cost_summary.get("total_cost", 0.0),
                fresh_cost=cost_summary.get("fresh_cost", 0.0),
                total_prompt_tokens=cost_summary.get("total_prompt_tokens", 0),
                total_completion_tokens=cost_summary.get("total_completion_tokens", 0),
                fresh_count=cost_summary.get("fresh_count", 0),
                cached_count=cost_summary.get("cached_count", 0),
                has_unknown_costs=cost_summary.get("has_unknown_costs", False),
                all_costs_unknown=cost_summary.get("all_costs_unknown", False),
            )

            # print base summary
            print(summary.format_summary(include_breakdown=True), file=sys.stderr)

            # add total API calls count
            total_calls = cost_summary.get("fresh_count", 0) + cost_summary.get(
                "cached_count", 0
            )
            if total_calls > 0:
                print(f"  Total API calls: {total_calls}", file=sys.stderr)

            # print per-node breakdown (only if verbose mode is enabled)
            if logging.getLogger("soak").level <= logging.INFO:
                for node_name, node_data in cost_summary.get("by_node", {}).items():
                    if node_data["cost"] > 0 or node_data.get("prompt_tokens", 0) > 0:
                        unknown_marker = "*" if node_data.get("has_unknown") else ""
                        # show cache info for this node if available
                        if node_data.get("cached_count", 0) > 0:
                            cache_info = f" ({node_data['fresh_count']} fresh, {node_data['cached_count']} cached)"
                        elif node_data.get("fresh_count", 0) > 0:
                            cache_info = " (fresh)"
                        else:
                            cache_info = ""

                        print(
                            f"  {node_name}{unknown_marker}: ${node_data['cost']:.4f} "
                            f"({node_data['prompt_tokens']:,} in / "
                            f"{node_data['completion_tokens']:,} out){cache_info}",
                            file=sys.stderr,
                        )

    except Exception as e:
        if _pdb_on_exception:
            traceback.print_exc()
            # unwrap ExceptionGroups from async TaskGroup to get the actual exception
            exc = e
            while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                exc = exc.exceptions[0]
            pdb.post_mortem(exc.__traceback__)
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Pipeline execution error:\n{error_msg}")

    # analysis.config.llm_credentials.api_key = (
    #     analysis.config.llm_credentials.api_key[:5] + "***"
    # )

    # remove documents from output if not requested
    if not include_documents:
        analysis.config.documents = []

    # generate output content using cached model dump (computed once, reused for HTML)
    jsoncontent = json.dumps(analysis.get_model_dump())

    # Generate HTML for each template (reuses cached model dump)
    # Use the original pipeline directly - don't serialize/deserialize as it corrupts node outputs
    pipeline_for_html = analysis
    html_outputs = generate_all_html_outputs(
        pipeline_for_html, template, on_error="raise"
    )

    # Write output files
    typer.echo(f"Writing output files")

    json_path = dump_path / f"{output}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(jsoncontent)
        logger.info(f"Wrote json output to {json_path}")

    for tmpl in template:
        template_stem = Path(resolve_template(tmpl)).stem
        html_filename = dump_path / f"{output}_{template_stem}.html"
        logger.info(
            f"Wrote HTML output with template '{template_stem}' to {html_filename}"
        )
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(html_outputs[tmpl])

    # Note: Execution details already exported incrementally to {dump_path}
    # during pipeline execution (nodes exported as they finished)
    logger.info(f"✓ Execution dump saved to: {dump_path}")


@app.command()
def dump(
    input_json: str = typer.Argument(
        ..., help="Path to JSON file from a saved pipeline run"
    ),
    output_folder: str = typer.Option(
        None,
        "--output-folder",
        "-o",
        help="Output folder path. If not specified, creates <input_stem>_dump/ in current directory",
    ),
    template: list[str] = typer.Option(
        None,
        "--template",
        "-t",
        help="Generate HTML files in dump using template(s) (can be used multiple times)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output folder if it already exists",
    ),
):
    """Dump detailed DAG execution to folder structure for inspection."""

    # Load the pipeline
    pipeline = load_pipeline_json(input_json)

    # Determine output folder
    input_path = Path(input_json)
    if output_folder is None:
        output_folder = f"{input_path.stem}_dump"

    output_path = Path(output_folder)

    # Check if output folder exists
    if output_path.exists():
        if not force:
            print(
                f"Error: Output folder already exists: {output_path}", file=sys.stderr
            )
            print(f"Use --force/-F to overwrite", file=sys.stderr)
            raise typer.Exit(1)
        else:
            logger.info(f"Removing existing folder: {output_path}")
            shutil.rmtree(output_path)

    # Export execution details
    metadata = {
        "source_file": str(input_path),
        "command": f"soak dump {input_json}",
    }

    logger.info(f"Dumping execution details to {output_path}")
    pipeline.export_execution(output_path, metadata=metadata)

    # Generate HTML files if template(s) specified
    if template:
        html_outputs = generate_all_html_outputs(pipeline, template, on_error="warn")
        for tmpl, html_content in html_outputs.items():
            template_stem = Path(resolve_template(tmpl)).stem
            html_path = output_path / f"analysis_{template_stem}.html"
            logger.info(f"Writing HTML with template '{template_stem}' to {html_path}")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

    logger.info(f"✓ Execution dump saved to: {output_path}")
    if template:
        logger.info(f"✓ Generated {len(template)} HTML file(s) in dump folder")


def resolve_analysis_path(input_path: str) -> Path:
    """Resolve an input path to a JSON file.

    If input_path is a JSON file, return it directly.
    If input_path is a directory, look for a JSON file with the same name
    (optionally minus '_dump' suffix).
    """
    path = Path(input_path)

    # if it's already a file, return it
    if path.is_file():
        return path

    # if it's a directory, look for matching JSON file
    if path.is_dir():
        dir_name = path.name

        # try exact match first: folder/folder.json
        exact_match = path / f"{dir_name}.json"
        if exact_match.exists():
            return exact_match

        # try without _dump suffix: folder_dump/folder.json
        if dir_name.endswith("_dump"):
            base_name = dir_name[:-5]  # remove '_dump'
            stripped_match = path / f"{base_name}.json"
            if stripped_match.exists():
                return stripped_match

        # list available JSON files for error message
        json_files = list(path.glob("*.json"))
        if json_files:
            json_names = ", ".join(f.name for f in json_files[:5])
            raise typer.BadParameter(
                f"Could not find '{dir_name}.json' or '{dir_name[:-5] if dir_name.endswith('_dump') else dir_name}.json' "
                f"in {path}. Available JSON files: {json_names}"
            )
        raise typer.BadParameter(f"No JSON files found in directory: {path}")

    raise typer.BadParameter(f"Path does not exist: {path}")


@app.command()
def compare(
    input_files: list[str] = typer.Argument(
        ...,
        help="JSON files or directories containing QualitativeAnalysis results to compare (minimum 2)",
    ),
    output: str = typer.Option(
        "comparison.html",
        "--output",
        "-o",
        help="Output HTML file path",
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
        "{name}: {description}",
        "--embedding-template",
        "-e",
        envvar="SOAK_EMBEDDING_TEMPLATE",
        help="Python format string for generating theme embeddings. Available: {name}, {description}.",
    ),
    embedding_model: str = typer.Option(
        "text-embedding-3-large",
        "--embedding-model",
        envvar="SOAK_EMBEDDING_MODEL",
        help="Embedding model (use 'local/model-name' for sentence-transformers, e.g., 'local/all-MiniLM-L6-v2')",
    ),
    k: float = typer.Option(
        1.0,
        "--k",
        "-k",
        envvar="SOAK_K",
        help="Shepard similarity decay parameter (default: 1.0, higher = steeper decay)",
    ),
    reg_m: float = typer.Option(
        0.2,
        "--reg-m",
        "-K",
        envvar="SOAK_REG_M",
        help="OT mass penalty (K). Fixed value for cross-analysis comparability. Lower = more selective matching.",
    ),
    distance: str = typer.Option(
        "angular",
        "--distance",
        "-d",
        envvar="SOAK_DISTANCE",
        help="Distance metric for similarity: angular (default), cosine, shepard. Angular is preferred as it satisfies the triangle inequality and avoids the high-similarity compression of cosine.",
    ),
):
    """Compare multiple analysis results and generate comparison report."""
    from jinja2 import Environment, FileSystemLoader

    from .comparators.similarity_comparator import SimilarityComparator
    from .helpers import format_exception_concise
    from .models import QualitativeAnalysis, QualitativeAnalysisPipeline

    if len(input_files) < 2:
        logger.error("At least 2 JSON files required for comparison")
        raise typer.Exit(1)
        
    logger.info(f"Loading {len(input_files)} analyses...")
    analyses = []

    for input_arg in input_files:
        input_path = resolve_analysis_path(input_arg)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            # Load as pipeline and extract result
            if "nodes" in data:
                # This is a pipeline
                pipeline = QualitativeAnalysisPipeline.model_validate(data)
                analysis = pipeline.result()
                # Set name from filename (more useful for comparisons)
                analysis.name = input_path.stem
            else:
                # This is already a QualitativeAnalysis

                analysis = QualitativeAnalysis.model_validate(data)
                if not analysis.name or analysis.name == analysis.sha256()[:8]:
                    analysis.name = input_path.stem

            analyses.append(analysis)
        except Exception as e:
            if _pdb_on_exception:
                traceback.print_exc()
                pdb.post_mortem()
            error_msg = format_exception_concise(e)
            raise typer.BadParameter(f"Error loading {input_path}:\n{error_msg}")
        logger.info(
            f"  Loaded: {analysis.name} ({len(analysis.themes)} themes, {len(analysis.codes)} codes)"
        )

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
            "embedding_template": embedding_template,
            "embedding_model": embedding_model,
            "k": k,
            "reg_m": reg_m,
            "distance": distance,
        },
    )

    logger.info("Generating HTML report...")
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    env.globals["enumerate"] = enumerate
    template = env.get_template("comparison.html")
    html_content = template.render(comparison=comparison)

    logger.info(f"Writing to {output}...")
    with open(output, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"✓ Comparison report saved to: {output}")

    # Print summary statistics
    logger.debug("\nSummary:")
    for key, comp in comparison.by_comparisons().items():
        logger.debug(f"  {key}:")
        logger.debug(
            f"    Hit Rate A: {comp['stats']['hit_rate_a']:.1%}, Hit Rate B: {comp['stats']['hit_rate_b']:.1%}, Fidelity: {comp['stats']['fidelity']:.3f}"
        )


@app.command()
def show(
    item_type: str = typer.Argument(
        ..., help="Type of item to show: 'pipeline', 'template', or a pipeline name directly"
    ),
    name: str = typer.Argument(
        None,
        help="Name of pipeline or template to show (optional - lists all if omitted)",
    ),
):
    """Show the contents of a built-in pipeline or template.

    Examples:
        soak show pipeline          # List all available pipelines
        soak show template          # List all available templates
        soak show pipeline demo     # Show contents of demo pipeline
        soak show template default  # Show contents of default template
        soak show demo              # Show contents of demo pipeline (shorthand)

    You can redirect output to create your own custom versions:
        soak show pipeline demo > my_pipeline.soak
        soak show template default > my_template.html
    """

    # if item_type is not a known type, treat it as a pipeline name
    if item_type not in ["pipeline", "template"]:
        # shift arguments: item_type becomes the name, search for pipeline
        name = item_type
        item_type = "pipeline"

        # check current directory first, then built-in
        cwd_candidate = Path.cwd() / name
        cwd_candidate_soak = Path.cwd() / f"{name}.soak"

        if cwd_candidate.is_file():
            print(cwd_candidate.read_text(), file=sys.stdout)
            return
        if cwd_candidate_soak.is_file():
            print(cwd_candidate_soak.read_text(), file=sys.stdout)
            return

        # fall through to built-in pipeline search below

    if item_type == "pipeline":
        items_dir = PIPELINE_DIR
        extensions = [".soak"]
    else:  # template
        items_dir = Path(__file__).parent / "templates"
        extensions = [".html"]

    # List all if no name provided
    if name is None:
        logger.info(f"Available {item_type}s:")
        for ext in extensions:
            for item_path in sorted(items_dir.glob(f"*{ext}")):
                logger.info(f"  {item_path.stem}")
        logger.info(f"\nUsage: soak show {item_type} <name>")
        return

    # Find the item
    candidates = [items_dir / name]
    for ext in extensions:
        candidates.extend(
            [
                items_dir / f"{name}{ext}",
            ]
        )

    item_path = None
    for cand in candidates:
        if cand.is_file():
            item_path = cand
            break

    if item_path is None:
        logger.error(f"{item_type} '{name}' not found in {items_dir}")
        logger.debug(f"Tried: {[str(c) for c in candidates]}")
        raise typer.Exit(1)

    # Print contents to stdout
    print(item_path.read_text(), file=sys.stdout)


@app.command()
def compare_strings(
    xlsx_file: str = typer.Argument(
        ..., help="Path to XLSX file with strings to compare"
    ),
    col_a: str = typer.Option(
        "A", "--col-a", help="First column name or letter (default: A)"
    ),
    col_b: str = typer.Option(
        "B", "--col-b", help="Second column name or letter (default: B)"
    ),
    threshold: float = typer.Option(
        0.6,
        "--threshold",
        envvar="SOAK_THRESHOLD",
        help="Similarity threshold for matching",
    ),
    embedding_model: str = typer.Option(
        "text-embedding-3-large",
        "--embedding-model",
        envvar="SOAK_EMBEDDING_MODEL",
        help="Embedding model (use 'local/model-name' for sentence-transformers, e.g., 'local/all-MiniLM-L6-v2')",
    ),
    k: float = typer.Option(
        1.0,
        "--k",
        envvar="SOAK_K",
        help="Shepard similarity decay parameter (default: 1.0, higher = steeper decay)",
    ),
):
    """Compare similarity of two lists of strings from an XLSX file.

    Reads column A and column B from the specified XLSX file and computes
    similarity metrics between the two lists using embeddings.

    Example:
        soak compare-strings data.xlsx --threshold 0.7
        soak compare-strings data.xlsx --col-a "List1" --col-b "List2"
    """
    import pandas as pd
    import numpy as np

    from .comparators.similarity_comparator import compare_result_similarity
    from .models import QualitativeAnalysis, Theme

    # Read the xlsx file - check current dir first, then package soak-data
    xlsx_path = Path(xlsx_file)
    if not xlsx_path.exists():
        # Try resolving relative to package directory
        package_data_path = Path(__file__).parent / xlsx_file
        if package_data_path.exists():
            xlsx_path = package_data_path
        else:
            logger.error(f"File not found: {xlsx_file}")
            raise typer.Exit(1)

    logger.info(f"Reading {xlsx_path}...")
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        logger.error(f"Error reading XLSX file: {e}")
        raise typer.Exit(1)

    # Get the two columns
    try:
        list_a = df[col_a].dropna().astype(str).tolist()
        list_b = df[col_b].dropna().astype(str).tolist()
    except KeyError as e:
        logger.error(f"Column not found: {e}")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        raise typer.Exit(1)

    if not list_a or not list_b:
        logger.error("One or both columns are empty")
        raise typer.Exit(1)

    logger.info(f"Column {col_a}: {len(list_a)} items")
    logger.info(f"Column {col_b}: {len(list_b)} items")

    # Check for API credentials if using api backend
    if not embedding_model.startswith("local/"):
        check_and_prompt_credentials(Path.cwd())

    # Create minimal QualitativeAnalysis objects with themes from the string lists
    themes_a = [Theme(name=s, description=s, code_slugs=[]) for s in list_a]
    themes_b = [Theme(name=s, description=s, code_slugs=[]) for s in list_b]

    analysis_a = QualitativeAnalysis(name=col_a, themes=themes_a)
    analysis_b = QualitativeAnalysis(name=col_b, themes=themes_b)

    logger.info("Computing similarity...")

    # Compare using existing function
    result = compare_result_similarity(
        analysis_a,
        analysis_b,
        threshold=threshold,
        embedding_template="{name}",
        embedding_model=embedding_model,
        k=k,
    )

    # Print results to stdout
    print("\n" + "=" * 60)
    print(f"Similarity Comparison: {col_a} vs {col_b}")
    print("=" * 60)
    print(f"\nThreshold: {threshold}")
    print(f"Embedding: {embedding_model}")
    print(f"Shepard k: {k}")

    # Print the two lists being compared
    print(f"\n{col_a} ({len(list_a)} items):")
    for i, item in enumerate(list_a):
        print(f"  {i}: {item}")

    print(f"\n{col_b} ({len(list_b)} items):")
    for i, item in enumerate(list_b):
        print(f"  {i}: {item}")
    print(f"\nCoverage (Hit Rates):")
    print(f"  Hit Rate A: {result['hit_rate_a']:.1%}")
    print(f"  Hit Rate B: {result['hit_rate_b']:.1%}")
    print(f"  Jaccard:    {result['jaccard']:.3f}")

    print(f"\nFidelity (Mean Best-Match Similarity):")
    print(f"  {col_a} → {col_b}: {result['mean_max_sim_a_to_b']:.3f}")
    print(f"  {col_b} → {col_a}: {result['mean_max_sim_b_to_a']:.3f}")
    print(f"  Fidelity:          {result['fidelity']:.3f}")

    print(f"\nHungarian Matching (1-to-1):")
    print(f"  Coverage A:   {result['hungarian']['thresholded_metrics']['coverage_a']:.1%}")
    print(f"  Coverage B:   {result['hungarian']['thresholded_metrics']['coverage_b']:.1%}")
    print(f"  True Jaccard: {result['hungarian']['thresholded_metrics']['true_jaccard']:.3f}")

    print(f"\nOptimal Transport (many-to-many):")
    print(f"  Shared Mass:        {result['ot']['shared_mass']:.3f}")
    print(f"  Shared Mass Rel:    {result['ot'].get('shared_mass_relative', 0):.3f} (0=random, 1=perfect)")
    print(f"  Avg Cost:           {result['ot']['avg_cost']:.3f}")
    print(f"  Unmatched Mass:     {result['ot']['unmatched_mass']:.3f}")
    print("\n" + "=" * 60)

    # Helper function for printing matrices
    def print_matrix(matrix, title, list_a, list_b):
        print(f"\n{title} ({len(list_a)} x {len(list_b)}):")
        print("-" * 60)

        # Truncate long strings for display
        def truncate(s, max_len=30):
            return s if len(s) <= max_len else s[:max_len-3] + "..."

        # Show a summary view if matrix is large
        if len(list_a) > 10 or len(list_b) > 10:
            print(f"(Matrix too large to display fully)")
            print(f"\nTop 5 most similar pairs:")
            # Find top N pairs
            flat_indices = np.argsort(matrix.ravel())[::-1][:5]
            for idx in flat_indices:
                i, j = np.unravel_index(idx, matrix.shape)
                print(f"  {truncate(list_a[i])} <-> {truncate(list_b[j])}: {matrix[i, j]:.3f}")
        else:
            # Print full matrix for small datasets
            print(f"\n{' ':30s} | ", end="")
            for item_b in list_b:
                print(f"{truncate(item_b, 15):15s} ", end="")
            print()
            print("-" * (32 + len(list_b) * 16))

            for i, item_a in enumerate(list_a):
                print(f"{truncate(item_a, 30):30s} | ", end="")
                for j in range(len(list_b)):
                    print(f"{matrix[i, j]:15.3f} ", end="")
                print()

    # Print all similarity matrices
    print_matrix(result['similarity_matrix'], "Cosine Similarity Matrix", list_a, list_b)
    print_matrix(result['angle_similarity_matrix'], "Angular Similarity Matrix", list_a, list_b)
    print_matrix(result['shepard_similarity_matrix'], f"Shepard Similarity Matrix (k={k})", list_a, list_b)


@app.command()
def coverage(
    analysis_file: str = typer.Argument(
        ..., help="Path to JSON file containing QualitativeAnalysis or Pipeline results"
    ),
    output: str = typer.Option(
        "coverage", "--output", "-o", help="Output file path (without extension)"
    ),
    documents: list[Path] = typer.Option(
        None, "--documents", "-d", help="Override documents (default: use documents from analysis)"
    ),
    groups_file: Path = typer.Option(
        None, "--groups", "-g", help="XLSX file with 'filename' and 'group' columns for grouping"
    ),
    format: str = typer.Option(
        "all", "--format", "-f", help="Output format: json, html, csv, or all"
    ),
    chunk_size: int = typer.Option(
        500, "--chunk-size", help="Size of document chunks"
    ),
    overlap_size: int = typer.Option(
        50, "--overlap", help="Overlap between chunks"
    ),
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

    from .coverage import ThemeCoverageAnalyzer
    from .coverage.analyzer import (
        compute_within_doc_variation,
        generate_absolute_chunk_heatmap,
        generate_chunk_heatmap,
        generate_coverage_heatmap,
        generate_group_heatmap,
        generate_normalized_chunk_heatmap,
        generate_theme_trajectories,
    )
    from .document_utils import unpack_zip_to_temp_paths_if_needed
    from .helpers import format_exception_concise

    # validate format
    valid_formats = {"json", "html", "csv", "all"}
    if format not in valid_formats:
        logger.error(f"Invalid format '{format}'. Must be one of: {valid_formats}")
        raise typer.Exit(1)

    # validate split_unit
    valid_split_units = {"words", "tokens", "chars"}
    if split_unit not in valid_split_units:
        logger.error(f"Invalid split-unit '{split_unit}'. Must be one of: {valid_split_units}")
        raise typer.Exit(1)

    # validate aggregation
    valid_aggregations = {"max", "mean", "p95"}
    if aggregation not in valid_aggregations:
        logger.error(f"Invalid aggregation '{aggregation}'. Must be one of: {valid_aggregations}")
        raise typer.Exit(1)

    # validate embed
    valid_embed_sources = {"quotes", "themes", "both"}
    if embed not in valid_embed_sources:
        logger.error(f"Invalid embed source '{embed}'. Must be one of: {valid_embed_sources}")
        raise typer.Exit(1)

    # check for API credentials if using api backend
    # check for API credentials if using api backend
    if not embedding_model.startswith("local/"):
        check_and_prompt_credentials(Path.cwd())

    # load analysis JSON
    analysis_path = Path(analysis_file)
    if not analysis_path.exists():
        logger.error(f"Analysis file not found: {analysis_file}")
        raise typer.Exit(1)

    logger.info(f"Loading analysis from {analysis_file}...")
    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # extract QualitativeAnalysis from pipeline or direct
    from .models import QualitativeAnalysis, QualitativeAnalysisPipeline

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
        if _pdb_on_exception:
            traceback.print_exc()
            pdb.post_mortem()
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Error loading analysis:\n{error_msg}")

    if not analysis.themes:
        logger.error("Analysis has no themes. Cannot compute coverage.")
        raise typer.Exit(1)

    logger.info(f"Loaded analysis with {len(analysis.themes)} themes")

    # load documents - from CLI, or from analysis config
    from .models.base import TrackedItem
    from .document_utils import extract_text

    doc_items = []

    if documents:
        # use documents specified via --documents option
        logger.info("Using documents specified via --documents")
        try:
            with unpack_zip_to_temp_paths_if_needed(documents) as docfiles:
                if not docfiles:
                    logger.error(f"No files found matching input patterns: {', '.join(map(str, documents))}")
                    raise typer.Exit(1)

                for docpath, doc_metadata in docfiles:
                    content = extract_text(str(docpath))
                    if isinstance(content, list):
                        logger.warning(f"Skipping spreadsheet file {docpath} - use text documents for coverage analysis")
                        continue
                    doc_id = Path(docpath).stem
                    doc_items.append(
                        TrackedItem(
                            content=content,
                            id=doc_id,
                            sources=[doc_id],
                            metadata={"filename": Path(docpath).name, "original_path": str(docpath), **doc_metadata},
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
                logger.info(f"Loading documents from {len(doc_paths)} paths in analysis config")
                for path_entry in doc_paths:
                    # handle both tuple format [(path, metadata), ...] and plain paths
                    if isinstance(path_entry, (list, tuple)):
                        docpath = path_entry[0]
                        doc_metadata = path_entry[1] if len(path_entry) > 1 else {}
                    else:
                        docpath = path_entry
                        doc_metadata = {}

                    if not Path(docpath).exists():
                        logger.warning(f"Document not found (may have moved): {docpath}")
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
                            metadata={"filename": Path(docpath).name, "original_path": str(docpath), **(doc_metadata or {})},
                        )
                    )

    if not doc_items:
        logger.error("No documents found. Specify --documents or ensure analysis contains document paths.")
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

            groups = dict(zip(groups_df["filename"].astype(str), groups_df["group"].astype(str)))
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

    # generate outputs
    logger.info("Generating outputs...")

    # generate heatmaps (both continuous and thresholded versions)
    heatmap = generate_coverage_heatmap(result)
    heatmap_thresholded = generate_coverage_heatmap(result, threshold=threshold)
    group_heatmap = generate_group_heatmap(result) if result.groups else None
    chunk_heatmap = generate_chunk_heatmap(result, window_size=chunk_window)

    # generate new chunk-level visualizations
    normalized_heatmap = generate_normalized_chunk_heatmap(result, n_bins=n_bins)
    normalized_heatmap_zscore = generate_normalized_chunk_heatmap(result, n_bins=n_bins, z_score=True)
    absolute_heatmap = generate_absolute_chunk_heatmap(result, n_bins_target=n_bins)
    theme_trajectories = generate_theme_trajectories(result, n_bins=n_bins)
    within_doc_variation = compute_within_doc_variation(result, n_bins=n_bins)

    # write outputs based on format
    if format in {"json", "all"}:
        json_path = f"{output}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        logger.info(f"Wrote JSON output to {json_path}")

    if format in {"csv", "all"}:
        csv_path = f"{output}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(result.to_csv())
        logger.info(f"Wrote CSV output to {csv_path}")

    if format in {"html", "all"}:
        html_path = f"{output}.html"

        # render HTML template
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("coverage.html")
        html_content = template.render(
            result=result,
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
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Wrote HTML output to {html_path}")

    logger.info(f"Coverage analysis complete")


def check_and_prompt_credentials(cwd: Path) -> tuple[str | None, str | None]:
    """Check for LLM credentials and prompt user if missing.

    Returns:
        Tuple of (api_key, base_url)
    """
    from .helpers import load_env_file, save_env_file

    env_path = cwd / ".env"

    # First check environment variables
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_BASE")

    # If not in env, check .env file
    if not api_key or not base_url:
        env_vars = load_env_file(env_path)
        api_key = api_key or env_vars.get("LLM_API_KEY")
        base_url = base_url or env_vars.get("LLM_API_BASE")

    # Prompt for missing credentials
    missing = []
    if not api_key:
        missing.append("LLM_API_KEY")
    if not base_url:
        missing.append("LLM_API_BASE")

    if missing:
        logger.warning("Missing required LLM credentials:")
        for var in missing:
            logger.warning(f"   - {var}")

        response = typer.confirm(
            "Would you like to provide them now?", default=True, err=True
        )
        if not response:
            logger.error("Cannot proceed without LLM credentials")
            raise typer.Exit(1)

        # Load existing .env vars to preserve them
        env_vars = load_env_file(env_path)

        if not api_key:
            api_key = typer.prompt("Enter LLM_API_KEY", err=True)
            env_vars["LLM_API_KEY"] = api_key

        if not base_url:
            default_url = "https://api.openai.com/v1"
            base_url = typer.prompt("Enter LLM_API_BASE", default=default_url, err=True)
            env_vars["LLM_API_BASE"] = base_url

        # Save to .env file
        save_env_file(env_path, env_vars)
        logger.info(f"✓ Credentials saved to {env_path}")

        # Set in current process environment
        os.environ["LLM_API_KEY"] = api_key
        os.environ["LLM_API_BASE"] = base_url

    return api_key, base_url


@app.command()
def tui():
    """Open the terminal user interface for building commands."""
    from trogon import Trogon
    from typer.main import get_group

    Trogon(get_group(app), app_name="soak", command="tui").run()


def setup_logging(verbose: int):
    """Configure logging levels based on verbosity (0=WARNING, 1=INFO, 2+=DEBUG)."""

    # map verbosity to levels
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    if verbose > 0:
        logging.basicConfig(
            level=logging.WARNING,  # Root level stays at WARNING
            format="%(name)s: %(message)s",
        )

    # Set package-specific levels
    pkg = __name__.split(".")[0]
    logging.getLogger(pkg).setLevel(level)
    logging.getLogger("struckdown").setLevel(level)


def main_with_default_command():
    """Entry point that makes 'run' the default command."""
    # Make 'run' the default command when first non-flag arg is not a built-in command
    # Find first non-flag argument
    first_arg = None
    first_arg_index = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if not arg.startswith("-"):
            first_arg = arg
            first_arg_index = i
            break

    # If first non-flag arg exists and is not a command, inject 'run' before it
    if first_arg and first_arg not in COMMANDS:
        sys.argv.insert(first_arg_index, "run")

    app()


if __name__ == "__main__":
    main_with_default_command()
