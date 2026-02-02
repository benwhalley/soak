"""Command-line interface for running qualitative analysis pipelines."""

import asyncio
import importlib.metadata
import json
import logging
import os
import pdb
import shutil
import sys
import traceback
import warnings
from pathlib import Path

import typer

# suppress litellm's unawaited coroutine warning (benign, occurs on event loop shutdown)
warnings.filterwarnings(
    "ignore",
    message="coroutine 'Logging.async_success_handler' was never awaited",
    category=RuntimeWarning,
)

app = typer.Typer(name="soak", pretty_exceptions_show_locals=False)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
logger = logging.getLogger(__name__)

# module-level flag for pdb on exception
_pdb_on_exception = False

PIPELINE_DIR = Path(__file__).parent / "pipelines"
COMMANDS = {"run", "export", "compare", "show", "tui", "coverage", "test"}


def _derive_input_source(docfiles: list) -> str:
    """Derive a summary string from document paths.

    Attempts to find common directory and file extension pattern.
    E.g., [("data/a.txt", {}), ("data/b.txt", {})] -> "data/*.txt"
    """
    if not docfiles:
        return ""

    # extract paths from tuples
    paths = [Path(p[0]) if isinstance(p, tuple) else Path(p) for p in docfiles]

    # find common parent directory
    parents = [p.parent for p in paths]
    if parents and all(p == parents[0] for p in parents):
        common_dir = str(parents[0])
    else:
        # try to find longest common prefix
        parent_strs = [str(p) for p in parents]
        common_prefix = os.path.commonpath(parent_strs) if parent_strs else ""
        common_dir = common_prefix if common_prefix else "."

    # find common extension
    extensions = {p.suffix for p in paths}
    if len(extensions) == 1 and extensions != {""}:
        ext_pattern = f"*{list(extensions)[0]}"
    else:
        ext_pattern = "*"

    return f"{common_dir}/{ext_pattern}"


def get_soak_version() -> str:
    """Get soak package version."""
    try:
        return importlib.metadata.version("soaking")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def generate_compare_filename(
    input_names: list[str],
    embedding_model: str = "text-embedding-3-large",
    threshold: float = 0.6,
    similarity: str = "angular",
    shepard_k: float = 1.0,
    rescale: str = "clip",
    rescale_min: float = 0.5,
    rescale_max: float = 0.9,
    rescale_percentiles: str = "5,95",
    rescale_sigmoid: str = "0.7,10",
    rescale_temp: float = 0.5,
    no_paraphrase_bound: bool = False,
    extension: str = ".html",
) -> str:
    """Generate an auto-filename for compare output based on inputs and options.

    Creates a concise but unique filename encoding key parameters.
    Format: {inputs}_{options}.{ext}

    Examples:
        PROF_vs_NOTPROF_e5b_t75.html
        A_vs_B_vs_C_cos_t60_rsada.html
    """
    # abbreviate input names (remove common suffixes, shorten)
    def abbreviate(name: str) -> str:
        for suffix in ["_dump", "_analysis", "_results", "_output"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        if len(name) > 12:
            name = name[:12]
        return name

    abbrev_inputs = [abbreviate(n) for n in input_names]
    inputs_part = "_vs_".join(abbrev_inputs)

    # build options part (only non-default values)
    opts = []

    # embedding model: strip "local/" prefix and take last part after "/"
    # e.g. "local/intfloat/e5-base" -> "e5-base"
    # "text-embedding-3-large" (default) -> omit
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

    # rescale method and parameters
    if rescale == "clip":
        if rescale_min != 0.5 or rescale_max != 0.9:
            opts.append(f"clip{int(rescale_min*100)}-{int(rescale_max*100)}")
    elif rescale == "adaptive":
        if rescale_percentiles != "5,95":
            opts.append(f"ada{rescale_percentiles.replace(',', '-')}")
        else:
            opts.append("ada")
    elif rescale == "sigmoid":
        if rescale_sigmoid != "0.7,10":
            opts.append(f"sig{rescale_sigmoid.replace(',', '-').replace('.', '')}")
        else:
            opts.append("sig")
    elif rescale == "temperature":
        if rescale_temp != 0.5:
            opts.append(f"temp{str(rescale_temp).replace('.', '')}")
        else:
            opts.append("temp")
    elif rescale == "rank":
        opts.append("rank")
    elif rescale == "off":
        opts.append("rsoff")

    # no paraphrase bound
    if no_paraphrase_bound:
        opts.append("nopara")

    # combine parts
    if opts:
        filename = f"{inputs_part}_{'_'.join(opts)}{extension}"
    else:
        filename = f"{inputs_part}{extension}"

    # sanitise: replace spaces and special chars
    filename = filename.replace(" ", "_").replace("/", "-")

    return filename


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(get_soak_version())
        raise typer.Exit()


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
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
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
    pipeline: str = typer.Argument(..., help="Pipeline name to run (e.g., 'zs')"),
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
        ["simple.html"],
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
    from .helpers import (format_exception_concise, hash_run_config,
                          resolve_pipeline)
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

    # Check for existing dump folder and outputs
    dump_path = Path(f"{output}_dump")
    existing_json = dump_path / f"{output}.json"

    # Check which templates already exist vs new ones requested
    existing_html_files = []
    new_templates = []
    for tmpl in template:
        template_stem = Path(resolve_template(tmpl)).stem
        html_path = dump_path / f"{output}_{template_stem}.html"
        if html_path.exists():
            existing_html_files.append(html_path)
        else:
            new_templates.append(tmpl)

    # Template-only mode: if JSON exists, no -f, and new templates requested
    if existing_json.exists() and not force and new_templates:
        logger.info(f"Found existing analysis at {existing_json}")
        logger.info(
            f"Rendering {len(new_templates)} new template(s): {', '.join(new_templates)}"
        )

        # Load existing pipeline and render new templates only
        pipeline_for_html = load_pipeline_json(str(existing_json))
        html_outputs = generate_all_html_outputs(
            pipeline_for_html, new_templates, on_error="raise"
        )

        for tmpl in new_templates:
            template_stem = Path(resolve_template(tmpl)).stem
            html_filename = dump_path / f"{output}_{template_stem}.html"
            logger.info(
                f"Writing HTML with template '{template_stem}' to {html_filename}"
            )
            with open(html_filename, "w", encoding="utf-8") as f:
                f.write(html_outputs[tmpl])

        logger.info(f"✓ Generated {len(new_templates)} new template(s)")
        raise typer.Exit(0)

    # Check for conflicts when running full pipeline
    if dump_path.exists() and not force:
        print(
            f"Error: Output folder already exists: {dump_path}/",
            file=sys.stderr,
        )
        print(f"Use --force/-f to overwrite", file=sys.stderr)
        raise typer.Exit(1)
    elif dump_path.exists() and force:
        logger.warning(f"Overwriting existing output folder: {dump_path}/")

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

    # Remove existing dump folder if force is enabled (dump_path declared earlier)
    if dump_path.exists() and force:
        logger.info(f"Removing existing dump folder: {dump_path}")
        shutil.rmtree(dump_path)

    # Build command string for metadata
    cmd_parts = ["soak", "run", pipeline_arg]
    for inp in input:
        cmd_parts.append(str(inp))
    cmd_parts.extend(["-o", output])
    if model_name:
        cmd_parts.extend(["--model", model_name])
    if sample is not None:
        cmd_parts.extend(["--sample", str(sample)])
    if head is not None:
        cmd_parts.extend(["--head", str(head)])
    if seed is not None:
        cmd_parts.extend(["--seed", str(seed)])
    if context:
        for ctx in context:
            cmd_parts.extend(["-c", ctx])
    for tmpl in template:
        cmd_parts.extend(["-t", tmpl])

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
        "sample_n": sample,
        "head_n": head,
        "seed": seed,
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
            pipeline.config.input_source = _derive_input_source(docfiles)
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
    prompt_path = Path(__file__).parent / "pipelines" / "make_labels.sd"
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
    lines.append("─" * 70)
    lines.append("COVERAGE (Hit Rates)")
    lines.append("─" * 70)
    lines.append(
        f"  Hit Rate {name_a}: {result['hit_rate_a']:.1%}  (items with ≥1 match above threshold)"
    )
    lines.append(f"  Hit Rate {name_b}: {result['hit_rate_b']:.1%}")
    lines.append(
        f"  Jaccard:         {result['jaccard']:.3f}  (proportion of pairs above threshold)"
    )

    # fidelity
    lines.append("")
    lines.append("─" * 70)
    lines.append("FIDELITY (Mean Best-Match Similarity)")
    lines.append("─" * 70)
    lines.append(f"  {name_a} → {name_b}: {result['mean_max_sim_a_to_b']:.3f}")
    lines.append(f"  {name_b} → {name_a}: {result['mean_max_sim_b_to_a']:.3f}")
    lines.append(f"  Fidelity:        {result['fidelity']:.3f}  (harmonic mean)")

    # hungarian matching
    lines.append("")
    lines.append("─" * 70)
    lines.append("HUNGARIAN MATCHING (Optimal 1-to-1 Assignment)")
    lines.append("─" * 70)
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
    lines.append("─" * 70)
    lines.append("OPTIMAL TRANSPORT (Many-to-Many Alignment)")
    lines.append("─" * 70)

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
                lines.append(f"    Paraphrase ceiling:   {ceiling_sim:.1%} shared mass, {ceiling_align:.1%} alignment")
            if floor_sim is not None:
                floor_align = 1 - (floor_cost or 0)
                lines.append(f"    Word-salad floor:     {floor_sim:.1%} shared mass, {floor_align:.1%} alignment")

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
            shared_mass = ot_data.get('shared_mass', 0)
            null_shared_mass = ot_data.get('null_shared_mass_mean')
            paraphrase_ceiling = ot_data.get('paraphrase_upper_bound')

            if paraphrase_ceiling is not None and null_shared_mass is not None:
                lines.append(f"    Shared Mass:          {shared_mass:.1%}  (floor: {null_shared_mass:.1%}, ceiling: {paraphrase_ceiling:.1%})")
            else:
                lines.append(f"    Shared Mass:          {shared_mass:.1%}")

            # paraphrase-scaled metrics (if available)
            pct_ceiling = ot_data.get('shared_mass_pct_of_ceiling')
            pct_improvement = ot_data.get('shared_mass_improvement_vs_null')
            if pct_ceiling is not None:
                lines.append(f"      % of ceiling:       {pct_ceiling:.0%}  (vs paraphrase upper bound)")
            if pct_improvement is not None:
                lines.append(f"      vs word-salad:      {pct_improvement:.0%} better, relative to ceiling")

            # semantic alignment (1 - cost, so higher = better)
            avg_cost = ot_data.get('avg_cost', 0)
            alignment = 1 - avg_cost
            alignment_ceiling = ot_data.get('alignment_paraphrase_ceiling')
            alignment_floor = ot_data.get('alignment_null_floor')

            if alignment_ceiling is not None and alignment_floor is not None:
                lines.append(f"    Semantic Alignment:   {alignment:.1%}  (floor: {alignment_floor:.1%}, ceiling: {alignment_ceiling:.1%})")
            else:
                lines.append(f"    Semantic Alignment:   {alignment:.1%}  (quality of matches)")

            # alignment paraphrase-scaled metrics
            align_pct_ceiling = ot_data.get('alignment_pct_of_ceiling')
            align_improvement = ot_data.get('alignment_improvement_vs_null')
            if align_pct_ceiling is not None:
                lines.append(f"      % of ceiling:       {align_pct_ceiling:.0%}  (vs paraphrase upper bound)")
            if align_improvement is not None:
                lines.append(f"      vs word-salad:      {align_improvement:.0%} better, relative to ceiling")

    if elbow_k:
        lines.append(f"")
        lines.append(f"  Knee detected at K = {elbow_k} (point of diminishing returns)")

    # text sankey diagram
    lines.append("")
    lines.append("─" * 70)
    lines.append(
        f"TRANSPORT FLOWS (cost labels: low <K, med K-2K, high >2K where K={default_k})"
    )
    lines.append("─" * 70)

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
    lines.append("─" * 70)
    lines.append("TRANSPORT MASS MATRIX (%)")
    lines.append("─" * 70)

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
    lines.append("─" * 70)
    lines.append(f"SIMILARITY MATRIX ({sim_metric.upper()} -- used for all metrics)")
    lines.append("─" * 70)

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
        return s[: max_len - 1] + "…" if len(s) > max_len else s

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
    header = " " * (max_label_len + 2) + "│"
    for label in labels_b:
        header += f" {label:>{max_label_len}}"
    lines.append(f"  {header}")

    # separator
    sep = "─" * (max_label_len + 1) + "┼" + "─" * (len(labels_b) * (max_label_len + 1))
    lines.append(f"  {sep}")

    # data rows
    for i, label_a in enumerate(labels_a):
        row = f"{label_a:>{max_label_len}} │"
        for j in range(n_b):
            val = fmt.format(matrix[i, j])
            row += f" {val:>{max_label_len}}"
        lines.append(f"  {row}")


@app.command()
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
    rescale: str = typer.Option(
        "clip",
        "--rescale",
        envvar="SOAK_RESCALE",
        help="Rescaling method: 'off', 'clip' (default), 'adaptive', 'sigmoid', 'temperature', 'rank'. "
             "For clip: use --rescale-min/--rescale-max. For adaptive: use --rescale-percentiles. "
             "For sigmoid: use --rescale-sigmoid. For temperature: use --rescale-temp.",
    ),
    rescale_min: float = typer.Option(
        0.5,
        "--rescale-min",
        envvar="SOAK_RESCALE_MIN",
        help="Floor for clip method (default: 0.5). Values below become 0.",
    ),
    rescale_max: float = typer.Option(
        0.9,
        "--rescale-max",
        envvar="SOAK_RESCALE_MAX",
        help="Ceiling for clip method (default: 0.9). Values above become 1.",
    ),
    rescale_percentiles: str = typer.Option(
        "5,95",
        "--rescale-percentiles",
        envvar="SOAK_RESCALE_PERCENTILES",
        help="Percentile bounds for adaptive method (default: '5,95'). Format: 'lower,upper'.",
    ),
    rescale_sigmoid: str = typer.Option(
        "0.7,10",
        "--rescale-sigmoid",
        envvar="SOAK_RESCALE_SIGMOID",
        help="Sigmoid params (default: '0.7,10'). Format: 'center,steepness'. Higher steepness = sharper.",
    ),
    rescale_temp: float = typer.Option(
        0.5,
        "--rescale-temp",
        envvar="SOAK_RESCALE_TEMP",
        help="Temperature for power method (default: 0.5). <1 sharpens, >1 flattens.",
    ),
    filter_threshold: float = typer.Option(
        0.05,
        "--filter-threshold",
        envvar="SOAK_FILTER_THRESHOLD",
        help="Filter weak edges from transport plan. Edges with mass < threshold * row/col sum are removed. Default 0.05 (5%). Set to 0 to disable.",
    ),
    color_green: float = typer.Option(
        0.2,
        "--color-green",
        envvar="SOAK_COLOR_GREEN",
        help="Green color threshold as fraction of K. Links with cost < green*K appear green. Default 0.2.",
    ),
    color_red: float = typer.Option(
        1.1,
        "--color-red",
        envvar="SOAK_COLOR_RED",
        help="Red color threshold as fraction of K. Links with cost > red*K appear red. Default 1.1.",
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

    from .comparators.similarity_comparator import (SimilarityComparator,
                                                    compare_result_similarity)
    from .helpers import format_exception_concise
    from .models import QualitativeAnalysis, QualitativeAnalysisPipeline, Theme

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

    # parse rescale options
    rescale_method = rescale.lower()
    valid_methods = ["off", "clip", "adaptive", "sigmoid", "temperature", "rank"]
    if rescale_method not in valid_methods:
        logger.error(f"Invalid --rescale method: {rescale}. Must be one of: {', '.join(valid_methods)}")
        raise typer.Exit(1)

    # parse percentile bounds for adaptive method
    try:
        pct_parts = rescale_percentiles.split(",")
        rescale_lower_pct = float(pct_parts[0].strip())
        rescale_upper_pct = float(pct_parts[1].strip())
    except (ValueError, IndexError):
        logger.error(f"Invalid --rescale-percentiles format: {rescale_percentiles}. Use 'lower,upper' (e.g., '5,95').")
        raise typer.Exit(1)

    # parse sigmoid params
    try:
        sig_parts = rescale_sigmoid.split(",")
        rescale_sigmoid_center = float(sig_parts[0].strip())
        rescale_sigmoid_steepness = float(sig_parts[1].strip())
    except (ValueError, IndexError):
        logger.error(f"Invalid --rescale-sigmoid format: {rescale_sigmoid}. Use 'center,steepness' (e.g., '0.7,10').")
        raise typer.Exit(1)

    logger.info(f"Rescaling method: {rescale_method}")
    if rescale_method == "clip":
        logger.info(f"  Clip bounds: [{rescale_min}, {rescale_max}]")
    elif rescale_method == "adaptive":
        logger.info(f"  Percentile bounds: [{rescale_lower_pct}%, {rescale_upper_pct}%]")
    elif rescale_method == "sigmoid":
        logger.info(f"  Sigmoid: center={rescale_sigmoid_center}, steepness={rescale_sigmoid_steepness}")
    elif rescale_method == "temperature":
        logger.info(f"  Temperature: {rescale_temp}")

    # determine mode: strings or JSON
    if strings:
        # STRINGS MODE: compare columns from XLSX/CSV
        xlsx_path = Path(strings)
        if not xlsx_path.exists():
            # try package soak-data directory
            package_data_path = Path(__file__).parent / strings
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
                "rescale_method": rescale_method,
                "rescale_min": rescale_min,
                "rescale_max": rescale_max,
                "rescale_lower_pct": rescale_lower_pct,
                "rescale_upper_pct": rescale_upper_pct,
                "rescale_sigmoid_center": rescale_sigmoid_center,
                "rescale_sigmoid_steepness": rescale_sigmoid_steepness,
                "rescale_temp": rescale_temp,
                "filter_threshold": filter_threshold,
                "color_green": color_green,
                "color_red": color_red,
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
                rescale=rescale_method,
                rescale_min=rescale_min,
                rescale_max=rescale_max,
                rescale_percentiles=rescale_percentiles,
                rescale_sigmoid=rescale_sigmoid,
                rescale_temp=rescale_temp,
                no_paraphrase_bound=no_paraphrase_bound,
                extension=output,
            )
            logger.info(f"Auto-generated filename: {effective_output}")

        # determine if we should print stats to console
        # only print if: no output specified, or output is .txt
        output_path = Path(effective_output) if effective_output else None
        print_to_console = not effective_output or (output_path and output_path.suffix == ".txt")

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

        # save output if requested
        if effective_output:
            if output_path.suffix == ".txt":
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(all_stats_text))
                logger.info(f"✓ Statistics saved to: {effective_output}")
            else:
                # HTML output
                template_dir = Path(__file__).parent / "templates"
                env = Environment(loader=FileSystemLoader(template_dir))
                env.globals["enumerate"] = enumerate
                template = env.get_template("comparison.html")
                html_content = template.render(
                    comparison=comparison,
                    soak_version=get_soak_version(),
                    text_output="\n\n".join(all_stats_text),
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"✓ HTML report saved to: {effective_output}")

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
                if _pdb_on_exception:
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
                "rescale_method": rescale_method,
                "rescale_min": rescale_min,
                "rescale_max": rescale_max,
                "rescale_lower_pct": rescale_lower_pct,
                "rescale_upper_pct": rescale_upper_pct,
                "rescale_sigmoid_center": rescale_sigmoid_center,
                "rescale_sigmoid_steepness": rescale_sigmoid_steepness,
                "rescale_temp": rescale_temp,
                "filter_threshold": filter_threshold,
                "color_green": color_green,
                "color_red": color_red,
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
                rescale=rescale_method,
                rescale_min=rescale_min,
                rescale_max=rescale_max,
                rescale_percentiles=rescale_percentiles,
                rescale_sigmoid=rescale_sigmoid,
                rescale_temp=rescale_temp,
                no_paraphrase_bound=no_paraphrase_bound,
                extension=output,
            )
            logger.info(f"Auto-generated filename: {effective_output}")

        # determine if we should print stats to console
        # only print if: no output specified, or output is .txt
        output_path = Path(effective_output) if effective_output else Path("comparison.html")
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

        # save output
        if output_path.suffix == ".txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_stats_text))
            logger.info(f"✓ Statistics saved to: {output_path}")
        else:
            # HTML output
            logger.info("Generating HTML report...")
            template_dir = Path(__file__).parent / "templates"
            env = Environment(loader=FileSystemLoader(template_dir))
            env.globals["enumerate"] = enumerate
            template = env.get_template("comparison.html")
            html_content = template.render(
                comparison=comparison,
                soak_version=get_soak_version(),
                text_output="\n\n".join(all_stats_text),
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"✓ HTML report saved to: {output_path}")


@app.command()
def show(
    item_type: str = typer.Argument(
        ...,
        help="Type of item to show: 'pipeline', 'template', or a pipeline name directly",
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
def coverage(
    analysis_file: str = typer.Argument(
        ..., help="Path to JSON file containing QualitativeAnalysis or Pipeline results"
    ),
    output: str = typer.Option(
        "coverage", "--output", "-o", help="Output file path (without extension)"
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
    from .coverage.analyzer import (compute_within_doc_variation,
                                    generate_absolute_chunk_heatmap,
                                    generate_chunk_heatmap,
                                    generate_coverage_heatmap,
                                    generate_group_heatmap,
                                    generate_normalized_chunk_heatmap,
                                    generate_theme_trajectories)
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
    from .document_utils import extract_text
    from .models.base import TrackedItem

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
            api_key = (
                api_key.strip().strip('"').strip("'")
            )  # strip quotes from user input
            env_vars["LLM_API_KEY"] = api_key

        if not base_url:
            default_url = "https://api.openai.com/v1"
            base_url = typer.prompt("Enter LLM_API_BASE", default=default_url, err=True)
            base_url = (
                base_url.strip().strip('"').strip("'")
            )  # strip quotes from user input
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

    Trogon(get_group(app), app_name="soak").run()


@app.command()
def test():
    """Test LLM and embedding connections with current settings.

    Passes through to struckdown's `sd test` command.

    Examples:
        soak test
    """
    from struckdown.sd_cli import test as sd_test

    # check and prompt for credentials first (soak-specific)
    check_and_prompt_credentials(Path.cwd())

    # delegate to struckdown's test command directly
    sd_test()


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
