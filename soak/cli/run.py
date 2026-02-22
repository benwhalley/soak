"""Run command for executing pipelines."""

import asyncio
import json
import logging
import pdb
import shutil
import sys
import traceback
from pathlib import Path

import typer

from ._common import (PIPELINE_DIR, _derive_input_source,
                      check_and_prompt_credentials, generate_all_html_outputs,
                      get_pdb_on_exception, load_pipeline_json,
                      resolve_template)

logger = logging.getLogger(__name__)


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
    model: list[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model configuration. Use 'model_id' for default or 'alias=model_id' format (can be used multiple times)",
    ),
    embeddings: str = typer.Option(
        None,
        "--embeddings",
        "-e",
        help="Embedding model to use (e.g., 'text-embedding-3-large', 'text-embedding-3-small')",
    ),
    instructor_mode: str = typer.Option(
        "json",
        "--instructor-mode",
        envvar="SOAK_INSTRUCTOR_MODE",
        help="Instructor mode for structured outputs: 'json' (default, broad compatibility) or 'json_schema' (stricter, uses native structured outputs)",
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

    from ..document_utils import unpack_zip_to_temp_paths_if_needed
    from ..helpers import (format_exception_concise, hash_run_config,
                           resolve_pipeline)
    from ..specs import load_template_bundle

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

        logger.info(f"Generated {len(new_templates)} new template(s)")
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

    # Parse model configurations
    # Supports: --model gpt-4 (sets default) or --model best=gpt-5 (sets alias)
    model_aliases = {}
    if model:
        for m in model:
            if "=" in m:
                alias, model_id = m.split("=", 1)
                model_aliases[alias.strip()] = model_id.strip()
                logger.info(f"Set model alias: {alias}={model_id}")
            else:
                # Simple model name - set as default
                model_aliases["default"] = m.strip()
                pipeline.config.model_name = m.strip()
                logger.info(f"Set default model: {m}")

    # Pass model aliases to pipeline config (merge with existing from pipeline YAML)
    if model_aliases:
        pipeline.config.models = {**pipeline.config.models, **model_aliases}

    if seed is not None:
        pipeline.config.seed = seed
        logger.info(f"Set seed to {seed}")
    if embeddings is not None:
        pipeline.config.embedding_model = embeddings
        logger.info(f"Set embedding model to {embeddings}")
    pipeline.config.llm_credentials = LLMCredentials(
        api_key=api_key,
        base_url=base_url,
        instructor_mode=instructor_mode,
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
    pipeline.config.pdb_on_exception = get_pdb_on_exception()

    # Remove existing dump folder if force is enabled (dump_path declared earlier)
    if dump_path.exists() and force:
        logger.info(f"Removing existing dump folder: {dump_path}")
        shutil.rmtree(dump_path)

    # Build command string for metadata
    cmd_parts = ["soak", "run", pipeline_arg]
    for inp in input:
        cmd_parts.append(str(inp))
    cmd_parts.extend(["-o", output])
    if model:
        for m in model:
            cmd_parts.extend(["--model", m])
    if sample is not None:
        cmd_parts.extend(["--sample", str(sample)])
    if head is not None:
        cmd_parts.extend(["--head", str(head)])
    if seed is not None:
        cmd_parts.extend(["--seed", str(seed)])
    if embeddings is not None:
        cmd_parts.extend(["--embeddings", embeddings])
    if context:
        for ctx in context:
            cmd_parts.extend(["-c", ctx])
    for tmpl in template:
        cmd_parts.extend(["-t", tmpl])

    # Generate config hash for dump folder naming
    config_hash = hash_run_config(
        input_files=input,
        model_name=model_aliases.get("default") if model_aliases else None,
        context=context,
        template=template,
    )

    metadata = {
        "command": " ".join(cmd_parts),
        "pipeline_file": str(pipyml),
        "pipeline_version": pipeline.pipeline_version,
        "model_aliases": model_aliases or {},
        "templates": template,
        "unique_id": config_hash,
        "sample_n": sample,
        "head_n": head,
        "seed": seed,
        "embedding_model": embeddings,
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
        if get_pdb_on_exception():
            traceback.print_exc()
            # unwrap ExceptionGroups from async TaskGroup to get the actual exception
            exc = e
            while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                exc = exc.exceptions[0]
            pdb.post_mortem(exc.__traceback__)
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Pipeline execution error:\n{error_msg}")

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
    logger.info(f"Execution dump saved to: {dump_path}")
