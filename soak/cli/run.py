"""Run command for executing pipelines."""

import json
import logging
import pdb
import sys
import traceback
from pathlib import Path

import typer

from ._common import (PIPELINE_DIR, check_and_prompt_credentials,
                      generate_all_html_outputs, get_pdb_on_exception,
                      load_pipeline_json, resolve_template)

logger = logging.getLogger(__name__)


def run(
    pipeline: str = typer.Argument(..., help="Pipeline name to run (e.g., 'zs')"),
    input: list[Path] = typer.Argument(
        None,
        help="Input file paths (supports globs). For spreadsheets with --document-template, "
        "each row becomes a document.",
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
    document_template: Path = typer.Option(
        None,
        "--document-template",
        "-T",
        help="Jinja2 template file for generating documents from spreadsheet rows. "
        "Use {{ column_name }} syntax to insert values.",
    ),
    document_names_from: str = typer.Option(
        None,
        "--document-names-from",
        help="Column name to use for document names when using --document-template. "
        "If not specified, documents are named 'Row 1', 'Row 2', etc.",
    ),
):
    """Run a pipeline on input files."""
    from ..api import CredentialsError, RunError
    from ..api import run as api_run
    from ..helpers import format_exception_concise, resolve_pipeline

    # validate that input files are provided
    if not input:
        logger.error("No input files specified.")
        raise typer.Exit(1)

    # validate that no bare directories are passed
    for inp in input:
        if inp.is_dir():
            print(
                f"Error: '{inp}' is a directory. Use a glob pattern instead (e.g., '{inp}/*.txt')",
                file=sys.stderr,
            )
            raise typer.Exit(1)

    # handle document template mode (mail merge)
    temp_doc_dir = None
    if document_template:
        import tempfile

        from ..tabular import (generate_documents, parse_tabular_file,
                               validate_template)

        # read template
        if not document_template.exists():
            logger.error(f"Template file not found: {document_template}")
            raise typer.Exit(1)

        template_content = document_template.read_text(encoding="utf-8")

        # find spreadsheet inputs
        spreadsheet_inputs = [p for p in input if p.suffix.lower() in (".csv", ".xlsx")]
        if not spreadsheet_inputs:
            logger.error("--document-template requires a CSV or XLSX input file")
            raise typer.Exit(1)

        if len(spreadsheet_inputs) > 1:
            logger.warning(
                f"Multiple spreadsheets found, using first: {spreadsheet_inputs[0]}"
            )

        spreadsheet_path = spreadsheet_inputs[0]
        logger.info(f"Generating documents from {spreadsheet_path} using template")

        # parse spreadsheet
        try:
            parse_result = parse_tabular_file(spreadsheet_path)
        except ValueError as e:
            logger.error(f"Failed to parse spreadsheet: {e}")
            raise typer.Exit(1)

        logger.info(
            f"Found {parse_result.total_rows} rows, {len(parse_result.columns)} columns"
        )

        # validate template
        errors = validate_template(template_content, parse_result.columns)
        validation_errors = [e for e in errors if not e.startswith("Warning:")]
        if validation_errors:
            for err in validation_errors:
                logger.error(f"Template error: {err}")
            raise typer.Exit(1)

        # generate documents
        documents = generate_documents(
            template_content,
            parse_result,
            source_file=spreadsheet_path.name,
            name_column=document_names_from,
            skip_empty=False,
        )

        if not documents:
            logger.error("No documents generated from spreadsheet")
            raise typer.Exit(1)

        logger.info(f"Generated {len(documents)} documents from spreadsheet rows")

        # write to temp directory
        temp_doc_dir = tempfile.mkdtemp(prefix="soak_tabular_")
        generated_paths = []
        for doc in documents:
            # sanitise filename
            safe_name = "".join(
                c if c.isalnum() or c in " -_" else "_" for c in doc.name
            )
            doc_path = Path(temp_doc_dir) / f"{safe_name}.txt"
            doc_path.write_text(doc.content, encoding="utf-8")
            generated_paths.append(doc_path)

        # replace input with generated files
        input = generated_paths
        logger.info(f"Documents written to temp directory: {temp_doc_dir}")

    # validate mutually exclusive options
    if sample is not None and head is not None:
        logger.error("--sample and --head are mutually exclusive")
        raise typer.Exit(1)

    # auto-detect progress bar setting
    if progress is None:
        is_debug = logging.getLogger().level <= logging.DEBUG
        progress = sys.stderr.isatty() and not is_debug

    # determine output name
    pipeline_arg = pipeline
    if output is None:
        pipyml = resolve_pipeline(pipeline_arg, Path.cwd(), PIPELINE_DIR)
        output = Path(pipyml).stem
        logger.info(f"Using default output name: {output}")

    dump_path = Path(f"{output}_dump")
    existing_json = dump_path / f"{output}.json"

    # check which templates already exist vs new ones requested
    existing_html_files = []
    new_templates = []
    for tmpl in template:
        template_stem = Path(resolve_template(tmpl)).stem
        html_path = dump_path / f"{output}_{template_stem}.html"
        if html_path.exists():
            existing_html_files.append(html_path)
        else:
            new_templates.append(tmpl)

    # template-only mode: if JSON exists, no -f, and new templates requested
    if existing_json.exists() and not force and new_templates:
        logger.info(f"Found existing analysis at {existing_json}")
        logger.info(
            f"Rendering {len(new_templates)} new template(s): {', '.join(new_templates)}"
        )

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

    # check for conflicts when running full pipeline
    if dump_path.exists() and not force:
        print(
            f"Error: Output folder already exists: {dump_path}/",
            file=sys.stderr,
        )
        print("Use --force/-f to overwrite", file=sys.stderr)
        raise typer.Exit(1)

    # check and prompt for credentials (CLI-specific interactive prompt)
    check_and_prompt_credentials(Path.cwd())

    # parse context variables
    context_dict = None
    if context:
        context_dict = {}
        for item in context:
            if "=" not in item:
                print(
                    f"Error: Context variable must be in format 'key=value', got: {item}",
                    file=sys.stderr,
                )
                raise typer.Exit(1)
            key, value = item.split("=", 1)
            context_dict[key] = value
            logger.info(f"Set context variable: {key}={value}")

    # parse model configuration
    model_config = None
    if model:
        model_config = {}
        for m in model:
            if "=" in m:
                alias, model_id = m.split("=", 1)
                model_config[alias.strip()] = model_id.strip()
                logger.info(f"Set model alias: {alias}={model_id}")
            else:
                model_config["default"] = m.strip()
                logger.info(f"Set default model: {m}")

    # run pipeline via API
    try:
        result = api_run(
            pipeline_arg,
            [str(p) for p in input],
            context=context_dict,
            output=output,
            model=model_config,
            seed=seed,
            sample=sample,
            head=head,
            skip_nodes=skip_node,
            stop_at=stop_at,
            embedding_model=embeddings,
            timeout=timeout,
            progress=progress,
            force=force,
            include_documents=include_documents,
            instructor_mode=instructor_mode,
        )
    except CredentialsError as e:
        logger.error(str(e))
        raise typer.Exit(1)
    except RunError as e:
        logger.error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        if get_pdb_on_exception():
            traceback.print_exc()
            exc = e
            while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                exc = exc.exceptions[0]
            pdb.post_mortem(exc.__traceback__)
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Pipeline execution error:\n{error_msg}")

    if result.errors:
        raise typer.BadParameter(f"Pipeline execution failed:\n{result.errors}")

    # print cost summary
    if result.cost_summary:
        print(result.cost_summary.format(include_breakdown=True), file=sys.stderr)

        # print per-node breakdown if verbose
        if logging.getLogger("soak").level <= logging.INFO:
            for node_name, node_data in result.cost_summary.by_node.items():
                if node_data["cost"] > 0 or node_data.get("prompt_tokens", 0) > 0:
                    unknown_marker = "*" if node_data.get("has_unknown") else ""
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

    # generate HTML outputs
    html_outputs = generate_all_html_outputs(
        result.pipeline, template, on_error="raise"
    )

    # write output files
    typer.echo("Writing output files")

    json_path = dump_path / f"{output}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.to_json())
        logger.info(f"Wrote json output to {json_path}")

    for tmpl in template:
        template_stem = Path(resolve_template(tmpl)).stem
        html_filename = dump_path / f"{output}_{template_stem}.html"
        logger.info(
            f"Wrote HTML output with template '{template_stem}' to {html_filename}"
        )
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(html_outputs[tmpl])

    logger.info(f"Execution dump saved to: {dump_path}")

    # clean up temp directory from document template mode
    if temp_doc_dir:
        import shutil

        shutil.rmtree(temp_doc_dir, ignore_errors=True)
        logger.debug(f"Cleaned up temp directory: {temp_doc_dir}")
