"""Command-line interface for running qualitative analysis pipelines."""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import typer
from jinja2 import Environment, FileSystemLoader
from struckdown import LLMCredentials
from trogon.typer import init_tui

from .comparators.similarity_comparator import SimilarityComparator
from .document_utils import unpack_zip_to_temp_paths_if_needed
from .models import QualitativeAnalysis, QualitativeAnalysisPipeline
from .specs import load_template_bundle

app = typer.Typer()
init_tui(app)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
logger = logging.getLogger(__name__)


def format_exception_concise(exc: Exception) -> str:
    """Format exception with minimal context - just the error and relevant code location."""
    tb = traceback.extract_tb(exc.__traceback__)

    # Get the last frame (where the error actually occurred)
    if tb:
        last_frame = tb[-1]
        error_msg = f"\n{type(exc).__name__}: {exc}\n"
        error_msg += f"  File: {last_frame.filename}:{last_frame.lineno}\n"
        error_msg += f"  In: {last_frame.name}\n"
        if last_frame.line:
            error_msg += f"    {last_frame.line}\n"
    else:
        error_msg = f"\n{type(exc).__name__}: {exc}\n"

    return error_msg


PIPELINE_DIR = Path(__file__).parent / "pipelines"


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load environment variables from a .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    return env_vars


def save_env_file(env_path: Path, env_vars: dict[str, str]):
    """Save environment variables to a .env file."""
    with open(env_path, "w") as f:
        for key, value in env_vars.items():
            f.write(f'{key}="{value}"\n')


def check_and_prompt_credentials(cwd: Path) -> tuple[str | None, str | None]:
    """Check for LLM credentials and prompt user if missing.

    Returns:
        Tuple of (api_key, base_url)
    """
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
        print("\n⚠️  Missing required LLM credentials:", file=sys.stderr)
        for var in missing:
            print(f"   - {var}", file=sys.stderr)
        print(file=sys.stderr)

        response = typer.confirm("Would you like to provide them now?", default=True)
        if not response:
            print("Error: Cannot proceed without LLM credentials", file=sys.stderr)
            raise typer.Exit(1)

        # Load existing .env vars to preserve them
        env_vars = load_env_file(env_path)

        if not api_key:
            api_key = typer.prompt("Enter LLM_API_KEY")
            env_vars["LLM_API_KEY"] = api_key

        if not base_url:
            default_url = "https://api.openai.com/v1"
            base_url = typer.prompt("Enter LLM_API_BASE", default=default_url)
            env_vars["LLM_API_BASE"] = base_url

        # Save to .env file
        save_env_file(env_path, env_vars)
        print(f"\n✓ Credentials saved to {env_path}", file=sys.stderr)

        # Set in current process environment
        os.environ["LLM_API_KEY"] = api_key
        os.environ["LLM_API_BASE"] = base_url

    return api_key, base_url


def setup_logging(verbose: int):

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
            format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
        )

    # Set package-specific levels
    pkg = __name__.split(".")[0]
    logging.getLogger(pkg).setLevel(level)
    logging.getLogger("struckdown").setLevel(level)


@app.callback()
def main(
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,  # allows -v -vv -vvv
        help="Increase verbosity (-v=INFO, -vv=DEBUG)",
    )
):
    """Global options."""
    setup_logging(verbose)


def sanitize_for_filename(text: str) -> str:
    """Remove or replace characters that are problematic in filenames.

    Args:
        text: Input string

    Returns:
        Sanitized string safe for use in filenames
    """
    # Replace problematic characters with underscores
    dangerous_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]
    result = text
    for char in dangerous_chars:
        result = result.replace(char, "_")
    return result


def hash_run_config(
    input_files: list[str],
    model_name: str | None = None,
    context: list[str] | None = None,
    template: str | None = None,
    length: int = 4,
) -> str:
    """Generate a short hash from run configuration.

    Args:
        input_files: List of input file paths
        model_name: Model name if specified
        context: Context overrides if specified
        template: Template name if specified
        length: Length of hash to return (default: 4)

    Returns:
        Short hash string of specified length (hex chars only - always safe)
    """
    # Build configuration string from all parameters
    parts = [
        "files:" + "|".join(sorted(input_files)),
    ]
    if model_name:
        parts.append(f"model:{model_name}")
    if context:
        parts.append("context:" + "|".join(sorted(context)))
    if template:
        parts.append(f"template:{template}")

    config_str = "||".join(parts)
    hash_obj = hashlib.sha256(config_str.encode("utf-8"))
    return hash_obj.hexdigest()[:length]


def resolve_pipeline(pipeline: str, localdir: Path, pipelinedir: Path) -> Path:
    candidates = [
        localdir / pipeline,
        localdir / f"{pipeline}.yaml",
        localdir / f"{pipeline}.yml",
        pipelinedir / f"{pipeline}",
        pipelinedir / f"{pipeline}.yaml",
        pipelinedir / f"{pipeline}.yml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Pipeline file not found. Tried: {candidates}")


def generate_html_output(pipeline: QualitativeAnalysisPipeline, template: str) -> str:
    """Generate HTML output from a pipeline using specified template.

    Args:
        pipeline: QualitativeAnalysisPipeline instance
        template: Template name or path

    Returns:
        Rendered HTML string
    """
    template_path = resolve_template(template)
    return pipeline.to_html(template_path=str(template_path))


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
    print(f"Error: Template not found: {template}", file=sys.stderr)
    print(f"Looked in: {templates_dir}", file=sys.stderr)
    raise typer.Exit(1)


@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Pipeline name to run (e.g., 'poc')"),
    input_files: list[str] = typer.Argument(
        ..., help="File patterns or zip files (supports globs like '*.txt')"
    ),
    model_name: str = typer.Option(None, help="LLM model name"),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path name (without extensions) (stdout if not specified)",
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format: json or html"
    ),
    template: str = typer.Option(
        "pipeline.html",
        "--template",
        "-t",
        help="Template name (in soak/templates) or path to custom HTML template",
    ),
    include_documents: bool = typer.Option(
        False, "--include-documents", help="Include original documents in output"
    ),
    context: list[str] = typer.Option(
        None,
        "--context",
        "-c",
        help="Override context variables (format: key=value, can be used multiple times)",
    ),
    dump: bool = typer.Option(
        False, "--dump", "-d", help="Dump execution details to folder"
    ),
    dump_folder: str = typer.Option(
        None,
        "--dump-folder",
        help="Custom dump folder path (default: <output>_dump or <pipeline>_<timestamp>_dump)",
    ),
    force: bool = typer.Option(
        False, "--force", "-F", help="Overwrite dump folder if it already exists"
    ),
):
    """Run a pipeline on input files."""

    # Check if output files exist (if output specified)
    if output:
        output_json = Path(f"{output}.json")
        output_html = Path(f"{output}.html")

        if output_json.exists() or output_html.exists():
            if not force:
                existing = []
                if output_json.exists():
                    existing.append(str(output_json))
                if output_html.exists():
                    existing.append(str(output_html))
                print(
                    f"Error: Output file(s) already exist: {', '.join(existing)}",
                    file=sys.stderr,
                )
                print(f"Use --force/-f to overwrite", file=sys.stderr)
                raise typer.Exit(1)
            else:
                logger.warning(
                    f"Overwriting existing output files: {output}.json, {output}.html"
                )

    # Check and prompt for credentials first
    api_key, base_url = check_and_prompt_credentials(Path.cwd())

    # Save the string pipeline argument before it gets reassigned
    pipeline_arg = pipeline

    pipyml = resolve_pipeline(pipeline_arg, Path.cwd(), PIPELINE_DIR)
    logger.info(f"Loading pipeline from {pipyml}")

    try:
        pipeline = load_template_bundle(pipyml)
    except ValueError as e:
        raise typer.BadParameter(f"Pipeline validation error: {e}")

    # Override default_context with CLI-provided values
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
    pipeline.config.llm_credentials = LLMCredentials(
        api_key=api_key,
        base_url=base_url,
    )

    with unpack_zip_to_temp_paths_if_needed(input_files) as docfiles:
        pipeline.config.document_paths = docfiles
        pipeline.config.documents = pipeline.config.load_documents()

    try:
        analysis, errors = asyncio.run(pipeline.run())
        if errors:
            raise typer.BadParameter(f"Pipeline execution failed:\n{errors}")

    except Exception as e:
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Pipeline execution error:\n{error_msg}")

    # analysis.config.llm_credentials.api_key = (
    #     analysis.config.llm_credentials.api_key[:5] + "***"
    # )

    # remove documents from output if not requested
    if not include_documents:
        analysis.config.documents = []

    # generate output content based on format
    jsoncontent = analysis.model_dump_json()

    # this roundtrip to dict is weird, but needed because of a bug in how ChatterResult is reserialised?
    try:
        htmlcontent = generate_html_output(
            QualitativeAnalysisPipeline.model_validate(analysis.model_dump()), template
        )
    except Exception as e:
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Error generating HTML output:\n{error_msg}")
    if output is None:
        if format == "json":
            print(jsoncontent, file=sys.stdout)
        elif format == "html":
            print(htmlcontent, file=sys.stdout)
        else:
            raise typer.BadParameter(
                "Format must be 'json' or 'html' or specify output file name"
            )

    else:
        logger.info(f"Writing output to {output}.json and {output}.html")
        with open(output + ".html", "w", encoding="utf-8") as f:
            f.write(htmlcontent)
        with open(output + ".json", "w", encoding="utf-8") as f:
            f.write(jsoncontent)

    # Dump execution details if requested
    if dump:
        # Generate unique ID for this run
        config_hash = hash_run_config(
            input_files=input_files,
            model_name=model_name,
            context=context,
            template=template,
        )

        if dump_folder is None:
            if output:
                dump_folder = f"{output}_dump"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize pipeline name for safe folder name (use pipeline.name from the loaded object)
                safe_pipeline = sanitize_for_filename(pipeline.name)
                dump_folder = f"{safe_pipeline}_{config_hash}_{timestamp}_dump"

        dump_path = Path(dump_folder)

        # Check if dump folder exists
        if dump_path.exists():
            if not force:
                print(
                    f"\nError: Dump folder already exists: {dump_path}", file=sys.stderr
                )
                print(f"Use --force/-f to overwrite", file=sys.stderr)
                raise typer.Exit(1)
            else:
                logger.info(f"Removing existing dump folder: {dump_path}")

                shutil.rmtree(dump_path)

        # Build command string for metadata
        cmd_parts = [f"soak run {pipeline_arg}"] + input_files
        if output:
            cmd_parts.append(f"--output {output}")
        cmd_parts.append(f"--model-name {model_name}")
        if context:
            for ctx in context:
                cmd_parts.append(f"--context {ctx}")

        metadata = {
            "command": " ".join(cmd_parts),
            "pipeline_file": str(pipyml),
            "model_name": model_name,
            "template": template,
            "unique_id": config_hash,
        }
        if context:
            metadata["context_overrides"] = dict([c.split("=", 1) for c in context])

        logger.info(f"Dumping execution details to {dump_path}")
        analysis.export_execution(dump_path, metadata=metadata)
        print(f"✓ Execution dump saved to: {str(dump_path)}", file=sys.stderr)


@app.command()
def compare(
    input_files: list[str] = typer.Argument(
        ...,
        help="JSON files containing QualitativeAnalysis results to compare (minimum 2)",
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
        help="Similarity threshold for matching themes",
    ),
    method: str = typer.Option(
        "umap",
        "--method",
        help="Dimensionality reduction method (umap, mds, pca)",
    ),
    label: str = typer.Option(
        "{name}",
        "--label",
        "-l",
        help="Python format string for theme labels in visualizations. Available: {name}, {description}",
    ),
    embedding_template: str = typer.Option(
        "{name}",
        "--embedding-template",
        "-e",
        help="Python format string for generating theme embeddings. Available: {name}, {description}",
    ),
):
    """Compare multiple analysis results and generate comparison report."""

    if len(input_files) < 2:
        print("Error: At least 2 JSON files required for comparison", file=sys.stderr)
        raise typer.Exit(1)

    logger.info(f"Loading {len(input_files)} analyses...")
    analyses = []

    for input_json in input_files:
        input_path = Path(input_json)
        if not input_path.exists():
            print(f"Error: File not found: {input_json}", file=sys.stderr)
            raise typer.Exit(1)

        with open(input_json, "r", encoding="utf-8") as f:
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
            error_msg = format_exception_concise(e)
            raise typer.BadParameter(f"Error loading {input_json}:\n{error_msg}")
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
        },
    )

    logger.info("Generating HTML report...")
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("comparison.html")
    html_content = template.render(comparison=comparison)

    logger.info(f"Writing to {output}...")
    with open(output, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✓ Comparison report saved to: {output}")

    # Print summary statistics
    logger.debug("\nSummary:")
    for key, comp in comparison.by_comparisons().items():
        logger.debug(f"  {key}:")
        logger.debug(
            f"    F1: {comp['stats']['f1']:.3f}, Similarity F1: {comp['stats']['similarity_f1']:.3f}"
        )


@app.command()
def show(
    item_type: str = typer.Argument(
        ..., help="Type of item to show: 'pipeline' or 'template'"
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

    You can redirect output to create your own custom versions:
        soak show pipeline demo > my_pipeline.yaml
        soak show template default > my_template.html
    """

    if item_type not in ["pipeline", "template"]:
        print(
            f"Error: item_type must be 'pipeline' or 'template', got: {item_type}",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    if item_type == "pipeline":
        items_dir = PIPELINE_DIR
        extensions = [".yaml", ".yml"]
    else:  # template
        items_dir = Path(__file__).parent / "templates"
        extensions = [".html"]

    # List all if no name provided
    if name is None:
        print(f"Available {item_type}s:", file=sys.stderr)
        for ext in extensions:
            for item_path in sorted(items_dir.glob(f"*{ext}")):
                print(f"  {item_path.stem}", file=sys.stderr)
        print(f"\nUsage: soak show {item_type} <name>", file=sys.stderr)
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
        print(f"Error: {item_type} '{name}' not found in {items_dir}", file=sys.stderr)
        print(f"Tried: {[str(c) for c in candidates]}", file=sys.stderr)
        raise typer.Exit(1)

    # Print contents to stdout
    print(item_path.read_text(), file=sys.stdout)


@app.command()
def export(
    input_json: str = typer.Argument(
        ..., help="Path to JSON file containing QualitativeAnalysis"
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (without extension). If not specified, replaces input .json with .html",
    ),
    template: str = typer.Option(
        "pipeline.html",
        "--template",
        "-t",
        help="Template name (in soak/templates) or path to custom HTML template",
    ),
):
    """Export a saved analysis result to HTML."""

    # Load the JSON file
    input_path = Path(input_json)
    if not input_path.exists():
        print(f"Error: File not found: {input_json}", file=sys.stderr)
        raise typer.Exit(1)

    logger.info(f"Loading analysis from {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try to parse as QualitativeAnalysisPipeline first (which has to_html method)
    # If that fails, try QualitativeAnalysis
    try:
        pipeline = QualitativeAnalysisPipeline.model_validate(data)
        logger.info("Loaded as QualitativeAnalysisPipeline")
        htmlcontent = generate_html_output(pipeline, template)

    except Exception as e:
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Could not load as pipeline:\n{error_msg}")

    # Determine output path
    if output is None:
        output = str(input_path.with_suffix(""))

    output_html = output + ".html"
    logger.info(f"Writing HTML to {output_html}")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(htmlcontent)

    print(f"Successfully exported to {output_html}")


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
    force: bool = typer.Option(
        False,
        "--force",
        "-F",
        help="Overwrite output folder if it already exists",
    ),
):
    """Dump detailed DAG execution to folder structure for inspection."""

    # Load the JSON file
    input_path = Path(input_json)
    if not input_path.exists():
        print(f"Error: File not found: {input_json}", file=sys.stderr)
        raise typer.Exit(1)

    logger.info(f"Loading pipeline from {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load as QualitativeAnalysisPipeline
    try:
        pipeline = QualitativeAnalysisPipeline.model_validate(data)
        logger.info(f"Loaded pipeline: {pipeline.name}")
    except Exception as e:
        error_msg = format_exception_concise(e)
        raise typer.BadParameter(f"Could not load as pipeline:\n{error_msg}")

    # Determine output folder
    if output_folder is None:
        output_folder = f"{input_path.stem}_dump"

    output_path = Path(output_folder)

    # Check if output folder exists
    if output_path.exists():
        if not force:
            print(
                f"Error: Output folder already exists: {output_path}", file=sys.stderr
            )
            print(f"Use --force/-f to overwrite", file=sys.stderr)
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

    print(f"\n✓ Execution dump saved to: {output_path}")


if __name__ == "__main__":
    app()
