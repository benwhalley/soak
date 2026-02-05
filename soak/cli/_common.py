"""Shared utilities for CLI commands."""

import importlib.metadata
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from ..models import QualitativeAnalysisPipeline

# suppress litellm's unawaited coroutine warning (benign, occurs on event loop shutdown)
warnings.filterwarnings(
    "ignore",
    message="coroutine 'Logging.async_success_handler' was never awaited",
    category=RuntimeWarning,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
logger = logging.getLogger(__name__)

# module-level flag for pdb on exception
_pdb_on_exception = False

PIPELINE_DIR = Path(__file__).parent.parent / "pipelines"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
COMMANDS = {"run", "export", "compare", "show", "tui", "coverage", "test", "calibrate"}


def set_pdb_on_exception(value: bool) -> None:
    """Set the global pdb on exception flag."""
    global _pdb_on_exception
    _pdb_on_exception = value


def get_pdb_on_exception() -> bool:
    """Get the global pdb on exception flag."""
    return _pdb_on_exception


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


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(get_soak_version())
        raise typer.Exit()


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
    from ..helpers import format_exception_concise

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
    from ..helpers import format_exception_concise
    from ..models import QualitativeAnalysisPipeline

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

    if template_path.exists():
        # Absolute or relative path that exists
        logger.info(f"Using custom template: {template_path}")
        return template_path

    # Try to find it in soak/templates directory
    candidate = TEMPLATES_DIR / template
    if candidate.exists():
        logger.info(f"Using template: {template} from soak/templates")
        return candidate

    # If template doesn't have a file extension, try adding .html
    if not template_path.suffix:
        candidate_with_ext = TEMPLATES_DIR / f"{template}.html"
        if candidate_with_ext.exists():
            logger.info(f"Using template: {template}.html from soak/templates")
            return candidate_with_ext

    # Template not found
    logger.error(f"Template not found: {template}")
    logger.error(f"Looked in: {TEMPLATES_DIR}")
    raise typer.Exit(1)


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


def check_and_prompt_credentials(cwd: Path) -> tuple[str | None, str | None]:
    """Check for LLM credentials and prompt user if missing.

    Returns:
        Tuple of (api_key, base_url)
    """
    from ..helpers import load_env_file, save_env_file

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
        logger.info(f"Credentials saved to {env_path}")

        # Set in current process environment
        os.environ["LLM_API_KEY"] = api_key
        os.environ["LLM_API_BASE"] = base_url

    return api_key, base_url


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
    pkg = "soak"
    logging.getLogger(pkg).setLevel(level)
    logging.getLogger("struckdown").setLevel(level)
