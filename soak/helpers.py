import hashlib
import traceback
from pathlib import Path


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
