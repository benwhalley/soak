"""Show/list implementation for soak API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from ..cli._common import PIPELINE_DIR, TEMPLATES_DIR


class ShowError(Exception):
    """Error during show/list operations."""

    pass


def list_pipelines() -> list[str]:
    """List all available built-in pipelines.

    Returns:
        List of pipeline names (without .soak extension)

    Example:
        >>> from soak import api
        >>> pipelines = api.list_pipelines()
        >>> print(pipelines)
        ['demo', 'thematic_analysis/zs', ...]
    """
    pipelines = []
    for path in sorted(PIPELINE_DIR.glob("**/*.soak")):
        rel_path = path.relative_to(PIPELINE_DIR)
        name = str(rel_path.with_suffix(""))
        pipelines.append(name)
    return pipelines


def list_templates() -> list[str]:
    """List all available built-in templates.

    Returns:
        List of template names (without .html extension)

    Example:
        >>> from soak import api
        >>> templates = api.list_templates()
        >>> print(templates)
        ['simple', 'pipeline', ...]
    """
    templates = []
    for path in sorted(TEMPLATES_DIR.glob("**/*.html")):
        # skip partials (files starting with _)
        if path.name.startswith("_"):
            continue
        rel_path = path.relative_to(TEMPLATES_DIR)
        name = str(rel_path.with_suffix(""))
        templates.append(name)
    return templates


def get_pipeline(
    name: str,
    cwd: Optional[Path] = None,
) -> str:
    """Get the contents of a pipeline by name.

    Searches in order:
    1. Current working directory (or cwd parameter)
    2. Built-in pipelines directory

    Args:
        name: Pipeline name (e.g., "zs", "demo", or "thematic_analysis/zs")
        cwd: Working directory for local pipeline search

    Returns:
        Pipeline file contents as string

    Raises:
        ShowError: If pipeline not found

    Example:
        >>> from soak import api
        >>> content = api.get_pipeline("demo")
        >>> print(content[:100])
    """
    cwd = cwd or Path.cwd()

    # check current directory first
    local_candidates = [
        cwd / name,
        cwd / f"{name}.soak",
    ]
    for candidate in local_candidates:
        if candidate.is_file():
            return candidate.read_text()

    # check built-in pipelines
    builtin_candidates = [
        PIPELINE_DIR / name,
        PIPELINE_DIR / f"{name}.soak",
    ]
    # also search subfolders
    for path in PIPELINE_DIR.glob(f"**/{name}.soak"):
        builtin_candidates.append(path)

    for candidate in builtin_candidates:
        if isinstance(candidate, Path) and candidate.is_file():
            return candidate.read_text()

    raise ShowError(f"Pipeline '{name}' not found")


def get_template(
    name: str,
    cwd: Optional[Path] = None,
) -> str:
    """Get the contents of a template by name.

    Searches in order:
    1. Current working directory (or cwd parameter)
    2. Built-in templates directory

    Args:
        name: Template name (e.g., "simple", "pipeline")
        cwd: Working directory for local template search

    Returns:
        Template file contents as string

    Raises:
        ShowError: If template not found

    Example:
        >>> from soak import api
        >>> content = api.get_template("simple")
    """
    cwd = cwd or Path.cwd()

    # check current directory first
    local_candidates = [
        cwd / name,
        cwd / f"{name}.html",
    ]
    for candidate in local_candidates:
        if candidate.is_file():
            return candidate.read_text()

    # check built-in templates
    builtin_candidates = [
        TEMPLATES_DIR / name,
        TEMPLATES_DIR / f"{name}.html",
    ]
    # also search subfolders
    for path in TEMPLATES_DIR.glob(f"**/{name}.html"):
        builtin_candidates.append(path)

    for candidate in builtin_candidates:
        if isinstance(candidate, Path) and candidate.is_file():
            return candidate.read_text()

    raise ShowError(f"Template '{name}' not found")
