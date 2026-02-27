"""Render implementation for soak API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from ..models import QualitativeAnalysisPipeline


class RenderError(Exception):
    """Error during rendering."""

    pass


def render(
    analysis: Union[str, Path, "QualitativeAnalysisPipeline"],
    templates: Union[str, list[str]] = "simple.html",
    output: Optional[Union[str, Path]] = None,
) -> dict[str, str]:
    """Render an analysis with HTML templates.

    Args:
        analysis: JSON file path or QualitativeAnalysisPipeline object
        templates: Template name(s) to render
        output: Output directory (writes files if provided)

    Returns:
        Dict mapping template name to rendered HTML content

    Raises:
        RenderError: If rendering fails
    """
    from ..cli._common import (
        generate_html_output,
        load_pipeline_json,
        resolve_template,
    )

    # normalize templates to list
    if isinstance(templates, str):
        templates = [templates]

    # load pipeline if needed
    if isinstance(analysis, (str, Path)):
        pipeline = load_pipeline_json(str(analysis))
    else:
        pipeline = analysis

    # render each template
    html_outputs = {}
    for tmpl in templates:
        try:
            html_outputs[tmpl] = generate_html_output(pipeline, tmpl)
        except Exception as e:
            raise RenderError(f"Error rendering template '{tmpl}': {e}")

    # write files if output specified
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for tmpl in templates:
            template_stem = Path(resolve_template(tmpl)).stem
            html_path = output_dir / f"analysis_{template_stem}.html"
            html_path.write_text(html_outputs[tmpl])

    return html_outputs


def load(path: Union[str, Path]) -> "RunResult":
    """Load an analysis from JSON file.

    Args:
        path: Path to JSON file or dump folder

    Returns:
        RunResult object with .to_html(), .to_json(), etc.

    Raises:
        RenderError: If loading fails
    """
    from ..cli._common import load_pipeline_json

    from ._results import RunResult

    path = Path(path)

    # if path is a folder, look for JSON inside
    if path.is_dir():
        json_files = list(path.glob("*.json"))
        if not json_files:
            raise RenderError(f"No JSON file found in folder: {path}")
        # prefer file matching folder name
        matching = [f for f in json_files if f.stem == path.name.replace("_dump", "")]
        json_path = matching[0] if matching else json_files[0]
    else:
        json_path = path

    try:
        pipeline = load_pipeline_json(str(json_path))
        return RunResult(
            pipeline=pipeline,
            output_folder=json_path.parent,
            errors=[],
        )
    except Exception as e:
        raise RenderError(f"Error loading analysis: {e}")
