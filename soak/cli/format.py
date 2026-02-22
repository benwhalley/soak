"""Format command for rendering existing JSON results with templates."""

import logging
import sys
from pathlib import Path

import typer

from ._common import (
    generate_all_html_outputs,
    load_pipeline_json,
    resolve_template,
)

logger = logging.getLogger(__name__)


def format(
    input_json: Path = typer.Argument(
        ...,
        help="Path to JSON file from a previous pipeline run",
        exists=True,
        dir_okay=False,
    ),
    template: list[str] = typer.Option(
        ["simple.html"],
        "--template",
        "-t",
        help="Template name (in soak/templates) or path to custom HTML template (can be used multiple times)",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory or file path. Defaults to same directory as input JSON",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing output files",
    ),
):
    """Render an existing JSON analysis result with one or more templates.

    This allows re-rendering results with different templates without re-running
    the analysis pipeline.

    Examples:

        # Render with a new template
        soak format results.json -t detailed.html

        # Render with multiple templates
        soak format results.json -t simple.html -t pipeline.html

        # Output to a different directory
        soak format results.json -t simple.html -o ./exports/
    """
    # Load the pipeline from JSON
    logger.info(f"Loading analysis from {input_json}")
    pipeline = load_pipeline_json(str(input_json))

    # Determine output directory
    if output is None:
        output_dir = input_json.parent
        output_stem = input_json.stem
    elif Path(output).suffix:
        # Output is a file path - use its directory and stem
        output_dir = Path(output).parent
        output_stem = Path(output).stem
    else:
        # Output is a directory
        output_dir = Path(output)
        output_stem = input_json.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing files
    output_files = []
    for tmpl in template:
        template_stem = Path(resolve_template(tmpl)).stem
        html_path = output_dir / f"{output_stem}_{template_stem}.html"
        output_files.append((tmpl, html_path))

        if html_path.exists() and not force:
            print(
                f"Error: Output file already exists: {html_path}",
                file=sys.stderr,
            )
            print("Use --force/-f to overwrite", file=sys.stderr)
            raise typer.Exit(1)

    # Generate HTML outputs
    logger.info(f"Rendering {len(template)} template(s)")
    html_outputs = generate_all_html_outputs(pipeline, template, on_error="raise")

    # Write output files
    for tmpl, html_path in output_files:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_outputs[tmpl])
        logger.info(f"Wrote {html_path}")

    print(f"Generated {len(template)} output(s) in {output_dir}")
