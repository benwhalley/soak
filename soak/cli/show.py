"""Show command for displaying pipelines and templates."""

import logging
import sys
from pathlib import Path

import typer

from ._common import PIPELINE_DIR, TEMPLATES_DIR

logger = logging.getLogger(__name__)


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
        items_dir = TEMPLATES_DIR
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
