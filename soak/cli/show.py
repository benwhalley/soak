"""Show command for displaying pipelines and templates."""

import logging
import sys

import typer

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

    # deferred import to avoid circular dependency
    from ..api import ShowError, get_pipeline, get_template, list_pipelines, list_templates

    # if item_type is not a known type, treat it as a pipeline name
    if item_type not in ["pipeline", "template"]:
        # shift arguments: item_type becomes the name, search for pipeline
        name = item_type
        item_type = "pipeline"

        # try to get the pipeline content
        try:
            content = get_pipeline(name)
            print(content, file=sys.stdout)
            return
        except ShowError:
            logger.error(f"Pipeline '{name}' not found")
            raise typer.Exit(1)

    # List all if no name provided
    if name is None:
        if item_type == "pipeline":
            items = list_pipelines()
        else:
            items = list_templates()

        logger.info(f"Available {item_type}s:")
        for item in items:
            logger.info(f"  {item}")
        logger.info(f"\nUsage: soak show {item_type} <name>")
        return

    # Get specific item content
    try:
        if item_type == "pipeline":
            content = get_pipeline(name)
        else:
            content = get_template(name)
        print(content, file=sys.stdout)
    except ShowError as e:
        logger.error(str(e))
        raise typer.Exit(1)
