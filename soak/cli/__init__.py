"""Command-line interface for running qualitative analysis pipelines.

This package organizes the CLI into separate modules for each command group.
"""

import sys

import typer

from ._common import (
    COMMANDS,
    PIPELINE_DIR,
    TEMPLATES_DIR,
    check_and_prompt_credentials,
    generate_all_html_outputs,
    generate_html_output,
    get_pdb_on_exception,
    get_soak_version,
    load_pipeline_json,
    resolve_analysis_path,
    resolve_template,
    set_pdb_on_exception,
    setup_logging,
    version_callback,
)

# create the main app
app = typer.Typer(name="soak", pretty_exceptions_show_locals=False)


@app.callback()
def main(
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,  # allows -v -vv -vvv
        help="Increase verbosity (-v=INFO, -vv=DEBUG)",
    ),
    pdb_on_exception: bool = typer.Option(
        False,
        "--pdb",
        help="Drop into pdb debugger on unhandled exceptions",
    ),
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
):
    """soak: DAG-based pipeline system for LLM-assisted qualitative analysis.

    Try `soak run --help` to get started
    """
    set_pdb_on_exception(pdb_on_exception)
    setup_logging(verbose)


# import and register commands
from .run import run
from .compare import compare
from .coverage_cmd import coverage
from .show import show
from .misc import tui, test, calibrate

app.command()(run)
app.command()(compare)
app.command()(coverage)
app.command()(show)
app.command()(tui)
app.command()(test)
app.command()(calibrate)


def main_with_default_command():
    """Entry point that makes 'run' the default command."""
    # Make 'run' the default command when first non-flag arg is not a built-in command
    # Find first non-flag argument
    first_arg = None
    first_arg_index = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if not arg.startswith("-"):
            first_arg = arg
            first_arg_index = i
            break

    # If first non-flag arg exists and is not a command, inject 'run' before it
    if first_arg and first_arg not in COMMANDS:
        sys.argv.insert(first_arg_index, "run")

    app()


# export for backwards compatibility
__all__ = [
    "app",
    "main_with_default_command",
    "COMMANDS",
    "PIPELINE_DIR",
    "TEMPLATES_DIR",
    "check_and_prompt_credentials",
    "generate_all_html_outputs",
    "generate_html_output",
    "get_pdb_on_exception",
    "get_soak_version",
    "load_pipeline_json",
    "resolve_analysis_path",
    "resolve_template",
    "set_pdb_on_exception",
    "setup_logging",
    "version_callback",
]
