"""CLI command for extracting themes from text."""

import json
import sys
from pathlib import Path
from typing import Optional

import anyio
import typer

from ._common import check_and_prompt_credentials


def extract_themes(
    input_file: str = typer.Argument(
        ..., help="Input file (PDF, DOCX, TXT) or '-' for stdin"
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Guidance for theme extraction",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (default: gpt-4.1-mini)",
    ),
    output_format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json or yaml",
    ),
):
    """Extract structured themes from text (PDF, DOCX, TXT, or stdin).

    Useful for creating analytical frameworks from papers or notes.

    Examples:
        soak extract-themes paper.pdf
        soak extract-themes paper.pdf --prompt "Focus on patient experience"
        cat notes.txt | soak extract-themes -
        soak extract-themes paper.pdf --format yaml
    """
    credentials = check_and_prompt_credentials()

    # read input
    if input_file == "-":
        text = sys.stdin.read()
    else:
        path = Path(input_file)
        if not path.exists():
            typer.echo(f"File not found: {input_file}", err=True)
            raise typer.Exit(1)

        # use soak's document extraction for PDF/DOCX
        from soak.document_utils import extract_text

        text = extract_text(str(path))

    if not text.strip():
        typer.echo("No text to extract themes from.", err=True)
        raise typer.Exit(1)

    from soak.extract_themes import extract_themes as _extract_themes

    themes = anyio.run(_extract_themes, text, prompt, model, credentials)

    if output_format == "yaml":
        try:
            import yaml

            print(yaml.dump({"themes": themes}, default_flow_style=False, allow_unicode=True))
        except ImportError:
            typer.echo("PyYAML required for YAML output. Install with: pip install pyyaml", err=True)
            raise typer.Exit(1)
    else:
        print(json.dumps(themes, indent=2, ensure_ascii=False))
