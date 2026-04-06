"""Export command for soak CLI.

Exports analysis results to various formats including PDF and XLSX.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from ._common import resolve_analysis_path

logger = logging.getLogger(__name__)

app = typer.Typer(name="export", help="Export analysis results to various formats.")


@app.command()
def export(
    input_path: str = typer.Argument(
        ...,
        help="Path to JSON file containing analysis results",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: derived from input with new extension)",
    ),
    format: str = typer.Option(
        "pdf",
        "--format",
        "-f",
        help="Output format: pdf, xlsx, themes-xlsx, codes-xlsx, similarity-xlsx",
    ),
    template: str = typer.Option(
        "default",
        "--template",
        "-t",
        help="PDF template style: default or apa (APA 7th edition)",
    ),
    pipeline_name: str = typer.Option(
        "Analysis",
        "--pipeline",
        "-p",
        help="Pipeline name for PDF metadata",
    ),
    model_name: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Model name for PDF metadata",
    ),
) -> None:
    """Export analysis results to PDF or XLSX formats.

    Takes a JSON file containing analysis results and exports to the
    specified format.

    Formats:
        pdf             - Full PDF report (requires typst)
        xlsx            - All XLSX files (themes, codes, similarity)
        themes-xlsx     - Themes only
        codes-xlsx      - Codes only
        similarity-xlsx - Self-similarity matrix

    Example:
        soak export results.json --format pdf --output report.pdf
        soak export results.json --format pdf --template apa --output report.pdf
        soak export results.json --format similarity-xlsx --output overlap.xlsx
        soak export results.json --format xlsx  # exports all xlsx files
    """
    json_path = resolve_analysis_path(input_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fmt = format.lower()
    if fmt == "pdf":
        _export_pdf(data, json_path, output, pipeline_name, model_name, template)
    elif fmt == "xlsx":
        _export_all_xlsx(data, json_path, output)
    elif fmt == "themes-xlsx":
        _export_themes_xlsx(data, json_path, output)
    elif fmt == "codes-xlsx":
        _export_codes_xlsx(data, json_path, output)
    elif fmt in ("similarity-xlsx", "self-similarity-xlsx", "overlap-xlsx"):
        _export_similarity_xlsx(data, json_path, output)
    else:
        logger.error(f"Unsupported format: {format}")
        raise typer.Exit(1)


def _export_pdf(
    data: dict,
    json_path: Path,
    output: Optional[str],
    pipeline_name: str,
    model_name: str,
    template: str = "default",
) -> None:
    """Export analysis to PDF."""
    from ..exports.typst_export import export_analysis_pdf

    # determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = json_path.with_suffix(".pdf")

    # extract analysis data
    analysis = None
    doc_count = 0
    word_count = 0

    if "result" in data:
        analysis = data["result"]
    elif "themes" in data or "codes" in data:
        analysis = data
    else:
        # try to extract from pipeline structure
        nodes = data.get("nodes", {})
        if isinstance(nodes, dict):
            analysis = {
                "themes": [],
                "codes": [],
                "narrative": "",
            }
            if "themes" in nodes:
                theme_output = nodes["themes"].get("output", [])
                if theme_output:
                    analysis["themes"] = _extract_from_output(theme_output, "themes")
            if "codes" in nodes:
                code_output = nodes["codes"].get("output", [])
                if code_output:
                    analysis["codes"] = _extract_from_output(code_output, "codes")
            if "narrative" in nodes:
                narr_output = nodes["narrative"].get("output", [])
                if narr_output:
                    analysis["narrative"] = _extract_from_output(narr_output, "report")

    if analysis is None:
        logger.error("Could not extract analysis from JSON file")
        raise typer.Exit(1)

    # extract metadata from config if available
    config = data.get("config", {})
    if not pipeline_name or pipeline_name == "Analysis":
        pipeline_name = data.get("name", config.get("name", "Analysis"))
    if not model_name:
        model_name = config.get("model_name", "")
    doc_paths = config.get("document_paths", [])
    doc_count = len(doc_paths)

    export_analysis_pdf(
        analysis=analysis,
        output_path=output_path,
        pipeline_name=pipeline_name,
        model_name=model_name,
        doc_count=doc_count,
        word_count=word_count,
        template=template,
    )

    logger.info(f"Exported PDF to {output_path}")
    typer.echo(f"PDF exported to: {output_path}")


def _extract_from_output(output_list: list, key: str):
    """Extract data from node output structure."""
    if not output_list:
        return []

    item = output_list[0]

    # handle StruckdownResult structure
    if isinstance(item, dict) and "results" in item:
        results = item["results"]
        if key in results:
            segment = results[key]
            if isinstance(segment, dict) and "output" in segment:
                output = segment["output"]
                if isinstance(output, dict) and key in output:
                    return output[key]
                return output

    # direct extraction
    if isinstance(item, dict):
        if key in item:
            return item[key]

    return []


def _extract_analysis(data: dict) -> tuple[dict, list, list]:
    """Extract analysis data, themes, and codes from JSON structure.

    Returns:
        Tuple of (analysis_dict, themes_list, codes_list)
    """
    analysis = None
    themes = []
    codes = []

    if "result" in data:
        analysis = data["result"]
    elif "themes" in data or "codes" in data:
        analysis = data
    else:
        # try to extract from pipeline structure
        nodes = data.get("nodes", {})
        if isinstance(nodes, dict):
            analysis = {
                "themes": [],
                "codes": [],
                "narrative": "",
            }
            if "themes" in nodes:
                theme_output = nodes["themes"].get("output", [])
                if theme_output:
                    analysis["themes"] = _extract_from_output(theme_output, "themes")
            if "codes" in nodes:
                code_output = nodes["codes"].get("output", [])
                if code_output:
                    analysis["codes"] = _extract_from_output(code_output, "codes")

    if analysis:
        themes = analysis.get("themes", [])
        codes = analysis.get("codes", [])

    return analysis, themes, codes


def _export_all_xlsx(
    data: dict,
    json_path: Path,
    output: Optional[str],
) -> None:
    """Export all XLSX files (themes, codes, similarity)."""
    analysis, themes, codes = _extract_analysis(data)

    if analysis is None:
        logger.error("Could not extract analysis from JSON file")
        raise typer.Exit(1)

    # determine output prefix
    if output:
        output_base = Path(output).stem
        output_dir = Path(output).parent
    else:
        output_base = json_path.stem
        output_dir = json_path.parent

    # export themes
    themes_path = output_dir / f"{output_base}-themes.xlsx"
    _do_export_themes(themes, codes, themes_path)

    # export codes
    codes_path = output_dir / f"{output_base}-codes.xlsx"
    _do_export_codes(codes, codes_path)

    # export similarity if enough themes
    if len(themes) >= 2:
        sim_path = output_dir / f"{output_base}-similarity.xlsx"
        _do_export_similarity(themes, sim_path)


def _export_themes_xlsx(
    data: dict,
    json_path: Path,
    output: Optional[str],
) -> None:
    """Export themes to XLSX."""
    analysis, themes, codes = _extract_analysis(data)

    if not themes:
        logger.error("No themes found in JSON file")
        raise typer.Exit(1)

    if output:
        output_path = Path(output)
    else:
        output_path = json_path.with_name(f"{json_path.stem}-themes.xlsx")

    _do_export_themes(themes, codes, output_path)


def _export_codes_xlsx(
    data: dict,
    json_path: Path,
    output: Optional[str],
) -> None:
    """Export codes to XLSX."""
    analysis, themes, codes = _extract_analysis(data)

    if not codes:
        logger.error("No codes found in JSON file")
        raise typer.Exit(1)

    if output:
        output_path = Path(output)
    else:
        output_path = json_path.with_name(f"{json_path.stem}-codes.xlsx")

    _do_export_codes(codes, output_path)


def _export_similarity_xlsx(
    data: dict,
    json_path: Path,
    output: Optional[str],
) -> None:
    """Export self-similarity matrix to XLSX."""
    analysis, themes, codes = _extract_analysis(data)

    if len(themes) < 2:
        logger.error("Need at least 2 themes for self-similarity matrix")
        raise typer.Exit(1)

    if output:
        output_path = Path(output)
    else:
        output_path = json_path.with_name(f"{json_path.stem}-similarity.xlsx")

    _do_export_similarity(themes, output_path)


def _do_export_themes(themes: list, codes: list, output_path: Path) -> None:
    """Generate themes XLSX file."""
    from ..exports.xlsx_export import generate_themes_xlsx

    generate_themes_xlsx(themes, codes, output_path)
    logger.info(f"Exported themes to {output_path}")
    typer.echo(f"Themes exported to: {output_path}")


def _do_export_codes(codes: list, output_path: Path) -> None:
    """Generate codes XLSX file."""
    from ..exports.xlsx_export import generate_codes_xlsx

    generate_codes_xlsx(codes, output_path)
    logger.info(f"Exported codes to {output_path}")
    typer.echo(f"Codes exported to: {output_path}")


def _do_export_similarity(themes: list, output_path: Path) -> None:
    """Generate self-similarity XLSX file."""
    from ..exports.xlsx_export import generate_self_similarity_xlsx

    success = generate_self_similarity_xlsx(themes, output_path)
    if success:
        logger.info(f"Exported self-similarity to {output_path}")
        typer.echo(f"Self-similarity exported to: {output_path}")
    else:
        logger.error("Failed to generate self-similarity matrix")
        raise typer.Exit(1)
