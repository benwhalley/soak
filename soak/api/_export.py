"""Export implementation for soak API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union


class ExportError(Exception):
    """Error during export."""

    pass


def export_pdf(
    analysis: Union[str, Path, dict],
    output: Optional[Union[str, Path]] = None,
    template: str = "default",
    pipeline_name: str = "Analysis",
    model_name: str = "",
) -> Path:
    """Export analysis to PDF.

    Args:
        analysis: JSON file path or analysis dict
        output: Output path (default: derived from input)
        template: PDF template style: "default" or "apa"
        pipeline_name: Name for PDF metadata
        model_name: Model name for PDF metadata

    Returns:
        Path to generated PDF

    Raises:
        ExportError: If export fails
    """
    from ..exports.typst_export import export_analysis_pdf

    # load data if path
    if isinstance(analysis, (str, Path)):
        input_path = Path(analysis)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        default_output = input_path.with_suffix(".pdf")
    else:
        data = analysis
        default_output = Path("analysis.pdf")

    output_path = Path(output) if output else default_output

    # extract analysis data
    analysis_data = _extract_analysis_data(data)
    if analysis_data is None:
        raise ExportError("Could not extract analysis from data")

    # extract metadata from config if available
    config = data.get("config", {})
    if pipeline_name == "Analysis":
        pipeline_name = data.get("name", config.get("name", "Analysis"))
    if not model_name:
        model_name = config.get("model_name", "")
    doc_count = len(config.get("document_paths", []))

    export_analysis_pdf(
        analysis=analysis_data,
        output_path=output_path,
        pipeline_name=pipeline_name,
        model_name=model_name,
        doc_count=doc_count,
        word_count=0,
        template=template,
    )

    return output_path


def export_xlsx(
    analysis: Union[str, Path, dict],
    output: Optional[Union[str, Path]] = None,
    format: str = "all",
) -> list[Path]:
    """Export analysis to XLSX file(s).

    Args:
        analysis: JSON file path or analysis dict
        output: Output path prefix (default: derived from input)
        format: Export format: "all", "themes", "codes", or "similarity"

    Returns:
        List of generated file paths

    Raises:
        ExportError: If export fails
    """
    from ..exports.xlsx_export import (generate_codes_xlsx,
                                       generate_self_similarity_xlsx,
                                       generate_themes_xlsx)

    # load data if path
    if isinstance(analysis, (str, Path)):
        input_path = Path(analysis)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        output_base = input_path.stem
        output_dir = input_path.parent
    else:
        data = analysis
        output_base = "analysis"
        output_dir = Path.cwd()

    if output:
        output_path = Path(output)
        output_base = output_path.stem
        output_dir = output_path.parent

    # extract themes and codes
    analysis_data, themes, codes = _extract_themes_codes(data)
    if analysis_data is None:
        raise ExportError("Could not extract analysis from data")

    output_files = []

    if format in ("all", "themes"):
        if not themes:
            raise ExportError("No themes found")
        themes_path = output_dir / f"{output_base}-themes.xlsx"
        generate_themes_xlsx(themes, codes, themes_path)
        output_files.append(themes_path)

    if format in ("all", "codes"):
        if not codes:
            raise ExportError("No codes found")
        codes_path = output_dir / f"{output_base}-codes.xlsx"
        generate_codes_xlsx(codes, codes_path)
        output_files.append(codes_path)

    if format in ("all", "similarity"):
        if len(themes) < 2:
            raise ExportError("Need at least 2 themes for similarity matrix")
        sim_path = output_dir / f"{output_base}-similarity.xlsx"
        generate_self_similarity_xlsx(themes, sim_path)
        output_files.append(sim_path)

    return output_files


def _extract_analysis_data(data: dict) -> Optional[dict]:
    """Extract analysis data from various JSON formats."""
    if "result" in data:
        return data["result"]
    if "themes" in data or "codes" in data:
        return data

    nodes = data.get("nodes", {})
    if isinstance(nodes, dict):
        analysis = {"themes": [], "codes": [], "narrative": ""}
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
        return analysis

    return None


def _extract_from_output(output_list: list, key: str):
    """Extract data from node output structure."""
    if not output_list:
        return []

    item = output_list[0]

    if isinstance(item, dict) and "results" in item:
        results = item["results"]
        if key in results:
            segment = results[key]
            if isinstance(segment, dict) and "output" in segment:
                output = segment["output"]
                if isinstance(output, dict) and key in output:
                    return output[key]
                return output

    if isinstance(item, dict) and key in item:
        return item[key]

    return []


def _extract_themes_codes(data: dict) -> tuple[Optional[dict], list, list]:
    """Extract analysis data, themes, and codes from JSON structure."""
    analysis = _extract_analysis_data(data)

    themes = []
    codes = []

    if analysis:
        themes = analysis.get("themes", [])
        codes = analysis.get("codes", [])

    return analysis, themes, codes
